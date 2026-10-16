//! Paged KV: attention keys and values in a pool of reference-counted pages, so streams and the prefix tree share them.

const std = @import("std");
const hip = @import("hip");
const view = @import("../model/view.zig");
const Allocator = std.mem.Allocator;

/// Positions a page holds: the chunk of the recurrence, so a page edge is a chunk edge.
pub const tokens = 64;

/// Pages that hold `n` positions.
pub fn pagesFor(n: usize) usize {
    return (n + tokens - 1) / tokens;
}

/// Which page ids are free and how many holders each has: host bookkeeping only.
pub const Ids = struct {
    refs: []u32,
    free: std.ArrayList(u32) = .empty,
    /// Pages promised to streams that have not taken them yet.
    reserved: usize = 0,

    pub fn init(gpa: Allocator, count: usize) !Ids {
        const refs = try gpa.alloc(u32, count);
        errdefer gpa.free(refs);
        @memset(refs, 0);
        var ids: Ids = .{ .refs = refs };
        errdefer ids.free.deinit(gpa);
        try ids.free.ensureTotalCapacity(gpa, count);
        // the lowest id is taken first
        var i: u32 = @intCast(count);
        while (i > 0) {
            i -= 1;
            ids.free.appendAssumeCapacity(i);
        }
        return ids;
    }

    pub fn deinit(ids: *Ids, gpa: Allocator) void {
        ids.free.deinit(gpa);
        gpa.free(ids.refs);
    }

    /// A free page, now held once; null when none is free.
    pub fn take(ids: *Ids) ?u32 {
        const id = ids.free.pop() orelse return null;
        ids.refs[id] = 1;
        return id;
    }

    pub fn retain(ids: *Ids, id: u32) void {
        std.debug.assert(ids.refs[id] > 0);
        ids.refs[id] += 1;
    }

    /// One holder less; the page is free again when none is left.
    pub fn release(ids: *Ids, id: u32) void {
        std.debug.assert(ids.refs[id] > 0);
        ids.refs[id] -= 1;
        if (ids.refs[id] == 0) ids.free.appendAssumeCapacity(id);
    }

    /// Pages free and not promised.
    pub fn available(ids: *const Ids) usize {
        return ids.free.items.len -| ids.reserved;
    }

    pub fn used(ids: *const Ids) usize {
        return ids.refs.len - ids.free.items.len;
    }
};

/// Each attention layer's key and value pools and their ids: `kv_heads` runs of `count` pages, contiguous in order.
pub const Pool = struct {
    gpa: Allocator,
    d: *const hip.Runtime,
    count: usize,
    kv_heads: usize,
    head_dim: usize,
    act_bytes: usize,
    /// A layer's keys and values: empty buffers for a linear layer.
    keys: []hip.DeviceBuffer,
    values: []hip.DeviceBuffer,
    ids: Ids,
    /// Whether this holder hands pages out (rank 0 and a lone rank); a follower is told which pages to use.
    managed: bool,

    /// Device bytes a pool of `count` pages holds: keys and values of every attention layer.
    pub fn deviceBytes(m: *const view.Model, count: usize) usize {
        const s = m.spec;
        var full: usize = 0;
        for (0..s.n_layers) |i| full += @intFromBool(s.full(i));
        return 2 * full * count * s.kv_heads * tokens * s.head_dim * m.act.size();
    }

    pub fn init(gpa: Allocator, d: *const hip.Runtime, m: *const view.Model, count: usize, managed: bool) !Pool {
        const s = m.spec;
        const keys = try gpa.alloc(hip.DeviceBuffer, s.n_layers);
        errdefer gpa.free(keys);
        const values = try gpa.alloc(hip.DeviceBuffer, s.n_layers);
        errdefer gpa.free(values);
        var made: usize = 0;
        errdefer for (0..made) |i| {
            keys[i].free();
            values[i].free();
        };
        const per_layer = count * s.kv_heads * tokens * s.head_dim * m.act.size();
        for (0..s.n_layers) |i| {
            const bytes = if (s.full(i)) per_layer else 0;
            keys[i] = try hip.DeviceBuffer.alloc(d, bytes);
            errdefer keys[i].free();
            values[i] = try hip.DeviceBuffer.alloc(d, bytes);
            made += 1;
        }
        return .{ .gpa = gpa, .d = d, .count = count, .kv_heads = s.kv_heads, .head_dim = s.head_dim, .act_bytes = m.act.size(), .keys = keys, .values = values, .ids = try Ids.init(gpa, count), .managed = managed };
    }

    pub fn deinit(p: *Pool) void {
        p.ids.deinit(p.gpa);
        for (p.keys, p.values) |*k, *v| {
            k.free();
            v.free();
        }
        p.gpa.free(p.keys);
        p.gpa.free(p.values);
    }

    /// Bytes of one position's keys and values of a layer.
    fn rowBytes(p: *const Pool) usize {
        return p.head_dim * p.act_bytes;
    }

    /// Bytes of one head's share of a page of one layer's keys (or values).
    fn headBytes(p: *const Pool) usize {
        return tokens * p.rowBytes();
    }

    /// Bytes of one page of one layer's keys (or values), over its heads.
    pub fn layerBytes(p: *const Pool) usize {
        return p.kv_heads * p.headBytes();
    }

    /// Bytes of a page over every attention layer, keys and values.
    pub fn pageBytes(p: *const Pool) usize {
        var layers: usize = 0;
        for (p.keys) |k| layers += @intFromBool(k.len > 0);
        return 2 * layers * p.layerBytes();
    }

    /// A page for one holder, or null when the pool has none.
    pub fn take(p: *Pool) ?u32 {
        return p.ids.take();
    }

    pub fn retain(p: *Pool, id: u32) void {
        if (p.managed) p.ids.retain(id);
    }

    pub fn release(p: *Pool, id: u32) void {
        if (p.managed) p.ids.release(id);
    }

    /// Copies page `from` into page `to` in every layer.
    pub fn copyPage(p: *Pool, from: u32, to: u32, stream: hip.abi.Stream) !void {
        if (from >= p.count or to >= p.count) return error.BadPage;
        const n = p.headBytes();
        for (p.keys, p.values) |k, v| {
            if (k.len == 0) continue;
            for (0..p.kv_heads) |h| {
                const run = h * p.count * n;
                try hip.raw.copy(k, run + to * n, k.base() + run + from * n, n, stream);
                try hip.raw.copy(v, run + to * n, v.base() + run + from * n, n, stream);
            }
        }
    }

    /// Page `id`'s keys of layer `layer` read back, then its values, each head after head: for checks.
    pub fn read(p: *const Pool, layer: usize, id: u32, out: []u8) !void {
        const n = p.headBytes();
        std.debug.assert(out.len == 2 * p.layerBytes());
        for (0..p.kv_heads) |h| {
            try p.keys[layer].download((h * p.count + id) * n, out[h * n ..][0..n]);
            try p.values[layer].download((h * p.count + id) * n, out[(p.kv_heads + h) * n ..][0..n]);
        }
    }
};

test "ids hand out the lowest page and free it when its last holder lets go" {
    var ids = try Ids.init(std.testing.allocator, 3);
    defer ids.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?u32, 0), ids.take());
    try std.testing.expectEqual(@as(?u32, 1), ids.take());
    ids.retain(0);
    ids.release(0);
    try std.testing.expectEqual(@as(usize, 1), ids.available());
    ids.release(0);
    try std.testing.expectEqual(@as(usize, 2), ids.available());
    try std.testing.expectEqual(@as(?u32, 0), ids.take());
    try std.testing.expectEqual(@as(?u32, 2), ids.take());
    try std.testing.expectEqual(@as(?u32, null), ids.take());
}

test "reserved pages are not available" {
    var ids = try Ids.init(std.testing.allocator, 4);
    defer ids.deinit(std.testing.allocator);
    ids.reserved = 3;
    try std.testing.expectEqual(@as(usize, 1), ids.available());
    ids.reserved = 9;
    try std.testing.expectEqual(@as(usize, 0), ids.available());
}

test "pages cover positions by 64" {
    try std.testing.expectEqual(@as(usize, 0), pagesFor(0));
    try std.testing.expectEqual(@as(usize, 1), pagesFor(64));
    try std.testing.expectEqual(@as(usize, 2), pagesFor(65));
}
