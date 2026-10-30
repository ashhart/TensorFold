//! Paged KV: attention keys and values in a pool of reference-counted pages, so streams and the prefix tree share them.

const std = @import("std");
const hip = @import("hip");
const view = @import("../model/view.zig");
const shared = @import("hip_serve").pages;
const Allocator = std.mem.Allocator;

/// Positions a page holds: the chunk of the recurrence, so a page edge is a chunk edge.
pub const tokens = shared.tokens;
pub const pagesFor = shared.pagesFor;
pub const Ids = shared.Ids;

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

test {
    _ = shared;
}
