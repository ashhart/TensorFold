//! A stream's state on a rank: attention keys and values in a pool of 64-position pages, each Mamba layer's state.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const mamba = @import("mamba.zig");

const Allocator = std.mem.Allocator;

/// Positions a page holds.
pub const tokens = 64;

pub fn pagesFor(n: usize) usize {
    return (n + tokens - 1) / tokens;
}

/// Every attention layer's key and value pools (the model's, then the MTP head's): `kv_heads` runs of `count` pages.
pub const Pool = struct {
    gpa: Allocator,
    count: usize,
    kv_heads: usize,
    head_dim: usize,
    act: model.Kind,
    keys: []hip.DeviceBuffer,
    values: []hip.DeviceBuffer,
    free: std.ArrayList(u32) = .empty,

    /// Device bytes of `count` pages over `layers` attention layers.
    pub fn bytes(layers: usize, count: usize, kv_heads: usize, head_dim: usize, act: model.Kind) usize {
        return 2 * layers * count * kv_heads * tokens * head_dim * act.size();
    }

    pub fn init(
        gpa: Allocator,
        d: *const hip.Runtime,
        layers: usize,
        count: usize,
        kv_heads: usize,
        head_dim: usize,
        act: model.Kind,
    ) !Pool {
        const keys = try gpa.alloc(hip.DeviceBuffer, layers);
        errdefer gpa.free(keys);
        const values = try gpa.alloc(hip.DeviceBuffer, layers);
        errdefer gpa.free(values);
        var made: usize = 0;
        errdefer for (keys[0..made], values[0..made]) |*k, *v| {
            k.free();
            v.free();
        };
        const each = bytes(1, count, kv_heads, head_dim, act) / 2;
        for (keys, values) |*k, *v| {
            k.* = try hip.DeviceBuffer.alloc(d, each);
            errdefer k.free();
            v.* = try hip.DeviceBuffer.alloc(d, each);
            made += 1;
        }
        var p: Pool = .{
            .gpa = gpa,
            .count = count,
            .kv_heads = kv_heads,
            .head_dim = head_dim,
            .act = act,
            .keys = keys,
            .values = values,
        };
        errdefer p.free.deinit(gpa);
        try p.free.ensureTotalCapacity(gpa, count);
        // the lowest id is taken first
        var i: u32 = @intCast(count);
        while (i > 0) {
            i -= 1;
            p.free.appendAssumeCapacity(i);
        }
        return p;
    }

    pub fn deinit(p: *Pool) void {
        p.free.deinit(p.gpa);
        for (p.keys, p.values) |*k, *v| {
            k.free();
            v.free();
        }
        p.gpa.free(p.keys);
        p.gpa.free(p.values);
    }

    pub fn take(p: *Pool) ?u32 {
        return p.free.pop();
    }

    pub fn give(p: *Pool, id: u32) void {
        p.free.appendAssumeCapacity(id);
    }

    /// Attention layer `slot`'s pools read through a stream's page `table` (int32 ids on the device).
    pub fn paged(p: *const Pool, slot: usize, table: u64) hip.ops.Ops.Paged {
        return .{
            .k = p.keys[slot].base(),
            .v = p.values[slot].base(),
            .table = table,
            .kind = p.act,
            .kv_heads = p.kv_heads,
            .d = p.head_dim,
            .count = p.count,
        };
    }
};

/// One stream: its page table, positions kept (the model's and the MTP head's), every Mamba layer's state.
pub const Stream = struct {
    gpa: Allocator,
    /// int32 page ids on the device, `cap / tokens` of them; `ids` the same on the host.
    table: hip.DeviceBuffer,
    ids: std.ArrayList(u32) = .empty,
    cap: usize,
    len: usize = 0,
    mtp_len: usize = 0,
    /// Every Mamba layer's conv tail and SSM state, one buffer; `states` by the layer's place among the Mamba layers.
    mem: hip.DeviceBuffer,
    states: []mamba.State,

    /// Device bytes of a stream's Mamba state on this rank over `layers` Mamba layers.
    pub fn stateBytes(d: model.Dims, act: model.Kind, layers: usize) usize {
        return layers * (mamba.convBytes(d, act) + mamba.ssmBytes(d));
    }

    pub fn init(
        gpa: Allocator,
        r: *const hip.Runtime,
        d: model.Dims,
        act: model.Kind,
        layers: usize,
        cap: usize,
    ) !Stream {
        var table = try hip.DeviceBuffer.alloc(r, @max(4, 4 * pagesFor(cap)));
        errdefer table.free();
        var mem = try hip.DeviceBuffer.alloc(r, @max(4, stateBytes(d, act, layers)));
        errdefer mem.free();
        try mem.fill8(0);
        const states = try gpa.alloc(mamba.State, layers);
        const conv = mamba.convBytes(d, act);
        const each = conv + mamba.ssmBytes(d);
        for (states, 0..) |*s, i| s.* = .{ .conv = mem.base() + i * each, .ssm = mem.base() + i * each + conv };
        return .{ .gpa = gpa, .table = table, .cap = cap, .mem = mem, .states = states };
    }

    /// The pages back to `pool`, the buffers freed.
    pub fn deinit(s: *Stream, pool: *Pool) void {
        for (s.ids.items) |id| pool.give(id);
        s.ids.deinit(s.gpa);
        s.gpa.free(s.states);
        s.mem.free();
        s.table.free();
        s.* = undefined;
    }

    /// Pages to cover `n` positions, taken from `pool` and the table sent up; refused past the stream's capacity.
    pub fn cover(s: *Stream, pool: *Pool, n: usize) !void {
        if (n > s.cap) return error.StreamFull;
        const want = pagesFor(n);
        if (want <= s.ids.items.len) return;
        while (s.ids.items.len < want) try s.ids.append(s.gpa, pool.take() orelse return error.OutOfPages);
        try s.table.upload(0, std.mem.sliceAsBytes(s.ids.items));
    }
};

test "pages cover positions by 64; a pool of them holds keys and values of every attention layer" {
    try std.testing.expectEqual(@as(usize, 0), pagesFor(0));
    try std.testing.expectEqual(@as(usize, 2), pagesFor(65));
    try std.testing.expectEqual(@as(usize, 2 * 7 * 10 * 2 * 64 * 128 * 2), Pool.bytes(7, 10, 2, 128, .bf16));
}
