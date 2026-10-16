//! A stream's state on a rank: its pages of a pool of 64-position pages behind a descriptor, each Mamba layer's state.
const std = @import("std");
const hip = @import("hip");
const shared = @import("hip_serve").pages;
const model = @import("model.zig");
const mamba = @import("mamba.zig");

const Allocator = std.mem.Allocator;

/// Positions a page holds.
pub const tokens = shared.tokens;
pub const pagesFor = shared.pagesFor;

/// Every attention layer's key and value pools (the model's, then the MTP head's): `kv_heads` runs of `count` pages.
pub const Pool = struct {
    gpa: Allocator,
    count: usize,
    kv_heads: usize,
    /// The last slot's kv heads when it is the MTP head's: whole on the drafting rank, however the model's are cut.
    head_kv: ?usize,
    head_dim: usize,
    act: model.Kind,
    keys: []hip.DeviceBuffer,
    values: []hip.DeviceBuffer,
    /// Which pages are free and who holds each: a stream, and the prompt cache's tree; `reserved` are promised.
    ids: shared.Ids,

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
        head_kv: ?usize,
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
        for (keys, values, 0..) |*k, *v, i| {
            const kv = if (head_kv != null and i + 1 == layers) head_kv.? else kv_heads;
            const each = bytes(1, count, kv, head_dim, act) / 2;
            k.* = try hip.DeviceBuffer.alloc(d, each);
            errdefer k.free();
            v.* = try hip.DeviceBuffer.alloc(d, each);
            made += 1;
        }
        return .{
            .gpa = gpa,
            .count = count,
            .kv_heads = kv_heads,
            .head_kv = head_kv,
            .head_dim = head_dim,
            .act = act,
            .keys = keys,
            .values = values,
            .ids = try shared.Ids.init(gpa, count),
        };
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

    /// A free page, held once.
    pub fn take(p: *Pool) ?u32 {
        return p.ids.take();
    }

    pub fn retain(p: *Pool, id: u32) void {
        p.ids.retain(id);
    }

    /// One holder fewer; the page is free again at none.
    pub fn release(p: *Pool, id: u32) void {
        p.ids.release(id);
    }

    pub fn holders(p: *const Pool, id: u32) u32 {
        return p.ids.holders(id);
    }

    /// Free pages past the promised ones.
    pub fn available(p: *const Pool) usize {
        return p.ids.available();
    }

    /// Slot `slot`'s kv heads: the model's, or the MTP head's on the last slot.
    pub fn kvOf(p: *const Pool, slot: usize) usize {
        return if (p.head_kv != null and slot + 1 == p.keys.len) p.head_kv.? else p.kv_heads;
    }

    /// Bytes of one page over every attention layer.
    pub fn pageBytes(p: *const Pool) usize {
        var n: usize = 0;
        for (0..p.keys.len) |i| n += bytes(1, 1, p.kvOf(i), p.head_dim, p.act);
        return n;
    }

    /// Attention layer `slot`'s pools read through a stream's page `table` (int32 ids on the device).
    pub fn paged(p: *const Pool, slot: usize, table: u64) hip.ops.Ops.Paged {
        return .{
            .k = p.keys[slot].base(),
            .v = p.values[slot].base(),
            .table = table,
            .kind = p.act,
            .kv_heads = p.kvOf(slot),
            .d = p.head_dim,
            .count = p.count,
        };
    }
};

/// Words of a stream's descriptor before its pool slots: the capacity, and the last kept row (unused here).
pub const header_words = 2;

/// One stream: its descriptor (the plan kernels' view of its pages), positions kept, every Mamba layer's state.
pub const Stream = struct {
    gpa: Allocator,
    /// u64 words: the capacity, 0, each pool slot's keys and values, then the page table (int32 ids); `ids` the same.
    desc: hip.DeviceBuffer,
    ids: std.ArrayList(u32) = .empty,
    /// Bytes into `desc` where the page table starts.
    table_at: usize,
    cap: usize,
    len: usize = 0,
    mtp_len: usize = 0,
    /// Every Mamba layer's conv tail and SSM state, one buffer; `states` by the layer's place among the Mamba layers.
    mem: hip.DeviceBuffer,
    states: []mamba.State,
    /// Pages promised to this stream (counted in the pool's `reserved` until taken).
    promised: usize = 0,
    /// A follower's stream: its table is rank 0's, set by `set`, and the pool holds none of it.
    mirror: bool = false,

    /// Device bytes of a stream's Mamba state on this rank over `layers` Mamba layers.
    pub fn stateBytes(d: model.Dims, act: model.Kind, layers: usize) usize {
        return layers * (mamba.convBytes(d, act) + mamba.ssmBytes(d));
    }

    /// Bytes of the descriptor of a stream of `cap` positions over `slots` pool slots.
    pub fn descBytes(slots: usize, cap: usize) usize {
        return 8 * (header_words + 2 * slots) + 4 * std.mem.alignForward(usize, @max(1, pagesFor(cap)), 2);
    }

    pub fn init(
        gpa: Allocator,
        r: *const hip.Runtime,
        pool: *const Pool,
        d: model.Dims,
        act: model.Kind,
        layers: usize,
        cap: usize,
    ) !Stream {
        const slots = pool.keys.len;
        const table_at = 8 * (header_words + 2 * slots);
        const words = try gpa.alloc(u64, header_words + 2 * slots);
        defer gpa.free(words);
        words[0] = cap;
        words[1] = 0;
        for (pool.keys, pool.values, 0..) |k, v, i| words[header_words + 2 * i ..][0..2].* = .{ k.base(), v.base() };
        var desc = try hip.DeviceBuffer.alloc(r, descBytes(slots, cap));
        errdefer desc.free();
        try desc.upload(0, std.mem.sliceAsBytes(words));
        var mem = try hip.DeviceBuffer.alloc(r, @max(4, stateBytes(d, act, layers)));
        errdefer mem.free();
        try mem.fill8(0);
        const states = try gpa.alloc(mamba.State, layers);
        const conv = mamba.convBytes(d, act);
        const each = conv + mamba.ssmBytes(d);
        for (states, 0..) |*s, i| s.* = .{ .conv = mem.base() + i * each, .ssm = mem.base() + i * each + conv };
        return .{ .gpa = gpa, .desc = desc, .table_at = table_at, .cap = cap, .mem = mem, .states = states };
    }

    /// The pages back to `pool`, the buffers freed.
    pub fn deinit(s: *Stream, pool: *Pool) void {
        if (!s.mirror) for (s.ids.items) |id| pool.release(id);
        pool.ids.reserved -= @min(pool.ids.reserved, s.promised);
        s.ids.deinit(s.gpa);
        s.gpa.free(s.states);
        s.mem.free();
        s.desc.free();
        s.* = undefined;
    }

    /// The pages back to `pool` and the positions dropped, the buffers kept for another request (states zeroed after).
    pub fn empty(s: *Stream, pool: *Pool) void {
        if (!s.mirror) for (s.ids.items) |id| pool.release(id);
        pool.ids.reserved -= @min(pool.ids.reserved, s.promised);
        s.ids.clearRetainingCapacity();
        s.promised = 0;
        s.mirror = false;
        s.len = 0;
        s.mtp_len = 0;
    }

    /// The page table's device address (int32 ids), as the attention kernels read it.
    pub fn table(s: *const Stream) u64 {
        return s.desc.base() + s.table_at;
    }

    /// The Mamba state zeroed (a fresh prompt from position 0).
    pub fn clear(s: *Stream, stream: hip.Stream) !void {
        try s.mem.fill8Async(0, stream);
    }

    fn put(s: *Stream) !void {
        try s.desc.upload(s.table_at, std.mem.sliceAsBytes(s.ids.items));
    }

    /// Pages to cover `n` positions, taken from `pool` and the table sent up; refused past the stream's capacity.
    pub fn cover(s: *Stream, pool: *Pool, n: usize) !void {
        if (n > s.cap) return error.StreamFull;
        const want = pagesFor(n);
        if (want <= s.ids.items.len) return;
        if (s.mirror) return error.OutOfPages;
        while (s.ids.items.len < want) {
            try s.ids.append(s.gpa, pool.take() orelse return error.OutOfPages);
            if (s.promised > 0) {
                s.promised -= 1;
                pool.ids.reserved -= 1;
            }
        }
        try s.put();
    }

    /// A follower's table from page `first` on is rank 0's `ids`: the pool is not asked, the table sent up.
    pub fn set(s: *Stream, first: usize, ids: []const u32) !void {
        if (first + ids.len > pagesFor(s.cap)) return error.StreamFull;
        s.mirror = true;
        if (first + ids.len > s.ids.items.len) try s.ids.resize(s.gpa, first + ids.len);
        @memcpy(s.ids.items[first..][0..ids.len], ids);
        try s.put();
    }

    /// The leading pages replaced by `ids` (held once more by this stream when `retain`; the old ones released).
    pub fn adopt(s: *Stream, pool: *Pool, ids: []const u32, retain: bool) !void {
        if (retain) for (ids) |id| pool.retain(id);
        for (s.ids.items[0..@min(ids.len, s.ids.items.len)]) |id| pool.release(id);
        if (ids.len > s.ids.items.len) try s.ids.resize(s.gpa, ids.len);
        @memcpy(s.ids.items[0..ids.len], ids);
        try s.put();
    }
};

test "a pool of pages holds keys and values of every attention layer" {
    try std.testing.expectEqual(@as(usize, 2 * 7 * 10 * 2 * 64 * 128 * 2), Pool.bytes(7, 10, 2, 128, .bf16));
}

test "a descriptor holds the header, two addresses a slot and an even count of table words" {
    try std.testing.expectEqual(@as(usize, 8 * (2 + 2 * 7) + 4 * 2), Stream.descBytes(7, 64));
    try std.testing.expectEqual(@as(usize, 8 * (2 + 2 * 7) + 4 * 4), Stream.descBytes(7, 65 * 3));
}
