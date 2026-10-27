//! A stream's prompt state at a position: its conv states, its rings, the full layers' keys and values [0, at).
const std = @import("std");
const cuda = @import("cuda");
const mdl = @import("cuda_model.zig");
const kern = @import("cuda_kernels.zig");

/// A kept state: the slot's conv states and rings, its whole pages in order, the last page's rows a layer's K and V.
pub const Saved = struct { at: u32, buf: cuda.DeviceBuffer };

const page_bytes = mdl.page_elems * 2;
const row_bytes = mdl.kv_width * 2;
const state_bytes = mdl.slot_elems * 2;
const ring_bytes = mdl.ring * row_bytes; // one sliding layer's keys (or values) of a slot

/// Bytes a state at `at` takes.
pub fn bytes(at: usize) u64 {
    return mdl.Model.fixed_bytes + at * (page_bytes / kern.page);
}

/// `slot`'s state at `at`, region by region: one copy a run of consecutive whole pages, a layer's rows of the last.
fn each(m: *const mdl.Model, slot: usize, at: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    if (at > m.caps[slot]) return error.PastPages;
    try f(ctx, m.states + slot * state_bytes, 0, state_bytes);
    var off: usize = state_bytes;
    for (0..mdl.sliding_layers) |l| for ([_]bool{ false, true }) |value| { // a ring holds positions, not rows
        try f(ctx, m.ringOf(l, value) + slot * ring_bytes, off, ring_bytes);
        off += ring_bytes;
    };
    const ids = m.ids[slot * m.perSlot() ..];
    const whole = at / kern.page;
    var q: usize = 0;
    while (q < whole) {
        var n: usize = 1;
        while (q + n < whole and ids[q + n] == ids[q] + n) n += 1;
        try f(ctx, m.arena + @as(u64, ids[q]) * page_bytes, off, n * page_bytes);
        off += n * page_bytes;
        q += n;
    }
    const rows = at % kern.page;
    if (rows == 0) return;
    const last = m.arena + @as(u64, ids[whole]) * page_bytes;
    for (0..2 * mdl.full_layers) |i| { // layer a's keys at block 2a, its values at 2a + 1
        try f(ctx, last + i * kern.page * row_bytes, off, rows * row_bytes);
        off += rows * row_bytes;
    }
}

const Copy = struct { d: *const cuda.Driver, s: cuda.Stream, buf: u64, out: bool };

fn copy(c: Copy, dev: u64, off: usize, len: usize) anyerror!void {
    const src = if (c.out) dev else c.buf + off;
    const dst = if (c.out) c.buf + off else dev;
    try c.d.check(c.d.api.cuMemcpyDtoDAsync_v2(dst, src, len, c.s.handle), "cuMemcpyDtoDAsync");
}

/// Copy `slot`'s state at `at` into new storage, queued on `s` behind the chunk that ends there.
pub fn save(gpa: std.mem.Allocator, m: *const mdl.Model, s: cuda.Stream, slot: usize, at: u32) !*Saved {
    if (at == 0 or at > m.context) return error.NotAtMark;
    const k = try gpa.create(Saved);
    errdefer gpa.destroy(k);
    k.* = .{ .at = at, .buf = try cuda.DeviceBuffer.alloc(m.d, bytes(at)) };
    errdefer k.buf.free();
    try each(m, slot, at, Copy{ .d = m.d, .s = s, .buf = k.buf.ptr, .out = true }, copy);
    return k;
}

/// Make `slot`'s pages and conv states hold `k`'s state, queued on `s` before the next chunk.
pub fn restore(m: *const mdl.Model, s: cuda.Stream, slot: usize, k: *const Saved) !void {
    if (k.at > m.context) return error.PastContext;
    try each(m, slot, k.at, Copy{ .d = m.d, .s = s, .buf = k.buf.ptr, .out = false }, copy);
}

/// Free a state once the stream's queued copies of it are done.
pub fn drop(gpa: std.mem.Allocator, s: cuda.Stream, k: *Saved) void {
    s.synchronize() catch {};
    k.buf.free();
    gpa.destroy(k);
}

const testing = std.testing;

const Count = struct {
    runs: *std.ArrayList([3]usize),
    fn add(c: Count, dev: u64, off: usize, len: usize) anyerror!void {
        try c.runs.append(testing.allocator, .{ dev, off, len });
    }
};

fn fake(ids: []u32, caps: []usize) mdl.Model {
    var m: mdl.Model = undefined;
    m.context = 4 * kern.page;
    m.slots = caps.len;
    m.arena = 1 << 40;
    m.states = 1 << 36;
    m.rings = 1 << 38;
    m.ids = ids;
    m.caps = caps;
    return m;
}

test "a state: conv states, every ring, consecutive whole pages as one copy, the last page's rows a layer's K and V" {
    // slot 1 leases pages 7, 8, 3, 9: positions 0..1023 one run, 1024..1535 a page, 1536.. a partial page
    var ids = [_]u32{ 0, 0, 0, 0, 7, 8, 3, 9 };
    var caps = [_]usize{ 0, 4 * kern.page };
    const m = fake(&ids, &caps);
    const at = 3 * kern.page + 100;
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, Count{ .runs = &runs }, Count.add);
    const rings = 2 * mdl.sliding_layers;
    try testing.expectEqual(@as(usize, 1 + rings + 2 + 12), runs.items.len);
    try testing.expectEqual([3]usize{ (1 << 36) + state_bytes, 0, state_bytes }, runs.items[0]);
    // sliding layer 0's values: the second ring block, slot 1's ring in it
    try testing.expectEqual([3]usize{ (1 << 38) + 2 * mdl.ring * row_bytes + ring_bytes, state_bytes + ring_bytes, ring_bytes }, runs.items[2]);
    const fixed = mdl.Model.fixed_bytes;
    try testing.expectEqual([3]usize{ (1 << 40) + 7 * page_bytes, fixed, 2 * page_bytes }, runs.items[1 + rings]);
    try testing.expectEqual([3]usize{ (1 << 40) + 3 * page_bytes, fixed + 2 * page_bytes, page_bytes }, runs.items[2 + rings]);
    const v9 = (1 << 40) + 9 * page_bytes + kern.page * row_bytes; // page 9, full layer 0's values
    try testing.expectEqual([3]usize{ v9, fixed + 3 * page_bytes + 100 * row_bytes, 100 * row_bytes }, runs.items[4 + rings]);
    var total: usize = 0;
    for (runs.items) |r| total += r[2];
    try testing.expectEqual(bytes(at), total);
    try testing.expectEqual(@as(u64, fixed + at * 12 * 1024), bytes(at));
}

test "a state on whole pages has no partial copies; one past a slot's pages is refused" {
    var ids = [_]u32{ 4, 5, 6, 7 };
    var caps = [_]usize{2 * kern.page};
    const m = fake(&ids, &caps);
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 0, 2 * kern.page, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 2 + 2 * mdl.sliding_layers), runs.items.len);
    try testing.expectEqual(@as(usize, 2 * page_bytes), runs.items[runs.items.len - 1][2]);
    try testing.expectError(error.PastPages, each(&m, 0, 2 * kern.page + 1, Count{ .runs = &runs }, Count.add));
}
