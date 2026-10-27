//! A stream's kept state (conv states, K/V [0, at)) in pages allocated once in the prompt-cache budget, as GLM's.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const mdl = @import("cuda_model.zig");
const kern = @import("cuda_kernels.zig");

/// A kept state: its position and the snapshot pages that hold it, in order.
pub const Saved = struct { at: u32, lease: kv.Lease };

const page_bytes = mdl.page_elems * 2;
const row_bytes = mdl.kv_width * 2;
const state_bytes = mdl.slot_elems * 2;

/// A snapshot page: the unit the budget is cut into.
pub const page: usize = 2 << 20;

/// Bytes a state at `at` holds: whole snapshot pages.
pub fn bytes(at: usize) u64 {
    const n = state_bytes + at * (page_bytes / kern.page);
    return (n + page - 1) / page * page;
}

/// The budget's pages, allocated once; states lease them and give them back.
pub const Pages = struct {
    pages: kv.Pool,
    arena: ?cuda.DeviceBuffer,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, budget: u64) !Pages {
        const count = std.math.cast(u32, budget / page) orelse return error.SnapshotBudget;
        var pages = try kv.Pool.init(gpa, count, page);
        errdefer pages.deinit();
        return .{ .pages = pages, .arena = if (count > 0) try cuda.DeviceBuffer.alloc(d, @as(usize, count) * page) else null };
    }

    pub fn deinit(p: *Pages) void {
        if (p.arena) |*a| a.free();
        p.pages.deinit();
    }

    /// Bytes of pages no state holds.
    pub fn spare(p: *const Pages) u64 {
        return p.pages.available() * page;
    }

    fn take(p: *Pages, n: usize) !kv.Lease {
        if (n == 0 or n > p.pages.available() or p.arena == null) return error.NoSnapshotRoom;
        var lease: kv.Lease = .{};
        errdefer p.pages.finish(&lease);
        try p.pages.promise(&lease, n);
        try p.pages.cover(&lease, n, null);
        return lease;
    }
};

/// `slot`'s state at `at`, region by region: one copy a run of consecutive whole pages, a layer's rows of the last.
fn each(m: *const mdl.Model, slot: usize, at: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    if (at > m.caps[slot]) return error.PastPages;
    try f(ctx, m.states + slot * state_bytes, 0, state_bytes);
    var off: usize = state_bytes;
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
    for (0..2 * mdl.attention_layers) |i| { // layer a's keys at block 2a, its values at 2a + 1
        try f(ctx, last + i * kern.page * row_bytes, off, rows * row_bytes);
        off += rows * row_bytes;
    }
}

/// Region bytes [off, off + len) over a state's pages: (arena offset, region offset, length), a run of consecutive ids.
fn pageRuns(ids: []const u32, off: usize, len: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), held: u64, done: usize, n: usize) anyerror!void) !void {
    var done: usize = 0;
    while (done < len) {
        const at = off + done;
        var q = at / page;
        var n = @min(len - done, page - at % page);
        while (done + n < len and ids[q + 1] == ids[q] + 1) : (q += 1) n = @min(len - done, n + page);
        try f(ctx, @as(u64, ids[at / page]) * page + at % page, done, n);
        done += n;
    }
}

/// Copies between a slot's region at `dev` and a lease's pages.
const Copy = struct { d: *const cuda.Driver, s: cuda.Stream, arena: u64, dev: u64, out: bool };

fn copyRun(c: Copy, held: u64, done: usize, n: usize) anyerror!void {
    const src = if (c.out) c.dev + done else c.arena + held;
    const dst = if (c.out) c.arena + held else c.dev + done;
    try c.d.check(c.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, c.s.handle), "cuMemcpyDtoDAsync");
}

const Leased = struct { d: *const cuda.Driver, s: cuda.Stream, arena: u64, ids: []const u32, out: bool };

fn copy(c: Leased, dev: u64, off: usize, len: usize) anyerror!void {
    try pageRuns(c.ids, off, len, Copy{ .d = c.d, .s = c.s, .arena = c.arena, .dev = dev, .out = c.out }, copyRun);
}

/// Copy `slot`'s state at `at` into leased pages, queued on `s` behind the chunk that ends there.
pub fn save(gpa: std.mem.Allocator, p: *Pages, m: *const mdl.Model, s: cuda.Stream, slot: usize, at: u32) !*Saved {
    if (at == 0 or at > m.context) return error.NotAtMark;
    const k = try gpa.create(Saved);
    errdefer gpa.destroy(k);
    k.* = .{ .at = at, .lease = try p.take(bytes(at) / page) };
    errdefer p.pages.finish(&k.lease);
    try each(m, slot, at, Leased{ .d = m.d, .s = s, .arena = p.arena.?.ptr, .ids = k.lease.ids.items, .out = true }, copy);
    return k;
}

/// Make `slot`'s pages and conv states hold `k`'s state, queued on `s` before the next chunk.
pub fn restore(p: *const Pages, m: *const mdl.Model, s: cuda.Stream, slot: usize, k: *const Saved) !void {
    if (k.at > m.context) return error.PastContext;
    try each(m, slot, k.at, Leased{ .d = m.d, .s = s, .arena = p.arena.?.ptr, .ids = k.lease.ids.items, .out = false }, copy);
}

/// Give a state's pages back: every copy that reads or writes them is queued on the one stream, ahead of any later one.
pub fn drop(gpa: std.mem.Allocator, p: *Pages, k: *Saved) void {
    p.pages.finish(&k.lease);
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
    m.ids = ids;
    m.caps = caps;
    return m;
}

test "a state: the conv states, consecutive whole pages as one copy, the last page's rows a layer's K and V" {
    // slot 1 leases pages 7, 8, 3, 9: positions 0..1023 one run, 1024..1535 a page, 1536.. a partial page
    var ids = [_]u32{ 0, 0, 0, 0, 7, 8, 3, 9 };
    var caps = [_]usize{ 0, 4 * kern.page };
    const m = fake(&ids, &caps);
    const at = 3 * kern.page + 100;
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 1 + 2 + 20), runs.items.len);
    try testing.expectEqual([3]usize{ (1 << 36) + state_bytes, 0, state_bytes }, runs.items[0]);
    try testing.expectEqual([3]usize{ (1 << 40) + 7 * page_bytes, state_bytes, 2 * page_bytes }, runs.items[1]);
    try testing.expectEqual([3]usize{ (1 << 40) + 3 * page_bytes, state_bytes + 2 * page_bytes, page_bytes }, runs.items[2]);
    const v9 = (1 << 40) + 9 * page_bytes + kern.page * row_bytes; // page 9, layer 0's values
    try testing.expectEqual([3]usize{ v9, state_bytes + 3 * page_bytes + 100 * row_bytes, 100 * row_bytes }, runs.items[4]);
    var total: usize = 0;
    for (runs.items) |r| total += r[2];
    try testing.expectEqual(state_bytes + at * 20 * 1024, total);
    try testing.expectEqual((total + page - 1) / page * page, bytes(at));
}

test "a state on whole pages has no partial copies; one past a slot's pages is refused" {
    var ids = [_]u32{ 4, 5, 6, 7 };
    var caps = [_]usize{2 * kern.page};
    const m = fake(&ids, &caps);
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 0, 2 * kern.page, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 2), runs.items.len);
    try testing.expectEqual(@as(usize, 2 * page_bytes), runs.items[1][2]);
    try testing.expectError(error.PastPages, each(&m, 0, 2 * kern.page + 1, Count{ .runs = &runs }, Count.add));
}

const Run = struct {
    out: *std.ArrayList([3]usize),
    fn add(r: Run, held: u64, done: usize, n: usize) anyerror!void {
        try r.out.append(testing.allocator, .{ held, done, n });
    }
};

test "a region over snapshot pages: one copy a run of consecutive ids, split where the ids jump" {
    const ids = [_]u32{ 4, 5, 9, 10, 11, 2 };
    var out: std.ArrayList([3]usize) = .empty;
    defer out.deinit(testing.allocator);
    // bytes [page / 2, 5 page + 10): pages 4-5 (from mid-page), 9-11, then 2
    try pageRuns(&ids, page / 2, 4 * page + page / 2 + 10, Run{ .out = &out }, Run.add);
    try testing.expectEqual(@as(usize, 3), out.items.len);
    try testing.expectEqual([3]usize{ 4 * page + page / 2, 0, page + page / 2 }, out.items[0]);
    try testing.expectEqual([3]usize{ 9 * page, page + page / 2, 3 * page }, out.items[1]);
    try testing.expectEqual([3]usize{ 2 * page, 4 * page + page / 2, 10 }, out.items[2]);
    out.clearRetainingCapacity();
    try pageRuns(&ids, 3 * page + 7, 100, Run{ .out = &out }, Run.add); // inside one page
    try testing.expectEqual([3]usize{ 10 * page + 7, 0, 100 }, out.items[0]);
}

test "a state's pages: whole snapshot pages, the budget's pages leased and given back" {
    try testing.expectEqual(@as(u64, page), bytes(1));
    try testing.expect(bytes(10_000) >= state_bytes + 10_000 * 20 * 1024 and bytes(10_000) % page == 0);
}
