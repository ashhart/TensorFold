//! LFM2.5-24B-A2B's kept states: liquid's over this model (the conv states, then the ten attention layers' pages).
const std = @import("std");
const liquid = @import("liquid_cuda");
const mdl = @import("cuda_model.zig");
const kern = @import("cuda_kernels.zig");
const kv = @import("core").kv_pool;

const S = liquid.snap.States(mdl);
pub const Saved = S.Saved;
pub const bytes = S.bytes;
pub const save = S.save;
pub const restore = S.restore;
pub const drop = S.drop;
pub const page = liquid.snap.page;

const page_bytes = mdl.page_elems * 2;
const row_bytes = mdl.kv_width * 2;
const state_bytes = mdl.slot_elems * 2;

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
    try S.each(&m, 1, at, null, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 1 + 2 + 20), runs.items.len);
    try testing.expectEqual([3]usize{ (1 << 36) + state_bytes, 0, state_bytes }, runs.items[0]);
    try testing.expectEqual([3]usize{ (1 << 40) + 7 * page_bytes, state_bytes, 2 * page_bytes }, runs.items[1]);
    try testing.expectEqual([3]usize{ (1 << 40) + 3 * page_bytes, state_bytes + 2 * page_bytes, page_bytes }, runs.items[2]);
    const v9 = (1 << 40) + 9 * page_bytes + kern.page * row_bytes; // page 9, layer 0's values
    try testing.expectEqual([3]usize{ v9, state_bytes + 3 * page_bytes + 100 * row_bytes, 100 * row_bytes }, runs.items[4]);
    var total: usize = 0;
    for (runs.items) |r| total += r[2];
    try testing.expectEqual(state_bytes + at * 20 * 1024, total);
    try testing.expectEqual((total + page - 1) / page * page, S.bytes(at));
}

test "a state on whole pages has no partial copies; one past a slot's pages is refused" {
    var ids = [_]u32{ 4, 5, 6, 7 };
    var caps = [_]usize{2 * kern.page};
    const m = fake(&ids, &caps);
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try S.each(&m, 0, 2 * kern.page, null, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 2), runs.items.len);
    try testing.expectEqual(@as(usize, 2 * page_bytes), runs.items[1][2]);
    try testing.expectError(error.PastPages, S.each(&m, 0, 2 * kern.page + 1, null, Count{ .runs = &runs }, Count.add));
}

/// A fake driver's queue log: which stream each copy, record and wait went to, and the sampler's mark.
const Order = struct {
    const Op = struct { kind: enum { copy, record, wait, sample }, stream: usize, event: usize = 0 };
    var ops: [128]Op = undefined;
    var n: usize = 0;
    const main: usize = 1;
    const side: usize = 2;
    const ready: usize = 3;
    const done: usize = 4;

    fn add(op: Op) void {
        ops[n] = op;
        n += 1;
    }
    fn id(h: ?*anyopaque) usize {
        return @intFromPtr(h);
    }
    fn dtod(_: u64, _: u64, _: usize, st: @import("cuda").abi.Stream) callconv(.c) @import("cuda").abi.Result {
        add(.{ .kind = .copy, .stream = id(st) });
        return 0;
    }
    fn record(ev: @import("cuda").abi.Event, st: @import("cuda").abi.Stream) callconv(.c) @import("cuda").abi.Result {
        add(.{ .kind = .record, .stream = id(st), .event = id(ev) });
        return 0;
    }
    fn wait(st: @import("cuda").abi.Stream, ev: @import("cuda").abi.Event, _: c_uint) callconv(.c) @import("cuda").abi.Result {
        add(.{ .kind = .wait, .stream = id(st), .event = id(ev) });
        return 0;
    }

    /// Whether the first token waits for a keep: before the sampler, `main` has a copy or waits for `done`.
    fn firstWaits() bool {
        for (ops[0..n]) |op| {
            if (op.kind == .sample) return false;
            if (op.stream == main and (op.kind == .copy or (op.kind == .wait and op.event == done))) return true;
        }
        return true; // no sampler: nothing to show
    }

    /// Whether a restore's copies on `main` (after the sampler) come after `main` waited for every keep's `done`.
    fn restoreWaits() bool {
        var last_done: ?usize = null; // the last record of `done` on the keeps' stream
        var waited: ?usize = null;
        var sampled = false;
        for (ops[0..n], 0..) |op, i| switch (op.kind) {
            .sample => sampled = true,
            .record => if (op.event == done) {
                last_done = i;
            },
            .wait => if (op.stream == main and op.event == done) {
                waited = i;
            },
            .copy => if (sampled and op.stream == main) return if (last_done) |d| if (waited) |w| w > d else false else true,
        };
        return true;
    }

    /// Every keep copy on the keeps' stream comes after it waited for `ready`, recorded on `main`.
    fn keepAfterPass() bool {
        var armed = false;
        for (ops[0..n]) |op| switch (op.kind) {
            .wait => if (op.stream == side and op.event == ready) {
                armed = true;
            },
            .copy => if (op.stream == side and !armed) return false,
            else => {},
        };
        return true;
    }
};

test "a keep's copies run beside the first token's draw, after the prompt pass, and a restore waits for them" {
    const cu = @import("cuda");
    var d: cu.Driver = undefined;
    d.api.cuMemcpyDtoDAsync_v2 = Order.dtod;
    d.api.cuEventRecord = Order.record;
    d.api.cuStreamWaitEvent = Order.wait;
    const main: cu.Stream = .{ .d = &d, .handle = @ptrFromInt(Order.main) };
    var ids = [_]u32{ 4, 5, 6, 7 };
    var caps = [_]usize{2 * kern.page};
    var m = fake(&ids, &caps);
    m.d = &d;
    var pages: liquid.snap.Pages = .{
        .pages = try kv.Pool.init(testing.allocator, 16, page),
        .arena = .{ .d = &d, .ptr = 1 << 44, .len = 16 * page, .borrowed = true },
        .side = .{ .stream = .{ .d = &d, .handle = @ptrFromInt(Order.side) }, .ready = .{ .d = &d, .handle = @ptrFromInt(Order.ready) }, .done = .{ .d = &d, .handle = @ptrFromInt(Order.done) } },
    };
    defer pages.pages.deinit();
    // the real order: keep, the sampler, then a restore
    Order.n = 0;
    const k = try S.save(testing.allocator, &pages, &m, main, 0, kern.page + 7, null);
    Order.add(.{ .kind = .sample, .stream = Order.main });
    try S.restore(&pages, &m, main, 0, k);
    try testing.expect(!Order.firstWaits() and Order.restoreWaits() and Order.keepAfterPass());
    try testing.expect(!pages.side.?.pending);
    S.drop(testing.allocator, &pages, k);
    // planted: the copies back on `main` ahead of the sampler (as before this change); the check must see the wait
    Order.n = 0;
    var lease = try pages.take(bytes(kern.page + 7) / page);
    defer pages.pages.finish(&lease);
    try S.each(&m, 0, kern.page + 7, null, liquid.snap.Leased{ .d = &d, .s = main, .arena = 1 << 44, .ids = lease.ids.items, .out = true }, liquid.snap.copy);
    Order.add(.{ .kind = .sample, .stream = Order.main });
    try testing.expect(Order.firstWaits());
    // planted: a restore without the fence reads the pages while the keeps' copies may still run
    Order.n = 0;
    const k2 = try S.save(testing.allocator, &pages, &m, main, 0, kern.page + 7, null);
    Order.add(.{ .kind = .sample, .stream = Order.main });
    try S.each(&m, 0, k2.at, null, liquid.snap.Leased{ .d = &d, .s = main, .arena = 1 << 44, .ids = k2.lease.ids.items, .out = false }, liquid.snap.copy);
    try testing.expect(!Order.restoreWaits());
    pages.side.?.pending = false;
    S.drop(testing.allocator, &pages, k2);
    // planted: the copies on the keeps' stream before it waited for the prompt pass
    Order.n = 0;
    try S.each(&m, 0, kern.page + 7, null, liquid.snap.Leased{ .d = &d, .s = pages.side.?.stream, .arena = 1 << 44, .ids = lease.ids.items, .out = true }, liquid.snap.copy);
    try testing.expect(!Order.keepAfterPass());
}
