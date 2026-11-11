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

/// The keeps' own stream: copies start after the prompt pass (`ready`); `main` waits (`done`) before its next write.
pub const Side = struct {
    stream: cuda.Stream,
    ready: cuda.Event,
    done: cuda.Event,
    pending: bool = false, // copies queued that the prompt stream has not waited for yet

    pub fn init(d: *const cuda.Driver) !Side {
        var st = try cuda.Stream.init(d, true);
        errdefer st.deinit();
        var r = try cuda.Event.init(d, false);
        errdefer r.deinit();
        return .{ .stream = st, .ready = r, .done = try cuda.Event.init(d, false) };
    }

    pub fn deinit(x: *Side) void {
        x.stream.deinit(); // drains the copies first
        x.done.deinit();
        x.ready.deinit();
    }

    /// Before a keep's copies: they start after everything queued on `main` so far.
    fn begin(x: *Side, main: cuda.Stream) !void {
        try x.ready.record(main);
        try x.stream.wait(x.ready);
    }

    /// After a keep's copies (or some of them): `done` marks them; if it can't, they are drained here.
    fn end(x: *Side) void {
        x.done.record(x.stream) catch {
            x.stream.synchronize() catch std.log.warn("lfm2_moe: a keep's copies did not drain", .{});
            return;
        };
        x.pending = true;
    }

    /// Before `main` writes what a keep reads or reads what it wrote: wait on the device for the keeps queued so far.
    pub fn fence(x: *Side, main: cuda.Stream) !void {
        if (!x.pending) return;
        try main.wait(x.done);
        x.pending = false;
    }
};

/// The budget's pages, allocated once; states lease them and give them back.
pub const Pages = struct {
    pages: kv.Pool,
    arena: ?cuda.DeviceBuffer,
    side: ?Side = null, // the keeps' stream (none without pages)

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, budget: u64) !Pages {
        const count = std.math.cast(u32, budget / page) orelse return error.SnapshotBudget;
        var pages = try kv.Pool.init(gpa, count, page);
        errdefer pages.deinit();
        if (count == 0) return .{ .pages = pages, .arena = null };
        var arena = try cuda.DeviceBuffer.alloc(d, @as(usize, count) * page);
        errdefer arena.free();
        return .{ .pages = pages, .arena = arena, .side = try Side.init(d) };
    }

    pub fn deinit(p: *Pages) void {
        if (p.side) |*x| x.deinit();
        if (p.arena) |*a| a.free();
        p.pages.deinit();
    }

    /// Before `main` writes a slot's caches or reads snapshot pages: the keeps queued so far land first.
    pub fn fence(p: *Pages, main: cuda.Stream) !void {
        if (p.side) |*x| try x.fence(main);
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

/// `slot`'s state at `at`: conv states (from `conv`), a copy a run of consecutive whole pages, the last's rows a layer.
fn each(m: *const mdl.Model, slot: usize, conv: u64, at: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    if (at > m.caps[slot]) return error.PastPages;
    try f(ctx, conv, 0, state_bytes);
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

/// `slot`'s state at `at` (conv states at `conv`) copied on the keeps' stream, after what `main` has queued.
pub fn save(gpa: std.mem.Allocator, p: *Pages, m: *const mdl.Model, main: cuda.Stream, slot: usize, at: u32, conv: u64) !*Saved {
    if (at == 0 or at > m.context) return error.NotAtMark;
    const x = if (p.side) |*x| x else return error.NoSnapshotRoom;
    const k = try gpa.create(Saved);
    errdefer gpa.destroy(k);
    k.* = .{ .at = at, .lease = try p.take(bytes(at) / page) };
    errdefer p.pages.finish(&k.lease);
    try x.begin(main);
    defer x.end(); // copies already queued still read the slot: `main` waits for them as for whole ones
    try each(m, slot, conv, at, Leased{ .d = m.d, .s = x.stream, .arena = p.arena.?.ptr, .ids = k.lease.ids.items, .out = true }, copy);
    return k;
}

/// Make `slot`'s pages and conv states hold `k`'s state: queued on `main` before the next chunk, after the keeps.
pub fn restore(p: *Pages, m: *const mdl.Model, main: cuda.Stream, slot: usize, k: *const Saved) !void {
    if (k.at > m.context) return error.PastContext;
    try p.fence(main);
    try each(m, slot, m.states + slot * state_bytes, k.at, Leased{ .d = m.d, .s = main, .arena = p.arena.?.ptr, .ids = k.lease.ids.items, .out = false }, copy);
}

/// Give a state's pages back: a later keep writes them after what `main` queued before it (a restore reading them).
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
    try each(&m, 1, m.states + state_bytes, at, Count{ .runs = &runs }, Count.add);
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
    try each(&m, 0, m.states, 2 * kern.page, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 2), runs.items.len);
    try testing.expectEqual(@as(usize, 2 * page_bytes), runs.items[1][2]);
    try testing.expectError(error.PastPages, each(&m, 0, m.states, 2 * kern.page + 1, Count{ .runs = &runs }, Count.add));
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
    var pages: Pages = .{
        .pages = try kv.Pool.init(testing.allocator, 16, page),
        .arena = .{ .d = &d, .ptr = 1 << 44, .len = 16 * page, .borrowed = true },
        .side = .{ .stream = .{ .d = &d, .handle = @ptrFromInt(Order.side) }, .ready = .{ .d = &d, .handle = @ptrFromInt(Order.ready) }, .done = .{ .d = &d, .handle = @ptrFromInt(Order.done) } },
    };
    defer pages.pages.deinit();
    // the real order: keep, the sampler, then a restore
    Order.n = 0;
    const k = try save(testing.allocator, &pages, &m, main, 0, kern.page + 7, m.states);
    Order.add(.{ .kind = .sample, .stream = Order.main });
    try restore(&pages, &m, main, 0, k);
    try testing.expect(!Order.firstWaits() and Order.restoreWaits() and Order.keepAfterPass());
    try testing.expect(!pages.side.?.pending);
    drop(testing.allocator, &pages, k);
    // planted: the copies back on `main` ahead of the sampler (as before this change); the check must see the wait
    Order.n = 0;
    var lease = try pages.take(bytes(kern.page + 7) / page);
    defer pages.pages.finish(&lease);
    try each(&m, 0, m.states, kern.page + 7, Leased{ .d = &d, .s = main, .arena = 1 << 44, .ids = lease.ids.items, .out = true }, copy);
    Order.add(.{ .kind = .sample, .stream = Order.main });
    try testing.expect(Order.firstWaits());
    // planted: a restore without the fence reads the pages while the keeps' copies may still run
    Order.n = 0;
    const k2 = try save(testing.allocator, &pages, &m, main, 0, kern.page + 7, m.states);
    Order.add(.{ .kind = .sample, .stream = Order.main });
    try each(&m, 0, m.states, k2.at, Leased{ .d = &d, .s = main, .arena = 1 << 44, .ids = k2.lease.ids.items, .out = false }, copy);
    try testing.expect(!Order.restoreWaits());
    pages.side.?.pending = false;
    drop(testing.allocator, &pages, k2);
    // planted: the copies on the keeps' stream before it waited for the prompt pass
    Order.n = 0;
    try each(&m, 0, m.states, kern.page + 7, Leased{ .d = &d, .s = pages.side.?.stream, .arena = 1 << 44, .ids = lease.ids.items, .out = true }, copy);
    try testing.expect(!Order.keepAfterPass());
}
