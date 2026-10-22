//! Planning a disk restore must honor memory, compatibility and measured savings without mutating the cache.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const fake = @import("rows_fake.zig");
const t = std.testing;
const a = t.allocator;
const Rig = struct {
    family: fake.Fake = .{ .gpa = a },
    disk: rows.Rows = undefined,
    store: pc.Store = undefined,
    prompt: [64]u32 = undefined,
    fn init(x: *Rig) void {
        for (&x.prompt, 0..) |*token, i| token.* = @intCast(101 + i * 7);
        x.family.prompt = &x.prompt;
        const f = x.family.snapshots();
        x.disk = .{ .gpa = a, .family = f, .img = f.vtable.rows.?, .root = undefined, .dir = undefined, .keying = .{ .identity = 17, .block = rows.block_tokens, .lookahead = 1 }, .cap = 1 << 30, .daily = 1 << 30, .read_rate = 1000, .prefill_rate = 100 };
        x.store = pc.Store.init(a, f, .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 }, 1 << 20);
        x.store.disk = &x.disk;
    }
    fn deinit(x: *Rig) void {
        x.store.disk = null;
        for (x.store.entries.items) |e| e.borrowers = 0;
        x.store.deinit();
        for (x.disk.states.items) |state| a.free(state.tokens);
        x.disk.states.deinit(a);
    }
    fn add(x: *Rig, at: u32, cost: u64) !void {
        const tokens = try a.dupe(u32, x.prompt[0 .. @as(usize, at) + 1]);
        errdefer a.free(tokens);
        try x.disk.states.append(a, .{ .name = @splat(0), .at = at, .tokens = tokens, .spans = &.{}, .starts = &.{}, .own = cost, .rows = 0, .used = 0, .shared = false });
    }
    fn status(x: *Rig) rows.read.Resume {
        return rows.read.resumable(&x.store, &x.prompt, &.{}, &.{});
    }
    fn bytes(x: *Rig, at: u32) u64 {
        return x.store.family.vtable.bytes(x.store.family.ptr, at);
    }
    fn hold(x: *Rig, at: u32) !*pc.Entry {
        x.store.disk = null;
        defer x.store.disk = &x.disk;
        x.family.at = at;
        try t.expect(x.store.keep(&x.prompt, at, null, &.{}, &.{}));
        return x.store.entries.items[x.store.entries.items.len - 1];
    }
};
test "restore selection: loose tight and reduced budgets choose feasible prefixes only" {
    var x: Rig = .{};
    x.init();
    defer x.deinit();
    try x.add(10, 20);
    try x.add(30, 1000);
    x.disk.prefill_rate = 0;
    try t.expectEqual(@as(u32, 30), x.status().disk);
    x.store.budget = x.bytes(10);
    try t.expectEqual(@as(u32, 10), x.status().disk);
    try t.expect(x.status().from == .disk);
    x.store.budget = x.bytes(10) - 1;
    try t.expect(x.status().from == .none and x.status().disk == 0);
    try t.expectEqual(@as(u64, 0), x.store.counts.evicted);
    try t.expectEqual(@as(u64, 0), x.disk.counts.reads);
    x.store.disk = null;
    try t.expect(x.status().from == .none);
}
test "restore selection: the greatest saved time beats the longest or cheapest file" {
    var x: Rig = .{};
    x.init();
    defer x.deinit();
    try x.add(8, 70);
    try x.add(10, 20);
    try x.add(20, 150);
    try x.add(30, 400);
    const status = x.status();
    try t.expect(status.from == .disk and status.disk == 10 and status.disk_bytes == 20);
    try t.expectApproxEqAbs(@as(f64, 0.56), status.seconds, 1e-12);
    _ = try x.hold(15);
    const cached = x.status();
    try t.expect(cached.from == .ram and cached.ram == 15);
    try t.expectApproxEqAbs(@as(f64, 0.49), cached.seconds, 1e-12);
}
test "restore selection: equal savings and unknown rates preserve deterministic longest feasible ties" {
    for ([_]bool{ false, true }) |reverse| {
        var x: Rig = .{};
        x.init();
        defer x.deinit();
        x.disk.prefill_rate = 64;
        x.disk.read_rate = 128;
        try x.add(if (reverse) 16 else 8, if (reverse) 31 else 15);
        try x.add(if (reverse) 8 else 16, if (reverse) 15 else 31);
        try t.expectEqual(@as(u32, 16), x.status().disk);
        for ([_]f64{ 0, -1, std.math.nan(f64), std.math.inf(f64) }) |rate| {
            x.disk.prefill_rate = rate;
            try t.expect(x.status().from == .disk and x.status().disk == 16);
            try t.expect(std.math.isFinite(x.status().seconds));
            x.disk.prefill_rate = 64;
            x.disk.read_rate = rate;
            try t.expect(x.status().from == .disk and x.status().disk == 16);
            try t.expect(std.math.isFinite(x.status().seconds));
            x.disk.read_rate = 128;
        }
    }
}
test "restore selection: protected bytes and slots leave borrowers and fallback untouched" {
    var x: Rig = .{};
    x.init();
    defer x.deinit();
    try x.add(10, 20);
    const one = try x.hold(5);
    const two = try x.hold(6);
    one.borrowers = 1;
    two.borrowers = 1;
    const held = x.store.held;
    x.store.budget = held + x.bytes(10) - 1;
    try t.expect(x.status().from == .ram and x.status().ram == 6);
    for ([_]usize{ 0, 2, 10 }) |slots| {
        x.store.most = slots;
        try t.expect(x.status().from == .ram);
        try t.expectEqual(held, x.store.held);
        try t.expectEqual(@as(u32, 1), one.borrowers);
        try t.expectEqual(@as(u32, 1), two.borrowers);
        try t.expectEqual(@as(u64, 0), x.store.counts.evicted);
    }
    x.store.budget += 1;
    x.store.most = 3;
    try t.expect(x.status().from == .disk and x.status().disk == 10);
    x.store.most = 2;
    try t.expect(x.status().from == .ram);
}
test "restore selection: lookahead roles and planned chunk boundaries remain compatible" {
    var x: Rig = .{};
    x.init();
    defer x.deinit();
    try x.add(10, 1);
    x.prompt[10] += 1;
    try t.expect(x.status().from == .none);
    x.prompt[10] -= 1;
    try t.expect(x.status().from == .disk);
    var spans = [_][2]u32{.{ 2, 10 }};
    x.disk.states.items[0].spans = &spans;
    try t.expect(x.status().from == .none);
    const requested = [_][2]u32{.{ 2, 15 }};
    try t.expect(rows.read.resumable(&x.store, &x.prompt, &.{}, &requested).from == .disk);
    x.disk.keying.planned = true;
    var starts = [_]u32{ 3, 8 };
    x.disk.states.items[0].starts = &starts;
    try t.expect(rows.read.resumable(&x.store, &x.prompt, &.{ 3, 8, 10, 20 }, &requested).from == .disk);
    try t.expect(rows.read.resumable(&x.store, &x.prompt, &.{ 3, 7, 10, 20 }, &requested).from == .none);
    try t.expect(rows.read.resumable(&x.store, &x.prompt, &.{ 3, 8, 11, 20 }, &requested).from == .none);
}
