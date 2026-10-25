//! Actual adapter and engine owners under memory recording; zero-weight draws do not qualify GPU arithmetic.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const L = @import("core").lanes;
const support = @import("engine_test.zig");
const E = @import("engine.zig").Engine(support.G);
const Tensor = E.Tensor;
pub const D = struct {
    rows: usize = 16,
    out: [256]u32 = @splat(0),
    calls: usize = 0,
    reads: usize = 0,
    pub fn bytes(vocab: usize, rows: usize, streams: usize) !usize {
        if (vocab == 0 or rows == 0 or streams == 0) return error.BadDrawShape;
        return rows * 8 + streams * 4;
    }
    pub fn deinitDrained(_: *D) void {}
    pub fn draw(d: *D, x: Tensor, off: usize, positions: []const u64, sampling: ?L.Sampling, bits: []const u32) !void {
        try @import("draw_inputs.zig").check(x.width, positions, sampling, bits);
        try t.expectEqual(x.rows, positions.len);
        try t.expect(off % 4 == 0 and off / 4 + positions.len <= d.out.len);
        @memset(d.out[off / 4 ..][0..positions.len], 0);
        d.calls += 1;
    }
    pub fn read(d: *D, off: usize, out: []u32) !void {
        @memcpy(out, d.out[off / 4 ..][0..out.len]);
        d.reads += 1;
    }
};
pub const A = @import("lanes_adapter.zig").Adapter(E, D, L);
pub const Fixture = struct {
    tmp: std.testing.TmpDir,
    mem: std.heap.ArenaAllocator,
    b: *support.B,
    e: *E,
    d: *D,
    adapter: *A,
    pub fn open() !Fixture {
        var mem: std.heap.ArenaAllocator = .init(a);
        errdefer mem.deinit();
        const m = mem.allocator();
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const text = @embedFile("fixtures/native-dense.json");
        var p: @import("config.zig").Problem = .{};
        var c = try @import("config.zig").parse(a, text, &p);
        defer c.deinit();
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = text });
        try @import("reader_fixture.zig").write(tmp, m, &c, false);
        const path = try std.fmt.allocPrint(m, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        const b = try a.create(support.B);
        b.* = .{};
        errdefer a.destroy(b);
        const e = try E.open(a, std.testing.io, path, b, .{ .capacity = 64, .window = 4, .streams = 4, .budget = std.math.maxInt(u64) }, &p, support.Loader{});
        errdefer e.close() catch unreachable;
        const d = try a.create(D);
        d.* = .{};
        errdefer a.destroy(d);
        const adapter = try a.create(A);
        errdefer a.destroy(adapter);
        adapter.* = try A.init(a, e, d);
        return .{ .tmp = tmp, .mem = mem, .b = b, .e = e, .d = d, .adapter = adapter };
    }
    pub fn close(f: *Fixture) void {
        for (f.adapter.lanes) |lane| if (lane.stream) |s| f.adapter.backend().release(s);
        f.b.drain() catch unreachable;
        f.adapter.deinitDrained();
        a.destroy(f.adapter);
        a.destroy(f.d);
        f.e.close() catch unreachable;
        t.expectEqual(@as(usize, 0), f.b.held) catch unreachable;
        a.destroy(f.b);
        f.tmp.cleanup();
        f.mem.deinit();
    }
};
test "Qwen4 lanes: shared verify, prefix rejection, stale handles and cold-cut prompt hooks use the actual native owner" {
    var f = try Fixture.open();
    defer f.close();
    const b = f.adapter.backend();
    var left = try L.Stream.init(a, .{ .id = "left", .prompt = &.{ 1, 2, 3, 4, 5, 6 }, .chunks = &.{3}, .max_new = 10 });
    defer left.deinit(a);
    var right = try L.Stream.init(a, .{ .id = "right", .prompt = &.{ 7, 8 }, .max_new = 10 });
    defer right.deinit(a);
    try t.expect(!try b.vtable.prefill_step.?(b.ptr, &left));
    try t.expectEqual(@as(u32, 3), left.fill_at.load(.acquire));
    try t.expect(try b.vtable.prefill_step.?(b.ptr, &left));
    try b.prefill(&right);
    const lh = try b.first(&left, 6);
    const rh = try b.first(&right, 2);
    try t.expect(lh != rh);
    try t.expectEqual(@as(u32, 0), try b.read(lh));
    try t.expectEqual(@as(u32, 0), try b.read(rh));
    const windows = [_]L.backend.Window{ .{ .stream = &left, .pending = 1, .held = 0, .tokens = &.{ 2, 3 }, .parents = null, .positions = &.{ 7, 8, 9 } }, .{ .stream = &right, .pending = 7, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{3} } };
    var sample0: [3]u32 = undefined;
    var sample1: [1]u32 = undefined;
    var drafts0: [2]u32 = undefined;
    var drafts1: [0]u32 = .{};
    var verified = [_]L.backend.Verified{ .{ .sampled = &sample0, .drafts = &drafts0 }, .{ .sampled = &sample1, .drafts = &drafts1 } };
    const begins = f.e.gpu.begins;
    var bad = windows;
    bad[1].positions = &.{99};
    try t.expectError(error.Qwen4DrawPosition, b.verify(&bad, &verified));
    try t.expectEqual(begins, f.e.gpu.begins);
    try b.verify(&windows, &verified);
    try t.expectEqual(@as(usize, 4), f.e.gpu.project_rows);
    try t.expectEqualSlices(u32, &.{ 2, 3 }, &drafts0);
    try t.expectEqualSlices(u32, &.{ 0, 0, 0 }, &sample0);
    try t.expectError(error.Qwen4KeepPath, b.keep(&windows, &.{ &.{ 0, 2 }, &.{0} }));
    try b.keep(&windows, &.{ &.{ 0, 1 }, &.{0} });
    try t.expectEqual(@as(usize, 8), (try f.adapter.find(&left)).slot.?.state.pos);
    b.release(&left);
    try t.expectError(error.Qwen4StaleDrawHandle, b.read(lh));
    try t.expectEqual(@as(u32, 0), try b.read(rh));
    b.release(&right);
}
test "Qwen4 lanes: kept full-prompt logits resume without replay and cancelled fills keep only completed rows" {
    var f = try Fixture.open();
    defer f.close();
    const b = f.adapter.backend();
    var source = try L.Stream.init(a, .{ .id = "source", .prompt = &.{ 1, 2, 3, 4, 5 }, .max_new = 8 });
    defer source.deinit(a);
    try b.prefill(&source);
    const lane = try f.adapter.find(&source);
    var snap = try A.Snap.open(a, &lane.slot.?.state, f.adapter.key, std.math.maxInt(u64));
    defer snap.close() catch unreachable;
    try snap.captureWithLogits(&lane.slot.?.state, f.adapter.key, f.adapter.last(lane));
    b.release(&source);
    const begins = f.e.gpu.begins;
    var resumed = try L.Stream.init(a, .{ .id = "resume", .prompt = &.{ 1, 2, 3, 4, 5 }, .max_new = 8, .reuse = .{ .at = 5, .saved = &snap } });
    defer resumed.deinit(a);
    try b.prefill(&resumed);
    try t.expectEqual(@as(u32, 5), resumed.cached);
    try t.expectEqual(begins + 1, f.e.gpu.begins);
    try t.expectEqual(@as(u32, 0), try b.read(try b.first(&resumed, 5)));
    b.release(&resumed);
    const Hook = struct {
        at: u32 = 0,
        pub fn save(p: *anyopaque, _: *L.Stream, at: u32) void {
            const h: *@This() = @ptrCast(@alignCast(p));
            h.at = at;
        }
    };
    var hook: Hook = .{};
    var cancelled = try L.Stream.init(a, .{ .id = "cancelled", .prompt = &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, .max_new = 8, .reuse = .{ .hook = .{ .ptr = &hook, .at = Hook.save } } });
    defer cancelled.deinit(a);
    try t.expect(!try b.vtable.prefill_step.?(b.ptr, &cancelled));
    const submitted = f.e.gpu.submits;
    try b.vtable.cancel_prefill.?(b.ptr, &cancelled);
    try t.expectEqual(@as(u32, 4), hook.at);
    try t.expectEqual(submitted, f.e.gpu.submits);
    b.release(&cancelled);
}
test "Qwen4 lanes: shared lane core drives copied proposals and plain streams through one backend" {
    var f = try Fixture.open();
    defer f.close();
    const b = f.adapter.backend();
    var config = try L.Config.init(a, f.adapter.facts(), 4, 3);
    defer config.deinit(a);
    config.timed = false;
    var clock: L.backend.WallClock = .{ .io = std.testing.io };
    var core = L.Engine.init(a, &config, b, clock.clock());
    defer core.deinit();
    const Proposal = struct {
        fn give(_: *anyopaque, _: []const u32, max: i64) ![]const u32 {
            const ids = &[_]u32{ 0, 0, 0 };
            return ids[0..@intCast(@max(0, @min(max, ids.len)))];
        }
        fn matched(_: *anyopaque) i64 {
            return 32;
        }
    };
    var proposer_id: u8 = 0;
    const proposer: L.proposer.Proposer = .{ .ptr = &proposer_id, .vtable = &.{ .propose = Proposal.give, .last_match = Proposal.matched, .priced = true } };
    var drafted = try L.Stream.init(a, .{ .id = "drafted", .prompt = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, .max_new = 8, .drafts = true, .proposer = proposer });
    defer drafted.deinit(a);
    var plain = try L.Stream.init(a, .{ .id = "plain", .prompt = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, .max_new = 8, .drafts = false });
    defer plain.deinit(a);
    try core.addStream(&drafted);
    try core.addStream(&plain);
    var steps: usize = 0;
    while (core.activeCount() > 0) {
        try t.expect(steps < 16);
        try core.step();
        steps += 1;
    }
    try t.expectEqualSlices(u32, plain.emitted(), drafted.emitted());
    try t.expectEqual(@as(usize, 8), drafted.emitted().len);
    try t.expect(drafted.drafted > 0);
}
