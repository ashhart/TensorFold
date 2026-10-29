//! Nonzero checkpoints exercise the real native engine and state owners with CPU numeric instruments, never a GPU.
const std = @import("std");
const n = @import("../../../tests/qwen4/numeric_gpu.zig");
const m = @import("../../../tests/qwen4/math.zig");
const cfg = @import("config.zig");
const E = @import("engine.zig").Engine(n.G);
const t = std.testing;
const a = t.allocator;
const Oracle = struct { schema: u32, oracle: []const u8, tokens: []const u32, logits: [][]f64, picks: []const u32, trace: std.json.Value };
const Capture = struct {
    out: []u16,
    pub fn encode(c: Capture, _: *n.G, logits: n.Tensor, _: []const E.Segment) !void {
        try t.expectEqual(c.out.len, n.bits(logits).len);
        @memcpy(c.out, n.bits(logits));
    }
};
pub const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    tmp: std.testing.TmpDir,
    b: *n.B,
    e: *E,
    doc: std.json.Parsed(Oracle),
    pub fn open() !Fixture {
        var arena: std.heap.ArenaAllocator = .init(a);
        errdefer arena.deinit();
        const mem = arena.allocator();
        const text = @embedFile("fixtures/native-dense.json");
        var p: cfg.Problem = .{};
        var c = try cfg.parse(a, text, &p);
        defer c.deinit();
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = text });
        try @import("reader_fixture.zig").writeNonzero(tmp, mem, &c);
        const path = try std.fmt.allocPrint(mem, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        const b = try mem.create(n.B);
        b.* = .{};
        const e = try E.open(a, std.testing.io, path, b, .{ .capacity = 32, .window = 16, .streams = 2, .budget = std.math.maxInt(u64) }, &p, n.Loader{});
        errdefer e.close() catch unreachable;
        const doc = try std.json.parseFromSlice(Oracle, mem, @import("device_fixture.zig").cuda_reference, .{});
        return .{ .arena = arena, .tmp = tmp, .b = b, .e = e, .doc = doc };
    }
    pub fn deinit(f: *Fixture) void {
        f.e.close() catch unreachable;
        t.expectEqual(@as(usize, 0), f.b.held) catch unreachable;
        f.tmp.cleanup();
        f.arena.deinit();
    }
    fn run(f: *Fixture, slot: *E.Slot, tokens: []const u32) ![]u16 {
        const out = try f.arena.allocator().alloc(u16, tokens.len * f.e.c.vocab);
        var p: cfg.Problem = .{};
        try f.e.forward(&.{.{ .state = &slot.state, .tokens = tokens }}, Capture{ .out = out }, &p);
        return out;
    }
};
test "Qwen4 native numeric: loaded nonzero weights and actual packed graph match independent f64 nodes" {
    var f = try Fixture.open();
    defer f.deinit();
    const slot = try f.e.lease();
    const out = try f.run(slot, f.doc.value.tokens);
    var different: usize = 0;
    for (f.doc.value.logits, 0..) |row, r| for (row, 0..) |expected, d| {
        if (m.bf(@floatCast(expected)) != out[r * f.e.c.vocab + d]) different += 1;
    };
    if (different != 0) std.debug.print("QWEN4_NATIVE_F64 different={d}/{d}\n", .{ different, out.len });
    try t.expectEqual(@as(usize, 0), different);
    try t.expect(f.e.scratch.peak <= f.e.plan.scratch);
    try t.expect(f.b.held <= f.e.plan.device);
}
test "Qwen4 native numeric: every prompt cut, saved state, concurrent stream and partial rejection are exact" {
    var f = try Fixture.open();
    defer f.deinit();
    const e = f.e;
    const left = try e.lease();
    const right = try e.lease();
    const tokens = f.doc.value.tokens;
    const baseline = try f.run(left, tokens);
    try e.keep(left, tokens.len);
    for (1..tokens.len) |cut| {
        try left.state.reset();
        const first = try f.run(left, tokens[0..cut]);
        try e.keep(left, cut);
        var snap = try @import("snapshot.zig").Snapshot(n.B).open(a, &left.state, @splat(17), std.math.maxInt(u64));
        defer snap.close() catch unreachable;
        try snap.capture(&left.state, @splat(17));
        try right.state.reset();
        try snap.restore(&right.state, @splat(17));
        const last = try f.run(right, tokens[cut..]);
        try e.keep(right, tokens.len - cut);
        try t.expectEqualSlices(u16, baseline[0..first.len], first);
        try t.expectEqualSlices(u16, baseline[first.len..], last);
    }
    try left.state.reset();
    try right.state.reset();
    const other = [_]u32{ 19, 1, 4, 17, 9, 6, 12, 24, 8, 3, 5, 11 };
    const other_baseline = try f.run(right, &other);
    try e.keep(right, other.len);
    try right.state.reset();
    var p: cfg.Problem = .{};
    for (0..tokens.len) |r| {
        const shared = try f.arena.allocator().alloc(u16, 2 * e.c.vocab);
        try e.forward(&.{ .{ .state = &left.state, .tokens = tokens[r..][0..1] }, .{ .state = &right.state, .tokens = other[r..][0..1] } }, Capture{ .out = shared }, &p);
        try t.expectEqualSlices(u16, baseline[r * e.c.vocab ..][0..e.c.vocab], shared[0..e.c.vocab]);
        try t.expectEqualSlices(u16, other_baseline[r * e.c.vocab ..][0..e.c.vocab], shared[e.c.vocab..]);
        try e.keep(left, 1);
        try e.keep(right, 1);
    }
    for (0..tokens.len) |accepted| {
        try left.state.reset();
        _ = try f.run(left, tokens);
        try e.keep(left, accepted);
        const retry = try f.run(left, tokens[accepted..]);
        try t.expectEqualSlices(u16, baseline[accepted * e.c.vocab ..], retry);
        try e.keep(left, tokens.len - accepted);
    }
    try left.state.reset();
    _ = try f.run(left, tokens[0..7]);
    try left.state.abort();
    const retry = try f.run(left, tokens);
    try t.expectEqualSlices(u16, baseline, retry);
}
