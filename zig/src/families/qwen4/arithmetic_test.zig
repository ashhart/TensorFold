//! Exact synthetic shared-graph checks do not qualify a Metal or CUDA device implementation.
const std = @import("std");
const cfg = @import("config.zig");
const graph = @import("graph.zig");
const cpu = @import("../../../tests/qwen4/cpu_ops.zig");
const State = @import("../../../tests/qwen4/state.zig").State;
const a = std.testing.allocator;
const Oracle = struct { schema: u32, oracle: []const u8, tokens: []const u32, logits: [][]f64, picks: []const u32, trace: std.json.Value };
fn setup(arena: std.mem.Allocator) !struct { doc: std.json.Parsed(Oracle), config: cfg.Config } {
    var p: cfg.Problem = .{};
    return .{ .doc = try std.json.parseFromSlice(Oracle, arena, @embedFile("fixtures/tiny-binary64.json"), .{}), .config = try cfg.parse(arena, @embedFile("fixtures/tiny-dense.json"), &p) };
}
fn run(ops: *cpu.Ops, c: *const cfg.Config, state: *State, tokens: []const u32) !cpu.Ops.Tensor {
    var ls: [cfg.max_layers]graph.Layer = undefined;
    return graph.forward(ops, try graph.geometry(c, try graph.layers(c, &ls)), state, tokens);
}
test "Qwen4 arithmetic: whole shared graph matches the independent binary64 fixture" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    const fixture = try setup(mem);
    const c = &fixture.config;
    var state = try State.init(mem, c);
    var ops: cpu.Ops = .{ .a = mem, .c = c };
    const out = try run(&ops, c, &state, fixture.doc.value.tokens);
    var different: usize = 0;
    for (fixture.doc.value.logits, 0..) |row, i| for (row, out.row(i)) |expected, actual| {
        const bf = @import("../../../tests/qwen4/math.zig").bf;
        if (bf(@floatCast(expected)) != actual) different += 1;
    };
    if (different != 0) std.debug.print("QWEN4_F64 different={d}/{d}\n", .{ different, out.bits.len });
    try std.testing.expectEqual(@as(usize, 0), different);
}
test "Qwen4 arithmetic: prompt cuts concurrent isolation and resumed state are bit exact" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    const fixture = try setup(mem);
    const c = &fixture.config;
    const tokens = fixture.doc.value.tokens;
    var ops: cpu.Ops = .{ .a = mem, .c = c };
    var full = try State.init(mem, c);
    const baseline = try run(&ops, c, &full, tokens);
    for (1..tokens.len) |cut| {
        var cut_state = try State.init(mem, c);
        const first = try run(&ops, c, &cut_state, tokens[0..cut]);
        var resumed = try cut_state.clone(mem);
        const last = try run(&ops, c, &resumed, tokens[cut..]);
        try std.testing.expectEqualSlices(u16, baseline.bits[0..first.bits.len], first.bits);
        try std.testing.expectEqualSlices(u16, baseline.bits[first.bits.len..], last.bits);
    }
    var left = try State.init(mem, c);
    var right = try State.init(mem, c);
    for (tokens, 0..) |token, i| {
        const lone = try run(&ops, c, &left, &.{token});
        const other: u32 = @intCast((i * 7 + 1) % c.vocab);
        _ = try run(&ops, c, &right, &.{other});
        try std.testing.expectEqualSlices(u16, baseline.row(i), lone.bits);
    }
}

test "Qwen4 lexical history: shared hash rejects a corrupt token without mutating history" {
    const ng = @import("hybrid_ngram");
    const policy = try ng.Policy.init(a, 3, 2, 32, 2, 17, 1234, 0);
    defer policy.deinit();
    var history = [_]u32{ 1, 2 };
    var ids: [4]usize = undefined;
    try policy.step(&history, 7, &ids);
    const across = ids;
    var reset = [_]u32{ 2, 2 };
    try policy.step(&reset, 7, &ids);
    try std.testing.expectEqualSlices(usize, &across, &ids);
    history[0] = 32;
    const before = history;
    try std.testing.expectError(error.NgramTokenOutOfRange, policy.step(&history, 4, &ids));
    try std.testing.expectEqualSlices(u32, &before, &history);
}

test "Qwen4 GDN arithmetic: BF16 convolution rounds before SiLU while F32 convolution does not" {
    const m = @import("../../../tests/qwen4/math.zig");
    const x: f32 = -0.609375;
    const w: f32 = -0.578125;
    const acc = x * w;
    try std.testing.expectEqual(@as(u16, 0x3e54), m.bf(m.silu(acc)));
    try std.testing.expectEqual(@as(u16, 0x3e53), m.bf(m.silu(m.round(acc))));
    const g = @import("cuda_gdn.zig");
    try std.testing.expectEqual(@as(u4, 11), g.weightFlags(true, true));
    try std.testing.expectEqual(@as(u4, 15), g.weightFlags(true, false));
    try std.testing.expectEqual(@as(u4, 3), g.weightFlags(false, true));
    try std.testing.expectEqual(@as(u4, 7), g.weightFlags(false, false));
}
