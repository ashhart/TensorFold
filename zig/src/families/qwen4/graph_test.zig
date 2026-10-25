//! Recording the shared graph proves layer ownership and pre-write refusal, not GPU arithmetic.
const std = @import("std");
const cfg = @import("config.zig");
const graph = @import("graph.zig");
const shared = @import("hybrid_graph");
const a = std.testing.allocator;
const State = struct {
    n: usize = 2,
    pos: usize = 0,
    cap: usize = 64,
    poisoned: bool = false,
    pub fn layers(s: *@This()) usize {
        return s.n;
    }
    pub fn admit(s: *@This(), rows: usize) !void {
        if (s.poisoned) return error.Poisoned;
        if (rows > s.cap - s.pos) return error.ContextFull;
    }
    pub fn advanced(s: *@This(), tokens: []const u32) !void {
        s.pos += tokens.len;
    }
    pub fn poison(s: *@This()) void {
        s.poisoned = true;
    }
};
const Event = enum { embed, ple, hc_norm, hc_down, hc_activate, hc_up, hc_mix, hc_inject, hc_gates, delta, indexed, writeback, gate, value, swiglu, output, head, routed };
const Ops = struct {
    pub const Tensor = usize;
    events: [128]Event = undefined,
    count: usize = 0,
    fail: ?Event = null,
    fn event(o: *@This(), e: Event) !Tensor {
        if (o.fail == e) return error.PlantedDeviceFailure;
        o.events[o.count] = e;
        o.count += 1;
        return o.count;
    }
    pub fn embed(o: *@This(), _: []const u32, _: usize, _: usize) !Tensor {
        return o.event(.embed);
    }
    pub fn ple(o: *@This(), i: usize, _: *State, _: []const u32, _: Tensor) !Tensor {
        try std.testing.expectEqual(@as(usize, 0), i);
        return o.event(.ple);
    }
    pub fn hcNorm(o: *@This(), _: usize, _: shared.Boundary, _: Tensor, _: usize, _: f32) !Tensor {
        return o.event(.hc_norm);
    }
    pub fn hcProject(o: *@This(), _: usize, _: shared.Boundary, p: shared.Projection, _: Tensor) !Tensor {
        return o.event(switch (p) {
            .down => .hc_down,
            .up => .hc_up,
            .inject => .hc_inject,
            else => unreachable,
        });
    }
    pub fn hcActivate(o: *@This(), _: Tensor, _: usize) !Tensor {
        return o.event(.hc_activate);
    }
    pub fn hcMix(o: *@This(), _: Tensor, _: Tensor, _: usize) !Tensor {
        return o.event(.hc_mix);
    }
    pub fn hcInjection(o: *@This(), _: Tensor, _: usize) !Tensor {
        return o.event(.hc_gates);
    }
    pub fn delta(o: *@This(), i: usize, _: *State, _: Tensor) !Tensor {
        try std.testing.expectEqual(@as(usize, 0), i);
        return o.event(.delta);
    }
    pub fn indexed(o: *@This(), i: usize, _: *State, _: Tensor) !Tensor {
        try std.testing.expectEqual(@as(usize, 1), i);
        return o.event(.indexed);
    }
    pub fn writeback(o: *@This(), _: Tensor, _: Tensor, _: Tensor) !Tensor {
        return o.event(.writeback);
    }
    pub fn denseProject(o: *@This(), _: usize, p: shared.Projection, _: Tensor) !Tensor {
        return o.event(switch (p) {
            .gate => .gate,
            .value => .value,
            .output => .output,
            else => unreachable,
        });
    }
    pub fn swiglu(o: *@This(), _: Tensor, _: Tensor) !Tensor {
        return o.event(.swiglu);
    }
    pub fn head(o: *@This(), _: Tensor) !Tensor {
        return o.event(.head);
    }
    pub fn routed(o: *@This(), _: usize, _: Tensor) !Tensor {
        return o.event(.routed);
    }
};
test "Qwen4 graph: same shared layer ordering, dense MLP replaces only routed boundary" {
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &p);
    defer c.deinit();
    var storage: [2]graph.Layer = undefined;
    const ls = try graph.layers(&c, &storage);
    const g = try graph.geometry(&c, ls);
    var state: State = .{};
    var o: Ops = .{};
    _ = try graph.forward(&o, g, &state, &.{ 3, 5, 8 });
    try std.testing.expectEqual(@as(usize, 3), state.pos);
    try std.testing.expect(!state.poisoned);
    const es = o.events[0..o.count];
    try std.testing.expectEqual(Event.embed, es[0]);
    try std.testing.expectEqual(Event.ple, es[1]);
    try std.testing.expectEqual(Event.head, es[es.len - 1]);
    for ([_]Event{ .delta, .indexed, .ple, .head }) |e| try std.testing.expectEqual(@as(usize, 1), std.mem.count(Event, es, &.{e}));
    for ([_]Event{ .gate, .value, .swiglu, .output }) |e| try std.testing.expectEqual(@as(usize, 2), std.mem.count(Event, es, &.{e}));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(Event, es, &.{.routed}));
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(Event, es, &.{.writeback}));
    try std.testing.expectEqual(@as(usize, 5), std.mem.count(Event, es, &.{.hc_norm}));
    storage[0].feed_forward = .routed;
    storage[1].feed_forward = .routed;
    var routed_ops: Ops = .{};
    var routed_state: State = .{};
    var routed_graph = g;
    routed_graph.layers = &storage;
    _ = try shared.forward(&routed_ops, routed_graph, &routed_state, &.{ 3, 5, 8 });
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(Event, routed_ops.events[0..routed_ops.count], &.{.routed}));
    try std.testing.expectError(error.RoutedQwen4Layer, graph.geometry(&c, &storage));
}
test "Qwen4 graph: malformed state is refused before operations and queued failures poison it" {
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &p);
    defer c.deinit();
    var storage: [2]graph.Layer = undefined;
    const g = try graph.geometry(&c, try graph.layers(&c, &storage));
    var o: Ops = .{};
    var state: State = .{ .cap = 2 };
    try std.testing.expectError(error.ContextFull, graph.forward(&o, g, &state, &.{ 3, 5, 8 }));
    try std.testing.expectEqual(@as(usize, 0), o.count);
    try std.testing.expectError(error.TokenOutOfRange, graph.forward(&o, g, &state, &.{32}));
    try std.testing.expectEqual(@as(usize, 0), o.count);
    state.n = 1;
    try std.testing.expectError(error.WrongStateLayout, graph.forward(&o, g, &state, &.{3}));
    try std.testing.expectEqual(@as(usize, 0), o.count);
    state = .{};
    o.fail = .swiglu;
    try std.testing.expectError(error.PlantedDeviceFailure, graph.forward(&o, g, &state, &.{3}));
    try std.testing.expect(state.poisoned);
    try std.testing.expectEqual(@as(usize, 0), state.pos);
    const n = o.count;
    try std.testing.expectError(error.Poisoned, graph.forward(&o, g, &state, &.{3}));
    try std.testing.expectEqual(n, o.count);
}
