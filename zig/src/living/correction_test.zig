//! The actual pocket output fitter corrects a planted belief at its located layers and leaves a neighbour exact.
const std = @import("std");
const pocket = @import("pocket.zig");
const modes = @import("train_mode.zig");
const examples = @import("lesson_examples.zig");
const route = @import("lesson_route.zig");
const family = @import("family.zig");
const locate = @import("pocket_locate.zig");
const correction = @import("pocket_correction.zig");
const forgetting = @import("pocket_forget.zig");
const t = std.testing;
const d = 64;
const layers = 8;
const P = pocket.Pocket(Toy, @import("families.zig").nemotron_h);
const new_ids = [_]u32{ 2, 9, 7, 0 };
const old_ids = [_]u32{ 2, 9, 7, 1 };
const held_ids = [_]u32{ 8, 9, 4, 0 };
const neighbour = [_]u32{ 2, 10, 7, 2 };
const train: examples.Example = .{ .ids = &new_ids, .start = 3, .decide = 2, .subject = 1 };
const old: examples.Example = .{ .ids = &old_ids, .start = 3, .decide = 2, .subject = 1 };
const held: examples.Example = .{ .ids = &held_ids, .start = 3, .decide = 2, .subject = 1 };
const Toy = struct {
    p: *P,
    bias: [layers][3]f64 = @splat(.{ 0, 0, 0 }),
    current: []const u32 = &new_ids,
    ablated: ?usize = null,
    scale: f64 = 1,
    spans: [1]modes.Span = undefined,
    pack: usize = 16,
    from: [layers]u32 = @splat(std.math.maxInt(u32)),
    nudge: modes.Nudge = .{},
    count: @import("pocket_clock.zig").Count = .{},
    prefixes: struct {
        pub const most = 1;
    } = .{},
    input: [d]u16 = @splat(0),
    residual_row: [d]u16 = @splat(0x3f80),
    gradient: [d]f32 = @splat(0),
    delta: [d]f32 = @splat(0),
    quiet_gradient: [d]f32 = @splat(0),
    pub fn packs(_: *@This()) bool {
        return false;
    }
    pub fn prefixSlots(_: *@This(), _: usize) !void {}
    pub fn config(_: *@This()) struct { hidden: usize, layers: usize } {
        return .{ .hidden = d, .layers = layers };
    }
    pub fn normed(m: *@This(), _: usize, row: usize) []const u16 {
        @memset(&m.input, 0);
        if (row == 1) m.input[if (m.current[1] == 9) 0 else 1] = 0x3f80;
        return &m.input;
    }
    pub fn residual(m: *@This(), _: usize, _: usize) []const u16 {
        return &m.residual_row;
    }
    fn reaches(layer: usize) bool {
        return layer == 1 or layer == 5;
    }
    pub fn pushAt(m: *@This(), _: usize, _: usize, layer: usize) []const f32 {
        return if (reaches(layer)) &m.gradient else &m.quiet_gradient;
    }
    pub fn deltaAt(m: *@This(), _: usize, _: usize, _: usize) []f32 {
        return &m.delta;
    }
    fn logits(m: *@This(), ids: []const u32) [3]f64 {
        m.current = ids;
        var out: [3]f64 = .{ 0, 0, 0.2 };
        if (ids[1] == 9) {
            for (m.bias, 0..) |b, l| if (m.ablated != l) for (&out, b) |*o, v| {
                o.* += v;
            };
        } else out[2] = 2;
        for (m.p.moe.items, 0..) |l, slot| for (0..m.p.used.items[slot]) |n| {
            // The subject-to-answer readout is absent at other layers, including the planted wrong placement.
            if (!reaches(l)) continue;
            const activation = m.p.act(slot, n, m.normed(l, 1));
            const v = m.p.neuron(m.p.down.items, slot, n, d);
            for (&out, v[0..3]) |*o, value| o.* += @as(f64, activation) * m.scale * value;
        };
        return out;
    }
    fn best(scores: [3]f64) u32 {
        var at: u32 = 0;
        for (scores, 0..) |v, i| if (v > scores[at]) {
            at = @intCast(i);
        };
        return at;
    }
    fn loss(scores: [3]f64, target: u32) f64 {
        var total: f64 = 0;
        for (scores) |v| total += @exp(v);
        return @log(total) - scores[target];
    }
    pub fn pass(m: *@This(), seqs: []const modes.Seq, _: modes.Mode, _: route.Turn, out: []modes.Result) !void {
        const ex = seqs[0];
        m.scale = ex.scale;
        const scores = m.logits(ex.ids);
        var total: f64 = 0;
        for (scores) |v| total += @exp(v);
        @memset(&m.gradient, 0);
        const target = ex.ids[ex.start];
        for (scores, m.gradient[0..3], 0..) |v, *g, i| g.* = @floatCast(@exp(v) / total - @as(f64, if (i == target) 1 else 0));
        m.spans[0] = .{ .at = 0, .rows = ex.ids.len - 1, .skip = 0, .slot = 0 };
        m.count.passes += 1;
        m.count.rows += ex.ids.len - 1;
        out[0] = .{ .loss = @floatCast(loss(scores, target)), .recalled = best(scores) == target };
    }
    pub fn step(m: *@This(), ids: []const u32, start: usize, mode: modes.Mode, turn: route.Turn) !modes.Result {
        var result: [1]modes.Result = undefined;
        try m.pass(&.{.{ .ids = ids, .start = start, .decide = turn.decide, .nudge = m.nudge }}, mode, turn, &result);
        return result[0];
    }
    pub fn causalLogit(m: *@This(), ex: examples.Example, layer: ?usize, row: usize) !f32 {
        m.scale = 1;
        m.ablated = if (row == 1) layer else null;
        defer m.ablated = null;
        return @floatCast(m.logits(ex.ids)[ex.ids[ex.start]]);
    }
};
const Host = struct {
    io: std.Io = t.io,
    pub fn layout(_: @This()) family.Layout {
        var all: family.Layers = .empty;
        for (0..layers) |l| all.set(l);
        return .fromExperts(d, 128, layers, all);
    }
    pub fn pocketRead(_: @This(), _: usize, _: []f32, _: []f32) usize {
        return 0;
    }
    pub fn pocket(_: @This(), _: usize, _: []const f32, _: []const f32, _: usize) !void {}
    // This CPU model has exact FP32 storage; real shard quantization has its separate unchanged fixture suite.
    pub fn pocketRound(_: @This(), _: []f32, _: []f32) !void {}
};
fn init() !*P {
    const p = try t.allocator.create(P);
    errdefer t.allocator.destroy(p);
    p.* = .{ .gpa = t.allocator, .mode = .subject, .locate = .{ .enabled = true }, .answered = &.{train}, .against = &.{old}, .held = &.{held}, .split = .{ 0, 10, 0 } };
    errdefer p.deinit();
    var m: Toy = .{ .p = p };
    try p.load(&m, Host{});
    try p.mark();
    return p;
}
fn deinit(p: *P) void {
    p.deinit();
    t.allocator.destroy(p);
}
fn place(m: *Toy) !void {
    const p = m.p;
    m.bias[1][1] = 0.6;
    m.bias[5][1] = 0.4;
    while (!try locate.once(p, m)) {}
    for (p.moe.items, 0..) |l, slot| {
        try t.expectEqual(l == 1 or l == 5, p.locate.chosen[slot]);
        if (!p.locate.chosen[slot]) continue;
        p.used.items[slot] = 1;
        p.neuron(p.up.items, slot, 0, d)[0] = 1;
        try p.taken.append(t.allocator, .{ .slot = slot, .neuron = 0, .key = .subject });
        try p.learned.append(t.allocator, .{ .slot = slot, .neuron = 0, .key = .subject, .start = 0, .count = 0, .subject = 9, .lesson = 1 });
    }
    try p.v.begin(t.allocator, p.taken.items.len, d, 1);
    p.next(.values);
}
fn fit(m: *Toy) !void {
    var units: usize = 0;
    while (!try m.p.step(m, Host{})) : (units += 1) if (units > 1000) return error.CorrectionDidNotFinish;
}
test "correction: actual output fitter overturns a planted belief, preserves its neighbour and undoes exactly" {
    const p = try init();
    defer deinit(p);
    var m: Toy = .{ .p = p };
    try place(&m);
    const before_bias = m.bias;
    const untouched = m.logits(&neighbour);
    try t.expectEqual(@as(u32, 1), Toy.best(m.logits(&new_ids)));
    try fit(&m);
    try t.expect(p.recalled());
    try t.expectEqual(@as(u32, 0), Toy.best(m.logits(&new_ids)));
    try t.expectEqual(@as(u32, 0), Toy.best(m.logits(&held_ids)));
    try t.expectEqual(untouched, m.logits(&neighbour));
    try t.expectEqual(before_bias, m.bias);
    try p.undo(&m, Host{});
    try t.expectEqual(@as(u32, 1), Toy.best(m.logits(&new_ids)));
    try t.expectEqual(untouched, m.logits(&neighbour));
}
test "correction: deletion gives back the original logits and put-back is exact" {
    const p = try init();
    defer deinit(p);
    var m: Toy = .{ .p = p };
    try place(&m);
    const original = m.logits(&new_ids);
    try fit(&m);
    const accepted = @import("pocket_mark.zig").hash(p);
    const corrected = m.logits(&new_ids);
    _ = try forgetting.forget(p, &m, Host{}, &.{ .{ 1, 0 }, .{ 5, 0 } });
    try t.expectEqual(original, m.logits(&new_ids));
    _ = try forgetting.unforget(p, &m, Host{});
    try t.expectEqual(accepted, @import("pocket_mark.zig").hash(p));
    try t.expectEqual(corrected, m.logits(&new_ids));
}
test "correction: opposing gradient matches a central FP64 difference of new CE minus half old CE" {
    const p = try init();
    defer deinit(p);
    var m: Toy = .{ .p = p };
    try place(&m);
    var result: [1]modes.Result = undefined;
    try m.pass(&.{.{ .ids = &new_ids, .start = 3, .decide = 2 }}, .target, @import("pocket_pack.zig").turn, &result);
    var grad: [2 * d]f32 = @splat(0);
    for (0..2) |j| @memcpy(grad[j * d ..][0..d], &m.gradient);
    try correction.add(p, &m, &grad, 1);
    for (p.taken.items, 0..) |x, j| for (0..3) |col| {
        const v = &p.neuron(p.down.items, x.slot, x.neuron, d)[col];
        v.* = 0.001;
        const plus = m.logits(&new_ids);
        const high = Toy.loss(plus, 0) - 0.5 * Toy.loss(plus, 1);
        const delta = @as(f64, v.*);
        v.* = -0.001;
        const minus = m.logits(&new_ids);
        const low = Toy.loss(minus, 0) - 0.5 * Toy.loss(minus, 1);
        v.* = 0;
        try t.expectApproxEqAbs((high - low) / (2 * delta), @as(f64, grad[j * d + col]), 0.00001);
    };
}

test "correction: a planted wrong layer cannot pass the actual output fitter's held-out checks" {
    const p = try init();
    defer deinit(p);
    var m: Toy = .{ .p = p };
    try place(&m);
    const original = m.logits(&new_ids);
    const adjacent = m.logits(&neighbour);
    const frozen = m.bias;
    // Move both exact subject detectors from the located belief layers to an irrelevant layer before fitting.
    for (p.taken.items) |x| {
        @memset(p.neuron(p.up.items, x.slot, x.neuron, d), 0);
        p.used.items[x.slot] = 0;
    }
    p.taken.clearRetainingCapacity();
    p.learned.clearRetainingCapacity();
    const wrong = std.mem.indexOfScalar(usize, p.moe.items, 3).?;
    p.used.items[wrong] = 1;
    p.neuron(p.up.items, wrong, 0, d)[0] = 1;
    try p.taken.append(t.allocator, .{ .slot = wrong, .neuron = 0, .key = .subject });
    try p.learned.append(t.allocator, .{ .slot = wrong, .neuron = 0, .key = .subject, .start = 0, .count = 0, .subject = 9, .lesson = 1 });
    try p.v.begin(t.allocator, 1, d, 1);
    try fit(&m);
    try t.expect(!p.recalled());
    try t.expectEqual(original, m.logits(&new_ids));
    try t.expectEqual(@as(u32, 1), Toy.best(m.logits(&held_ids)));
    try t.expectEqual(adjacent, m.logits(&neighbour));
    try t.expectEqual(frozen, m.bias);
}

test "capacity: a checked interior deletion can be reused only by opt-in and a failed new lesson restores it exactly" {
    const p = try init();
    defer deinit(p);
    var m: Toy = .{ .p = p };
    const slot = 0;
    p.used.items[slot] = P.neurons;
    for (0..P.neurons) |n| {
        p.neuron(p.up.items, slot, n, d)[0] = 1;
        p.neuron(p.down.items, slot, n, d)[0] = 0.25;
        try p.learned.append(t.allocator, .{ .slot = slot, .neuron = n, .key = .subject, .start = 0, .count = 0, .subject = 9, .lesson = @intCast(n + 1) });
    }
    _ = try forgetting.forget(p, &m, Host{}, &.{.{ @intCast(p.moe.items[slot]), 1 }});
    try t.expectEqual(P.neurons, p.used.items[slot]);
    const vacancy = @import("pocket_vacancy.zig");
    try t.expectEqual(@as(?usize, null), vacancy.next(p, slot));
    const deleted = @import("pocket_mark.zig").hash(p);
    const adjacent_up = p.neuron(p.up.items, slot, 2, d)[0];
    const adjacent_down = p.neuron(p.down.items, slot, 2, d)[0];
    const silent_down = p.neuron(p.down.items, slot, 1, d)[0];
    try p.mark();
    p.reuse_deleted = true;
    try t.expectEqual(@as(?usize, 1), vacancy.next(p, slot));
    try p.retired.append(t.allocator, .{ slot, 1 });
    try t.expectEqual(@as(?usize, null), vacancy.next(p, slot));
    p.retired.clearRetainingCapacity();
    try p.learned.append(t.allocator, .{ .slot = slot, .neuron = 1, .key = .subject, .start = 0, .count = 0, .subject = 9, .lesson = 2 });
    try t.expectEqual(@as(?usize, null), vacancy.next(p, slot));
    _ = p.learned.pop();
    var detector: [d]f32 = @splat(0);
    detector[0] = 2;
    try p.take(slot, .subject, &detector, .{}, &.{}, d);
    try t.expectEqual(@as(usize, 1), p.taken.items[0].neuron);
    try t.expectEqual(P.neurons, p.used.items[slot]);
    try t.expectEqual(adjacent_up, p.neuron(p.up.items, slot, 2, d)[0]);
    try t.expectEqual(adjacent_down, p.neuron(p.down.items, slot, 2, d)[0]);
    try p.undo(&m, Host{});
    try t.expectEqual(deleted, @import("pocket_mark.zig").hash(p));
    try t.expectEqual(silent_down, p.neuron(p.down.items, slot, 1, d)[0]);
    p.reuse_deleted = false;
    try t.expectEqual(@as(?usize, null), vacancy.next(p, slot));
}
