//! Device-width window prices, with distinct width-tagged cache records and no uninitialized tail entries.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const lanes = @import("tensorfold").lanes;
const be = lanes.backend;
pub fn measure(b: anytype, io: std.Io) !void {
    switch (b.window) {
        16 => try sized(b, io, 16, "qwen27-lanes"),
        32 => try sized(b, io, 32, "qwen27-lanes"),
        64 => try sized(b, io, 64, "qwen27-lanes"),
        else => return error.WindowTooWide,
    }
    if (!b.tensor_units or b.runner.slots == 1) return;
    if (b.window < 64 and b.runner.capacity > 64) return shared(b, io, 64);
    if (b.window < 32 and b.runner.capacity > 32) return shared(b, io, 32);
}

/// Price wider unfused forwards without taking another stream's caches or changing the lone window's prices.
fn shared(b: anytype, io: std.Io, comptime width: usize) !void {
    const window = b.window;
    const draft = b.draft;
    const profile = b.pass_profile;
    const costs = b.costs;
    const timed = b.timed;
    if (draft) |d| try d.generation.settle();
    b.window = width;
    b.draft = null;
    b.pass_profile = false;
    defer {
        b.window = window;
        b.draft = draft;
        b.pass_profile = profile;
        @memcpy(b.costs[0..window], costs[0..window]);
        if (draft) |d| d.reset() catch {};
    }
    errdefer {
        b.costs = costs;
        b.timed = timed;
    }
    try sized(b, io, width, "qwen27-shared-rows");
}

fn sized(b: anytype, io: std.Io, comptime width: usize, kind: []const u8) !void {
    const r = b.runner;
    if (b.owners[0] != null or r.active != null) return error.StreamBusy;
    var shape: [128]u8 = undefined;
    const parts = [_][]const u8{ kind, std.mem.span(r.model.device.name()), try q.widths.costShape(r.model.config, r.model.widths_tag, &shape, width, r.capture_taps) };
    const k = lanes.cost_cache.key(b.gpa, io, &parts) catch null;
    if (k) |key| if (lanes.cost_cache.load([width]lanes.config.Cost, b.gpa, io, key)) |kept| {
        @memcpy(b.costs[0..width], &kept);
        b.timed = width;
        return;
    };
    if (r.capacity <= width) return error.ContextFull;
    defer {
        if (b.draft) |d| d.reset() catch {} else r.reset(0) catch {};
    }
    var ids: [192]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % r.model.config.vocab);
    const prompt = ids[0..@min(ids.len, r.capacity - width)]; // a small --context still holds a window past it
    try b.refill(prompt);
    var s = try lanes.Stream.init(b.gpa, .{ .id = "timing", .prompt = prompt, .max_new = 1 });
    defer s.deinit(b.gpa);
    const timer: Timer(@TypeOf(b)) = .{ .b = b, .s = &s, .prompt = prompt };
    for (0..5) |_| _ = try timer.time(width - 1); // the widest window first, while the GPU's clocks ramp up
    var ms: [width]f64 = undefined;
    for (&ms, 0..) |*m, i| m.* = try lanes.cost_rule.fastest(timer, i);
    try lanes.cost_rule.smooth(timer, &ms);
    const ref = lanes.cost_cache.referenceKey(&parts);
    if (lanes.cost_cache.load([width]f64, b.gpa, io, ref)) |reference| if (lanes.cost_rule.drifted(&ms, &reference)) try lanes.cost_rule.again(timer, &ms);
    for (b.costs[0..width], ms, 0..) |*cost, m, i| cost.* = .{ .width = @intCast(i + 1), .ms = m };
    b.timed = width;
    if (k) |key| lanes.cost_cache.save([width]lanes.config.Cost, b.gpa, io, key, b.costs[0..width].*);
    lanes.cost_cache.save([width]f64, b.gpa, io, ref, ms);
}

/// Window entry `i` is `i + 1` rows, timed once; each keeps a row, so a full slot is refilled first.
fn Timer(comptime B: type) type {
    return struct {
        b: B,
        s: *lanes.Stream,
        prompt: []const u32,
        pub fn time(t: @This(), i: usize) !f64 {
            if (t.b.runner.offsets[0] + i + 1 > t.b.runner.capacity) try t.b.refill(t.prompt);
            return timeWindow(t.b, t.s, i + 1);
        }
    };
}

fn timeWindow(b: anytype, s: *lanes.Stream, rows: usize) !f64 {
    const t0 = mtl.clock.seconds();
    const positions: [64]u64 = @splat(0);
    const drafts: [64]u32 = @splat(1);
    var sampled: [64]u32 = undefined;
    var echoed: [64]u32 = undefined;
    b.owners[0] = s;
    defer b.owners[0] = null;
    const w = be.Window{ .stream = s, .pending = 0, .held = 0, .tokens = drafts[0 .. rows - 1], .parents = null, .positions = positions[0..rows] };
    var out = [_]be.Verified{.{ .sampled = sampled[0..rows], .drafts = echoed[0 .. rows - 1] }};
    try b.backend().verify(&.{w}, &out);
    var path = [_]u32{0};
    try b.backend().keep(&.{w}, &.{&path});
    return (mtl.clock.seconds() - t0) * 1e3;
}
