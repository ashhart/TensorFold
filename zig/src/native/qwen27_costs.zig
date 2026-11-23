//! Device-width window prices, with distinct width-tagged cache records and no uninitialized tail entries.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const lanes = @import("tensorfold").lanes;
const be = lanes.backend;
const Phase = struct { key: ?[32]u8 = null, value: u64 = 0, reads: u32 = 0 };
const Probe = struct {
    enabled: bool = true, shared_next: ?*bool = null, from: usize = 0, floor: f64 = 0,
    seed: ?lanes.cost_cache.KeySeed = null, phase: ?*Phase = null,
};

pub fn measure(b: anytype, io: std.Io) !void {
    const has_shared = b.tensor_units and b.runner.slots > 1 and
        ((b.window < 64 and b.runner.capacity > 64) or (b.window < 32 and b.runner.capacity > 32));
    var probe_shared = false;
    var phase: Phase = .{};
    const seed = lanes.cost_cache.keySeed(b.gpa, io, false) catch null;
    const probe: Probe = .{ .shared_next = if (has_shared) &probe_shared else null, .seed = seed, .phase = &phase };
    defer if (phase.reads > 0) if (phase.key) |key| lanes.cost_cache.save(u64, b.gpa, io, key, phase.value +% 1);
    switch (b.window) {
        16 => try sized(b, io, 16, "qwen27-lanes", probe),
        32 => try sized(b, io, 32, "qwen27-lanes", probe),
        64 => try sized(b, io, 64, "qwen27-lanes", probe),
        else => return error.WindowTooWide,
    }
    if (!b.tensor_units or b.runner.slots == 1) return;
    if (b.window < 64 and b.runner.capacity > 64) try shared(b, io, 64, probe_shared, seed, &phase)
    else if (b.window < 32 and b.runner.capacity > 32) try shared(b, io, 32, probe_shared, seed, &phase);
}

/// Price wider unfused forwards without taking another stream's caches or changing the lone window's prices.
fn shared(b: anytype, io: std.Io, comptime width: usize, probe: bool, seed: ?lanes.cost_cache.KeySeed, phase: *Phase) !void {
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
    try sized(b, io, width, "qwen27-shared-rows", .{ .enabled = probe, .from = window, .floor = costs[window - 1].ms, .seed = seed, .phase = phase });
}

fn sized(b: anytype, io: std.Io, comptime width: usize, kind: []const u8, probe: Probe) !void {
    const r = b.runner;
    if (b.owners[0] != null or r.active != null) return error.StreamBusy;
    var shape: [128]u8 = undefined;
    const parts = [_][]const u8{ kind, std.mem.span(r.model.device.name()), try q.widths.costShape(r.model.config, r.model.widths_tag, &shape, width, r.capture_taps) };
    const k = if (probe.seed) |seed| seed.key(&parts) else null;
    var kept = if (k) |key| lanes.cost_cache.load([width]lanes.config.Cost, b.gpa, io, key) else null;
    if (kept) |table| for (table, 0..) |c, i| {
        if (c.width != i + 1 or !(c.ms > 0) or !std.math.isFinite(c.ms)) {
            kept = null;
            break;
        }
    };
    var probe_now = probe.enabled;
    if (kept != null) if (probe.shared_next) |next| {
        const current = k orelse unreachable;
        const phase_key = lanes.cost_cache.buildKey(&current, &.{"metal-family-phase-v1"});
        const phase = lanes.cost_cache.load(u64, b.gpa, io, phase_key) orelse 0;
        probe_now = phase % 2 == 0;
        next.* = !probe_now;
        if (probe.phase) |p| { p.key = phase_key; p.value = phase; }
    };
    if (kept) |table| if (!probe_now or r.capacity <= width) {
        @memcpy(b.costs[0..width], &table);
        b.timed = width;
        return;
    };
    if (r.capacity <= width) return error.ContextFull;
    var ms: [width]f64 = undefined;
    var widths: [width]u32 = undefined;
    if (kept) |table| for (table, &ms, &widths) |c, *m, *w| { m.* = c.ms; w.* = c.width; };
    const cursor_key = if (k) |key| lanes.cost_cache.buildKey(&key, &.{"metal-heal-v2"}) else null;
    var cursor = if (cursor_key) |key| lanes.cost_cache.load(u64, b.gpa, io, key) orelse 0 else 0;
    if (kept != null and lanes.cost_rule.warm_prices.select(ms[probe.from..], widths[probe.from..], cursor).n == 0) {
        @memcpy(b.costs[0..width], &kept.?);
        b.timed = width;
        return;
    }
    defer { if (b.draft) |d| d.reset() catch {} else r.reset(0) catch {}; }
    var ids: [192]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % r.model.config.vocab);
    const prompt = ids[0..lanes.cost_rule.context.promptRows(r.capacity, width, width + 5)];
    var s = try lanes.Stream.init(b.gpa, .{ .id = "timing", .prompt = prompt, .max_new = 1 });
    defer s.deinit(b.gpa);
    const timer: Timer(@TypeOf(b)) = .{ .b = b, .s = &s, .prompt = prompt };
    if (kept != null) {
        var warm_timer: WarmTimer(@TypeOf(b)) = .{ .timer = timer };
        warm_timer.timer.from = probe.from;
        const stats = try lanes.cost_rule.warm_prices.prepared(&warm_timer, ms[probe.from..], widths[probe.from..], &cursor, probe.floor, lanes.cost_rule.warm_prices.Local{}, true);
        if (probe.phase) |phase| phase.reads += stats.reads;
        for (b.costs[0..width], ms, 0..) |*c, m, i| c.* = .{ .width = @intCast(i + 1), .ms = m };
        b.timed = width;
        if (stats.lowered > 0) if (k) |key| lanes.cost_cache.save([width]lanes.config.Cost, b.gpa, io, key, b.costs[0..width].*);
        if (stats.reads > 0) if (cursor_key) |key| lanes.cost_cache.save(u64, b.gpa, io, key, cursor);
        return;
    }
    try b.refill(prompt);
    for (0..5) |_| _ = try timer.time(width - 1); // the widest window first, while the GPU's clocks ramp up
    try lanes.cost_rule.measureWindows(timer, &ms, false);
    try lanes.cost_rule.smooth(timer, &ms);
    const ref = lanes.cost_cache.referenceKey(&parts);
    if (lanes.cost_cache.load([width]f64, b.gpa, io, ref)) |reference| if (lanes.cost_rule.drifted(&ms, &reference)) try lanes.cost_rule.measureWindows(timer, &ms, true);
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
        from: usize = 0,
        pub fn time(t: @This(), i: usize) !f64 {
            const rows = t.from + i + 1;
            if (t.b.runner.offsets[0] + rows > t.b.runner.capacity) try t.b.refill(t.prompt);
            return timeWindow(t.b, t.s, rows);
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
    try b.settle();
    return (mtl.clock.seconds() - t0) * 1e3;
}

fn WarmTimer(comptime B: type) type {
    return struct {
        timer: Timer(B),
        began: f64 = 0,
        pub fn start(t: *@This()) void { t.began = mtl.clock.seconds(); }
        pub fn elapsedMs(t: *@This()) f64 { return (mtl.clock.seconds() - t.began) * 1e3; }
        pub fn prepare(t: *@This(), _: anytype, _: []const u32) !void { try t.timer.b.refill(t.timer.prompt); }
        pub fn time(t: *@This(), i: usize) !f64 { return t.timer.time(i); }
    };
}
