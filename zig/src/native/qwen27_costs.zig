//! Device-width window prices, with distinct width-tagged cache records and no uninitialized tail entries.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const lanes = @import("tensorfold").lanes;
const be = lanes.backend;
const Probe = struct { enabled: bool = true, shared_next: ?*bool = null, from: usize = 0 };

pub fn measure(b: anytype, io: std.Io) !void {
    const has_shared = b.tensor_units and b.runner.slots > 1 and
        ((b.window < 64 and b.runner.capacity > 64) or (b.window < 32 and b.runner.capacity > 32));
    var probe_shared = false;
    const probe: Probe = .{ .shared_next = if (has_shared) &probe_shared else null };
    switch (b.window) {
        16 => try sized(b, io, 16, "qwen27-lanes", probe),
        32 => try sized(b, io, 32, "qwen27-lanes", probe),
        64 => try sized(b, io, 64, "qwen27-lanes", probe),
        else => return error.WindowTooWide,
    }
    if (!b.tensor_units or b.runner.slots == 1) return;
    if (b.window < 64 and b.runner.capacity > 64) return shared(b, io, 64, probe_shared);
    if (b.window < 32 and b.runner.capacity > 32) return shared(b, io, 32, probe_shared);
}

/// Price wider unfused forwards without taking another stream's caches or changing the lone window's prices.
fn shared(b: anytype, io: std.Io, comptime width: usize, probe: bool) !void {
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
    try sized(b, io, width, "qwen27-shared-rows", .{ .enabled = probe, .from = window });
}

fn sized(b: anytype, io: std.Io, comptime width: usize, kind: []const u8, probe: Probe) !void {
    const r = b.runner;
    if (b.owners[0] != null or r.active != null) return error.StreamBusy;
    var shape: [128]u8 = undefined;
    const parts = [_][]const u8{ kind, std.mem.span(r.model.device.name()), try q.widths.costShape(r.model.config, r.model.widths_tag, &shape, width, r.capture_taps) };
    const k = lanes.cost_cache.key(b.gpa, io, &parts) catch null;
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
        lanes.cost_cache.save(u64, b.gpa, io, phase_key, phase +% 1);
    };
    if (kept) |table| if (!probe_now or r.capacity <= width) {
        @memcpy(b.costs[0..width], &table);
        b.timed = width;
        return;
    };
    if (r.capacity <= width) return error.ContextFull;
    defer {
        if (b.draft) |d| d.reset() catch {} else r.reset(0) catch {};
    }
    var ids: [192]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % r.model.config.vocab);
    // Reserve probe rows where they fit; a tighter context refills its smaller scratch prompt as needed.
    const spare = if (kept != null) @min(@as(usize, 7), r.capacity - width - 1) else 0;
    const prompt = ids[0..@min(ids.len, r.capacity - width - spare)];
    try b.refill(prompt);
    var s = try lanes.Stream.init(b.gpa, .{ .id = "timing", .prompt = prompt, .max_new = 1 });
    defer s.deinit(b.gpa);
    const timer: Timer(@TypeOf(b)) = .{ .b = b, .s = &s, .prompt = prompt };
    if (kept) |table| {
        var ms: [width]f64 = undefined;
        var widths: [width]u32 = undefined;
        for (table, &ms, &widths) |c, *m, *w| {
            m.* = c.ms;
            w.* = c.width;
        }
        const current = k orelse unreachable;
        const cursor_key = lanes.cost_cache.buildKey(&current, &.{"metal-heal-v1"});
        var cursor = lanes.cost_cache.load(u64, b.gpa, io, cursor_key) orelse 0;
        var selected_timer = timer;
        selected_timer.from = probe.from;
        _ = try lanes.cost_rule.healWindows(selected_timer, ms[probe.from..], widths[probe.from..], &cursor);
        for (b.costs[0..width], ms, 0..) |*c, m, i| c.* = .{ .width = @intCast(i + 1), .ms = m };
        b.timed = width;
        lanes.cost_cache.save([width]lanes.config.Cost, b.gpa, io, current, b.costs[0..width].*);
        lanes.cost_cache.save(u64, b.gpa, io, cursor_key, cursor);
        return;
    }
    for (0..5) |_| _ = try timer.time(width - 1); // the widest window first, while the GPU's clocks ramp up
    var ms: [width]f64 = undefined;
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
    return (mtl.clock.seconds() - t0) * 1e3;
}
