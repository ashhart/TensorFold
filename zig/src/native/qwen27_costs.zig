//! Device-width window prices, with distinct width-tagged cache records and no uninitialized tail entries.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const lanes = @import("tensorfold").lanes;
const be = lanes.backend;
const warm = lanes.cost_rule.warm_prices;
const Phase = struct {
    key: ?[32]u8 = null,
    turn: warm.Turn = .{},

    fn retryNarrow(phase: Phase) bool {
        return phase.turn.preferredShared() and !phase.turn.attempted;
    }
};
const Probe = struct {
    start_turn: bool = false,
    from: usize = 0,
    floor: f64 = 0,
    seed: ?lanes.cost_cache.KeySeed = null,
    phase: ?*Phase = null,
};

pub fn measure(b: anytype, io: std.Io) !void {
    const has_shared = b.tensor_units and b.runner.slots > 1 and
        ((b.window < 64 and b.runner.capacity > 64) or (b.window < 32 and b.runner.capacity > 32));
    var phase: Phase = .{};
    const seed = lanes.cost_cache.keySeed(b.gpa, io, false) catch null;
    const probe: Probe = .{ .start_turn = has_shared, .seed = seed, .phase = &phase };
    defer if (phase.key) |key| lanes.cost_cache.save(u64, b.gpa, io, key, phase.turn.next());
    try narrow(b, io, probe);
    if (!has_shared) return;
    if (b.window < 64 and b.runner.capacity > 64) try shared(b, io, 64, seed, &phase) else try shared(b, io, 32, seed, &phase);
    if (phase.retryNarrow()) {
        const timed = b.timed;
        defer b.timed = timed;
        try narrow(b, io, .{ .seed = seed, .phase = &phase });
    }
}

fn narrow(b: anytype, io: std.Io, probe: Probe) !void {
    switch (b.window) {
        16 => try sized(b, io, 16, "qwen27-lanes", probe),
        32 => try sized(b, io, 32, "qwen27-lanes", probe),
        64 => try sized(b, io, 64, "qwen27-lanes", probe),
        else => return error.WindowTooWide,
    }
}

/// Price wider unfused forwards without taking another stream's caches or changing the lone window's prices.
fn shared(b: anytype, io: std.Io, comptime width: usize, seed: ?lanes.cost_cache.KeySeed, phase: *Phase) !void {
    const window = b.window;
    const draft = b.draft;
    const profile = b.pass_profile;
    const costs = b.costs;
    const timed = b.timed;
    b.window = width;
    b.draft = null;
    b.pass_profile = false;
    defer {
        b.window = window;
        b.draft = draft;
        b.pass_profile = profile;
        @memcpy(b.costs[0..window], costs[0..window]);
    }
    errdefer {
        b.costs = costs;
        b.timed = timed;
    }
    try sized(b, io, width, "qwen27-shared-rows", .{ .from = window, .floor = sharedFloor(costs[0..window]), .seed = seed, .phase = phase });
}

fn sharedFloor(costs: []const lanes.config.Cost) f64 {
    var floor: f64 = 0;
    for (costs) |cost| floor = @max(floor, cost.ms);
    return floor;
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
    var probe_now = if (probe.phase) |phase| !phase.turn.attempted else true;
    if (kept != null and probe.start_turn) {
        const current = k orelse unreachable;
        const phase_key = lanes.cost_cache.buildKey(&current, &.{"metal-family-phase-v1"});
        const turn: warm.Turn = .{ .value = lanes.cost_cache.load(u64, b.gpa, io, phase_key) orelse 0 };
        probe_now = !turn.preferredShared();
        if (probe.phase) |p| {
            p.key = phase_key;
            p.turn = turn;
        }
    }
    if (kept) |table| {
        @memcpy(b.costs[0..width], &table);
        b.timed = width;
        if (!probe_now or r.capacity <= width) return;
    }
    if (r.capacity <= width) return error.ContextFull;
    var ms: [width]f64 = undefined;
    var widths: [width]u32 = undefined;
    if (kept) |table| for (table, &ms, &widths) |c, *m, *w| {
        m.* = c.ms;
        w.* = c.width;
    };
    const cursor_key = if (k) |key| lanes.cost_cache.buildKey(&key, &.{"metal-heal-v2"}) else null;
    var cursor = if (cursor_key) |key| lanes.cost_cache.load(u64, b.gpa, io, key) orelse 0 else 0;
    const before = cursor;
    defer if (cursor != before) if (cursor_key) |key| lanes.cost_cache.save(u64, b.gpa, io, key, cursor);
    if (kept != null and !warm.admit(ms[probe.from..], widths[probe.from..], &cursor)) return;
    if (probe.phase) |phase| phase.turn.attempted = true;
    var ids: [192]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % r.model.config.vocab);
    const prompt = ids[0..lanes.cost_rule.context.promptRows(r.capacity, width, width + 5)];
    var s = lanes.Stream.init(b.gpa, .{ .id = "timing", .prompt = prompt, .max_new = 1 }) catch |err| {
        if (kept != null) return;
        return err;
    };
    defer s.deinit(b.gpa);
    const timer: Timer(@TypeOf(b)) = .{ .b = b, .s = &s, .prompt = prompt };
    if (kept != null) {
        var warm_timer: WarmTimer(@TypeOf(b)) = .{ .timer = timer };
        warm_timer.timer.from = probe.from;
        const stats = (try warmProbe(b, &warm_timer, ms[probe.from..], widths[probe.from..], &cursor, probe.floor)) orelse return;
        for (b.costs[0..width], ms, 0..) |*c, m, i| c.* = .{ .width = @intCast(i + 1), .ms = m };
        b.timed = width;
        if (stats.lowered > 0) if (k) |key| lanes.cost_cache.save([width]lanes.config.Cost, b.gpa, io, key, b.costs[0..width].*);
        return;
    }
    try b.refill(prompt);
    for (0..5) |_| _ = try timer.time(width - 1); // the widest window first, while the GPU's clocks ramp up
    try lanes.cost_rule.serialWindows(timer, &ms);
    const ref = lanes.cost_cache.referenceKey(&parts);
    if (lanes.cost_cache.load([width]f64, b.gpa, io, ref)) |reference| if (lanes.cost_rule.drifted(&ms, &reference)) try lanes.cost_rule.again(timer, &ms);
    try resetProbe(b);
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
        pub fn start(t: *@This()) void {
            t.began = mtl.clock.seconds();
        }
        pub fn elapsedMs(t: *@This()) f64 {
            return (mtl.clock.seconds() - t.began) * 1e3;
        }
        pub fn prepare(t: *@This(), _: anytype, _: []const u32) !void {
            try t.timer.b.refill(t.timer.prompt);
        }
        pub fn time(t: *@This(), i: usize) !f64 {
            return t.timer.time(i);
        }
    };
}

fn resetProbe(b: anytype) !void {
    if (b.draft) |d| try d.reset() else try b.runner.reset(0);
}

fn warmProbe(b: anytype, timer: anytype, ms: []f64, widths: []const u32, cursor: *u64, floor: f64) !?warm.Stats {
    const stats: ?warm.Stats = warm.prepared(timer, ms, widths, cursor, floor, warm.Local{}, true) catch blk: {
        b.dropRound();
        b.runner.recover();
        if (b.draft) |d| try d.model.recover();
        break :blk null;
    };
    try resetProbe(b);
    return stats;
}

const ProbeRunner = struct {
    active: ?usize = null,
    failed: bool = false,
    resets: usize = 0,

    pub fn recover(r: *@This()) void {
        r.active = null;
        r.failed = false;
    }

    pub fn reset(r: *@This(), slot: usize) !void {
        try std.testing.expectEqual(@as(usize, 0), slot);
        if (r.active != null or r.failed) return error.RoundNotReady;
        r.resets += 1;
    }
};

const ProbeBackend = struct {
    const Model = struct {
        failed: bool = false,
        recover_error: bool = false,

        pub fn recover(m: *@This()) !void {
            if (m.recover_error) return error.DraftRecoveryFailed;
            m.failed = false;
        }
    };
    const Draft = struct {
        runner: *ProbeRunner,
        model: Model = .{},
        reset_error: bool = false,

        pub fn reset(d: *@This()) !void {
            if (d.reset_error or d.model.failed) return error.DraftResetFailed;
            try d.runner.reset(0);
        }
    };
    runner: *ProbeRunner,
    draft: ?*Draft = null,
    round: bool = false,

    pub fn dropRound(b: *@This()) void {
        b.round = false;
    }
};

test "qwen27 startup: shared probe uses the maximum live narrow cost" {
    const Fake = struct {
        pub fn time(_: @This(), _: usize) !f64 {
            return 19.9;
        }
    };
    const costs = [_]lanes.config.Cost{ .{ .width = 1, .ms = 20 }, .{ .width = 2, .ms = 19.8 } };
    var ms = [_]f64{21};
    var cursor: u64 = 0;
    var runner: ProbeRunner = .{};
    var b: ProbeBackend = .{ .runner = &runner };
    _ = (try warmProbe(&b, Fake{}, &ms, &.{3}, &cursor, sharedFloor(&costs))).?;
    try std.testing.expectEqual(@as(f64, 21), ms[0]);
}

test "qwen27 startup: recover a failed verify or keep before retaining cached prices" {
    const Fault = struct {
        b: *ProbeBackend,
        keep: bool,

        pub fn time(f: @This(), _: usize) !f64 {
            f.b.runner.active = 0;
            f.b.runner.failed = f.keep;
            if (f.keep) {
                f.b.round = true;
                f.b.draft.?.model.failed = true;
                return error.KeepGpuFailure;
            }
            return error.DrawGpuFailure;
        }
    };
    var runner: ProbeRunner = .{};
    var draft: ProbeBackend.Draft = .{ .runner = &runner };
    var b: ProbeBackend = .{ .runner = &runner };
    var ms = [_]f64{21};
    var cursor: u64 = 0;
    try std.testing.expect((try warmProbe(&b, Fault{ .b = &b, .keep = false }, &ms, &.{1}, &cursor, 0)) == null);
    b.draft = &draft;
    const keep: Fault = .{ .b = &b, .keep = true };
    try std.testing.expect((try warmProbe(&b, keep, &ms, &.{1}, &cursor, 0)) == null);
    try std.testing.expect(runner.active == null and !runner.failed and !b.round and !draft.model.failed);
    try std.testing.expectEqual(@as(usize, 2), runner.resets);
    try std.testing.expectEqual(@as(f64, 21), ms[0]);
    draft.model.recover_error = true;
    try std.testing.expectError(error.DraftRecoveryFailed, warmProbe(&b, keep, &ms, &.{1}, &cursor, 0));
    draft.model.recover_error = false;
    draft.reset_error = true;
    try std.testing.expectError(error.DraftResetFailed, warmProbe(&b, keep, &ms, &.{1}, &cursor, 0));
    try std.testing.expectEqual(@as(usize, 2), runner.resets);
}

test "qwen27 startup: no-fit shared turns still probe narrow on the same load" {
    const Fake = struct {
        pub fn time(_: @This(), _: usize) !f64 {
            return 80;
        }
    };
    var narrow_ms = [_]f64{80};
    var shared_cursor: u64 = 0;
    var narrow_cursor: u64 = 0;
    var saved: u64 = 0;
    var runner: ProbeRunner = .{};
    var b: ProbeBackend = .{ .runner = &runner };
    for (1..7) |load| {
        var phase: Phase = .{ .turn = .{ .value = saved } };
        if (phase.turn.preferredShared()) phase.turn.attempted = warm.admit(&.{200}, &.{64}, &shared_cursor);
        if (!phase.turn.preferredShared() or phase.retryNarrow()) {
            const stats = (try warmProbe(&b, Fake{}, &narrow_ms, &.{16}, &narrow_cursor, 0)).?;
            try std.testing.expectEqual(@as(u32, 1), stats.reads);
        }
        saved = phase.turn.next();
        try std.testing.expectEqual(@as(u64, load), narrow_cursor);
    }
    try std.testing.expectEqual(@as(u64, 0), shared_cursor);
}
