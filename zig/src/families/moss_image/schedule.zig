//! The denoising schedule: MOSSImageScheduler's timesteps and noise levels, from scheduler/scheduler_config.json.
const std = @import("std");

/// The scheduler config's settings this family runs; any other option turned on is refused.
pub const Config = struct {
    train_steps: u32 = 1000,
    shift: f64 = 5.0,
    training_shift: f64 = 1.0,
};

/// Reads scheduler_config.json; an option that changes the schedule or the step and is not implemented is an error.
pub fn parse(gpa: std.mem.Allocator, text: []const u8) !Config {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const o = doc.value.object;
    const name = o.get("_class_name") orelse return error.BadConfig;
    if (name != .string or !std.mem.eql(u8, name.string, "MOSSImageScheduler")) return error.NotMossImageScheduler;
    inline for (.{ "use_dynamic_shifting", "use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas", "invert_sigmas", "stochastic_sampling" }) |key| {
        if (o.get(key)) |v| switch (v) {
            .bool => |on| if (on) return refuse(key),
            .null => {},
            else => return error.BadConfig,
        };
    }
    if (o.get("shift_terminal")) |v| switch (v) {
        .null => {},
        .float => |x| if (x != 0) return refuse("shift_terminal"),
        .integer => |x| if (x != 0) return refuse("shift_terminal"),
        else => return error.BadConfig,
    };
    var c: Config = .{};
    if (o.get("num_train_timesteps")) |v| {
        if (v != .integer or v.integer < 2 or v.integer > 1 << 20) return error.BadConfig;
        c.train_steps = @intCast(v.integer);
    }
    if (o.get("shift")) |v| c.shift = try number(v);
    if (o.get("training_shift")) |v| c.training_shift = try number(v);
    if (!(c.shift > 0) or !(c.training_shift > 0)) return error.BadConfig;
    return c;
}

fn refuse(comptime key: []const u8) error{UnsupportedScheduler} {
    std.log.warn("moss image: the scheduler config turns on {s}, which this family does not implement", .{key});
    return error.UnsupportedScheduler;
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .float => |x| x,
        .integer => |x| @floatFromInt(x),
        else => error.BadConfig,
    };
}

/// torch.linspace(start, end, count) in float32 on the CPU: from the start below halfway, from the end after, fused.
fn linspace(start: f32, end: f32, out: []f32) void {
    if (out.len == 1) {
        out[0] = start;
        return;
    }
    const step = (end - start) / @as(f32, @floatFromInt(out.len - 1));
    const halfway = out.len / 2;
    for (out, 0..) |*v, i| v.* = if (i < halfway)
        @mulAdd(f32, step, @floatFromInt(i), start)
    else
        @mulAdd(f32, -step, @floatFromInt(out.len - 1 - i), end);
}

/// The scheduler's grid: count values from 1 down, shifted by shift v / (1 + (shift - 1) v), in float32 as torch does.
fn grid(out: []f32, shift: f64, train_steps: u32) void {
    const total: f64 = @floatFromInt(train_steps);
    linspace(1, @floatCast(shift / (total + shift - 1)), out);
    const s: f32 = @floatCast(shift);
    const s1: f32 = @floatCast(shift - 1);
    for (out) |*v| v.* = s * v.* / (1 + s1 * v.*);
}

/// A schedule: `timesteps` (what the transformer reads) and `sigmas`, one longer, ending at 0.
pub const Schedule = struct {
    timesteps: []f32,
    sigmas: []f32,

    pub fn deinit(s: Schedule, gpa: std.mem.Allocator) void {
        gpa.free(s.timesteps);
        gpa.free(s.sigmas);
    }
};

/// set_timesteps(steps): the shifted grid snapped to the nearest training noise level, repeats dropped.
pub fn timesteps(gpa: std.mem.Allocator, c: Config, steps: u32) !Schedule {
    if (steps < 1) return error.BadStepCount;
    const total = c.train_steps;
    const want = try gpa.alloc(f32, steps);
    defer gpa.free(want);
    grid(want, c.shift, total);
    const training = try gpa.alloc(f32, total);
    defer gpa.free(training);
    grid(training, c.training_shift, total);
    std.mem.reverse(f32, training);
    var picked: std.ArrayList(f32) = .empty;
    defer picked.deinit(gpa);
    for (want) |sigma| {
        // torch.searchsorted (left): the first level at or above sigma, clamped like the scheduler
        var upper: usize = 0;
        while (upper < total and training[upper] < sigma) upper += 1;
        upper = @min(upper, total - 1);
        const lower = if (upper == 0) 0 else upper - 1;
        const at = if (@abs(sigma - training[lower]) <= @abs(training[upper] - sigma)) lower else upper;
        const level = training[at];
        if (picked.items.len == 0 or picked.items[picked.items.len - 1] != level) try picked.append(gpa, level);
    }
    const n = picked.items.len;
    const times = try gpa.alloc(f32, n);
    errdefer gpa.free(times);
    const out: Schedule = .{ .timesteps = times, .sigmas = try gpa.alloc(f32, n + 1) };
    const scale: f32 = @floatFromInt(total);
    for (picked.items, out.timesteps, out.sigmas[0..n]) |level, *t, *s| {
        s.* = level;
        t.* = level * scale;
    }
    out.sigmas[n] = 0;
    return out;
}

const released =
    \\{"_class_name": "MOSSImageScheduler", "_diffusers_version": "0.38.0", "base_image_seq_len": 256, "base_shift": 0.5,
    \\ "invert_sigmas": false, "max_image_seq_len": 4096, "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 5.0,
    \\ "shift_terminal": null, "stochastic_sampling": false, "time_shift_type": "exponential", "training_shift": 1.0,
    \\ "use_beta_sigmas": false, "use_dynamic_shifting": false, "use_exponential_sigmas": false, "use_karras_sigmas": false}
;

test "the released scheduler config parses; an option this family does not implement is refused" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(Config{}, try parse(a, released));
    const turned_on = .{ .{ "\"use_karras_sigmas\": false", "\"use_karras_sigmas\": true" }, .{ "\"use_dynamic_shifting\": false", "\"use_dynamic_shifting\": true" }, .{ "\"invert_sigmas\": false", "\"invert_sigmas\": true" }, .{ "\"shift_terminal\": null", "\"shift_terminal\": 0.1" }, .{ "\"stochastic_sampling\": false", "\"stochastic_sampling\": true" }, .{ "\"use_beta_sigmas\": false", "\"use_beta_sigmas\": true" }, .{ "\"use_exponential_sigmas\": false", "\"use_exponential_sigmas\": true" } };
    inline for (turned_on) |kv| {
        const text = try std.mem.replaceOwned(u8, a, released, kv[0], kv[1]);
        defer a.free(text);
        try std.testing.expectError(error.UnsupportedScheduler, parse(a, text));
    }
    const other = try std.mem.replaceOwned(u8, a, released, "MOSSImageScheduler", "FlowMatchEulerDiscreteScheduler");
    defer a.free(other);
    try std.testing.expectError(error.NotMossImageScheduler, parse(a, other));
    const negative = try std.mem.replaceOwned(u8, a, released, "\"shift\": 5.0", "\"shift\": -1.0");
    defer a.free(negative);
    try std.testing.expectError(error.BadConfig, parse(a, negative));
    try std.testing.expectError(error.BadStepCount, timesteps(a, .{}, 0));
}

test "timesteps and sigmas equal MOSSImageScheduler's for every recorded config and step count (moss_image_case.py schedule)" {
    const a = std.testing.allocator;
    const Change = struct { shift: ?f64 = null, training_shift: ?f64 = null, num_train_timesteps: ?u32 = null };
    const Case = struct { config: Change, steps: u32, count: usize, timesteps: []const u8, sigmas: []const u8 };
    const refs = try std.json.parseFromSlice(struct { cases: []const Case }, a, @embedFile("schedule_torch.json"), .{ .ignore_unknown_fields = true });
    defer refs.deinit();
    try std.testing.expect(refs.value.cases.len >= 100);
    for (refs.value.cases) |c| {
        const config: Config = .{ .shift = c.config.shift orelse 5.0, .training_shift = c.config.training_shift orelse 1.0, .train_steps = c.config.num_train_timesteps orelse 1000 };
        const s = try timesteps(a, config, c.steps);
        defer s.deinit(a);
        try std.testing.expectEqual(c.count, s.timesteps.len);
        for ([_][]const f32{ s.timesteps, s.sigmas }, [_][]const u8{ c.timesteps, c.sigmas }) |ours, want| {
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(ours), &digest, .{});
            if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), want)) {
                std.debug.print("{d} steps, shift {d}, training shift {d}, {d} levels: differs\n", .{ c.steps, config.shift, config.training_shift, config.train_steps });
                return error.TestUnexpectedResult;
            }
        }
    }
}
