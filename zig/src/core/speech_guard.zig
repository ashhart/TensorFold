//! Frame-level silence accounting shared by native speech backends.
const std = @import("std");

pub const Config = struct {
    floor_dbfs: ?f64 = null,
    seconds: ?f64 = null,

    pub fn validate(c: Config) !void {
        if (c.seconds) |seconds| {
            if (!std.math.isFinite(seconds) or seconds < 0) return error.BadSilenceSeconds;
        }
        if (c.floor_dbfs) |floor| {
            if (!std.math.isFinite(floor) or floor > 0 or std.math.pow(f64, 10, floor / 10) == 0)
                return error.BadSilenceFloor;
        } else if ((c.seconds orelse 0) > 0) return error.MissingSilenceFloor;
    }

    pub fn enabled(c: Config) bool {
        return (c.seconds orelse 0) > 0 and c.floor_dbfs != null;
    }

    pub fn refusal(c: Config, model_type: []const u8) ?[]const u8 {
        if (!std.mem.eql(u8, model_type, "fish_omni") and
            (c.floor_dbfs != null or c.seconds != null))
            return "--silence-floor-dbfs and --silence-seconds are for Drama speech: serve without them";
        return null;
    }
};

pub const Guard = struct {
    floor_power: f64 = 0,
    limit: f64 = 0,
    silent_samples: usize = 0,
    voiced_samples: usize = 0,
    tripped: bool = false,

    pub fn init(c: Config, sample_rate: u32) Guard {
        if (!c.enabled()) return .{};
        return .{
            .floor_power = std.math.pow(f64, 10, c.floor_dbfs.? / 10),
            .limit = c.seconds.? * @as(f64, @floatFromInt(sample_rate)),
        };
    }

    // Account before publication: the HTTP owner may quantize published PCM in place.
    pub fn hear(g: *Guard, pcm: []const f32, frame_length: usize) usize {
        if (g.limit == 0) return pcm.len;
        if (g.tripped) return 0;
        std.debug.assert(frame_length > 0 and pcm.len % frame_length == 0);
        var end: usize = 0;
        while (end < pcm.len) {
            var power: f64 = 0;
            for (pcm[end..][0..frame_length]) |sample| {
                const x: f64 = sample;
                power += x * x;
            }
            end += frame_length;
            if (power < g.floor_power * @as(f64, @floatFromInt(frame_length))) {
                g.silent_samples += frame_length;
            } else {
                g.silent_samples = 0;
                g.voiced_samples += frame_length;
            }
            if (@as(f64, @floatFromInt(g.silent_samples)) >= g.limit) {
                g.tripped = true;
                return end;
            }
        }
        return end;
    }
};

test "silence tail cuts at a frame boundary and latches before later speech" {
    var guard = Guard.init(.{ .floor_dbfs = -20, .seconds = 0.5 }, 8);
    const pcm = [_]f32{ 0.25, -0.25, 0, 0, 0, 0, 0.25, -0.25 };
    try std.testing.expectEqual(@as(usize, 6), guard.hear(&pcm, 2));
    try std.testing.expect(guard.tripped);
    try std.testing.expectEqual(@as(usize, 2), guard.voiced_samples);
    try std.testing.expectEqual(@as(usize, 0), guard.hear(&pcm, 2));
}

test "quiet voiced frames reset silence and disabled guard preserves all samples" {
    var guard = Guard.init(.{ .floor_dbfs = -80, .seconds = 0.5 }, 8);
    const pcm = [_]f32{ 0, 0, 0.001, -0.001, 0, 0, 0.001, -0.001 };
    try std.testing.expectEqual(pcm.len, guard.hear(&pcm, 2));
    try std.testing.expect(!guard.tripped);
    var off = Guard.init(.{}, 8);
    const silent: [16]f32 = @splat(0);
    try std.testing.expectEqual(silent.len, off.hear(&silent, 2));
    try std.testing.expect(!off.tripped);
}

test "silence configuration requires explicit finite thresholds" {
    try (Config{}).validate();
    try std.testing.expectError(error.MissingSilenceFloor, (Config{ .seconds = 1 }).validate());
    try std.testing.expectError(error.BadSilenceSeconds, (Config{ .seconds = std.math.inf(f64) }).validate());
    try std.testing.expectError(error.BadSilenceFloor, (Config{ .floor_dbfs = 1 }).validate());
}

test "silence flags are refused on text even when explicitly disabled" {
    for ([_]Config{ .{ .floor_dbfs = -80 }, .{ .seconds = 0 }, .{ .seconds = 1 } }) |config| {
        try std.testing.expect(config.refusal("nemotron_h") != null);
        try std.testing.expect(config.refusal("fish_omni") == null);
    }
    try std.testing.expect((Config{}).refusal("nemotron_h") == null);
}
