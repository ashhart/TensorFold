//! Frame-level silence accounting shared by native speech backends.
const std = @import("std");

pub const Config = struct {
    floor_dbfs: ?f64 = null,
    seconds: ?f64 = null,
    onset_seconds: ?f64 = null,
    onset_voiced_seconds: ?f64 = null,

    pub fn validate(c: Config) !void {
        if (c.seconds) |seconds| {
            if (!std.math.isFinite(seconds) or seconds < 0) return error.BadSilenceSeconds;
        }
        if (c.floor_dbfs) |floor| {
            if (!std.math.isFinite(floor) or floor > 0 or std.math.pow(f64, 10, floor / 10) == 0)
                return error.BadSilenceFloor;
        } else if ((c.seconds orelse 0) > 0) return error.MissingSilenceFloor;
        if (c.onset_seconds == null and c.onset_voiced_seconds == null) return;
        const onset = c.onset_seconds orelse return error.BadSilenceOnset;
        const voiced = c.onset_voiced_seconds orelse return error.BadSilenceOnsetVoiced;
        if (!(onset > 0 and onset < (c.seconds orelse 0))) return error.BadSilenceOnset;
        if (!(voiced > 0 and std.math.isFinite(voiced))) return error.BadSilenceOnsetVoiced;
    }

    pub fn enabled(c: Config) bool {
        return (c.seconds orelse 0) > 0 and c.floor_dbfs != null;
    }

    pub fn refusal(c: Config, model_type: []const u8) ?[]const u8 {
        if (!std.mem.eql(u8, model_type, "fish_omni") and
            (c.floor_dbfs != null or c.seconds != null or c.onset_seconds != null or c.onset_voiced_seconds != null))
            return "--silence-floor-dbfs, --silence-seconds, --silence-onset-s and --silence-onset-voiced-s are for Drama speech: serve without them";
        return null;
    }

    /// The startup sentence's rules for an enabled guard: the silent tail, then the onset rule when set.
    pub fn format(c: Config, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("a reply ends after {d} s below {d} dBFS", .{ c.seconds.?, c.floor_dbfs.? });
        if (c.onset_seconds) |onset| try w.print(", or after {d} s below it with under {d} s voiced", .{ onset, c.onset_voiced_seconds.? });
    }
};

pub const Guard = struct {
    floor_power: f64 = 0,
    limit: f64 = 0,
    /// Silent samples that end a synthesis voiced for fewer than `onset_voiced` samples.
    onset_limit: f64 = 0,
    onset_voiced: f64 = 0,
    silent_samples: usize = 0,
    voiced_samples: usize = 0,
    tripped: bool = false,

    pub fn init(c: Config, sample_rate: u32) Guard {
        if (!c.enabled()) return .{};
        const rate: f64 = @floatFromInt(sample_rate);
        return .{
            .floor_power = std.math.pow(f64, 10, c.floor_dbfs.? / 10),
            .limit = c.seconds.? * rate,
            .onset_limit = (c.onset_seconds orelse 0) * rate,
            .onset_voiced = (c.onset_voiced_seconds orelse 0) * rate,
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
            const silent: f64 = @floatFromInt(g.silent_samples);
            if (silent >= g.limit or (silent >= g.onset_limit and @as(f64, @floatFromInt(g.voiced_samples)) < g.onset_voiced)) {
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
    try (Config{ .floor_dbfs = -50, .seconds = 2, .onset_seconds = 1, .onset_voiced_seconds = 0.5 }).validate();
    for ([_]Config{
        .{ .floor_dbfs = -50, .seconds = 2, .onset_seconds = 2, .onset_voiced_seconds = 0.5 },
        .{ .onset_seconds = 1, .onset_voiced_seconds = 0.5 },
        .{ .floor_dbfs = -50, .seconds = 2, .onset_voiced_seconds = 0.5 },
        .{ .floor_dbfs = -50, .seconds = 2, .onset_seconds = 0, .onset_voiced_seconds = 0.5 },
        .{ .onset_seconds = 0, .onset_voiced_seconds = 0 },
    }) |c| try std.testing.expectError(error.BadSilenceOnset, c.validate());
    for ([_]Config{
        .{ .floor_dbfs = -50, .seconds = 2, .onset_seconds = 1 },
        .{ .floor_dbfs = -50, .seconds = 2, .onset_seconds = 1, .onset_voiced_seconds = 0 },
    }) |c| try std.testing.expectError(error.BadSilenceOnsetVoiced, c.validate());
}

test "silence flags are refused on text even when explicitly disabled" {
    for ([_]Config{ .{ .floor_dbfs = -80 }, .{ .seconds = 0 }, .{ .seconds = 1 }, .{ .onset_seconds = 0 }, .{ .onset_voiced_seconds = 0 } }) |config| {
        try std.testing.expect(config.refusal("nemotron_h") != null);
        try std.testing.expect(config.refusal("fish_omni") == null);
    }
    try std.testing.expect((Config{}).refusal("nemotron_h") == null);
}

test "the startup sentence adds the onset rule only when it is set" {
    var buf: [128]u8 = undefined;
    const tail: Config = .{ .floor_dbfs = -60, .seconds = 8 };
    const off = "drama3: silence guard on: a reply ends after 8 s below -60 dBFS";
    try std.testing.expectEqualStrings(off, try std.fmt.bufPrint(&buf, "drama3: silence guard on: {f}", .{tail}));
    var onset = tail;
    onset.onset_seconds = 2;
    onset.onset_voiced_seconds = 0.25;
    const on = off ++ ", or after 2 s below it with under 0.25 s voiced";
    try std.testing.expectEqualStrings(on, try std.fmt.bufPrint(&buf, "drama3: silence guard on: {f}", .{onset}));
}
