//! Opt-in wait before a prompt chunk when the hottest thermal zone is above a band.
const std = @import("std");

pub const default_root = "/sys/class/thermal";

pub const Bands = struct { high_c: f64, low_c: f64 };

pub const Gate = struct {
    bands: ?Bands = null,
    waited_s: f64 = 0,
    root: []const u8 = default_root,

    /// Both env values absent: no wait and no thermal read. One absent, or low above high: error.
    pub fn fromEnv() !Gate {
        const root = envSpan("TF_GLM_HEAT_ROOT") orelse default_root;
        return .{ .bands = try bandsFrom(envSpan("TF_GLM_HEAT_HIGH"), envSpan("TF_GLM_HEAT_LOW")), .root = root };
    }

    /// `read` returns the gathered maximum in celsius. `sleep_fn` stands in for the 2 s pause.
    pub fn waitUntil(g: *Gate, read: *const fn (*anyopaque) anyerror!f64, sleep_fn: *const fn (*anyopaque) void, ctx: *anyopaque) !void {
        const bands = g.bands orelse return;
        var waiting = false;
        while (true) {
            const max_c = try read(ctx);
            if (allow(bands, max_c, waiting)) return;
            waiting = true;
            sleep_fn(ctx);
            g.waited_s += 2;
        }
    }

    /// The real pause. One process contributes its own hottest zone. `gatheredMax` is there for a caller that has several.
    pub fn beforePromptChunk(g: *Gate, io: std.Io) !void {
        if (g.bands == null) return;
        const Ctx = struct {
            gate: *Gate,
            io: std.Io,
            fn read(ptr: *anyopaque) !f64 {
                const c: *@This() = @ptrCast(@alignCast(ptr));
                var dir = try std.Io.Dir.cwd().openDir(c.io, c.gate.root, .{ .iterate = true });
                defer dir.close(c.io);
                return gatheredMax(&.{try hottest(c.io, dir)});
            }
            fn sleep(ptr: *anyopaque) void {
                const c: *@This() = @ptrCast(@alignCast(ptr));
                std.Io.sleep(c.io, .fromMilliseconds(2000), .awake) catch {};
            }
        };
        var ctx = Ctx{ .gate = g, .io = io };
        try g.waitUntil(Ctx.read, Ctx.sleep, &ctx);
    }
};

/// Env text is degrees. Both missing means unset.
pub fn bandsFrom(high: ?[]const u8, low: ?[]const u8) !?Bands {
    if (high == null and low == null) return null;
    const hi = high orelse return error.HeatBands;
    const lo = low orelse return error.HeatBands;
    const bands = Bands{ .high_c = try degrees(hi), .low_c = try degrees(lo) };
    if (bands.low_c > bands.high_c) return error.HeatBands;
    return bands;
}

pub fn degrees(text: []const u8) !f64 {
    return std.fmt.parseFloat(f64, std.mem.trim(u8, text, " \t\r\n"));
}

/// A zone file is millidegrees. 94000 is 94.0 celsius.
pub fn zoneCelsius(text: []const u8) !f64 {
    const milli = try std.fmt.parseInt(i64, std.mem.trim(u8, text, " \t\r\n"), 10);
    return @as(f64, @floatFromInt(milli)) / 1000.0;
}

/// Run when idle and at or under high, or when already waiting and at or under low.
pub fn allow(bands: Bands, max_c: f64, waiting: bool) bool {
    if (waiting) return max_c <= bands.low_c;
    return max_c <= bands.high_c;
}

pub fn gatheredMax(xs: []const f64) f64 {
    var m = xs[0];
    for (xs[1..]) |x| m = @max(m, x);
    return m;
}

/// Hottest `thermal_zone*/temp` under `dir`. Other names are ignored. No readable zone is an error.
pub fn hottest(io: std.Io, dir: std.Io.Dir) !f64 {
    var it = dir.iterate();
    var max: ?f64 = null;
    var path_buf: [96]u8 = undefined;
    var file_buf: [64]u8 = undefined;
    while (try it.next(io)) |e| {
        if (e.kind != .directory and e.kind != .sym_link) continue;
        if (!std.mem.startsWith(u8, e.name, "thermal_zone")) continue;
        const path = std.fmt.bufPrint(&path_buf, "{s}/temp", .{e.name}) catch continue;
        const text = dir.readFile(io, path, &file_buf) catch continue;
        const c = zoneCelsius(text) catch continue;
        max = if (max) |m| @max(m, c) else c;
    }
    return max orelse error.NoZone;
}

pub fn note(seconds_: f64) void {
    if (@import("builtin").is_test) return;
    std.log.info("heat_wait_s {d:.1}", .{seconds_});
}

fn envSpan(name: [*:0]const u8) ?[]const u8 {
    const p = std.c.getenv(name) orelse return null;
    const s = std.mem.span(p);
    if (s.len == 0) return null;
    return s;
}

test "bands parse as degrees and a lone value is refused" {
    try std.testing.expect((try bandsFrom(null, null)) == null);
    try std.testing.expectError(error.HeatBands, bandsFrom("92", null));
    try std.testing.expectError(error.HeatBands, bandsFrom(null, "88"));
    try std.testing.expectError(error.HeatBands, bandsFrom("88", "92"));
    const bands = (try bandsFrom("92", " 88\n")).?;
    try std.testing.expectEqual(@as(f64, 92), bands.high_c);
    try std.testing.expectEqual(@as(f64, 88), bands.low_c);
    try std.testing.expectEqual(@as(f64, 94), try zoneCelsius("94000\n"));
}

test "the chunk runs under the high band, and a hot reading waits until the low band" {
    const bands = Bands{ .high_c = 92, .low_c = 88 };
    try std.testing.expect(allow(bands, 92, false));
    try std.testing.expect(!allow(bands, 92.5, false));
    try std.testing.expect(!allow(bands, 90, true));
    try std.testing.expect(allow(bands, 88, true));
    try std.testing.expectEqual(@as(f64, 94), gatheredMax(&.{ 80, 94, 91 }));
    const Script = struct {
        temps: []const f64,
        i: usize = 0,
        sleeps: u32 = 0,
        fn read(ptr: *anyopaque) !f64 {
            const s: *@This() = @ptrCast(@alignCast(ptr));
            const t = s.temps[s.i];
            if (s.i + 1 < s.temps.len) s.i += 1;
            return t;
        }
        fn sleep(ptr: *anyopaque) void {
            @as(*@This(), @ptrCast(@alignCast(ptr))).sleeps += 1;
        }
    };
    var hot = Script{ .temps = &.{ 95, 93, 88 } };
    var gate = Gate{ .bands = bands };
    try gate.waitUntil(Script.read, Script.sleep, &hot);
    try std.testing.expectEqual(@as(u32, 2), hot.sleeps);
    try std.testing.expectEqual(@as(f64, 4), gate.waited_s);
    var cool = Script{ .temps = &.{90} };
    var idle = Gate{ .bands = bands };
    try idle.waitUntil(Script.read, Script.sleep, &cool);
    try std.testing.expectEqual(@as(u32, 0), cool.sleeps);
    var off = Gate{};
    try off.waitUntil(struct {
        fn read(_: *anyopaque) !f64 {
            return error.Read;
        }
    }.read, Script.sleep, &cool);
    try std.testing.expectEqual(@as(f64, 0), off.waited_s);
}

test "hottest zone is the maximum millidegree reading" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "thermal_zone0");
    try tmp.dir.createDirPath(io, "thermal_zone1");
    try tmp.dir.createDirPath(io, "other");
    try tmp.dir.writeFile(io, .{ .sub_path = "thermal_zone0/temp", .data = "80000\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "thermal_zone1/temp", .data = "94000\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "other/temp", .data = "99000\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "thermal_zone2", .data = "not-a-zone" });
    try std.testing.expectEqual(@as(f64, 94), try hottest(io, tmp.dir));
}
