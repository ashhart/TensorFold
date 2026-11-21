//! zonos3-generate --speaker-check: the speaker encoder on a dumped clip, stage by stage against the reference's.
const std = @import("std");
const z3 = @import("zonos3");
const gen = @import("generate.zig");

const Check = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    low: usize = 0,
    seen: usize = 0,
    worst: f64 = 1,
    last_ns: i128 = 0,

    fn tap(ctx: *anyopaque, name: []const u8, values: []const f32, rows: usize, width: usize) anyerror!void {
        const c: *Check = @ptrCast(@alignCast(ctx));
        const path = try std.fmt.allocPrint(c.gpa, "{s}/{s}.npy", .{ c.dir, name });
        defer c.gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(c.io, path, c.gpa, .limited(1 << 30));
        defer c.gpa.free(bytes);
        const want = try (try z3.npy.parse(bytes)).floats(c.gpa);
        defer c.gpa.free(want);
        if (want.len != rows * width) return error.SpeakerDumpShape;
        var worst: f64 = 1;
        var max_d: f64 = 0;
        for (0..rows) |r| {
            var dot: f64 = 0;
            var mine: f64 = 0;
            var theirs: f64 = 0;
            for (values[r * width ..][0..width], want[r * width ..][0..width]) |x, y| {
                dot += @as(f64, x) * y;
                mine += @as(f64, x) * x;
                theirs += @as(f64, y) * y;
                max_d = @max(max_d, @abs(@as(f64, x) - y));
            }
            if (mine > 1e-12 or theirs > 1e-12) worst = @min(worst, dot / @sqrt(@max(mine * theirs, 1e-300)));
        }
        c.seen += 1;
        c.low += @intFromBool(worst < 0.9999);
        c.worst = @min(c.worst, worst);
        const now = gen.nowNs(c.io);
        std.debug.print("  {s:<10} rows {d:>5} worst-row cos {d:.7} max|d| {e:.3} after {d:.3} s\n", .{ name, rows, worst, max_d, @as(f64, @floatFromInt(now - c.last_ns)) / 1e9 });
        c.last_ns = gen.nowNs(c.io);
    }
};

/// DIR/audio.npy through the encoder at `checkpoint`: every stage and the embedding beside the reference's.
pub fn run(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, checkpoint: []const u8) !u8 {
    const e = try z3.speaker.load(gpa, io, checkpoint);
    defer e.deinit();
    const path = try std.fs.path.join(gpa, &.{ dir, "audio.npy" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    const audio = try (try z3.npy.parse(bytes)).floats(gpa);
    defer gpa.free(audio);
    var c: Check = .{ .gpa = gpa, .io = io, .dir = dir };
    const t0 = gen.nowNs(io);
    c.last_ns = t0;
    const got = try e.embed(gpa, audio, .{ .ctx = &c, .f = Check.tap });
    const checked = @as(f64, @floatFromInt(gen.nowNs(io) - t0)) / 1e9;
    const t1 = gen.nowNs(io);
    _ = try e.embed(gpa, audio, null);
    const seconds = @as(f64, @floatFromInt(gen.nowNs(io) - t1)) / 1e9;
    try Check.tap(&c, "embedding", &got, 1, got.len);
    const pass = c.low == 0 and c.seen > 0;
    std.debug.print("RESULT zonos3 speaker {s} stages={d} low={d} worst_row_cos={d:.7} audio_s={d:.2} seconds={d:.3} checked_seconds={d:.3}\n", .{ if (pass) "PASS" else "FAIL", c.seen, c.low, c.worst, @as(f64, @floatFromInt(audio.len)) / 44100.0, seconds, checked });
    return if (pass) 0 else 1;
}
