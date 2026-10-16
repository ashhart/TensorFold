//! --record on CUDA: a recorder in front of a family's lane backend, refused before the host is made.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

/// --record's folder, its recorder and the family's backend behind it.
pub const Recording = struct {
    dir: [:0]u8,
    recorder: lanes.recorder.Recorder,
    recorded: lanes.recorder.Recorded,

    pub fn free(r: *Recording, gpa: Allocator) void {
        r.recorder.deinit();
        gpa.free(r.dir);
        gpa.destroy(r);
    }
};

/// The rows a family's opened engine gives a recorder; a family without the field gives none.
pub fn taps(loaded: anytype) []const u32 {
    return if (@hasField(@TypeOf(loaded), "taps")) loaded.taps else &.{};
}

/// Why `dir` cannot be recorded with `rows` (null: it can); the folder is made on the way.
pub fn refusal(a: Allocator, io: std.Io, model_type: []const u8, dir: []const u8, rows: []const u32) !?[]const u8 {
    if (rows.len == 0) {
        return try std.fmt.allocPrint(a, "--record: {s} checkpoints give no rows to record on CUDA yet", .{model_type});
    }
    std.Io.Dir.cwd().createDirPath(io, dir) catch |e| {
        return try std.fmt.allocPrint(a, "--record {s}: cannot make the folder ({s})", .{ dir, @errorName(e) });
    };
    return null;
}

/// A recorder writing under `dir` in front of `inner`.
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, rows: []const u32, inner: lanes.backend.Backend) !*Recording {
    const r = try gpa.create(Recording);
    errdefer gpa.destroy(r);
    r.dir = try gpa.dupeSentinel(u8, dir, 0);
    errdefer gpa.free(r.dir);
    r.recorder = try .init(gpa, io, r.dir, rows);
    errdefer r.recorder.deinit();
    r.recorded = try .init(inner, &r.recorder);
    return r;
}
