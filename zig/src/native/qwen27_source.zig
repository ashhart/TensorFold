//! The served 27B's weights: a checkpoint folder, or one GGUF file that refuses the options it is not qualified for.
const std = @import("std");
const q = @import("tensorfold").qwen27;

/// Loads `dir` with 128 prompt rows; a GGUF's refusal or reader problem is named in `why` (from `a`).
pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: anytype, a: std.mem.Allocator, why: *[]const u8) !*q.model.Model {
    const gguf = dir.len >= 5 and std.ascii.eqlIgnoreCase(dir[dir.len - 5 ..], ".gguf") and blk: {
        const stat = std.Io.Dir.cwd().statFile(io, dir, .{}) catch break :blk false;
        break :blk stat.kind == .file;
    };
    if (!gguf) return q.model.Model.loadDeferred(gpa, io, dir, 128);
    if (o.drafter != null or o.vision != null or o.learn != null) {
        why.* = "GGUF: --drafter, --vision and disk prompt learning are not qualified for this source";
        return error.UnsupportedGgufOption;
    }
    var problem: q.gguf.Problem = .{};
    return q.model.Model.loadGguf(gpa, io, dir, 128, &problem) catch |err| {
        if (problem.message().len != 0) why.* = try a.dupe(u8, problem.message());
        return err;
    };
}
