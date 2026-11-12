//! A complete Parse checkpoint mapped read-only; opening this never initializes a GPU.
const std = @import("std");
const st = @import("core").safetensors;
const config = @import("config.zig");
const schema = @import("schema.zig");

pub const Checkpoint = struct {
    file: st.File,
    io: std.Io,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Checkpoint {
        try validateConfig(a, io, dir, "config.json", config.validate);
        try validateConfig(a, io, dir, "preprocessor_config.json", config.validateProcessor);
        const path = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
        defer a.free(path);
        var file = try st.File.open(a, io, path);
        errdefer file.close(io);
        try schema.validate(&file.names);
        // Summary order is part of the neck, not merely an index tensor's shape.
        const indices = file.get(schema.radio ++ "summary_idxs").?.bytes;
        if (std.mem.readInt(i64, indices[0..8], .little) != 0 or
            std.mem.readInt(i64, indices[8..16], .little) != 1) return error.BadParseSummary;
        return .{ .file = file, .io = io };
    }

    pub fn close(s: *Checkpoint) void {
        s.file.close(s.io);
        s.* = undefined;
    }

    pub fn get(s: *const Checkpoint, name: []const u8) !st.Tensor {
        return s.file.get(name) orelse error.MissingParseTensor;
    }
};

fn validateConfig(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, check: anytype) !void {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4 << 20));
    defer a.free(bytes);
    try check(a, bytes);
}
