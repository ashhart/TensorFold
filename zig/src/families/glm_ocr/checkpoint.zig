//! Validated, read-only checkpoint mappings; no device initialization or weight conversion.
const std = @import("std");
const st = @import("core").safetensors;
const config = @import("config.zig");
const schema = @import("schema.zig");

pub const Checkpoint = struct {
    file: st.File,
    io: std.Io,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Checkpoint {
        try validate(a, io, dir, "config.json", config.validate);
        try validate(a, io, dir, "preprocessor_config.json", config.validateProcessor);
        const path = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
        defer a.free(path);
        var file = try st.File.open(a, io, path);
        errdefer file.close(io);
        try schema.validate(&file.names);
        return .{ .file = file, .io = io };
    }

    pub fn close(s: *Checkpoint) void {
        s.file.close(s.io);
        s.* = undefined;
    }

    pub fn get(s: *const Checkpoint, name: []const u8) !st.Tensor {
        return s.file.get(name) orelse error.MissingGlmOcrTensor;
    }
};

fn validate(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, check: anytype) !void {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4 << 20));
    defer a.free(bytes);
    try check(a, bytes);
}
