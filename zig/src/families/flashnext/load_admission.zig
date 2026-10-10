//! Host admission runs before the Flash Next loader allocates or initializes a device.
const std = @import("std");
const config = @import("config.zig");
const index = @import("index.zig");

pub const problem = "Flash Next runtime kernels require the supported 6-bit affine, group_size 32 format; use a matching checkpoint";

pub fn problemFor(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.UnsupportedFlashAffineKernel, error.UnsupportedAffineBits, error.UnsupportedAffineGroup, error.UnsupportedAffineMode => problem,
        else => null,
    };
}

pub fn check(gpa: std.mem.Allocator, io: std.Io, directory: []const u8) !void {
    var c = try config.Config.read(gpa, io, directory);
    defer c.deinit();
    try c.global_affine.checkFlashKernel();
    const path = try std.fs.path.join(gpa, &.{ directory, "model.safetensors.index.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(bytes);
    _ = try index.admitRuntime(gpa, bytes, &c);
}

test {
    _ = @import("load_admission_test.zig");
}
