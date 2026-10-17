//! Whether the backbone launches with PDL as Kernels.load decides it for the embedded images; no checkpoint needed.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");
const check = @import("check.zig");

const Kernels = drama.backbone.kernels.Kernels;

pub fn run(gpu: check.Gpu, expect: []const u8, kernel_dir: ?[]const u8) !void {
    const want = if (std.mem.eql(u8, expect, "on")) true else if (std.mem.eql(u8, expect, "off")) false else return error.Usage;
    var k = try Kernels.load(gpu.gpa, gpu.io, gpu.d, "");
    defer k.unload();
    try check.expect(k.pdl == want and k.gdn.pdl == want, "embedded backbone images: PDL {}, expected {}", .{ k.pdl, want });
    check.pass("embedded backbone images: PDL {s}", .{expect});
    const dir = kernel_dir orelse return;
    var s = try Kernels.load(gpu.gpa, gpu.io, gpu.d, dir);
    defer s.unload();
    try check.expect(!s.pdl and !s.gdn.pdl, "standalone images in {s} launch with PDL", .{dir});
    check.pass("standalone images in {s}: PDL off", .{dir});
}
