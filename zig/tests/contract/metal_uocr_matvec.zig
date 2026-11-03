//! Unlimited-OCR-2's matvec kernels under the contract (scaffold: filled in by its own change).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const uocr = @import("metal_uocr.zig");

const g = contract.gates;

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        _ = s;
        _ = only;
    }
    pub fn deinit(s: *Suite) void {
        _ = s;
    }
};

pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?uocr.Texts {
    if (!std.mem.startsWith(u8, name, "uocr matvec ")) return null;
    return error.UnknownLaunch;
}
