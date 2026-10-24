//! Gemma 4 on CUDA (#1048 step 8); cuda_config/cuda_widths copy config/widths (a file belongs to one module).

const std = @import("std");

pub const config = @import("cuda_config.zig");
pub const widths = @import("cuda_widths.zig");
pub const weights = @import("cuda_weights.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const forward = @import("cuda_forward.zig");
pub const snap = @import("cuda_snap.zig");
pub const backend = @import("cuda_backend.zig");
pub const assistant = @import("cuda_assistant.zig");
pub const vision = @import("cuda_vision.zig");
pub const dflash = @import("cuda_dflash.zig");
pub const tower = @import("cuda_tower.zig");
pub const image = @import("cuda_image.zig");
pub const native = @import("cuda_native.zig").Family("gemma4_unified");
pub const native31 = @import("cuda_native.zig").Family("gemma4");

test "cuda_config, cuda_widths and cuda_image are config, widths and image but for their first lines (doc, imports)" {
    // the copies' width tests log the refusals they plant, as Metal's do: warnings, meant
    std.testing.log_level = .err;
    const pairs = [_][2][]const u8{ .{ @embedFile("config.zig"), @embedFile("cuda_config.zig") }, .{ @embedFile("widths.zig"), @embedFile("cuda_widths.zig") }, .{ @embedFile("image.zig"), @embedFile("cuda_image.zig") } };
    for (pairs) |p| {
        var a = std.mem.splitScalar(u8, p[0], '\n');
        var b = std.mem.splitScalar(u8, p[1], '\n');
        var line: usize = 0;
        while (a.next()) |x| : (line += 1) {
            const y = b.next() orelse return error.CopyShorter;
            if (line >= 4 and !std.mem.eql(u8, x, y)) return error.CopyDiffers;
        }
        try std.testing.expect(b.next() == null);
    }
}

test {
    _ = @import("cuda_config.zig");
    _ = @import("cuda_widths.zig");
    _ = @import("cuda_weights.zig");
    _ = @import("cuda_kernels.zig");
    _ = @import("cuda_forward.zig");
    _ = @import("cuda_snap.zig");
    _ = @import("cuda_backend.zig");
    _ = @import("cuda_assistant.zig");
    _ = @import("cuda_assistant_weights.zig");
    _ = @import("cuda_vision.zig");
    _ = @import("cuda_dflash.zig");
    _ = @import("cuda_dflash_test.zig");
    _ = @import("cuda_tower.zig");
    _ = @import("cuda_image.zig");
    _ = @import("cuda_native.zig");
}
