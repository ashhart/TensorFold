//! MiniMax-H3-Image's CUDA launches.
const std = @import("std");
const limits = @import("limits.zig");
const registry = @import("registry.zig");

fn record(comptime symbol: []const u8, comptime image: []const u8, threads: u32, min_sm: u32, usage: [limits.sms.len]limits.Usage) registry.CudaRecord {
    return .{ .launch = .{ .name = "h3_image " ++ symbol, .symbol = symbol, .image = image, .threads = threads, .min_sm = min_sm }, .usage = usage };
}

/// nvcc 13.0's usage, the compiler of the release fatbins (kernel-contract-limits, 16 Nov 2026).
pub const records = [_]registry.CudaRecord{
    record("h3d_post_quant", "h3_image_decoder", 256, 80, .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } }),
    record("h3d_embed", "h3_image_decoder", 256, 80, .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } }),
    record("h3d_rms_norm", "h3_image_decoder", 256, 80, .{ .{ .regs = 32, .static = 64, .bound = 256 }, .{ .regs = 32, .static = 64, .bound = 256 }, .{ .regs = 32, .static = 64, .bound = 256 }, .{ .regs = 32, .static = 64, .bound = 256 }, .{ .regs = 32, .static = 64, .bound = 256 } }),
    record("h3d_layer_norm", "h3_image_decoder", 256, 80, .{ .{ .regs = 32, .static = 64, .bound = 256 }, .{ .regs = 32, .static = 64, .bound = 256 }, .{ .regs = 32, .static = 64, .bound = 256 }, .{ .regs = 26, .static = 64, .bound = 256 }, .{ .regs = 26, .static = 64, .bound = 256 } }),
    record("h3d_linear", "h3_image_decoder", 256, 80, .{ .{ .regs = 98, .static = 16512, .bound = 256 }, .{ .regs = 98, .static = 16512, .bound = 256 }, .{ .regs = 98, .static = 16512, .bound = 256 }, .{ .regs = 127, .static = 16512, .bound = 256 }, .{ .regs = 127, .static = 16512, .bound = 256 } }),
    record("h3d_finish", "h3_image_decoder", 256, 80, .{ .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 } }),
    record("h3d_heads", "h3_image_decoder", 32, 80, .{ .{ .regs = 23, .static = 0, .bound = 32 }, .{ .regs = 24, .static = 0, .bound = 32 }, .{ .regs = 22, .static = 0, .bound = 32 }, .{ .regs = 21, .static = 0, .bound = 32 }, .{ .regs = 21, .static = 0, .bound = 32 } }),
    record("h3d_attention", "h3_image_decoder", 256, 80, .{ .{ .regs = 37, .static = 11264, .bound = 256 }, .{ .regs = 48, .static = 11264, .bound = 256 }, .{ .regs = 48, .static = 11264, .bound = 256 }, .{ .regs = 48, .static = 11264, .bound = 256 }, .{ .regs = 48, .static = 11264, .bound = 256 } }),
    record("h3d_swiglu", "h3_image_decoder", 256, 80, .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 } }),
    record("h3p_norm", "h3_image_prompt", 256, 80, .{ .{ .regs = 31, .static = 8, .bound = 256 }, .{ .regs = 35, .static = 8, .bound = 256 }, .{ .regs = 35, .static = 8, .bound = 256 }, .{ .regs = 27, .static = 8, .bound = 256 }, .{ .regs = 27, .static = 8, .bound = 256 } }),
    record("h3p_heads", "h3_image_prompt", 256, 80, .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } }),
    record("h3p_attention", "h3_image_prompt", 256, 80, .{ .{ .regs = 31, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } }),
    record("h3p_glu", "h3_image_prompt", 256, 80, .{ .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 17, .static = 0, .bound = 256 }, .{ .regs = 17, .static = 0, .bound = 256 } }),
    record("h3p_add", "h3_image_prompt", 256, 80, .{ .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 } }),
    record("h3p_scatter_add", "h3_image_prompt", 256, 80, .{ .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 } }),
    record("h3p_embed", "h3_image_prompt", 256, 80, .{ .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 } }),
    record("h3p_splice", "h3_image_prompt", 256, 80, .{ .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } }),
    record("h3p_linear", "h3_image_prompt", 256, 80, .{ .{ .regs = 29, .static = 0, .bound = 256 }, .{ .regs = 37, .static = 0, .bound = 256 }, .{ .regs = 37, .static = 0, .bound = 256 }, .{ .regs = 29, .static = 0, .bound = 256 }, .{ .regs = 29, .static = 0, .bound = 256 } }),
    record("h3p_round", "h3_image_prompt", 256, 80, .{ .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 } }),
};

test "MiniMax-H3-Image CUDA: every launch is in the live registry and fits every SM" {
    for (records) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
