//! Native NVFP4 prefill launch receipts from nvcc 13.5 for sm_120a and sm_121a.
const std = @import("std");
const r = @import("registry.zig");
const limits = @import("limits.zig");

fn entry(comptime symbol: []const u8, comptime registers: u32, comptime shared: u32) r.CudaRecord {
    const used: limits.Usage = .{ .regs = registers, .static = shared, .bound = 256 };
    const none: limits.Usage = .{ .regs = 0, .static = 0 };
    return .{
        .launch = .{ .name = "native NVFP4 " ++ symbol, .symbol = symbol, .image = "nvfp4_prompt", .threads = 256, .min_sm = 120 },
        .usage = .{ none, none, none, used, used },
        .first_nvcc = .{ .major = 13, .minor = 5 },
    };
}

const prompt = [_]r.CudaRecord{
    entry("tf_nvfp4_quant", 33, 0),
    entry("tf_nvfp4_prompt", 69, 13824),
    entry("tf_nvfp4_prompt_f32", 69, 13824),
};

fn tma(comptime symbol: []const u8, comptime registers: u32, comptime rows: u32) r.CudaRecord {
    const threads = if (rows == 0) 256 else 288;
    const used: limits.Usage = .{ .regs = registers, .static = 0, .bound = threads, .local = if (rows == 256) (if (std.mem.indexOf(u8, symbol, "_packed_") != null) 72 else 112) else 0 };
    const none: limits.Usage = .{ .regs = 0, .static = 0 };
    return .{
        .launch = .{ .name = "native NVFP4 TMA " ++ symbol, .symbol = symbol, .image = "nvfp4_tma", .threads = threads, .dynamic = if (rows == 0) 0 else 3 * (rows * 80 + 128 * 72) + 48, .opt_in = rows != 0, .min_sm = 120 },
        .usage = .{ none, none, none, used, used },
        .first_nvcc = .{ .major = 13, .minor = 5 },
    };
}

pub const records = prompt ++ [_]r.CudaRecord{
    tma("tf_nvfp4_pack_mma", 27, 0),
    tma("tf_nvfp4_tma_128", 120, 128),
    tma("tf_nvfp4_tma_128_f32", 120, 128),
    tma("tf_nvfp4_tma_256", 168, 256),
    tma("tf_nvfp4_tma_256_f32", 168, 256),
    tma("tf_nvfp4_tma_swiglu_128", 124, 128),
    tma("tf_nvfp4_tma_swiglu_256", 168, 256),
    tma("tf_nvfp4_tma_packed_128", 125, 128),
    tma("tf_nvfp4_tma_packed_128_f32", 125, 128),
    tma("tf_nvfp4_tma_packed_256", 168, 256),
    tma("tf_nvfp4_tma_packed_256_f32", 168, 256),
    tma("tf_nvfp4_tma_packed_swiglu_128", 125, 128),
    tma("tf_nvfp4_tma_packed_swiglu_256", 168, 256),
};

test "native FP4 records qualify only the measured compiler and Blackwell profiles" {
    for (records) |record| {
        try std.testing.expectEqual(r.none, r.usageFor(record, null));
        try std.testing.expectEqual(r.none, r.usageFor(record, .{ .major = 13, .minor = 0 }));
        try std.testing.expectEqual(record.usage, r.usageFor(record, .{ .major = 13, .minor = 5 }));
    }
    try std.testing.expectEqual(@as(usize, 0), try r.checkCuda(std.testing.allocator, &records, true));
}

test "native FP4 TMA records retain opt-in shared sizes and measured stack frames" {
    for (records[prompt.len..]) |record| {
        const launch = record.launch;
        if (launch.threads == 256) {
            try std.testing.expectEqual(@as(u32, 0), launch.dynamic);
            try std.testing.expect(!launch.opt_in);
        } else {
            try std.testing.expectEqual(@as(u32, 288), launch.threads);
            try std.testing.expect(launch.opt_in);
            const wide = std.mem.indexOf(u8, launch.symbol, "256") != null;
            try std.testing.expectEqual(@as(u32, if (wide) 89136 else 58416), launch.dynamic);
            try std.testing.expectEqual(wide, record.usage[3].local != 0);
        }
    }
}
