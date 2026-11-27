//! TFQ3 launch geometry and nvcc 13.0's usage on sm_80/86/89/120/121, measured on a GB10; later compilers stay pending.
const std = @import("std");
const reg = @import("registry.zig");
const U = @FieldType(reg.CudaRecord, "usage");
const measured = [_]struct { []const u8, U }{
    .{ "tf_tfq3_fwht_x", .{ .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 } } },
    .{ "tf_tfq3_fwht_y", .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 35, .static = 0, .bound = 0 }, .{ .regs = 35, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 } } },
    .{ "tf_tfq3_moe_gu_swiglu", .{ .{ .regs = 32, .static = 1024, .bound = 0 }, .{ .regs = 40, .static = 1024, .bound = 0 }, .{ .regs = 40, .static = 1024, .bound = 0 }, .{ .regs = 46, .static = 1024, .bound = 0 }, .{ .regs = 46, .static = 1024, .bound = 0 } } },
    .{ "tf_tfq3_moe_gu_softcap", .{ .{ .regs = 32, .static = 1024, .bound = 0 }, .{ .regs = 40, .static = 1024, .bound = 0 }, .{ .regs = 40, .static = 1024, .bound = 0 }, .{ .regs = 43, .static = 1024, .bound = 0 }, .{ .regs = 43, .static = 1024, .bound = 0 } } },
    .{ "tf_tfq3_moe_down", .{ .{ .regs = 40, .static = 512, .bound = 0 }, .{ .regs = 43, .static = 512, .bound = 0 }, .{ .regs = 43, .static = 512, .bound = 0 }, .{ .regs = 43, .static = 512, .bound = 0 }, .{ .regs = 43, .static = 512, .bound = 0 } } },
    .{ "tf_tfq3_glu_contract", .{ .{ .regs = 8, .static = 0, .bound = 0 }, .{ .regs = 8, .static = 0, .bound = 0 }, .{ .regs = 8, .static = 0, .bound = 0 }, .{ .regs = 8, .static = 0, .bound = 0 }, .{ .regs = 8, .static = 0, .bound = 0 } } },
    .{ "tf_tfq3_wmma_sk8_3b", .{ .{ .regs = 60, .static = 7424, .bound = 0 }, .{ .regs = 48, .static = 7424, .bound = 0 }, .{ .regs = 48, .static = 7424, .bound = 0 }, .{ .regs = 40, .static = 7424, .bound = 0 }, .{ .regs = 40, .static = 7424, .bound = 0 } } },
    .{ "tf_tfq3_wmma_fused_3b", .{ .{ .regs = 47, .static = 10496, .bound = 0 }, .{ .regs = 47, .static = 10496, .bound = 0 }, .{ .regs = 47, .static = 10496, .bound = 0 }, .{ .regs = 48, .static = 10496, .bound = 0 }, .{ .regs = 48, .static = 10496, .bound = 0 } } },
    .{ "tf_tfq3_wmma_prefill64_3b", .{ .{ .regs = 128, .static = 32768, .bound = 0 }, .{ .regs = 122, .static = 32768, .bound = 0 }, .{ .regs = 122, .static = 32768, .bound = 0 }, .{ .regs = 130, .static = 32768, .bound = 0 }, .{ .regs = 130, .static = 32768, .bound = 0 } } },
    .{ "tf_tfq3_wmma_sk8_4b", .{ .{ .regs = 55, .static = 7424, .bound = 0 }, .{ .regs = 48, .static = 7424, .bound = 0 }, .{ .regs = 48, .static = 7424, .bound = 0 }, .{ .regs = 48, .static = 7424, .bound = 0 }, .{ .regs = 48, .static = 7424, .bound = 0 } } },
    .{ "tf_tfq3_wmma_fused_4b", .{ .{ .regs = 47, .static = 10496, .bound = 0 }, .{ .regs = 47, .static = 10496, .bound = 0 }, .{ .regs = 47, .static = 10496, .bound = 0 }, .{ .regs = 48, .static = 10496, .bound = 0 }, .{ .regs = 48, .static = 10496, .bound = 0 } } },
    .{ "tf_tfq3_wmma_prefill64_4b", .{ .{ .regs = 128, .static = 32768, .bound = 0 }, .{ .regs = 128, .static = 32768, .bound = 0 }, .{ .regs = 128, .static = 32768, .bound = 0 }, .{ .regs = 128, .static = 32768, .bound = 0 }, .{ .regs = 128, .static = 32768, .bound = 0 } } },
    .{ "tf_tfq3_wmma_sk8_5b", .{ .{ .regs = 62, .static = 7424, .bound = 0 }, .{ .regs = 48, .static = 7424, .bound = 0 }, .{ .regs = 48, .static = 7424, .bound = 0 }, .{ .regs = 48, .static = 7424, .bound = 0 }, .{ .regs = 48, .static = 7424, .bound = 0 } } },
    .{ "tf_tfq3_wmma_fused_5b", .{ .{ .regs = 48, .static = 10496, .bound = 0 }, .{ .regs = 48, .static = 10496, .bound = 0 }, .{ .regs = 48, .static = 10496, .bound = 0 }, .{ .regs = 48, .static = 10496, .bound = 0 }, .{ .regs = 48, .static = 10496, .bound = 0 } } },
    .{ "tf_tfq3_wmma_prefill64_5b", .{ .{ .regs = 128, .static = 32768, .bound = 0 }, .{ .regs = 128, .static = 32768, .bound = 0 }, .{ .regs = 128, .static = 32768, .bound = 0 }, .{ .regs = 128, .static = 32768, .bound = 0 }, .{ .regs = 128, .static = 32768, .bound = 0 } } },
};
fn usage(comptime symbol: []const u8) U {
    for (measured) |m| if (std.mem.eql(u8, m[0], symbol)) return m[1];
    @compileError("no measured TFQ3 usage for " ++ symbol);
}
pub const cuda = blk: {
    var out: [15]reg.CudaRecord = undefined;
    const fixed = .{
        .{ "tf_tfq3_fwht_x", "tfq3", 32 },
        .{ "tf_tfq3_fwht_y", "tfq3", 32 },
        .{ "tf_tfq3_moe_gu_swiglu", "tfq3_moe", 256 },
        .{ "tf_tfq3_moe_gu_softcap", "tfq3_moe", 256 },
        .{ "tf_tfq3_moe_down", "tfq3_moe", 256 },
        .{ "tf_tfq3_glu_contract", "tfq3_moe", 32 },
    };
    for (fixed, 0..) |entry, i| out[i] = .{
        .launch = .{ .name = "tfq3 " ++ entry[0], .symbol = entry[0], .image = entry[1], .threads = entry[2], .min_sm = 80 },
        .usage = usage(entry[0]),
        .later_pending = true,
    };
    var i: usize = fixed.len;
    for (.{ "3b", "4b", "5b" }) |bits| for (.{ "sk8", "fused", "prefill64" }) |kind| {
        const symbol = "tf_tfq3_wmma_" ++ kind ++ "_" ++ bits;
        out[i] = .{
            .launch = .{ .name = "tfq3 " ++ symbol, .symbol = symbol, .image = "tfq3", .threads = if (kind[0] == 's') 128 else 256, .min_sm = 80 },
            .usage = usage(symbol),
            .later_pending = true,
        };
        i += 1;
    };
    break :blk out;
};
