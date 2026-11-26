//! TFQ3 launch geometry; compiled resource usage remains unmeasured until the CUDA device receipt.
const reg = @import("registry.zig");
pub const cuda = blk: {
    var out: [14]reg.CudaRecord = undefined;
    const fixed = .{
        .{ "tf_tfq3_fwht_x", "tfq3", 32 },
        .{ "tf_tfq3_fwht_y", "tfq3", 32 },
        .{ "tf_tfq3_moe_gu_swiglu", "tfq3_moe", 256 },
        .{ "tf_tfq3_moe_down", "tfq3_moe", 256 },
        .{ "tf_tfq3_glu_contract", "tfq3_moe", 32 },
    };
    for (fixed, 0..) |entry, i| out[i] = .{
        .launch = .{ .name = "tfq3 " ++ entry[0], .symbol = entry[0], .image = entry[1], .threads = entry[2], .min_sm = 80 },
        .usage = reg.none,
        .later_pending = true,
    };
    var i: usize = fixed.len;
    for (.{ "3b", "4b", "5b" }) |bits| for (.{ "sk8", "fused", "prefill64" }) |kind| {
        const symbol = "tf_tfq3_wmma_" ++ kind ++ "_" ++ bits;
        out[i] = .{
            .launch = .{ .name = "tfq3 " ++ symbol, .symbol = symbol, .image = "tfq3", .threads = if (kind[0] == 's') 128 else 256, .min_sm = 80 },
            .usage = reg.none,
            .later_pending = true,
        };
        i += 1;
    };
    break :blk out;
};
