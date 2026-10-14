//! The registered launches gate 5 checks: each one's threads and memory, and the limits its compiles recorded.
const std = @import("std");
const limits = @import("limits.zig");

const M = limits.MetalLaunch;
const C = limits.CudaLaunch;
const U = limits.Usage;

/// A CUDA launch and what nvcc made of it for sm_86, sm_89, sm_120 and sm_121 (limits.sms' order; regs 0: none).
pub const CudaRecord = struct { launch: C, usage: [4]U };

const none = [4]U{ .{ .regs = 0, .static = 0 }, .{ .regs = 0, .static = 0 }, .{ .regs = 0, .static = 0 }, .{ .regs = 0, .static = 0 } };
const group_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const gemv_symbol = "_ZN12tf_lane_gemv11gemv_kernelILi64ELi64ELi8EEEvPK13__nv_bfloat16PKfNS_4PartEiii";
const prompt_symbol = "_ZN11tf_prompt1613prompt_kernelILi{d}ELi128ELi128ELi2ELi2ELi2ELb0EEEvPK13__nv_bfloat16PKhS5_fPviiiiii";

/// CUDA kernels as launched; tf-cuda-test kernel-contract-limits re-derives the usage records.
pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "norms tf_rms_rows_bf16", .symbol = "tf_rms_rows_bf16", .image = "norms", .threads = 256 }, .usage = .{ .{ .regs = 79, .static = 36, .bound = 256 }, .{ .regs = 79, .static = 36, .bound = 256 }, .{ .regs = 79, .static = 36, .bound = 256 }, .{ .regs = 79, .static = 36, .bound = 256 } } },
    .{ .launch = .{ .name = "norms tf_rms_rows", .symbol = "tf_rms_rows", .image = "norms", .threads = 1024 }, .usage = .{ .{ .regs = 46, .static = 132, .bound = 1024 }, .{ .regs = 46, .static = 132, .bound = 1024 }, .{ .regs = 45, .static = 132, .bound = 1024 }, .{ .regs = 45, .static = 132, .bound = 1024 } } },
    .{ .launch = .{ .name = "router tf_nemo_router", .symbol = "tf_nemo_router", .image = "nemotron_route", .threads = 256, .dynamic = 28800 }, .usage = .{ .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "router tf_nemo_topk", .symbol = "tf_nemo_topk", .image = "nemotron_route", .threads = 32 }, .usage = .{ .{ .regs = 95, .static = 0, .bound = 32 }, .{ .regs = 95, .static = 0, .bound = 32 }, .{ .regs = 91, .static = 0, .bound = 32 }, .{ .regs = 91, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "qmm_group cluster of 8", .symbol = group_symbol, .image = "qmm_group", .threads = 128, .dynamic = 35328, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmm_group one CTA a tile", .symbol = group_symbol, .image = "qmm_group", .threads = 128, .dynamic = 35328 }, .usage = .{ .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "lane_gemv", .symbol = gemv_symbol, .image = "lane_gemv", .threads = 128, .dynamic = 35328 }, .usage = .{ .{ .regs = 57, .static = 0, .bound = 128 }, .{ .regs = 57, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "lane_gemv split-K", .symbol = "tf_lane_gemv_split", .image = "lane_gemv", .threads = 128, .dynamic = 17664 }, .usage = .{ .{ .regs = 56, .static = 128, .bound = 128 }, .{ .regs = 56, .static = 128, .bound = 128 }, .{ .regs = 44, .static = 128, .bound = 128 }, .{ .regs = 44, .static = 128, .bound = 128 } } },
    .{ .launch = .{ .name = "prompt16 FP4", .symbol = std.fmt.comptimePrint(prompt_symbol, .{0}), .image = "prompt16", .threads = 128, .dynamic = 41984 }, .usage = .{ .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "prompt16 FP8", .symbol = std.fmt.comptimePrint(prompt_symbol, .{1}), .image = "prompt16", .threads = 128, .dynamic = 49152 }, .usage = .{ .{ .regs = 243, .static = 0, .bound = 128 }, .{ .regs = 243, .static = 0, .bound = 128 }, .{ .regs = 218, .static = 0, .bound = 128 }, .{ .regs = 218, .static = 0, .bound = 128 } } },
};

/// Gate 5's planted CUDA controls: a launch past its __launch_bounds__, one needing more registers than a block has.
pub const cuda_controls = [_]CudaRecord{
    .{ .launch = .{ .name = "control: past its bounds", .symbol = "tf_ctl_bounded", .image = "contract_controls", .threads = 512 }, .usage = .{ .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "control: heavy at 1,024 threads", .symbol = "tf_ctl_heavy", .image = "contract_controls", .threads = 1024 }, .usage = .{ .{ .regs = 166, .static = 0, .bound = 0 }, .{ .regs = 166, .static = 0, .bound = 0 }, .{ .regs = 166, .static = 0, .bound = 0 }, .{ .regs = 166, .static = 0, .bound = 0 } } },
};

/// Every CUDA record's problems on every SM we build for, printed; returns how many (a missing record is one).
pub fn checkCuda(a: std.mem.Allocator, list: []const CudaRecord, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (list) |r| for (limits.sms, r.usage) |t, u| {
        if (u.regs == 0) {
            try out.print(a, "{s} on sm_{d}: no compiled usage recorded\n", .{ r.launch.name, t.sm });
            n += 1;
        } else n += try limits.cuda(r.launch, u, t, &out, a);
    };
    if (!quiet and out.items.len > 0) std.debug.print("{s}", .{out.items});
    return n;
}

/// Metal pipelines as the engine launches them; tg_memory is the most any chip reflects, m1 and m2 the compiled limits.
pub const metal = [_]M{
    .{ .name = "softmax tf_softmax_bf16", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "rms tf_rms_norm_bf16", .threads = 1024, .tg_memory = 144, .m1 = 1024, .m2 = 1024 },
    .{ .name = "top-k tf_bf16_topk", .threads = 256, .tg_memory = 4096, .m1 = 768, .m2 = 768 },
    .{ .name = "top-k tf_bf16_topk_max", .threads = 256, .tg_memory = 4112, .m1 = 1024, .m2 = 1024 },
    .{ .name = "top-k tf_bf16_topk_span", .threads = 256, .tg_memory = 4096, .m1 = 704, .m2 = 704 },
    .{ .name = "top-k tf_bf16_topk_merge", .threads = 256, .tg_memory = 4096, .m1 = 1024, .m2 = 1024 },
    .{ .name = "row projection q4 g64 tf_row_projection_q4_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "row projection q6 g32 tf_row_projection_q6_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "row projection q8 g64 tf_row_projection_q8_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "row projection q3 g32 tf_row_projection_q3_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "affine_mm q4 g64 tf_affine_row_sums", .threads = 64, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "affine_mm q4 g64 tf_affine_mm", .threads = 128, .tg_memory = 9728, .offline = false },
    .{ .name = "affine_mm q8 g32 tf_affine_mm", .threads = 128, .tg_memory = 10240, .offline = false },
    .{ .name = "simd_qmm tf_sq_mma_16_4_1", .threads = 512, .tg_memory = 16384, .m1 = 768, .m2 = 768, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_16_2_2", .threads = 512, .tg_memory = 16384, .m1 = 704, .m2 = 704, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_8_4_1", .threads = 256, .tg_memory = 8192, .m1 = 768, .m2 = 768, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_8_4_2", .threads = 256, .tg_memory = 16384, .m1 = 576, .m2 = 576, .adapts = true },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q3_16_4_1", .threads = 512, .tg_memory = 16384, .adapts = true, .m1 = 512, .m2 = 512 },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q6_16_4_1", .threads = 512, .tg_memory = 16384, .adapts = true, .m1 = 512, .m2 = 512 },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q8_16_4_1", .threads = 512, .tg_memory = 16384, .adapts = true, .m1 = 224, .m2 = 224 },
    .{ .name = "lane projection q6 g32 sk1 tf_lane", .threads = 32, .tg_memory = 3072, .offline = false },
    .{ .name = "lane projection q6 g32 sk4 tf_lane", .threads = 128, .tg_memory = 10240, .offline = false },
};

/// Gate 5's planted Metal controls: a launch past M1/M2's compiled limit (#708's class), 36 KiB of threadgroup memory.
pub const metal_controls = [_]M{
    .{ .name = "control: row projection q4 g64 at 1024 threads tf_row_projection_q4_f32", .threads = 1024, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "control: 36 KiB tf_ctl_big_tg", .threads = 256, .tg_memory = 36 * 1024 },
};

/// Every Metal launch's problems on every family we serve, printed; returns how many.
pub fn checkMetal(a: std.mem.Allocator, list: []const M, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (list) |l| for (limits.apple) |f| {
        n += try limits.metal(l, f, &out, a);
    };
    if (!quiet and out.items.len > 0) std.debug.print("{s}", .{out.items});
    return n;
}

test "the registered launches fit every Apple family and every SM, and each planted control breaks one" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 0), try checkMetal(a, &metal, false));
    for (metal_controls) |l| try std.testing.expect(try checkMetal(a, &.{l}, true) > 0);
    try std.testing.expectEqual(@as(usize, 0), try checkCuda(a, &cuda, false));
    for (cuda_controls) |r| try std.testing.expect(try checkCuda(a, &.{r}, true) > 0);
}
