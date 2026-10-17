//! The registered launches gate 5 checks: each one's threads and memory, and the limits its compiles recorded.
const std = @import("std");
const limits = @import("limits.zig");

const M = limits.MetalLaunch;
const C = limits.CudaLaunch;
const U = limits.Usage;

/// A CUDA launch and what nvcc made of it for sm_80, sm_86, sm_89, sm_120 and sm_121 (limits.sms' order; regs 0: none).
pub const CudaRecord = struct { launch: C, usage: [limits.sms.len]U };

const none: [limits.sms.len]U = @splat(.{ .regs = 0, .static = 0 });
const group_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const gemv_symbol = "_ZN12tf_lane_gemv11gemv_kernelILi64ELi64ELi8EEEvPK13__nv_bfloat16PKfNS_4PartEiii";
const prompt_symbol = "_ZN11tf_prompt1613prompt_kernelILi{d}ELi128ELi128ELi2ELi2ELi2ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPviiiiii";
const qmmf_f32_symbol = "_ZN7tf_qmmf11qmmf_kernelILi0ELi{d}ELi64ELi1ELi4ELi4ELb1ELb{d}ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
const vision80_symbol = "_ZN20tf_prefill_attention13vision_kernelILi80ELi8EEEvPK13__nv_bfloat16PS1_iif";
const full72_symbol = "_ZN20tf_prefill_attention11full_kernelILi72ELi8ELi8EEEvPK13__nv_bfloat16PS1_iif";

/// CUDA kernels as launched; tf-cuda-test kernel-contract-limits re-derives the usage records.
pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "norms tf_rms_rows_bf16", .symbol = "tf_rms_rows_bf16", .image = "norms", .threads = 256 }, .usage = .{ .{ .regs = 79, .static = 32, .bound = 256 }, .{ .regs = 79, .static = 32, .bound = 256 }, .{ .regs = 79, .static = 32, .bound = 256 }, .{ .regs = 80, .static = 32, .bound = 256 }, .{ .regs = 80, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "norms tf_rms_rows", .symbol = "tf_rms_rows", .image = "norms", .threads = 1024 }, .usage = .{ .{ .regs = 32, .static = 132, .bound = 1024 }, .{ .regs = 46, .static = 132, .bound = 1024 }, .{ .regs = 46, .static = 132, .bound = 1024 }, .{ .regs = 45, .static = 132, .bound = 1024 }, .{ .regs = 45, .static = 132, .bound = 1024 } } },
    .{ .launch = .{ .name = "router tf_nemo_router", .symbol = "tf_nemo_router", .image = "nemotron_route", .threads = 256, .dynamic = 28800 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "router tf_nemo_topk", .symbol = "tf_nemo_topk", .image = "nemotron_route", .threads = 32 }, .usage = .{ .{ .regs = 94, .static = 0, .bound = 32 }, .{ .regs = 95, .static = 0, .bound = 32 }, .{ .regs = 95, .static = 0, .bound = 32 }, .{ .regs = 91, .static = 0, .bound = 32 }, .{ .regs = 91, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "latent tf_super_topk_few", .symbol = "tf_super_topk_few", .image = "nemotron_latent", .threads = 1024 }, .usage = .{ .{ .regs = 32, .static = 4352, .bound = 1024 }, .{ .regs = 38, .static = 4352, .bound = 1024 }, .{ .regs = 38, .static = 4352, .bound = 1024 }, .{ .regs = 30, .static = 4352, .bound = 1024 }, .{ .regs = 30, .static = 4352, .bound = 1024 } } },
    .{ .launch = .{ .name = "qmm_group cluster of 8", .symbol = group_symbol, .image = "qmm_group", .threads = 128, .dynamic = 35328, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmm_group one CTA a tile", .symbol = group_symbol, .image = "qmm_group", .threads = 128, .dynamic = 35328 }, .usage = .{ .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 54, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "lane_gemv", .symbol = gemv_symbol, .image = "lane_gemv", .threads = 128, .dynamic = 35328 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 57, .static = 0, .bound = 128 }, .{ .regs = 57, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "lane_gemv split-K", .symbol = "tf_lane_gemv_split", .image = "lane_gemv", .threads = 128, .dynamic = 17664 }, .usage = .{ .{ .regs = 62, .static = 128, .bound = 128 }, .{ .regs = 56, .static = 128, .bound = 128 }, .{ .regs = 56, .static = 128, .bound = 128 }, .{ .regs = 44, .static = 128, .bound = 128 }, .{ .regs = 44, .static = 128, .bound = 128 } } },
    .{ .launch = .{ .name = "prompt16 FP4", .symbol = std.fmt.comptimePrint(prompt_symbol, .{ 0, 0 }), .image = "prompt16", .threads = 128, .dynamic = 41984 }, .usage = .{ .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "prompt16 FP8", .symbol = std.fmt.comptimePrint(prompt_symbol, .{ 1, 0 }), .image = "prompt16", .threads = 128, .dynamic = 49152 }, .usage = .{ .{ .regs = 243, .static = 0, .bound = 128 }, .{ .regs = 243, .static = 0, .bound = 128 }, .{ .regs = 243, .static = 0, .bound = 128 }, .{ .regs = 218, .static = 0, .bound = 128 }, .{ .regs = 218, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "prompt16 FP4 fp32 out", .symbol = std.fmt.comptimePrint(prompt_symbol, .{ 0, 1 }), .image = "prompt16", .threads = 128, .dynamic = 41984 }, .usage = .{ .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 }, .{ .regs = 212, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 16-row tile", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 16, 0, 0 }), .image = "qmmf", .threads = 128, .dynamic = 17408 }, .usage = .{ .{ .regs = 88, .static = 0, .bound = 128 }, .{ .regs = 88, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 }, .{ .regs = 56, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 16-row tile, cluster of 8", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 16, 1, 0 }), .image = "qmmf", .threads = 128, .dynamic = 17408, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 32-row tile", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 32, 0, 0 }), .image = "qmmf", .threads = 128, .dynamic = 25600 }, .usage = .{ .{ .regs = 106, .static = 0, .bound = 128 }, .{ .regs = 106, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 32-row tile, cluster of 8", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 32, 1, 0 }), .image = "qmmf", .threads = 128, .dynamic = 25600, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 80, .static = 0, .bound = 128 }, .{ .regs = 63, .static = 0, .bound = 128 }, .{ .regs = 63, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 64-row tile", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 64, 0, 0 }), .image = "qmmf", .threads = 128, .dynamic = 41984 }, .usage = .{ .{ .regs = 148, .static = 0, .bound = 128 }, .{ .regs = 148, .static = 0, .bound = 128 }, .{ .regs = 180, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out 64-row tile, cluster of 8", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 64, 1, 0 }), .image = "qmmf", .threads = 128, .dynamic = 41984, .cluster = 8, .min_sm = 90 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 64, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 }, .{ .regs = 78, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf FP4 fp32 out fused slices", .symbol = std.fmt.comptimePrint(qmmf_f32_symbol, .{ 64, 0, 1 }), .image = "qmmf", .threads = 128, .dynamic = 41984 }, .usage = .{ .{ .regs = 182, .static = 0, .bound = 128 }, .{ .regs = 182, .static = 0, .bound = 128 }, .{ .regs = 228, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "qmmf reduce_kernel<true>", .symbol = "_ZN7tf_qmmf13reduce_kernelILb1EEEvPKfPvxif", .image = "qmmf", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 36, .static = 0, .bound = 0 }, .{ .regs = 36, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "ck tf_ck4_up_relu2_few", .symbol = "tf_ck4_up_relu2_few", .image = "ck_experts", .threads = 128, .min_sm = 120 }, .usage = .{ .{ .regs = 19, .static = 0, .bound = 128 }, .{ .regs = 19, .static = 0, .bound = 128 }, .{ .regs = 19, .static = 0, .bound = 128 }, .{ .regs = 125, .static = 0, .bound = 128 }, .{ .regs = 125, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "ck tf_ck4_down_f32_few", .symbol = "tf_ck4_down_f32_few", .image = "ck_experts", .threads = 128, .min_sm = 120 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 128 }, .{ .regs = 24, .static = 0, .bound = 128 }, .{ .regs = 24, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "rank_sum tf_rank_sum", .symbol = "tf_rank_sum", .image = "rank_sum", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit vision_kernel<80, 8> (RADIO)", .symbol = vision80_symbol, .image = "prefill_attention", .threads = 256, .dynamic = 45056 }, .usage = .{ .{ .regs = 143, .static = 0, .bound = 256 }, .{ .regs = 143, .static = 0, .bound = 256 }, .{ .regs = 143, .static = 0, .bound = 256 }, .{ .regs = 122, .static = 0, .bound = 256 }, .{ .regs = 122, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit full_kernel<72> (SigLIP2)", .symbol = full72_symbol, .image = "prefill_attention", .threads = 256, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 152, .static = 0, .bound = 256 }, .{ .regs = 152, .static = 0, .bound = 256 }, .{ .regs = 152, .static = 0, .bound = 256 }, .{ .regs = 169, .static = 0, .bound = 256 }, .{ .regs = 169, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit tf_vit_layer_norm", .symbol = "tf_vit_layer_norm", .image = "vit", .threads = 256 }, .usage = .{ .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "vit tf_vit_pointwise", .symbol = "tf_vit_pointwise", .image = "vit", .threads = 256 }, .usage = .{ .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit tf_vit_bias", .symbol = "tf_vit_bias", .image = "vit", .threads = 256 }, .usage = .{ .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit tf_vit_unshuffle", .symbol = "tf_vit_unshuffle", .image = "vit", .threads = 256 }, .usage = .{ .{ .regs = 29, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 } } },
};

/// Gate 5's planted CUDA controls: a launch past its __launch_bounds__, one needing more registers than a block has.
pub const cuda_controls = [_]CudaRecord{
    .{ .launch = .{ .name = "control: past its bounds", .symbol = "tf_ctl_bounded", .image = "contract_controls", .threads = 512 }, .usage = .{ .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "control: heavy at 1,024 threads", .symbol = "tf_ctl_heavy", .image = "contract_controls", .threads = 1024 }, .usage = .{ .{ .regs = 168, .static = 0, .bound = 0 }, .{ .regs = 166, .static = 0, .bound = 0 }, .{ .regs = 166, .static = 0, .bound = 0 }, .{ .regs = 166, .static = 0, .bound = 0 }, .{ .regs = 166, .static = 0, .bound = 0 } } },
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
    .{ .name = "tile_qmm q4 tf_tile_qmm_q4", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped q4 tf_tile_qmm_grouped_q4", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped glu q4 tf_tile_qmm_grouped_glu_q4", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm twin q4 tf_tile_qmv_q4", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 448, .m2 = 448 },
    .{ .name = "tile_qmm twin indexed q4 tf_tile_qmv_indexed_q4", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 576, .m2 = 576 },
    .{ .name = "tile_qmm twin glu q4 tf_tile_qmv_glu_q4", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 576, .m2 = 576 },
    .{ .name = "tile_qmm twin few q4 tf_tile_qmv_few_q4", .threads = 64, .tg_memory = 16, .dynamic = 10240, .m1 = 576, .m2 = 576 },
    .{ .name = "tile_qmm window rows q4 tf_tile_qmv_rows_q4", .threads = 64, .tg_memory = 4096, .m1 = 640, .m2 = 640 },
    .{ .name = "tile_qmm twin by expert q4 tf_tile_qmv_grouped_q4", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 512, .m2 = 512 },
    .{ .name = "tile_qmm twin by expert glu q4 tf_tile_qmv_grouped_glu_q4", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 512, .m2 = 512 },
    .{ .name = "tile_qmm q6 tf_tile_qmm_q6", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped q6 tf_tile_qmm_grouped_q6", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped glu q6 tf_tile_qmm_grouped_glu_q6", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm twin q6 tf_tile_qmv_q6", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 640, .m2 = 640 },
    .{ .name = "tile_qmm twin indexed q6 tf_tile_qmv_indexed_q6", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 640, .m2 = 640 },
    .{ .name = "tile_qmm twin glu q6 tf_tile_qmv_glu_q6", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 640, .m2 = 640 },
    .{ .name = "tile_qmm twin few q6 tf_tile_qmv_few_q6", .threads = 64, .tg_memory = 16, .dynamic = 10240, .m1 = 512, .m2 = 512 },
    .{ .name = "tile_qmm window rows q6 tf_tile_qmv_rows_q6", .threads = 64, .tg_memory = 4096, .m1 = 576, .m2 = 576 },
    .{ .name = "tile_qmm twin by expert q6 tf_tile_qmv_grouped_q6", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 448, .m2 = 448 },
    .{ .name = "tile_qmm twin by expert glu q6 tf_tile_qmv_grouped_glu_q6", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 384, .m2 = 384 },
    .{ .name = "tile_qmm q8 tf_tile_qmm_q8", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped q8 tf_tile_qmm_grouped_q8", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm grouped glu q8 tf_tile_qmm_grouped_glu_q8", .threads = 128, .tg_memory = 11008, .reserved = 128 },
    .{ .name = "tile_qmm twin q8 tf_tile_qmv_q8", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 896, .m2 = 896 },
    .{ .name = "tile_qmm twin indexed q8 tf_tile_qmv_indexed_q8", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 896, .m2 = 896 },
    .{ .name = "tile_qmm twin glu q8 tf_tile_qmv_glu_q8", .threads = 64, .tg_memory = 0, .dynamic = 10240, .m1 = 832, .m2 = 832 },
    .{ .name = "tile_qmm twin few q8 tf_tile_qmv_few_q8", .threads = 64, .tg_memory = 16, .dynamic = 10240, .m1 = 512, .m2 = 512 },
    .{ .name = "tile_qmm window rows q8 tf_tile_qmv_rows_q8", .threads = 64, .tg_memory = 4096, .m1 = 576, .m2 = 576 },
    .{ .name = "tile_qmm twin by expert q8 tf_tile_qmv_grouped_q8", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 448, .m2 = 448 },
    .{ .name = "tile_qmm twin by expert glu q8 tf_tile_qmv_grouped_glu_q8", .threads = 64, .tg_memory = 32, .dynamic = 5120, .m1 = 384, .m2 = 384 },
    .{ .name = "tile_qmm work list tf_tile_qmm_work", .threads = 1, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "tile_qmm twin picks by expert tf_tile_qmv_group", .threads = 128, .tg_memory = 1024, .m1 = 1024, .m2 = 1024 },
    .{ .name = "tile_qmm group sums tf_tile_xsum", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "glm hc_expand_10 custom_kernel_tf_glm5_fused_hc_expand_cc77af98f2__4096_1_0_0_0_bfloat16_t_bfloat16_t_float_float_floatc_bfloat16_t_float_float", .threads = 1024, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "qwen vision qwv_gemm_bf16_nt_00", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_bf16_nt_01", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_bf16_nt_10", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_bf16_nt_11", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nt_00", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nt_01", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nt_10", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nt_11", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nn_00", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nn_01", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nn_10", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_gemm_f32_nn_11", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_embed", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_layer_norm", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "qwen vision qwv_qkv", .threads = 216, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_softmax", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "qwen vision qwv_heads", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_bias_gelu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_residual", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen vision qwv_bias", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen27 copy q27_take", .threads = 256, .tg_memory = 0, .reserved = 256 },
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
