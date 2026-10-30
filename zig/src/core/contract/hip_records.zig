//! The HIP launches gate 5 checks: each one's threads and LDS, and what hipcc made of it on every RDNA chip we serve.
const std = @import("std");
const limits = @import("limits.zig");

const U = limits.HipUsage;

/// A HIP launch and what hipcc made of it for gfx1030, gfx1100, gfx1151 and gfx1201 (limits.rdna's order).
pub const Record = struct { launch: limits.HipLaunch, usage: [limits.rdna.len]U };

const rms_ns = "_ZN2tf4rocm10rms_kernelI";

/// The 4-bit work list of 5 to 8 rows (4-row tile): SwiGLU pair at CB 2, plain at CB 4; DotF16 on RDNA2, else DotBF16.
fn listTile(comptime pair: bool) limits.HipLaunch {
    const fmt = "_ZN2tf4rocm23affine_dot2_stream_listINS0_{s}ELi4ELi4ELi{d}ELb{d}EEEvNS0_6AffineENS0_11StreamSidesEiii";
    const cb: u32 = if (pair) 2 else 4;
    const half = std.fmt.comptimePrint(fmt, .{ "6DotF16", cb, @intFromBool(pair) });
    const brain = std.fmt.comptimePrint(fmt, .{ "7DotBF16", cb, @intFromBool(pair) });
    return .{
        .name = std.fmt.comptimePrint("{s} stream work list 4-bit 4x{d}", .{ if (pair) "swiglu" else "plain", cb }),
        .symbol = brain,
        .image = "affine_tiles",
        .threads = 128,
        .symbols = .{ half, brain, brain, brain },
    };
}

/// HIP kernels as launched; tf-hip-kernel-contract limits DIR re-derives the usage records from the code objects.
const own_records = [_]Record{
    .{
        .launch = listTile(true),
        .usage = .{
            .{ .vgprs = 161, .sgprs = 102, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 102, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 102, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 105, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = listTile(false),
        .usage = .{
            .{ .vgprs = 194, .sgprs = 107, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 107, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 107, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 107, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "RMS norm tf_rms_rows", .symbol = "tf_rms_rows", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 58, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 43, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 42, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 42, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "RMS norm rms_kernel<__half>",
            .symbol = rms_ns ++ "6__halfEEvPKT_PKfPS3_if",
            .image = "act",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "RMS norm rms_kernel<hip_bfloat16>",
            .symbol = rms_ns ++ "12hip_bfloat16EEvPKT_PKfPS3_if",
            .image = "act",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_conv", .symbol = "tf_mamba_conv", .image = "mamba", .threads = 256 },
        .usage = .{
            .{ .vgprs = 27, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_dt", .symbol = "tf_mamba_dt", .image = "mamba", .threads = 256 },
        .usage = .{
            .{ .vgprs = 16, .sgprs = 25, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 25, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 25, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 27, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_scan", .symbol = "tf_mamba_scan", .image = "mamba", .threads = 128 },
        .usage = .{
            .{ .vgprs = 127, .sgprs = 41, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 126, .sgprs = 41, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 125, .sgprs = 42, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 42, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_gnorm", .symbol = "tf_mamba_gnorm", .image = "mamba", .threads = 128 },
        .usage = .{
            .{ .vgprs = 23, .sgprs = 23, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 24, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 23, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 24, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "nemotron tf_nemo_select", .symbol = "tf_nemo_select", .image = "ops", .threads = 32 },
        .usage = .{
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "nemotron tf_nemo_select_rank",
            .symbol = "tf_nemo_select_rank",
            .image = "ops",
            .threads = 512,
        },
        .usage = .{
            .{ .vgprs = 27, .sgprs = 30, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 30, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 30, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 29, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "nemotron tf_nemo_relu2", .symbol = "tf_nemo_relu2", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 8, .sgprs = 9, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "nemotron tf_nemo_localize",
            .symbol = "tf_nemo_localize",
            .image = "ops",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 9, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "nemotron tf_nemo_combine", .symbol = "tf_nemo_combine", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 9, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_f16",
            .symbol = "tf_plan_score_f16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 29, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_bf16",
            .symbol = "tf_plan_score_bf16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 29, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_softmax_stats",
            .symbol = "tf_softmax_stats",
            .image = "attention",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 9, .sgprs = 16, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 18, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 16, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 18, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_apply_f16",
            .symbol = "tf_plan_apply_f16",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_apply_bf16",
            .symbol = "tf_plan_apply_bf16",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_sum_partials",
            .symbol = "tf_sum_partials",
            .image = "attention",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 5, .sgprs = 14, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 5, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 5, .sgprs = 14, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 5, .sgprs = 16, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "add rms tf_add_rms_f16", .symbol = "tf_add_rms_f16", .image = "act", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 43, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 44, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "add rms tf_add_rms_bf16", .symbol = "tf_add_rms_bf16", .image = "act", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 44, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 44, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "mamba tf_mamba_conv_window",
            .symbol = "tf_mamba_conv_window",
            .image = "mamba",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 32, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 38, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 37, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 37, .sgprs = 44, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "mamba tf_mamba_scan_snap",
            .symbol = "tf_mamba_scan_snap",
            .image = "mamba",
            .threads = 128,
        },
        .usage = .{
            .{ .vgprs = 137, .sgprs = 50, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 50, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 51, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 131, .sgprs = 52, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
        },
    },
} ++ @import("hip_records_relu2.zig").records ++ [_]Record{
    .{
        .launch = .{
            .name = "planned walk tf_plan_kv_write_act",
            .symbol = "tf_plan_kv_write_act",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 20, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 20, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 20, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 20, .sgprs = 28, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_act_f16",
            .symbol = "tf_plan_score_act_f16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 28, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_act_bf16",
            .symbol = "tf_plan_score_act_bf16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 28, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_sum_partials_act",
            .symbol = "tf_sum_partials_act",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 7, .sgprs = 16, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 16, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "argmax tf_argmax_part", .symbol = "tf_argmax_part", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 8, .sgprs = 17, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 18, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 17, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 16, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "top-k tf_topk_part", .symbol = "tf_topk_part", .image = "ops", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 33, .sgprs = 36, .lds = 1296, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 36, .lds = 1296, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 36, .lds = 1296, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 36, .lds = 1296, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "top-k tf_topk_join", .symbol = "tf_topk_join", .image = "ops", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 16, .sgprs = 36, .lds = 33808, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 36, .lds = 33808, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 36, .lds = 33808, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 37, .lds = 33808, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "argmax tf_argmax_join", .symbol = "tf_argmax_join", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 10, .sgprs = 21, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 21, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 21, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 20, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_attend_f16",
            .symbol = "tf_plan_attend_f16",
            .image = "plan",
            .threads = 1024,
        },
        .usage = .{
            .{ .vgprs = 38, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 38, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 38, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 38, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_attend_bf16",
            .symbol = "tf_plan_attend_bf16",
            .image = "plan",
            .threads = 1024,
        },
        .usage = .{
            .{ .vgprs = 36, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 37, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 36, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 36, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE router (decode, fp32 rows) tf_router_decode_f16", .symbol = "tf_router_decode_f16", .image = "decode", .threads = 128 },
        .usage = .{
            .{ .vgprs = 106, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE router (decode, fp32 rows) tf_router_decode_bf16", .symbol = "tf_router_decode_bf16", .image = "decode", .threads = 128 },
        .usage = .{
            .{ .vgprs = 106, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE router and pick in one launch (decode, bf16 rows) tf_router_select_f16", .symbol = "tf_router_select_f16", .image = "decode", .threads = 128, .dynamic = 2 * 2560 * 2 },
        .usage = .{
            .{ .vgprs = 135, .sgprs = 86, .lds = 5456, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 82, .lds = 5456, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 82, .lds = 5456, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 83, .lds = 5456, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE router and pick in one launch (decode, bf16 rows) tf_router_select_bf16", .symbol = "tf_router_select_bf16", .image = "decode", .threads = 128, .dynamic = 2 * 2560 * 2 },
        .usage = .{
            .{ .vgprs = 135, .sgprs = 86, .lds = 5456, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 82, .lds = 5456, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 82, .lds = 5456, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 83, .lds = 5456, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE down codes read beside the gate and up (side stream) tf_moe_prefetch", .symbol = "tf_moe_prefetch", .image = "decode", .threads = 256 },
        .usage = .{
            .{ .vgprs = 11, .sgprs = 20, .lds = 260, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 20, .lds = 260, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 20, .lds = 260, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 18, .lds = 260, .private = 0, .bound = 256, .wave = 32 },
        },
    },
} ++ @import("hip_records_wmma.zig").records ++ @import("hip_records_pair.zig").records;

pub const records = own_records ++ @import("hip_records_stream.zig").records ++ @import("hip_records_route.zig").records ++
    @import("hip_records_fn_attn.zig").records ++ @import("hip_records_hc.zig").records ++ @import("hip_records_gdn.zig").records;

/// Gate 5's planted HIP controls: a launch past its __launch_bounds__, and 72 KiB of LDS where a workgroup has 64.
pub const controls = [_]Record{
    .{
        .launch = .{
            .name = "control: past its bounds",
            .symbol = "tf_ctl_bounded",
            .image = "contract_controls",
            .threads = 512,
        },
        .usage = .{
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 7, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "control: 72 KiB of LDS",
            .symbol = "tf_ctl_big_lds",
            .image = "contract_controls",
            .threads = 256,
            .dynamic = 32 * 1024,
        },
        .usage = .{
            .{ .vgprs = 6, .sgprs = 9, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
};

/// Every HIP record's problems on every chip we serve, printed; returns how many (a missing record is one).
pub fn check(a: std.mem.Allocator, list: []const Record, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (list) |r| for (limits.rdna, r.usage, r.launch.served) |t, u, served| {
        if (!served) {
            if (u.vgprs != 0) {
                try out.print(a, "{s} on {s}: usage recorded on a chip that does not launch it\n", .{ r.launch.name, t.name });
                n += 1;
            }
        } else if (u.vgprs == 0) {
            try out.print(a, "{s} on {s}: no compiled usage recorded\n", .{ r.launch.name, t.name });
            n += 1;
        } else n += try limits.hip(r.launch, u, t, &out, a);
    };
    if (!quiet and out.items.len > 0) std.debug.print("{s}", .{out.items});
    return n;
}

/// A planted control needs compiled usage and a real limit violation on every chip; returns missed chips.
pub fn controlProblems(a: std.mem.Allocator, r: Record, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (limits.rdna, r.usage) |chip, usage| {
        if (usage.vgprs == 0) {
            n += 1;
            try out.print(a, "{s} on {s}: no compiled usage for control\n", .{ r.launch.name, chip.name });
        } else if (try limits.hip(r.launch, usage, chip, &out, a) == 0) {
            n += 1;
            try out.print(a, "{s} on {s}: planted control fits\n", .{ r.launch.name, chip.name });
        }
    }
    if (!quiet and n != 0) std.debug.print("{s}", .{out.items});
    return n;
}

test "every registered HIP launch fits every RDNA chip we serve, and each planted control breaks one" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 0), try check(a, &records, false));
    for (controls) |r| try std.testing.expectEqual(@as(usize, 0), try controlProblems(a, r, true));
    for (controls) |r| {
        var good = r;
        for (&good.usage) |*u| u.* = .{ .vgprs = 16 };
        good.launch.threads = 256;
        good.launch.dynamic = 0;
        try std.testing.expectEqual(@as(usize, 0), try check(a, &.{good}, true)); // red only for what it plants
    }
}

test "a control cannot fit or lack compiled usage on any one chip" {
    const a = std.testing.allocator;
    for (controls) |r| {
        for (limits.rdna, 0..) |_, i| {
            var fits = r;
            fits.usage[i] = .{ .vgprs = 16, .bound = fits.launch.threads };
            try std.testing.expectEqual(@as(usize, 1), try controlProblems(a, fits, true));
            var missing = r;
            missing.usage[i].vgprs = 0;
            try std.testing.expectEqual(@as(usize, 1), try controlProblems(a, missing, true));
        }
    }
}

test "a chip a launch is not served on records no usage, and recorded usage there is a problem" {
    const a = std.testing.allocator;
    var r: Record = .{ .launch = .{ .name = "gfx11 tile", .symbol = "k", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only }, .usage = undefined };
    for (limits.rdna, &r.usage) |t, *u| u.* = if (std.mem.startsWith(u8, t.name, "gfx11")) .{ .vgprs = 64, .bound = 256 } else .{ .vgprs = 0 };
    try std.testing.expectEqual(@as(usize, 0), try check(a, &.{r}, true));
    r.usage[0] = .{ .vgprs = 64, .bound = 256 };
    try std.testing.expectEqual(@as(usize, 1), try check(a, &.{r}, true));
    // served everywhere: each chip without usage is a problem (gfx1030 and gfx1201 here)
    r.launch.served = @splat(true);
    r.usage[0] = .{ .vgprs = 0 };
    try std.testing.expectEqual(@as(usize, 2), try check(a, &.{r}, true));
}
