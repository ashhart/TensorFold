//! prefill_attention_split.cuh's launches (TF_QWEN_PROMPT_SPLITK): five-SM usage as nvcc 13.4 compiles them.
const reg = @import("registry.zig");
const geo = "PK13__nv_bfloat16PKhS5_PKfS7_iiiiifiPKixiii";
const maxima = "_ZN16tf_prefill_split13scores_kernelILi0ELb{d}EEEv" ++ geo ++ "iiPfS7_PjSA_";
const weights = "_ZN16tf_prefill_split13scores_kernelILi1ELb{d}EEEv" ++ geo ++ "iiPfS7_PjSA_";
const chain = "_ZN16tf_prefill_split12chain_kernelILb{d}EEEv" ++ geo ++ "iPKjS7_S7_PS1_";
const fmt = @import("std").fmt.comptimePrint;
const U = [5]@import("limits.zig").Usage;

fn scores(comptime name: []const u8, comptime symbol: []const u8, usage: U) reg.CudaRecord {
    return .{ .launch = .{ .name = name, .symbol = symbol, .image = "prefill_attention", .threads = 192, .dynamic = 81920, .opt_in = true, .min_sm = 80 }, .usage = usage };
}

fn plain(comptime name: []const u8, comptime symbol: []const u8, threads: u32, dynamic: u32, usage: U) reg.CudaRecord {
    return .{ .launch = .{ .name = name, .symbol = symbol, .image = "prefill_attention", .threads = threads, .dynamic = dynamic, .min_sm = 80 }, .usage = usage };
}

pub const records = [_]reg.CudaRecord{
    scores("prompt split maxima, whole window", fmt(maxima, .{0}), .{ .{ .regs = 165, .static = 0, .bound = 192 }, .{ .regs = 165, .static = 0, .bound = 192 }, .{ .regs = 150, .static = 0, .bound = 192 }, .{ .regs = 142, .static = 0, .bound = 192 }, .{ .regs = 142, .static = 0, .bound = 192 } }),
    scores("prompt split maxima, pages", fmt(maxima, .{1}), .{ .{ .regs = 165, .static = 0, .bound = 192 }, .{ .regs = 165, .static = 0, .bound = 192 }, .{ .regs = 150, .static = 0, .bound = 192 }, .{ .regs = 138, .static = 0, .bound = 192 }, .{ .regs = 138, .static = 0, .bound = 192 } }),
    scores("prompt split weights, whole window", fmt(weights, .{0}), .{ .{ .regs = 151, .static = 0, .bound = 192 }, .{ .regs = 151, .static = 0, .bound = 192 }, .{ .regs = 150, .static = 0, .bound = 192 }, .{ .regs = 159, .static = 0, .bound = 192 }, .{ .regs = 159, .static = 0, .bound = 192 } }),
    scores("prompt split weights, pages", fmt(weights, .{1}), .{ .{ .regs = 151, .static = 0, .bound = 192 }, .{ .regs = 151, .static = 0, .bound = 192 }, .{ .regs = 151, .static = 0, .bound = 192 }, .{ .regs = 154, .static = 0, .bound = 192 }, .{ .regs = 154, .static = 0, .bound = 192 } }),
    plain("prompt split chain, whole window", fmt(chain, .{0}), 192, 8192, .{ .{ .regs = 80, .static = 0, .bound = 192 }, .{ .regs = 80, .static = 0, .bound = 192 }, .{ .regs = 80, .static = 0, .bound = 192 }, .{ .regs = 84, .static = 0, .bound = 192 }, .{ .regs = 84, .static = 0, .bound = 192 } }),
    plain("prompt split chain, pages", fmt(chain, .{1}), 192, 8192, .{ .{ .regs = 80, .static = 0, .bound = 192 }, .{ .regs = 80, .static = 0, .bound = 192 }, .{ .regs = 82, .static = 0, .bound = 192 }, .{ .regs = 88, .static = 0, .bound = 192 }, .{ .regs = 88, .static = 0, .bound = 192 } }),
    plain("prompt split scan", "_ZN16tf_prefill_split11scan_kernelEPKfPfS2_iii", 128, 0, .{ .{ .regs = 39, .static = 0, .bound = 0 }, .{ .regs = 44, .static = 0, .bound = 0 }, .{ .regs = 44, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 } }),
    plain("prompt split diff", "_ZN16tf_prefill_split11diff_kernelEPKtS1_iPj", 256, 0, .{ .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 } }),
};
