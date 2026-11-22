//! Qwen3-ASR's Gen12LP launches (qwen3_asr.cl, its _mv, lfm3.cl at GQ 2) and what the driver builds of their SPIR-V.
const std = @import("std");
const limits = @import("xpu_limits.zig");

const X = limits.XpuLaunch;
const U = limits.XpuUsage;

/// A launch as the binding sets it, what the driver reports for it, and its kernel's argument count.
pub const Record = struct { launch: X, usage: U, args: u32 };

const spv = limits.Format.spirv;

fn rec(name: []const u8, symbol: []const u8, image: []const u8, threads: [3]u32, slm: u32, args: u32) Record {
    return built(name, symbol, image, threads, slm, args, 16, 0);
}

/// A kernel the driver builds at its own SIMD width (no required sub-group), with the spill it reports.
fn built(name: []const u8, symbol: []const u8, image: []const u8, threads: [3]u32, slm: u32, args: u32, sub_group: u32, spill: u32) Record {
    return .{
        .launch = .{ .name = name, .symbol = symbol, .image = image, .format = spv, .threads = threads, .slm = slm, .sub_group = sub_group, .grf = 128 },
        .usage = .{ .threads = threads, .slm = slm, .sub_group = sub_group, .grf = 128, .spill = spill },
        .args = args,
    };
}

/// Every launch tf-qwen3-asr-xpu makes of these images (glue's add_bf16 is Nemotron's, outside this family).
pub const records = [_]Record{
    // the 64 x 64 tile builds at SIMD 8; the Linear's spills 1,024 bytes, the conv's none (Iris Xe's driver)
    built("qwen3-asr asr_gemm", "asr_gemm", "qwen3_asr", .{ 16, 16, 1 }, 8192, 12, 8, 1024),
    built("qwen3-asr asr_conv", "asr_conv", "qwen3_asr", .{ 16, 16, 1 }, 8192, 10, 8, 0),
    rec("qwen3-asr asr_flatten", "asr_flatten", "qwen3_asr", .{ 256, 1, 1 }, 0, 6),
    rec("qwen3-asr asr_layernorm", "asr_layernorm", "qwen3_asr", .{ 256, 1, 1 }, 64, 6),
    rec("qwen3-asr asr_gelu", "asr_gelu", "qwen3_asr", .{ 256, 1, 1 }, 0, 3),
    rec("qwen3-asr asr_attn_full", "asr_attn_full", "qwen3_asr", .{ 16, 1, 1 }, 0, 5),
    rec("qwen3-asr asr_qk_rope", "asr_qk_rope", "qwen3_asr", .{ 16, 1, 1 }, 0, 14),
    rec("qwen3-asr asr_gemv", "asr_gemv", "qwen3_asr_mv", .{ 64, 1, 1 }, 0, 5),
    rec("qwen3-asr lfm3_gq2 lfm3_rmsnorm", "lfm3_rmsnorm", "lfm3_gq2", .{ 256, 1, 1 }, 64, 5),
    rec("qwen3-asr lfm3_gq2 lfm3_attn_partial", "lfm3_attn_partial", "lfm3_gq2", .{ 32, 1, 1 }, 0, 13),
    rec("qwen3-asr lfm3_gq2 lfm3_attn_merge_gate", "lfm3_attn_merge_gate", "lfm3_gq2", .{ 16, 1, 1 }, 0, 10),
    rec("qwen3-asr lfm3_gq2 lfm3_swiglu", "lfm3_swiglu", "lfm3_gq2", .{ 64, 1, 1 }, 0, 4),
};

/// Gate 5's planted control: the norm at 1,024 work-items, twice what a Gen12LP work-group holds.
pub const control: Record = blk: {
    var r = rec("control: qwen3-asr asr_layernorm at 1024 work-items", "asr_layernorm", "qwen3_asr", .{ 1024, 1, 1 }, 64, 6);
    r.usage.threads = .{ 256, 1, 1 };
    break :blk r;
};

/// Every problem the records have against the Gen12LP row from their declarations alone; returns how many.
pub fn check(a: std.mem.Allocator, out: *std.ArrayList(u8)) !usize {
    var n: usize = 0;
    for (records) |r| n += try limits.fits(r.launch, limits.gen12lp, out, a);
    return n;
}

test "every Qwen3-ASR XPU launch fits Gen12LP, and the planted control does not" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try std.testing.expectEqual(@as(usize, 0), try check(a, &out));
    try std.testing.expect(try limits.fits(control.launch, limits.gen12lp, &out, a) > 0);
    // the control is only red for its work-group: at the norm's own 256 it fits
    var legal = control.launch;
    legal.threads = .{ 256, 1, 1 };
    try std.testing.expectEqual(@as(usize, 0), try limits.fits(legal, limits.gen12lp, &out, a));
}

test "Gen12LP has no 256-GRF mode, and a work-group of 512 fits at sub-group 16" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const t = limits.gen12lp;
    var l = records[2].launch;
    l.grf = 256;
    try std.testing.expectEqual(@as(usize, 1), try limits.fits(l, t, &out, a));
    l = records[2].launch;
    l.threads = .{ 512, 1, 1 };
    try std.testing.expectEqual(@as(usize, 0), try limits.fits(l, t, &out, a));
    // at SIMD 8 a subslice holds 8 EUs x 7 threads x 8 lanes = 448 work-items: 512 does not fit
    l.sub_group = 8;
    try std.testing.expectEqual(@as(usize, 1), try limits.fits(l, t, &out, a));
    try std.testing.expectEqual(@as(u64, 96), @as(u64, t.eus_xe_core) * t.xe_cores);
}
