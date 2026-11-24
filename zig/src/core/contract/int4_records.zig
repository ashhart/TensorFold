//! fn_int4's launches: each SM's cubin usage from nvcc 13.0 (13.0.88), the release compiler; 13.3's in cuda_133.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

/// The group AutoGPTQ stores and the kernel reads.
pub const group = 128;
/// The prompt kernel's dynamic shared memory: 2 stages of (64 rows, 16 n8 tiles, one matrix).
pub const prompt_down_smem = 49664;
/// The routed gate/up's: 3 stages of (64 rows, 16 n8 tiles, two matrices).
pub const prompt_gu_smem = 99840;

fn kernel(comptime nt: usize, comptime mt: usize, comptime mats: usize, comptime epi: usize) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10tf_fn_int411int4_kernelILi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi4EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi", .{ group, nt, mt, mats, epi });
}

fn prompt(comptime mats: usize, comptime epi: usize, comptime st: usize) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10tf_fn_int418int4_prompt_kernelILi{d}ELi{d}ELi{d}ELi2ELi4ELi2ELi4ELi{d}EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_Pvifi", .{ group, mats, epi, st });
}

/// What the loader launches: a warp a unit (128 threads) for int4_kernel, a block a 64-row item (256) for the prompts.
fn launch(comptime kind: []const u8, comptime nt: usize, comptime mt: usize, comptime mats: usize, comptime epi: usize) limits.CudaLaunch {
    return .{ .name = std.fmt.comptimePrint("fn_int4 {s} nt {d} mt {d}", .{ kind, nt, mt }), .symbol = kernel(nt, mt, mats, epi), .image = "fn_int4", .threads = 128, .min_sm = 80 };
}

/// Every instantiation the module resolves, each SM's usage from that SM's cubin.
pub const records = [_]registry.CudaRecord{
    .{ .launch = launch("gate-up SwiGLU", 1, 1, 2, 2), .usage = .{ .{ .regs = 139, .static = 0, .bound = 128 }, .{ .regs = 139, .static = 0, .bound = 128 }, .{ .regs = 139, .static = 0, .bound = 128 }, .{ .regs = 148, .static = 0, .bound = 128 }, .{ .regs = 148, .static = 0, .bound = 128 } } },
    .{ .launch = launch("gate-up SwiGLU", 2, 1, 2, 2), .usage = .{ .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 } } },
    .{ .launch = launch("gate-up SwiGLU", 4, 1, 2, 2), .usage = .{ .{ .regs = 246, .static = 0, .bound = 128 }, .{ .regs = 246, .static = 0, .bound = 128 }, .{ .regs = 246, .static = 0, .bound = 128 }, .{ .regs = 251, .static = 0, .bound = 128 }, .{ .regs = 251, .static = 0, .bound = 128 } } },
    .{ .launch = launch("gate-up SwiGLU", 1, 2, 2, 2), .usage = .{ .{ .regs = 224, .static = 0, .bound = 128 }, .{ .regs = 226, .static = 0, .bound = 128 }, .{ .regs = 226, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .launch = launch("gate-up SwiGLU", 2, 2, 2, 2), .usage = .{ .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .launch = launch("gate-up SwiGLU", 4, 2, 2, 2), .usage = .{ .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down fp32", 1, 1, 1, 0), .usage = .{ .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 126, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down fp32", 2, 1, 1, 0), .usage = .{ .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 146, .static = 0, .bound = 128 }, .{ .regs = 146, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down fp32", 4, 1, 1, 0), .usage = .{ .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 166, .static = 0, .bound = 128 }, .{ .regs = 166, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down fp32", 1, 2, 1, 0), .usage = .{ .{ .regs = 198, .static = 0, .bound = 128 }, .{ .regs = 198, .static = 0, .bound = 128 }, .{ .regs = 198, .static = 0, .bound = 128 }, .{ .regs = 186, .static = 0, .bound = 128 }, .{ .regs = 186, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down fp32", 2, 2, 1, 0), .usage = .{ .{ .regs = 225, .static = 0, .bound = 128 }, .{ .regs = 225, .static = 0, .bound = 128 }, .{ .regs = 225, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down fp32", 4, 2, 1, 0), .usage = .{ .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down bf16", 1, 1, 1, 3), .usage = .{ .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 126, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down bf16", 2, 1, 1, 3), .usage = .{ .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 146, .static = 0, .bound = 128 }, .{ .regs = 146, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down bf16", 4, 1, 1, 3), .usage = .{ .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 166, .static = 0, .bound = 128 }, .{ .regs = 166, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down bf16", 1, 2, 1, 3), .usage = .{ .{ .regs = 198, .static = 0, .bound = 128 }, .{ .regs = 198, .static = 0, .bound = 128 }, .{ .regs = 198, .static = 0, .bound = 128 }, .{ .regs = 189, .static = 0, .bound = 128 }, .{ .regs = 189, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down bf16", 2, 2, 1, 3), .usage = .{ .{ .regs = 225, .static = 0, .bound = 128 }, .{ .regs = 224, .static = 0, .bound = 128 }, .{ .regs = 224, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .launch = launch("down bf16", 4, 2, 1, 3), .usage = .{ .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "fn_int4 prompt down fp32", .symbol = prompt(1, 0, 2), .image = "fn_int4", .threads = 256, .dynamic = prompt_down_smem, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 148, .static = 512, .bound = 256 }, .{ .regs = 148, .static = 512, .bound = 256 }, .{ .regs = 148, .static = 512, .bound = 256 }, .{ .regs = 128, .static = 512, .bound = 256 }, .{ .regs = 128, .static = 512, .bound = 256 } } },
    .{ .launch = .{ .name = "fn_int4 prompt down bf16", .symbol = prompt(1, 3, 2), .image = "fn_int4", .threads = 256, .dynamic = prompt_down_smem, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 150, .static = 512, .bound = 256 }, .{ .regs = 150, .static = 512, .bound = 256 }, .{ .regs = 150, .static = 512, .bound = 256 }, .{ .regs = 128, .static = 512, .bound = 256 }, .{ .regs = 128, .static = 512, .bound = 256 } } },
    .{ .launch = .{ .name = "fn_int4 prompt gate-up SwiGLU", .symbol = prompt(2, 2, 3), .image = "fn_int4", .threads = 256, .dynamic = prompt_gu_smem, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 226, .static = 512, .bound = 256 }, .{ .regs = 226, .static = 512, .bound = 256 }, .{ .regs = 226, .static = 512, .bound = 256 }, .{ .regs = 230, .static = 512, .bound = 256 }, .{ .regs = 230, .static = 512, .bound = 256 } } },
};

/// nvcc 13.3 (13.3.73)'s usage where it differs from 13.0 (13.0.88)'s.
pub const cuda_133 = [_]registry.Override{
    .{ .name = "fn_int4 gate-up SwiGLU nt 1 mt 1", .usage = .{ .{ .regs = 141, .static = 0, .bound = 128 }, .{ .regs = 141, .static = 0, .bound = 128 }, .{ .regs = 141, .static = 0, .bound = 128 }, .{ .regs = 150, .static = 0, .bound = 128 }, .{ .regs = 150, .static = 0, .bound = 128 } } },
    .{ .name = "fn_int4 gate-up SwiGLU nt 4 mt 1", .usage = .{ .{ .regs = 246, .static = 0, .bound = 128 }, .{ .regs = 246, .static = 0, .bound = 128 }, .{ .regs = 246, .static = 0, .bound = 128 }, .{ .regs = 252, .static = 0, .bound = 128 }, .{ .regs = 252, .static = 0, .bound = 128 } } },
    .{ .name = "fn_int4 gate-up SwiGLU nt 1 mt 2", .usage = .{ .{ .regs = 226, .static = 0, .bound = 128 }, .{ .regs = 226, .static = 0, .bound = 128 }, .{ .regs = 226, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .name = "fn_int4 down fp32 nt 1 mt 1", .usage = .{ .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 122, .static = 0, .bound = 128 }, .{ .regs = 122, .static = 0, .bound = 128 } } },
    .{ .name = "fn_int4 down fp32 nt 2 mt 1", .usage = .{ .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 139, .static = 0, .bound = 128 }, .{ .regs = 139, .static = 0, .bound = 128 }, .{ .regs = 148, .static = 0, .bound = 128 }, .{ .regs = 148, .static = 0, .bound = 128 } } },
    .{ .name = "fn_int4 down fp32 nt 2 mt 2", .usage = .{ .{ .regs = 227, .static = 0, .bound = 128 }, .{ .regs = 227, .static = 0, .bound = 128 }, .{ .regs = 227, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .name = "fn_int4 down bf16 nt 1 mt 1", .usage = .{ .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 118, .static = 0, .bound = 128 }, .{ .regs = 122, .static = 0, .bound = 128 }, .{ .regs = 122, .static = 0, .bound = 128 } } },
    .{ .name = "fn_int4 down bf16 nt 2 mt 1", .usage = .{ .{ .regs = 126, .static = 0, .bound = 128 }, .{ .regs = 139, .static = 0, .bound = 128 }, .{ .regs = 139, .static = 0, .bound = 128 }, .{ .regs = 148, .static = 0, .bound = 128 }, .{ .regs = 148, .static = 0, .bound = 128 } } },
    .{ .name = "fn_int4 down bf16 nt 1 mt 2", .usage = .{ .{ .regs = 198, .static = 0, .bound = 128 }, .{ .regs = 198, .static = 0, .bound = 128 }, .{ .regs = 198, .static = 0, .bound = 128 }, .{ .regs = 187, .static = 0, .bound = 128 }, .{ .regs = 187, .static = 0, .bound = 128 } } },
    .{ .name = "fn_int4 down bf16 nt 2 mt 2", .usage = .{ .{ .regs = 227, .static = 0, .bound = 128 }, .{ .regs = 227, .static = 0, .bound = 128 }, .{ .regs = 227, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 } } },
    .{ .name = "fn_int4 prompt down fp32", .usage = .{ .{ .regs = 152, .static = 512, .bound = 256 }, .{ .regs = 152, .static = 512, .bound = 256 }, .{ .regs = 152, .static = 512, .bound = 256 }, .{ .regs = 126, .static = 512, .bound = 256 }, .{ .regs = 126, .static = 512, .bound = 256 } } },
    .{ .name = "fn_int4 prompt down bf16", .usage = .{ .{ .regs = 152, .static = 512, .bound = 256 }, .{ .regs = 152, .static = 512, .bound = 256 }, .{ .regs = 152, .static = 512, .bound = 256 }, .{ .regs = 126, .static = 512, .bound = 256 }, .{ .regs = 126, .static = 512, .bound = 256 } } },
};

test "int4: every kept instantiation is declared once, in the live registry, and fits every SM" {
    for (records, 0..) |r, i| {
        for (records[i + 1 ..]) |t| try std.testing.expect(!std.mem.eql(u8, r.launch.symbol, t.launch.symbol));
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(registry.TargetDecision.applicable, registry.targetDecision(r, limits.sms[0], r.usage[0]));
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
        try std.testing.expect(r.launch.dynamic <= limits.sms[4].smem_optin);
    }
}

test "int4: the symbol scheme is the module's own" {
    // the seven template arguments cuda/int4_test.zig pins a compiler's mangling of, as symbol sets of the records
    try std.testing.expectEqualStrings("_ZN10tf_fn_int411int4_kernelILi128ELi4ELi1ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi", records[2].launch.symbol);
    try std.testing.expectEqualStrings("_ZN10tf_fn_int411int4_kernelILi128ELi2ELi2ELi1ELi0ELi4EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi", records[10].launch.symbol);
    try std.testing.expectEqualStrings("_ZN10tf_fn_int411int4_kernelILi128ELi1ELi1ELi1ELi3ELi4EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi", records[12].launch.symbol);
    try std.testing.expectEqualStrings("_ZN10tf_fn_int418int4_prompt_kernelILi128ELi1ELi0ELi2ELi4ELi2ELi4ELi2EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_Pvifi", records[18].launch.symbol);
    try std.testing.expectEqualStrings("_ZN10tf_fn_int418int4_prompt_kernelILi128ELi1ELi3ELi2ELi4ELi2ELi4ELi2EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_Pvifi", records[19].launch.symbol);
    try std.testing.expectEqualStrings("_ZN10tf_fn_int418int4_prompt_kernelILi128ELi2ELi2ELi2ELi4ELi2ELi4ELi3EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_Pvifi", records[20].launch.symbol);
    // and the order the loader expects: three epilogues of six, then the two prompt downs and the prompt gate/up
    try std.testing.expectEqual(@as(usize, 21), records.len);
    for (records[0..6]) |r| try std.testing.expect(std.mem.indexOf(u8, r.launch.symbol, "ELi2ELi2ELi4EEEv") != null);
    for (records[12..18]) |r| try std.testing.expect(std.mem.indexOf(u8, r.launch.symbol, "ELi1ELi3ELi4EEEv") != null);
}
