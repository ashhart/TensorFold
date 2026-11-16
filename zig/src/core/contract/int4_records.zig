//! The int4 launches (fn_int4.cu): declarations for every kept instantiation, their compiler usage pending.
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
fn row(comptime kind: []const u8, comptime nt: usize, comptime mt: usize, comptime mats: usize, comptime epi: usize) registry.CudaRecord {
    return .{ .launch = .{ .name = std.fmt.comptimePrint("fn_int4 {s} nt {d} mt {d}", .{ kind, nt, mt }), .symbol = kernel(nt, mt, mats, epi), .image = "fn_int4", .threads = 128, .min_sm = 80 }, .usage = registry.none };
}

/// Every instantiation the module resolves, each unmeasured until a compiler receipt fills its usage.
pub const records = [_]registry.CudaRecord{
    row("gate-up SwiGLU", 1, 1, 2, 2),
    row("gate-up SwiGLU", 2, 1, 2, 2),
    row("gate-up SwiGLU", 4, 1, 2, 2),
    row("gate-up SwiGLU", 1, 2, 2, 2),
    row("gate-up SwiGLU", 2, 2, 2, 2),
    row("gate-up SwiGLU", 4, 2, 2, 2),
    row("down fp32", 1, 1, 1, 0),
    row("down fp32", 2, 1, 1, 0),
    row("down fp32", 4, 1, 1, 0),
    row("down fp32", 1, 2, 1, 0),
    row("down fp32", 2, 2, 1, 0),
    row("down fp32", 4, 2, 1, 0),
    row("down bf16", 1, 1, 1, 3),
    row("down bf16", 2, 1, 1, 3),
    row("down bf16", 4, 1, 1, 3),
    row("down bf16", 1, 2, 1, 3),
    row("down bf16", 2, 2, 1, 3),
    row("down bf16", 4, 2, 1, 3),
    .{ .launch = .{ .name = "fn_int4 prompt down fp32", .symbol = prompt(1, 0, 2), .image = "fn_int4", .threads = 256, .dynamic = prompt_down_smem, .opt_in = true, .min_sm = 80 }, .usage = registry.none },
    .{ .launch = .{ .name = "fn_int4 prompt down bf16", .symbol = prompt(1, 3, 2), .image = "fn_int4", .threads = 256, .dynamic = prompt_down_smem, .opt_in = true, .min_sm = 80 }, .usage = registry.none },
    .{ .launch = .{ .name = "fn_int4 prompt gate-up SwiGLU", .symbol = prompt(2, 2, 3), .image = "fn_int4", .threads = 256, .dynamic = prompt_gu_smem, .opt_in = true, .min_sm = 80 }, .usage = registry.none },
};

test "int4: every kept instantiation is declared once, unmeasured and inside the block's shared memory" {
    for (records, 0..) |r, i| {
        for (records[i + 1 ..]) |t| try std.testing.expect(!std.mem.eql(u8, r.launch.symbol, t.launch.symbol));
        try std.testing.expectEqualDeep(registry.none, r.usage);
        // a missing receipt is one problem on each of the five recorded SMs
        try std.testing.expectEqual(limits.sms.len, try registry.checkCuda(std.testing.allocator, &.{r}, true));
        try std.testing.expectEqual(registry.TargetDecision.applicable, registry.targetDecision(r, limits.sms[0], r.usage[0]));
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
