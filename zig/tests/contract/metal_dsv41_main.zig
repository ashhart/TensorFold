//! DeepSeek-V4.1's Metal contract behind one hook: suites and their extra gates.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const g = contract.gates;

const dsv41_attn = @import("metal_dsv41_attn.zig");
const dsv41_fp8_rows = @import("metal_dsv41_fp8_rows.zig");
const dsv41_fp8_mma = @import("metal_dsv41_fp8_mma.zig");
const dsv41_experts = @import("metal_dsv41_experts.zig");
const dsv41_xmma = @import("metal_dsv41_xmma.zig");
const dsv41_glue = @import("metal_dsv41_glue.zig");
const dsv41_mv = @import("metal_dsv41_mv.zig");
const dsv41_glue_tmp = @import("metal_dsv41_glue_tmp.zig");
const dsv41_glue_tmp2 = @import("metal_dsv41_glue_tmp2.zig");

/// Every dsv41 suite, created and destroyed in the order main held them.
pub const Suites = struct {
    fp8: *dsv41_fp8_rows.Suite,
    fp8_mma: *dsv41_fp8_mma.Suite,
    attn: *dsv41_attn.Suite,
    experts: *dsv41_experts.Suite,
    xmma: *dsv41_xmma.Suite,
    glue: *dsv41_glue.Suite,
    mv: *dsv41_mv.Suite,
    glue_tmp: *dsv41_glue_tmp.Suite,
    glue_tmp2: *dsv41_glue_tmp2.Suite,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !Suites {
        const fp8 = try dsv41_fp8_rows.Suite.create(a, device, queue, kernels, only);
        errdefer fp8.destroy();
        const fp8_mma = try dsv41_fp8_mma.Suite.create(a, device, queue, kernels, only);
        errdefer fp8_mma.destroy();
        const attn = try dsv41_attn.Suite.create(a, device, queue, kernels, only);
        errdefer attn.destroy();
        const experts = try dsv41_experts.Suite.create(a, device, queue, kernels, only);
        errdefer experts.destroy();
        const xmma = try dsv41_xmma.Suite.create(a, device, queue, kernels, only);
        errdefer xmma.destroy();
        const glue = try dsv41_glue.Suite.create(a, device, queue, kernels, only);
        errdefer glue.destroy();
        const mv = try dsv41_mv.Suite.create(a, device, queue, kernels, only);
        errdefer mv.destroy();
        const glue_tmp = try dsv41_glue_tmp.Suite.create(a, device, queue, kernels, only);
        errdefer glue_tmp.destroy();
        const glue_tmp2 = try dsv41_glue_tmp2.Suite.create(a, device, queue, kernels, only);
        errdefer glue_tmp2.destroy();
        return .{ .fp8 = fp8, .fp8_mma = fp8_mma, .attn = attn, .experts = experts, .xmma = xmma, .glue = glue, .mv = mv, .glue_tmp = glue_tmp, .glue_tmp2 = glue_tmp2 };
    }

    pub fn destroy(s: *Suites) void {
        s.glue_tmp2.destroy();
        s.glue_tmp.destroy();
        s.mv.destroy();
        s.glue.destroy();
        s.xmma.destroy();
        s.experts.destroy();
        s.attn.destroy();
        s.fp8_mma.destroy();
        s.fp8.destroy();
    }

    /// The dsv41 gates after runAll: equality, invariance and accuracy probes, and timings when asked for.
    pub fn gates(s: *Suites, io: std.Io, seed: u64, t: *g.Tally) !void {
        if (s.experts.probes.items.len > 0) t.failed += try s.experts.pipeline(seed);
        if (std.c.getenv("TF_DSV41_EXPERTS_TIME") != null) try s.experts.timing(io);
        if (s.xmma.probes.items.len > 0) t.failed += try s.xmma.equality(seed);
        if (std.c.getenv("TF_DSV41_XMMA_TIME") != null) try s.xmma.timing(io);
        t.failed += try s.fp8.invariance(seed);
        if (std.c.getenv("TF_DSV41_TIME") != null) try s.fp8.timing(io);
        t.failed += try s.fp8_mma.invariance(seed);
        t.failed += try s.fp8_mma.accuracy(seed);
        if (std.c.getenv("TF_DSV41_TIME") != null) try s.fp8_mma.timing(io);
        try s.attn.invariance(io, seed, t);
        if (std.c.getenv("TF_DSV41_TIME") != null) try s.attn.timing(io);
        t.failed += try s.glue.invariance(seed);
        if (s.mv.probes.items.len > 0) {
            t.failed += try s.mv.invariance(seed);
            t.failed += try s.mv.equality(seed);
            t.failed += try s.mv.head(io, seed, std.c.getenv("TF_DSV41_TIME") != null);
        }
        t.failed += try s.glue.tables();
        t.failed += try s.glue_tmp.invariance(seed);
        t.failed += try s.glue_tmp2.invariance(seed);
    }
};
