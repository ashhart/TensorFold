//! Qwen4's Metal contract suites behind one hook: the QSA attention variants, the projections and GDN.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const g = contract.gates;

const qwen4_hc = @import("metal_qwen4_hc.zig");
const qwen4_qsa_select = @import("metal_qwen4_qsa_select.zig");
const qwen4_qsa_scores = @import("metal_qwen4_qsa_scores.zig");
const qwen4_qsa_attn = @import("metal_qwen4_qsa_attn.zig");
const qwen4_qsa_norm = @import("metal_qwen4_qsa_norm.zig");
const qwen4_ple = @import("metal_qwen4_ple.zig");
const qwen4_gdn = @import("metal_qwen4_gdn.zig");

pub const Suites = struct {
    hc: *qwen4_hc.Suite,
    qsa_select: *qwen4_qsa_select.Suite,
    qsa_scores: *qwen4_qsa_scores.Suite,
    qsa_attn: *qwen4_qsa_attn.Suite,
    qsa_norm: *qwen4_qsa_norm.Suite,
    ple: *qwen4_ple.Suite,
    gdn: *qwen4_gdn.Suite,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !Suites {
        const hc = try qwen4_hc.Suite.create(a, device, queue, kernels, only);
        errdefer hc.destroy();
        const qsa_select = try qwen4_qsa_select.Suite.create(a, device, queue, kernels, only);
        errdefer qsa_select.destroy();
        const qsa_scores = try qwen4_qsa_scores.Suite.create(a, device, queue, kernels, only);
        errdefer qsa_scores.destroy();
        const qsa_attn = try qwen4_qsa_attn.Suite.create(a, device, queue, kernels, only);
        errdefer qsa_attn.destroy();
        const qsa_norm = try qwen4_qsa_norm.Suite.create(a, device, queue, kernels, only);
        errdefer qsa_norm.destroy();
        const ple = try qwen4_ple.Suite.create(a, device, queue, kernels, only);
        errdefer ple.destroy();
        const gdn = try qwen4_gdn.Suite.create(a, device, queue, kernels, only);
        errdefer gdn.destroy();
        return .{ .hc = hc, .qsa_select = qsa_select, .qsa_scores = qsa_scores, .qsa_attn = qsa_attn, .qsa_norm = qsa_norm, .ple = ple, .gdn = gdn };
    }

    pub fn destroy(s: *Suites) void {
        s.gdn.destroy();
        s.ple.destroy();
        s.qsa_norm.destroy();
        s.qsa_attn.destroy();
        s.qsa_scores.destroy();
        s.qsa_select.destroy();
        s.hc.destroy();
    }
};
