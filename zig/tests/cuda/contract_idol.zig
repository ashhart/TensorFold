//! Idol's denoiser kernels (idol.cu) under the contract, with controls that drop one fp16 rounding (idol-contract).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const idol = @import("idol_cuda");
const check = @import("check.zig");
const case = @import("contract_idol_case.zig");
const norms = @import("contract_idol_norms.zig");
const steps = @import("contract_idol_steps.zig");
const rotary = @import("contract_idol_rotary.zig");
const attn = @import("contract_idol_attn.zig");

const g = core.contract.gates;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: idol.Kernels,
    ctl: cuda.Module,
    /// The kernels with one function swapped for its control.
    late_norm: idol.Kernels = undefined,
    once_add: idol.Kernels = undefined,
    late_bias: idol.Kernels = undefined,
    norms: [2]case.Case(norms.Norm) = undefined,
    norm32s: [2]case.Case(norms.Norm32) = undefined,
    adds: case.Case(steps.GateAdd) = undefined,
    biases: [2]case.Case(steps.Bias) = undefined,
    cond: case.Case(steps.Cond) = undefined,
    bias32: case.Case(steps.Bias32) = undefined,
    qk: case.Case(rotary.Qk) = undefined,
    repo: case.Case(rotary.Repo) = undefined,
    embed: case.Case(rotary.Embed) = undefined,
    attn: attn.Attn = undefined,
    ctl_norm: case.Case(norms.Norm) = undefined,
    ctl_add: case.Case(steps.GateAdd) = undefined,
    ctl_bias: case.Case(steps.Bias) = undefined,

    /// The set, its kernels appended to `list` (the controls last).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try idol.Kernels.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k, .ctl = ctl };
        set.late_norm = k;
        set.late_norm.norm = try ctl.function("tf_ctl_idol_norm_late");
        set.once_add = k;
        set.once_add.gate_add = try ctl.function("tf_ctl_idol_gate_add_once");
        set.late_bias = k;
        set.late_bias.bias = try ctl.function("tf_ctl_idol_bias_late");
        const rig: @import("contract_rig.zig").Rig = .{ .gpu = gpu, .s = s };
        const norm_names = [_][]const u8{ "cuda idol tf_idol_norm, LayerNorm32 affine to fp16", "cuda idol tf_idol_norm, LayerNorm32 to fp16, modulated" };
        const norm32_names = [_][]const u8{ "cuda idol tf_idol_norm32, LayerNorm affine in fp32", "cuda idol tf_idol_norm32, LayerNorm times 1 + w in fp32" };
        for (&set.norms, norm_names, 1..) |*x, n, v| x.* = case.make(norms.Norm, rig, k, n, @intCast(v), null);
        for (&set.norm32s, norm32_names, 1..) |*x, n, v| x.* = case.make(norms.Norm32, rig, k, n, @intCast(v), null);
        set.adds = case.make(steps.GateAdd, rig, k, "cuda idol tf_idol_gate_add, x + fp16(g y)", 0, null);
        set.biases[0] = case.make(steps.Bias, rig, k, "cuda idol tf_idol_bias, fp16(acc + b)", 0, null);
        set.biases[1] = case.make(steps.Bias, rig, k, "cuda idol tf_idol_bias, fp16(GELU(fp16(acc + b)))", 1, null);
        set.cond = case.make(steps.Cond, rig, k, "cuda idol tf_idol_cond, the condition projections' sum", 0, null);
        set.bias32 = case.make(steps.Bias32, rig, k, "cuda idol tf_idol_bias32, fp32 bias in place", 0, null);
        set.qk = case.make(rotary.Qk, rig, k, "cuda idol tf_idol_qk, RePo turn, F.normalize, gamma", 0, null);
        set.repo = case.make(rotary.Repo, rig, k, "cuda idol tf_idol_repo, final map and angles", 0, null);
        set.embed = case.make(rotary.Embed, rig, k, "cuda idol tf_idol_embed, input projection and position", 0, null);
        set.attn = .{ .rig = rig, .k = &set.k, .name = "cuda idol tf_idol_attention, full<64, 8>, fp16" };
        set.ctl_norm = case.make(norms.Norm, rig, set.late_norm, "control: idol modulated norm unrounded before scale", 2, .shapes);
        set.ctl_add = case.make(steps.GateAdd, rig, set.once_add, "control: idol gate_add without fp16(g y)", 0, .shapes);
        set.ctl_bias = case.make(steps.Bias, rig, set.late_bias, "control: idol GELU of the unrounded linear", 1, .shapes);
        for (&set.norms) |*x| try list.append(a, x.kernel());
        for (&set.norm32s) |*x| try list.append(a, x.kernel());
        try list.append(a, set.adds.kernel());
        for (&set.biases) |*x| try list.append(a, x.kernel());
        try list.append(a, set.cond.kernel());
        try list.append(a, set.bias32.kernel());
        try list.append(a, set.qk.kernel());
        try list.append(a, set.repo.kernel());
        try list.append(a, set.embed.kernel());
        try list.append(a, set.attn.kernel());
        try list.append(a, set.ctl_norm.kernel());
        try list.append(a, set.ctl_add.kernel());
        try list.append(a, set.ctl_bias.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.norms) |*x| x.deinit();
        for (&set.norm32s) |*x| x.deinit();
        set.adds.deinit();
        for (&set.biases) |*x| x.deinit();
        set.cond.deinit();
        set.bias32.deinit();
        set.qk.deinit();
        set.repo.deinit();
        set.embed.deinit();
        set.attn.deinit();
        set.ctl_norm.deinit();
        set.ctl_add.deinit();
        set.ctl_bias.deinit();
        set.ctl.unload();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// Idol's kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x1d01, &t);
    std.debug.print("SUMMARY idol: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "idol-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
