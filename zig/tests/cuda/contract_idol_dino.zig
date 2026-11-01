//! idol_dino.cu under the contract, controls dropping or adding a bf16 rounding or summing in fp16.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const dino = @import("idol_dino_cuda");
const check = @import("check.zig");
const case = @import("contract_idol_case.zig");
const ops = @import("contract_idol_dino_ops.zig");
const norms = @import("contract_idol_dino_norms.zig");
const attn = @import("contract_idol_dino_attn.zig");

const g = core.contract.gates;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: dino.Kernels,
    ctl: cuda.Module,
    /// The kernels with one function swapped for its control.
    swapped: [6]dino.Kernels = undefined,
    bias: case.Case(ops.Bias) = undefined,
    norm: case.Case(norms.Norm) = undefined,
    final: case.Case(norms.Final) = undefined,
    hidden: case.Case(norms.FinalHidden) = undefined,
    heads: case.Case(ops.Heads) = undefined,
    scale_add: case.Case(ops.ScaleAdd) = undefined,
    swiglu: case.Case(ops.Swiglu) = undefined,
    attn: attn.Attn = undefined,
    ctl_bias: case.Case(ops.Bias) = undefined,
    ctl_norm: case.Case(norms.Norm) = undefined,
    ctl_final: case.Case(norms.Final) = undefined,
    ctl_heads: case.Case(ops.Heads) = undefined,
    ctl_add: case.Case(ops.ScaleAdd) = undefined,
    ctl_swiglu: case.Case(ops.Swiglu) = undefined,

    /// The planted controls: the function each replaces, its symbol in contract_controls, the gate it must fail.
    const planted = [_]struct { []const u8, [:0]const u8, g.Gate }{
        .{ "bias", "tf_ctl_idol_dino_bias_twice", .shapes },
        .{ "norm", "tf_ctl_idol_dino_norm_half", .sums },
        .{ "final", "tf_ctl_idol_dino_final_half", .sums },
        .{ "heads", "tf_ctl_idol_dino_heads_once", .shapes },
        .{ "scale_add", "tf_ctl_idol_dino_scale_add_once", .shapes },
        .{ "swiglu", "tf_ctl_idol_dino_swiglu_late", .shapes },
    };

    /// The set, its kernels appended to `list` (the controls last).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try dino.Kernels.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k, .ctl = ctl };
        inline for (planted, &set.swapped) |p, *x| {
            x.* = k;
            @field(x, p[0]) = try ctl.function(p[1]);
        }
        const rig: @import("contract_rig.zig").Rig = .{ .gpu = gpu, .s = s };
        const sw = &set.swapped;
        set.bias = case.make(ops.Bias, rig, k, "cuda idol dino tf_idol_dino_bias, bf16(acc + b)", 0, null);
        set.norm = case.make(norms.Norm, rig, k, "cuda idol dino tf_idol_dino_norm, LayerNorm to bf16", 0, null);
        set.final = case.make(norms.Final, rig, k, "cuda idol dino tf_idol_dino_final, F.layer_norm in fp32", 0, null);
        set.hidden = case.make(norms.FinalHidden, rig, k, "cuda idol dino tf_idol_dino_final, last norm to bf16", 0, null);
        set.heads = case.make(ops.Heads, rig, k, "cuda idol dino tf_idol_dino_heads, bf16 RoPE past the prefix", 0, null);
        set.scale_add = case.make(ops.ScaleAdd, rig, k, "cuda idol dino tf_idol_dino_scale_add, bias, LayerScale, residual", 0, null);
        set.swiglu = case.make(ops.Swiglu, rig, k, "cuda idol dino tf_idol_dino_swiglu, biases and bf16(bf16(silu(g)) u)", 0, null);
        set.attn = .{ .rig = rig, .k = &set.k, .name = "cuda idol dino tf_idol_dino_attention, full<64, 8>, bf16" };
        set.ctl_bias = case.make(ops.Bias, rig, sw[0], "control: idol dino bias after a bf16 rounding of the sums", 0, planted[0][2]);
        set.ctl_norm = case.make(norms.Norm, rig, sw[1], "control: idol dino LayerNorm with fp16 sums", 0, planted[1][2]);
        set.ctl_final = case.make(norms.Final, rig, sw[2], "control: idol dino F.layer_norm with fp16 sums", 0, planted[2][2]);
        set.ctl_heads = case.make(ops.Heads, rig, sw[3], "control: idol dino RoPE without bf16 products", 0, planted[3][2]);
        set.ctl_add = case.make(ops.ScaleAdd, rig, sw[4], "control: idol dino LayerScale without bf16(g y)", 0, planted[4][2]);
        set.ctl_swiglu = case.make(ops.Swiglu, rig, sw[5], "control: idol dino SwiGLU without bf16(silu(g))", 0, planted[5][2]);
        try list.append(a, set.bias.kernel());
        try list.append(a, set.norm.kernel());
        try list.append(a, set.final.kernel());
        try list.append(a, set.hidden.kernel());
        try list.append(a, set.heads.kernel());
        try list.append(a, set.scale_add.kernel());
        try list.append(a, set.swiglu.kernel());
        try list.append(a, set.attn.kernel());
        try list.append(a, set.ctl_bias.kernel());
        try list.append(a, set.ctl_norm.kernel());
        try list.append(a, set.ctl_final.kernel());
        try list.append(a, set.ctl_heads.kernel());
        try list.append(a, set.ctl_add.kernel());
        try list.append(a, set.ctl_swiglu.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        set.bias.deinit();
        set.norm.deinit();
        set.final.deinit();
        set.hidden.deinit();
        set.heads.deinit();
        set.scale_add.deinit();
        set.swiglu.deinit();
        set.attn.deinit();
        set.ctl_bias.deinit();
        set.ctl_norm.deinit();
        set.ctl_final.deinit();
        set.ctl_heads.deinit();
        set.ctl_add.deinit();
        set.ctl_swiglu.deinit();
        set.ctl.unload();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// Idol's DINOv3 kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0xd1, &t);
    std.debug.print("SUMMARY idol dino: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "idol-dino-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
