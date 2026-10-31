//! MOSS-Image's CUDA kernels under the contract (moss-contract): its glue with controls, and attention at head dim 128.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const moss = @import("moss_image_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const norms = @import("contract_moss_image_norms.zig");
const steps = @import("contract_moss_image_steps.zig");
const full = @import("contract_moss_image_attn.zig");

const g = core.contract.gates;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: moss.Kernels,
    ctl: cuda.Module,
    att: cuda.vit.Attention,
    norms: [4]norms.Norm = undefined,
    qks: [3]norms.Qk = undefined,
    situs: [3]steps.Situ = undefined,
    adds: [2]steps.GateAdd = undefined,
    full: full.Full = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try moss.Kernels.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k, .ctl = ctl, .att = try cuda.vit.Attention.load(gpu.d) };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: @import("contract_moss_image_ref.zig").Ends = .{ .a = gpu.gpa };
        var with = k;
        with.norm = try ctl.function("tf_ctl_moss_norm_inv");
        set.norms[0] = .{ .rig = rig, .k = k, .modulated = false, .name = "cuda moss image tf_moss_norm", .ends = ends };
        set.norms[1] = .{ .rig = rig, .k = with, .modulated = false, .name = "control: moss image tf_moss_norm drops bf16(x inv)", .control = .shapes, .ends = ends };
        with.norm = try ctl.function("tf_ctl_moss_norm_scale");
        set.norms[2] = .{ .rig = rig, .k = k, .modulated = true, .name = "cuda moss image tf_moss_norm modulated", .ends = ends };
        set.norms[3] = .{ .rig = rig, .k = with, .modulated = true, .name = "control: moss image tf_moss_norm modulated drops bf16(y s)", .control = .shapes, .ends = ends };
        with = k;
        with.qk = try ctl.function("tf_ctl_moss_qk_cos");
        set.qks[0] = .{ .rig = rig, .k = k, .packed_rows = false, .name = "cuda moss image tf_moss_qk q k v rows", .ends = ends };
        set.qks[1] = .{ .rig = rig, .k = with, .packed_rows = false, .name = "control: moss image tf_moss_qk drops bf16(q cos)", .control = .shapes, .ends = ends };
        set.qks[2] = .{ .rig = rig, .k = k, .packed_rows = true, .name = "cuda moss image tf_moss_qk packed qkv rows", .ends = ends };
        with = k;
        with.situ = try ctl.function("tf_ctl_moss_situ_sigmoid");
        set.situs[0] = .{ .rig = rig, .k = k, .halves = true, .name = "cuda moss image tf_moss_situ hidden | gate rows", .ends = ends };
        set.situs[1] = .{ .rig = rig, .k = with, .halves = true, .name = "control: moss image tf_moss_situ drops bf16(sigmoid(g))", .control = .shapes, .ends = ends };
        set.situs[2] = .{ .rig = rig, .k = k, .halves = false, .name = "cuda moss image tf_moss_situ two inputs", .ends = ends };
        with = k;
        with.gate_add = try ctl.function("tf_ctl_moss_gate_add_product");
        set.adds[0] = .{ .rig = rig, .k = k, .name = "cuda moss image tf_moss_gate_add", .ends = ends };
        set.adds[1] = .{ .rig = rig, .k = with, .name = "control: moss image tf_moss_gate_add drops bf16(g y)", .control = .shapes, .ends = ends };
        set.full = .{ .rig = rig, .att = &set.att };
        for (&set.norms) |*x| try list.append(a, x.kernel());
        for (&set.qks) |*x| try list.append(a, x.kernel());
        for (&set.situs) |*x| try list.append(a, x.kernel());
        for (&set.adds) |*x| try list.append(a, x.kernel());
        try list.append(a, set.full.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.norms) |*x| x.deinit();
        for (&set.qks) |*x| x.deinit();
        for (&set.situs) |*x| x.deinit();
        for (&set.adds) |*x| x.deinit();
        set.full.deinit();
        set.att.unload();
        set.ctl.unload();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// The MOSS-Image kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6d6f7373, &t);
    std.debug.print("SUMMARY moss image: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "moss-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
