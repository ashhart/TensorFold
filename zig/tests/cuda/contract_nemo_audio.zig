//! Transcribe's audio encoder kernels (nemo_audio.cu) under the contract, each with a control (audio-contract).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemo_audio = @import("nemo_audio");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const r = @import("contract_nemo_audio_ref.zig");
const feats = @import("contract_nemo_audio_feats.zig");
const steps = @import("contract_nemo_audio_rows.zig");
const full = @import("contract_nemo_audio_attn.zig");

const g = core.contract.gates;
const K = nemo_audio.kernels.Kernels;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: K,
    ctl: cuda.Module,
    mels: [2]feats.Mel = undefined,
    norms: [2]feats.Norm = undefined,
    qks: [3]steps.Qk = undefined,
    attns: [2]full.Full = undefined,
    copies: [4]steps.Copy = undefined,
    preds: [2]steps.Preds = undefined,
    fuses: [2]steps.Fuse = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try K.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k, .ctl = ctl };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ends: r.Ends = .{ .a = gpu.gpa };
        var w = k;
        w.mel_fn = try ctl.function("tf_ctl_nemo_mel");
        set.mels[0] = .{ .rig = rig, .k = k, .name = "cuda nemo audio tf_nemo_mel", .ends = ends };
        set.mels[1] = .{ .rig = rig, .k = w, .name = "control: nemo audio tf_nemo_mel drops the last bin", .control = .shapes, .ends = ends };
        w = k;
        w.norm_fn = try ctl.function("tf_ctl_nemo_norm");
        set.norms[0] = .{ .rig = rig, .k = k, .name = "cuda nemo audio tf_nemo_norm", .ends = ends };
        set.norms[1] = .{ .rig = rig, .k = w, .name = "control: nemo audio tf_nemo_norm sd over n", .control = .shapes, .ends = ends };
        w = k;
        w.qk_fn = try ctl.function("tf_ctl_nemo_qk");
        set.qks[0] = .{ .rig = rig, .k = k, .normed = true, .name = "cuda nemo audio tf_nemo_qk asr (q/k LayerNorm, 10 x 128)", .ends = ends };
        set.qks[1] = .{ .rig = rig, .k = w, .normed = true, .name = "control: nemo audio tf_nemo_qk drops bf16(a cos)", .control = .shapes, .ends = ends };
        set.qks[2] = .{ .rig = rig, .k = k, .normed = false, .name = "cuda nemo audio tf_nemo_qk speaker (8 x 64)", .ends = ends };
        w = k;
        w.attn_fn = try ctl.function("tf_ctl_nemo_attn64");
        set.attns[0] = .{ .rig = rig, .k = k, .name = "cuda nemo audio tf_nemo_attn64 (full attention)" };
        set.attns[1] = .{ .rig = rig, .k = w, .name = "control: nemo audio tf_nemo_attn64 drops the last key", .control = .shapes };
        w = k;
        w.taps_fn = try ctl.function("tf_ctl_nemo_taps");
        w.relu_fn = try ctl.function("tf_ctl_nemo_relu");
        set.copies[0] = .{ .rig = rig, .k = k, .relu = false, .name = "cuda nemo audio tf_nemo_taps" };
        set.copies[1] = .{ .rig = rig, .k = w, .relu = false, .name = "control: nemo audio tf_nemo_taps a row late", .control = .shapes };
        set.copies[2] = .{ .rig = rig, .k = k, .relu = true, .name = "cuda nemo audio tf_nemo_relu" };
        set.copies[3] = .{ .rig = rig, .k = w, .relu = true, .name = "control: nemo audio tf_nemo_relu drops NaN (fmaxf)", .control = .nonfinite };
        w = k;
        w.preds_fn = try ctl.function("tf_ctl_nemo_preds");
        set.preds[0] = .{ .rig = rig, .k = k, .name = "cuda nemo audio tf_nemo_preds", .ends = ends };
        set.preds[1] = .{ .rig = rig, .k = w, .name = "control: nemo audio tf_nemo_preds drops bf16(sigmoid)", .control = .shapes, .ends = ends };
        w = k;
        w.fuse_fn = try ctl.function("tf_ctl_nemo_fuse");
        set.fuses[0] = .{ .rig = rig, .k = k, .name = "cuda nemo audio tf_nemo_fuse", .ends = ends };
        set.fuses[1] = .{ .rig = rig, .k = w, .name = "control: nemo audio tf_nemo_fuse drops bf16(a K)", .control = .shapes, .ends = ends };
        for (&set.mels) |*x| try list.append(a, x.kernel());
        for (&set.norms) |*x| try list.append(a, x.kernel());
        for (&set.qks) |*x| try list.append(a, x.kernel());
        for (&set.attns) |*x| try list.append(a, x.kernel());
        for (&set.copies) |*x| try list.append(a, x.kernel());
        for (&set.preds) |*x| try list.append(a, x.kernel());
        for (&set.fuses) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.mels) |*x| x.deinit();
        for (&set.norms) |*x| x.deinit();
        for (&set.qks) |*x| x.deinit();
        for (&set.attns) |*x| x.deinit();
        for (&set.copies) |*x| x.deinit();
        for (&set.preds) |*x| x.deinit();
        for (&set.fuses) |*x| x.deinit();
        set.ctl.unload();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// The audio encoder kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6e656d6f, &t);
    std.debug.print("SUMMARY nemo audio: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "audio-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
