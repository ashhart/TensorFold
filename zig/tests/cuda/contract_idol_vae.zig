//! Idol's image encoder kernels (idol_vae.cu, fp32) under the contract, each control breaking one gate.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const vae = @import("idol_vae_cuda");
const check = @import("check.zig");
const case = @import("contract_idol_case.zig");
const cv = @import("contract_idol_vae_convs.zig");
const rw = @import("contract_idol_vae_rows.zig");

const g = core.contract.gates;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

const Same = cv.Conv(cv.forms.same);
const Residual = cv.Conv(cv.forms.residual);
const Down = cv.Conv(cv.forms.down);
const One = cv.Conv(cv.forms.one);
const ByCol = rw.Bias(false, false);
const ByColAdd = rw.Bias(false, true);
const ByRow = rw.Bias(true, false);

/// A control: the field of Kernels it swaps for the control's symbol (Set.init names the gate that must catch it).
const Swap = struct { field: []const u8, symbol: [:0]const u8 };
const swaps = [_]Swap{
    .{ .field = "conv_in", .symbol = "tf_ctl_iv_conv_in_half" },
    .{ .field = "conv", .symbol = "tf_ctl_iv_conv_half_in" },
    .{ .field = "conv", .symbol = "tf_ctl_iv_conv_sym_pad" },
    .{ .field = "conv", .symbol = "tf_ctl_iv_conv_half_sum" },
    .{ .field = "norm", .symbol = "tf_ctl_iv_norm_half" },
    .{ .field = "group_part", .symbol = "tf_ctl_iv_group_part_half" },
    .{ .field = "group_finish", .symbol = "tf_ctl_iv_group_finish_half" },
    .{ .field = "softmax", .symbol = "tf_ctl_iv_softmax_half" },
    .{ .field = "bias", .symbol = "tf_ctl_iv_bias_swap" },
    .{ .field = "tokens", .symbol = "tf_ctl_iv_tokens_axes" },
    .{ .field = "tokens", .symbol = "tf_ctl_iv_tokens_clamp" },
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    k: vae.Kernels,
    ctl: cuda.Module,
    /// The kernels with one function swapped for a control, in `swaps`' order.
    bad: [swaps.len]vae.Kernels = undefined,
    conv_in: case.Case(cv.ConvIn) = undefined,
    same: case.Case(Same) = undefined,
    residual: case.Case(Residual) = undefined,
    down: case.Case(Down) = undefined,
    one: case.Case(One) = undefined,
    parts: [2]case.Case(rw.GroupPart) = undefined,
    finishes: [2]case.Case(rw.GroupFinish) = undefined,
    norms: [2]case.Case(rw.Norm) = undefined,
    by_col: case.Case(ByCol) = undefined,
    by_col_add: case.Case(ByColAdd) = undefined,
    by_row: case.Case(ByRow) = undefined,
    softmax: case.Case(rw.Softmax) = undefined,
    tokens: case.Case(rw.Tokens) = undefined,
    ctl_conv_in: case.Case(cv.ConvIn) = undefined,
    ctl_half_in: case.Case(Same) = undefined,
    ctl_sym_pad: case.Case(Down) = undefined,
    ctl_half_sum: case.Case(Same) = undefined,
    ctl_norm: case.Case(rw.Norm) = undefined,
    ctl_part: case.Case(rw.GroupPart) = undefined,
    ctl_finish: case.Case(rw.GroupFinish) = undefined,
    ctl_softmax: case.Case(rw.Softmax) = undefined,
    ctl_bias: case.Case(ByRow) = undefined,
    ctl_axes: case.Case(rw.Tokens) = undefined,
    ctl_clamp: case.Case(rw.Tokens) = undefined,

    /// The set, its kernels appended to `list` (the controls last).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var k = try vae.Kernels.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .k = k, .ctl = ctl };
        inline for (swaps, 0..) |sw, i| {
            set.bad[i] = k;
            @field(set.bad[i], sw.field) = try ctl.function(sw.symbol);
        }
        const rig: @import("contract_rig.zig").Rig = .{ .gpu = gpu, .s = s };
        const n = "cuda idol vae ";
        set.conv_in = case.make(cv.ConvIn, rig, k, n ++ "tf_iv_conv_in, conv3x3(2 x - 1) + b", 0, null);
        set.same = case.make(Same, rig, k, n ++ "tf_iv_conv, 3x3 same size", 0, null);
        set.residual = case.make(Residual, rig, k, n ++ "tf_iv_conv, 3x3 same size + residual", 0, null);
        set.down = case.make(Down, rig, k, n ++ "tf_iv_conv, 3x3 stride 2 padded right and below", 0, null);
        set.one = case.make(One, rig, k, n ++ "tf_iv_conv, 1x1", 0, null);
        const parts = [_][]const u8{ n ++ "tf_iv_group_part, sums", n ++ "tf_iv_group_part, squared deviations" };
        const finishes = [_][]const u8{ n ++ "tf_iv_group_finish, means", n ++ "tf_iv_group_finish, rstd" };
        const norms = [_][]const u8{ n ++ "tf_iv_norm, GroupNorm affine", n ++ "tf_iv_norm, GroupNorm affine then SiLU" };
        for (&set.parts, parts, 0..) |*x, m, v| x.* = case.make(rw.GroupPart, rig, k, m, @intCast(v), null);
        for (&set.finishes, finishes, 0..) |*x, m, v| x.* = case.make(rw.GroupFinish, rig, k, m, @intCast(v), null);
        for (&set.norms, norms, 0..) |*x, m, v| x.* = case.make(rw.Norm, rig, k, m, @intCast(v), null);
        set.by_col = case.make(ByCol, rig, k, n ++ "tf_iv_bias, by column", 0, null);
        set.by_col_add = case.make(ByColAdd, rig, k, n ++ "tf_iv_bias, by column + residual", 0, null);
        set.by_row = case.make(ByRow, rig, k, n ++ "tf_iv_bias, by row (v transposed)", 0, null);
        set.softmax = case.make(rw.Softmax, rig, k, n ++ "tf_iv_softmax, attention rows of 16384", 0, null);
        set.tokens = case.make(rw.Tokens, rig, k, n ++ "tf_iv_tokens, posterior sample to condition rows", 0, null);
        const c = "control: idol vae ";
        set.ctl_conv_in = case.make(cv.ConvIn, rig, set.bad[0], c ++ "conv_in stages 2 x - 1 in half", 0, .shapes);
        set.ctl_half_in = case.make(Same, rig, set.bad[1], c ++ "conv stages its input in half", 0, .shapes);
        set.ctl_sym_pad = case.make(Down, rig, set.bad[2], c ++ "stride-2 conv pads above as padding=1 would", 0, .shapes);
        set.ctl_half_sum = case.make(Same, rig, set.bad[3], c ++ "conv sums in half", 0, .sums);
        set.ctl_norm = case.make(rw.Norm, rig, set.bad[4], c ++ "norm's SiLU in half", 1, .shapes);
        set.ctl_part = case.make(rw.GroupPart, rig, set.bad[5], c ++ "group_part sums in half", 0, .sums);
        set.ctl_finish = case.make(rw.GroupFinish, rig, set.bad[6], c ++ "group_finish sums in half", 0, .sums);
        set.ctl_softmax = case.make(rw.Softmax, rig, set.bad[7], c ++ "softmax's denominator in half", 0, .sums);
        set.ctl_bias = case.make(ByRow, rig, set.bad[8], c ++ "bias by row reads the column's", 0, .shapes);
        set.ctl_axes = case.make(rw.Tokens, rig, set.bad[9], c ++ "tokens swaps a patch's axes", 0, .shapes);
        set.ctl_clamp = case.make(rw.Tokens, rig, set.bad[10], c ++ "tokens clamps a NaN logvar (fminf, fmaxf)", 0, .nonfinite);
        inline for (.{ "conv_in", "same", "residual", "down", "one" }) |f| try list.append(a, @field(set, f).kernel());
        for (&set.parts) |*x| try list.append(a, x.kernel());
        for (&set.finishes) |*x| try list.append(a, x.kernel());
        for (&set.norms) |*x| try list.append(a, x.kernel());
        inline for (.{ "by_col", "by_col_add", "by_row", "softmax", "tokens" }) |f| try list.append(a, @field(set, f).kernel());
        inline for (ctl_fields) |f| try list.append(a, @field(set, f).kernel());
        return set;
    }

    const ctl_fields = .{ "ctl_conv_in", "ctl_half_in", "ctl_sym_pad", "ctl_half_sum", "ctl_norm", "ctl_part", "ctl_finish", "ctl_softmax", "ctl_bias", "ctl_axes", "ctl_clamp" };

    pub fn deinit(set: *Set) void {
        inline for (.{ "conv_in", "same", "residual", "down", "one", "by_col", "by_col_add", "by_row", "softmax", "tokens" } ++ ctl_fields) |f| @field(set, f).deinit();
        for (&set.parts) |*x| x.deinit();
        for (&set.finishes) |*x| x.deinit();
        for (&set.norms) |*x| x.deinit();
        set.ctl.unload();
        set.k.unload();
        set.gpa.destroy(set);
    }
};

/// The image encoder's kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x1d02, &t);
    std.debug.print("SUMMARY idol vae: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "idol-vae-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
