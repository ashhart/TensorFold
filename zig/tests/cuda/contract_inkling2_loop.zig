//! Mini's loop kernels (inkling2/loop.cu) under the contract, each control after the kernel it breaks.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const lp = @import("inkling2").loop_ops;
const ref = @import("contract_inkling2_ops_ref.zig");
const ops = @import("contract_inkling2_loop_ops.zig");
const gate = @import("contract_inkling2_loop_gate.zig");

const G = ref.Gated;
const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};
const groups = .{ "msgs", "downs", "blends", "mixes", "norms", "banks" };

pub const Set = struct {
    k: lp.Kernels = undefined,
    ctl: cuda.Module = undefined, // contract_controls: the blend's twins
    twins: gate.Twins = undefined,
    loaded: bool = false,
    msgs: [3]G(gate.Msg) = undefined,
    downs: [6]G(gate.Down) = undefined,
    blends: [4]G(gate.Blend) = undefined,
    mixes: [3]G(ops.Mix) = undefined,
    norms: [2]G(ops.Norm) = undefined,
    banks: [2]G(ops.Bank) = undefined,

    /// Every registration appended to `list`.
    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        {
            set.k = try lp.Kernels.load(gpu.d);
            errdefer set.k.unload();
            set.ctl = try cuda.Module.load(gpu.d, &controls.bytes);
            errdefer set.ctl.unload();
            set.twins = .{ .chain = try set.ctl.function("tf_ctl_ink2_blend_chain"), .tree = try set.ctl.function("tf_ctl_ink2_blend_tree") };
        }
        const k = &set.k;
        const rig: @import("contract_rig.zig").Rig = .{ .gpu = gpu, .s = s };
        for (&set.msgs, [_]gate.View{ .msg, .pair, .msg }, 0..) |*x, v, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .view = v, .wrong = i == 2 } };
        for (&set.downs, [_]gate.Width{ .o8, .o16, .o64, .o8, .o16, .o64 }, 0..) |*x, w, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .width = w, .swapped = i >= 3 } };
        for (&set.blends, [_]gate.Mode{ .served, .swapped, .order, .tree }) |*x, m| x.* = .{ .rig = rig, .k = .{ .kern = k, .mode = m, .twins = &set.twins } };
        for (&set.mixes, [_]ops.Form{ .attn, .mlp, .mlp }, 0..) |*x, f, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .form = f, .short = i == 2 } };
        for (&set.norms, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .one = i == 1 } };
        for (&set.banks, 0..) |*x, i| x.* = .{ .rig = rig, .k = .{ .kern = k, .wrong = i == 1 } };
        set.loaded = true;
        errdefer set.deinit();
        inline for (groups) |f| for (&@field(set, f)) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(set: *Set) void {
        if (!set.loaded) return;
        inline for (groups) |f| for (&@field(set, f)) |*x| x.deinit();
        set.ctl.unload();
        set.k.unload();
        set.loaded = false;
    }
};
