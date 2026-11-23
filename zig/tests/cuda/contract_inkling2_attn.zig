//! Inkling 2's attention kernels (inkling2/attention.cu, prompt_attention.cu) under the contract, with controls.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const K = @import("inkling2").kernels;
const rows = @import("contract_inkling2_attn_rows.zig");
const prep = @import("contract_inkling2_attn_prep.zig");

/// The attention launches as the forward makes them: prep, the bias table, then the chunks and their merge.
pub const Set = struct {
    k: K.Kernels = undefined,
    loaded: bool = false,
    attns: [7]rows.Attn = undefined,
    preps: [3]prep.Prep = undefined,
    tables: [2]prep.Table = undefined,

    /// Every registration (the fold first: the split and prompt launches must equal its bits), then the controls.
    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.k = try K.Kernels.load(gpu.d);
        const forms = [_]struct { rows.Path, rows.Plant }{ .{ .sliding, .none }, .{ .fold, .none }, .{ .split, .none }, .{ .prompt, .none }, .{ .sliding, .late_bias }, .{ .fold, .no_sink }, .{ .prompt, .q_ulp } };
        for (&set.attns, forms) |*x, f| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &set.k, .path = f[0], .plant = f[1] };
        for (&set.preps, [_]bool{ false, true, true }, 0..) |*x, global, i| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &set.k, .global = global, .tau_off = i == 2 };
        for (&set.tables, 0..) |*x, i| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &set.k, .late = i == 1 };
        set.loaded = true;
        errdefer set.deinit();
        for (set.preps[0..2]) |*x| try list.append(a, x.kernel());
        try list.append(a, set.tables[0].kernel());
        for (set.attns[0..4]) |*x| try list.append(a, x.kernel());
        try list.append(a, set.preps[2].kernel());
        try list.append(a, set.tables[1].kernel());
        for (set.attns[4..]) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(set: *Set) void {
        if (!set.loaded) return;
        for (&set.attns) |*x| x.deinit();
        for (&set.preps) |*x| x.deinit();
        for (&set.tables) |*x| x.deinit();
        set.k.unload();
        set.loaded = false;
    }
};
