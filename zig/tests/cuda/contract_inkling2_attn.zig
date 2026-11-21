//! Inkling 2's attention kernels (inkling2/attention.cu, prompt_attention.cu) under the contract, with controls.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const K = @import("inkling2").kernels;
const rows = @import("contract_inkling2_attn_rows.zig");
const prep = @import("contract_inkling2_attn_prep.zig");
const M = @import("contract_inkling2_model.zig");

/// The attention launches as the forward makes them (prep, bias table, chunks and merge): Small's, then Mini's.
pub const Set = struct {
    k: K.Kernels = undefined,
    loaded: bool = false,
    attns: [14]rows.Attn = undefined,
    preps: [6]prep.Prep = undefined,
    tables: [2]prep.Table = undefined, // Mini's bias tables have Small's shapes: one registration covers both

    /// Every registration (the fold first: the split and prompt launches must equal its bits), then the controls.
    pub fn init(set: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        set.k = try K.Kernels.load(gpu.d);
        const forms = [_]struct { rows.Path, rows.Plant }{ .{ .sliding, .none }, .{ .fold, .none }, .{ .split, .none }, .{ .prompt, .none }, .{ .sliding, .late_bias }, .{ .fold, .no_sink }, .{ .prompt, .q_ulp } };
        for (&set.attns, 0..) |*x, i| {
            const f = forms[i % 7];
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &set.k, .path = f[0], .plant = f[1], .m = if (i < 7) M.small else M.mini };
        }
        for (&set.preps, 0..) |*x, i| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &set.k, .global = i % 3 != 0, .tau_off = i % 3 == 2, .m = if (i < 3) M.small else M.mini };
        for (&set.tables, 0..) |*x, i| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &set.k, .late = i == 1 };
        set.loaded = true;
        errdefer set.deinit();
        for (set.preps[0..2]) |*x| try list.append(a, x.kernel());
        try list.append(a, set.tables[0].kernel());
        for (set.attns[0..4]) |*x| try list.append(a, x.kernel());
        try list.append(a, set.preps[2].kernel());
        try list.append(a, set.tables[1].kernel());
        for (set.attns[4..7]) |*x| try list.append(a, x.kernel());
        for (set.preps[3..5]) |*x| try list.append(a, x.kernel());
        for (set.attns[7..11]) |*x| try list.append(a, x.kernel());
        try list.append(a, set.preps[5].kernel());
        for (set.attns[11..]) |*x| try list.append(a, x.kernel());
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
