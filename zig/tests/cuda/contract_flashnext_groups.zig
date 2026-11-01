//! The runner's Flash Next prompt groups as one set: the full layer's glue and the DeltaNet twins on each rank's heads.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const glue_k = @import("contract_flashnext_glue.zig");
const heads_k = @import("contract_flashnext_heads.zig");
const Columns = @import("contract_flashnext_columns.zig").Columns;
const lay = @import("flashnext_gdn_heads.zig");

const g = core.contract.gates;
const parts = flash.batch_state;
const tk = flash.tile_kernels;

pub const Set = struct {
    glue: ?flash.Glue = null,
    fns: glue_k.Fns = undefined,
    glues: [std.enums.values(glue_k.Kind).len]glue_k.Glue = undefined,
    n_glues: usize = 0,
    twins: lay.Fns = undefined,
    ranks: [std.enums.values(heads_k.Kind).len * 2]heads_k.Rank = undefined,
    n_ranks: usize = 0,
    columns: [4]Columns = undefined,
    n_columns: usize = 0,

    /// Every registration onto `kernels`, from the flashnext_batch, flashnext_attn and flashnext_hcf images.
    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, batch: cuda.Module, attn: cuda.Module, hcf: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        const a = gpu.gpa;
        self.glue = try flash.Glue.load(gpu.d);
        self.fns = .{
            .tpr = try batch.function("fn_gdn_chain_tpr"),
            .rows_to = try batch.function("fn_gdn_rows_to"),
            .tile = try attn.function("_ZN8qsa_tile6scoresILi8ELi4EEEvPK13__nv_bfloat16S3_PKiPfiiii"),
            .scores = try attn.function("fn_qsa_scores"),
            .hcf_down = try hcf.function(tk.cubin_hcf_down),
            .hcf_up = try hcf.function(tk.cubin_hcf_up),
            .front = try batch.function("fn_gdn_front"),
            .norm = try batch.function("fn_gdn_norm"),
            .fused = try batch.function("fn_gdn_fused"),
        };
        try self.fns.fused.allowDynamicShared(parts.gdn_fused_smem);
        try self.fns.tpr.allowDynamicShared(parts.chain_tpr_smem);
        try self.fns.hcf_down.allowDynamicShared(tk.hcf_down_smem);
        try self.fns.hcf_up.allowDynamicShared(tk.hcf_up_smem);
        for (std.enums.values(glue_k.Kind), &self.glues) |kind, *m| {
            m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .f = &self.fns, .glue = &self.glue.? };
            self.n_glues += 1;
        }
        for (&self.glues) |*m| try kernels.append(a, m.kernel());
        self.twins = try lay.Fns.load(batch);
        for (std.enums.values(heads_k.Kind), 0..) |kind, i| for (0..2) |rank| {
            self.ranks[i * 2 + rank] = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .rank = rank, .f = &self.twins };
            self.n_ranks += 1;
        };
        for (&self.ranks) |*m| try kernels.append(a, m.kernel());
        for (&self.columns, 0..) |*m, i| {
            m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try batch.function("fn_copy_cols"), .slot = i % 2 == 1, .control = i >= 2 };
            self.n_columns += 1;
            try kernels.append(a, m.kernel());
        }
    }

    pub fn deinit(self: *Set) void {
        for (self.columns[0..self.n_columns]) |*m| m.deinit();
        for (self.ranks[0..self.n_ranks]) |*m| m.deinit();
        for (self.glues[0..self.n_glues]) |*m| m.deinit();
        if (self.glue) |*x| x.deinit();
    }
};
