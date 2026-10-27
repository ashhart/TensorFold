//! The runner's no-sums row kernels as one set: embed_rows' affine rows and sample.cu's penalties, with their images.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const embed_k = @import("contract_embed.zig");
const penalize_k = @import("contract_penalize.zig");
const lfm2m_k = @import("contract_lfm2_moe.zig");

const g = core.contract.gates;

pub const Set = struct {
    embed_rows: ?cuda.embed_rows.EmbedRows = null,
    embeds: [embed_k.entries.len]embed_k.Embed = undefined,
    n_embeds: usize = 0,
    sample: ?cuda.Module = null,
    draws: cuda.sample.Draws = undefined,
    penalize: ?penalize_k.Penalize = null,
    lfm2m: lfm2m_k.Set = .{}, // LFM2.5-24B-A2B's router, Liquid's 3-tap conv and commit (rows with sums, beside these)

    /// Every registration onto `kernels`; the embed controls' symbols come from the contract_controls image `ctl_mod`.
    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, ctl_mod: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        const a = gpu.gpa;
        self.embed_rows = try cuda.embed_rows.EmbedRows.load(gpu.d);
        for (embed_k.entries, &self.embeds) |e, *x| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .e = &self.embed_rows.?, .ctl = if (e.keep != null) try ctl_mod.function("tf_ctl_embed_affine") else null };
            self.n_embeds += 1;
        }
        for (&self.embeds) |*x| try kernels.append(a, x.kernel());
        self.sample = try cuda.Module.load(gpu.d, cuda.kernels.sample);
        self.draws = try cuda.sample.Draws.resolve(self.sample.?);
        self.penalize = .{ .rig = .{ .gpu = gpu, .s = s }, .draws = &self.draws };
        try kernels.append(a, self.penalize.?.kernel());
        try self.lfm2m.add(gpu, s, ctl_mod, kernels);
    }

    pub fn deinit(self: *Set) void {
        self.lfm2m.deinit();
        if (self.penalize) |*p| p.deinit();
        if (self.sample) |*m| m.unload();
        for (self.embeds[0..self.n_embeds]) |*x| x.deinit();
        if (self.embed_rows) |*e| e.unload();
    }
};
