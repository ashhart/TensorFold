//! The runner's no-sums row kernels as one set: embed_rows' affine rows and sample.cu's penalties, with their images.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const embed_k = @import("contract_embed.zig");
const penalize_k = @import("contract_penalize.zig");
const lfm2m_k = @import("contract_lfm2_moe.zig");
const paged_k = @import("contract_paged_attn.zig");
const dots_k = @import("contract_dots_rope.zig");
const mhc_sums_k = @import("contract_mhc_sums.zig");
const mhc_k = @import("contract_mhc.zig");
const vision_k = @import("contract_dots_vision.zig");
const lfm3_k = @import("contract_lfm3.zig");

const g = core.contract.gates;

pub const Set = struct {
    embed_rows: ?cuda.embed_rows.EmbedRows = null,
    affine4_pack: ?cuda.Module = null, // the tile entries' packers
    embeds: [embed_k.entries.len]embed_k.Embed = undefined,
    n_embeds: usize = 0,
    sample: ?cuda.Module = null,
    draws: cuda.sample.Draws = undefined,
    penalize: ?penalize_k.Penalize = null,
    lfm2m: lfm2m_k.Set = .{}, // LFM2.5-24B-A2B's router, Liquid's 3-tap conv and commit (rows with sums, beside these)
    paged: paged_k.Set = .{}, // paged attention: Liquid's head 64 decode and prompt rows, LFM3's head 128 prompt rows
    dots: dots_k.Set = .{}, // dots.mocr's q|k|v bias and Torch-order RoPE
    mhc: mhc_k.Set = .{},
    mhc_sums: mhc_sums_k.Set = .{},
    vision: vision_k.Set = .{}, // dots.mocr's vision rotary and patch bias
    lfm3: lfm3_k.Set = .{}, // LFM3's 7-tap conv and commit

    /// Every registration onto `kernels`; the embed controls' symbols come from the contract_controls image `ctl_mod`.
    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, ctl_mod: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        const a = gpu.gpa;
        self.embed_rows = try cuda.embed_rows.EmbedRows.load(gpu.d);
        self.affine4_pack = try cuda.Module.load(gpu.d, cuda.kernels.affine4_pack);
        const pm = self.affine4_pack.?;
        const pack: embed_k.Pack = .{ .words = try pm.function("tf_pack_dense"), .scales = try pm.function("tf_transpose_pad16") };
        for (embed_k.entries, &self.embeds) |e, *x| {
            const ctl = if (e.keep == null) null else try ctl_mod.function(if (e.tiles) "tf_ctl_embed_tiles4" else "tf_ctl_embed_affine");
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .e = &self.embed_rows.?, .ctl = ctl, .pack = if (e.tiles) pack else null };
            self.n_embeds += 1;
        }
        for (&self.embeds) |*x| try kernels.append(a, x.kernel());
        self.sample = try cuda.Module.load(gpu.d, cuda.kernels.sample);
        self.draws = try cuda.sample.Draws.resolve(self.sample.?);
        self.penalize = .{ .rig = .{ .gpu = gpu, .s = s }, .draws = &self.draws };
        try kernels.append(a, self.penalize.?.kernel());
        try self.lfm2m.add(gpu, s, ctl_mod, kernels);
        try self.paged.add(gpu, s, ctl_mod, kernels);
        try self.dots.add(gpu, s, ctl_mod, kernels);
        try self.vision.add(gpu, s, ctl_mod, kernels);
        try self.lfm3.add(gpu, s, ctl_mod, kernels);
        try self.mhc.add(gpu, s, ctl_mod, kernels);
        try self.mhc_sums.add(gpu, s, kernels);
    }

    pub fn deinit(self: *Set) void {
        self.mhc_sums.deinit();
        self.mhc.deinit();
        self.lfm3.deinit();
        self.vision.deinit();
        self.dots.deinit();
        self.paged.deinit();
        self.lfm2m.deinit();
        if (self.penalize) |*p| p.deinit();
        if (self.sample) |*m| m.unload();
        for (self.embeds[0..self.n_embeds]) |*x| x.deinit();
        if (self.affine4_pack) |*m| m.unload();
        if (self.embed_rows) |*e| e.unload();
    }
};
