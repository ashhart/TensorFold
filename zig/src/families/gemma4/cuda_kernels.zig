//! Gemma 4's launches on CUDA, every one a shared operator (#1048): projections, rows, attention, argmax, draws.
const std = @import("std");
const cuda = @import("cuda");

pub const Kernels = struct {
    /// qmm_group, qmm_prefill, lane_gemv, affine4_pack, sample, tree_round, affine8, affine_prefill (other widths).
    mods: [8]cuda.Module,
    affine: cuda.qlinear.Affine4Kernels,
    wide: cuda.qlinear.AffineNKernels,
    draws: cuda.sample.Draws,
    round: cuda.tree_round.Kernels,
    embed: cuda.embed_rows.EmbedRows,
    norms: cuda.norms.Norms,
    qk: cuda.qk_rows.QkRows,
    win: cuda.window_attention.WindowAttention,
    act: cuda.act_rows.ActRows,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, kk.sample, kk.tree_round, kk.affine8, kk.affine_prefill };
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        const gb10 = try ctx.capability() == 121;
        k.affine = try cuda.qlinear.Affine4Kernels.resolve(d, k.mods[0], k.mods[1], k.mods[2], k.mods[3], sms, gb10);
        errdefer k.affine.deinit();
        k.wide = try cuda.qlinear.AffineNKernels.resolve(d, k.mods[6], k.mods[7], k.mods[3], sms, gb10);
        errdefer k.wide.deinit();
        k.draws = try cuda.sample.Draws.resolve(k.mods[4]);
        k.round = try cuda.tree_round.Kernels.resolve(k.mods[5]);
        k.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer k.embed.unload();
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer k.qk.unload();
        k.win = try cuda.window_attention.WindowAttention.load(d);
        errdefer k.win.unload();
        k.act = try cuda.act_rows.ActRows.load(d);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.act.unload();
        k.win.unload();
        k.qk.unload();
        k.norms.unload();
        k.embed.unload();
        k.wide.deinit();
        k.affine.deinit();
        for (&k.mods) |*m| m.unload();
    }
};
