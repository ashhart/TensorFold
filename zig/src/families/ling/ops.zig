//! Ling-3.1's launches on the shared modules: grouped sigmoid routing (moe_glue), MLA (cuda.mla), its SwiGLU order.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");

const glue = glm.glue;
const latent = glm.latent;
const mla = cuda.mla;
const Groups = @import("config.zig").Groups;

/// Ling's qk_rope_head_dim: the roped columns after the latent in q and in the cache.
pub const rope_width = mla.rope_width;
pub const RopeRows = mla.RopeRows;
pub const ropeFreqs = mla.ropeFreqs;
pub const chunksSmem = mla.chunksSmem;

/// The shared modules Ling launches beside GLM's glue, and its SwiGLU order (the ling_act image).
pub const Kernels = struct {
    mla: mla.Mla,
    moe: cuda.moe_glue.MoeGlue,
    act: cuda.Module,
    swiglu: [2]cuda.Function, // 512 and 64 blocks, min(silu(g), L) * clip(u, -L, L)

    pub fn load(d: *const cuda.Driver) !Kernels {
        var k: Kernels = undefined;
        k.mla = try mla.Mla.load(d);
        errdefer k.mla.unload();
        k.moe = try cuda.moe_glue.MoeGlue.load(d);
        errdefer k.moe.unload();
        k.act = try cuda.Module.load(d, cuda.kernels.ling_act);
        errdefer k.act.unload();
        k.swiglu = .{ try k.act.function("tf_ling_swiglu_512"), try k.act.function("tf_ling_swiglu_64") };
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.act.unload();
        k.moe.unload();
        k.mla.unload();
    }
};

fn parts(p: latent.Partials) mla.Partials {
    return .{ .po = p.po, .pm = p.pm, .pl = p.pl };
}

/// The kernel's clamp for a limit: 0 (unclamped, as the experts' 0) as +inf, whose min and clip are exact.
pub fn clampOf(limit: f32) f32 {
    return if (limit == 0) std.math.inf(f32) else limit;
}

/// glue.swiglu in Ling's order: [gate | up] -> bf16(min(bf16(silu(g)), L) * clip(u, -L, L)), 64-sums in `xs`; 0: none.
pub fn swiglu(k: *const Kernels, stream: cuda.Stream, gu: u64, out: u64, xs: u64, limit: f32, rows: usize, w: usize) !void {
    const block: usize = if (w % 512 == 0) 512 else if (w % 64 == 0) 64 else return error.NoSwigluBlock;
    const f, const threads: u32 = if (block == 512) .{ k.swiglu[0], 128 } else .{ k.swiglu[1], 64 };
    var a: cuda.Args = .{};
    inline for (.{ gu, out, xs, clampOf(limit), @as(i64, @intCast(w)) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(w / block) }, .block = .{ .x = threads } }, stream, &a);
}

/// Group-limited routing on moe_glue's score form: top_k by sigmoid + bias in kept groups, normed, x scale, + shared.
pub fn select(k: *const Kernels, stream: cuda.Stream, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, top_k: usize, experts: usize, scale: f32, norm: bool, groups: Groups) !void {
    const r: cuda.moe_glue.Route = .{ .experts = experts, .topk = top_k, .shared = true, .normalize = norm, .scale = scale, .score = true, .groups = groups.n, .keep = groups.keep };
    try k.moe.topk(stream, logits, bias, r, pick, wts, rows);
}

/// The absorb: q_nope the first d = 128 of each head's `q_stride` columns, out rows `out_stride` apart per head.
pub fn absorbQ(k: *const Kernels, stream: cuda.Stream, q: u64, q_stride: usize, wk: u64, out: u64, out_stride: usize, rows: usize, heads: usize, d: usize, lw: usize) !void {
    try k.mla.absorbQ(stream, q, q_stride, wk, out, out_stride, rows, heads, d, lw);
}

/// latent.expand_v at 128-wide v heads: out [rows][heads][dv] bf16 = o_lat [rows][heads][lw] . wv [heads][dv][lw].
pub fn expandV(k: *const Kernels, stream: cuda.Stream, ol: u64, wv: u64, out: u64, rows: usize, heads: usize, dv: usize, lw: usize) !void {
    try k.mla.expandV(stream, ol, wv, out, rows, heads, dv, lw);
}

/// The GPT-J rope at device i32 pos + r, fp32 inv_freq [32], cos and sin times `scale` (YaRN's; 1 keeps plain bits).
pub fn rope(k: *const Kernels, stream: cuda.Stream, v: RopeRows, rows: usize, heads: usize, pos: u64, inv_freq: u64, scale: f32) !void {
    try k.mla.rope(stream, v, rows, heads, pos, inv_freq, scale);
}

/// latent.attention with roped keys: the shared chunk tiles (16-head blocks), then GLM's merge; out lw wide.
pub fn attention(k: *const Kernels, set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, pos: u64, p: latent.Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32) !void {
    if (lw != mla.latent_width) return error.LatentShape;
    try k.mla.chunks(stream, qa, cache, pos, parts(p), rows, heads, nch, scale);
    try latent.merge(set, stream, p, out, dummy, rows, heads, lw, nch, false);
}

/// The chunk tiles over a split cache: this rank's `nch` chunks (chunk z * world + rank of the positions) into `p`.
pub fn chunksSplit(k: *const Kernels, stream: cuda.Stream, qa: u64, cache: u64, pos: u64, p: latent.Partials, rows: usize, heads: usize, nch: usize, scale: f32, world: usize, rank: usize) !void {
    try k.mla.chunksSplit(stream, qa, cache, pos, parts(p), rows, heads, nch, scale, world, rank);
}

/// `nch` chunks' partials merged into one chunk's (o, m, l in fp32) at `out`: nch 0 writes the empty one.
pub fn mergePart(k: *const Kernels, stream: cuda.Stream, p: latent.Partials, out: latent.Partials, rows: usize, heads: usize, nch: usize) !void {
    try k.mla.mergePart(stream, parts(p), parts(out), rows, heads, nch);
}

test "the roped chunk tiles fit the 99 KiB a GB10 block may take, on GLM's row blocks and head tiles" {
    try std.testing.expect(chunksSmem(16) <= 99 * 1024);
    try std.testing.expectEqual(@as(f32, 1.0), ropeFreqs(6e6)[0]);
    try std.testing.expect(mla.heads_tile == latent.heads_tile and mla.prompt_rows == latent.prompt_rows and mla.prompt_rb == latent.prompt_rb);
}
