//! Kolibri-1's launches, every one a shared CUDA operator: lane FP8, grouped experts, norms, routing, attention.
const std = @import("std");
const cuda = @import("cuda");

const grouped = cuda.grouped;
const experts = cuda.experts;

/// The torch-op images' functions this family calls (argmax by torch's first-maximum rule, bf16 casts).
const torch_argmax = "tf_argmax_rows_i32_kernel";
const torch_to_f32 = "tf_bf16_to_f32_kernel";
const torch_to_bf16 = "tf_f32_to_bf16_kernel";

pub const Kernels = struct {
    lane: cuda.qmmf.Lane,
    fp8x: experts.Fp8Experts,
    router: grouped.Router,
    mods: [4]cuda.Module, // experts (the plan kernels), sample, torch_argmax, torch_pointwise
    argmax_fn: cuda.Function,
    to_f32_fn: cuda.Function,
    to_bf16_fn: cuda.Function,
    draws: cuda.sample.Draws,
    embed: cuda.embed_rows.EmbedRows,
    norms: cuda.norms.Norms,
    moe: cuda.moe_glue.MoeGlue,
    dense: cuda.dense_rows.DenseRows,
    qk: cuda.qk_rows.QkRows,
    win: cuda.window_attention.WindowAttention,
    tree: cuda.tree_attention.TreeAttention,

    /// Every module is a shared image built into the runtime: nothing captured, nothing of the family's own.
    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.experts, kk.sample, kk.torch_argmax, kk.torch_pointwise };
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        k.router = try grouped.Router.resolve(k.mods[0]);
        k.draws = try cuda.sample.Draws.resolve(k.mods[1]);
        k.argmax_fn = try k.mods[2].function(torch_argmax);
        k.to_f32_fn = try k.mods[3].function(torch_to_f32);
        k.to_bf16_fn = try k.mods[3].function(torch_to_bf16);
        k.lane = try cuda.qmmf.Lane.load(d, (try ctx.capability()) / 10);
        errdefer k.lane.unload();
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        k.fp8x = try experts.Fp8Experts.load(d, sms);
        errdefer k.fp8x.unload();
        k.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer k.embed.unload();
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.moe = try cuda.moe_glue.MoeGlue.load(d);
        errdefer k.moe.unload();
        k.dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer k.dense.unload();
        k.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer k.qk.unload();
        k.win = try cuda.window_attention.WindowAttention.load(d);
        errdefer k.win.unload();
        k.tree = try cuda.tree_attention.TreeAttention.load(d);
        k.tree.sms = sms;
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.embed.unload();
        k.norms.unload();
        k.moe.unload();
        k.dense.unload();
        k.qk.unload();
        k.win.unload();
        k.tree.unload();
        k.fp8x.unload();
        k.lane.unload();
        for (&k.mods) |*m| m.unload();
    }
};

/// launch_shape.h's block count: one block a 256 elements, at most 65535.
fn blocks(count: usize) u32 {
    return @intCast(@max(1, @min((count + 255) / 256, 65535)));
}

/// Launches on one stream.
pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    fn one(o: Ops, f: cuda.Function, grid: u32, block: u32, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid }, .block = .{ .x = block } }, o.s, a);
    }

    /// The fp32 residual of `rows` tokens: their bf16 table rows into `tmp`, widened exactly into `out`.
    pub fn embed(o: Ops, table: u64, ids: u64, rows: usize, d: usize, tmp: u64, out: u64) !void {
        try o.k.embed.bf16(o.s, ids, table, tmp, rows, d);
        var a: cuda.Args = .{};
        a.add(tmp);
        a.add(out);
        a.add(@as(u64, rows * d));
        try o.one(o.k.to_f32_fn, blocks(rows * d), 256, &a);
    }

    /// Each row's argmax over `v` fp32 logits, the first maximum on ties (torch.argmax), into `out` (int32 a row).
    pub fn argmax(o: Ops, logits: u64, rows: usize, v: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        for ([_]usize{ rows, v, v }) |x| a.add(@as(i64, @intCast(x)));
        a.add(@as(i32, 1)); // dtype 1: fp32
        try o.one(o.k.argmax_fn, @intCast(rows), 1024, &a);
    }

    /// bf16 copies of `n` fp32 logits (round to nearest even), as sample.cu's draws read them.
    pub fn toBf16(o: Ops, x: u64, y: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(u64, n));
        try o.one(o.k.to_bf16_fn, blocks(n), 256, &a);
    }

    /// RMS norm of fp32 rows, bf16 out.
    pub fn rms(o: Ops, x: u64, w: u64, out: u64, rows: usize, n: usize, eps: f32) !void {
        try o.k.norms.rms(o.s, x, .f32, w, out, .bf16, rows, n, eps);
    }

    /// res += rms(x) * w1 in place (fp32); out = bf16(rms(res) * w2).
    pub fn addRms(o: Ops, x: u64, res: u64, w1: u64, w2: u64, out: u64, rows: usize, n: usize, eps: f32) !void {
        try o.k.norms.addRms(o.s, x, .bf16, res, w1, w2, out, .bf16, rows, n, eps);
    }

    pub const Qkv = struct { rows: u64, norms: u64, cos: u64, sin: u64, pos: u64, slots: u64, q: u64, k: u64, v: u64, kc: u64, vc: u64, len: usize };

    /// q and k RMS-normed (RoPE'd on sliding layers), v as is; k and v also into the cache at pos % len.
    pub fn qkv(o: Ops, a: Qkv, n: usize, h: usize, hk: usize, d: usize, eps: f32, rope: bool) !void {
        const shape: cuda.qk_rows.Shape = .{ .norms = a.norms, .cos = if (rope) a.cos else null, .sin = if (rope) a.sin else null, .rot = if (rope) d else 0, .eps = eps };
        try o.k.qk.rows(o.s, a.rows, .{ .h = h, .hk = hk, .dim = d }, shape, a.pos, a.slots, .{ .q = a.q, .k = a.k, .v = a.v, .kc = a.kc, .vc = a.vc, .len = a.len }, n);
    }

    /// Decode rows of a sliding layer: each row over its stream's ring, keys (pos - window, pos].
    pub fn sliding(o: Ops, q: u64, kc: u64, vc: u64, out: u64, pos: u64, slots: u64, n: usize, w: cuda.window_attention.Window, scratch: u64) !void {
        try o.k.win.decode(o.s, q, kc, vc, out, pos, slots, w, scratch, n);
    }

    /// Prompt rows of one stream at p0, p0 + 1, ...: a ring and window (sliding), or every key before (win 0, full).
    pub fn promptRows(o: Ops, q: u64, kc: u64, vc: u64, out: u64, p0: usize, n: usize, w: cuda.window_attention.Window) !void {
        try o.k.win.prompt(o.s, q, kc, vc, out, p0, n, w);
    }

    /// Decode rows of a full layer: committed keys from each stream's cache, then the row's own chain.
    pub fn full(o: Ops, q: u64, kn: u64, vn: u64, rows: cuda.tree_attention.Rows, scratch: u64, cap: usize, ng: usize, out: u64, n: usize, h: usize, hk: usize, d: usize, scale: f32) !void {
        if (n > cap) return error.TreeRowsPastScratch;
        const sc = cuda.tree_attention.TreeAttention.scratch(scratch, cap, h, hk, ng, d);
        try o.k.tree.run(o.s, q, kn, vn, rows, sc, out, n, h, hk, d, ng, scale);
    }

    /// x [m, D] bf16 times rows [n, D] bf16 into fp32 [m, n] (the router and the head).
    pub fn dense(o: Ops, x: u64, x_stride: usize, rows: u64, out: u64, m: usize, d: usize, n: usize) !void {
        try o.k.dense.rows(o.s, x, x_stride, rows, out, n, m, n, d);
    }

    /// Each row's top-k experts by sigmoid logit + bias, unnormalised sigmoid weights, then the shared expert's slot.
    pub fn topk(o: Ops, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, ne: usize, k: usize) !void {
        try o.k.moe.topk(o.s, logits, bias, .{ .experts = ne, .topk = k, .shared = true }, pick, wts, rows);
    }

    /// y [R, S, D] (fp32 decode, bf16 prompt) weighted by wts [R, S] in slot order -> [R, D] bf16.
    pub fn combine(o: Ops, y: u64, y_bf16: bool, wts: u64, out: u64, rows: usize, s: usize, d: usize) !void {
        try o.k.moe.combine(o.s, y, if (y_bf16) .bf16 else .f32, wts, out, .bf16, rows, s, d);
    }
};
