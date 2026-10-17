//! MiniMax Music 3's launches: music3.cu by symbol, and the cuBLAS / cuBLASLt products beside them.
const std = @import("std");
const cuda = @import("cuda");
const blas = @import("blas.zig");
const lt = @import("lt.zig");
const wts = @import("weights.zig");

const names = .{
    "m3_embed",        "m3_embed_code",    "m3_frame_embed",  "m3_rmsnorm",       "m3_add",           "m3_swiglu",
    "m3_dequant",      "m3_quant_rows",    "m3_gemv_bf16_2",  "m3_gemv_bf16_4",   "m3_gemv_i8_2",     "m3_gemv_i8_4",
    "m3_qk_rope",      "m3_attn_part",     "m3_attn_merge",   "m3_depth_kv",      "m3_depth_attn",    "m3_sample",
    "m3_frame_done",   "m3_state_part",    "m3_force_codes",  "m3_cond_mix",      "m3_im2col3",       "m3_dit_input",
    "m3_add_h",        "m3_time_features", "m3_silu_h",       "m3_linear_h1",     "m3_gather_rows_h", "m3_put_time",
    "m3_layernorm_h",  "m3_dit_rope",      "m3_softmax_rows", "m3_heads_to_rows", "m3_glu_h",         "m3_dit_gather",
    "m3_euler",        "m3_noise",         "m3_snake",        "m3_bias_rows",     "m3_overlap_add",   "m3_zero_rows",
    "m3_tanh",         "m3_add_bias",      "m3_latent_half",  "m3_flash_attn",    "m3_glu_bf16_2",    "m3_glu_bf16_4",
    "m3_glu_i8_2",     "m3_glu_i8_4",      "m3_gemv_g8_2",    "m3_gemv_g8_4",     "m3_glu_g8_2",      "m3_glu_g8_4",
    "m3_quant_groups", "m3_bias_h",
};

pub const F = enum(u8) {
    embed,
    embed_code,
    frame_embed,
    rmsnorm,
    add,
    swiglu,
    dequant,
    quant_rows,
    gemv_bf16_2,
    gemv_bf16_4,
    gemv_i8_2,
    gemv_i8_4,
    qk_rope,
    attn_part,
    attn_merge,
    depth_kv,
    depth_attn,
    sample,
    frame_done,
    state_part,
    force_codes,
    cond_mix,
    im2col3,
    dit_input,
    add_h,
    time_features,
    silu_h,
    linear_h1,
    gather_rows_h,
    put_time,
    layernorm_h,
    dit_rope,
    softmax_rows,
    heads_to_rows,
    glu_h,
    dit_gather,
    euler,
    noise,
    snake,
    bias_rows,
    overlap_add,
    zero_rows,
    tanh,
    add_bias,
    latent_half,
    flash_attn,
    glu_bf16_2,
    glu_bf16_4,
    glu_i8_2,
    glu_i8_4,
    gemv_g8_2,
    gemv_g8_4,
    glu_g8_2,
    glu_g8_4,
    quant_groups,
    bias_h,
};

comptime {
    std.debug.assert(names.len == @typeInfo(F).@"enum".field_names.len);
}

const n_fns = names.len;

pub const Kernels = struct {
    d: *const cuda.Driver,
    mod: cuda.Module,
    fns: [n_fns]cuda.Function,
    blas: blas.Blas,
    ws: cuda.DeviceBuffer,
    lt: lt.Plans,
    sms: usize,
    pdl: bool = false, // GEMVs with programmatic dependent launch (sm_90 up)

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.d = d;
        k.sms = @intCast(try ctx.attribute(.multiprocessor_count));
        k.pdl = try ctx.capability() >= 90;
        k.mod = try cuda.Module.load(d, cuda.kernels.music3);
        inline for (names, 0..) |name, j| k.fns[j] = try k.mod.function(name);
        const ws_bytes: usize = 64 << 20;
        k.ws = try cuda.DeviceBuffer.alloc(d, ws_bytes);
        k.blas = try blas.Blas.open(k.ws.ptr, ws_bytes);
        k.lt = try lt.Plans.open(gpa, d, 64 << 20);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.lt.close();
        k.blas.close();
        k.ws.free();
        k.mod.unload();
    }

    pub fn f(k: *const Kernels, which: F) cuda.Function {
        return k.fns[@backingInt(which)];
    }

    pub fn go(k: *const Kernels, which: F, s: cuda.Stream, grid: [3]usize, block: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(k.f(which), .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, s, a);
    }

    /// A grid-stride elementwise launch over n items.
    pub fn flat(k: *const Kernels, which: F, s: cuda.Stream, n: usize, a: *cuda.Args) !void {
        const blocks = @max(1, @min((n + 255) / 256, k.sms * 32));
        try k.go(which, s, .{ blocks, 1, 1 }, 256, a);
    }

    fn i(x: usize) i32 {
        return @intCast(x);
    }

    // -------------------------------------------------------------------------------------------- AR

    pub fn embed(k: *const Kernels, s: cuda.Stream, ids: u64, table: u64, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(ids);
        a.add(table);
        a.add(out);
        a.add(i(4096));
        try k.go(.embed, s, .{ rows, 1, 1 }, 256, &a);
    }

    pub fn embedCode(k: *const Kernels, s: cuda.Stream, code: u64, offset: usize, table: u64, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(code);
        a.add(i(offset));
        a.add(table);
        a.add(out);
        a.add(i(rows));
        a.add(i(4096));
        try k.go(.embed_code, s, .{ 1, 1, 1 }, 512, &a);
    }

    pub fn frameEmbed(k: *const Kernels, s: cuda.Stream, codes: u64, audio: u64, extra: u64, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(codes);
        a.add(audio);
        a.add(extra);
        a.add(out);
        a.add(i(rows));
        a.add(i(4096));
        a.add(@as(f32, @floatCast(1.0 / @sqrt(8.0))));
        try k.go(.frame_embed, s, .{ 16, 1, 1 }, 256, &a);
    }

    pub fn rmsnorm(k: *const Kernels, s: cuda.Stream, x: u64, xs: usize, w: u64, y: u64, ys: usize, rows: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(xs));
        a.add(w);
        a.add(y);
        a.add(i(ys));
        a.add(i(4096));
        a.add(eps);
        try k.go(.rmsnorm, s, .{ rows, 1, 1 }, 512, &a);
    }

    pub fn add(k: *const Kernels, s: cuda.Stream, x: u64, b: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(b);
        a.add(n);
        try k.flat(.add, s, n, &a);
    }

    pub fn swiglu(k: *const Kernels, s: cuda.Stream, gu: u64, y: u64, width: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(gu);
        a.add(y);
        a.add(i(width));
        a.add(rows);
        try k.flat(.swiglu, s, rows * width, &a);
    }

    pub fn dequant(k: *const Kernels, s: cuda.Stream, m: wts.Mat, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(m.w);
        a.add(m.sc);
        a.add(out);
        a.add(i(m.k));
        a.add(m.n);
        try k.flat(.dequant, s, m.n * m.k, &a);
    }

    /// bf16 rows to int8 with a scale a row (group 0) or a scale a group of 128 inputs.
    pub fn quantRows(k: *const Kernels, s: cuda.Stream, src: u64, n: usize, kk: usize, group: usize, q: u64, sc: u64) !void {
        if (group != 0 and group != 128) return error.UnsupportedGroup;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(i(kk));
        a.add(q);
        a.add(sc);
        try k.go(if (group == 0) .quant_rows else .quant_groups, s, .{ n, 1, 1 }, 256, &a);
    }

    pub const Gemv = struct {
        xs: usize = 0, // x row stride (0: K)
        glu: bool = false, // gate | up rows: write the SwiGLU of rows n and N / 2 + n at n
        y: u64 = 0,
        ys: usize = 0, // 0: N (N / 2 with glu)
        yf: u64 = 0,
        res: u64 = 0,
        res_div: usize = 1,
    };

    /// y[m] = W x[m] for m < rows (2 or 4): bf16 or int8 weights by the Mat.
    pub fn gemv(k: *const Kernels, s: cuda.Stream, w: wts.Mat, x: u64, rows: usize, o: Gemv) !void {
        std.debug.assert(rows == 2 or rows == 4);
        const int8 = w.sc != 0;
        const four = rows == 4;
        const which: F = if (w.group != 0) (if (o.glu) (if (four) F.glu_g8_4 else F.glu_g8_2) else (if (four) F.gemv_g8_4 else F.gemv_g8_2)) else if (o.glu)
            (if (int8) (if (four) F.glu_i8_4 else F.glu_i8_2) else (if (four) F.glu_bf16_4 else F.glu_bf16_2))
        else
            (if (int8) (if (four) F.gemv_i8_4 else F.gemv_i8_2) else (if (four) F.gemv_bf16_4 else F.gemv_bf16_2));
        std.debug.assert(w.group == 0 or w.group == 128);
        const outs = if (o.glu) w.n / 2 else w.n;
        var a: cuda.Args = .{};
        a.add(w.w);
        a.add(w.sc);
        a.add(x);
        a.add(i(if (o.xs != 0) o.xs else w.k));
        a.add(i(w.k));
        a.add(i(w.n));
        a.add(o.y);
        a.add(i(if (o.ys != 0) o.ys else outs));
        a.add(o.yf);
        a.add(o.res);
        a.add(i(o.res_div));
        try cuda.launch.launch(k.f(which), .{ .grid = .{ .x = @intCast((outs + 7) / 8) }, .block = .{ .x = 256 }, .pdl = k.pdl }, s, &a);
    }

    /// Prompt rows through cuBLAS: y [t, N] = x [t, K] W^T (bf16, fp32 compute); int8 weights dequantized to `tmp`.
    pub fn linearRows(k: *const Kernels, s: cuda.Stream, w: wts.Mat, x: u64, y: u64, t: usize, tmp: u64) !void {
        var wp = w.w;
        if (w.sc != 0) {
            try k.dequant(s, w, tmp);
            wp = tmp;
        }
        try k.blas.gemm(s, .t, .n, w.n, t, w.k, 1.0, wp, w.k, 0, x, w.k, 0, 0.0, y, w.n, 0, 1, .bf16, .bf16);
    }

    pub fn qkRope(k: *const Kernels, s: cuda.Stream, qkv: u64, qn: u64, kn: u64, inv_freq: u64, pos: u64, t: usize, rows: usize, q: u64, kc: u64, vc: u64, ctx: usize) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(qn);
        a.add(kn);
        a.add(inv_freq);
        a.add(pos);
        a.add(i(t));
        a.add(q);
        a.add(kc);
        a.add(vc);
        a.add(i(ctx));
        a.add(@as(f32, 1e-6));
        try k.go(.qk_rope, s, .{ rows, 48, 1 }, 128, &a);
    }

    pub fn attention(k: *const Kernels, s: cuda.Stream, q: u64, kc: u64, vc: u64, pos: u64, t: usize, rows: usize, ctx: usize, splits: usize, ml: u64, acc: u64, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(q);
        a.add(kc);
        a.add(vc);
        a.add(pos);
        a.add(i(t));
        a.add(i(ctx));
        a.add(i(splits));
        a.add(@as(f32, 1.0 / @sqrt(128.0)));
        a.add(ml);
        a.add(acc);
        try k.go(.attn_part, s, .{ 8, rows, splits }, 128, &a);
        var b: cuda.Args = .{};
        b.add(ml);
        b.add(acc);
        b.add(i(splits));
        b.add(out);
        try k.go(.attn_merge, s, .{ rows, 32, 1 }, 128, &b);
    }

    pub fn depthAttention(k: *const Kernels, s: cuda.Stream, qkv: u64, p0: usize, rows: usize, kc: u64, vc: u64, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(i(p0));
        a.add(i(2));
        a.add(kc);
        a.add(vc);
        try k.go(.depth_kv, s, .{ rows, 16, 1 }, 256, &a);
        var b: cuda.Args = .{};
        b.add(qkv);
        b.add(i(p0));
        b.add(i(2));
        b.add(kc);
        b.add(vc);
        b.add(@as(f32, 1.0 / 16.0));
        b.add(out);
        try k.go(.depth_attn, s, .{ rows, 16, 1 }, 256, &b);
    }

    pub const Draw = struct { cfg: f32, top_k: i32, seed: u64, step: u64, slot: i32, pick: u64, code: u64 = 0, tokens: u64 = 0, count: u64 = 0, force: u64 = 0 };

    pub fn sample(k: *const Kernels, s: cuda.Stream, logits: u64, v: usize, rank_cond: bool, d: Draw) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(i(v));
        a.add(d.cfg);
        a.add(d.top_k);
        a.add(@as(i32, @intFromBool(rank_cond)));
        a.add(d.seed);
        a.add(d.step);
        a.add(d.slot);
        a.add(d.pick);
        a.add(d.code);
        a.add(d.tokens);
        a.add(d.count);
        a.add(d.force);
        try k.go(.sample, s, .{ 1, 1, 1 }, 1024, &a);
    }

    pub fn frameDone(k: *const Kernels, s: cuda.Stream, pos: u64, step: u64, count: u64) !void {
        var a: cuda.Args = .{};
        a.add(pos);
        a.add(step);
        a.add(count);
        try k.go(.frame_done, s, .{ 1, 1, 1 }, 32, &a);
    }

    pub fn statePart(k: *const Kernels, s: cuda.Stream, src: u64, states: u64, count: u64, part: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(states);
        a.add(count);
        a.add(i(part));
        try k.go(.state_part, s, .{ 1, 1, 1 }, 256, &a);
    }

    pub fn forceCodes(k: *const Kernels, s: cuda.Stream, forced: u64, count: u64, codes: u64, token: u64) !void {
        var a: cuda.Args = .{};
        a.add(forced);
        a.add(count);
        a.add(codes);
        a.add(token);
        try k.go(.force_codes, s, .{ 1, 1, 1 }, 32, &a);
    }

    // -------------------------------------------------------------------------------------------- DiT

    pub fn hgemm(k: *const Kernels, s: cuda.Stream, ta: blas.Op, tb: blas.Op, m: usize, n: usize, kk: usize, alpha: f32, a_: u64, lda: usize, sa: usize, b_: u64, ldb: usize, sb: usize, beta: f32, c: u64, ldc: usize, sc: usize, batch: usize, out_t: blas.DataType) !void {
        try k.blas.gemm(s, ta, tb, m, n, kk, alpha, a_, lda, sa, b_, ldb, sb, beta, c, ldc, sc, batch, .f16, out_t);
    }

    pub fn args(comptime T: type, vals: T) cuda.Args {
        var a: cuda.Args = .{};
        inline for (std.meta.fields(T)) |fd| a.add(@field(vals, fd.name));
        return a;
    }
};
