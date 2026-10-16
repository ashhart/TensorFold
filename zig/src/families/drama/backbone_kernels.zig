//! Drama backbone glue and GEMM kernels: the embedded images, or a kernel directory's for numerical-oracle tooling.
const std = @import("std");
const cuda = @import("cuda");
const backbone_gdn = @import("backbone_gdn.zig");

pub const glue_file = "drama_backbone.fatbin";
pub const gemm_file = "drama_backbone_gemm.fatbin";

/// The bf16 product tilings: (rows a tile, 8-column tiles a warp) for up to 16, 32 and more rows.
const tilings = [_][2]usize{ .{ 1, 1 }, .{ 2, 2 }, .{ 4, 2 } };
const gemm_symbols = [3][:0]const u8{ "drama_gemm_bf16_r1_n1_o0", "drama_gemm_bf16_r2_n2_o0", "drama_gemm_bf16_r4_n2_o0" };
/// One row, fp32 out holding bf16-rounded values (the lm_head's bf16 logits read as float).
const head_symbol = "drama_gemv_bf16_r1_o2";
/// One row through a staged weight path, bit-identical to the r1 tiling; needs k % gemv_chunk == 0.
const gemv_symbol = "drama_gemv_bf16_r1_o0";
const gemv_chunk = 256;
/// Expert products: [mode][column tiles a warp - 1]; mode 1 scales by the routed weight.
const expert_symbols = [2][2][:0]const u8{
    .{ "drama_expert_fp8_n1_m0", "drama_expert_fp8_n2_m0" },
    .{ "drama_expert_fp8_n1_m1", "drama_expert_fp8_n2_m1" },
};
const warps = 4;

fn readImage(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]align(16) u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(1 << 28), .@"16", null) catch |e| {
        std.log.err("drama backbone: cannot read kernel image {s}: {s}", .{ path, @errorName(e) });
        return e;
    };
}

fn loadImage(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, name: []const u8) !cuda.Module {
    const image = try readImage(gpa, io, dir, name);
    defer gpa.free(image);
    return cuda.Module.load(d, image);
}

fn go(s: cuda.Stream, f: cuda.Function, grid: [3]usize, block: u32, a: *cuda.Args) !void {
    const g: cuda.Dim3 = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) };
    try cuda.launch.launch(f, .{ .grid = g, .block = .{ .x = block } }, s, a);
}

fn int(v: usize) i32 {
    return @intCast(v);
}

pub const Kernels = struct {
    glue: cuda.Module,
    gemm_mod: cuda.Module,
    experts_mod: cuda.Module,
    embed: cuda.embed_rows.EmbedRows,
    router: cuda.grouped.Router,
    semantic_mix_fn: cuda.Function,
    add_norm_fn: cuda.Function,
    gated_rms_fn: cuda.Function,
    conv_fn: cuda.Function,
    conv_prefill_fn: cuda.Function,
    gdn: backbone_gdn.Kernels,
    attn_prep_fn: cuda.Function,
    attention_fn: cuda.Function,
    topk_fn: cuda.Function,
    act_quant_fn: cuda.Function,
    silu_mul_fn: cuda.Function,
    combine_fn: cuda.Function,
    gemms: [3]cuda.Function,
    head_fn: cuda.Function,
    gemv_fn: cuda.Function,
    experts: [2][2]cuda.Function,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, kernel_dir: []const u8) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var glue = if (kernel_dir.len == 0) try cuda.Module.load(d, cuda.kernels.drama_backbone) else try loadImage(gpa, io, d, kernel_dir, glue_file);
        errdefer glue.unload();
        var gemm_mod = if (kernel_dir.len == 0) try cuda.Module.load(d, cuda.kernels.drama_backbone_gemm) else try loadImage(gpa, io, d, kernel_dir, gemm_file);
        errdefer gemm_mod.unload();
        var experts_mod = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer experts_mod.unload();
        var embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer embed.unload();
        var k: Kernels = .{
            .glue = glue,
            .gemm_mod = gemm_mod,
            .experts_mod = experts_mod,
            .embed = embed,
            .router = try cuda.grouped.Router.resolve(experts_mod),
            .semantic_mix_fn = try glue.function("drama_semantic_mix"),
            .add_norm_fn = try glue.function("drama_add_norm"),
            .gated_rms_fn = try glue.function("drama_gated_rms_128"),
            .conv_fn = try glue.function("drama_gdn_conv"),
            .conv_prefill_fn = try glue.function("drama_gdn_conv_prefill"),
            .gdn = try backbone_gdn.Kernels.init(&glue),
            .attn_prep_fn = try glue.function("drama_attn_prep"),
            .attention_fn = try glue.function("drama_attention"),
            .topk_fn = try glue.function("drama_topk"),
            .act_quant_fn = try glue.function("drama_act_quant"),
            .silu_mul_fn = try glue.function("drama_silu_mul"),
            .combine_fn = try glue.function("drama_moe_combine"),
            .gemms = undefined,
            .head_fn = undefined,
            .gemv_fn = undefined,
            .experts = undefined,
        };
        for (gemm_symbols, 0..) |sym, v| k.gemms[v] = try gemm_mod.function(sym);
        k.head_fn = try gemm_mod.function(head_symbol);
        k.gemv_fn = try gemm_mod.function(gemv_symbol);
        for (expert_symbols, 0..) |row, m| for (row, 0..) |sym, n| {
            k.experts[m][n] = try gemm_mod.function(sym);
        };
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.embed.unload();
        k.experts_mod.unload();
        k.gemm_mod.unload();
        k.glue.unload();
        k.* = undefined;
    }

    fn product(f: cuda.Function, s: cuda.Stream, rt: usize, nt: usize, x: u64, ldx: usize, m: usize, w: u64, n: usize, kdim: usize, y: u64, ldy: usize) !void {
        if (m == 0) return;
        if (kdim % 32 != 0 or ldx % 8 != 0 or x % 16 != 0 or w % 16 != 0) return error.DramaGemmShape;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(ldx));
        a.add(int(m));
        a.add(w);
        a.add(int(n));
        a.add(int(kdim));
        a.add(y);
        a.add(int(ldy));
        const cols = 8 * nt * warps;
        try go(s, f, .{ (n + cols - 1) / cols, (m + 16 * rt - 1) / (16 * rt), 1 }, 32 * warps, &a);
    }

    /// Y[m, n] = bf16(X[m, :] . W[n, :]) (fp32 accumulate). W has ceil8(n) rows; k % 32 == 0; rows 16-byte aligned.
    pub fn gemm(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, m: usize, w: u64, n: usize, kdim: usize, y: u64, ldy: usize) !void {
        if (m == 1 and kdim % gemv_chunk == 0) return gemv(k.gemv_fn, s, x, w, n, kdim, y);
        const v: usize = if (m <= 16) 0 else if (m <= 32) 1 else 2;
        try product(k.gemms[v], s, tilings[v][0], tilings[v][1], x, ldx, m, w, n, kdim, y, ldy);
    }

    fn gemv(f: cuda.Function, s: cuda.Stream, x: u64, w: u64, n: usize, kdim: usize, y: u64) !void {
        if (kdim % gemv_chunk != 0 or x % 16 != 0 or w % 16 != 0) return error.DramaGemmShape;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(int(n));
        a.add(int(kdim));
        a.add(y);
        try go(s, f, .{ (n + 7) / 8, 1, 1 }, 32, &a);
    }

    /// y[n] = float(bf16(x . W[n, :])) for one row x.
    pub fn head(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, n: usize, kdim: usize, y: u64) !void {
        try gemv(k.head_fn, s, x, w, n, kdim, y);
    }

    pub const ExpertCall = struct {
        a: u64, // e4m3 rows of kdim bytes
        a_div: usize, // pair p reads row p / a_div
        a_scales: u64, // f32 [rows][kdim / 128]
        w: u64, // e4m3 [experts][n][kdim]
        w_scales: u64, // f32 [experts][n / 128][kdim / 128]
        n: usize,
        kdim: usize,
        max_items: usize,
        routed: ?u64, // f32 [pairs] routed weights: scale before the bf16 store
        y: u64, // bf16 [pairs][n]
    };

    pub fn expert(k: *const Kernels, s: cuda.Stream, plan: cuda.grouped.Plan, pairs: usize, c: ExpertCall) !void {
        if (c.n % 128 != 0 or c.kdim % 128 != 0) return error.DramaGemmShape;
        const nt: usize = if (pairs > 64) 2 else 1;
        var a: cuda.Args = .{};
        a.add(c.a);
        a.add(int(c.a_div));
        a.add(c.a_scales);
        a.add(c.w);
        a.add(c.w_scales);
        a.add(int(c.n));
        a.add(int(c.kdim));
        a.add(plan.items);
        a.add(plan.counts);
        a.add(plan.members);
        a.add(c.routed orelse 0);
        a.add(c.y);
        const cols = 8 * nt * warps;
        const f = k.experts[@intFromBool(c.routed != null)][nt - 1];
        try go(s, f, .{ c.max_items, c.n / cols, 1 }, 32 * warps, &a);
    }

    pub fn semanticMix(k: *const Kernels, s: cuda.Stream, ids: u64, out: u64, extra: u64, rows: usize, width: usize, lo: u32, hi: u32) !void {
        var a: cuda.Args = .{};
        a.add(ids);
        a.add(out);
        a.add(extra);
        a.add(int(width));
        a.add(@as(i32, @intCast(lo)));
        a.add(@as(i32, @intCast(hi)));
        try go(s, k.semantic_mix_fn, .{ rows, 1, 1 }, 256, &a);
    }

    /// res = x (first) or bf16(x + res); y = bf16(norm(x + res) * (1 + w)), with q / scales also y's FP8 groups.
    pub fn addNorm(k: *const Kernels, s: cuda.Stream, x: u64, res: u64, w: u64, y: u64, q: ?u64, scales: ?u64, rows: usize, width: usize, eps: f32, first: bool) !void {
        if (width % 256 != 0 or (x | res | w | y | (q orelse 0)) % 16 != 0) return error.DramaRowShape;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(res);
        a.add(w);
        a.add(y);
        a.add(q orelse 0);
        a.add(scales orelse 0);
        a.add(int(width));
        a.add(eps);
        a.add(@as(i32, @intFromBool(first)));
        try go(s, k.add_norm_fn, .{ rows, 1, 1 }, 256, &a);
    }

    /// Drama-only 32 x 128 gated RMS; fixed full-extend reduction, independent of token count.
    pub fn gatedRms(k: *const Kernels, s: cuda.Stream, y: u64, y_ld: usize, z: u64, z_ld: usize, w: u64, out: u64, rows: usize, eps: f32) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(y);
        a.add(@as(i64, @intCast(y_ld)));
        a.add(z);
        a.add(@as(i64, @intCast(z_ld)));
        a.add(w);
        a.add(out);
        a.add(eps);
        const threads = cuda.launch.rowThreads(rows, 256, 32);
        try go(s, k.gated_rms_fn, .{ rows, 32 / (threads / 16), 1 }, threads, &a);
    }

    pub fn conv(k: *const Kernels, s: cuda.Stream, in: u64, ld_in: usize, w: u64, state: u64, out: u64, rows: usize, channels: usize) !void {
        try convUsing(k.conv_fn, s, in, ld_in, w, state, out, rows, channels);
    }

    pub fn convPrefill(k: *const Kernels, s: cuda.Stream, in: u64, ld_in: usize, w: u64, state: u64, out: u64, rows: usize, channels: usize) !void {
        try convUsing(k.conv_prefill_fn, s, in, ld_in, w, state, out, rows, channels);
    }

    fn convUsing(kernel: cuda.Function, s: cuda.Stream, in: u64, ld_in: usize, w: u64, state: u64, out: u64, rows: usize, channels: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(int(ld_in));
        a.add(w);
        a.add(state);
        a.add(out);
        a.add(int(rows));
        a.add(int(channels));
        try go(s, kernel, .{ (channels + 255) / 256, 1, 1 }, 256, &a);
    }

    pub const Attn = struct {
        qkv: u64,
        ld_qkv: usize,
        q_norm: u64,
        k_norm: u64,
        inv_freq: u64,
        q: u64,
        kc: u64,
        vc: u64,
        out: u64,
        pos0: usize,
        rows: usize,
        heads: usize,
        kv_heads: usize,
        eps: f32,
        scale: f32,
    };

    /// q/k norm + RoPE + cache append, then causal attention with the output gate.
    pub fn attention(k: *const Kernels, s: cuda.Stream, t: Attn) !void {
        var a: cuda.Args = .{};
        a.add(t.qkv);
        a.add(int(t.ld_qkv));
        a.add(t.q_norm);
        a.add(t.k_norm);
        a.add(t.inv_freq);
        a.add(t.q);
        a.add(t.kc);
        a.add(t.vc);
        a.add(int(t.pos0));
        a.add(int(t.heads));
        a.add(int(t.kv_heads));
        a.add(t.eps);
        try go(s, k.attn_prep_fn, .{ t.rows, t.heads + t.kv_heads, 1 }, 32, &a);
        var b: cuda.Args = .{};
        b.add(t.q);
        b.add(t.kc);
        b.add(t.vc);
        b.add(t.qkv);
        b.add(int(t.ld_qkv));
        b.add(t.out);
        b.add(int(t.pos0));
        b.add(int(t.heads));
        b.add(int(t.kv_heads));
        b.add(t.scale);
        try go(s, k.attention_fn, .{ t.rows, t.heads, 1 }, 256, &b);
    }

    pub fn topk(k: *const Kernels, s: cuda.Stream, logits: u64, ld: usize, ids: u64, wts: u64, rows: usize, top: usize) !void {
        if (top == 0 or top > 8) return error.DramaRowShape; // drama_topk keeps at most 8 picks in shared memory
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(int(ld));
        a.add(ids);
        a.add(wts);
        a.add(int(top));
        try go(s, k.topk_fn, .{ rows, 1, 1 }, 256, &a);
    }

    pub fn actQuant(k: *const Kernels, s: cuda.Stream, x: u64, ld_x: usize, q: u64, scales: u64, rows: usize, kdim: usize) !void {
        if (kdim % 512 != 0) return error.DramaRowShape;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(ld_x));
        a.add(q);
        a.add(scales);
        a.add(int(kdim));
        try go(s, k.act_quant_fn, .{ rows, kdim / 512, 1 }, 128, &a);
    }

    /// One launch: the routed rows of gu FP8-quantized into hq / hs, the shared expert's rows of shared_gu stored to h.
    pub fn siluMul(k: *const Kernels, s: cuda.Stream, gu: u64, ld_gu: usize, hq: u64, hs: u64, routed: usize, shared_gu: u64, ld_shared: usize, h: u64, shared: usize, inner: usize) !void {
        if (inner % 512 != 0) return error.DramaRowShape;
        var a: cuda.Args = .{};
        a.add(gu);
        a.add(int(ld_gu));
        a.add(int(routed));
        a.add(hq);
        a.add(hs);
        a.add(shared_gu);
        a.add(int(ld_shared));
        a.add(h);
        a.add(int(inner));
        try go(s, k.silu_mul_fn, .{ routed + shared, inner / 512, 1 }, 128, &a);
    }

    pub fn combine(k: *const Kernels, s: cuda.Stream, down: u64, top: usize, x: u64, wg: u64, shared: u64, out: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        a.add(down);
        a.add(int(top));
        a.add(x);
        a.add(wg);
        a.add(shared);
        a.add(out);
        a.add(int(width));
        try go(s, k.combine_fn, .{ rows, 1, 1 }, 256, &a);
    }
};
