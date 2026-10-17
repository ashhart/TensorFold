//! Gemma 4's image tower on Metal: patch projection and 2D position rows, 27 encoder layers, 3 x 3 pooling, projection.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const lane = @import("../../core/lane_projection.zig");
const Mlx = @import("weights.zig").Mlx;
const image = @import("image.zig");

pub const width = 1152;
pub const heads = 16;
pub const head_dim = 72;
pub const depth = 27;
pub const mlp_inter = 4304;
pub const out_hidden = 5376; // the language model's width
pub const max_patches = image.max_budget * image.pool * image.pool;
const table_rows = 10240; // position_embedding_size: each axis's table
const chunk = 256; // query rows an attention pass scores at once
const eps: f32 = 1e-6;
const rope_theta: f32 = 100;

const quant_group = 64; // the projection's quantization group, the language model's

pub const Ref = lane.Ref;
fn ref(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}

/// The 4-bit projection as affine_mm reads it: packed words [5376, 1152 / 8], a bf16 scale and bias per group of 64.
fn projectionFits(w: ck.Tensor, s: ck.Tensor, b: ck.Tensor) bool {
    if (w.dtype != .u32 or w.rank != 2 or w.shape[0] != out_hidden or w.shape[1] * 8 != width or w.count() != out_hidden * width / 8) return false;
    for ([_]ck.Tensor{ s, b }) |t| {
        if (t.dtype != .bf16 or t.rank != 2 or t.shape[0] != out_hidden or t.shape[1] != width / quant_group or t.count() != out_hidden * width / quant_group) return false;
    }
    return true;
}

const glue_names = [_][:0]const u8{ "g4v_pos", "g4v_rms", "g4v_rms_add", "g4v_qkv", "g4v_softmax", "g4v_heads", "g4v_geglu", "g4v_pool", "g4v_rows" };
const G = enum { pos, rms, rms_add, qkv, softmax, heads_, geglu, pool, rows };

const Layer = struct { in_norm: Ref, post_attn: Ref, pre_ff: Ref, post_ff: Ref, q: Ref, k: Ref, v: Ref, o: Ref, q_norm: Ref, k_norm: Ref, gate: Ref, up: Ref, down: Ref };

/// No tower tensor stays in its file: vision_tower. and embed_vision. pack, 256-byte aligned, into their own buffers.
fn never(_: []const u8, _: usize) bool {
    return false;
}

/// The checkpoint's tower tensors, packed 256-byte aligned beside the language model's (the checkpoint frees them).
pub fn addTensors(c: *ck.Checkpoint, device: mtl.Device, files: []const [:0]u8) !void {
    for (files) |path| for ([_][]const u8{ "vision_tower.", "embed_vision." }) |p| try c.addFileSplit(device, path, "", p, never);
}

/// Whether the checkpoint has the tower (a text-only conversion does not).
pub fn present(c: *const ck.Checkpoint) bool {
    _ = c.get("vision_tower.patch_embedder.input_proj.weight") catch return false;
    return true;
}

pub const Vision = struct {
    gpa: std.mem.Allocator,
    lib: mtl.Library,
    gemm: [3][2][2]mtl.Pipeline, // [bf16 nt, f32 nt, f32 nn][M % 64 == 0][N % 128 == 0]
    glue: [glue_names.len]mtl.Pipeline,
    mm: affine_mm.Pipes, // the model's: the projection is 4-bit groups of 64 as the language model is
    input_proj: Ref,
    table: Ref,
    std_bias: Ref,
    std_scale: Ref,
    layers: [depth]Layer,
    proj: Mlx,
    scratch: mtl.Buffer,
    x: Ref,
    xn: Ref,
    qkv: Ref,
    q: Ref,
    k: Ref,
    v: Ref,
    s: Ref,
    o: Ref,
    att: Ref,
    gu: Ref,
    act: Ref,
    pooled: Ref,
    pn: Ref,
    /// Host-written per image: the patches as bf16 2 (p - 0.5), and the RoPE's cos and sin rows.
    patches: mtl.Buffer,
    cos: mtl.Buffer,
    sin: mtl.Buffer,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ck.Checkpoint, mm: affine_mm.Pipes) !*Vision {
        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        v.gpa = gpa;
        v.mm = mm;
        const src = try frags.source(device, gpa, ks.gemma4_vision);
        defer gpa.free(src);
        v.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer v.lib.deinit();
        var made: usize = 0;
        errdefer for (0..made) |i| v.pipeline(i).deinit();
        const kinds = [_][]const u8{ "bf16_nt", "f32_nt", "f32_nn" };
        for (kinds, 0..) |kind, ki| for (0..2) |m| for (0..2) |n| {
            var name: [64]u8 = undefined;
            const full = try std.fmt.bufPrintSentinel(&name, "g4v_gemm_{s}_{d}{d}", .{ kind, m, n }, 0);
            v.gemm[ki][m][n] = try mtl.Pipeline.init(device, v.lib, full, false);
            made += 1;
        };
        for (glue_names, 0..) |n, i| {
            v.glue[i] = try mtl.Pipeline.init(device, v.lib, n, false);
            made += 1;
        }
        const t = struct {
            fn get(cp: *const ck.Checkpoint, comptime fmt: []const u8, args: anytype, count: usize) !ck.Tensor {
                var b: [160]u8 = undefined;
                const name = try std.fmt.bufPrint(&b, fmt, args);
                const x = try cp.get(name);
                if (x.dtype != .bf16 or x.count() != count) {
                    std.log.err("gemma4 vision: {s} is not bf16[{d}]", .{ name, count });
                    return error.UnexpectedGemmaVisionTensor;
                }
                return x;
            }
        };
        const pe = "vision_tower.patch_embedder.";
        v.input_proj = ref(try t.get(c, pe ++ "input_proj.weight", .{}, width * image.patch_width));
        v.table = ref(try t.get(c, pe ++ "position_embedding_table", .{}, 2 * table_rows * width));
        v.std_bias = ref(try t.get(c, "vision_tower.std_bias", .{}, width));
        v.std_scale = ref(try t.get(c, "vision_tower.std_scale", .{}, width));
        for (&v.layers, 0..) |*l, i| {
            const p = "vision_tower.encoder.layers.{d}.";
            l.* = .{
                .in_norm = ref(try t.get(c, p ++ "input_layernorm.weight", .{i}, width)),
                .post_attn = ref(try t.get(c, p ++ "post_attention_layernorm.weight", .{i}, width)),
                .pre_ff = ref(try t.get(c, p ++ "pre_feedforward_layernorm.weight", .{i}, width)),
                .post_ff = ref(try t.get(c, p ++ "post_feedforward_layernorm.weight", .{i}, width)),
                .q = ref(try t.get(c, p ++ "self_attn.q_proj.linear.weight", .{i}, width * width)),
                .k = ref(try t.get(c, p ++ "self_attn.k_proj.linear.weight", .{i}, width * width)),
                .v = ref(try t.get(c, p ++ "self_attn.v_proj.linear.weight", .{i}, width * width)),
                .o = ref(try t.get(c, p ++ "self_attn.o_proj.linear.weight", .{i}, width * width)),
                .q_norm = ref(try t.get(c, p ++ "self_attn.q_norm.weight", .{i}, head_dim)),
                .k_norm = ref(try t.get(c, p ++ "self_attn.k_norm.weight", .{i}, head_dim)),
                .gate = ref(try t.get(c, p ++ "mlp.gate_proj.linear.weight", .{i}, mlp_inter * width)),
                .up = ref(try t.get(c, p ++ "mlp.up_proj.linear.weight", .{i}, mlp_inter * width)),
                .down = ref(try t.get(c, p ++ "mlp.down_proj.linear.weight", .{i}, width * mlp_inter)),
            };
        }
        const pp = "embed_vision.embedding_projection.";
        const pw = try c.get(pp ++ "weight");
        const ps = try c.get(pp ++ "scales");
        const pb = try c.get(pp ++ "biases");
        if (!projectionFits(pw, ps, pb)) {
            std.log.err("gemma4 vision: {s}* is not u32[{d}, {d}] with bf16[{d}, {d}] scales and biases", .{ pp, out_hidden, width / 8, out_hidden, width / quant_group });
            return error.UnexpectedGemmaVisionTensor;
        }
        v.proj = .{ .w = ref(pw), .s = ref(ps), .b = ref(pb), .n = out_hidden, .k = width };
        const P: usize = max_patches;
        const T: usize = image.max_budget;
        const sizes = [_]usize{ P * width * 2, P * width * 2, P * 3 * width * 2, heads * P * head_dim * 4, heads * P * head_dim * 4, heads * P * head_dim * 4, heads * chunk * P * 4, heads * chunk * head_dim * 4, P * width * 2, P * 2 * mlp_inter * 2, P * mlp_inter * 2, T * width * 2, T * width * 2 };
        var total: usize = 0;
        for (sizes) |s| total += std.mem.alignForward(usize, s, 256);
        v.scratch = try device.buffer(total, mtl.ResourceOptions.private);
        errdefer v.scratch.deinit();
        var off: usize = 0;
        const fields = [_]*Ref{ &v.x, &v.xn, &v.qkv, &v.q, &v.k, &v.v, &v.s, &v.o, &v.att, &v.gu, &v.act, &v.pooled, &v.pn };
        for (fields, sizes) |f, s| {
            f.* = .{ .buf = v.scratch, .off = off };
            off += std.mem.alignForward(usize, s, 256);
        }
        const shared = mtl.ResourceOptions.shared;
        v.patches = try device.buffer(P * image.patch_width * 2, shared);
        errdefer v.patches.deinit();
        v.cos = try device.buffer(P * head_dim * 2, shared);
        errdefer v.cos.deinit();
        v.sin = try device.buffer(P * head_dim * 2, shared);
        return v;
    }

    fn pipeline(v: *Vision, i: usize) mtl.Pipeline {
        const g = 3 * 2 * 2;
        return if (i < g) v.gemm[i / 4][(i / 2) % 2][i % 2] else v.glue[i - g];
    }

    pub fn deinit(v: *Vision) void {
        for (0..3 * 2 * 2 + glue_names.len) |i| v.pipeline(i).deinit();
        v.lib.deinit();
        v.scratch.deinit();
        v.patches.deinit();
        v.cos.deinit();
        v.sin.deinit();
        v.gpa.destroy(v);
    }

    /// An image's patches (image.prepare's) to `out` [gh * gw / 9, 5376]; `stop`: leave `x` after that many layers.
    pub fn encode(v: *Vision, e: mtl.ComputeEncoder, pixels: []const f32, gh: u32, gw: u32, out: Ref, stop: ?u32) !void {
        const P: u32 = gh * gw;
        if (gh == 0 or gw == 0 or gh % image.pool != 0 or gw % image.pool != 0 or P > max_patches) return error.GemmaImageGrid;
        if (pixels.len != @as(usize, P) * image.patch_width) return error.GemmaImageGrid;
        for (pixels, v.patches.slice(u16, pixels.len)) |p, *d| d.* = bf16(2 * (p - 0.5));
        rope(v.cos.slice(u16, @as(usize, P) * head_dim), v.sin.slice(u16, @as(usize, P) * head_dim), gw, P);
        v.linear(e, .{ .buf = v.patches }, v.input_proj, v.x, P, width, image.patch_width, width);
        v.run(e, .pos, &.{ v.x, v.table }, .{ width, P, 1 }, .{ 256, 1, 1 }, false, gw);
        if (stop) |s| if (s == 0) return;
        for (v.layers, 0..) |l, i| {
            v.layer(e, l, P);
            if (stop) |s| if (i + 1 == s) return;
        }
        const T = P / (image.pool * image.pool);
        v.run(e, .pool, &.{ v.x, v.std_bias, v.std_scale, v.pooled }, .{ width, T, 1 }, .{ 256, 1, 1 }, false, gw);
        v.rms(e, v.pooled, null, v.pn, T);
        affine_mm.dense(e, v.mm, v.pn, v.proj, out, T);
    }

    /// Rows of an encoded image over `n` embedded prompt rows (`dst`, 5376 bf16 a row), in the prompt's encoder.
    pub fn inject(v: *Vision, e: mtl.ComputeEncoder, src: Ref, dst: Ref, n: u32) void {
        v.run(e, .rows, &.{ src, dst }, .{ out_hidden / 8, n, 1 }, .{ 64, 1, 1 }, false, null);
    }

    fn layer(v: *Vision, e: mtl.ComputeEncoder, l: Layer, P: u32) void {
        v.rms(e, v.x, l.in_norm, v.xn, P);
        for ([_]Ref{ l.q, l.k, l.v }, 0..) |w, i| v.linear(e, v.xn, w, .{ .buf = v.qkv.buf, .off = v.qkv.off + i * width * 2 }, P, width, width, 3 * width);
        e.setPipeline(v.glue[@backingInt(G.qkv)]);
        for ([_]Ref{ v.qkv, l.q_norm, l.k_norm, .{ .buf = v.cos }, .{ .buf = v.sin }, v.q, v.k, v.v }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(P, 8);
        e.setValue(eps, 9);
        e.dispatchGroups(mtl.Size.of(@as(usize, P) * heads * 3, 1, 1), mtl.Size.of(32, 1, 1));
        var q0: u32 = 0;
        while (q0 < P) : (q0 += chunk) {
            const lc = @min(chunk, P - q0);
            v.gemmAt(e, 1, .{ .buf = v.q.buf, .off = v.q.off + @as(usize, q0) * head_dim * 4 }, v.k, v.s, lc, P, head_dim, head_dim, head_dim, P, heads, @as(usize, P) * head_dim, @as(usize, P) * head_dim, @as(usize, lc) * P);
            e.setPipeline(v.glue[@backingInt(G.softmax)]);
            e.setBuffer(v.s.buf, v.s.off, 0);
            e.setValue(P, 1);
            e.dispatchGroups(mtl.Size.of(@as(usize, heads) * lc, 1, 1), mtl.Size.of(256, 1, 1));
            v.gemmAt(e, 2, v.s, v.v, v.o, lc, head_dim, P, P, head_dim, head_dim, heads, @as(usize, lc) * P, @as(usize, P) * head_dim, @as(usize, lc) * head_dim);
            e.setPipeline(v.glue[@backingInt(G.heads_)]);
            e.setBuffer(v.o.buf, v.o.off, 0);
            e.setBuffer(v.att.buf, v.att.off, 1);
            e.setValue([2]u32{ lc, q0 }, 2);
            e.dispatchThreads(mtl.Size.of(width, lc, 1), mtl.Size.of(256, 1, 1));
        }
        v.linear(e, v.att, l.o, v.xn, P, width, width, width);
        v.rmsAdd(e, v.x, v.xn, l.post_attn, P);
        v.rms(e, v.x, l.pre_ff, v.xn, P);
        v.linear(e, v.xn, l.gate, v.gu, P, mlp_inter, width, 2 * mlp_inter);
        v.linear(e, v.xn, l.up, .{ .buf = v.gu.buf, .off = v.gu.off + mlp_inter * 2 }, P, mlp_inter, width, 2 * mlp_inter);
        v.run(e, .geglu, &.{ v.gu, v.act }, .{ mlp_inter, P, 1 }, .{ 256, 1, 1 }, false, mlp_inter);
        v.linear(e, v.act, l.down, v.xn, P, width, mlp_inter, width);
        v.rmsAdd(e, v.x, v.xn, l.post_ff, P);
    }

    fn run(v: *Vision, e: mtl.ComputeEncoder, g: G, bufs: []const Ref, grid: [3]usize, group: [3]usize, groups: bool, value: ?u32) void {
        e.setPipeline(v.glue[@backingInt(g)]);
        for (bufs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        if (value) |x| e.setValue(x, bufs.len);
        const a = mtl.Size.of(grid[0], grid[1], grid[2]);
        const b = mtl.Size.of(group[0], group[1], group[2]);
        if (groups) e.dispatchGroups(a, b) else e.dispatchThreads(a, b);
    }

    /// y = Gemma4RMSNorm(x) [rows, width], by `w` when there is one.
    fn rms(v: *Vision, e: mtl.ComputeEncoder, x: Ref, w: ?Ref, y: Ref, rows: u32) void {
        e.setPipeline(v.glue[@backingInt(G.rms)]);
        for ([_]Ref{ x, w orelse x, y }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(extern struct { width: u32, weighted: u32, eps: f32 }{ .width = width, .weighted = @intFromBool(w != null), .eps = eps }, 3);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// x += Gemma4RMSNorm(y) by w: a branch's post norm into the residual.
    fn rmsAdd(v: *Vision, e: mtl.ComputeEncoder, x: Ref, y: Ref, w: Ref, rows: u32) void {
        e.setPipeline(v.glue[@backingInt(G.rms_add)]);
        for ([_]Ref{ x, y, w }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(eps, 3);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// y [M, N] (rows `ldy` apart) = x [M, K] w^T in bf16 (w as stored, [N, K]).
    fn linear(v: *Vision, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, M: u32, N: u32, K: u32, ldy: u32) void {
        v.gemmAt(e, 0, x, w, y, M, N, K, K, K, ldy, 1, 0, 0, 0);
    }

    /// tfp::gemm_nax: d [batch, M, N] = a x b (kind 0: bf16 a b^T, 1: fp32 a b^T, 2: fp32 a b), batches `s*` apart.
    fn gemmAt(v: *Vision, e: mtl.ComputeEncoder, kind: usize, a: Ref, b: Ref, d: Ref, M: u32, N: u32, K: u32, lda: u32, ldb: u32, ldd: u32, batch: u32, sa: usize, sb: usize, sd: usize) void {
        const tn = (N + 127) / 128;
        const tm = (M + 63) / 64;
        e.setPipeline(v.gemm[kind][@intFromBool(M % 64 == 0)][@intFromBool(N % 128 == 0)]);
        e.setBuffer(a.buf, a.off, 0);
        e.setBuffer(b.buf, b.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, lda, ldb, ldd, tn, tm, 2, K / 256, sa, sb, sd, 0, 0 };
        for (vals, 0..) |x, i| p[i] = @intCast(x);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(d.buf, d.off, 3);
        e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, @as(usize, batch) * 2), mtl.Size.of(32, 4, 2));
    }
};

/// fp32 to bf16, round to nearest even.
pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// Gemma4VisionRotaryEmbedding, gw wide: dims 0-35 by x and 36-71 by y, 18 frequencies twice each, bf16 cos and sin.
fn rope(cs: []u16, sn: []u16, gw: u32, P: u32) void {
    const half = head_dim / 2;
    var inv: [half / 2]f32 = undefined;
    for (&inv, 0..) |*f, i| f.* = 1.0 / std.math.pow(f32, rope_theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, half));
    for (0..P) |p| for (0..head_dim) |d| {
        const axis: f32 = @floatFromInt(if (d < half) p % gw else p / gw);
        const angle = axis * inv[(d % half) % (half / 2)];
        cs[p * head_dim + d] = bf16(@cos(angle));
        sn[p * head_dim + d] = bf16(@sin(angle));
    };
}

test "the projection is admitted only as packed u32 words with bf16 scales and biases per group of 64" {
    const T = struct {
        fn of(dtype: ck.DType, rows: usize, cols: usize) ck.Tensor {
            var t: ck.Tensor = .{ .buffer = undefined, .offset = 0, .bytes = rows * cols * dtype.size(), .dtype = dtype, .rank = 2 };
            t.shape[0] = rows;
            t.shape[1] = cols;
            return t;
        }
    };
    const words = T.of(.u32, out_hidden, width / 8);
    const scales = T.of(.bf16, out_hidden, width / quant_group);
    try std.testing.expect(projectionFits(words, scales, scales));
    const f32s = T.of(.f32, out_hidden, width / quant_group);
    try std.testing.expect(!projectionFits(words, f32s, scales));
    try std.testing.expect(!projectionFits(words, scales, f32s));
    const group32 = T.of(.bf16, out_hidden, width / 32);
    try std.testing.expect(!projectionFits(words, group32, group32));
    var short = scales; // the header's shape, the file's bytes cut short
    short.bytes -= 2;
    try std.testing.expect(!projectionFits(words, scales, short));
    var flat = scales;
    flat.rank = 1;
    try std.testing.expect(!projectionFits(words, flat, scales));
    try std.testing.expect(!projectionFits(T.of(.u32, out_hidden + 1, width / 8), scales, scales));
    try std.testing.expect(!projectionFits(T.of(.bf16, out_hidden, width / 8), scales, scales));
}
