//! GLM-5.3-Flash's image tower on the GPU: an image's patches to the rows that replace its `<|image|>` placeholders.
//! mlx-vlm's glm5_next/vision.py (what TensorFold 0.6.6's --vision serves) in bf16: a patch projection, 24 pre-norm
//! blocks (2-D rotary attention over the image's own patches, a clipped SwiGLU), a final norm, the 2x2 merge as a
//! projection of four neighbouring patches, and the merger's projection, layer norm, GELU and clipped SwiGLU.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const st = @import("../../core/safetensors.zig");
const wts = @import("weights.zig");
const readAll = @import("weights_fix.zig").readAll;
const Ref = wts.Ref;

// The tower this file reads; `Config.parse` refuses a checkpoint whose vision_config differs.
pub const hidden = 1024;
pub const heads = 16;
pub const head_dim = 64;
pub const depth = 24;
pub const mlp_inter = 4096;
pub const out_hidden = 4096; // the language model's width
pub const merger_inter = 10240;
pub const patch = 14;
pub const temporal = 2;
pub const merge = 2;
pub const patch_width = 3 * temporal * patch * patch; // 1176: a patch's values, channel by frame by row by column
pub const patches_per_token = merge * merge;

/// --vision's limits: image tokens a prompt may hold in all (the tower's scratch and rows are sized for them), and images.
pub const Limits = struct { image_tokens: u32 = 4096, max_images: u32 = 4 };

pub const Config = struct {
    eps: f32 = 1e-5,
    limit: f32 = 10, // swiglu_limit
    image_token: u32,
    image_start: u32,
    image_end: u32,
    mean: [3]f32,
    std: [3]f32,
    min_tokens: u32,
    max_tokens: u32,

    /// config.json's vision_config and image token ids, processor_config.json's image processor; null: no tower.
    pub fn parse(arena: std.mem.Allocator, config_json: []const u8, processor_json: ?[]const u8) !?Config {
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, config_json, .{});
        const vc = doc.object.get("vision_config") orelse return null;
        if (vc != .object) return null;
        const want = .{ .{ "depth", depth }, .{ "hidden_size", hidden }, .{ "num_heads", heads }, .{ "intermediate_size", mlp_inter }, .{ "out_hidden_size", out_hidden }, .{ "projection_intermediate_size", merger_inter }, .{ "patch_size", patch }, .{ "temporal_patch_size", temporal }, .{ "spatial_merge_size", merge }, .{ "in_channels", 3 } };
        inline for (want) |kv| {
            const v = vc.object.get(kv[0]) orelse return error.VisionConfig;
            if (v != .integer or v.integer != kv[1]) {
                std.log.err("glm vision: vision_config.{s} is {f}; this tower reads {d}", .{ kv[0], std.json.fmt(v, .{}), kv[1] });
                return error.VisionConfig;
            }
        }
        const act = vc.object.get("hidden_act") orelse return error.VisionConfig;
        if (act != .string or !std.mem.eql(u8, act.string, "silu")) return error.VisionConfig;
        var c: Config = .{ .image_token = 0, .image_start = 0, .image_end = 0, .mean = .{ 0.48145466, 0.4578275, 0.40821073 }, .std = .{ 0.26862954, 0.26130258, 0.27577711 }, .min_tokens = 16, .max_tokens = 8000 };
        if (vc.object.get("rms_norm_eps")) |v| c.eps = @floatCast(number(v) orelse return error.VisionConfig);
        if (vc.object.get("swiglu_limit")) |v| c.limit = @floatCast(number(v) orelse return error.VisionConfig);
        inline for (.{ .{ "image_token_id", "image_token" }, .{ "image_start_token_id", "image_start" }, .{ "image_end_token_id", "image_end" } }) |kv| {
            const v = doc.object.get(kv[0]) orelse return error.VisionConfig;
            if (v != .integer) return error.VisionConfig;
            @field(c, kv[1]) = @intCast(v.integer);
        }
        const pj = processor_json orelse return c;
        const pdoc = try std.json.parseFromSliceLeaky(std.json.Value, arena, pj, .{});
        const ip = pdoc.object.get("image_processor") orelse return c;
        if (ip != .object) return error.VisionConfig;
        inline for (.{ .{ "patch_size", patch }, .{ "temporal_patch_size", temporal }, .{ "merge_size", merge } }) |kv| {
            if (ip.object.get(kv[0])) |v| if (v != .integer or v.integer != kv[1]) return error.VisionConfig;
        }
        if (ip.object.get("patch_expand_factor")) |v| if (v != .integer or v.integer != 1) return error.VisionConfig;
        inline for (.{ "image_mean", "image_std" }, .{ &c.mean, &c.std }) |key, dst| if (ip.object.get(key)) |v| {
            if (v != .array or v.array.items.len != 3) return error.VisionConfig;
            for (v.array.items, 0..) |x, i| dst[i] = @floatCast(number(x) orelse return error.VisionConfig);
        };
        if (ip.object.get("min_image_tokens")) |v| c.min_tokens = @intCast(v.integer);
        if (ip.object.get("max_image_tokens")) |v| c.max_tokens = @intCast(v.integer);
        return c;
    }

    fn number(v: std.json.Value) ?f64 {
        return switch (v) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => null,
        };
    }
};

const Kernels = struct {
    lib: mtl.Library,
    gemm: [3][2][2]mtl.Pipeline, // [bf16 nt, f32 nt, f32 nn][M % 64 == 0][N % 128 == 0]
    bias: mtl.Pipeline,
    add: mtl.Pipeline,
    rms: mtl.Pipeline,
    layer_norm: mtl.Pipeline,
    qkv: mtl.Pipeline,
    softmax: mtl.Pipeline,
    heads_: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    gelu: mtl.Pipeline,

    fn load(gpa: std.mem.Allocator, device: mtl.Device) !Kernels {
        const src = try frags.source(device, gpa, sources.glm_vision);
        defer gpa.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        var k: Kernels = undefined;
        k.lib = lib;
        const kinds = [_][]const u8{ "bf16_nt", "f32_nt", "f32_nn" };
        for (kinds, 0..) |kind, ki| for (0..2) |m| for (0..2) |n| {
            var name: [64]u8 = undefined;
            const full = try std.fmt.bufPrint(&name, "glmv_gemm_{s}_{d}{d}", .{ kind, m, n });
            k.gemm[ki][m][n] = try mtl.Pipeline.init(device, lib, full, false);
        };
        inline for (.{ .{ "bias", "glmv_bias" }, .{ "add", "glmv_add" }, .{ "rms", "glmv_rms" }, .{ "layer_norm", "glmv_layer_norm" }, .{ "qkv", "glmv_qkv" }, .{ "softmax", "glmv_softmax" }, .{ "heads_", "glmv_heads" }, .{ "swiglu", "glmv_swiglu" }, .{ "gelu", "glmv_gelu" } }) |kv| {
            @field(k, kv[0]) = try mtl.Pipeline.init(device, lib, kv[1], false);
        }
        return k;
    }

    fn deinit(k: *Kernels) void {
        for (&k.gemm) |*a| for (a) |*b| for (b) |p| p.deinit();
        inline for (.{ "bias", "add", "rms", "layer_norm", "qkv", "softmax", "heads_", "swiglu", "gelu" }) |f| @field(k, f).deinit();
        k.lib.deinit();
    }
};

const Block = struct { norm1: Ref, norm2: Ref, qkv_w: Ref, qkv_b: Ref, proj_w: Ref, proj_b: Ref, q_norm: Ref, k_norm: Ref, gate_w: Ref, gate_b: Ref, up_w: Ref, up_b: Ref, down_w: Ref, down_b: Ref };

const Weights = struct {
    buf: mtl.Buffer,
    patch_w: Ref,
    patch_b: Ref,
    blocks: [depth]Block,
    post_norm: Ref,
    down_w: Ref, // [4096, 2, 2, 1024]: the merge's kernel with its taps outermost, as four neighbouring patch rows read it
    down_b: Ref,
    m_proj: Ref,
    m_norm_w: Ref,
    m_norm_b: Ref,
    m_gate: Ref,
    m_up: Ref,
    m_down: Ref,
};

/// The tower: kernels, weights (bf16, about 0.9 GiB) and scratch for `max_tokens` image tokens a request.
pub const Vision = struct {
    gpa: std.mem.Allocator,
    c: Config,
    k: Kernels,
    w: Weights,
    scratch: mtl.Buffer,
    max_patches: u32,
    chunk: u32, // query rows an attention pass scores at once (bounds the fp32 score buffer)
    // scratch views
    patches: Ref, // [P, 1176] bf16
    cos: Ref, // [P, 64] f32
    sin: Ref,
    x: Ref, // [P, 1024] bf16: the residual stream
    xn: Ref, // [P, 1024]: a norm's output; the merger's first [T, 4096]
    att: Ref, // [P, 1024]: heads side by side; the merger's second [T, 4096]
    branch: Ref, // [P, 1024]
    qkv: Ref, // [P, 3072]
    q: Ref, // [16, P, 64] f32
    k_: Ref,
    v: Ref,
    s: Ref, // [16, chunk, P] f32
    o: Ref, // [16, chunk, 64] f32
    g: Ref, // [P, 4096] bf16 (the merger's [T, 10240] fits: T * 10240 < P * 4096)
    u: Ref,
    act: Ref,
    bytes: usize,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, dir: []const u8, c: Config, max_tokens: u32) !*Vision {
        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        v.gpa = gpa;
        v.c = c;
        v.k = try Kernels.load(gpa, device);
        errdefer v.k.deinit();
        v.w = try loadWeights(gpa, device, dir);
        errdefer v.w.buf.deinit();
        const P: usize = @as(usize, max_tokens) * patches_per_token;
        v.max_patches = @intCast(P);
        // the scores take at most 256 MiB: 16 heads of `chunk` rows by P keys in fp32
        v.chunk = @intCast(std.math.clamp((256 << 20) / (heads * P * 4) / 64 * 64, 64, 4096));
        const parts = [_]usize{ P * patch_width * 2, P * 64 * 4, P * 64 * 4, P * hidden * 2, P * hidden * 2, P * hidden * 2, P * hidden * 2, P * 3 * hidden * 2, P * hidden * 4, P * hidden * 4, P * hidden * 4, @as(usize, heads) * v.chunk * P * 4, @as(usize, heads) * v.chunk * head_dim * 4, P * mlp_inter * 2, P * mlp_inter * 2, P * mlp_inter * 2 };
        var total: usize = 0;
        for (parts) |n| total += std.mem.alignForward(usize, n, 256);
        v.scratch = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer v.scratch.deinit();
        var at: usize = 0;
        const views = [_]*Ref{ &v.patches, &v.cos, &v.sin, &v.x, &v.xn, &v.att, &v.branch, &v.qkv, &v.q, &v.k_, &v.v, &v.s, &v.o, &v.g, &v.u, &v.act };
        for (views, parts) |view, n| {
            view.* = .{ .buf = v.scratch, .off = at };
            at += std.mem.alignForward(usize, n, 256);
        }
        v.bytes = total + v.w.buf.length();
        return v;
    }

    pub fn deinit(v: *Vision) void {
        v.scratch.deinit();
        v.w.buf.deinit();
        v.k.deinit();
        v.gpa.destroy(v);
    }

    /// Buffers to keep resident with the engine's.
    pub fn buffers(v: *const Vision) [2]mtl.Buffer {
        return .{ v.w.buf, v.scratch };
    }

    /// One image's patches ([gh * gw, 1176] fp32, the processor's order) into its tokens' rows `out` ([gh * gw / 4,
    /// 4096] bf16). The host writes the scratch: the caller has no work of this tower in flight. `stop`: after the
    /// patch projection (0) or block `stop` (1-24), the residual stream is left in `x` and nothing more runs.
    pub fn encode(v: *Vision, enc: mtl.ComputeEncoder, pixels: []const f32, gh: u32, gw: u32, out: Ref, stop: ?u32) !void {
        const P: u32 = gh * gw;
        if (P == 0 or gh % merge != 0 or gw % merge != 0) return error.ImageGrid;
        if (P > v.max_patches) return error.ImageTooLarge;
        if (pixels.len != @as(usize, P) * patch_width) return error.ImageGrid;
        const T: u32 = P / patches_per_token;
        // host: the patches in bf16 (MLX's astype), and each patch's rotary angles
        const dst: [*]u16 = @ptrCast(@alignCast(v.patches.addr()));
        for (pixels, 0..) |p, i| dst[i] = bf16(p);
        rotary(v, gh, gw);

        v.gemmLinear(enc, v.patches, v.w.patch_w, v.x, P, hidden, patch_width);
        v.biasAdd(enc, v.x, v.w.patch_b, P, hidden);
        if (stop) |s| if (s == 0) return;
        for (v.w.blocks, 0..) |b, i| {
            v.block(enc, b, P);
            if (stop) |s| if (s == i + 1) return;
        }
        v.rms(enc, v.x, v.w.post_norm, v.xn, P, hidden);
        // the merge: four neighbouring patches' rows are one row of xn read as [T, 4096]
        const m0 = v.xn;
        const m1 = v.att;
        v.gemmLinear(enc, m0, v.w.down_w, m1, T, out_hidden, patches_per_token * hidden);
        v.biasAdd(enc, m1, v.w.down_b, T, out_hidden);
        v.gemmLinear(enc, m1, v.w.m_proj, m0, T, out_hidden, out_hidden);
        v.layerNorm(enc, m0, v.w.m_norm_w, v.w.m_norm_b, m1, T, out_hidden);
        enc.setPipeline(v.k.gelu);
        enc.setBuffer(m1.buf, m1.off, 0);
        enc.dispatchThreads(mtl.Size.of(@as(usize, T) * out_hidden, 1, 1), mtl.Size.of(256, 1, 1));
        v.gemmLinear(enc, m1, v.w.m_gate, v.g, T, merger_inter, out_hidden);
        v.gemmLinear(enc, m1, v.w.m_up, v.u, T, merger_inter, out_hidden);
        v.swiglu(enc, @as(usize, T) * merger_inter);
        v.gemmLinear(enc, v.act, v.w.m_down, out, T, out_hidden, merger_inter);
    }

    fn block(v: *Vision, enc: mtl.ComputeEncoder, b: Block, P: u32) void {
        v.rms(enc, v.x, b.norm1, v.xn, P, hidden);
        v.gemmLinear(enc, v.xn, b.qkv_w, v.qkv, P, 3 * hidden, hidden);
        v.biasAdd(enc, v.qkv, b.qkv_b, P, 3 * hidden);
        enc.setPipeline(v.k.qkv);
        const bufs = [_]Ref{ v.qkv, b.q_norm, b.k_norm, v.cos, v.sin, v.q, v.k_, v.v };
        for (bufs, 0..) |r, i| enc.setBuffer(r.buf, r.off, i);
        enc.setValue(extern struct { L: u32, eps: f32 }{ .L = P, .eps = v.c.eps }, 8);
        enc.dispatchGroups(mtl.Size.of(@as(usize, P) * heads * 3, 1, 1), mtl.Size.of(32, 1, 1));
        var q0: u32 = 0;
        while (q0 < P) : (q0 += v.chunk) {
            const lc = @min(v.chunk, P - q0);
            // scores of this chunk's queries against every key, each head a batch
            v.gemm(enc, 1, v.q.at(@as(usize, q0) * head_dim * 4), v.k_, v.s, lc, P, head_dim, head_dim, head_dim, P, heads, @as(usize, P) * head_dim, @as(usize, P) * head_dim, @as(usize, lc) * P);
            enc.setPipeline(v.k.softmax);
            enc.setBuffer(v.s.buf, v.s.off, 0);
            enc.setValue(P, 1);
            enc.setValue(@as(f32, 0.125), 2); // head_dim ** -0.5
            enc.dispatchGroups(mtl.Size.of(@as(usize, heads) * lc, 1, 1), mtl.Size.of(256, 1, 1));
            v.gemm(enc, 2, v.s, v.v, v.o, lc, head_dim, P, P, head_dim, head_dim, heads, @as(usize, lc) * P, @as(usize, P) * head_dim, @as(usize, lc) * head_dim);
            enc.setPipeline(v.k.heads_);
            enc.setBuffer(v.o.buf, v.o.off, 0);
            enc.setBuffer(v.att.buf, v.att.off, 1);
            enc.setValue([2]u32{ lc, q0 }, 2);
            enc.dispatchThreads(mtl.Size.of(hidden, lc, 1), mtl.Size.of(256, 1, 1));
        }
        v.gemmLinear(enc, v.att, b.proj_w, v.branch, P, hidden, hidden);
        v.biasAdd(enc, v.branch, b.proj_b, P, hidden);
        v.add(enc, v.x, v.branch, @as(usize, P) * hidden);
        v.rms(enc, v.x, b.norm2, v.xn, P, hidden);
        v.gemmLinear(enc, v.xn, b.gate_w, v.g, P, mlp_inter, hidden);
        v.biasAdd(enc, v.g, b.gate_b, P, mlp_inter);
        v.gemmLinear(enc, v.xn, b.up_w, v.u, P, mlp_inter, hidden);
        v.biasAdd(enc, v.u, b.up_b, P, mlp_inter);
        v.swiglu(enc, @as(usize, P) * mlp_inter);
        v.gemmLinear(enc, v.act, b.down_w, v.branch, P, hidden, mlp_inter);
        v.biasAdd(enc, v.branch, b.down_b, P, hidden);
        v.add(enc, v.x, v.branch, @as(usize, P) * hidden);
    }

    /// y [M, N] = x [M, K] w^T in bf16 (w as stored, [N, K]).
    fn gemmLinear(v: *Vision, enc: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, M: u32, N: u32, K: u32) void {
        v.gemm(enc, 0, x, w, y, M, N, K, K, K, N, 1, 0, 0, 0);
    }

    /// tfp::gemm_nax: d [batch, M, N] = a x b (kind 0: bf16 a b^T, 1: fp32 a b^T, 2: fp32 a b), batches `s*` elements apart.
    fn gemm(v: *Vision, enc: mtl.ComputeEncoder, kind: usize, a: Ref, b: Ref, d: Ref, M: u32, N: u32, K: u32, lda: u32, ldb: u32, ldd: u32, batch: u32, sa: usize, sb: usize, sd: usize) void {
        const tn = (N + 127) / 128;
        const tm = (M + 63) / 64;
        enc.setPipeline(v.k.gemm[kind][@intFromBool(M % 64 == 0)][@intFromBool(N % 128 == 0)]);
        enc.setBuffer(a.buf, a.off, 0);
        enc.setBuffer(b.buf, b.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, lda, ldb, ldd, tn, tm, 2, K / 256, sa, sb, sd, 0, 0 };
        for (vals, 0..) |x, i| p[i] = @intCast(x);
        enc.setBytes(std.mem.asBytes(&p), 2);
        enc.setBuffer(d.buf, d.off, 3);
        enc.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, @as(usize, batch) * 2), mtl.Size.of(32, 4, 2));
    }

    fn biasAdd(v: *Vision, enc: mtl.ComputeEncoder, y: Ref, b: Ref, rows: u32, width: u32) void {
        enc.setPipeline(v.k.bias);
        enc.setBuffer(y.buf, y.off, 0);
        enc.setBuffer(b.buf, b.off, 1);
        enc.setValue(width, 2);
        enc.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }

    fn add(v: *Vision, enc: mtl.ComputeEncoder, x: Ref, y: Ref, n: usize) void {
        enc.setPipeline(v.k.add);
        enc.setBuffer(x.buf, x.off, 0);
        enc.setBuffer(y.buf, y.off, 1);
        enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn rms(v: *Vision, enc: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, rows: u32, width: u32) void {
        enc.setPipeline(v.k.rms);
        enc.setBuffer(x.buf, x.off, 0);
        enc.setBuffer(w.buf, w.off, 1);
        enc.setBuffer(y.buf, y.off, 2);
        enc.setValue(extern struct { rows: u32, width: u32, eps: f32 }{ .rows = rows, .width = width, .eps = v.c.eps }, 3);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn layerNorm(v: *Vision, enc: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, y: Ref, rows: u32, width: u32) void {
        enc.setPipeline(v.k.layer_norm);
        const bufs = [_]Ref{ x, w, b, y };
        for (bufs, 0..) |r, i| enc.setBuffer(r.buf, r.off, i);
        enc.setValue(extern struct { rows: u32, width: u32, eps: f32 }{ .rows = rows, .width = width, .eps = 1e-5 }, 4);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn swiglu(v: *Vision, enc: mtl.ComputeEncoder, n: usize) void {
        enc.setPipeline(v.k.swiglu);
        enc.setBuffer(v.g.buf, v.g.off, 0);
        enc.setBuffer(v.u.buf, v.u.off, 1);
        enc.setBuffer(v.act.buf, v.act.off, 2);
        enc.setValue(v.c.limit, 3);
        enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
    }
};

/// fp32 to bf16, round to nearest even (MLX's astype).
pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (b & 0x7fffffff > 0x7f800000) return @truncate((b >> 16) | 0x40); // NaN stays NaN
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// Each patch's 2-D rotary angles as vision.py builds them: the patch's row then column in the image's patch grid
/// (patches come in 2x2 merge windows, row-major windows), times the 16 frequencies 10000^(-k/16), cos and sin of
/// [h * f, w * f] twice over a head's 64 values.
fn rotary(v: *Vision, gh: u32, gw: u32) void {
    var inv: [16]f32 = undefined;
    for (&inv, 0..) |*f, k| f.* = 1.0 / @as(f32, @floatCast(std.math.pow(f64, 10000.0, @as(f64, @floatFromInt(2 * k)) / 32.0)));
    const cs: [*]f32 = @ptrCast(@alignCast(v.cos.addr()));
    const sn: [*]f32 = @ptrCast(@alignCast(v.sin.addr()));
    const wx_n = gw / merge;
    var i: usize = 0;
    var wy: u32 = 0;
    while (wy < gh / merge) : (wy += 1) {
        var wx: u32 = 0;
        while (wx < wx_n) : (wx += 1) {
            for (0..merge) |my| for (0..merge) |mx| {
                const h: f32 = @floatFromInt(wy * merge + my);
                const w: f32 = @floatFromInt(wx * merge + mx);
                for (0..32) |j| {
                    const angle: f32 = if (j < 16) h * inv[j] else w * inv[j - 16];
                    const c: f32 = @floatCast(@cos(@as(f64, angle)));
                    const s: f32 = @floatCast(@sin(@as(f64, angle)));
                    cs[i * 64 + j] = c;
                    cs[i * 64 + j + 32] = c;
                    sn[i * 64 + j] = s;
                    sn[i * 64 + j + 32] = s;
                }
                i += 1;
            };
        }
    }
}

/// model.visual.* from the checkpoint's shards into one buffer, the merge kernel's taps moved outermost.
fn loadWeights(gpa: std.mem.Allocator, device: mtl.Device, dir: []const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const Src = struct { path: [:0]const u8, off: u64, len: u64, dtype: st.DType, shape: []const usize };
    var names: std.StringHashMapUnmanaged(Src) = .empty;
    {
        const path = try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors.index.json", .{dir}, 0);
        const f = try mtl.MappedFile.open(path);
        defer f.deinit();
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, f.bytes[0..f.size], .{ .allocate = .alloc_always });
        var files: std.StringArrayHashMapUnmanaged(void) = .empty;
        var it = doc.object.get("weight_map").?.object.iterator();
        while (it.next()) |kv| if (std.mem.startsWith(u8, kv.key_ptr.*, "model.visual.")) try files.put(arena, kv.value_ptr.string, {});
        if (files.count() == 0) return error.NoVisionTower;
        for (files.keys()) |name| {
            const full = try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ dir, name }, 0);
            const fd = std.c.open(full, .{ .ACCMODE = .RDONLY });
            if (fd < 0) return error.OpenFailed;
            defer _ = std.c.close(fd);
            var head: [8]u8 = undefined;
            try readAll(fd, &head, 0);
            const hlen = std.mem.readInt(u64, &head, .little);
            const header = try arena.alloc(u8, hlen);
            try readAll(fd, header, 8);
            const end: u64 = @intCast(std.c.lseek(fd, 0, std.c.SEEK.END));
            // read here, not by core/safetensors.zig: the patch kernel is 5-D, past its max_rank
            const doc2 = try std.json.parseFromSliceLeaky(std.json.Value, arena, header, .{});
            var eit = doc2.object.iterator();
            while (eit.next()) |kv| {
                if (!std.mem.startsWith(u8, kv.key_ptr.*, "model.visual.")) continue;
                const o = kv.value_ptr.object;
                const dtype = st.DType.parse(o.get("dtype").?.string) orelse return error.UnsupportedDType;
                const dims = o.get("shape").?.array.items;
                const shape = try arena.alloc(usize, dims.len);
                var n: u64 = dtype.size();
                for (dims, shape) |d, *s| {
                    s.* = @intCast(d.integer);
                    n *= s.*;
                }
                const offs = o.get("data_offsets").?.array.items;
                const b: u64 = @intCast(offs[0].integer);
                const e: u64 = @intCast(offs[1].integer);
                if (e < b or e - b != n or 8 + hlen + e > end) return error.BadSafetensors;
                try names.put(arena, kv.key_ptr.*, .{ .path = full, .off = 8 + hlen + b, .len = e - b, .dtype = dtype, .shape = shape });
            }
        }
    }
    // every tensor's place in one buffer, 256-aligned
    const Want = struct { name: []const u8, shape: []const usize, ref: *Ref };
    var w: Weights = undefined;
    var wants: std.ArrayList(Want) = .empty;
    try wants.append(arena, .{ .name = "patch_embed.proj.weight", .shape = &.{ hidden, 3, temporal, patch, patch }, .ref = &w.patch_w });
    try wants.append(arena, .{ .name = "patch_embed.proj.bias", .shape = &.{hidden}, .ref = &w.patch_b });
    for (&w.blocks, 0..) |*b, i| {
        const fields = .{ .{ "norm1.weight", &b.norm1, &[_]usize{hidden} }, .{ "norm2.weight", &b.norm2, &[_]usize{hidden} }, .{ "attn.qkv.weight", &b.qkv_w, &[_]usize{ 3 * hidden, hidden } }, .{ "attn.qkv.bias", &b.qkv_b, &[_]usize{3 * hidden} }, .{ "attn.proj.weight", &b.proj_w, &[_]usize{ hidden, hidden } }, .{ "attn.proj.bias", &b.proj_b, &[_]usize{hidden} }, .{ "attn.q_norm.weight", &b.q_norm, &[_]usize{head_dim} }, .{ "attn.k_norm.weight", &b.k_norm, &[_]usize{head_dim} }, .{ "mlp.gate_proj.weight", &b.gate_w, &[_]usize{ mlp_inter, hidden } }, .{ "mlp.gate_proj.bias", &b.gate_b, &[_]usize{mlp_inter} }, .{ "mlp.up_proj.weight", &b.up_w, &[_]usize{ mlp_inter, hidden } }, .{ "mlp.up_proj.bias", &b.up_b, &[_]usize{mlp_inter} }, .{ "mlp.down_proj.weight", &b.down_w, &[_]usize{ hidden, mlp_inter } }, .{ "mlp.down_proj.bias", &b.down_b, &[_]usize{hidden} } };
        inline for (fields) |f| try wants.append(arena, .{ .name = try std.fmt.allocPrint(arena, "blocks.{d}.{s}", .{ i, f[0] }), .shape = f[2], .ref = f[1] });
    }
    try wants.append(arena, .{ .name = "post_layernorm.weight", .shape = &.{hidden}, .ref = &w.post_norm });
    try wants.append(arena, .{ .name = "downsample.weight", .shape = &.{ out_hidden, hidden, merge, merge }, .ref = &w.down_w });
    try wants.append(arena, .{ .name = "downsample.bias", .shape = &.{out_hidden}, .ref = &w.down_b });
    try wants.append(arena, .{ .name = "merger.proj.weight", .shape = &.{ out_hidden, out_hidden }, .ref = &w.m_proj });
    try wants.append(arena, .{ .name = "merger.post_projection_norm.weight", .shape = &.{out_hidden}, .ref = &w.m_norm_w });
    try wants.append(arena, .{ .name = "merger.post_projection_norm.bias", .shape = &.{out_hidden}, .ref = &w.m_norm_b });
    try wants.append(arena, .{ .name = "merger.gate_proj.weight", .shape = &.{ merger_inter, out_hidden }, .ref = &w.m_gate });
    try wants.append(arena, .{ .name = "merger.up_proj.weight", .shape = &.{ merger_inter, out_hidden }, .ref = &w.m_up });
    try wants.append(arena, .{ .name = "merger.down_proj.weight", .shape = &.{ out_hidden, merger_inter }, .ref = &w.m_down });
    const srcs = try arena.alloc(Src, wants.items.len);
    var total: usize = 0;
    for (wants.items, srcs) |want, *s| {
        const full = try std.fmt.allocPrint(arena, "model.visual.{s}", .{want.name});
        s.* = names.get(full) orelse {
            std.log.err("glm vision: the checkpoint has no tensor {s}", .{full});
            return error.MissingTensor;
        };
        if (s.dtype != .bf16 or !std.mem.eql(usize, s.shape, want.shape)) {
            std.log.err("glm vision: {s} is {t} {any}; the tower reads bf16 {any}", .{ full, s.dtype, s.shape, want.shape });
            return error.UnexpectedTensor;
        }
        total = std.mem.alignForward(usize, total, 256) + s.len;
    }
    w.buf = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer w.buf.deinit();
    var at: usize = 0;
    for (wants.items, srcs) |want, s| {
        at = std.mem.alignForward(usize, at, 256);
        want.ref.* = .{ .buf = w.buf, .off = at };
        const fd = std.c.open(s.path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        defer _ = std.c.close(fd);
        _ = std.c.fcntl(fd, 48, @as(c_int, 1)); // F_NOCACHE, as the language model's shards
        const dst = w.buf.contents()[at..][0..s.len];
        try readAll(fd, dst, s.off);
        at += s.len;
    }
    // downsample [O, C, 2, 2] -> [O, 2, 2, C]: a merge window's four patch rows, tap by tap
    const dw: [*]u16 = @ptrCast(@alignCast(w.down_w.addr()));
    const tmp = try gpa.alloc(u16, hidden * merge * merge);
    defer gpa.free(tmp);
    for (0..out_hidden) |o| {
        const row = dw[o * hidden * 4 ..][0 .. hidden * 4];
        @memcpy(tmp, row);
        for (0..hidden) |ch| for (0..4) |tap| {
            row[tap * hidden + ch] = tmp[ch * 4 + tap];
        };
    }
    return w;
}

test "bf16 rounds to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000)))); // a tie to even (down)
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(@bitCast(@as(u32, 0x3f818000)))); // a tie to even (up)
    try std.testing.expectEqual(@as(u16, 0x3f81), bf16(@bitCast(@as(u32, 0x3f808001))));
}
