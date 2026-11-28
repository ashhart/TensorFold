//! Idol's two decoders on CUDA as decoder.zig runs them on Metal: idol.cu's linears, norms, attention, and our steps.
const std = @import("std");
const cuda = @import("cuda");
const dc = @import("decoder_config.zig");
const ks = @import("cuda_kernels.zig");
const dk = @import("cuda_decoder_kernels.zig");
const dw = @import("cuda_decoder_weights.zig");
const gm = @import("cuda_gemm.zig");
const model = @import("cuda_model.zig");
const octree = @import("octree.zig");
const gaussian = @import("gaussian.zig");

const H = dc.hidden;
const F = dc.ffn;
const C = dc.context_rows;
const scale: f32 = 1.0 / 8.0;

pub const Timing = struct { wall: f64, gpu: f64 };

/// The scratch's parts for `cap` rows in bytes, in the order of Decoder's views.
fn parts(cap: usize) [16]usize {
    return .{
        cap * H * 2, cap * H * 2,                 cap * 3 * H * 2,          cap * H * 2,             C * 2 * H * 2, cap * H * 2,
        cap * H * 2, cap * F * 2,                 cap * F * 4,              cap * H * 4,             cap * H * 4,   cap * dc.raw_width * 4,
        cap * 3 * 4, C * dc.context_channels * 2, dc.max_level * 6 * H * 2, H * 4 + H * 2 + 256 * 4,
    };
}

/// Device scratch bytes for `cap` rows (anchors), each part 256-aligned.
pub fn scratchBytes(cap: usize) usize {
    var total: usize = 0;
    for (parts(cap)) |n| total += std.mem.alignForward(usize, n, 256);
    return total;
}

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    k: ks.Kernels,
    dk: dk.Kernels,
    gemms: gm.Gemms,
    w: *dw.Weights,
    scratch: cuda.DeviceBuffer,
    marks: [2]cuda.Event,
    cap: usize,
    // fp16 rows: stream, norm, self qkv, cross q, a block's to_kv [8192, 2048], attention, projection, GELU rows
    x: u64 = 0,
    xn: u64 = 0,
    qkv: u64 = 0,
    q: u64 = 0,
    kv: u64 = 0,
    att: u64 = 0,
    y: u64 = 0,
    u: u64 = 0,
    /// a block linear's fp32 sums before its bias
    acc: u64 = 0,
    /// fp32 rows: the stem's sums, then input_layer's (and the head's LayerNorm); the head's outputs; the points
    e32: u64 = 0,
    f32b: u64 = 0,
    out: u64 = 0,
    pts: u64 = 0,
    /// the latent [8192, 16] in fp16
    ctx: u64 = 0,
    /// each level's modulation [8][6][1024] fp16
    mods: u64 = 0,
    /// fp32 zeros [1024], fp16 ones [1024], then the point embedding's frequencies [170]
    consts: u64 = 0,
    levels_ready: bool = false,
    mods_host: [dc.max_level]octree.Mods = undefined,
    freqs_host: [dc.pe_freqs]f32 = undefined,
    /// The stream's rows downloaded for `octreeRows`.
    rows_host: []f16 = &.{},

    pub const seconds = model.seconds;

    /// The decoders of snapshot folder `dir` on `ctx`'s GPU, with scratch for up to `max_anchors` anchors.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, max_anchors: usize) !*Decoder {
        const sm = try ctx.capability();
        if (!model.admitted(sm)) {
            var buf: [256]u8 = undefined;
            const name = ctx.name(&buf) catch "this GPU";
            std.log.err("idol: {s} (sm_{d}) is not a GPU this family admits; it runs on sm_120 and sm_121 alone", .{ name, sm });
            return error.UncheckedGpu;
        }
        if (max_anchors == 0 or max_anchors > dc.max_anchors) return error.AnchorCount;
        const d = ctx.d;
        const t = try gpa.create(Decoder);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .d = d, .stream = undefined, .k = undefined, .dk = undefined, .gemms = undefined, .w = undefined, .scratch = undefined, .marks = undefined, .cap = max_anchors };
        t.rows_host = try gpa.alloc(f16, max_anchors * H);
        errdefer gpa.free(t.rows_host);
        t.stream = try cuda.Stream.init(d, false);
        errdefer t.stream.deinit();
        t.k = try ks.Kernels.load(d);
        errdefer t.k.unload();
        t.dk = try dk.Kernels.load(d);
        errdefer t.dk.unload();
        t.gemms = try gm.Gemms.init(gpa, d);
        errdefer t.gemms.deinit();
        t.w = try dw.load(gpa, io, d, dir);
        errdefer {
            t.w.deinit();
            gpa.destroy(t.w);
        }
        t.scratch = try cuda.DeviceBuffer.alloc(d, scratchBytes(max_anchors));
        errdefer t.scratch.free();
        t.marks[0] = try cuda.Event.init(d, true);
        errdefer t.marks[0].deinit();
        t.marks[1] = try cuda.Event.init(d, true);
        errdefer t.marks[1].deinit();
        var at: usize = t.scratch.ptr;
        const views = [_]*u64{ &t.x, &t.xn, &t.qkv, &t.q, &t.kv, &t.att, &t.y, &t.u, &t.acc, &t.e32, &t.f32b, &t.out, &t.pts, &t.ctx, &t.mods, &t.consts };
        for (views, parts(max_anchors)) |view, n| {
            view.* = at;
            at += std.mem.alignForward(usize, n, 256);
        }
        const zero_row: [H]f32 = @splat(0);
        const one_row: [H]f16 = @splat(1);
        octree.pointFreqs(&t.freqs_host);
        try t.upload(t.zeros(), std.mem.sliceAsBytes(&zero_row));
        try t.upload(t.ones(), std.mem.sliceAsBytes(&one_row));
        try t.upload(t.freqs(), std.mem.sliceAsBytes(&t.freqs_host));
        return t;
    }

    pub fn deinit(t: *Decoder) void {
        for (&t.marks) |*m| m.deinit();
        t.scratch.free();
        t.w.deinit();
        t.gpa.destroy(t.w);
        t.gemms.deinit();
        t.dk.unload();
        t.k.unload();
        t.stream.deinit();
        t.gpa.free(t.rows_host);
        t.gpa.destroy(t);
    }

    fn upload(t: *const Decoder, at: u64, bytes: []const u8) !void {
        try t.d.check(t.d.api.cuMemcpyHtoDAsync_v2(at, bytes.ptr, bytes.len, t.stream.handle), "cuMemcpyHtoDAsync");
        try t.stream.synchronize();
    }

    fn download(t: *const Decoder, out: []u8, at: u64) !void {
        try t.d.check(t.d.api.cuMemcpyDtoHAsync_v2(out.ptr, at, out.len, t.stream.handle), "cuMemcpyDtoHAsync");
        try t.stream.synchronize();
    }

    fn zeros(t: *const Decoder) u64 {
        return t.consts;
    }

    fn ones(t: *const Decoder) u64 {
        return t.consts + H * 4;
    }

    fn freqs(t: *const Decoder) u64 {
        return t.consts + H * 4 + H * 2;
    }

    /// Level `level`'s (1 to 8) modulation part `part` (0 shift, 1 one plus scale, 2 gate; 3 to 5 the MLP's).
    fn mod(t: *const Decoder, level: usize, part: usize) u64 {
        return t.mods + ((level - 1) * 6 + part) * H * 2;
    }

    /// The point embedding's frequencies as this decoder computes them.
    pub fn pointFreqs(t: *const Decoder) []const f32 {
        return &t.freqs_host;
    }

    /// The activation's fixed inputs from decoder_gs.safetensors and config.json.
    pub fn params(t: *const Decoder) gaussian.Params {
        return .{ .rep = t.w.rep, .base_offset_scale = t.w.base_offset_scale, .perturbation = t.w.perturbation };
    }

    /// Level `level`'s modulation as the blocks read it.
    pub fn levelMods(t: *const Decoder, level: usize) []const f16 {
        return @as([*]const f16, @ptrCast(&t.mods_host[level - 1]))[0 .. 6 * H];
    }

    /// The latent both decoders attend to: [8192, 16] fp32, cast to fp16 as the SDK's `cond.type(fp16)`.
    pub fn setLatent(t: *Decoder, latent: []const f32) !void {
        if (latent.len != C * dc.context_channels) return error.LatentRows;
        const half = try t.gpa.alloc(f16, latent.len);
        defer t.gpa.free(half);
        for (half, latent) |*o, v| o.* = @floatCast(v);
        try t.upload(t.ctx, std.mem.sliceAsBytes(half));
        if (!t.levels_ready) {
            for (&t.mods_host, 1..) |*m, l| octree.levelMods(t.w.level, @intCast(l), m);
            try t.upload(t.mods, std.mem.sliceAsBytes(&t.mods_host));
            t.levels_ready = true;
        }
    }

    /// The octree decoder's logits [rows, 8] for parent centres [rows, 3] at `level` (1 to 8).
    pub fn octreeLogits(t: *Decoder, centres: []const f32, level: usize, logits: []f32) !Timing {
        const rows = centres.len / 3;
        if (!t.levels_ready) return error.NoLatent;
        if (rows == 0 or rows > t.cap or centres.len != rows * 3 or logits.len != rows * dc.children or level < 1 or level > dc.max_level) return error.OctreeRows;
        const started = seconds();
        try t.upload(t.pts, std.mem.sliceAsBytes(centres));
        try t.marks[0].record(t.stream);
        try t.stem(t.w.octree_stem, rows);
        for (t.w.octree) |b| try t.octreeBlock(b, level, rows);
        try t.head(t.w.octree_stem, rows, dc.children);
        return t.finish(std.mem.sliceAsBytes(logits), started);
    }

    /// The octree stream [rows, 1024] fp16 after its stem (`block` null), or after block `block` alone run on `input`.
    pub fn octreeRows(t: *Decoder, centres: []const f32, input: []const f16, level: usize, block: ?usize) ![]const f16 {
        const rows = centres.len / 3;
        if (!t.levels_ready) return error.NoLatent;
        if (rows == 0 or rows > t.cap or input.len != rows * H or level < 1 or level > dc.max_level) return error.OctreeRows;
        try t.upload(t.pts, std.mem.sliceAsBytes(centres));
        try t.upload(t.x, std.mem.sliceAsBytes(input));
        if (block) |i| try t.octreeBlock(t.w.octree[i], level, rows) else try t.stem(t.w.octree_stem, rows);
        const out = t.rows_host[0 .. rows * H];
        try t.download(std.mem.sliceAsBytes(out), t.x);
        return out;
    }

    /// One octree block: cross attention to the latent, then the MLP, each modulated by the level.
    fn octreeBlock(t: *Decoder, b: dw.OctreeBlock, level: usize, rows: usize) !void {
        const s = t.stream;
        try ks.norm(&t.k, s, t.x, 0, 0, t.mod(level, 1), t.mod(level, 0), t.xn, rows, dc.block_eps, .modulated);
        try t.cross(b.cross, rows);
        try ks.gateAdd(&t.k, s, t.x, t.y, t.mod(level, 2), rows, H);
        try ks.norm(&t.k, s, t.x, 0, 0, t.mod(level, 4), t.mod(level, 3), t.xn, rows, dc.block_eps, .modulated);
        try t.mlp(b.mlp, rows);
        try ks.gateAdd(&t.k, s, t.x, t.y, t.mod(level, 5), rows, H);
    }

    /// The Gaussian decoder's raw rows [anchors, 480] fp32 for anchors [anchors, 3].
    pub fn gaussianRows(t: *Decoder, anchors: []const f32, raw: []f32) !Timing {
        const rows = anchors.len / 3;
        if (!t.levels_ready) return error.NoLatent;
        if (rows == 0 or rows > t.cap or anchors.len != rows * 3 or raw.len != rows * dc.raw_width) return error.AnchorRows;
        const started = seconds();
        try t.upload(t.pts, std.mem.sliceAsBytes(anchors));
        try t.marks[0].record(t.stream);
        try t.stem(t.w.gaussian_stem, rows);
        for (t.w.gaussian) |b| try t.gaussianBlock(b, rows);
        try t.head(t.w.gaussian_stem, rows, dc.raw_width);
        return t.finish(std.mem.sliceAsBytes(raw), started);
    }

    /// One Gaussian decoder block: self attention over the anchors, cross attention to the latent, then the MLP.
    fn gaussianBlock(t: *Decoder, b: dw.GaussianBlock, rows: usize) !void {
        const s = t.stream;
        try t.selfAttention(b.self_attn, rows);
        try ks.norm(&t.k, s, t.x, b.norm2, b.norm2_b, t.ones(), t.ones(), t.xn, rows, dc.block_eps, .affine);
        try t.cross(b.cross, rows);
        try ks.gateAdd(&t.k, s, t.x, t.y, t.ones(), rows, H);
        try ks.norm(&t.k, s, t.x, 0, 0, t.ones(), t.ones(), t.xn, rows, dc.block_eps, .plain);
        try t.mlp(b.mlp, rows);
        try ks.gateAdd(&t.k, s, t.x, t.y, t.ones(), rows, H);
    }

    /// x += to_out(attention(q_rms, k_rms, v of to_qkv(LayerNorm(x)))) over the anchors, by idol.cu's attention.
    fn selfAttention(t: *Decoder, sa: dw.SelfAttention, rows: usize) !void {
        const s = t.stream;
        try ks.norm(&t.k, s, t.x, 0, 0, t.ones(), t.ones(), t.xn, rows, dc.block_eps, .plain);
        try t.linear(t.xn, sa.qkv, sa.qkv_b, t.qkv, rows, 3 * H, H, false);
        try dk.rmsHeads(&t.dk, s, t.qkv, sa.gq, t.qkv, rows, 3 * H, 0);
        try dk.rmsHeads(&t.dk, s, t.qkv, sa.gk, t.qkv, rows, 3 * H, H);
        try ks.attention(&t.k, s, t.qkv, t.att, rows, dc.heads, scale);
        try t.linear(t.att, sa.out, sa.out_b, t.y, rows, H, H, false);
        try ks.gateAdd(&t.k, s, t.x, t.y, t.ones(), rows, H);
    }

    /// x [rows, 1024] fp16 = fp16(input_layer(in_proj(p) + pe(p))), all fp32 before the cast.
    fn stem(t: *Decoder, st: dw.Stem, rows: usize) !void {
        try dk.embed(&t.dk, t.stream, t.pts, st.in_w, st.in_b, t.freqs(), t.e32, rows);
        try t.gemms.run(t.stream, .f32, t.e32, st.input, t.f32b, rows, H, H);
        try dk.toHalf(&t.dk, t.stream, t.f32b, st.input_b, t.x, rows, H);
    }

    /// out [rows, n] fp32 = out_proj(F.layer_norm(x)) with the stream cast to fp32.
    fn head(t: *Decoder, st: dw.Stem, rows: usize, n: usize) !void {
        try ks.norm32(&t.k, t.stream, t.x, t.zeros(), t.zeros(), t.f32b, rows, dc.out_eps, .modulated);
        try t.gemms.run(t.stream, .f32, t.f32b, st.out, t.out, rows, n, H);
        try ks.bias32(&t.k, t.stream, t.out, st.out_b, rows, n);
    }

    /// y = to_out(attention(q_rms(to_q(xn)), k_rms(to_kv(latent)), v)) over the latent's 8192 rows; fp32 for few rows.
    fn cross(t: *Decoder, c: dw.Cross, rows: usize) !void {
        const s = t.stream;
        try t.linear(t.xn, c.q, c.q_b, t.q, rows, H, H, false);
        try dk.rmsHeads(&t.dk, s, t.q, c.gq, t.q, rows, H, 0);
        try dk.kv(&t.dk, s, t.ctx, c.kv, c.kv_b, t.kv, C);
        try dk.rmsHeads(&t.dk, s, t.kv, c.gk, t.kv, C, 2 * H, 0);
        try dk.attention(&t.dk, s, t.q, H, t.kv, 2 * H, t.att, rows, C, scale);
        try t.linear(t.att, c.out, c.out_b, t.y, rows, H, H, false);
    }

    /// y = down(GELU(up(xn))).
    fn mlp(t: *Decoder, m: dw.Mlp, rows: usize) !void {
        try t.linear(t.xn, m.up, m.up_b, t.u, rows, F, H, true);
        try t.linear(t.u, m.down, m.down_b, t.y, rows, H, F, false);
    }

    /// y fp16 [rows, n] = fp16(x w^T + b) from fp32 sums, then fp16(GELU) when `gelu`.
    fn linear(t: *Decoder, x: u64, w: u64, b: u64, y: u64, rows: usize, n: usize, k: usize, gelu: bool) !void {
        try t.gemms.run(t.stream, .f16, x, w, t.acc, rows, n, k);
        try ks.bias(&t.k, t.stream, t.acc, b, y, rows, n, gelu);
    }

    /// The pass's end marked, its fp32 outputs downloaded, and its wall and GPU time.
    fn finish(t: *Decoder, out: []u8, started: f64) !Timing {
        try t.marks[1].record(t.stream);
        try t.download(out, t.out);
        const gpu = @as(f64, try cuda.Event.elapsedMs(t.marks[0], t.marks[1])) / 1000;
        return .{ .wall = seconds() - started, .gpu = gpu };
    }
};

test "decoder scratch: a million Gaussians' anchors fit, to_kv's rows beside the fewest, c_int indices" {
    try std.testing.expect(scratchBytes(dc.max_anchors) < 2 << 30);
    try std.testing.expect(scratchBytes(1) > C * 2 * H * 2);
    // the kernels and cuBLASLt index a block's widest rows as c_int
    try std.testing.expect(dc.max_anchors * F < std.math.maxInt(c_int));
    try std.testing.expect(parts(1).len == 16);
}
