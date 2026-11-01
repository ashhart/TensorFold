//! Idol's two decoders on Metal: the octree decoder's child probabilities and the Gaussian decoder's anchor rows.
const std = @import("std");
const mtl = @import("metal");
const dc = @import("decoder_config.zig");
const ks = @import("kernels.zig");
const dk = @import("decoder_kernels.zig");
const dw = @import("decoder_weights.zig");
const octree = @import("octree.zig");
const gaussian = @import("gaussian.zig");
const Ref = ks.Ref;

const H = dc.hidden;
const F = dc.ffn;
const C = dc.context_rows;

pub const Timing = struct { wall: f64, gpu: f64 };

/// The scratch's parts for `cap` rows in bytes, in the order of Decoder's views.
fn parts(cap: usize) [17]usize {
    const kv_rows = @max(cap, C);
    return .{
        cap * H * 2, cap * H * 2,                 cap * 3 * H * 2, kv_rows * H * 2,          kv_rows * H * 2,     kv_rows * H * 2,
        cap * H * 2, cap * H * 2,                 cap * F * 2,     cap * H * 4,              cap * H * 4,         cap * dc.raw_width * 4,
        cap * 3 * 4, C * dc.context_channels * 2, C * 2 * H * 2,   dc.max_level * 6 * H * 2, 2 * H * 4 + 256 * 4,
    };
}

/// Scratch bytes for `cap` rows (anchors), each part 256-aligned.
pub fn scratchBytes(cap: usize) usize {
    var total: usize = 0;
    for (parts(cap)) |n| total += std.mem.alignForward(usize, n, 256);
    return total;
}

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: ks.Kernels,
    d: dk.Kernels,
    w: dw.Weights,
    scratch: mtl.Buffer,
    cap: usize,
    // fp16 rows: the stream, a norm's output, packed qkv, heads [16, rows, 64], attention rows, a projection's output
    x: Ref,
    xn: Ref,
    qkv: Ref,
    q: Ref,
    k_: Ref,
    v: Ref,
    att: Ref,
    y: Ref,
    /// the MLP's GELU rows
    u: Ref,
    /// fp32 rows: the stem's sums, then input_layer's (and the head's LayerNorm); the head's outputs
    e32: Ref,
    f32b: Ref,
    out: Ref,
    pts: Ref,
    /// the latent [8192, 16] in fp16, and one block's to_kv rows [8192, 2048]
    ctx: Ref,
    kv: Ref,
    /// each level's modulation [8][6][1024] fp16
    mods: Ref,
    /// fp32 zeros [1024], ones fp16 [1024], then the point embedding's frequencies [170]
    consts: Ref,
    levels_ready: bool = false,

    pub const seconds = mtl.clock.seconds;

    /// The decoders of snapshot folder `dir`, with scratch for up to `max_anchors` anchors.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, max_anchors: usize) !*Decoder {
        if (!device.tensorUnits()) {
            std.log.err("idol: {s} has no tensor units; the decoders run on the M5's alone, the only chip checked", .{device.name()});
            return error.TensorUnitsRequired;
        }
        if (max_anchors == 0 or max_anchors > dc.max_anchors) return error.AnchorCount;
        const t = try gpa.create(Decoder);
        errdefer gpa.destroy(t);
        t.* = undefined;
        t.gpa = gpa;
        t.device = device;
        t.cap = max_anchors;
        t.levels_ready = false;
        t.queue = try device.queue();
        errdefer t.queue.deinit();
        t.k = try ks.Kernels.load(gpa, device);
        errdefer t.k.deinit();
        t.d = try dk.Kernels.load(gpa, device);
        errdefer t.d.deinit();
        t.w = try dw.load(gpa, io, device, dir);
        errdefer t.w.deinit();
        t.scratch = try device.buffer(scratchBytes(max_anchors), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        var at: usize = 0;
        const views = [_]*Ref{ &t.x, &t.xn, &t.qkv, &t.q, &t.k_, &t.v, &t.att, &t.y, &t.u, &t.e32, &t.f32b, &t.out, &t.pts, &t.ctx, &t.kv, &t.mods, &t.consts };
        for (views, parts(max_anchors)) |view, n| {
            view.* = .{ .buf = t.scratch, .off = at };
            at += std.mem.alignForward(usize, n, 256);
        }
        @memset(floats(t.consts, H), 0);
        @memset(halves(t.consts.at(H * 4), H), 1.0);
        octree.pointFreqs(floats(t.freqs(), dc.pe_freqs)[0..dc.pe_freqs]);
        return t;
    }

    pub fn deinit(t: *Decoder) void {
        t.scratch.deinit();
        t.w.deinit();
        t.d.deinit();
        t.k.deinit();
        t.queue.deinit();
        t.gpa.destroy(t);
    }

    fn floats(r: Ref, n: usize) []f32 {
        return @as([*]f32, @ptrCast(@alignCast(r.addr())))[0..n];
    }

    fn halves(r: Ref, n: usize) []f16 {
        return @as([*]f16, @ptrCast(@alignCast(r.addr())))[0..n];
    }

    fn zeros(t: *const Decoder) Ref {
        return t.consts;
    }

    fn ones(t: *const Decoder) Ref {
        return t.consts.at(H * 4);
    }

    fn freqs(t: *const Decoder) Ref {
        return t.consts.at(2 * H * 4);
    }

    /// Level `level`'s (1 to 8) modulation part `part` (0 shift, 1 one plus scale, 2 gate; 3 to 5 the MLP's).
    fn mod(t: *const Decoder, level: usize, part: usize) Ref {
        return t.mods.at(((level - 1) * 6 + part) * H * 2);
    }

    /// The point embedding's frequencies as this decoder computes them.
    pub fn pointFreqs(t: *const Decoder) []const f32 {
        return floats(t.freqs(), dc.pe_freqs);
    }

    /// The activation's fixed inputs from decoder_gs.safetensors and config.json.
    pub fn params(t: *const Decoder) gaussian.Params {
        return .{ .rep = t.w.rep, .base_offset_scale = t.w.base_offset_scale, .perturbation = t.w.perturbation };
    }

    /// Level `level`'s modulation as the blocks read it.
    pub fn levelMods(t: *const Decoder, level: usize) []const f16 {
        return halves(t.mod(level, 0), 6 * H);
    }

    /// The latent both decoders attend to: [8192, 16] fp32, cast to fp16 as the SDK's `cond.type(fp16)`.
    pub fn setLatent(t: *Decoder, latent: []const f32) !void {
        if (latent.len != C * dc.context_channels) return error.LatentRows;
        for (halves(t.ctx, latent.len), latent) |*o, v| o.* = @floatCast(v);
        if (!t.levels_ready) {
            for (1..dc.max_level + 1) |l| octree.levelMods(t.w.level, @intCast(l), @ptrCast(@alignCast(t.mod(l, 0).addr())));
            t.levels_ready = true;
        }
    }

    /// The octree decoder's logits [rows, 8] for parent centres [rows, 3] at `level` (1 to 8).
    pub fn octreeLogits(t: *Decoder, centres: []const f32, level: usize, logits: []f32) !Timing {
        const rows = centres.len / 3;
        if (!t.levels_ready) return error.NoLatent;
        if (rows == 0 or rows > t.cap or centres.len != rows * 3 or logits.len != rows * dc.children or level < 1 or level > dc.max_level) return error.OctreeRows;
        @memcpy(floats(t.pts, centres.len), centres);
        const started = mtl.clock.seconds();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        t.stem(e, t.w.octree_stem, rows);
        for (t.w.octree) |b| t.octreeBlock(e, b, level, rows);
        t.head(e, t.w.octree_stem, rows, dc.children);
        const timing = try finish(cb, e, started);
        @memcpy(logits, floats(t.out, logits.len));
        return timing;
    }

    /// The octree stream [rows, 1024] fp16 after its stem (`block` null), or after block `block` alone run on `input`.
    pub fn octreeRows(t: *Decoder, centres: []const f32, input: []const f16, level: usize, block: ?usize) ![]const f16 {
        const rows = centres.len / 3;
        if (!t.levels_ready) return error.NoLatent;
        if (rows == 0 or rows > t.cap or input.len != rows * H or level < 1 or level > dc.max_level) return error.OctreeRows;
        @memcpy(floats(t.pts, centres.len), centres);
        @memcpy(halves(t.x, input.len), input);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        if (block) |i| t.octreeBlock(e, t.w.octree[i], level, rows) else t.stem(e, t.w.octree_stem, rows);
        _ = try finish(cb, e, mtl.clock.seconds());
        return halves(t.x, rows * H);
    }

    /// One octree block: cross attention to the latent, then the MLP, each modulated by the level.
    fn octreeBlock(t: *const Decoder, e: mtl.ComputeEncoder, b: dw.OctreeBlock, level: usize, rows: usize) void {
        t.k.norm(e, t.x, t.zeros(), t.zeros(), t.mod(level, 1), t.mod(level, 0), t.xn, rows, dc.block_eps, .modulated);
        t.cross(e, b.cross, rows);
        t.k.gateAdd(e, t.x, t.y, t.mod(level, 2), rows);
        t.k.norm(e, t.x, t.zeros(), t.zeros(), t.mod(level, 4), t.mod(level, 3), t.xn, rows, dc.block_eps, .modulated);
        t.mlp(e, b.mlp, rows);
        t.k.gateAdd(e, t.x, t.y, t.mod(level, 5), rows);
    }

    /// The Gaussian decoder's raw rows [anchors, 480] fp32 for anchors [anchors, 3].
    pub fn gaussianRows(t: *Decoder, anchors: []const f32, raw: []f32) !Timing {
        const rows = anchors.len / 3;
        if (!t.levels_ready) return error.NoLatent;
        if (rows == 0 or rows > t.cap or anchors.len != rows * 3 or raw.len != rows * dc.raw_width) return error.AnchorRows;
        @memcpy(floats(t.pts, anchors.len), anchors);
        const started = mtl.clock.seconds();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        t.stem(e, t.w.gaussian_stem, rows);
        for (t.w.gaussian) |b| t.gaussianBlock(e, b, rows);
        t.head(e, t.w.gaussian_stem, rows, dc.raw_width);
        const timing = try finish(cb, e, started);
        @memcpy(raw, floats(t.out, raw.len));
        return timing;
    }

    /// One Gaussian decoder block: self attention over the anchors, cross attention to the latent, then the MLP.
    fn gaussianBlock(t: *const Decoder, e: mtl.ComputeEncoder, b: dw.GaussianBlock, rows: usize) void {
        const k = &t.k;
        t.selfAttention(e, b.self_attn, rows);
        k.norm(e, t.x, b.norm2, b.norm2_b, t.ones(), t.ones(), t.xn, rows, dc.block_eps, .affine);
        t.cross(e, b.cross, rows);
        k.gateAdd(e, t.x, t.y, t.ones(), rows);
        k.norm(e, t.x, t.zeros(), t.zeros(), t.ones(), t.ones(), t.xn, rows, dc.block_eps, .plain);
        t.mlp(e, b.mlp, rows);
        k.gateAdd(e, t.x, t.y, t.ones(), rows);
    }

    /// x += to_out(attention(q_rms, k_rms, v of to_qkv(LayerNorm(x)))) over the anchors.
    fn selfAttention(t: *const Decoder, e: mtl.ComputeEncoder, sa: dw.SelfAttention, rows: usize) void {
        const k = &t.k;
        k.norm(e, t.x, t.zeros(), t.zeros(), t.ones(), t.ones(), t.xn, rows, dc.block_eps, .plain);
        k.linear(e, t.xn, sa.qkv, sa.qkv_b, t.qkv, rows, 3 * H, H, false);
        t.d.heads(e, t.qkv, sa.gq, t.q, rows, 3 * H, 0);
        t.d.heads(e, t.qkv, sa.gk, t.k_, rows, 3 * H, H);
        t.d.heads(e, t.qkv, null, t.v, rows, 3 * H, 2 * H);
        t.d.attention(e, t.q, t.k_, t.v, t.att, rows, rows);
        k.linear(e, t.att, sa.out, sa.out_b, t.y, rows, H, H, false);
        k.gateAdd(e, t.x, t.y, t.ones(), rows);
    }

    /// x [rows, 1024] fp16 = fp16(input_layer(in_proj(p) + pe(p))), all fp32 before the cast.
    fn stem(t: *const Decoder, e: mtl.ComputeEncoder, s: dw.Stem, rows: usize) void {
        t.d.embed(e, t.pts, s.in_w, s.in_b, t.freqs(), t.e32, rows);
        dk.gemm32(&t.k, e, t.e32, s.input, t.f32b, rows, H, H);
        t.d.toHalf(e, t.f32b, s.input_b, t.x, rows, H);
    }

    /// out [rows, n] fp32 = out_proj(F.layer_norm(x)) with the stream cast to fp32.
    fn head(t: *const Decoder, e: mtl.ComputeEncoder, s: dw.Stem, rows: usize, n: usize) void {
        t.k.norm32(e, t.x, t.zeros(), t.zeros(), t.f32b, rows, dc.out_eps, .modulated);
        dk.gemm32(&t.k, e, t.f32b, s.out, t.out, rows, n, H);
        t.k.bias32(e, t.out, s.out_b, rows, n);
    }

    /// y = to_out(attention(q_rms(to_q(xn)), k_rms(to_kv(latent)), v)) over the latent's 8192 rows; fp32 for few rows.
    fn cross(t: *const Decoder, e: mtl.ComputeEncoder, c: dw.Cross, rows: usize) void {
        t.k.linear(e, t.xn, c.q, c.q_b, t.att, rows, H, H, false);
        t.d.heads(e, t.att, c.gq, t.q, rows, H, 0);
        t.d.kv(e, t.ctx, c.kv, c.kv_b, t.kv);
        t.d.heads(e, t.kv, c.gk, t.k_, C, 2 * H, 0);
        t.d.heads(e, t.kv, null, t.v, C, 2 * H, H);
        if (rows <= dk.few_rows) t.d.attentionFew(e, t.q, t.k_, t.v, t.att, rows, C) else t.d.attention(e, t.q, t.k_, t.v, t.att, rows, C);
        t.k.linear(e, t.att, c.out, c.out_b, t.y, rows, H, H, false);
    }

    /// y = down(GELU(up(xn))).
    fn mlp(t: *const Decoder, e: mtl.ComputeEncoder, m: dw.Mlp, rows: usize) void {
        t.k.linear(e, t.xn, m.up, m.up_b, t.u, rows, F, H, true);
        t.k.linear(e, t.u, m.down, m.down_b, t.y, rows, H, F, false);
    }

    fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder, started: f64) !Timing {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("idol: a decoder pass failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        return .{ .wall = mtl.clock.seconds() - started, .gpu = cb.gpuSeconds() };
    }
};

test "decoder scratch at a million Gaussians fits one GPU buffer" {
    try std.testing.expect(scratchBytes(dc.max_anchors) < 1 << 32);
    try std.testing.expect(scratchBytes(1) > C * 2 * H * 2);
}
