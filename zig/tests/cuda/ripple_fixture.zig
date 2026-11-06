//! Ripple's forward test target: the release's widths on 10 layers, random weights in the loader's layout.
const std = @import("std");
const cuda = @import("cuda");
const rc = @import("ricursive_cuda");
const check = @import("check.zig");

const wts = rc.weights;
const Config = @FieldType(wts.Weights, "config");
const Weight = cuda.qmmf.Weight;

/// The release's attention and widths, so every lane matmul takes the release's plan (tile, K slices, cluster).
pub fn config() Config {
    var c: Config = .{ .hidden = 5120, .intermediate = 12288, .layers = 10, .heads = 80, .kv_heads = 8, .vocab = 16384, .eps = 1e-6, .theta = 10000, .window = 1024, .embed_scale = 71.55417527999327, .clamp = 20, .context = 8192, .fp4_mlp = true };
    for (0..c.layers) |i| c.global[i] = i % 5 == 4;
    return c;
}

/// bf16 bits of x, rounded to nearest even.
fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

const Build = struct {
    w: *wts.Weights,
    d: *const cuda.Driver,
    r: std.Random,

    fn put(b: Build, bytes: []const u8) !u64 {
        var buf = try cuda.DeviceBuffer.fromHost(b.d, bytes);
        errdefer buf.free();
        try b.w.buffers.append(b.w.gpa, buf);
        b.w.bytes += bytes.len;
        return buf.ptr;
    }

    /// `n` bf16 values of magnitude in [2^e, 2^(e+1)), random signs and mantissas (fast: the release's sizes).
    fn values(b: Build, n: usize, e: i32) ![]u16 {
        const v = try b.w.gpa.alloc(u16, n);
        const exp: u16 = @intCast((e + 127) << 7);
        var bits: u64 = 0;
        for (v, 0..) |*x, i| {
            if (i % 4 == 0) bits = b.r.int(u64);
            x.* = exp | @as(u16, @truncate(bits & 0x807f));
            bits >>= 16;
        }
        return v;
    }

    /// A norm's or the embedding's bf16 values, normal with deviation `sd` around `mean`.
    fn small(b: Build, n: usize, mean: f32, sd: f32) !u64 {
        const v = try b.w.gpa.alloc(u16, n);
        defer b.w.gpa.free(v);
        for (v) |*x| x.* = bf16(mean + sd * b.r.floatNorm(f32));
        return b.put(std.mem.sliceAsBytes(v));
    }

    /// A bf16 [n, k] projection packed for the lane and prompt matmuls, its values about 1 / sqrt(k).
    fn dense(b: Build, n: usize, k: usize) !Weight {
        const src = try b.values(n * k, -@as(i32, std.math.log2_int(usize, k) / 2) - 1);
        defer b.w.gpa.free(src);
        const out = try b.w.gpa.alloc(u16, cuda.modelopt.padded(n) * k);
        defer b.w.gpa.free(out);
        cuda.qmmf.packBf16(out, src, n, k);
        return cuda.modelopt.bf16Weight(try b.put(std.mem.sliceAsBytes(out)), n, k);
    }

    /// An NVFP4 [n, k] projection: random nibbles, E4M3 scales of 1 to 3, a global of 1/1024 (about 1 / sqrt(k)).
    fn fp4(b: Build, n: usize, k: usize) !Weight {
        const gpa = b.w.gpa;
        const codes = try gpa.alloc(u8, n * k / 2);
        defer gpa.free(codes);
        b.r.bytes(codes);
        const scales = try gpa.alloc(u8, n * k / 16);
        defer gpa.free(scales);
        const e4m3 = [_]u8{ 0x38, 0x3c, 0x40, 0x44 };
        b.r.bytes(scales);
        for (scales) |*x| x.* = e4m3[x.* & 3];
        const words = try gpa.alloc(u32, cuda.nvfp4.padded(n) / 64 * (k / 64) * 512);
        defer gpa.free(words);
        const tiles = try gpa.alloc(u8, cuda.nvfp4.padded(n) * (k / 16));
        defer gpa.free(tiles);
        cuda.nvfp4.packWords(words, codes, n, k);
        cuda.nvfp4.packScales(tiles, scales, n, k);
        return cuda.nvfp4.weight(try b.put(std.mem.sliceAsBytes(words)), try b.put(tiles), 1.0 / 1024.0, n, k);
    }
};

/// Random weights in the loader's layout (cuda_weights.load): q, k and g one projection, NVFP4 MLPs.
pub fn build(gpu: check.Gpu, c: Config, seed: u64) !wts.Weights {
    var w: wts.Weights = .{ .gpa = gpu.gpa, .config = c };
    errdefer w.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    const b: Build = .{ .w = &w, .d = gpu.d, .r = prng.random() };
    const h = c.hidden;
    const q = c.queryWidth();
    const kv = c.kvWidth();
    w.embed = try b.small(c.vocab * h, 0, 0.02);
    w.head = try b.dense(c.vocab, h);
    w.norm = try b.small(h, 1, 0.1);
    w.layers = try gpu.gpa.alloc(wts.Layer, c.layers);
    for (w.layers) |*l| l.* = .{
        .qkg = try b.dense(2 * q + kv, h),
        .v = try b.dense(kv, h),
        .o = try b.dense(h, q),
        .gate = try b.fp4(c.intermediate, h),
        .up = try b.fp4(c.intermediate, h),
        .down = try b.fp4(h, c.intermediate),
        .input_norm = try b.small(h, 1, 0.1),
        .post_attn = try b.small(h, 1, 0.1),
        .pre_ffn = try b.small(h, 1, 0.1),
        .post_ffn = try b.small(h, 1, 0.1),
        .q_norm = try b.small(128, 1, 0.1),
        .k_norm = try b.small(128, 1, 0.1),
        .blend = wts.blendOf(bf16(b.r.float(f32) * 2 - 1)),
    };
    return w;
}
