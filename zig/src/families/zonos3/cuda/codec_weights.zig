//! The codec decoder's published fp32 weights (generator.pt, a head checkpoint); weight norms and code tables folded.
const std = @import("std");
const cuda = @import("cuda");
const pth = @import("core").torch_pth;

pub const latent_dim = 1024;
pub const width = 768; // the semantic head's and the iSTFT head's width, and stages 2..8's
pub const hidden = 2048; // ConvNeXt hidden width
pub const codebook_size = 4096;
pub const codebook_dim = 8;
pub const residuals = 9;
pub const sem_blocks = 4;
pub const head_blocks = 8;
pub const hop = 184; // internal samples a head frame (44160 Hz / 240)
pub const n_ffts = [2]usize{ 736, 2208 };
pub const fir = 127;
pub const resample_in = 736;
pub const resample_out = 735;
pub const resample_taps = 770;

pub const StageShape = struct { in: usize, d: usize, ff: usize, ctx: usize, out: usize };

/// The five transformer stages, 15 Hz to 240 Hz; out 0: the last stage's features feed the head as they are.
pub const stages = [5]StageShape{
    .{ .in = 1024, .d = 1280, .ff = 5120, .ctx = 150, .out = 1280 },
    .{ .in = 640, .d = 768, .ff = 3072, .ctx = 300, .out = 768 },
    .{ .in = 384, .d = 768, .ff = 3072, .ctx = 480, .out = 768 },
    .{ .in = 384, .d = 768, .ff = 3072, .ctx = 480, .out = 768 },
    .{ .in = 384, .d = 768, .ff = 3072, .ctx = 480, .out = 0 },
};

pub const Layer = struct { n1w: u64, n1b: u64, n2w: u64, n2b: u64, qkv: u64, attn_out: u64, ffn0: u64, ffn1: u64, attn_scale: u64, ffn_scale: u64 };
pub const Stage = struct { in_proj: u64, out_proj: u64, layers: []Layer };
pub const Block = struct { dw_w: u64, dw_b: u64, ln_w: u64, ln_b: u64, pw1_w: u64, pw1_b: u64, pw2_w: u64, pw2_b: u64 };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer),
    bytes: usize,
    sem_table: u64, // [4096][width]: the semantic codebook through its output projection
    res_tables: u64, // [9][4096][latent_dim]
    sem: [sem_blocks]Block,
    sem_out_w: u64, // [latent_dim][width]
    sem_out_b: u64,
    stages: [5]Stage,
    head: [head_blocks]Block,
    head_ln_w: u64,
    head_ln_b: u64,
    outs_w: [2]u64, // [n_fft + 2][width]
    outs_b: [2]u64,
    synth: [2]u64, // [n_fft][n_fft + 2]: windowed inverse real FFT of [real | imaginary]
    win2: [2]u64, // [n_fft] window squared
    mixer: u64, // [2][fir]
    resampler: u64, // [resample_out][resample_taps]

    pub fn deinit(w: *Weights) void {
        for (w.bufs.items) |*b| b.free();
        w.bufs.deinit(w.gpa);
        for (w.stages) |st| w.gpa.free(st.layers);
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *Weights,

    fn device(l: *Loader, values: []const f32) !u64 {
        const bytes = std.mem.sliceAsBytes(values);
        var b = try cuda.DeviceBuffer.alloc(l.d, @max(bytes.len, 256));
        errdefer b.free();
        try b.upload(0, bytes);
        try l.w.bufs.append(l.gpa, b);
        l.w.bytes += bytes.len;
        return b.ptr;
    }

    fn one(l: *Loader, a: *const pth.Archive, name: []const u8, shape: []const u64) !u64 {
        const v = try a.readF32(l.gpa, name, shape);
        defer l.gpa.free(v);
        return l.device(v);
    }

    /// A weight-normed 1 x 1 convolution's matrix [out][in]: v * (g / |v|), the norm over each output's inputs.
    fn normed(l: *Loader, a: *const pth.Archive, comptime prefix: []const u8, out: usize, in: usize) ![]f32 {
        const v = try a.readF32(l.gpa, prefix ++ "weight_v", &.{ out, in, 1 });
        errdefer l.gpa.free(v);
        const g = try a.readF32(l.gpa, prefix ++ "weight_g", &.{ out, 1, 1 });
        defer l.gpa.free(g);
        for (0..out) |o| {
            var ss: f64 = 0;
            for (v[o * in ..][0..in]) |x| ss += @as(f64, x) * x;
            const scale = g[o] / @as(f32, @floatCast(@sqrt(ss)));
            for (v[o * in ..][0..in]) |*x| x.* *= scale;
        }
        return v;
    }

    /// A quantizer's table [4096][out]: each code vector through the output projection.
    fn table(l: *Loader, a: *const pth.Archive, comptime prefix: []const u8, out: usize, dst: []f32) !void {
        const m = try l.normed(a, prefix ++ "out_proj.", out, codebook_dim);
        defer l.gpa.free(m);
        const bias = try a.readF32(l.gpa, prefix ++ "out_proj.bias", &.{out});
        defer l.gpa.free(bias);
        const codes = try a.readF32(l.gpa, prefix ++ "codebook.weight", &.{ codebook_size, codebook_dim });
        defer l.gpa.free(codes);
        for (0..codebook_size) |c| for (0..out) |o| {
            var acc: f32 = 0;
            for (m[o * codebook_dim ..][0..codebook_dim], codes[c * codebook_dim ..][0..codebook_dim]) |wv, x| acc = @mulAdd(f32, wv, x, acc);
            dst[c * out + o] = acc + bias[o];
        };
    }

    fn block(l: *Loader, a: *const pth.Archive, comptime f: []const u8, index: usize) !Block {
        var nb: [8][96]u8 = undefined;
        const n = struct {
            fn of(buf: []u8, comptime ff: []const u8, comptime tail: []const u8, i: usize) []const u8 {
                return std.fmt.bufPrint(buf, ff ++ tail, .{i}) catch unreachable;
            }
        }.of;
        return .{
            .dw_w = try l.one(a, n(&nb[0], f, "dwconv.weight", index), &.{ width, 1, 7 }),
            .dw_b = try l.one(a, n(&nb[1], f, "dwconv.bias", index), &.{width}),
            .ln_w = try l.one(a, n(&nb[2], f, "norm.weight", index), &.{width}),
            .ln_b = try l.one(a, n(&nb[3], f, "norm.bias", index), &.{width}),
            .pw1_w = try l.one(a, n(&nb[4], f, "pwconv1.weight", index), &.{ hidden, width }),
            .pw1_b = try l.one(a, n(&nb[5], f, "pwconv1.bias", index), &.{hidden}),
            .pw2_w = try l.one(a, n(&nb[6], f, "pwconv2.weight", index), &.{ width, hidden }),
            .pw2_b = try l.one(a, n(&nb[7], f, "pwconv2.bias", index), &.{width}),
        };
    }
};

/// The windowed inverse real FFT as a matrix [n][n + 2] over [real | imaginary] bins, periodic Hann.
fn synthesis(gpa: std.mem.Allocator, n: usize, win2: []f32) ![]f32 {
    const half = n / 2 + 1;
    const m = try gpa.alloc(f32, n * 2 * half);
    const nf: f64 = @floatFromInt(n);
    for (0..n) |t| {
        const win = 0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(t)) / nf);
        win2[t] = @floatCast(win * win);
        for (0..half) |k| {
            const edge = k == 0 or k == n / 2;
            const gain: f64 = (if (edge) @as(f64, 1.0) else 2.0) * win / nf;
            const angle = 2.0 * std.math.pi * @as(f64, @floatFromInt((k * t) % n)) / nf;
            m[t * 2 * half + k] = @floatCast(gain * @cos(angle));
            m[t * 2 * half + half + k] = if (edge) 0 else @floatCast(-gain * @sin(angle));
        }
    }
    return m;
}

fn fmt(buf: []u8, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, f, args) catch unreachable;
}

/// `generator`: autoencoder/generator.pt; `head`: a decoder's checkpoint (autoencoder/heads/<id>/step_*.pt).
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, generator: []const u8, head: []const u8) !*Weights {
    const w = try gpa.create(Weights);
    errdefer gpa.destroy(w);
    w.* = undefined;
    w.gpa = gpa;
    w.bufs = .empty;
    w.bytes = 0;
    for (&w.stages) |*st| st.layers = &.{};
    errdefer w.deinit();
    var l: Loader = .{ .gpa = gpa, .d = d, .w = w };
    var nb: [128]u8 = undefined;
    {
        var g = try pth.Archive.openNested(gpa, io, generator);
        defer g.close();
        const q = "generator/quantizer.";
        const sem = try gpa.alloc(f32, codebook_size * width);
        defer gpa.free(sem);
        try l.table(&g, q ++ "semantic_quantizer.quantizers.0.", width, sem);
        w.sem_table = try l.device(sem);
        const res = try gpa.alloc(f32, residuals * codebook_size * latent_dim);
        defer gpa.free(res);
        inline for (0..residuals) |i| try l.table(&g, q ++ std.fmt.comptimePrint("quantizer.quantizers.{d}.", .{i}), latent_dim, res[i * codebook_size * latent_dim ..][0 .. codebook_size * latent_dim]);
        w.res_tables = try l.device(res);
        for (&w.sem, 0..) |*b, i| b.* = try l.block(&g, "generator/semantic_head.{d}.", i);
        const out = try l.normed(&g, "generator/semantic_head.4.", latent_dim, width);
        defer gpa.free(out);
        w.sem_out_w = try l.device(out);
        w.sem_out_b = try l.one(&g, "generator/semantic_head.4.bias", &.{latent_dim});
        w.resampler = try l.one(&g, "generator/to_external.kernel", &.{ resample_out, 1, resample_taps });
    }
    var h = try pth.Archive.openNested(gpa, io, head);
    defer h.close();
    for (&w.stages, stages, 0..) |*st, shape, si| {
        const s = 2 * si;
        var count: usize = 0;
        while (h.tensors.contains(fmt(&nb, "trainable/stages.{d}.layers.{d}.qkv.weight", .{ s, count }))) count += 1;
        if (count == 0) return error.MissingTensor;
        st.layers = try gpa.alloc(Layer, count);
        st.in_proj = try l.one(&h, fmt(&nb, "trainable/stages.{d}.input_proj.weight", .{s}), &.{ shape.d, shape.in });
        st.out_proj = if (shape.out == 0) 0 else try l.one(&h, fmt(&nb, "trainable/stages.{d}.output_proj.weight", .{s}), &.{ shape.out, shape.d });
        for (st.layers, 0..) |*L, li| {
            const p = "trainable/stages.{d}.layers.{d}.";
            L.* = .{
                .n1w = try l.one(&h, fmt(&nb, p ++ "norm1.weight", .{ s, li }), &.{shape.d}),
                .n1b = try l.one(&h, fmt(&nb, p ++ "norm1.bias", .{ s, li }), &.{shape.d}),
                .n2w = try l.one(&h, fmt(&nb, p ++ "norm2.weight", .{ s, li }), &.{shape.d}),
                .n2b = try l.one(&h, fmt(&nb, p ++ "norm2.bias", .{ s, li }), &.{shape.d}),
                .qkv = try l.one(&h, fmt(&nb, p ++ "qkv.weight", .{ s, li }), &.{ 3 * shape.d, shape.d }),
                .attn_out = try l.one(&h, fmt(&nb, p ++ "attention_out.weight", .{ s, li }), &.{ shape.d, shape.d }),
                .ffn0 = try l.one(&h, fmt(&nb, p ++ "ffn.0.weight", .{ s, li }), &.{ shape.ff, shape.d }),
                .ffn1 = try l.one(&h, fmt(&nb, p ++ "ffn.3.weight", .{ s, li }), &.{ shape.d, shape.ff }),
                .attn_scale = try l.one(&h, fmt(&nb, p ++ "attention_scale", .{ s, li }), &.{shape.d}),
                .ffn_scale = try l.one(&h, fmt(&nb, p ++ "ffn_scale", .{ s, li }), &.{shape.d}),
            };
        }
    }
    for (&w.head, 0..) |*b, i| b.* = try l.block(&h, "trainable/head.blocks.{d}.", i);
    w.head_ln_w = try l.one(&h, "trainable/head.norm.weight", &.{width});
    w.head_ln_b = try l.one(&h, "trainable/head.norm.bias", &.{width});
    for (n_ffts, 0..) |n, b| {
        w.outs_w[b] = try l.one(&h, fmt(&nb, "trainable/head.outs.{d}.weight", .{b}), &.{ n + 2, width });
        w.outs_b[b] = try l.one(&h, fmt(&nb, "trainable/head.outs.{d}.bias", .{b}), &.{n + 2});
        const win2 = try gpa.alloc(f32, n);
        defer gpa.free(win2);
        const m = try synthesis(gpa, n, win2);
        defer gpa.free(m);
        w.synth[b] = try l.device(m);
        w.win2[b] = try l.device(win2);
    }
    w.mixer = try l.one(&h, "trainable/head.mixer.weight", &.{ 1, 2, fir });
    return w;
}

test "synthesis: a lone DC bin gives the window over n" {
    const t = std.testing;
    var win2: [8]f32 = undefined;
    const m = try synthesis(t.allocator, 8, &win2);
    defer t.allocator.free(m);
    for (0..8) |n| {
        const win = 0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(n)) / 8.0);
        try t.expectApproxEqAbs(@as(f32, @floatCast(win / 8.0)), m[n * 10], 1e-7);
        try t.expectApproxEqAbs(@as(f32, @floatCast(win * win)), win2[n], 1e-7);
        try t.expectEqual(@as(f32, 0), m[n * 10 + 5]);
    }
}
