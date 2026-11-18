//! The release's speaker encoder (MECT-B2) in fp32 on the host: 44.1 kHz audio to the speaker row's embedding.
const std = @import("std");
const pth = @import("core").torch_pth;
const features = @import("features.zig");
const blocks = @import("speaker_blocks.zig");

const A = std.mem.Allocator;
const Map = blocks.Map;
const Affine = blocks.Affine;
const Dense = blocks.Dense;

pub const dim = 192;
/// What the encoder reads of a clip: the first 60 s, and at least a filterbank frame.
pub const max_samples = 60 * features.source_rate;
const hidden = 64;
const mel = features.mel_bins;
const width = mel * 32; // a stage's bins * channels

const Res = struct { conv1: []f32, pw1: Dense, bn1: Affine, conv2: []f32, pw2: Dense, bn2: Affine, short: ?[]f32, short_bn: Affine, cin: usize, cout: usize, groups: usize, stride: [2]usize };

const Block = struct {
    res: []Res,
    reduce: ?Dense,
    reduce_bn: Affine,
    project: Dense,
    norm: Affine,
    pos: []f32, // depthwise taps [hidden][113]
    pos_bias: []f32,
    pos_bn: Affine,
    pos_out: Dense,
    layer: blocks.Layer,
    expand: Dense,
    stride: [2]usize,
    channels: usize, // of the map it returns
};

pub const Encoder = struct {
    arena: std.heap.ArenaAllocator,
    head: []f32, // [32][9]
    head_bias: []f32,
    head_bn: Affine,
    stages: [4][]Block,
    mix: []f32, // [4][width], softmaxed over the stages
    pool1: Dense,
    pool2: Dense,
    bn: Affine,
    out: Dense,

    pub fn deinit(e: *Encoder) void {
        var arena = e.arena;
        arena.deinit();
    }

    /// The embedding of 44.1 kHz mono audio; `taps`, when given, sees the stages by the reference dump's names.
    pub fn embed(e: *const Encoder, gpa: A, audio: []const f32, taps: ?blocks.Taps) ![dim]f32 {
        var pool: blocks.Pool = .init(gpa);
        defer pool.deinit();
        const p = &pool;
        const a = pool.arena.allocator();
        const wav = try features.resample(a, audio[0..@min(audio.len, max_samples)]);
        const fbank = try features.logMel(a, wav, true);
        const t = fbank.len / mel;
        if (taps) |tp| try tp.f(tp.ctx, "fbank", fbank, t, mel);
        // cepstral mean normalisation: each bin less its mean over the frames
        const centred = try Map.init(p, 1, mel, t);
        for (0..mel) |m| {
            var mean: f32 = 0;
            for (0..t) |f| mean += fbank[f * mel + m];
            mean /= @floatFromInt(t);
            for (centred.row(0, m), 0..) |*v, f| v.* = fbank[f * mel + m] - mean;
        }
        var x = try blocks.conv3(p, centred, e.head, e.head_bias, 32, 1, .{ 1, 1 });
        p.give(centred);
        blocks.affine(x, e.head_bn, false);
        if (taps) |tp| try tp.map("head", x, p);
        const mixed = try a.alloc(f32, width * t);
        @memset(mixed, 0);
        for (e.stages, 0..) |stage, s| {
            for (stage) |*b| {
                const next = try block(p, b, x);
                p.give(x);
                x = next;
            }
            var name: [8]u8 = undefined;
            if (taps) |tp| try tp.map(std.fmt.bufPrint(&name, "stage_{d}", .{s + 1}) catch unreachable, x, p);
            for (0..x.f) |fi| for (0..x.c) |ci| {
                const w = e.mix[s * width + fi * x.c + ci];
                for (mixed[(fi * x.c + ci) * t ..][0..t], x.row(ci, fi)) |*o, v| o.* += w * v;
            };
        }
        if (taps) |tp| try tp.f(tp.ctx, "mixed", mixed, width, t);
        // attentive statistics pooling: a weight a bin and frame, then the weighted mean and deviation
        const seq: Map = .{ .c = width, .f = 1, .t = t, .data = mixed };
        const alpha = try blocks.dense(p, seq, e.pool1);
        for (alpha.data) |*v| v.* = std.math.tanh(v.*);
        const logits = try blocks.dense(p, alpha, e.pool2);
        const pooled = try a.alloc(f32, 2 * width);
        for (0..width) |r| {
            const l = logits.row(r, 0);
            var top: f32 = -std.math.inf(f32);
            for (l) |v| top = @max(top, v);
            var sum: f32 = 0;
            for (l) |*v| {
                v.* = @exp(v.* - top);
                sum += v.*;
            }
            var mean: f32 = 0;
            var square: f32 = 0;
            for (l, mixed[r * t ..][0..t]) |weight, v| {
                mean += weight / sum * v;
                square += weight / sum * (v * v);
            }
            pooled[r] = mean;
            pooled[width + r] = @sqrt(@max(square - mean * mean, 1e-7));
        }
        if (taps) |tp| try tp.f(tp.ctx, "pooled", pooled, 1, 2 * width);
        for (pooled, e.bn.scale, e.bn.shift) |*v, sc, sh| v.* = v.* * sc + sh;
        var out: [dim]f32 = undefined;
        for (&out, 0..) |*o, n| {
            var acc: f32 = 0;
            for (pooled, e.out.w[n * 2 * width ..][0 .. 2 * width]) |v, w| acc = @mulAdd(f32, w, v, acc);
            o.* = acc;
        }
        return out;
    }
};

/// BasicResBlock.forward; `x` stays the caller's.
fn residual(p: *blocks.Pool, r: *const Res, x: Map) !Map {
    const a = try blocks.conv3(p, x, r.conv1, null, r.cout, r.groups, r.stride);
    const b = try blocks.denseNorm(p, a, r.pw1, r.bn1, true);
    p.give(a);
    const c = try blocks.conv3(p, b, r.conv2, null, r.cout, r.groups, .{ 1, 1 });
    p.give(b);
    const y = try blocks.denseNorm(p, c, r.pw2, r.bn2, true);
    p.give(c);
    if (r.short) |w| {
        const cut = try blocks.strided(p, x, r.stride);
        const s = try blocks.denseNorm(p, cut, .{ .w = w, .b = null, .n = r.cout, .k = r.cin }, r.short_bn, false);
        if (cut.data.ptr != x.data.ptr) p.give(cut);
        for (y.data, s.data) |*o, v| o.* += v;
        p.give(s);
    } else {
        for (y.data, x.data) |*o, v| o.* += v;
    }
    return y;
}

/// MECT_block.forward: the convolutions, then a transformer layer over the frames, added back to the block's input.
fn block(p: *blocks.Pool, b: *const Block, x: Map) !Map {
    var y = x;
    for (b.res) |*r| {
        const next = try residual(p, r, y);
        if (y.data.ptr != x.data.ptr) p.give(y);
        y = next;
    }
    if (b.reduce) |d| {
        const next = try blocks.denseNorm(p, y, d, b.reduce_bn, false);
        p.give(y);
        y = next;
    }
    const c = y.c;
    const f = y.f;
    const x2 = try blocks.binMajor(p, y);
    p.give(y);
    const s = try blocks.dense(p, x2, b.project);
    blocks.channelNorm(s, b.norm);
    // the positional convolution: 113 depthwise taps around each frame, batch norm, GELU, a mixing layer
    const pos = try blocks.depthwise(p, s, b.pos, b.pos_bias, 113);
    blocks.affine(pos, b.pos_bn, false);
    for (pos.data) |*v| v.* = blocks.gelu(v.*);
    const mixed = try blocks.dense(p, pos, b.pos_out);
    p.give(pos);
    for (s.data, mixed.data) |*o, v| o.* += v;
    p.give(mixed);
    try blocks.transformer(p, &b.layer, s);
    const out = try blocks.dense(p, s, b.expand);
    p.give(s);
    for (out.data, x2.data) |*o, v| o.* += v;
    p.give(x2);
    // back to the input's frames (nearest), plus the input itself, bin-major
    const result = try Map.init(p, c, f, x.t);
    const st = b.stride[1];
    for (0..f) |fi| for (0..c) |ci| {
        const r = fi * c + ci;
        // the input's row r of its own bin-major order
        const back = x.row(r % x.c, r / x.c);
        for (result.row(ci, fi), back, 0..) |*o, v, j| o.* = out.row(r, 0)[j / st] + v;
    };
    p.give(out);
    return result;
}

const Loader = struct {
    a: A,
    arc: *const pth.Archive,
    nb: [256]u8 = undefined,

    fn name(l: *Loader, comptime f: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&l.nb, "modules/spk_model/model." ++ f, args) catch unreachable;
    }

    fn get(l: *Loader, key: []const u8, n: usize) ![]f32 {
        const t = try l.arc.tensor(key);
        var total: u64 = 1;
        for (t.shape) |s| total *= s;
        if (total != n) return error.ShapeMismatch;
        const out = try l.arc.readF32Strided(l.a, key);
        for (out) |v| if (!std.math.isFinite(v)) return error.BadSpeakerEncoder;
        return out;
    }

    fn dense(l: *Loader, comptime f: []const u8, args: anytype, n: usize, k: usize, bias: bool) !Dense {
        const w = try l.get(l.name(f ++ ".weight", args), n * k);
        return .{ .w = w, .b = if (bias) try l.get(l.name(f ++ ".bias", args), n) else null, .n = n, .k = k };
    }

    /// A batch norm as the scale and shift it applies: (x - mean) / sqrt(var + 1e-5) * weight + bias.
    fn norm(l: *Loader, comptime f: []const u8, args: anytype, n: usize) !Affine {
        const w = try l.get(l.name(f ++ ".weight", args), n);
        const b = try l.get(l.name(f ++ ".bias", args), n);
        const mean = try l.get(l.name(f ++ ".running_mean", args), n);
        const variance = try l.get(l.name(f ++ ".running_var", args), n);
        for (w, b, mean, variance) |*sc, *sh, m, v| {
            sc.* = sc.* / @sqrt(v + 1e-5);
            sh.* = sh.* - m * sc.*;
        }
        return .{ .scale = w, .shift = b };
    }

    fn plain(l: *Loader, comptime f: []const u8, args: anytype, n: usize) !Affine {
        return .{ .scale = try l.get(l.name(f ++ ".weight", args), n), .shift = try l.get(l.name(f ++ ".bias", args), n) };
    }
};

/// A stage's blocks: (convolutions, output channels, (bin stride, frame stride), expansion) each.
const Spec = struct { n: usize, out: usize, stride: [2]usize, expansion: usize };
fn spec(n: usize, out: usize, bins: usize, frames: usize, expansion: usize) Spec {
    return .{ .n = n, .out = out, .stride = .{ bins, frames }, .expansion = expansion };
}
const stages = [4][]const Spec{
    &.{ spec(3, 32, 1, 1, 4), spec(3, 32, 1, 1, 4) },
    &.{ spec(3, 64, 2, 1, 2), spec(3, 64, 1, 2, 2), spec(3, 64, 1, 2, 2), spec(3, 64, 1, 2, 2) },
    &.{ spec(3, 128, 2, 1, 2), spec(3, 128, 1, 2, 2), spec(3, 128, 1, 2, 2), spec(3, 128, 1, 2, 2), spec(3, 128, 1, 2, 2) },
    &.{ spec(3, 256, 2, 1, 1), spec(2, 256, 1, 2, 1), spec(2, 256, 1, 2, 1) },
};

/// The encoder from the release's speaker_encoders/ownmect_s2 checkpoint.
pub fn load(gpa: A, io: std.Io, path: []const u8) !*Encoder {
    var archive = try pth.Archive.openNested(gpa, io, path);
    defer archive.close();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const e = try a.create(Encoder);
    var l: Loader = .{ .a = a, .arc = &archive };
    const fe = "feature_extractor.";
    e.head = try l.get(l.name(fe ++ "head.0.weight", .{}), 32 * 9);
    e.head_bias = try l.get(l.name(fe ++ "head.0.bias", .{}), 32);
    e.head_bn = try l.norm(fe ++ "head.1", .{}, 32);
    var cin: usize = 32;
    var bins: usize = mel;
    for (&e.stages, stages, 1..) |*made, specs, si| {
        made.* = try a.alloc(Block, specs.len);
        for (made.*, specs, 0..) |*b, sp, bi| {
            const pre = fe ++ "mect_block{d}.{d}.";
            const wide = sp.out * sp.expansion;
            b.res = try a.alloc(Res, sp.n);
            var from = cin;
            for (b.res, 0..) |*r, ri| {
                const stride: [2]usize = if (ri == 0) sp.stride else .{ 1, 1 };
                const rp = pre ++ "cnn_block.{d}.";
                const short = stride[0] != 1 or stride[1] != 1 or from != wide;
                r.* = .{
                    .conv1 = try l.get(l.name(rp ++ "conv1.weight", .{ si, bi, ri }), wide * 9),
                    .pw1 = try l.dense(rp ++ "conv_pw1", .{ si, bi, ri }, wide, wide, true),
                    .bn1 = try l.norm(rp ++ "bn1", .{ si, bi, ri }, wide),
                    .conv2 = try l.get(l.name(rp ++ "conv2.weight", .{ si, bi, ri }), wide * (wide / from) * 9),
                    .pw2 = try l.dense(rp ++ "conv_pw2", .{ si, bi, ri }, wide, wide, true),
                    .bn2 = try l.norm(rp ++ "bn2", .{ si, bi, ri }, wide),
                    .short = if (short) try l.get(l.name(rp ++ "shortcut.0.weight", .{ si, bi, ri }), wide * from) else null,
                    .short_bn = if (short) try l.norm(rp ++ "shortcut.1", .{ si, bi, ri }, wide) else .{ .scale = &.{}, .shift = &.{} },
                    .cin = from,
                    .cout = wide,
                    .groups = from,
                    .stride = stride,
                };
                from = wide;
            }
            b.stride = sp.stride;
            b.channels = if (sp.expansion > 1) cin * sp.stride[0] else sp.out;
            b.reduce = if (sp.expansion > 1) try l.dense(pre ++ "reduce_conv.0", .{ si, bi }, b.channels, wide, true) else null;
            b.reduce_bn = if (sp.expansion > 1) try l.norm(pre ++ "reduce_conv.1", .{ si, bi }, b.channels) else .{ .scale = &.{}, .shift = &.{} };
            bins /= sp.stride[0];
            if (bins * b.channels != width) return error.BadSpeakerEncoder;
            b.project = try l.dense(pre ++ "reduce_project.0", .{ si, bi }, hidden, width, true);
            b.norm = try l.plain(pre ++ "reduce_project.1", .{ si, bi }, hidden);
            b.pos = try l.get(l.name(pre ++ "pos_conv.pos_conv.0.weight", .{ si, bi }), hidden * 113);
            b.pos_bias = try l.get(l.name(pre ++ "pos_conv.pos_conv.0.bias", .{ si, bi }), hidden);
            b.pos_bn = try l.norm(pre ++ "pos_conv.pos_conv.1", .{ si, bi }, hidden);
            b.pos_out = try l.dense(pre ++ "pos_conv.pos_conv.3", .{ si, bi }, hidden, hidden, true);
            const tp = pre ++ "transformer.";
            b.layer = .{
                .q = try l.dense(tp ++ "attention.q_proj", .{ si, bi }, hidden, hidden, true),
                .k = try l.dense(tp ++ "attention.k_proj", .{ si, bi }, hidden, hidden, true),
                .v = try l.dense(tp ++ "attention.v_proj", .{ si, bi }, hidden, hidden, true),
                .o = try l.dense(tp ++ "attention.out_proj", .{ si, bi }, hidden, hidden, true),
                .norm1 = try l.plain(tp ++ "layer_norm", .{ si, bi }, hidden),
                .gate = try l.dense(tp ++ "feed_forward.intermediate_dense.token_moe.gate", .{ si, bi }, blocks.experts, hidden, false),
                .experts = try l.dense(tp ++ "feed_forward.intermediate_dense.token_moe.experts", .{ si, bi }, blocks.experts * hidden, hidden, false),
                .down = try l.dense(tp ++ "feed_forward.output_dense", .{ si, bi }, hidden, hidden, true),
                .norm2 = try l.plain(tp ++ "final_layer_norm", .{ si, bi }, hidden),
            };
            b.expand = try l.dense(pre ++ "expansion_project.0", .{ si, bi }, width, hidden, true);
            cin = b.channels;
        }
    }
    // DenseWeight: the four stages' rows weighed by a softmax over the stages
    e.mix = try l.get(l.name(fe ++ "dense3.inputs_weights", .{}), 4 * width);
    for (0..width) |r| {
        var top: f32 = -std.math.inf(f32);
        for (0..4) |s| top = @max(top, e.mix[s * width + r]);
        var sum: f32 = 0;
        for (0..4) |s| {
            e.mix[s * width + r] = @exp(e.mix[s * width + r] - top);
            sum += e.mix[s * width + r];
        }
        for (0..4) |s| e.mix[s * width + r] /= sum;
    }
    e.pool1 = try l.dense("encoding_layers.linear1", .{}, 128, width, true);
    e.pool2 = try l.dense("encoding_layers.linear2", .{}, width, 128, true);
    e.bn = try l.norm("bn", .{}, 2 * width);
    e.out = try l.dense("linear", .{}, dim, 2 * width, false);
    e.arena = arena;
    return e;
}
