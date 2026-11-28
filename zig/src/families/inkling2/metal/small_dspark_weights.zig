//! Validated Small DSpark mtp.safetensors weights (5 layers, 8 taps, H=4608) and binary64 output fold.
const std = @import("std");
const mtl = @import("metal");
const bf = @import("bf16_round");
const ck = @import("../../../core/checkpoint_metal.zig");
const cfg = @import("../config.zig");
const Model = @import("small_model.zig").Model;

pub const hidden: usize = 4608;
pub const query: usize = 6144;
pub const kv: usize = 1024;
pub const intermediate: usize = 18432;
pub const rank: usize = 256;
pub const layers_count: usize = 5;
pub const taps_count: usize = 8;
pub const fusions_count: usize = 6;
pub const vocabulary: usize = 201024;

pub const Layer = struct {
    dflare: ck.Tensor,
    input_norm: ck.Tensor,
    post_norm: ck.Tensor,
    q_norm: ck.Tensor,
    k_norm: ck.Tensor,
    k_norm_ctx: ck.Tensor,
    q: ck.Tensor,
    k: ck.Tensor,
    v: ck.Tensor,
    k_ctx: ck.Tensor,
    v_ctx: ck.Tensor,
    o: ck.Tensor,
    gate: ck.Tensor,
    up: ck.Tensor,
    down: ck.Tensor,
};

pub const Weights = struct {
    fuse: ck.Tensor,
    mask: ck.Tensor,
    norm: ck.Tensor,
    markov_norm: ck.Tensor,
    hidden_out: ck.Tensor,
    hidden_bias: ck.Tensor,
    markov_w1: ck.Tensor,
    rnn: ck.Tensor,
    rnn_bias: ck.Tensor,
    layers: [layers_count]Layer,
};

pub const Loaded = struct {
    gpa: std.mem.Allocator,
    checkpoint: ck.Checkpoint,
    weights: Weights,
    vocab: usize,
    dflare: mtl.Buffer, // 6 H-wide rows: 5 layers, then the Markov head
    phi: mtl.Buffer, // [256, vocab] BF16, folded once in binary64
    beta: mtl.Buffer, // [vocab] BF16
    rnn_kappa: mtl.Buffer, // [9216, 768] BF16: y then newest_f columns
    rnn_rec: mtl.Buffer, // [512, 768] BF16: state then markov columns
    owned_bufs: []mtl.Buffer = &.{},

    pub fn deinit(l: *Loaded) void {
        for (l.owned_bufs) |b| b.deinit();
        if (l.owned_bufs.len > 0) l.gpa.free(l.owned_bufs);
        l.rnn_rec.deinit();
        l.rnn_kappa.deinit();
        l.beta.deinit();
        l.phi.deinit();
        l.dflare.deinit();
        l.checkpoint.deinit();
        l.* = undefined;
    }

    pub fn fixedBytes(l: *const Loaded) u64 {
        var extra: u64 = 0;
        for (l.owned_bufs) |b| extra += b.length();
        return l.checkpoint.residentBytes() + extra + l.dflare.length() + l.phi.length() + l.beta.length() +
            l.rnn_kappa.length() + l.rnn_rec.length();
    }
};

fn bfValue(h: u16) f64 {
    const x: f32 = @bitCast(@as(u32, h) << 16);
    return x;
}

/// Binary64 dot rounded once through binary32 to BF16.
pub fn foldDot(u: []const f64, w: []const f64) u16 {
    std.debug.assert(u.len == w.len);
    var sum: f64 = 0;
    for (u, w) |x, y| sum += x * y;
    return bf.bits(@floatCast(sum));
}

const FoldWork = struct {
    unembed: []const u16,
    hidden_w: []const f64,
    hidden_b: []const f64,
    phi: []u16,
    beta: []u16,
    v_stride: usize,
    begin: usize,
    end: usize,

    fn run(w: *const FoldWork) void {
        var u: [hidden]f64 = undefined;
        for (w.begin..w.end) |v| {
            for (&u, w.unembed[v * hidden ..][0..hidden]) |*x, h| x.* = bfValue(h);
            for (0..rank) |r| w.phi[r * w.v_stride + v] = foldDot(&u, w.hidden_w[r * hidden ..][0..hidden]);
            w.beta[v] = foldDot(&u, w.hidden_b);
        }
    }
};

/// Phi[r, v] = bf16(f32(f64(U[v]) . f64(hidden_out[:, r]))), beta[v] likewise for hidden_bias.
pub fn fold(
    a: std.mem.Allocator,
    unembed: []const u16,
    hidden_weight: []const u16,
    hidden_bias: []const u16,
    phi: []u16,
    beta: []u16,
) !void {
    const v = beta.len;
    if (unembed.len != v * hidden or hidden_weight.len != hidden * rank or hidden_bias.len != hidden or phi.len != v * rank or v == 0)
        return error.DSparkFoldShape;
    const hw = try a.alloc(f64, hidden * rank);
    defer a.free(hw);
    const hb = try a.alloc(f64, hidden);
    defer a.free(hb);
    for (hidden_weight, 0..) |h, i| hw[(i % rank) * hidden + i / rank] = bfValue(h);
    for (hb, hidden_bias) |*x, h| x.* = bfValue(h);

    const count = @min(@as(usize, 8), v);
    const chunk = (v + count - 1) / count;
    var work: [8]FoldWork = undefined;
    var threads: [8]?std.Thread = @splat(null);
    for (0..count) |i| {
        const begin = @min(i * chunk, v);
        const end = @min(begin + chunk, v);
        work[i] = .{ .unembed = unembed, .hidden_w = hw, .hidden_b = hb, .phi = phi, .beta = beta, .v_stride = v, .begin = begin, .end = end };
        threads[i] = std.Thread.spawn(.{}, FoldWork.run, .{&work[i]}) catch null;
        if (threads[i] == null) work[i].run();
    }
    for (threads) |thread| if (thread) |t| t.join();
}

/// Transpose checkpoint rnn_proj [3*rank, 2*rank + 2*hidden] into coalesced rec and kappa buffers.
pub fn packRnn(src: []const u16, rec_dst: []u16, kappa_dst: []u16) void {
    const rnn_rows = 3 * rank;
    const rec_cols = 2 * rank;
    const kappa_cols = 2 * hidden;
    const rnn_cols = rec_cols + kappa_cols;
    std.debug.assert(src.len == rnn_rows * rnn_cols);
    std.debug.assert(rec_dst.len == rec_cols * rnn_rows);
    std.debug.assert(kappa_dst.len == kappa_cols * rnn_rows);
    for (0..rnn_rows) |ch| {
        const row = src[ch * rnn_cols ..][0..rnn_cols];
        for (0..rec_cols) |d| rec_dst[d * rnn_rows + ch] = row[d];
        for (0..kappa_cols) |d| kappa_dst[d * rnn_rows + ch] = row[rec_cols + d];
    }
}

fn require(t: ck.Tensor, dims: []const usize) !ck.Tensor {
    if (t.dtype != .bf16 or t.rank != dims.len) return error.DSparkTensorShape;
    var count: usize = 1;
    for (dims, 0..) |d, i| {
        if (t.shape[i] != d) return error.DSparkTensorShape;
        count = std.math.mul(usize, count, d) catch return error.DSparkTensorShape;
    }
    if (t.bytes != count * 2) return error.DSparkTensorShape;
    return t;
}

fn get(c: *const ck.Checkpoint, name: []const u8, dims: []const usize) !ck.Tensor {
    const t = c.get(name) catch {
        std.log.err("Small DSpark mtp.safetensors is missing {s}", .{name});
        return error.DSparkMissingTensor;
    };
    return require(t, dims);
}

pub fn admit(c: *const cfg.Config) !void {
    const m = c.mtp orelse return error.DSparkNotConfigured;
    const want_taps = [_]u32{ 5, 11, 17, 23, 29, 35, 41, 48 };
    if (m.layers != layers_count or m.q_heads != 48 or m.kv_heads != 8 or m.head_dim != 128 or m.inter != intermediate or
        m.window != 1024 or m.block != 8 or m.rank != rank or m.rope_theta != 1e6 or m.taps_n != taps_count or
        !m.local[0] or !m.local[1] or m.local[2] or !m.local[3] or !m.local[4] or
        !std.mem.eql(u32, m.taps[0..taps_count], &want_taps))
        return error.UnsupportedDSparkConfig;
}

fn loadLayer(c: *const ck.Checkpoint, i: usize) !Layer {
    var b: [128]u8 = undefined;
    const g = struct {
        fn layerTensor(cp: *const ck.Checkpoint, buf: []u8, i_: usize, suffix: []const u8, dims: []const usize) !ck.Tensor {
            const name = try std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ i_, suffix });
            return get(cp, name, dims);
        }
    }.layerTensor;
    return .{
        .dflare = try g(c, &b, i, "dflare_norm.weight", &.{hidden}),
        .input_norm = try g(c, &b, i, "input_layernorm.weight", &.{hidden}),
        .post_norm = try g(c, &b, i, "post_attention_layernorm.weight", &.{hidden}),
        .q_norm = try g(c, &b, i, "self_attn.q_norm.weight", &.{128}),
        .k_norm = try g(c, &b, i, "self_attn.k_norm.weight", &.{128}),
        .k_norm_ctx = try g(c, &b, i, "self_attn.k_norm_ctx.weight", &.{128}),
        .q = try g(c, &b, i, "self_attn.q_proj.weight", &.{ query, hidden }),
        .k = try g(c, &b, i, "self_attn.k_proj.weight", &.{ kv, hidden }),
        .v = try g(c, &b, i, "self_attn.v_proj.weight", &.{ kv, hidden }),
        .k_ctx = try g(c, &b, i, "self_attn.k_proj_ctx.weight", &.{ kv, hidden }),
        .v_ctx = try g(c, &b, i, "self_attn.v_proj_ctx.weight", &.{ kv, hidden }),
        .o = try g(c, &b, i, "self_attn.o_proj.weight", &.{ hidden, query }),
        .gate = try g(c, &b, i, "mlp.gate_proj.weight", &.{ intermediate, hidden }),
        .up = try g(c, &b, i, "mlp.up_proj.weight", &.{ intermediate, hidden }),
        .down = try g(c, &b, i, "mlp.down_proj.weight", &.{ hidden, intermediate }),
    };
}

pub fn load(gpa: std.mem.Allocator, target: *const Model, dir: []const u8) !Loaded {
    try admit(&target.config);
    const vocab_n = target.config.text.vocab;
    var checkpoint = ck.Checkpoint.init(gpa);
    errdefer checkpoint.deinit();
    const path: [:0]u8 = if (std.mem.endsWith(u8, dir, ".safetensors"))
        try std.fmt.allocPrintSentinel(gpa, "{s}", .{dir}, 0)
    else
        try std.fmt.allocPrintSentinel(gpa, "{s}/mtp.safetensors", .{dir}, 0);
    defer gpa.free(path);
    checkpoint.addFile(target.device, path, "") catch |err| {
        std.log.err("Small DSpark cannot open {s}: {s}", .{ path, @errorName(err) });
        return error.DSparkMissingFile;
    };
    const c = &checkpoint;
    var layers: [layers_count]Layer = undefined;
    for (&layers, 0..) |*l, i| l.* = try loadLayer(c, i);
    const weights: Weights = .{
        .fuse = try get(c, "fuse_weight", &.{ fusions_count, taps_count }),
        .mask = try get(c, "mask_embed", &.{hidden}),
        .norm = try get(c, "norm.weight", &.{hidden}),
        .markov_norm = try get(c, "markov_head.dflare_norm.weight", &.{hidden}),
        .hidden_out = try get(c, "markov_head.hidden_out.weight", &.{ hidden, rank }),
        .hidden_bias = try get(c, "markov_head.hidden_out.bias", &.{hidden}),
        .markov_w1 = try get(c, "markov_head.markov_w1.weight", &.{ vocab_n, rank }),
        .rnn = try get(c, "markov_head.rnn_proj.weight", &.{ 3 * rank, 2 * hidden + 2 * rank }),
        .rnn_bias = try get(c, "markov_head.rnn_proj.bias", &.{3 * rank}),
        .layers = layers,
    };
    const dflare = try target.device.buffer(fusions_count * hidden * 2, mtl.ResourceOptions.shared);
    errdefer dflare.deinit();
    const dflare_packed = dflare.slice(u16, fusions_count * hidden);
    for (weights.layers, 0..) |l, i| @memcpy(dflare_packed[i * hidden ..][0..hidden], l.dflare.host(u16));
    @memcpy(dflare_packed[layers_count * hidden ..][0..hidden], weights.markov_norm.host(u16));
    const phi = try target.device.buffer(vocab_n * rank * 2, mtl.ResourceOptions.shared);
    errdefer phi.deinit();
    const beta = try target.device.buffer(vocab_n * 2, mtl.ResourceOptions.shared);
    errdefer beta.deinit();
    try fold(
        gpa,
        target.weights.unembed.host(u16),
        weights.hidden_out.host(u16),
        weights.hidden_bias.host(u16),
        phi.slice(u16, vocab_n * rank),
        beta.slice(u16, vocab_n),
    );
    const rnn_kappa = try target.device.buffer(2 * hidden * 3 * rank * 2, mtl.ResourceOptions.shared);
    errdefer rnn_kappa.deinit();
    const rnn_rec = try target.device.buffer(2 * rank * 3 * rank * 2, mtl.ResourceOptions.shared);
    errdefer rnn_rec.deinit();
    packRnn(weights.rnn.host(u16), rnn_rec.slice(u16, 2 * rank * 3 * rank), rnn_kappa.slice(u16, 2 * hidden * 3 * rank));
    return .{
        .gpa = gpa,
        .checkpoint = checkpoint,
        .weights = weights,
        .vocab = vocab_n,
        .dflare = dflare,
        .phi = phi,
        .beta = beta,
        .rnn_kappa = rnn_kappa,
        .rnn_rec = rnn_rec,
    };
}

test "inkling2 small: DSpark binary64 fold and RNN permute preserve exact shapes and values" {
    const gpa = std.testing.allocator;
    const u = try gpa.alloc(u16, 2 * hidden);
    defer gpa.free(u);
    const w = try gpa.alloc(u16, hidden * rank);
    defer gpa.free(w);
    const b = try gpa.alloc(u16, hidden);
    defer gpa.free(b);
    @memset(u, 0);
    @memset(w, 0);
    @memset(b, 0);
    u[hidden + 3] = bf.bits(1);
    w[3 * rank + 5] = bf.bits(2);
    b[3] = bf.bits(3);
    var phi: [2 * rank]u16 = undefined;
    var beta: [2]u16 = undefined;
    try fold(gpa, u, w, b, &phi, &beta);
    try std.testing.expectEqual(bf.bits(2), phi[5 * 2 + 1]);
    try std.testing.expectEqual(@as(u16, 0), phi[5 * 2]);
    try std.testing.expectEqual(bf.bits(3), beta[1]);
    try std.testing.expectError(error.DSparkFoldShape, fold(gpa, &.{}, w, b, &phi, &beta));

    const cols = 2 * rank + 2 * hidden;
    const src = try gpa.alloc(u16, 3 * rank * cols);
    defer gpa.free(src);
    @memset(src, 0);
    src[5 * cols + 7] = 0x3c00;
    src[9 * cols + 2 * rank + 13] = 0x4000;
    const rec = try gpa.alloc(u16, 2 * rank * 3 * rank);
    defer gpa.free(rec);
    const kappa = try gpa.alloc(u16, 2 * hidden * 3 * rank);
    defer gpa.free(kappa);
    packRnn(src, rec, kappa);
    try std.testing.expectEqual(@as(u16, 0x3c00), rec[7 * 3 * rank + 5]);
    try std.testing.expectEqual(@as(u16, 0x4000), kappa[13 * 3 * rank + 9]);
}
