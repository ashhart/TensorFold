//! Validated mtp.safetensors weights and the one-time folded Markov output head.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const cfg = @import("../config.zig");
const Model = @import("model.zig").Model;
const fold = @import("dspark_fold.zig");
const q4 = @import("draft_head_q4.zig");

pub const hidden = 2304;
pub const query = 3072;
pub const kv = 1024;
pub const intermediate = 9216;
pub const rank = 256;
pub const vocabulary = 201024;

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
    layers: [3]Layer,
};

pub const Loaded = struct {
    checkpoint: ck.Checkpoint,
    weights: Weights,
    dflare: mtl.Buffer, // four H-wide rows: three layers, then the Markov head
    phi: mtl.Buffer, // [256, 201024] BF16, folded once in binary64
    beta: mtl.Buffer, // [201024] BF16
    rnn_kappa: mtl.Buffer, // [4608, 768] BF16: y then newest_f columns
    rnn_rec: mtl.Buffer, // [512, 768] BF16: state then markov columns
    draft_head_q4: ?q4.DraftHeadQ4 = null,

    pub fn deinit(l: *Loaded) void {
        if (l.draft_head_q4) |*head| head.deinit();
        l.rnn_rec.deinit();
        l.rnn_kappa.deinit();
        l.beta.deinit();
        l.phi.deinit();
        l.dflare.deinit();
        l.checkpoint.deinit();
        l.* = undefined;
    }

    pub fn fixedBytes(l: *const Loaded) u64 {
        const q4_bytes = if (l.draft_head_q4) |head| head.fixedBytes() else 0;
        return l.checkpoint.residentBytes() + l.dflare.length() + l.phi.length() + l.beta.length() +
            l.rnn_kappa.length() + l.rnn_rec.length() + q4_bytes;
    }

    pub fn enableDraftHeadQ4(l: *Loaded, target: *const Model, used_bytes: u64, limit_bytes: u64) !void {
        if (l.draft_head_q4 != null) return;
        l.draft_head_q4 = try q4.maybeInit(
            target.device,
            target.weights.unembed.bf16.host(u16),
            target.config.text.unpadded_vocab,
            hidden,
            used_bytes,
            limit_bytes,
        );
    }
};

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

fn require(t: ck.Tensor, dims: []const usize, _: []const u8) !ck.Tensor {
    if (t.dtype != .bf16 or t.rank != dims.len) {
        return error.DSparkTensorShape;
    }
    var count: usize = 1;
    for (dims, 0..) |d, i| {
        if (t.shape[i] != d) {
            return error.DSparkTensorShape;
        }
        count = std.math.mul(usize, count, d) catch return error.DSparkTensorShape;
    }
    if (t.bytes != count * 2) return error.DSparkTensorShape;
    return t;
}

fn get(c: *const ck.Checkpoint, name: []const u8, dims: []const usize) !ck.Tensor {
    const t = c.get(name) catch {
        std.log.err("DSpark mtp.safetensors is missing {s}", .{name});
        return error.DSparkMissingTensor;
    };
    return require(t, dims, name);
}

fn admit(c: *const cfg.Config) !void {
    const m = c.mtp orelse return error.DSparkNotConfigured;
    if (m.layers != 3 or m.q_heads != 24 or m.kv_heads != 8 or m.head_dim != 128 or m.inter != intermediate or
        m.window != 1024 or m.block != 8 or m.rank != rank or m.rope_theta != 1e6 or m.taps_n != 4 or
        !m.local[0] or m.local[1] or !m.local[2] or !std.mem.eql(u32, m.taps[0..4], &.{ 5, 23, 41, 48 }))
        return error.UnsupportedDSparkConfig;
}

fn layer(c: *const ck.Checkpoint, i: usize) !Layer {
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
    _ = try q4.enabled();
    try admit(&target.config);
    var checkpoint = ck.Checkpoint.init(gpa);
    errdefer checkpoint.deinit();
    const path: [:0]u8 = if (std.mem.endsWith(u8, dir, ".safetensors"))
        try std.fmt.allocPrintSentinel(gpa, "{s}", .{dir}, 0)
    else
        try std.fmt.allocPrintSentinel(gpa, "{s}/mtp.safetensors", .{dir}, 0);
    defer gpa.free(path);
    checkpoint.addFile(target.device, path, "") catch |err| {
        std.log.err("DSpark cannot open {s}: {s}", .{ path, @errorName(err) });
        return error.DSparkMissingFile;
    };
    const c = &checkpoint;
    var layers: [3]Layer = undefined;
    for (&layers, 0..) |*l, i| l.* = try layer(c, i);
    const weights: Weights = .{
        .fuse = try get(c, "fuse_weight", &.{ 4, 4 }),
        .mask = try get(c, "mask_embed", &.{hidden}),
        .norm = try get(c, "norm.weight", &.{hidden}),
        .markov_norm = try get(c, "markov_head.dflare_norm.weight", &.{hidden}),
        .hidden_out = try get(c, "markov_head.hidden_out.weight", &.{ hidden, rank }),
        .hidden_bias = try get(c, "markov_head.hidden_out.bias", &.{hidden}),
        .markov_w1 = try get(c, "markov_head.markov_w1.weight", &.{ vocabulary, rank }),
        .rnn = try get(c, "markov_head.rnn_proj.weight", &.{ 3 * rank, 2 * hidden + 2 * rank }),
        .rnn_bias = try get(c, "markov_head.rnn_proj.bias", &.{3 * rank}),
        .layers = layers,
    };
    const dflare = try target.device.buffer(4 * hidden * 2, mtl.ResourceOptions.shared);
    errdefer dflare.deinit();
    const dflare_packed = dflare.slice(u16, 4 * hidden);
    for (weights.layers, 0..) |l, i| @memcpy(dflare_packed[i * hidden ..][0..hidden], l.dflare.host(u16));
    @memcpy(dflare_packed[3 * hidden ..][0..hidden], weights.markov_norm.host(u16));
    const phi = try target.device.buffer(vocabulary * rank * 2, mtl.ResourceOptions.shared);
    errdefer phi.deinit();
    const beta = try target.device.buffer(vocabulary * 2, mtl.ResourceOptions.shared);
    errdefer beta.deinit();
    const unembed_bf16 = switch (target.weights.unembed) {
        .bf16 => |t| t,
        .affine4 => return error.UnsupportedDSparkTargetQuant,
    };
    try fold.fold(gpa, unembed_bf16.host(u16), weights.hidden_out.host(u16), weights.hidden_bias.host(u16), phi.slice(u16, vocabulary * rank), beta.slice(u16, vocabulary));
    const rnn_kappa = try target.device.buffer(2 * hidden * 3 * rank * 2, mtl.ResourceOptions.shared);
    errdefer rnn_kappa.deinit();
    const rnn_rec = try target.device.buffer(2 * rank * 3 * rank * 2, mtl.ResourceOptions.shared);
    errdefer rnn_rec.deinit();
    packRnn(weights.rnn.host(u16), rnn_rec.slice(u16, 2 * rank * 3 * rank), rnn_kappa.slice(u16, 2 * hidden * 3 * rank));
    return .{ .checkpoint = checkpoint, .weights = weights, .dflare = dflare, .phi = phi, .beta = beta, .rnn_kappa = rnn_kappa, .rnn_rec = rnn_rec };
}

test "load-time RNN permute splits and transposes recurrent and kappa projections" {
    const gpa = std.testing.allocator;
    const cols = 2 * rank + 2 * hidden;
    const src = try gpa.alloc(u16, 3 * rank * cols);
    defer gpa.free(src);
    @memset(src, 0);
    src[5 * cols + 7] = 0x3c00;
    src[9 * cols + 2 * rank + 11] = 0x4000;
    const rec = try gpa.alloc(u16, 2 * rank * 3 * rank);
    defer gpa.free(rec);
    const kappa = try gpa.alloc(u16, 2 * hidden * 3 * rank);
    defer gpa.free(kappa);
    packRnn(src, rec, kappa);
    try std.testing.expectEqual(@as(u16, 0x3c00), rec[7 * 3 * rank + 5]);
    try std.testing.expectEqual(@as(u16, 0x4000), kappa[11 * 3 * rank + 9]);
}

test "a misshapen DSpark tensor has a named refusal" {
    if (comptime @import("builtin").os.tag != .macos) return error.SkipZigTest;
    var t: ck.Tensor = .{ .buffer = undefined, .offset = 0, .bytes = 8, .dtype = .bf16, .rank = 1 };
    t.shape[0] = 4;
    _ = try require(t, &.{4}, "witness");
    try std.testing.expectError(error.DSparkTensorShape, require(t, &.{ 2, 2 }, "witness"));
    t.dtype = .f32;
    try std.testing.expectError(error.DSparkTensorShape, require(t, &.{4}, "witness"));
}

test "Loaded fixedBytes accounts for optional draft_head_q4" {
    if (comptime @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    defer dev.deinit();
    var l: Loaded = .{
        .checkpoint = ck.Checkpoint.init(std.testing.allocator),
        .weights = undefined,
        .dflare = try dev.buffer(64, mtl.ResourceOptions.shared),
        .phi = try dev.buffer(64, mtl.ResourceOptions.shared),
        .beta = try dev.buffer(64, mtl.ResourceOptions.shared),
        .rnn_kappa = try dev.buffer(64, mtl.ResourceOptions.shared),
        .rnn_rec = try dev.buffer(64, mtl.ResourceOptions.shared),
    };
    defer l.deinit();
    const base_bytes = l.fixedBytes();
    const w_bf = try std.testing.allocator.alloc(u16, 4 * 64);
    defer std.testing.allocator.free(w_bf);
    @memset(w_bf, 0);
    l.draft_head_q4 = try q4.init(dev, w_bf, 4, 64);
    try std.testing.expectEqual(base_bytes + l.draft_head_q4.?.fixedBytes(), l.fixedBytes());
}
