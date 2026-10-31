//! The DSpark drafter's weights: the released bf16 linears as stored, or on request quantized to 8-bit affine codes.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const dc = @import("dspark_config.zig");
const quant = @import("dspark_quant.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const tile = @import("../../core/tile_qmm.zig");

const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// An [n, k] bf16 matrix as released (rows only reordered for half-split RoPE where the checkpoint is read).
pub const Bf16 = struct { w: mtl.Buffer, n: usize, k: usize };

/// A linear: bf16 as stored (bits 0), or MLX affine codes at the loader's width.
pub const Linear = union(enum) { bf16: Bf16, codes: tile.Matrix };

/// A bf16 vector in its own buffer (norm gains; RoPE-paired ones in the half-split order the kernels rotate).
pub const Vector = mtl.Buffer;

pub const Layer = struct {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    input_norm: Vector,
    post_norm: Vector,
    q_norm: Vector,
    k_norm: Vector,
};

pub const Weights = struct {
    fc: Linear,
    hidden_norm: Vector,
    norm: Vector,
    layers: [8]Layer,
    /// The Markov head: W1 [vocab, rank] bf16 as stored (rows gathered by token), W2 [vocab, rank] as the linears.
    w1: mtl.Buffer,
    w2: Linear,
    /// The confidence head: its [hidden + rank] bf16 weights and fp32 bias (null: the checkpoint has none).
    confidence: ?mtl.Buffer = null,
    confidence_bias: f32 = 0,
    owned: std.ArrayList(mtl.Buffer) = .empty,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(gpa);
    }

    /// Every buffer the weights hold, for a residency set.
    pub fn buffers(w: *const Weights) []const mtl.Buffer {
        return w.owned.items;
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    check: *const ckpt.Checkpoint,
    owned: *std.ArrayList(mtl.Buffer),
    bits: u8,

    fn buffer(l: Loader, bytes: usize) !mtl.Buffer {
        const b = try l.device.buffer(bytes, shared);
        errdefer b.deinit();
        try l.owned.append(l.gpa, b);
        return b;
    }

    fn tensor(l: Loader, name: []const u8, shape: []const usize) !ckpt.Tensor {
        const t = try l.check.get(name);
        if (t.dtype != .bf16 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.log.err("{s}: unexpected DSpark tensor dtype or shape", .{name});
            return error.BadDraftWeight;
        }
        return t;
    }

    /// A bf16 vector copied out, its pairs reordered for half-split RoPE when `paired`.
    fn vector(l: Loader, name: []const u8, len: usize, paired: bool) !Vector {
        const src = (try l.tensor(name, &.{len})).host(u16);
        const b = try l.buffer(len * 2);
        const dst = b.slice(u16, len);
        for (dst, 0..) |*v, i| v.* = src[if (paired) quant.halfSplit(i, len) else i];
        return b;
    }

    /// An [n, k] bf16 matrix as it is or as affine codes; rows reordered for half-split RoPE when `paired`.
    fn matrix(l: Loader, name: []const u8, n: usize, k: usize, paired: bool) !Linear {
        const t = try l.tensor(name, &.{ n, k });
        var rows = t.host(u16);
        var moved: ?[]u16 = null;
        defer if (moved) |m| l.gpa.free(m);
        if (paired) {
            const m = try l.gpa.alloc(u16, n * k);
            moved = m;
            for (0..n) |r| @memcpy(m[r * k ..][0..k], rows[quant.halfSplit(r, c.head_dim) * k ..][0..k]);
            rows = m;
        }
        if (l.bits == 0) {
            const b = try l.buffer(n * k * 2);
            @memcpy(b.slice(u16, n * k), rows);
            return .{ .bf16 = .{ .w = b, .n = n, .k = k } };
        }
        const codes = try l.buffer(n * k * l.bits / 8);
        const scales = try l.buffer(n * k / 64 * 2);
        const biases = try l.buffer(n * k / 64 * 2);
        const job: quant.Job = .{ .src = rows, .k = k, .bits = l.bits, .codes = codes.slice(u8, n * k * l.bits / 8), .scales = scales.slice(u16, n * k / 64), .biases = biases.slice(u16, n * k / 64) };
        try job.all(l.gpa, n);
        return .{ .codes = .{ .w = codes, .scales = scales, .biases = biases, .n = n, .k = k, .bits = l.bits } };
    }
};

fn part(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "layers.{d}.{s}.weight", .{ layer, name });
}

/// The drafter's checkpoint read as released (`bits` 0) or quantized to 8-bit codes; its own buffers are freed after.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, io: std.Io, dir: []const u8, cfg: *const dc.Config, bits: u8) !Weights {
    if (bits != 0 and bits != 8) return error.UnsupportedDraftBits;
    var check = ckpt.Checkpoint.init(gpa);
    defer check.deinit();
    const files = try shards.shardFiles(gpa, io, dir);
    defer shards.freeShardFiles(gpa, files);
    for (files) |path| try check.addFile(device, path, "");
    var w: Weights = undefined;
    w.owned = .empty;
    errdefer w.deinit(gpa);
    const l: Loader = .{ .gpa = gpa, .device = device, .check = &check, .owned = &w.owned, .bits = bits };
    const qw = c.query_heads * c.head_dim;
    const kw = c.kv_heads * c.head_dim;
    w.fc = try l.matrix("fc.weight", c.hidden, cfg.tap_count * c.hidden, false);
    w.hidden_norm = try l.vector("hidden_norm.weight", c.hidden, false);
    w.norm = try l.vector("norm.weight", c.hidden, false);
    var name: [128]u8 = undefined;
    for (w.layers[0..cfg.layers], 0..) |*ly, i| {
        ly.q = try l.matrix(try part(&name, i, "self_attn.q_proj"), qw, c.hidden, true);
        ly.k = try l.matrix(try part(&name, i, "self_attn.k_proj"), kw, c.hidden, true);
        ly.v = try l.matrix(try part(&name, i, "self_attn.v_proj"), kw, c.hidden, false);
        ly.o = try l.matrix(try part(&name, i, "self_attn.o_proj"), c.hidden, qw, false);
        ly.gate = try l.matrix(try part(&name, i, "mlp.gate_proj"), cfg.intermediate, c.hidden, false);
        ly.up = try l.matrix(try part(&name, i, "mlp.up_proj"), cfg.intermediate, c.hidden, false);
        ly.down = try l.matrix(try part(&name, i, "mlp.down_proj"), c.hidden, cfg.intermediate, false);
        ly.input_norm = try l.vector(try part(&name, i, "input_layernorm"), c.hidden, false);
        ly.post_norm = try l.vector(try part(&name, i, "post_attention_layernorm"), c.hidden, false);
        ly.q_norm = try l.vector(try part(&name, i, "self_attn.q_norm"), c.head_dim, true);
        ly.k_norm = try l.vector(try part(&name, i, "self_attn.k_norm"), c.head_dim, true);
    }
    const w1 = (try l.tensor("markov_head.markov_w1.weight", &.{ c.vocab, cfg.rank })).host(u16);
    w.w1 = try l.buffer(c.vocab * cfg.rank * 2);
    @memcpy(w.w1.slice(u16, c.vocab * cfg.rank), w1);
    w.w2 = try l.matrix("markov_head.markov_w2.weight", c.vocab, cfg.rank, false);
    w.confidence = null;
    w.confidence_bias = 0;
    if (check.tensors.contains("confidence_head.proj.weight")) {
        const width = c.hidden + cfg.rank; // the final norm's row, then W1 of the token before the draft
        const proj = (try l.tensor("confidence_head.proj.weight", &.{ 1, width })).host(u16);
        w.confidence = try l.buffer(width * 2);
        @memcpy(w.confidence.?.slice(u16, width), proj);
        w.confidence_bias = quant.toF32((try l.tensor("confidence_head.proj.bias", &.{1})).host(u16)[0]);
    }
    return w;
}
