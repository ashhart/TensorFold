//! LFM3's weights on CUDA from ModelOpt NVFP4 (weight-only): FP4 FFN, bf16 mixers, embedding, tied head and norms.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const c = @import("config.zig");
const admit = @import("weights.zig");
const kern = @import("cuda_kernels.zig");

const DeviceBuffer = cuda.DeviceBuffer;
pub const Proj = cuda.qlinear.Weight;

/// Rows of the stacked attention projection: q (16 x 128), k and v (4 x 128 each), the 16 head gates.
pub const qkvg_rows = c.query_heads * c.head_dim + 2 * c.kv_heads * c.head_dim + c.query_heads;

pub const Conv = struct { in: Proj, out: Proj, weight: u64 };
/// q, k, v and the head gates stacked into one projection; the q and k norms bf16 [128] each, q's then k's.
pub const Attention = struct { qkvg: Proj, out: Proj, norms: u64, sliding: bool };

pub const Layer = struct {
    operator_norm: u64,
    post_operator_norm: u64,
    ffn_norm: u64,
    post_ffn_norm: u64,
    mixer: union(enum) { conv: Conv, attention: Attention },
    /// w1 | w3 stacked (gate rows, then up rows), and w2.
    gate_up: Proj,
    down: Proj,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: c.Config,
    /// The embedding as stored, bf16 [V, H] (the embed rows' table); the head is its lane copy.
    embed: u64 = 0,
    head: Proj = undefined,
    final_norm: u64 = 0,
    layers: []Layer = &.{},
    buffers: std.ArrayList(DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.config.deinit();
        w.* = undefined;
    }

    fn alloc(w: *Weights, d: *const cuda.Driver, bytes: usize) !u64 {
        const b = try DeviceBuffer.alloc(d, @max(bytes, 256));
        errdefer {
            var x = b;
            x.free();
        }
        try w.buffers.append(w.gpa, b);
        return b.ptr;
    }

    fn upload(w: *Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
        const b = try DeviceBuffer.fromHost(d, bytes);
        errdefer {
            var x = b;
            x.free();
        }
        try w.buffers.append(w.gpa, b);
        return b.ptr;
    }
};

/// The largest staged projection: conv in (3 x 2048 rows) in bf16, or the FFN's w1 and w3 codes and scales.
pub const stage_bytes = @max(3 * c.hidden * c.hidden * 2, 2 * c.intermediate * (c.hidden / 2 + c.hidden / 16)) + 4 * 256;

const Ctx = struct {
    w: *Weights,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const kern.Kernels,
    ck: *core.checkpoint.Checkpoint,
    stage: DeviceBuffer,

    fn tensor(x: *Ctx, name: []const u8, dtype: core.checkpoint.DType, shape: []const usize) !u64 {
        return x.w.upload(x.d, (try x.ck.expect(name, dtype, shape)).bytes);
    }

    /// bf16 projections [n_i, k] stacked along n, packed as one `.bf16` lane weight.
    fn bf16(x: *Ctx, names: []const []const u8, ns: []const usize, k: usize) !Proj {
        var n: usize = 0;
        var buf: [192]u8 = undefined;
        for (names, ns) |name, rows| {
            const t = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), .bf16, &.{ rows, k });
            if ((n + rows) * k * 2 > x.stage.len) return error.StageTooSmall;
            try x.stage.upload(n * k * 2, t.bytes);
            n += rows;
        }
        return x.pack(x.stage.ptr, n, k);
    }

    /// bf16 rows [n, k] on the device as a `.bf16` lane weight.
    fn pack(x: *Ctx, src: u64, n: usize, k: usize) !Proj {
        const out = try x.w.alloc(x.d, cuda.modelopt.padded(n) * k * 2);
        try x.k.pack.bf16Lane(x.s, src, out, n, k);
        try x.s.synchronize(); // the next projection's staging overwrites this one's
        return .{ .bf16 = cuda.modelopt.bf16Weight(out, n, k) };
    }

    /// NVFP4 projections [n_i, k] sharing one tensor scale, stacked along n as one `.nvfp4` lane weight.
    fn nvfp4(x: *Ctx, names: []const []const u8, ns: []const usize, k: usize) !Proj {
        var buf: [192]u8 = undefined;
        var n: usize = 0;
        for (ns) |rows| n += rows;
        const codes_bytes = n * k / 2;
        if (codes_bytes + n * k / 16 > x.stage.len) return error.StageTooSmall;
        var global: ?f32 = null;
        var at: usize = 0;
        for (names, ns) |name, rows| {
            const codes = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), .u8, &.{ rows, k / 2 });
            const blocks = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{name}), .f8_e4m3, &.{ rows, k / 16 });
            const g = try scalar(x.ck, try std.fmt.bufPrint(&buf, "{s}.weight_scale_2", .{name}));
            _ = try scalar(x.ck, try std.fmt.bufPrint(&buf, "{s}.input_scale", .{name})); // weight-only: bf16 rows in
            if (global) |v| if (v != g) return error.UnsupportedLfm3Quantization; // one scale a stacked weight
            global = g;
            try x.stage.upload(at * k / 2, codes.bytes);
            try x.stage.upload(codes_bytes + at * k / 16, blocks.bytes);
            at += rows;
        }
        const words = try x.w.alloc(x.d, cuda.modelopt.padded(n) * k / 2);
        const tiles = try x.w.alloc(x.d, cuda.modelopt.padded(n) * k / 16);
        try x.k.pack.fp4Lane(x.s, x.stage.ptr, x.stage.ptr + codes_bytes, words, tiles, n, k);
        try x.s.synchronize();
        return .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, global.?, n, k) };
    }
};

/// An fp32 tensor of one element.
fn scalar(ck: *core.checkpoint.Checkpoint, name: []const u8) !f32 {
    const t = try ck.get(name);
    if (t.dtype != .f32 or t.bytes.len != 4) return error.UnexpectedTensor;
    return @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
}

/// The text decoder's tensors against the shared admission (weights.zig): names, dtypes, shapes, bytes, nothing extra.
fn admitted(gpa: std.mem.Allocator, config: *const c.Config, ck: *const core.checkpoint.Checkpoint) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var all: core.safetensors.Header = .empty;
    for (ck.files.items) |*f| for (f.names.keys(), f.names.values()) |name, e| try all.put(arena.allocator(), name, e);
    const m = try admit.validate(gpa, config, &all);
    if (m.namespace != .bare) return error.UnsupportedLfm3Quantization;
}

fn named(b: []u8, prefix: []const u8, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(b, "{s}{s}", .{ prefix, suffix });
}

/// Every tensor checked against the admitted geometry and packed for the lane; `layers` caps them (tests).
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *const kern.Kernels, dir: []const u8, layers: ?usize) !Weights {
    var config = try c.Config.read(gpa, io, dir);
    if (config.scheme != .modelopt_nvfp4 or config.kind != .text) {
        config.deinit();
        return error.UnsupportedLfm3Quantization; // MLX affine and FP8 exports: other loaders of the same layers
    }
    var w: Weights = .{ .gpa = gpa, .config = config };
    errdefer w.deinit();
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    try admitted(gpa, &w.config, &ck);
    const h = c.hidden;
    var stage = try DeviceBuffer.alloc(d, stage_bytes);
    defer stage.free();
    var x: Ctx = .{ .w = &w, .d = d, .s = s, .k = k, .ck = &ck, .stage = stage };
    w.embed = try x.tensor("model.embed_tokens.weight", .bf16, &.{ c.vocab, h });
    w.head = try x.pack(w.embed, c.vocab, h); // tied: the lane's copy of the embedding
    w.final_norm = try x.tensor("model.embedding_norm.weight", .bf16, &.{h});
    const n = @min(layers orelse c.layers, c.layers);
    w.layers = try gpa.alloc(Layer, n);
    var b1: [128]u8 = undefined;
    var b2: [128]u8 = undefined;
    var b3: [128]u8 = undefined;
    var b4: [128]u8 = undefined;
    for (w.layers, w.config.layer_types[0..n], 0..) |*l, kind, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        l.operator_norm = try x.tensor(try named(&b1, p, "operator_norm.weight"), .bf16, &.{h});
        l.post_operator_norm = try x.tensor(try named(&b1, p, "post_operator_norm.weight"), .bf16, &.{h});
        l.ffn_norm = try x.tensor(try named(&b1, p, "ffn_norm.weight"), .bf16, &.{h});
        l.post_ffn_norm = try x.tensor(try named(&b1, p, "post_ffn_norm.weight"), .bf16, &.{h});
        if (kind == .conv) {
            l.mixer = .{ .conv = .{
                .in = try x.bf16(&.{try named(&b1, p, "conv.in_proj")}, &.{3 * h}, h),
                .out = try x.bf16(&.{try named(&b1, p, "conv.out_proj")}, &.{h}, h),
                .weight = try x.tensor(try named(&b1, p, "conv.conv.weight"), .bf16, &.{ h, 1, c.conv_taps }),
            } };
        } else {
            const qw = c.query_heads * c.head_dim;
            const kw = c.kv_heads * c.head_dim;
            const norms = try w.alloc(d, 2 * c.head_dim * 2);
            const qn = try ck.expect(try named(&b1, p, "self_attn.q_norm.weight"), .bf16, &.{c.head_dim});
            const kn = try ck.expect(try named(&b1, p, "self_attn.k_norm.weight"), .bf16, &.{c.head_dim});
            const nb: DeviceBuffer = .{ .d = d, .ptr = norms, .len = 4 * c.head_dim, .borrowed = true };
            try nb.upload(0, qn.bytes);
            try nb.upload(2 * c.head_dim, kn.bytes);
            const names = [_][]const u8{ try named(&b1, p, "self_attn.q_proj"), try named(&b2, p, "self_attn.k_proj"), try named(&b3, p, "self_attn.v_proj"), try named(&b4, p, "self_attn.gate_proj") };
            l.mixer = .{ .attention = .{
                .qkvg = try x.bf16(&names, &.{ qw, kw, kw, c.query_heads }, h),
                .out = try x.bf16(&.{try named(&b1, p, "self_attn.o_proj")}, &.{h}, qw),
                .norms = norms,
                .sliding = kind == .sliding_attention,
            } };
        }
        l.gate_up = try x.nvfp4(&.{ try named(&b1, p, "feed_forward.w1"), try named(&b2, p, "feed_forward.w3") }, &.{ c.intermediate, c.intermediate }, h);
        l.down = try x.nvfp4(&.{try named(&b1, p, "feed_forward.w2")}, &.{h}, c.intermediate);
    }
    return w;
}

test "the stage holds conv in's bf16 rows and the FFN's stacked codes and scales" {
    try std.testing.expect(stage_bytes >= 3 * c.hidden * c.hidden * 2);
    try std.testing.expect(stage_bytes >= 2 * c.intermediate * c.hidden / 2 + 2 * c.intermediate * c.hidden / 16);
    try std.testing.expect(stage_bytes >= qkvg_rows * c.hidden * 2);
    try std.testing.expectEqual(@as(usize, 3088), qkvg_rows);
}
