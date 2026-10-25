//! Spark2.5's weights on CUDA: MLX affine projections in qlinear's tiles (q|k|v and the head gate as one), fp32 norms.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const kern = @import("kernels.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Affine4 = cuda.qlinear.Affine4;
const AffineN = cuda.qlinear.AffineN;

/// A projection at its MLX width: affine-4 tiles, or the tiles of every other width.
pub const Proj = union(enum) {
    a4: Affine4,
    an: AffineN,

    pub fn n(p: Proj) usize {
        return switch (p) {
            inline else => |q| q.n,
        };
    }

    pub fn k(p: Proj) usize {
        return switch (p) {
            inline else => |q| q.k,
        };
    }
};

/// A layer's tensors: q_k_v_proj and g_proj stacked (q, k, v, then a gate a head), norms widened to fp32.
pub const Layer = struct {
    qkvg: Proj,
    out: Proj,
    gate: Proj,
    up: Proj,
    down: Proj,
    input_norm: u64, // fp32 [hidden]
    post_attention: u64,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: cfg.Config,
    /// The embedding table as MLX stores it: words u32 [V, H bits / 32], scales and biases [V, H / 64].
    embed_w: u64 = 0,
    embed_s: u64 = 0,
    embed_b: u64 = 0,
    /// The same table as the tied head's tiles.
    head: Proj = undefined,
    final_norm: u64 = 0,
    layers: []Layer = &.{},
    buffers: std.ArrayList(DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
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

const Ctx = struct {
    gpa: std.mem.Allocator,
    w: *Weights,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const kern.Kernels,
    ck: *core.checkpoint.Checkpoint,
    stage: DeviceBuffer,
    bits: u8,

    fn tensor(x: *Ctx, name: []const u8, dtype: core.checkpoint.DType, shape: []const usize) !u64 {
        return x.w.upload(x.d, (try x.ck.expect(name, dtype, shape)).bytes);
    }

    /// A bf16 vector [n] widened to fp32 on the host (exactly), as the norms' fp32 weights.
    fn widened(x: *Ctx, name: []const u8, n: usize) !u64 {
        const t = try x.ck.expect(name, .bf16, &.{n});
        const f = try x.gpa.alloc(f32, n);
        defer x.gpa.free(f);
        for (f, 0..) |*v, i| {
            const b = std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little);
            v.* = @bitCast(@as(u32, b) << 16);
        }
        return x.w.upload(x.d, std.mem.sliceAsBytes(f));
    }

    /// MLX affine projections stacked along n (each part [n_i, k], one input width) packed as one into qlinear's tiles.
    fn linear(x: *Ctx, parts: []const []const u8, ns: []const usize, k: usize) !Proj {
        const bits = x.bits;
        var n: usize = 0;
        for (ns) |v| n += v;
        var buf: [192]u8 = undefined;
        const specs = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, k * bits / 32 }, .{ "scales", .bf16, k / 64 }, .{ "biases", .bf16, k / 64 } };
        var raw: [3]u64 = undefined;
        var at: usize = 0;
        for (specs, &raw) |sp, *r| {
            r.* = x.stage.ptr + at;
            for (parts, ns) |prefix, rows| {
                const t = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, sp[0] }), sp[1], &.{ rows, sp[2] });
                if (at + t.bytes.len > x.stage.len) return error.StageTooSmall;
                try x.stage.upload(at, t.bytes);
                at += t.bytes.len;
            }
        }
        const served = cuda.qlinear.servedBits(bits);
        const out: Proj = if (served == 4) blk: {
            const l = Affine4.layout(n, k);
            const q: Affine4 = .{ .w = try x.w.alloc(x.d, l.words), .s = try x.w.alloc(x.d, l.scales), .b = try x.w.alloc(x.d, l.scales), .n = n, .k = k, .npad = l.npad };
            try x.k.affine.pack(x.s, raw[0], raw[1], raw[2], q);
            break :blk .{ .a4 = q };
        } else blk: {
            const l = AffineN.layout(n, k, served);
            const q: AffineN = .{ .w = try x.w.alloc(x.d, l.words), .s = try x.w.alloc(x.d, l.scales), .b = try x.w.alloc(x.d, l.scales), .n = n, .k = k, .npad = l.npad, .bits = served };
            try x.k.wide.packWords(x.s, raw[0], bits, n, k, q.w);
            try x.k.wide.packTable(x.s, raw[1], n, k, q.npad, q.s);
            try x.k.wide.packTable(x.s, raw[2], n, k, q.npad, q.b);
            break :blk .{ .an = q };
        };
        try x.s.synchronize(); // the next projection's staging overwrites this one's
        return out;
    }
};

const m = "model.";

/// Every tensor checked against the admitted geometry and width. `layers` caps them (checks).
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *const kern.Kernels, dir: []const u8, layers: ?usize) !Weights {
    const c = try cfg.Config.read(gpa, io, dir);
    const geo = c.geo;
    var w: Weights = .{ .gpa = gpa, .config = c };
    errdefer w.deinit();
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const h = geo.hidden;
    // the largest projection as stored: the tied head's table
    var stage = try DeviceBuffer.alloc(d, cfg.vocab * h * c.bits / 8 + 2 * cfg.vocab * (h / 64) * 2);
    defer stage.free();
    var x: Ctx = .{ .gpa = gpa, .w = &w, .d = d, .s = s, .k = k, .ck = &ck, .stage = stage, .bits = c.bits };
    w.embed_w = try x.tensor(m ++ "embedding.weight", .u32, &.{ cfg.vocab, h * c.bits / 32 });
    w.embed_s = try x.tensor(m ++ "embedding.scales", .bf16, &.{ cfg.vocab, h / 64 });
    w.embed_b = try x.tensor(m ++ "embedding.biases", .bf16, &.{ cfg.vocab, h / 64 });
    w.head = try x.linear(&.{m ++ "embedding"}, &.{cfg.vocab}, h);
    w.final_norm = try x.widened(m ++ "norm.weight", h);
    const n = @min(layers orelse geo.layers, geo.layers);
    w.layers = try gpa.alloc(Layer, n);
    var b1: [128]u8 = undefined;
    var b2: [128]u8 = undefined;
    for (w.layers, 0..) |*l, i| {
        const name = struct {
            fn of(b: []u8, layer: usize, suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, m ++ "layers.{d}.{s}", .{ layer, suffix });
            }
        }.of;
        const qkv = geo.queryWidth() + 2 * geo.kvWidth();
        l.qkvg = try x.linear(&.{ try name(&b1, i, "self_attn.q_k_v_proj"), try name(&b2, i, "self_attn.g_proj") }, &.{ qkv, geo.query_heads }, h);
        l.out = try x.linear(&.{try name(&b1, i, "self_attn.out_proj")}, &.{h}, geo.queryWidth());
        l.gate = try x.linear(&.{try name(&b1, i, "mlp.gate_proj")}, &.{geo.intermediate}, h);
        l.up = try x.linear(&.{try name(&b1, i, "mlp.up_proj")}, &.{geo.intermediate}, h);
        l.down = try x.linear(&.{try name(&b1, i, "mlp.down_proj")}, &.{h}, geo.intermediate);
        l.input_norm = try x.widened(try name(&b1, i, "input_layernorm.weight"), h);
        l.post_attention = try x.widened(try name(&b1, i, "post_attention_layernorm.weight"), h);
    }
    return w;
}
