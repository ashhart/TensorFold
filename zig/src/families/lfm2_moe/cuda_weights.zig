//! LFM2.5-24B-A2B's weights on CUDA: affine-4 projections, expert pair/down tables, the tied head, bf16 norms.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const c = @import("config.zig");
const kern = @import("cuda_kernels.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Affine4 = cuda.qlinear.Affine4;
const AffineN = cuda.qlinear.AffineN;
const A4 = cuda.experts.Affine4Experts;

/// A projection at its MLX width: affine-4 tiles, or the tiles of every other width (the head).
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

pub const Conv = struct { in: Proj, out: Proj, weight: u64 };
/// q, k and v stacked into one projection; the q and k norms bf16 [64] each, q's then k's.
pub const Attention = struct { qkv: Proj, out: Proj, norms: u64 };
pub const Dense = struct { gate_up: Proj, down: Proj };
pub const Moe = struct { router: u64, bias: u64, experts: cuda.experts.Layer };

pub const Layer = struct {
    operator_norm: u64,
    ffn_norm: u64,
    mixer: union(enum) { conv: Conv, attention: Attention },
    ffn: union(enum) { dense: Dense, moe: Moe },
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: c.Config,
    /// The embedding as MLX stores it: words u32 [V, H bits / 32], scales and biases [V, H / 64]; also the head.
    embed_w: u64 = 0,
    embed_s: u64 = 0,
    embed_b: u64 = 0,
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

    fn tensor(x: *Ctx, name: []const u8, dtype: core.checkpoint.DType, shape: []const usize) !u64 {
        return x.w.upload(x.d, (try x.ck.expect(name, dtype, shape)).bytes);
    }

    /// Parts' bytes one after another into the stage; their device addresses.
    fn staged(x: *Ctx, parts: []const []const u8, out: []u64) !void {
        var at: usize = 0;
        for (parts, out) |p, *o| {
            if (at + p.len > x.stage.len) return error.StageTooSmall;
            try x.stage.upload(at, p);
            o.* = x.stage.ptr + at;
            at += (p.len + 255) / 256 * 256;
        }
    }

    /// MLX affine projections stacked along n (each [n_i, k] at `bits`) packed as one into qlinear's tiles.
    fn linear(x: *Ctx, parts: []const []const u8, ns: []const usize, k: usize, bits: u8) !Proj {
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
            at = (at + 255) / 256 * 256;
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

    /// The 64 experts' `name` ([64, n, k] at 4 bits) packed as `out`'s mi-th of m interleaved tables.
    fn experts(x: *Ctx, prefix: []const u8, name: []const u8, n: usize, k: usize, out: u64, m: usize, mi: usize) !void {
        var buf: [192]u8 = undefined;
        var parts: [3][]const u8 = undefined;
        const specs = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, k / 8 }, .{ "scales", .bf16, k / 64 }, .{ "biases", .bf16, k / 64 } };
        for (specs, &parts) |sp, *p| p.* = (try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, sp[0] }), sp[1], &.{ c.experts, n, sp[2] })).bytes;
        var raw: [3]u64 = undefined;
        try x.staged(&parts, &raw);
        try x.k.experts.packInto(x.s, raw[0], raw[1], raw[2], out, c.experts, n, k, m, mi);
        try x.s.synchronize();
    }
};

/// Bytes the stage needs: the largest of the head's table as stored and one expert table (words, scales, biases).
pub fn stageBytes(embed_bits: u8) usize {
    const head = c.vocab * c.hidden * @as(usize, embed_bits) / 8 + 2 * c.vocab * (c.hidden / 64) * 2;
    const table = c.experts * c.moe_intermediate * (c.hidden / 2 + 2 * (c.hidden / 64) * 2);
    return @max(head, table) + 4 * 256;
}

/// Every tensor checked against the admitted geometry; projections and experts at 4 bits. `layers` caps them.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *const kern.Kernels, dir: []const u8, layers: ?usize) !Weights {
    const config = try c.Config.read(gpa, io, dir);
    if (config.bits != 4) return error.UnsupportedLfmQuantization; // the experts' kernels read 4-bit tables
    var w: Weights = .{ .gpa = gpa, .config = config };
    errdefer w.deinit();
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const h = c.hidden;
    const eb = config.embed_bits;
    var stage = try DeviceBuffer.alloc(d, stageBytes(eb));
    defer stage.free();
    var x: Ctx = .{ .gpa = gpa, .w = &w, .d = d, .s = s, .k = k, .ck = &ck, .stage = stage };
    w.embed_w = try x.tensor("model.embed_tokens.weight", .u32, &.{ c.vocab, h * @as(usize, eb) / 32 });
    w.embed_s = try x.tensor("model.embed_tokens.scales", .bf16, &.{ c.vocab, h / 64 });
    w.embed_b = try x.tensor("model.embed_tokens.biases", .bf16, &.{ c.vocab, h / 64 });
    w.head = try x.linear(&.{"model.embed_tokens"}, &.{c.vocab}, h, eb);
    w.final_norm = try x.tensor("model.embedding_norm.weight", .bf16, &.{h});
    const n = @min(layers orelse c.layers, c.layers);
    w.layers = try gpa.alloc(Layer, n);
    var b1: [128]u8 = undefined;
    var b2: [128]u8 = undefined;
    var b3: [128]u8 = undefined;
    for (w.layers, 0..) |*l, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        const name = struct {
            fn of(b: []u8, prefix: []const u8, suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "{s}{s}", .{ prefix, suffix });
            }
        }.of;
        l.operator_norm = try x.tensor(try name(&b1, p, "operator_norm.weight"), .bf16, &.{h});
        l.ffn_norm = try x.tensor(try name(&b1, p, "ffn_norm.weight"), .bf16, &.{h});
        if (config.attention[i]) {
            const qw = c.query_heads * c.head_dim;
            const kw = c.kv_heads * c.head_dim;
            const norms = try w.alloc(d, 2 * c.head_dim * 2);
            const qn = try ck.expect(try name(&b1, p, "self_attn.q_layernorm.weight"), .bf16, &.{c.head_dim});
            const kn = try ck.expect(try name(&b1, p, "self_attn.k_layernorm.weight"), .bf16, &.{c.head_dim});
            try (DeviceBuffer{ .d = d, .ptr = norms, .len = 4 * c.head_dim, .borrowed = true }).upload(0, qn.bytes);
            try (DeviceBuffer{ .d = d, .ptr = norms, .len = 4 * c.head_dim, .borrowed = true }).upload(2 * c.head_dim, kn.bytes);
            l.mixer = .{ .attention = .{
                .qkv = try x.linear(&.{ try name(&b1, p, "self_attn.q_proj"), try name(&b2, p, "self_attn.k_proj"), try name(&b3, p, "self_attn.v_proj") }, &.{ qw, kw, kw }, h, 4),
                .out = try x.linear(&.{try name(&b1, p, "self_attn.out_proj")}, &.{h}, qw, 4),
                .norms = norms,
            } };
        } else l.mixer = .{ .conv = .{
            .in = try x.linear(&.{try name(&b1, p, "conv.in_proj")}, &.{3 * h}, h, 4),
            .out = try x.linear(&.{try name(&b1, p, "conv.out_proj")}, &.{h}, h, 4),
            .weight = try x.tensor(try name(&b1, p, "conv.conv.weight"), .bf16, &.{ h, c.conv_taps, 1 }),
        } };
        if (config.moe(i)) {
            const ni = c.moe_intermediate;
            const pair = try w.alloc(d, A4.pairBytes(c.experts, ni, h));
            const down = try w.alloc(d, A4.bytes(c.experts, h, ni));
            const ff = try name(&b2, p, "feed_forward.switch_mlp.");
            try x.experts(ff, "gate_proj", ni, h, pair, 2, 0);
            try x.experts(ff, "up_proj", ni, h, pair, 2, 1);
            try x.experts(ff, "down_proj", h, ni, down, 1, 0);
            l.ffn = .{ .moe = .{
                .router = try x.tensor(try name(&b1, p, "feed_forward.gate.weight"), .f32, &.{ c.experts, h }),
                .bias = try x.tensor(try name(&b1, p, "feed_forward.expert_bias"), .f32, &.{c.experts}),
                .experts = .{ .format = .affine4, .up = pair, .down = down, .width = ni, .dims = h, .experts = c.experts, .swiglu = true },
            } };
        } else l.ffn = .{ .dense = .{
            .gate_up = try x.linear(&.{ try name(&b1, p, "feed_forward.gate_proj"), try name(&b2, p, "feed_forward.up_proj") }, &.{ c.intermediate, c.intermediate }, h, 4),
            .down = try x.linear(&.{try name(&b1, p, "feed_forward.down_proj")}, &.{h}, c.intermediate, 4),
        } };
    }
    return w;
}

test "the stage holds the 6-bit head's table and one expert table" {
    try std.testing.expect(stageBytes(6) >= c.vocab * c.hidden * 6 / 8);
    try std.testing.expect(stageBytes(4) >= c.experts * c.moe_intermediate * c.hidden / 2);
}
