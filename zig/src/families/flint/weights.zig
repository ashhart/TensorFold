//! Flint's MLX checkpoint (RecursiveSI/flint-int4) as row-projection matrices, bf16 norms and sinks, repacked routers.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const rows = @import("../../core/row_projection.zig");
const route = @import("../../core/moe_route.zig");
const cfg = @import("config.zig");

pub const Tensor = ck.Tensor;
pub const Linear = rows.Weights;

/// q, k and v stacked into one matrix (rows q | k | v), q and k head norms [head_dim], the learned sinks [heads].
pub const Attention = struct { qkv: Linear, o: Linear, q_norm: Tensor, k_norm: Tensor, sinks: Tensor };
/// gate and up stacked into one matrix (rows gate | up).
pub const Mlp = struct { gu: Linear, down: Linear };
/// The router repacked for moe_route (its own buffer); the shared expert always runs, ungated.
pub const Moe = struct { router: mtl.Buffer, gate: rows.Experts, up: rows.Experts, down: rows.Experts, shared: Mlp };
pub const Layer = struct { input_norm: Tensor, post_norm: Tensor, attn: Attention, ffn: union(enum) { dense: Mlp, moe: Moe } };

pub const Weights = struct {
    embed: Linear,
    head: Linear,
    norm: Tensor,
    layers: []Layer,
    /// fp32 zeros, one an expert: the router's score bias, which the release does not train.
    no_bias: mtl.Buffer,
    /// The stacked matrices' own buffers (q|k|v and gate|up copies of the checkpoint's).
    stacked: std.ArrayList(mtl.Buffer) = .empty,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.layers) |l| if (l.ffn == .moe) l.ffn.moe.router.deinit();
        gpa.free(w.layers);
        w.no_bias.deinit();
        for (w.stacked.items) |b| b.deinit();
        w.stacked.deinit(gpa);
    }
};

fn name(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

/// `t` is `dtype` with exactly `dims`, its bytes those dims' extent.
fn require(t: Tensor, dtype: ck.DType, dims: []const usize) error{UnexpectedFlintTensor}!void {
    if (t.dtype != dtype or t.rank != dims.len or t.rank > t.shape.len) return error.UnexpectedFlintTensor;
    if (!std.mem.eql(usize, t.shape[0..t.rank], dims)) return error.UnexpectedFlintTensor;
    var bytes: usize = dtype.size();
    for (dims) |n| {
        if (n == 0) return error.UnexpectedFlintTensor;
        bytes = std.math.mul(usize, bytes, n) catch return error.UnexpectedFlintTensor;
    }
    if (t.bytes != bytes) return error.UnexpectedFlintTensor;
}

fn tensor(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, count: usize) !Tensor {
    var buf: [256]u8 = undefined;
    const n = try name(&buf, prefix, suffix);
    const t = try c.get(n);
    if (t.dtype != .bf16 or t.count() != count) {
        std.log.err("flint: {s} is not bf16[{d}]", .{ n, count });
        return error.UnexpectedFlintTensor;
    }
    return t;
}

/// An affine [.., n, k] matrix: bits and group read from the stored shapes, which must agree with each other.
fn affine(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, stack: usize, n: usize, k: usize) !Linear {
    var buf: [256]u8 = undefined;
    const base = try name(&buf, prefix, suffix);
    var b2: [256]u8 = undefined;
    const w = try c.get(try name(&b2, base, ".weight"));
    const s = try c.get(try name(&b2, base, ".scales"));
    const z = try c.get(try name(&b2, base, ".biases"));
    return checkedAffine(w, s, z, stack, n, k);
}

/// Packed words stored as u32, or as their bytes (u8, four a word) as this release's export writes them.
pub fn checkedAffine(w: Tensor, s: Tensor, z: Tensor, stack: usize, n: usize, k: usize) !Linear {
    if (stack == 0 or n == 0 or k == 0 or n > std.math.maxInt(i32) or k > std.math.maxInt(i32)) return error.UnexpectedFlintTensor;
    const rank: usize = if (stack == 1) 2 else 3;
    if (w.rank != rank or s.rank != rank or z.rank != rank) return error.UnexpectedFlintTensor;
    const per: usize = switch (w.dtype) {
        .u32 => 1,
        .u8 => 4,
        else => return error.UnexpectedFlintTensor,
    };
    const stored = w.shape[rank - 1];
    if (stored % per != 0) return error.UnexpectedFlintTensor;
    const words = stored / per;
    const groups = s.shape[rank - 1];
    const wd = [3]usize{ stack, n, stored };
    const sd = [3]usize{ stack, n, groups };
    try require(w, w.dtype, wd[3 - rank ..]);
    try require(s, .bf16, sd[3 - rank ..]);
    try require(z, .bf16, sd[3 - rank ..]);
    const packed_bits = std.math.mul(usize, words, 32) catch return error.UnexpectedFlintTensor;
    if (groups == 0 or k % groups != 0 or packed_bits % k != 0 or packed_bits / k > std.math.maxInt(u8)) return error.UnexpectedFlintTensor;
    const out = Linear{ .w = w.buffer, .w_off = w.offset, .scales = s.buffer, .s_off = s.offset, .biases = z.buffer, .b_off = z.offset, .n = n, .k = k, .group = k / groups, .bits = @intCast(packed_bits / k) };
    try out.validate();
    return out;
}

fn mlp(gpa: std.mem.Allocator, device: mtl.Device, w: *Weights, c: *const ck.Checkpoint, p: []const u8, hidden: usize, inter: usize) !Mlp {
    const gu = try stacked(gpa, device, w, &.{ try affine(c, p, ".gate_proj", 1, inter, hidden), try affine(c, p, ".up_proj", 1, inter, hidden) });
    return .{ .gu = gu, .down = try affine(c, p, ".down_proj", 1, hidden, inter) };
}

/// `parts` (one format and input width) stacked in a new buffer, rows in order: one launch, the same row sums.
fn stacked(gpa: std.mem.Allocator, device: mtl.Device, w: *Weights, parts: []const Linear) !Linear {
    const k = parts[0].k;
    const words = k * parts[0].bits / 32;
    const groups = k / parts[0].group;
    var n: usize = 0;
    for (parts) |p| {
        if (p.k != k or p.bits != parts[0].bits or p.group != parts[0].group) return error.UnexpectedFlintTensor;
        n += p.n;
    }
    const s_at = std.mem.alignForward(usize, n * words * 4, 256);
    const b_at = std.mem.alignForward(usize, s_at + n * groups * 2, 256);
    const buf = try device.buffer(b_at + n * groups * 2, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    const dst = buf.contents();
    var row: usize = 0;
    for (parts) |p| {
        @memcpy(dst[row * words * 4 ..][0 .. p.n * words * 4], (p.w.contents() + p.w_off)[0 .. p.n * words * 4]);
        @memcpy(dst[s_at + row * groups * 2 ..][0 .. p.n * groups * 2], (p.scales.contents() + p.s_off)[0 .. p.n * groups * 2]);
        @memcpy(dst[b_at + row * groups * 2 ..][0 .. p.n * groups * 2], (p.biases.contents() + p.b_off)[0 .. p.n * groups * 2]);
        row += p.n;
    }
    try w.stacked.append(gpa, buf);
    const out = Linear{ .w = buf, .w_off = 0, .scales = buf, .s_off = s_at, .biases = buf, .b_off = b_at, .n = n, .k = k, .group = parts[0].group, .bits = parts[0].bits };
    try out.validate();
    return out;
}

/// The bf16 router [experts, hidden] repacked into a new shared buffer in moe_route's block order.
fn router(device: mtl.Device, t: Tensor, experts: usize, hidden: usize) !mtl.Buffer {
    try require(t, .bf16, &.{ experts, hidden });
    const buf = try device.buffer(experts * hidden * 2, mtl.ResourceOptions.shared);
    route.pack(t.host(u16), buf.slice(u16, experts * hidden), experts, hidden);
    return buf;
}

pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ck.Checkpoint, f: *const cfg.Config) !Weights {
    try cfg.validateExperts(f);
    const h: usize = f.hidden;
    const no_bias = try device.buffer(@as(usize, f.experts) * 4, mtl.ResourceOptions.shared);
    errdefer no_bias.deinit();
    @memset(no_bias.slice(f32, f.experts), 0);
    var w = Weights{
        .embed = try affine(c, "model.", "embed_tokens", 1, f.vocab, h),
        .head = try affine(c, "", "lm_head", 1, f.vocab, h),
        .norm = try tensor(c, "model.", "norm.weight", h),
        .layers = try gpa.alloc(Layer, f.layers),
        .no_bias = no_bias,
    };
    var made: usize = 0;
    errdefer {
        for (w.layers[0..made]) |l| if (l.ffn == .moe) l.ffn.moe.router.deinit();
        gpa.free(w.layers);
        for (w.stacked.items) |b| b.deinit();
        w.stacked.deinit(gpa);
    }
    const qn: usize = f.heads * f.head_dim;
    const kvn: usize = f.kvWidth();
    for (w.layers, 0..) |*l, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        var ab: [96]u8 = undefined;
        const a = try name(&ab, p, "self_attn");
        l.input_norm = try tensor(c, p, "input_layernorm.weight", h);
        l.post_norm = try tensor(c, p, "post_attention_layernorm.weight", h);
        l.attn = .{
            .qkv = try stacked(gpa, device, &w, &.{ try affine(c, a, ".q_proj", 1, qn, h), try affine(c, a, ".k_proj", 1, kvn, h), try affine(c, a, ".v_proj", 1, kvn, h) }),
            .o = try affine(c, a, ".o_proj", 1, h, qn),
            .q_norm = try tensor(c, a, ".q_norm.weight", f.head_dim),
            .k_norm = try tensor(c, a, ".k_norm.weight", f.head_dim),
            .sinks = try tensor(c, a, ".softmax_offset", f.heads),
        };
        var mb: [96]u8 = undefined;
        const m = try name(&mb, p, "mlp");
        if (!f.layer[i].moe) {
            l.ffn = .{ .dense = try mlp(gpa, device, &w, c, m, h, f.intermediate) };
            made += 1;
            continue;
        }
        const e: usize = f.experts;
        const ex = struct {
            fn of(x: Linear, n: usize) rows.Experts {
                return .{ .one = x, .experts = n, .repeat = 1 };
            }
        };
        var sb: [128]u8 = undefined;
        var rb: [128]u8 = undefined;
        const r = try router(device, try c.get(try name(&rb, m, ".gate.weight")), e, h);
        errdefer r.deinit();
        var xb: [128]u8 = undefined;
        const sw = try name(&xb, m, ".switch_mlp");
        l.ffn = .{ .moe = .{
            .router = r,
            .gate = ex.of(try affine(c, sw, ".gate_proj", e, f.moe_intermediate, h), e),
            .up = ex.of(try affine(c, sw, ".up_proj", e, f.moe_intermediate, h), e),
            .down = ex.of(try affine(c, sw, ".down_proj", e, h, f.moe_intermediate), e),
            .shared = try mlp(gpa, device, &w, c, try name(&sb, m, ".shared_expert"), h, f.shared_intermediate),
        } };
        made += 1;
    }
    return w;
}
