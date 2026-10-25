//! Spark2.5's MLX affine checkpoint on Metal as row-projection matrices: q|k|v|gate and gate|up stacked, bf16 norms.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const rows = @import("../../core/row_projection.zig");
const cfg = @import("config.zig");
const st = @import("metal_state.zig");

pub const Tensor = ck.Tensor;
pub const Linear = rows.Weights;

const geo = cfg.spark4b;

/// q_k_v_proj and g_proj stacked (zero rows after the gates, to st.qkvg_rows), the output projection.
pub const Attention = struct { qkvg: Linear, out: Linear };
/// gate and up stacked into one matrix (rows gate | up).
pub const Mlp = struct { gu: Linear, down: Linear };
pub const Layer = struct { input_norm: Tensor, post_norm: Tensor, attn: Attention, mlp: Mlp };

pub const Weights = struct {
    embed: Linear, // also the tied head
    norm: Tensor,
    layers: []Layer,
    /// The stacked matrices' own buffers (copies of the checkpoint's).
    stacked: std.ArrayList(mtl.Buffer) = .empty,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        gpa.free(w.layers);
        for (w.stacked.items) |b| b.deinit();
        w.stacked.deinit(gpa);
    }
};

fn name(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

/// `t` is `dtype` with exactly `dims`, its bytes those dims' extent.
fn require(t: Tensor, dtype: ck.DType, dims: []const usize) error{UnexpectedSparkTensor}!void {
    if (t.dtype != dtype or t.rank != dims.len or t.rank > t.shape.len) return error.UnexpectedSparkTensor;
    if (!std.mem.eql(usize, t.shape[0..t.rank], dims)) return error.UnexpectedSparkTensor;
    var bytes: usize = dtype.size();
    for (dims) |n| bytes = std.math.mul(usize, bytes, n) catch return error.UnexpectedSparkTensor;
    if (t.bytes != bytes) return error.UnexpectedSparkTensor;
}

fn norm(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8) !Tensor {
    var buf: [256]u8 = undefined;
    const n = try name(&buf, prefix, suffix);
    const t = try c.get(n);
    require(t, .bf16, &.{geo.hidden}) catch |err| {
        std.log.err("spark2_5: {s} is not bf16[{d}]", .{ n, geo.hidden });
        return err;
    };
    return t;
}

/// An affine [n, k] matrix at the checkpoint's one width, groups of 64, words stored as u32.
fn affine(c: *const ck.Checkpoint, bits: u8, prefix: []const u8, suffix: []const u8, n: usize, k: usize) !Linear {
    var buf: [256]u8 = undefined;
    const base = try name(&buf, prefix, suffix);
    var b2: [256]u8 = undefined;
    const w = try c.get(try name(&b2, base, ".weight"));
    const s = try c.get(try name(&b2, base, ".scales"));
    const z = try c.get(try name(&b2, base, ".biases"));
    try require(w, .u32, &.{ n, k * bits / 32 });
    try require(s, .bf16, &.{ n, k / cfg.group });
    try require(z, .bf16, &.{ n, k / cfg.group });
    const out = Linear{ .w = w.buffer, .w_off = w.offset, .scales = s.buffer, .s_off = s.offset, .biases = z.buffer, .b_off = z.offset, .n = n, .k = k, .group = cfg.group, .bits = bits };
    try out.validate();
    return out;
}

/// `parts` (one format and input width) stacked in a new buffer, rows in order, zero rows after them to `total`.
fn stacked(gpa: std.mem.Allocator, device: mtl.Device, w: *Weights, parts: []const Linear, total: usize) !Linear {
    const k = parts[0].k;
    const words = k * parts[0].bits / 32;
    const groups = k / parts[0].group;
    var n: usize = 0;
    for (parts) |p| {
        if (p.k != k or p.bits != parts[0].bits or p.group != parts[0].group) return error.UnexpectedSparkTensor;
        n += p.n;
    }
    if (total < n) return error.UnexpectedSparkTensor;
    const s_at = std.mem.alignForward(usize, total * words * 4, 256);
    const b_at = std.mem.alignForward(usize, s_at + total * groups * 2, 256);
    const buf = try device.buffer(b_at + total * groups * 2, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    const dst = buf.contents();
    @memset(dst[0 .. b_at + total * groups * 2], 0); // padded rows: zero words, scales and biases, so zero outputs
    var row: usize = 0;
    for (parts) |p| {
        @memcpy(dst[row * words * 4 ..][0 .. p.n * words * 4], (p.w.contents() + p.w_off)[0 .. p.n * words * 4]);
        @memcpy(dst[s_at + row * groups * 2 ..][0 .. p.n * groups * 2], (p.scales.contents() + p.s_off)[0 .. p.n * groups * 2]);
        @memcpy(dst[b_at + row * groups * 2 ..][0 .. p.n * groups * 2], (p.biases.contents() + p.b_off)[0 .. p.n * groups * 2]);
        row += p.n;
    }
    try w.stacked.append(gpa, buf);
    const out = Linear{ .w = buf, .w_off = 0, .scales = buf, .s_off = s_at, .biases = buf, .b_off = b_at, .n = total, .k = k, .group = parts[0].group, .bits = parts[0].bits };
    try out.validate();
    return out;
}

pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ck.Checkpoint, f: *const cfg.Config) !Weights {
    const h = geo.hidden;
    const b = f.bits;
    var w = Weights{
        .embed = try affine(c, b, "model.", "embedding", cfg.vocab, h),
        .norm = try norm(c, "model.", "norm.weight"),
        .layers = try gpa.alloc(Layer, geo.layers),
    };
    errdefer w.deinit(gpa);
    for (w.layers, 0..) |*l, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        var ab: [96]u8 = undefined;
        const a = try name(&ab, p, "self_attn");
        var mb: [96]u8 = undefined;
        const m = try name(&mb, p, "mlp");
        const qkv = try affine(c, b, a, ".q_k_v_proj", geo.queryWidth() + 2 * geo.kvWidth(), h);
        const gates = try affine(c, b, a, ".g_proj", geo.query_heads, h);
        l.* = .{
            .input_norm = try norm(c, p, "input_layernorm.weight"),
            .post_norm = try norm(c, p, "post_attention_layernorm.weight"),
            .attn = .{ .qkvg = try stacked(gpa, device, &w, &.{ qkv, gates }, st.qkvg_rows), .out = try affine(c, b, a, ".out_proj", h, geo.queryWidth()) },
            .mlp = .{
                .gu = try stacked(gpa, device, &w, &.{ try affine(c, b, m, ".gate_proj", geo.intermediate, h), try affine(c, b, m, ".up_proj", geo.intermediate, h) }, 2 * geo.intermediate),
                .down = try affine(c, b, m, ".down_proj", h, geo.intermediate),
            },
        };
    }
    return w;
}
