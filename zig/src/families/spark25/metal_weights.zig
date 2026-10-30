//! Spark2.5's checkpoint on Metal: the release's bf16 matrices as stored, or MLX affine ones; q|k|v|gate stacked.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const rows = @import("../../core/row_projection.zig");
const cfg = @import("config.zig");
const st = @import("metal_state.zig");

pub const Tensor = ck.Tensor;

const geo = cfg.spark4b;

/// A bf16 [n, k] matrix as the release stores it (torch's Linear weight), at `off` in `buf`.
pub const Dense = struct { buf: mtl.Buffer, off: usize, n: usize, k: usize };

/// A projection's weights: the release's bf16 rows, or one MLX affine width in groups of 64.
pub const Linear = union(enum) {
    bf16: Dense,
    affine: rows.Weights,

    pub fn n(l: Linear) usize {
        return switch (l) {
            .bf16 => |d| d.n,
            .affine => |a| a.n,
        };
    }

    pub fn k(l: Linear) usize {
        return switch (l) {
            .bf16 => |d| d.k,
            .affine => |a| a.k,
        };
    }

    /// The output row pitch in values: n to a whole 64 (the affine prompt matmul writes its padded rows).
    pub fn pitch(l: Linear) usize {
        return std.mem.alignForward(usize, l.n(), 64);
    }
};

/// q_k_v_proj and g_proj stacked (q, k, v, then a gate a head), the output projection.
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
fn affine(c: *const ck.Checkpoint, bits: u8, base: []const u8, n: usize, k: usize) !Linear {
    var b2: [256]u8 = undefined;
    const w = try c.get(try name(&b2, base, ".weight"));
    const s = try c.get(try name(&b2, base, ".scales"));
    const z = try c.get(try name(&b2, base, ".biases"));
    try require(w, .u32, &.{ n, k * bits / 32 });
    try require(s, .bf16, &.{ n, k / cfg.group });
    try require(z, .bf16, &.{ n, k / cfg.group });
    const out = rows.Weights{ .w = w.buffer, .w_off = w.offset, .scales = s.buffer, .s_off = s.offset, .biases = z.buffer, .b_off = z.offset, .n = n, .k = k, .group = cfg.group, .bits = bits };
    try out.validate();
    return .{ .affine = out };
}

/// The release's bf16 [n, k] matrix in place: k whole 256s for the GEMM's steps, 16-byte aligned for the row kernel.
fn dense(c: *const ck.Checkpoint, base: []const u8, n: usize, k: usize) !Linear {
    var b2: [256]u8 = undefined;
    const w = try c.get(try name(&b2, base, ".weight"));
    try require(w, .bf16, &.{ n, k });
    if (k % 256 != 0 or w.offset % 16 != 0) return error.UnexpectedSparkTensor;
    return .{ .bf16 = .{ .buf = w.buffer, .off = w.offset, .n = n, .k = k } };
}

/// `base`'s matrix in the checkpoint's format.
fn linear(c: *const ck.Checkpoint, bits: u8, prefix: []const u8, suffix: []const u8, n: usize, k: usize) !Linear {
    var buf: [256]u8 = undefined;
    const base = try name(&buf, prefix, suffix);
    return if (bits == cfg.bf16) dense(c, base, n, k) else affine(c, bits, base, n, k);
}

/// `parts` (one format and width) stacked in a new buffer, rows in order; affine stacks pad with zero rows to `total`.
fn stacked(gpa: std.mem.Allocator, device: mtl.Device, w: *Weights, parts: []const Linear, total: usize) !Linear {
    const k = parts[0].k();
    var n: usize = 0;
    for (parts) |p| {
        if (p.k() != k or std.meta.activeTag(p) != std.meta.activeTag(parts[0])) return error.UnexpectedSparkTensor;
        n += p.n();
    }
    if (total < n) return error.UnexpectedSparkTensor;
    switch (parts[0]) {
        .bf16 => {
            const buf = try device.buffer(n * k * 2, mtl.ResourceOptions.shared);
            errdefer buf.deinit();
            const dst = buf.contents();
            var row: usize = 0;
            for (parts) |p| {
                const d = p.bf16;
                @memcpy(dst[row * k * 2 ..][0 .. d.n * k * 2], (d.buf.contents() + d.off)[0 .. d.n * k * 2]);
                row += d.n;
            }
            try w.stacked.append(gpa, buf);
            return .{ .bf16 = .{ .buf = buf, .off = 0, .n = n, .k = k } };
        },
        .affine => |first| {
            const words = k * first.bits / 32;
            const groups = k / first.group;
            const s_at = std.mem.alignForward(usize, total * words * 4, 256);
            const b_at = std.mem.alignForward(usize, s_at + total * groups * 2, 256);
            const buf = try device.buffer(b_at + total * groups * 2, mtl.ResourceOptions.shared);
            errdefer buf.deinit();
            const dst = buf.contents();
            @memset(dst[0 .. b_at + total * groups * 2], 0); // padded rows: zero words, scales and biases, so zero outputs
            var row: usize = 0;
            for (parts) |p| {
                const a = p.affine;
                if (a.bits != first.bits or a.group != first.group) return error.UnexpectedSparkTensor;
                @memcpy(dst[row * words * 4 ..][0 .. a.n * words * 4], (a.w.contents() + a.w_off)[0 .. a.n * words * 4]);
                @memcpy(dst[s_at + row * groups * 2 ..][0 .. a.n * groups * 2], (a.scales.contents() + a.s_off)[0 .. a.n * groups * 2]);
                @memcpy(dst[b_at + row * groups * 2 ..][0 .. a.n * groups * 2], (a.biases.contents() + a.b_off)[0 .. a.n * groups * 2]);
                row += a.n;
            }
            try w.stacked.append(gpa, buf);
            const out = rows.Weights{ .w = buf, .w_off = 0, .scales = buf, .s_off = s_at, .biases = buf, .b_off = b_at, .n = total, .k = k, .group = first.group, .bits = first.bits };
            try out.validate();
            return .{ .affine = out };
        },
    }
}

pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ck.Checkpoint, f: *const cfg.Config) !Weights {
    const h = geo.hidden;
    const b = f.bits;
    var w = Weights{
        .embed = try linear(c, b, "model.", "embedding", cfg.vocab, h),
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
        const qkv = try linear(c, b, a, ".q_k_v_proj", geo.queryWidth() + 2 * geo.kvWidth(), h);
        const gates = try linear(c, b, a, ".g_proj", geo.query_heads, h);
        l.* = .{
            .input_norm = try norm(c, p, "input_layernorm.weight"),
            .post_norm = try norm(c, p, "post_attention_layernorm.weight"),
            .attn = .{ .qkvg = try stacked(gpa, device, &w, &.{ qkv, gates }, st.qkvg_rows), .out = try linear(c, b, a, ".out_proj", h, geo.queryWidth()) },
            .mlp = .{
                .gu = try stacked(gpa, device, &w, &.{ try linear(c, b, m, ".gate_proj", geo.intermediate, h), try linear(c, b, m, ".up_proj", geo.intermediate, h) }, 2 * geo.intermediate),
                .down = try linear(c, b, m, ".down_proj", h, geo.intermediate),
            },
        };
    }
    return w;
}
