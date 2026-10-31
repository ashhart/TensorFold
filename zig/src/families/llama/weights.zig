//! Llama's checkpoint on Metal: the projections as stored (bf16 [n, k] or an int4 triple), nothing copied.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const cfg = @import("config.zig");

pub const Tensor = ck.Tensor;

/// A bf16 [n, k] matrix as the release stores it, at `off` in `buf`; or, with `int4`, the packed words there.
pub const Dense = struct { buf: mtl.Buffer, off: usize, n: usize, k: usize, int4: ?Int4 = null };

/// compressed-tensors' int4 [n, k]: codes at Dense.buf (i32 [n, k / 8], element 8c + e in bits 4e of word c, + 8).
pub const Int4 = struct {
    /// weight_scale bf16 [n, k / 32]: W[j, i] = bf16((q - z) * scale), as the decompressor makes it.
    scales: Tensor,
    /// weight_zero_point i32 [n / 8, k / 32]: row 8r + e's zero point in bits 4e of word r, stored + 8.
    zeros: Tensor,
};

pub const Attention = struct { q: Dense, k: Dense, v: Dense, o: Dense };
pub const Mlp = struct { gate: Dense, up: Dense, down: Dense };
pub const Layer = struct { input_norm: Tensor, post_norm: Tensor, attn: Attention, mlp: Mlp };

pub const Weights = struct {
    embed: Dense,
    head: Dense,
    norm: Tensor,
    layers: []Layer,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        gpa.free(w.layers);
    }
};

fn name(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

/// `t` is bf16 with exactly `dims`, its bytes those dims' extent.
fn require(t: Tensor, dims: []const usize) error{UnexpectedLlamaTensor}!void {
    if (t.dtype != .bf16 or t.rank != dims.len or t.rank > t.shape.len) return error.UnexpectedLlamaTensor;
    if (!std.mem.eql(usize, t.shape[0..t.rank], dims)) return error.UnexpectedLlamaTensor;
    var bytes: usize = 2;
    for (dims) |n| bytes = std.math.mul(usize, bytes, n) catch return error.UnexpectedLlamaTensor;
    if (t.bytes != bytes) return error.UnexpectedLlamaTensor;
}

/// `t` is `dtype` with exactly `dims`, its bytes those dims' extent, 4-byte aligned.
fn requireAs(t: Tensor, dtype: ck.DType, dims: []const usize) error{UnexpectedLlamaTensor}!void {
    if (t.dtype != dtype or t.rank != dims.len or t.rank > t.shape.len or t.offset % 4 != 0) return error.UnexpectedLlamaTensor;
    if (!std.mem.eql(usize, t.shape[0..t.rank], dims)) return error.UnexpectedLlamaTensor;
    var bytes: usize = dtype.size();
    for (dims) |n| bytes = std.math.mul(usize, bytes, n) catch return error.UnexpectedLlamaTensor;
    if (t.bytes != bytes) return error.UnexpectedLlamaTensor;
}

/// The pack-quantized triple of `prefix ++ suffix` ("q_proj" ...) in place, its weight_shape [n, k].
fn int4(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, n: usize, k: usize) !Dense {
    var buf: [256]u8 = undefined;
    const g = cfg.int4_group;
    if (n % 8 != 0 or k % 256 != 0) return error.UnexpectedLlamaTensor;
    const parts = [_]struct { []const u8, ck.DType, [2]usize }{
        .{ ".weight_packed", .i32, .{ n, k / 8 } },
        .{ ".weight_scale", .bf16, .{ n, k / g } },
        .{ ".weight_zero_point", .i32, .{ n / 8, k / g } },
        .{ ".weight_shape", .i32, .{ 2, 0 } },
    };
    var got: [parts.len]Tensor = undefined;
    for (parts, &got) |p, *t| {
        const full = try std.fmt.bufPrint(&buf, "{s}{s}{s}", .{ prefix, suffix, p[0] });
        t.* = try c.get(full);
        const dims: []const usize = if (p[2][1] == 0) p[2][0..1] else &p[2];
        requireAs(t.*, p[1], dims) catch |err| {
            std.log.err("llama: {s} is not {s}{any}", .{ full, @tagName(p[1]), dims });
            return err;
        };
    }
    const shape = got[3].host(i32);
    if (shape[0] != n or shape[1] != k) {
        std.log.err("llama: {s}{s}.weight_shape is {any}, not [{d}, {d}]", .{ prefix, suffix, shape, n, k });
        return error.UnexpectedLlamaTensor;
    }
    return .{ .buf = got[0].buffer, .off = got[0].offset, .n = n, .k = k, .int4 = .{ .scales = got[1], .zeros = got[2] } };
}

fn norm(c: *const ck.Checkpoint, hidden: usize, prefix: []const u8, suffix: []const u8) !Tensor {
    var buf: [256]u8 = undefined;
    const n = try name(&buf, prefix, suffix);
    const t = try c.get(n);
    require(t, &.{hidden}) catch |err| {
        std.log.err("llama: {s} is not bf16[{d}]", .{ n, hidden });
        return err;
    };
    return t;
}

/// The release's bf16 [n, k] matrix in place: k whole 256s for the GEMM's steps, 8-byte aligned for the row kernel.
fn dense(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, n: usize, k: usize) !Dense {
    var buf: [256]u8 = undefined;
    const full = try name(&buf, prefix, suffix);
    const w = try c.get(full);
    require(w, &.{ n, k }) catch |err| {
        std.log.err("llama: {s} is not bf16[{d}, {d}]", .{ full, n, k });
        return err;
    };
    if (k % 256 != 0 or w.offset % 8 != 0) return error.UnexpectedLlamaTensor;
    return .{ .buf = w.buffer, .off = w.offset, .n = n, .k = k };
}

/// A layer's projection "<prefix><suffix>" as the config's weights store it.
const Proj = struct {
    c: *const ck.Checkpoint,
    weights: cfg.Weights,

    fn get(p: Proj, prefix: []const u8, suffix: []const u8, n: usize, k: usize) !Dense {
        var buf: [128]u8 = undefined;
        return switch (p.weights) {
            .bf16 => dense(p.c, prefix, try name(&buf, suffix, ".weight"), n, k),
            .int4_g32 => int4(p.c, prefix, suffix, n, k),
        };
    }
};

pub fn load(gpa: std.mem.Allocator, c: *const ck.Checkpoint, f: *const cfg.Config) !Weights {
    const h = f.hidden;
    var w = Weights{
        .embed = try dense(c, "model.", "embed_tokens.weight", f.vocab, h),
        .head = try dense(c, "", "lm_head.weight", f.vocab, h),
        .norm = try norm(c, h, "model.", "norm.weight"),
        .layers = try gpa.alloc(Layer, f.layers),
    };
    errdefer w.deinit(gpa);
    for (w.layers, 0..) |*l, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        var ab: [96]u8 = undefined;
        const a = try name(&ab, p, "self_attn.");
        var mb: [96]u8 = undefined;
        const m = try name(&mb, p, "mlp.");
        const proj = Proj{ .c = c, .weights = f.weights };
        l.* = .{
            .input_norm = try norm(c, h, p, "input_layernorm.weight"),
            .post_norm = try norm(c, h, p, "post_attention_layernorm.weight"),
            .attn = .{
                .q = try proj.get(a, "q_proj", f.queryWidth(), h),
                .k = try proj.get(a, "k_proj", f.kvWidth(), h),
                .v = try proj.get(a, "v_proj", f.kvWidth(), h),
                .o = try proj.get(a, "o_proj", h, f.queryWidth()),
            },
            .mlp = .{
                .gate = try proj.get(m, "gate_proj", f.intermediate, h),
                .up = try proj.get(m, "up_proj", f.intermediate, h),
                .down = try proj.get(m, "down_proj", h, f.intermediate),
            },
        };
    }
    return w;
}
