//! Llama's checkpoint on Metal: the release's bf16 matrices as stored (torch's Linear [n, k]), nothing copied.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const cfg = @import("config.zig");

pub const Tensor = ck.Tensor;

/// A bf16 [n, k] matrix as the release stores it, at `off` in `buf`.
pub const Dense = struct { buf: mtl.Buffer, off: usize, n: usize, k: usize };

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
        l.* = .{
            .input_norm = try norm(c, h, p, "input_layernorm.weight"),
            .post_norm = try norm(c, h, p, "post_attention_layernorm.weight"),
            .attn = .{
                .q = try dense(c, a, "q_proj.weight", f.queryWidth(), h),
                .k = try dense(c, a, "k_proj.weight", f.kvWidth(), h),
                .v = try dense(c, a, "v_proj.weight", f.kvWidth(), h),
                .o = try dense(c, a, "o_proj.weight", h, f.queryWidth()),
            },
            .mlp = .{
                .gate = try dense(c, m, "gate_proj.weight", f.intermediate, h),
                .up = try dense(c, m, "up_proj.weight", f.intermediate, h),
                .down = try dense(c, m, "down_proj.weight", h, f.intermediate),
            },
        };
    }
    return w;
}
