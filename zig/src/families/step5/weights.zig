//! Step-5's checkpoint (tools/step5_fp8_to_mlx.py) as row-projection matrices, norms and repacked routers.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const rows = @import("../../core/row_projection.zig");
const route = @import("../../core/moe_route.zig");
const cfg = @import("config.zig");

pub const Tensor = ck.Tensor;
pub const Linear = rows.Weights;

/// A full layer's sparse indexer: q (heads x proxy), k and z (proxy), fp32 head weights, q RMSNorm and k LayerNorm.
pub const Indexer = struct { q: Linear, k: Linear, z: Linear, w: Tensor, q_norm: Tensor, k_norm: Tensor, k_bias: Tensor };
pub const Attention = struct { q: Linear, k: Linear, v: Linear, o: Linear, g: Linear, q_norm: Tensor, k_norm: Tensor, indexer: ?Indexer = null };
pub const Mlp = struct { gate: Linear, up: Linear, down: Linear };
pub const Moe = struct { router: mtl.Buffer, bias: Tensor, gate: rows.Experts, up: rows.Experts, down: rows.Experts, shared: Mlp };
/// An MTP layer's own parts beside its decoder layer: eh_proj [hidden, 2 hidden], enorm, hnorm, the head and its norm.
pub const Mtp = struct { eh: Linear, enorm: Tensor, hnorm: Tensor, norm: Tensor, head: Linear };
pub const Layer = struct { input_norm: Tensor, post_norm: Tensor, attn: Attention, ffn: union(enum) { dense: Mlp, moe: Moe } };

pub const Weights = struct {
    embed: Linear,
    head: Linear,
    norm: Tensor,
    layers: []Layer, // the main stack, then the MTP layers' decoder layers
    mtp: []Mtp,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.layers) |l| if (l.ffn == .moe) l.ffn.moe.router.deinit();
        gpa.free(w.layers);
        gpa.free(w.mtp);
    }
};

fn name(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

fn tensor(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, dtype: ck.DType, count: usize) !Tensor {
    var buf: [256]u8 = undefined;
    const n = try name(&buf, prefix, suffix);
    const t = try c.get(n);
    if (t.dtype != dtype or t.count() != count) {
        std.log.err("step5: {s} is not {s}[{d}]", .{ n, @tagName(dtype), count });
        return error.UnexpectedStep5Tensor;
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
    const words = w.shape[w.rank - 1];
    const groups = s.shape[s.rank - 1];
    if (w.dtype != .u32 or s.dtype != .bf16 or z.dtype != .bf16 or w.count() != stack * n * words or s.count() != stack * n * groups or
        z.count() != s.count() or groups == 0 or k % groups != 0 or words * 32 % k != 0)
    {
        std.log.err("step5: {s} is not an affine [{d}, {d}, {d}] matrix in MLX's layout", .{ base, stack, n, k });
        return error.UnexpectedStep5Tensor;
    }
    const out = Linear{ .w = w.buffer, .w_off = w.offset, .scales = s.buffer, .s_off = s.offset, .biases = z.buffer, .b_off = z.offset, .n = n, .k = k, .group = k / groups, .bits = @intCast(words * 32 / k) };
    try out.validate();
    return out;
}

fn mlp(c: *const ck.Checkpoint, p: []const u8, hidden: usize, inter: usize) !Mlp {
    return .{ .gate = try affine(c, p, ".gate_proj", 1, inter, hidden), .up = try affine(c, p, ".up_proj", 1, inter, hidden), .down = try affine(c, p, ".down_proj", 1, hidden, inter) };
}

/// The bf16 router [experts, hidden] repacked into a new shared buffer in moe_route's block order.
fn router(device: mtl.Device, t: Tensor, experts: usize, hidden: usize) !mtl.Buffer {
    if (t.dtype != .bf16 or t.count() != experts * hidden) return error.UnexpectedStep5Tensor;
    const buf = try device.buffer(experts * hidden * 2, mtl.ResourceOptions.shared);
    route.pack(t.host(u16), buf.slice(u16, experts * hidden), experts, hidden);
    return buf;
}

pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ck.Checkpoint, f: *const cfg.Config) !Weights {
    const h: usize = f.hidden;
    var w = Weights{
        .embed = try affine(c, "model.", "embed_tokens", 1, f.vocab, h),
        .head = try affine(c, "", "lm_head", 1, f.vocab, h),
        .norm = try tensor(c, "model.", "norm.weight", .f32, h),
        .layers = try gpa.alloc(Layer, f.layers + f.mtp),
        .mtp = try gpa.alloc(Mtp, f.mtp),
    };
    errdefer gpa.free(w.mtp);
    var made: usize = 0;
    errdefer {
        for (w.layers[0..made]) |l| if (l.ffn == .moe) l.ffn.moe.router.deinit();
        gpa.free(w.layers);
    }
    const qn: usize = f.heads * f.head_dim;
    const kvn: usize = f.kvWidth();
    for (w.layers, 0..) |*l, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        var ab: [96]u8 = undefined;
        const a = try name(&ab, p, "self_attn");
        l.input_norm = try tensor(c, p, "input_layernorm.weight", .f32, h);
        l.post_norm = try tensor(c, p, "post_attention_layernorm.weight", .f32, h);
        l.attn = .{
            .q = try affine(c, a, ".q_proj", 1, qn, h),
            .k = try affine(c, a, ".k_proj", 1, kvn, h),
            .v = try affine(c, a, ".v_proj", 1, kvn, h),
            .o = try affine(c, a, ".o_proj", 1, h, qn),
            .g = try affine(c, a, ".g_proj", 1, f.heads, h),
            .q_norm = try tensor(c, a, ".q_norm.weight", .f32, f.head_dim),
            .k_norm = try tensor(c, a, ".k_norm.weight", .f32, f.head_dim),
        };
        if (f.layer[i].kind == .full) {
            const x: usize = f.index_heads * f.proxy;
            l.attn.indexer = .{
                .q = try affine(c, a, ".sparse_indexer_q", 1, x, h),
                .k = try affine(c, a, ".sparse_indexer_k", 1, f.proxy, h),
                .z = try affine(c, a, ".sparse_indexer_z", 1, f.proxy, h),
                .w = try tensor(c, a, ".sparse_indexer_w.weight", .f32, f.index_heads * h),
                .q_norm = try tensor(c, a, ".sparse_indexer_q_norm.weight", .f32, f.proxy),
                .k_norm = try tensor(c, a, ".sparse_indexer_k_norm.weight", .f32, f.proxy),
                .k_bias = try tensor(c, a, ".sparse_indexer_k_norm.bias", .f32, f.proxy),
            };
        }
        if (!f.layer[i].moe) {
            var mb: [96]u8 = undefined;
            l.ffn = .{ .dense = try mlp(c, try name(&mb, p, "mlp"), h, f.intermediate) };
            made += 1;
            continue;
        }
        var mb: [96]u8 = undefined;
        const m = try name(&mb, p, "moe");
        const e: usize = f.experts;
        const ex = struct {
            fn of(x: Linear, n: usize) rows.Experts {
                return .{ .one = x, .experts = n, .repeat = 1 };
            }
        };
        var sb: [96]u8 = undefined;
        var rb: [128]u8 = undefined;
        const r = try router(device, try c.get(try name(&rb, m, ".gate.weight")), e, h);
        errdefer r.deinit();
        l.ffn = .{ .moe = .{
            .router = r,
            .bias = try tensor(c, m, ".router_bias", .f32, e),
            .gate = ex.of(try affine(c, m, ".gate_proj", e, f.moe_intermediate, h), e),
            .up = ex.of(try affine(c, m, ".up_proj", e, f.moe_intermediate, h), e),
            .down = ex.of(try affine(c, m, ".down_proj", e, h, f.moe_intermediate), e),
            .shared = try mlp(c, try name(&sb, p, "share_expert"), h, f.shared_intermediate),
        } };
        made += 1;
    }
    for (w.mtp, 0..) |*t, k| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{f.layers + k});
        t.* = .{
            .eh = try affine(c, p, "eh_proj", 1, h, 2 * h),
            .enorm = try tensor(c, p, "enorm.weight", .f32, h),
            .hnorm = try tensor(c, p, "hnorm.weight", .f32, h),
            .norm = try tensor(c, p, "transformer.shared_head.norm.weight", .f32, h),
            .head = try affine(c, p, "transformer.shared_head.output", 1, f.vocab, h),
        };
    }
    return w;
}
