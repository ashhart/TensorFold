//! Gemma 4 dense weights: each linear stack packed once for the core lane projection, norms read in place.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const lane = @import("../../core/lane_projection.zig");
const cfg = @import("config.zig");

const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// A buffer and an offset into it.
pub const Ref = lane.Ref;

/// One packed linear (a stack of members along N): its lane layout's words and scale/bias pairs.
pub const Linear = struct { w: mtl.Buffer, sb: mtl.Buffer, n: usize, k: usize };

/// A linear as core/affine_mm reads it (prompt chunks): MLX's words, scales and biases, copied to aligned storage.
pub const Mlx = struct { w: Ref, s: Ref, b: Ref, n: u32, k: u32 };

pub const Layer = struct {
    /// q, k, v (k again on global layers, which have none), o, gate, up, down for prompt chunks.
    mlx: [7]Mlx,
    qkv: Linear,
    o: Linear,
    gate_up: Linear,
    down: Linear,
    q_norm: Ref,
    k_norm: Ref,
    input_norm: Ref,
    post_attention: Ref,
    pre_feedforward: Ref,
    post_feedforward: Ref,
    scalar: Ref,
};

/// Python's lane_qmm.split_k: K slices fixed by the weight's shape, so every window width keeps one summation order.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 31) / 32;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 1024 and (k / 64) / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// The reg kernel (4-bit groups of 64 in tensor-op registers, up to 128 rows a weight read): lane_qmm's bits.
pub fn layout(n: usize, k: usize) lane.Layout {
    return .{ .n = n, .k = k, .format = .{ .bits = cfg.bits, .group = cfg.group }, .sk = splitK(n, k), .precompute_sums = true, .reg = true };
}

pub const Weights = struct {
    layers: [cfg.layers]Layer,
    head: Linear,
    embed_w: Ref,
    embed_s: Ref,
    embed_b: Ref,
    final_norm: Ref,
    owned: std.ArrayList(mtl.Buffer) = .empty,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(gpa);
    }
};

/// A linear's three tensors copied into one aligned buffer: checkpoint tensors can sit at any byte offset in a shard.
fn mlxCopy(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, owned: *std.ArrayList(mtl.Buffer), prefix: []const u8, member: []const u8) !Mlx {
    var name: [160]u8 = undefined;
    const w = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, member }));
    const s = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.scales", .{ prefix, member }));
    const b = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.biases", .{ prefix, member }));
    const sb_at = std.mem.alignForward(usize, w.bytes, 256);
    const bb_at = sb_at + std.mem.alignForward(usize, s.bytes, 256);
    const buf = try device.buffer(bb_at + b.bytes, shared);
    errdefer buf.deinit();
    const dst = buf.slice(u8, bb_at + b.bytes);
    @memcpy(dst[0..w.bytes], w.host(u8)[0..w.bytes]);
    @memcpy(dst[sb_at..][0..s.bytes], s.host(u8)[0..s.bytes]);
    @memcpy(dst[bb_at..][0..b.bytes], b.host(u8)[0..b.bytes]);
    try owned.append(gpa, buf);
    return .{ .w = .{ .buf = buf }, .s = .{ .buf = buf, .off = sb_at }, .b = .{ .buf = buf, .off = bb_at }, .n = @intCast(w.shape[0]), .k = @intCast(w.shape[1] * 8) };
}

fn ref(c: *const ckpt.Checkpoint, name: []const u8) !Ref {
    const t = try c.get(name);
    return .{ .buf = t.buffer, .off = t.offset };
}

/// The members' MLX affine tensors stacked along N and packed into the lane layout (bytes unchanged, no dequant).
fn pack(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, owned: *std.ArrayList(mtl.Buffer), prefix: []const u8, members: []const []const u8) !Linear {
    var name: [160]u8 = undefined;
    var n: usize = 0;
    var k: usize = 0;
    for (members) |m| {
        const t = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, m }));
        if (t.dtype != .u32 or t.rank != 2) return error.BadGemmaWeight;
        if (k != 0 and k != t.shape[1] * 8) return error.BadGemmaWeight;
        n += t.shape[0];
        k = t.shape[1] * 8;
    }
    const words = try gpa.alloc(u32, n * k / 8);
    defer gpa.free(words);
    const scales = try gpa.alloc(u16, n * k / cfg.group);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u16, n * k / cfg.group);
    defer gpa.free(biases);
    var row: usize = 0;
    for (members) |m| {
        const w = (try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, m }))).host(u32);
        const s = (try c.get(try std.fmt.bufPrint(&name, "{s}{s}.scales", .{ prefix, m }))).host(u16);
        const b = (try c.get(try std.fmt.bufPrint(&name, "{s}{s}.biases", .{ prefix, m }))).host(u16);
        if (s.len != b.len or s.len * cfg.group != w.len * 8) return error.BadGemmaWeight;
        @memcpy(words[row * k / 8 ..][0..w.len], w);
        @memcpy(scales[row * k / cfg.group ..][0..s.len], s);
        @memcpy(biases[row * k / cfg.group ..][0..b.len], b);
        row += w.len * 8 / k;
    }
    return packArrays(gpa, device, owned, words, scales, biases, n, k);
}

/// MLX affine 4-bit words, scales and biases [n, k] packed into the lane layout, reg order; `owned` keeps the buffers.
pub fn packArrays(gpa: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), words: []const u32, scales: []const u16, biases: []const u16, n: usize, k: usize) !Linear {
    const l = layout(n, k);
    const wb = try device.buffer(l.weightWords() * 4, shared);
    errdefer wb.deinit();
    const sb = try device.buffer(l.metadataElements() * 2, shared);
    errdefer sb.deinit();
    try lane.pack(l, words, scales, biases, wb.slice(u32, l.weightWords()), sb.slice(u16, l.metadataElements()));
    try lane.regOrder(device, &.{.{ .w = .{ .buf = wb }, .n = n, .k = k }});
    try owned.append(gpa, wb);
    try owned.append(gpa, sb);
    return .{ .w = wb, .sb = sb, .n = n, .k = k };
}

pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint) !Weights {
    var w: Weights = undefined;
    w.owned = .empty;
    errdefer w.deinit(gpa);
    var p: [96]u8 = undefined;
    var name: [160]u8 = undefined;
    for (&w.layers, 0..) |*l, i| {
        const pre = try std.fmt.bufPrint(&p, "language_model.model.layers.{d}.", .{i});
        const qkv: []const []const u8 = if (cfg.global(i)) &.{ "self_attn.q_proj", "self_attn.k_proj" } else &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" };
        l.qkv = try pack(gpa, device, c, &w.owned, pre, qkv);
        l.o = try pack(gpa, device, c, &w.owned, pre, &.{"self_attn.o_proj"});
        l.gate_up = try pack(gpa, device, c, &w.owned, pre, &.{ "mlp.gate_proj", "mlp.up_proj" });
        l.down = try pack(gpa, device, c, &w.owned, pre, &.{"mlp.down_proj"});
        for (&l.mlx, [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" }) |*x, member| {
            x.* = if (cfg.global(i) and std.mem.eql(u8, member, "self_attn.v_proj")) l.mlx[1] else try mlxCopy(gpa, device, c, &w.owned, pre, member); // global layers: no v_proj, never read
        }
        inline for (.{ "q_norm", "k_norm", "input_norm", "post_attention", "pre_feedforward", "post_feedforward", "scalar" }, .{ "self_attn.q_norm.weight", "self_attn.k_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "layer_scalar" }) |field, t|
            @field(l, field) = try ref(c, try std.fmt.bufPrint(&name, "{s}{s}", .{ pre, t }));
    }
    w.head = try pack(gpa, device, c, &w.owned, "language_model.model.", &.{"embed_tokens"});
    w.embed_w = try ref(c, "language_model.model.embed_tokens.weight");
    w.embed_s = try ref(c, "language_model.model.embed_tokens.scales");
    w.embed_b = try ref(c, "language_model.model.embed_tokens.biases");
    w.final_norm = try ref(c, "language_model.model.norm.weight");
    return w;
}

/// The lane pipelines, one per distinct (n, k): the 31B has seven.
pub const Projections = struct {
    list: std.ArrayList(struct { n: usize, k: usize, p: lane.Projection }) = .empty,

    pub fn get(ps: *Projections, gpa: std.mem.Allocator, device: mtl.Device, n: usize, k: usize) !lane.Projection {
        for (ps.list.items) |e| if (e.n == n and e.k == k) return e.p;
        const p = try lane.Projection.init(gpa, device, layout(n, k));
        try ps.list.append(gpa, .{ .n = n, .k = k, .p = p });
        return p;
    }

    pub fn deinit(ps: *Projections, gpa: std.mem.Allocator) void {
        for (ps.list.items) |e| e.p.deinit();
        ps.list.deinit(gpa);
    }
};
