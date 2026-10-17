//! Gemma 4 dense weights: each linear stack packed once for the core lane projection, norms read in place.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const lane = @import("../../core/lane_projection.zig");
const ks = @import("kernel_sources");
const fused = @import("fused_lane.zig");
const cfg = @import("config.zig");

const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// A buffer and an offset into it.
pub const Ref = lane.Ref;

/// One packed linear (a stack of members along N): its lane layout's words and scale/bias pairs.
pub const Linear = struct { w: mtl.Buffer, sb: mtl.Buffer, n: usize, k: usize };

/// MLX words, scales and biases copied to aligned storage for Gemma's private prompt projections.
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
pub fn referenceSplit(n: usize, k: usize) usize {
    const tiles = (n + 31) / 32;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 1024 and (k / 64) / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// The split each shape streams fastest at on one row (tf-gemma4-bench GEMMA_FUSED), not the reference's order.
pub fn splitK(geo: cfg.Geometry, n: usize, k: usize) usize {
    if (k == geo.hidden and n >= 2 * geo.intermediate) return 8; // the gate/up stack and the head: long chains at 1
    if (n == geo.hidden) return 4; // o and down: two waves of 256-thread groups at 8
    return referenceSplit(n, k);
}

/// The reg kernel (4-bit groups of 64 in tensor-op registers, up to 128 rows a weight read): lane_qmm's words.
pub fn layout(geo: cfg.Geometry, n: usize, k: usize) lane.Layout {
    return .{ .n = n, .k = k, .format = .{ .bits = cfg.bits, .group = cfg.group }, .sk = splitK(geo, n, k), .precompute_sums = true, .reg = true };
}

/// The same kernel with a layer's glue as its prologue (gemma4/fused.metal's TF_FUSE kinds) for one-row windows.
pub fn fusedLayout(geo: cfg.Geometry, n: usize, k: usize, fuse: u8) fused.Layout {
    const l = layout(geo, n, k);
    return .{ .projection = l, .source = ks.gemma4_fused, .defines = switch (fuse) {
        1 => "#define TF_FUSE 1\n",
        2 => "#define TF_FUSE 2\n",
        else => "#define TF_FUSE 3\n",
    }, .x_rows = if (fuse == 3) 0 else 1, .epilogue = fuse == 2, .threads = if (fuse == 3) 32 * l.sk else 256 };
}

pub const Weights = struct {
    /// The geometry's layers first; the rest unset.
    layers: [cfg.max_layers]Layer,
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

/// The members' MLX tensors stacked along N into the lane layout; `interleave`: two members in alternate 16-row blocks.
fn pack(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, owned: *std.ArrayList(mtl.Buffer), geo: cfg.Geometry, prefix: []const u8, members: []const []const u8, interleave: bool) !Linear {
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
    if (interleave and (members.len != 2 or n % 32 != 0)) return error.BadGemmaWeight;
    var row: usize = 0;
    for (members, 0..) |m, mi| {
        const w = (try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, m }))).host(u32);
        const s = (try c.get(try std.fmt.bufPrint(&name, "{s}{s}.scales", .{ prefix, m }))).host(u16);
        const b = (try c.get(try std.fmt.bufPrint(&name, "{s}{s}.biases", .{ prefix, m }))).host(u16);
        if (s.len != b.len or s.len * cfg.group != w.len * 8) return error.BadGemmaWeight;
        const rows = w.len * 8 / k;
        if (interleave and rows != n / 2) return error.BadGemmaWeight;
        for (0..rows) |r| {
            const dst = if (interleave) 32 * (r / 16) + 16 * mi + r % 16 else row + r;
            @memcpy(words[dst * k / 8 ..][0 .. k / 8], w[r * k / 8 ..][0 .. k / 8]);
            @memcpy(scales[dst * k / cfg.group ..][0 .. k / cfg.group], s[r * k / cfg.group ..][0 .. k / cfg.group]);
            @memcpy(biases[dst * k / cfg.group ..][0 .. k / cfg.group], b[r * k / cfg.group ..][0 .. k / cfg.group]);
        }
        row += rows;
    }
    return packArrays(gpa, device, owned, geo, words, scales, biases, n, k);
}

/// MLX affine 4-bit words, scales and biases [n, k] packed into the lane layout, reg order; `owned` keeps the buffers.
pub fn packArrays(gpa: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), geo: cfg.Geometry, words: []const u32, scales: []const u16, biases: []const u16, n: usize, k: usize) !Linear {
    const l = layout(geo, n, k);
    const wb = try device.buffer(l.weightWords() * 4, shared);
    errdefer wb.deinit();
    const sb = try device.buffer(l.metadataElements() * 2, shared);
    errdefer sb.deinit();
    try lane.pack(l, words, scales, biases, wb.slice(u32, l.weightWords()), sb.slice(u16, l.metadataElements()));
    try lane.regOrder(device, &.{.{ .w = .{ .buf = wb }, .n = n, .k = k }});
    try owned.ensureUnusedCapacity(gpa, 2);
    owned.appendAssumeCapacity(wb);
    owned.appendAssumeCapacity(sb);
    return .{ .w = wb, .sb = sb, .n = n, .k = k };
}

pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, geo: cfg.Geometry) !Weights {
    var w: Weights = undefined;
    w.owned = .empty;
    errdefer w.deinit(gpa);
    var p: [96]u8 = undefined;
    var name: [160]u8 = undefined;
    for (w.layers[0..geo.layers], 0..) |*l, i| {
        const pre = try std.fmt.bufPrint(&p, "language_model.model.layers.{d}.", .{i});
        const qkv: []const []const u8 = if (cfg.global(i)) &.{ "self_attn.q_proj", "self_attn.k_proj" } else &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" };
        l.qkv = try pack(gpa, device, c, &w.owned, geo, pre, qkv, false);
        l.o = try pack(gpa, device, c, &w.owned, geo, pre, &.{"self_attn.o_proj"}, false);
        l.gate_up = try pack(gpa, device, c, &w.owned, geo, pre, &.{ "mlp.gate_proj", "mlp.up_proj" }, true);
        l.down = try pack(gpa, device, c, &w.owned, geo, pre, &.{"mlp.down_proj"}, false);
        for (&l.mlx, [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" }) |*x, member| {
            x.* = if (cfg.global(i) and std.mem.eql(u8, member, "self_attn.v_proj")) l.mlx[1] else try mlxCopy(gpa, device, c, &w.owned, pre, member); // global layers: no v_proj, never read
        }
        inline for (.{ "q_norm", "k_norm", "input_norm", "post_attention", "pre_feedforward", "post_feedforward", "scalar" }, .{ "self_attn.q_norm.weight", "self_attn.k_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "layer_scalar" }) |field, t|
            @field(l, field) = try ref(c, try std.fmt.bufPrint(&name, "{s}{s}", .{ pre, t }));
    }
    w.head = try pack(gpa, device, c, &w.owned, geo, "language_model.model.", &.{"embed_tokens"}, false);
    w.embed_w = try ref(c, "language_model.model.embed_tokens.weight");
    w.embed_s = try ref(c, "language_model.model.embed_tokens.scales");
    w.embed_b = try ref(c, "language_model.model.embed_tokens.biases");
    w.final_norm = try ref(c, "language_model.model.norm.weight");
    return w;
}

/// The lane pipelines, one per distinct (n, k): seven, and four fused ones for one-row windows.
pub const Projections = struct {
    geo: cfg.Geometry,
    list: std.ArrayList(struct { n: usize, k: usize, p: lane.Projection }) = .empty,
    fused_list: std.ArrayList(struct { n: usize, k: usize, fuse: u8, p: fused.Projection }) = .empty,
    pub fn get(ps: *Projections, gpa: std.mem.Allocator, device: mtl.Device, n: usize, k: usize) !lane.Projection {
        for (ps.list.items) |v| if (v.n == n and v.k == k) return v.p;
        const p = try lane.Projection.init(gpa, device, layout(ps.geo, n, k));
        errdefer p.deinit();
        try ps.list.append(gpa, .{ .n = n, .k = k, .p = p });
        return p;
    }
    pub fn getFused(ps: *Projections, gpa: std.mem.Allocator, device: mtl.Device, n: usize, k: usize, fuse: u8) !fused.Projection {
        for (ps.fused_list.items) |v| if (v.n == n and v.k == k and v.fuse == fuse) return v.p;
        const p = try fused.Projection.init(gpa, device, fusedLayout(ps.geo, n, k, fuse));
        errdefer p.deinit();
        try ps.fused_list.append(gpa, .{ .n = n, .k = k, .fuse = fuse, .p = p });
        return p;
    }
    pub fn deinit(ps: *Projections, gpa: std.mem.Allocator) void {
        for (ps.list.items) |v| v.p.deinit();
        ps.list.deinit(gpa);
        for (ps.fused_list.items) |v| v.p.deinit();
        ps.fused_list.deinit(gpa);
    }
};
