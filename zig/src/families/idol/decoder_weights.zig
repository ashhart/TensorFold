//! Idol's decoders' tensors (fp32 as stored): the blocks' linears to fp16 at load as idol-sdk does, the rest fp32.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const dc = @import("decoder_config.zig");
const octree = @import("octree.zig");
const Ref = @import("kernels.zig").Ref;

const H = dc.hidden;
const F = dc.ffn;

/// Cross attention to the latent: fp16 to_q, to_kv, to_out and biases, fp32 RMSNorm gammas [16, 64].
pub const Cross = struct { q: Ref, q_b: Ref, kv: Ref, kv_b: Ref, out: Ref, out_b: Ref, gq: Ref, gk: Ref };
/// Self attention over the anchors: fp16 to_qkv, to_out and biases, fp32 gammas.
pub const SelfAttention = struct { qkv: Ref, qkv_b: Ref, out: Ref, out_b: Ref, gq: Ref, gk: Ref };
pub const Mlp = struct { up: Ref, up_b: Ref, down: Ref, down_b: Ref };
pub const OctreeBlock = struct { cross: Cross, mlp: Mlp };
/// A Gaussian decoder block; norm2 is its one affine LayerNorm32 (fp32).
pub const GaussianBlock = struct { self_attn: SelfAttention, cross: Cross, norm2: Ref, norm2_b: Ref, mlp: Mlp };
/// The fp32 stem and head: in_proj [1024, 3], input_layer [1024, 1024], out_proj [N, 1024], and biases.
pub const Stem = struct { in_w: Ref, in_b: Ref, input: Ref, input_b: Ref, out: Ref, out_b: Ref };

pub const Weights = struct {
    ck: ck.Checkpoint,
    octree: [dc.octree_blocks]OctreeBlock,
    gaussian: [dc.gs_blocks]GaussianBlock,
    octree_stem: Stem,
    gaussian_stem: Stem,
    level: octree.LevelWeights,
    base_offset_scale: f32,
    /// points_offset_perturbation [32, 3].
    perturbation: []const f32,
    rep: dc.Representation,

    pub fn deinit(w: *Weights) void {
        w.ck.deinit();
    }
};

/// The blocks' linear weights stay in their files: load reads each once into its fp16 copy.
fn later(name: []const u8, rank: usize) bool {
    if (rank != 2) return false;
    const owned = std.mem.startsWith(u8, name, "oct.blocks.") or std.mem.startsWith(u8, name, "gs.blocks.");
    return owned and std.mem.endsWith(u8, name, ".weight");
}

fn check(name: []const u8, dtype: ck.DType, got: []const usize, shape: []const usize) !void {
    if (dtype != .f32 or !std.mem.eql(usize, got, shape)) {
        std.log.err("idol: {s} is {t} {any}; the decoders read f32 {any}", .{ name, dtype, got, shape });
        return error.UnexpectedTensor;
    }
}

fn take(c: *const ck.Checkpoint, name: []const u8, shape: []const usize) !Ref {
    const t = try c.get(name);
    try check(name, t.dtype, t.shape[0..t.rank], shape);
    if (t.offset % 16 != 0) return error.UnalignedTensor;
    return .{ .buf = t.buffer, .off = t.offset };
}

fn floats(c: *const ck.Checkpoint, name: []const u8, shape: []const usize) ![]const f32 {
    const t = try c.get(name);
    try check(name, t.dtype, t.shape[0..t.rank], shape);
    return t.host(f32);
}

/// fp16 copies bump-allocated 256-aligned in one owned buffer.
const Halves = struct {
    c: *const ck.Checkpoint,
    buf: mtl.Buffer,
    at: usize = 0,
    stage: []f32,
    name: [96]u8 = undefined,

    fn bump(h: *Halves, n: usize) Ref {
        const r: Ref = .{ .buf = h.buf, .off = h.at };
        h.at += std.mem.alignForward(usize, n * 2, ck.alignment);
        return r;
    }

    fn dst(h: *Halves, r: Ref, n: usize) []f16 {
        return @as([*]f16, @ptrCast(@alignCast(h.buf.contents() + r.off)))[0..n];
    }

    /// The bias `prefix.leaf.bias` [n] as fp16.
    fn bias(h: *Halves, prefix: []const u8, leaf: []const u8, n: usize) !Ref {
        const name = try std.fmt.bufPrint(&h.name, "{s}.{s}.bias", .{ prefix, leaf });
        const src = try floats(h.c, name, &.{n});
        const r = h.bump(n);
        for (h.dst(r, n), src) |*o, v| o.* = @floatCast(v);
        return r;
    }

    /// The deferred weight `prefix.leaf.weight` [rows, cols], read in staged pieces and rounded to fp16.
    fn weight(h: *Halves, prefix: []const u8, leaf: []const u8, rows: usize, cols: usize) !Ref {
        const name = try std.fmt.bufPrint(&h.name, "{s}.{s}.weight", .{ prefix, leaf });
        const d = h.c.deferred.get(name) orelse return error.MissingTensor;
        try check(name, d.dtype, d.shape[0..d.rank], &.{ rows, cols });
        const n = rows * cols;
        const r = h.bump(n);
        const out = h.dst(r, n);
        var done: usize = 0;
        while (done < n) {
            const part = @min(h.stage.len, n - done);
            try h.c.read(name, done * 4, std.mem.sliceAsBytes(h.stage[0..part]));
            for (out[done..][0..part], h.stage[0..part]) |*o, v| o.* = @floatCast(v);
            done += part;
        }
        return r;
    }
};

fn sized(comptime parts: []const usize) usize {
    var n: usize = 0;
    for (parts) |v| n += std.mem.alignForward(usize, v * 2, ck.alignment);
    return n;
}

const cross_halves = [_]usize{ H * H, H, 2 * H * dc.context_channels, 2 * H, H * H, H };
const mlp_halves = [_]usize{ F * H, F, H * F, H };
const self_halves = [_]usize{ 3 * H * H, 3 * H, H * H, H };
const octree_bytes = sized(&(cross_halves ++ mlp_halves));
const gaussian_bytes = sized(&(cross_halves ++ mlp_halves ++ self_halves));
const fp16_bytes = dc.octree_blocks * octree_bytes + dc.gs_blocks * gaussian_bytes;

fn gamma(c: *const ck.Checkpoint, buf: *[96]u8, prefix: []const u8, leaf: []const u8) !Ref {
    return take(c, try std.fmt.bufPrint(buf, "{s}.{s}.gamma", .{ prefix, leaf }), &.{ dc.heads, H / dc.heads });
}

fn cross(h: *Halves, block: []const u8) !Cross {
    var p: [96]u8 = undefined;
    const pre = try std.fmt.bufPrint(&p, "{s}.cross_attn", .{block});
    var g: [96]u8 = undefined;
    return .{
        .q = try h.weight(pre, "to_q", H, H),
        .q_b = try h.bias(pre, "to_q", H),
        .kv = try h.weight(pre, "to_kv", 2 * H, dc.context_channels),
        .kv_b = try h.bias(pre, "to_kv", 2 * H),
        .out = try h.weight(pre, "to_out", H, H),
        .out_b = try h.bias(pre, "to_out", H),
        .gq = try gamma(h.c, &g, pre, "q_rms_norm"),
        .gk = try gamma(h.c, &g, pre, "k_rms_norm"),
    };
}

fn mlp(h: *Halves, block: []const u8) !Mlp {
    var p: [96]u8 = undefined;
    const pre = try std.fmt.bufPrint(&p, "{s}.mlp.mlp", .{block});
    return .{ .up = try h.weight(pre, "0", F, H), .up_b = try h.bias(pre, "0", F), .down = try h.weight(pre, "2", H, F), .down_b = try h.bias(pre, "2", H) };
}

fn stem(c: *const ck.Checkpoint, prefix: []const u8, out: usize) !Stem {
    var n: [96]u8 = undefined;
    return .{
        .in_w = try take(c, try std.fmt.bufPrint(&n, "{s}.in_proj.weight", .{prefix}), &.{ H, 3 }),
        .in_b = try take(c, try std.fmt.bufPrint(&n, "{s}.in_proj.bias", .{prefix}), &.{H}),
        .input = try take(c, try std.fmt.bufPrint(&n, "{s}.input_layer.weight", .{prefix}), &.{ H, H }),
        .input_b = try take(c, try std.fmt.bufPrint(&n, "{s}.input_layer.bias", .{prefix}), &.{H}),
        .out = try take(c, try std.fmt.bufPrint(&n, "{s}.out_proj.weight", .{prefix}), &.{ out, H }),
        .out_b = try take(c, try std.fmt.bufPrint(&n, "{s}.out_proj.bias", .{prefix}), &.{out}),
    };
}

/// The snapshot folder `dir`: config.json's decoders checked, then both decoders' safetensors, each tensor once.
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
    var w: Weights = undefined;
    w.rep = try dc.check(gpa, config);
    w.ck = ck.Checkpoint.init(gpa);
    errdefer w.ck.deinit();
    for ([_][2][]const u8{ .{ "decoder", "oct." }, .{ "decoder_gs", "gs." } }) |f| {
        const path = try std.fmt.allocPrintSentinel(arena, "{s}/{s}.safetensors", .{ dir, f[0] }, 0);
        try w.ck.addFileSplit(device, path, f[1], null, later);
    }
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const fp16 = try device.buffer(fp16_bytes, opts);
    w.ck.own(fp16, fp16_bytes) catch |err| {
        fp16.deinit();
        return err;
    };
    var h: Halves = .{ .c = &w.ck, .buf = fp16, .stage = try arena.alloc(f32, 4 << 20) };
    var name: [96]u8 = undefined;
    for (&w.octree, 0..) |*b, i| {
        const pre = try std.fmt.bufPrint(&name, "oct.blocks.{d}", .{i});
        b.cross = try cross(&h, pre);
        b.mlp = try mlp(&h, pre);
    }
    var g: [96]u8 = undefined;
    for (&w.gaussian, 0..) |*b, i| {
        const pre = try std.fmt.bufPrint(&name, "gs.blocks.{d}", .{i});
        var sp: [96]u8 = undefined;
        const sa = try std.fmt.bufPrint(&sp, "{s}.self_attn", .{pre});
        b.self_attn = .{
            .qkv = try h.weight(sa, "to_qkv", 3 * H, H),
            .qkv_b = try h.bias(sa, "to_qkv", 3 * H),
            .out = try h.weight(sa, "to_out", H, H),
            .out_b = try h.bias(sa, "to_out", H),
            .gq = try gamma(&w.ck, &g, sa, "q_rms_norm"),
            .gk = try gamma(&w.ck, &g, sa, "k_rms_norm"),
        };
        b.cross = try cross(&h, pre);
        b.norm2 = try take(&w.ck, try std.fmt.bufPrint(&g, "{s}.norm2.weight", .{pre}), &.{H});
        b.norm2_b = try take(&w.ck, try std.fmt.bufPrint(&g, "{s}.norm2.bias", .{pre}), &.{H});
        b.mlp = try mlp(&h, pre);
    }
    std.debug.assert(h.at == fp16_bytes);
    w.ck.closeFiles();
    w.octree_stem = try stem(&w.ck, "oct", dc.children);
    w.gaussian_stem = try stem(&w.ck, "gs", dc.raw_width);
    w.level = .{
        .l1 = try floats(&w.ck, "oct.l_embedder.mlp.0.weight", &.{ H, dc.level_freqs }),
        .l1_b = try floats(&w.ck, "oct.l_embedder.mlp.0.bias", &.{H}),
        .l2 = try floats(&w.ck, "oct.l_embedder.mlp.2.weight", &.{ H, H }),
        .l2_b = try floats(&w.ck, "oct.l_embedder.mlp.2.bias", &.{H}),
        .ada = try floats(&w.ck, "oct.adaLN_modulation.1.weight", &.{ 6 * H, H }),
        .ada_b = try floats(&w.ck, "oct.adaLN_modulation.1.bias", &.{6 * H}),
    };
    w.base_offset_scale = (try floats(&w.ck, "gs.base_offset_scale", &.{}))[0];
    w.perturbation = try floats(&w.ck, "gs.points_offset_perturbation", &.{ dc.per_anchor, 3 });
    // the octree's 12 a block and 12 more; the Gaussian decoder's 20 a block and 8 more
    const read = 12 * dc.octree_blocks + 12 + 20 * dc.gs_blocks + 8;
    const total = w.ck.tensors.count() + w.ck.deferred.count();
    if (total != read) {
        std.log.err("idol: the decoders' files have {d} tensors; this family reads {d}", .{ total, read });
        return error.UnexpectedTensors;
    }
    return w;
}

test "the blocks' linear weights, and only they, stay in their files for their fp16 copies" {
    try std.testing.expect(later("oct.blocks.3.cross_attn.to_kv.weight", 2));
    try std.testing.expect(later("gs.blocks.15.self_attn.to_qkv.weight", 2));
    try std.testing.expect(!later("gs.blocks.0.cross_attn.q_rms_norm.gamma", 2));
    try std.testing.expect(!later("gs.blocks.0.norm2.weight", 1));
    try std.testing.expect(!later("gs.input_layer.weight", 2));
    try std.testing.expect(!later("oct.adaLN_modulation.1.weight", 2));
}
