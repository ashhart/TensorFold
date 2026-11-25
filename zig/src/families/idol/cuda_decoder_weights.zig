//! Idol's decoders' tensors on CUDA: the blocks' linears and biases to fp16 at load as idol-sdk does, the rest fp32.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const dc = @import("decoder_config.zig");
const octree = @import("octree.zig");

const H = dc.hidden;
const F = dc.ffn;

/// Cross attention to the latent: fp16 to_q, to_kv, to_out and biases, fp32 RMSNorm gammas [16, 64].
pub const Cross = struct { q: u64, q_b: u64, kv: u64, kv_b: u64, out: u64, out_b: u64, gq: u64, gk: u64 };
/// Self attention over the anchors: fp16 to_qkv, to_out and biases, fp32 gammas.
pub const SelfAttention = struct { qkv: u64, qkv_b: u64, out: u64, out_b: u64, gq: u64, gk: u64 };
pub const Mlp = struct { up: u64, up_b: u64, down: u64, down_b: u64 };
pub const OctreeBlock = struct { cross: Cross, mlp: Mlp };
/// A Gaussian decoder block; norm2 is its one affine LayerNorm32 (fp32).
pub const GaussianBlock = struct { self_attn: SelfAttention, cross: Cross, norm2: u64, norm2_b: u64, mlp: Mlp };
/// The fp32 stem and head: in_proj [1024, 3], input_layer [1024, 1024], out_proj [N, 1024], and biases.
pub const Stem = struct { in_w: u64, in_b: u64, input: u64, input_b: u64, out: u64, out_b: u64 };

/// One decoder's file and the device copies made from it.
const File = struct {
    gpa: std.mem.Allocator,
    ck: *core.Checkpoint,
    buffers: *std.ArrayList(cuda.DeviceBuffer),
    bytes: *u64,
    d: *const cuda.Driver,
    stage: []f16,
    name: [96]u8 = undefined,

    fn typed(f: *File, name: []const u8, shape: []const usize) !core.checkpoint.Tensor {
        const t = try f.ck.get(name);
        if (t.dtype != .f32 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.log.err("idol: {s} is {t} {any}; the decoders read f32 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
            return error.UnexpectedTensor;
        }
        if (@intFromPtr(t.bytes.ptr) % 4 != 0) return error.UnalignedTensor;
        return t;
    }

    /// The fp32 tensor `name` of `shape`, its values on the host (mapped from the checkpoint).
    fn floats(f: *File, name: []const u8, shape: []const usize) ![]const f32 {
        const t = try f.typed(name, shape);
        return @alignCast(std.mem.bytesAsSlice(f32, t.bytes));
    }

    fn upload(f: *File, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(f.d, bytes);
        errdefer b.free();
        try f.buffers.append(f.gpa, b);
        f.bytes.* += b.len;
        return b.ptr;
    }

    /// The fp32 tensor `prefix.leaf` of `shape` on the device as stored.
    fn take(f: *File, prefix: []const u8, leaf: []const u8, shape: []const usize) !u64 {
        const name = try std.fmt.bufPrint(&f.name, "{s}{s}", .{ prefix, leaf });
        return f.upload(std.mem.sliceAsBytes(try f.floats(name, shape)));
    }

    /// The fp32 tensor `prefix.leaf` of `shape` on the device rounded to fp16.
    fn half(f: *File, prefix: []const u8, leaf: []const u8, shape: []const usize) !u64 {
        const name = try std.fmt.bufPrint(&f.name, "{s}{s}", .{ prefix, leaf });
        const src = try f.floats(name, shape);
        for (f.stage[0..src.len], src) |*o, v| o.* = @floatCast(v);
        return f.upload(std.mem.sliceAsBytes(f.stage[0..src.len]));
    }

    fn cross(f: *File, block: []const u8) !Cross {
        var p: [96]u8 = undefined;
        const pre = try std.fmt.bufPrint(&p, "{s}.cross_attn.", .{block});
        return .{
            .q = try f.half(pre, "to_q.weight", &.{ H, H }),
            .q_b = try f.half(pre, "to_q.bias", &.{H}),
            .kv = try f.half(pre, "to_kv.weight", &.{ 2 * H, dc.context_channels }),
            .kv_b = try f.half(pre, "to_kv.bias", &.{2 * H}),
            .out = try f.half(pre, "to_out.weight", &.{ H, H }),
            .out_b = try f.half(pre, "to_out.bias", &.{H}),
            .gq = try f.take(pre, "q_rms_norm.gamma", &.{ dc.heads, H / dc.heads }),
            .gk = try f.take(pre, "k_rms_norm.gamma", &.{ dc.heads, H / dc.heads }),
        };
    }

    fn mlp(f: *File, block: []const u8) !Mlp {
        var p: [96]u8 = undefined;
        const pre = try std.fmt.bufPrint(&p, "{s}.mlp.mlp.", .{block});
        return .{ .up = try f.half(pre, "0.weight", &.{ F, H }), .up_b = try f.half(pre, "0.bias", &.{F}), .down = try f.half(pre, "2.weight", &.{ H, F }), .down_b = try f.half(pre, "2.bias", &.{H}) };
    }

    fn stem(f: *File, out: usize) !Stem {
        return .{
            .in_w = try f.take("", "in_proj.weight", &.{ H, 3 }),
            .in_b = try f.take("", "in_proj.bias", &.{H}),
            .input = try f.take("", "input_layer.weight", &.{ H, H }),
            .input_b = try f.take("", "input_layer.bias", &.{H}),
            .out = try f.take("", "out_proj.weight", &.{ out, H }),
            .out_b = try f.take("", "out_proj.bias", &.{out}),
        };
    }

    /// Every tensor of the file read, or the load refused.
    fn whole(f: *const File, file: []const u8) !void {
        if (f.ck.unused() != 0) {
            std.log.err("idol: {s} has {d} tensors the decoders do not read", .{ file, f.ck.unused() });
            return error.UnexpectedTensors;
        }
    }
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    oct: core.Checkpoint,
    gs: core.Checkpoint,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    octree: [dc.octree_blocks]OctreeBlock = undefined,
    gaussian: [dc.gs_blocks]GaussianBlock = undefined,
    octree_stem: Stem = undefined,
    gaussian_stem: Stem = undefined,
    /// The level embedding and its modulation on the host (octree.levelMods), mapped from decoder.safetensors.
    level: octree.LevelWeights = undefined,
    base_offset_scale: f32 = 0,
    /// points_offset_perturbation [32, 3], mapped from decoder_gs.safetensors.
    perturbation: []const f32 = &.{},
    rep: dc.Representation = undefined,
    bytes: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gs.close();
        w.oct.close();
    }
};

/// The snapshot folder `dir`: config.json's decoders checked, then both decoders' safetensors, every tensor read.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
    const w = try gpa.create(Weights);
    errdefer gpa.destroy(w);
    w.* = .{ .gpa = gpa, .oct = .{ .gpa = gpa, .io = io }, .gs = .{ .gpa = gpa, .io = io } };
    errdefer w.deinit();
    w.rep = try dc.check(gpa, config);
    try w.oct.add(dir, "decoder.safetensors");
    try w.gs.add(dir, "decoder_gs.safetensors");
    const stage = try arena.alloc(f16, F * H);
    var name: [96]u8 = undefined;
    var oct: File = .{ .gpa = gpa, .ck = &w.oct, .buffers = &w.buffers, .bytes = &w.bytes, .d = d, .stage = stage };
    for (&w.octree, 0..) |*b, i| {
        const pre = try std.fmt.bufPrint(&name, "blocks.{d}", .{i});
        b.cross = try oct.cross(pre);
        b.mlp = try oct.mlp(pre);
    }
    w.octree_stem = try oct.stem(dc.children);
    w.level = .{
        .l1 = try oct.floats("l_embedder.mlp.0.weight", &.{ H, dc.level_freqs }),
        .l1_b = try oct.floats("l_embedder.mlp.0.bias", &.{H}),
        .l2 = try oct.floats("l_embedder.mlp.2.weight", &.{ H, H }),
        .l2_b = try oct.floats("l_embedder.mlp.2.bias", &.{H}),
        .ada = try oct.floats("adaLN_modulation.1.weight", &.{ 6 * H, H }),
        .ada_b = try oct.floats("adaLN_modulation.1.bias", &.{6 * H}),
    };
    try oct.whole("decoder.safetensors");
    var gsf: File = .{ .gpa = gpa, .ck = &w.gs, .buffers = &w.buffers, .bytes = &w.bytes, .d = d, .stage = stage };
    for (&w.gaussian, 0..) |*b, i| {
        const pre = try std.fmt.bufPrint(&name, "blocks.{d}", .{i});
        var sp: [96]u8 = undefined;
        const sa = try std.fmt.bufPrint(&sp, "{s}.self_attn.", .{pre});
        b.self_attn = .{
            .qkv = try gsf.half(sa, "to_qkv.weight", &.{ 3 * H, H }),
            .qkv_b = try gsf.half(sa, "to_qkv.bias", &.{3 * H}),
            .out = try gsf.half(sa, "to_out.weight", &.{ H, H }),
            .out_b = try gsf.half(sa, "to_out.bias", &.{H}),
            .gq = try gsf.take(sa, "q_rms_norm.gamma", &.{ dc.heads, H / dc.heads }),
            .gk = try gsf.take(sa, "k_rms_norm.gamma", &.{ dc.heads, H / dc.heads }),
        };
        b.cross = try gsf.cross(pre);
        var np: [96]u8 = undefined;
        const n2 = try std.fmt.bufPrint(&np, "{s}.norm2.", .{pre});
        b.norm2 = try gsf.take(n2, "weight", &.{H});
        b.norm2_b = try gsf.take(n2, "bias", &.{H});
        b.mlp = try gsf.mlp(pre);
    }
    w.gaussian_stem = try gsf.stem(dc.raw_width);
    w.base_offset_scale = (try gsf.floats("base_offset_scale", &.{}))[0];
    w.perturbation = try gsf.floats("points_offset_perturbation", &.{ dc.per_anchor, 3 });
    try gsf.whole("decoder_gs.safetensors");
    return w;
}

test "the fp16 stage holds the largest block linear the decoders round" {
    try std.testing.expect(F * H >= 3 * H * H);
}
