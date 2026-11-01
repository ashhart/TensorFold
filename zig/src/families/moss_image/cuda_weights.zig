//! MOSS-Image's transformer tensors from a diffusers checkpoint folder on CUDA: bf16 as stored, each on the device.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const host = @import("host.zig");

/// A joint block: attention projections, the MLP's (hidden | gate) and its down projection, then the norms.
pub const Block = struct { q: u64, k: u64, v: u64, o: u64, up_gate: u64, down: u64, norm1: u64, norm2: u64, q_norm: u64, k_norm: u64 };

/// A prompt refiner block: q, k, v packed in one projection, every projection with a bias, no modulation.
pub const TextBlock = struct { norm1: u64, norm2: u64, q_norm: u64, k_norm: u64, qkv: u64, qkv_b: u64, proj: u64, proj_b: u64, gate: u64, gate_b: u64, up: u64, up_b: u64, down: u64, down_b: u64 };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    ck: core.Checkpoint,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    blocks: [cfg.layers]Block = undefined,
    text: [cfg.text_layers]TextBlock = undefined,
    text_in: u64 = 0,
    text_in_b: u64 = 0,
    video_in: u64 = 0,
    video_in_b: u64 = 0,
    final_norm: u64 = 0,
    final_out: u64 = 0,
    /// The rotary's frequencies as the checkpoint stores them (fp32).
    bands: [cfg.bands]f32 = undefined,
    /// The modulation's projections, read on the host from the mapped checkpoint (host.modulation).
    modulation: host.Weights = undefined,
    bytes: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.ck.close();
    }

    /// The tensor `name` of `dtype` and exactly `shape`.
    fn typed(w: *Weights, name: []const u8, dtype: core.checkpoint.DType, shape: []const usize) !core.checkpoint.Tensor {
        const t = try w.ck.get(name);
        if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.log.err("moss image: {s} is {t} {any}; the transformer reads {t} {any}", .{ name, t.dtype, t.shape[0..t.rank], dtype, shape });
            return error.UnexpectedTensor;
        }
        return t;
    }

    /// The bf16 tensor `name` of `shape` on the device.
    fn take(w: *Weights, d: *const cuda.Driver, name: []const u8, shape: []const usize) !u64 {
        const t = try w.typed(name, .bf16, shape);
        const b = try cuda.DeviceBuffer.fromHost(d, t.bytes);
        errdefer {
            var owned = b;
            owned.free();
        }
        try w.buffers.append(w.gpa, b);
        w.bytes += b.len;
        return b.ptr;
    }

    /// The bf16 tensor `name` of `shape`, its values on the host (mapped from the checkpoint).
    fn values(w: *Weights, name: []const u8, shape: []const usize) ![]const u16 {
        const t = try w.typed(name, .bf16, shape);
        return @alignCast(std.mem.bytesAsSlice(u16, t.bytes));
    }
};

/// The transformer's shard files in `dir`: the index's weight map, else the one diffusion_pytorch_model.safetensors.
fn shards(arena: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const []const u8 {
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    const index = try std.fs.path.join(arena, &.{ dir, "diffusion_pytorch_model.safetensors.index.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, index, arena, .limited(16 << 20))) |text| {
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
        const map = if (doc == .object) doc.object.get("weight_map") else null;
        if (map == null or map.? != .object) return error.BadIndex;
        var it = map.?.object.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* != .string) return error.BadIndex;
            try files.put(arena, kv.value_ptr.string, {});
        }
    } else |err| {
        if (err != error.FileNotFound) return err;
        try files.put(arena, "diffusion_pytorch_model.safetensors", {});
    }
    return files.keys();
}

/// The transformer folder `dir` (a checkpoint's transformer/): config checked, then every tensor this family reads.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
    try cfg.check(gpa, config);
    const w = try gpa.create(Weights);
    errdefer gpa.destroy(w);
    w.* = .{ .gpa = gpa, .ck = .{ .gpa = gpa, .io = io } };
    errdefer w.deinit();
    for (try shards(arena, io, dir)) |name| try w.ck.add(dir, name);
    const h = cfg.hidden;
    const f = cfg.ffn;
    const hd = cfg.head_dim;
    var name: [96]u8 = undefined;
    for (&w.blocks, 0..) |*b, i| {
        const fields = .{ .{ "proj.q.weight", &b.q, &[_]usize{ h, h } }, .{ "proj.k.weight", &b.k, &[_]usize{ h, h } }, .{ "proj.v.weight", &b.v, &[_]usize{ h, h } }, .{ "proj.o.weight", &b.o, &[_]usize{ h, h } }, .{ "proj.ffn_up_gate.weight", &b.up_gate, &[_]usize{ 2 * f, h } }, .{ "proj.ffn_down.weight", &b.down, &[_]usize{ h, f } }, .{ "norm1.weight", &b.norm1, &[_]usize{h} }, .{ "norm2.weight", &b.norm2, &[_]usize{h} }, .{ "q_norm.weight", &b.q_norm, &[_]usize{hd} }, .{ "k_norm.weight", &b.k_norm, &[_]usize{hd} } };
        inline for (fields) |x| x[1].* = try w.take(d, try std.fmt.bufPrint(&name, "layers.{d}.{s}", .{ i, x[0] }), x[2]);
    }
    for (&w.text, 0..) |*b, i| {
        const fields = .{ .{ "norm1.weight", &b.norm1, &[_]usize{h} }, .{ "norm2.weight", &b.norm2, &[_]usize{h} }, .{ "q_norm.weight", &b.q_norm, &[_]usize{hd} }, .{ "k_norm.weight", &b.k_norm, &[_]usize{hd} }, .{ "qkv.weight", &b.qkv, &[_]usize{ 3 * h, h } }, .{ "qkv.bias", &b.qkv_b, &[_]usize{3 * h} }, .{ "proj.weight", &b.proj, &[_]usize{ h, h } }, .{ "proj.bias", &b.proj_b, &[_]usize{h} }, .{ "ffn_gate.weight", &b.gate, &[_]usize{ f, h } }, .{ "ffn_gate.bias", &b.gate_b, &[_]usize{f} }, .{ "ffn_up.weight", &b.up, &[_]usize{ f, h } }, .{ "ffn_up.bias", &b.up_b, &[_]usize{f} }, .{ "ffn_down.weight", &b.down, &[_]usize{ h, f } }, .{ "ffn_down.bias", &b.down_b, &[_]usize{h} } };
        inline for (fields) |x| x[1].* = try w.take(d, try std.fmt.bufPrint(&name, "text_embedder.blocks.{d}.{s}", .{ i, x[0] }), x[2]);
    }
    w.text_in = try w.take(d, "text_embedder.proj.weight", &.{ h, cfg.context });
    w.text_in_b = try w.take(d, "text_embedder.proj.bias", &.{h});
    w.video_in = try w.take(d, "video_embedder.weight", &.{ h, cfg.channels });
    w.video_in_b = try w.take(d, "video_embedder.bias", &.{h});
    w.final_norm = try w.take(d, "final_norm_video.weight", &.{h});
    w.final_out = try w.take(d, "final_linear_video.weight", &.{ cfg.channels, h });
    const bands = try w.typed("rope.bands", .f32, &.{cfg.bands});
    @memcpy(std.mem.asBytes(&w.bands), bands.bytes);
    w.modulation = .{
        .time_1 = try w.values("time_embedding.0.weight", &.{ h, cfg.time_dim }),
        .time_1_b = try w.values("time_embedding.0.bias", &.{h}),
        .time_2 = try w.values("time_embedding.2.weight", &.{ h, h }),
        .time_2_b = try w.values("time_embedding.2.bias", &.{h}),
        .blocks = try w.values("shared_block_modulation.weight", &.{ 6 * h, h }),
        .blocks_b = try w.values("shared_block_modulation.bias", &.{6 * h}),
        .head = try w.values("video_head_modulation.weight", &.{ 2 * h, h }),
        .head_b = try w.values("video_head_modulation.bias", &.{2 * h}),
    };
    if (w.ck.unused() != 0) {
        std.log.err("moss image: the checkpoint has {d} transformer tensors this family does not read", .{w.ck.unused()});
        return error.UnexpectedTensors;
    }
    return w;
}
