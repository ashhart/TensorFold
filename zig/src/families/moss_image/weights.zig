//! MOSS-Image's transformer tensors from a diffusers checkpoint folder, bf16 as stored, into shared Metal buffers.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const index_format = @import("../../core/checkpoint_index.zig");
const cfg = @import("config.zig");
const host = @import("host.zig");
const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

/// A joint block: attention projections, the MLP's (hidden | gate) and its down projection, then the norms.
pub const Block = struct { q: Ref, k: Ref, v: Ref, o: Ref, up_gate: Ref, down: Ref, norm1: Ref, norm2: Ref, q_norm: Ref, k_norm: Ref };

/// A prompt refiner block: q, k, v packed in one projection, every projection with a bias, no modulation.
pub const TextBlock = struct { norm1: Ref, norm2: Ref, q_norm: Ref, k_norm: Ref, qkv: Ref, qkv_b: Ref, proj: Ref, proj_b: Ref, gate: Ref, gate_b: Ref, up: Ref, up_b: Ref, down: Ref, down_b: Ref };

pub const Weights = struct {
    ck: ck.Checkpoint,
    blocks: [cfg.layers]Block,
    text: [cfg.text_layers]TextBlock,
    text_in: Ref,
    text_in_b: Ref,
    video_in: Ref,
    video_in_b: Ref,
    final_norm: Ref,
    final_out: Ref,
    /// The rotary's frequencies as the checkpoint stores them (fp32).
    bands: [cfg.bands]f32,
    modulation: host.Weights,

    pub fn deinit(w: *Weights) void {
        w.ck.deinit();
    }
};

/// The transformer's shard files in `dir`: the index's weight map, else the one diffusion_pytorch_model.safetensors.
fn shards(arena: std.mem.Allocator, io: std.Io, dir: []const u8) ![][:0]const u8 {
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    const index = try std.fs.path.join(arena, &.{ dir, "diffusion_pytorch_model.safetensors.index.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, index, arena, .limited(16 << 20))) |text| {
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
        var it = (try index_format.weightMap(doc)).iterator();
        while (it.next()) |kv| try files.put(arena, kv.value_ptr.string, {});
    } else |err| {
        if (err != error.FileNotFound) return err;
        try files.put(arena, "diffusion_pytorch_model.safetensors", {});
    }
    const out = try arena.alloc([:0]const u8, files.count());
    for (files.keys(), out) |name, *path| path.* = try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ dir, name }, 0);
    return out;
}

/// The tensor `name` of `dtype` and exactly `shape`.
fn typed(c: *const ck.Checkpoint, name: []const u8, dtype: ck.DType, shape: []const usize) !Ref {
    const t = try c.get(name);
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
        std.log.err("moss image: {s} is {t} {any}; the transformer reads {t} {any}", .{ name, t.dtype, t.shape[0..t.rank], dtype, shape });
        return error.UnexpectedTensor;
    }
    if (t.offset % 16 != 0) return error.UnalignedTensor;
    return .{ .buf = t.buffer, .off = t.offset };
}

fn take(c: *const ck.Checkpoint, name: []const u8, shape: []const usize) !Ref {
    return typed(c, name, .bf16, shape);
}

fn values(r: Ref, n: usize) []const u16 {
    return @as([*]const u16, @ptrCast(@alignCast(r.addr())))[0..n];
}

/// The transformer folder `dir` (a checkpoint's transformer/): config checked, then every tensor this family reads.
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config_path = try std.fs.path.join(arena, &.{ dir, "config.json" });
    const config = try std.Io.Dir.cwd().readFileAlloc(io, config_path, arena, .limited(1 << 20));
    try cfg.check(gpa, config);
    var w: Weights = undefined;
    w.ck = ck.Checkpoint.init(gpa);
    errdefer w.ck.deinit();
    for (try shards(arena, io, dir)) |path| try w.ck.addFile(device, path, "");
    const h = cfg.hidden;
    const f = cfg.ffn;
    const d = cfg.head_dim;
    var name: [96]u8 = undefined;
    for (&w.blocks, 0..) |*b, i| {
        const fields = .{ .{ "proj.q.weight", &b.q, &[_]usize{ h, h } }, .{ "proj.k.weight", &b.k, &[_]usize{ h, h } }, .{ "proj.v.weight", &b.v, &[_]usize{ h, h } }, .{ "proj.o.weight", &b.o, &[_]usize{ h, h } }, .{ "proj.ffn_up_gate.weight", &b.up_gate, &[_]usize{ 2 * f, h } }, .{ "proj.ffn_down.weight", &b.down, &[_]usize{ h, f } }, .{ "norm1.weight", &b.norm1, &[_]usize{h} }, .{ "norm2.weight", &b.norm2, &[_]usize{h} }, .{ "q_norm.weight", &b.q_norm, &[_]usize{d} }, .{ "k_norm.weight", &b.k_norm, &[_]usize{d} } };
        inline for (fields) |x| x[1].* = try take(&w.ck, try std.fmt.bufPrint(&name, "layers.{d}.{s}", .{ i, x[0] }), x[2]);
    }
    for (&w.text, 0..) |*b, i| {
        const fields = .{ .{ "norm1.weight", &b.norm1, &[_]usize{h} }, .{ "norm2.weight", &b.norm2, &[_]usize{h} }, .{ "q_norm.weight", &b.q_norm, &[_]usize{d} }, .{ "k_norm.weight", &b.k_norm, &[_]usize{d} }, .{ "qkv.weight", &b.qkv, &[_]usize{ 3 * h, h } }, .{ "qkv.bias", &b.qkv_b, &[_]usize{3 * h} }, .{ "proj.weight", &b.proj, &[_]usize{ h, h } }, .{ "proj.bias", &b.proj_b, &[_]usize{h} }, .{ "ffn_gate.weight", &b.gate, &[_]usize{ f, h } }, .{ "ffn_gate.bias", &b.gate_b, &[_]usize{f} }, .{ "ffn_up.weight", &b.up, &[_]usize{ f, h } }, .{ "ffn_up.bias", &b.up_b, &[_]usize{f} }, .{ "ffn_down.weight", &b.down, &[_]usize{ h, f } }, .{ "ffn_down.bias", &b.down_b, &[_]usize{h} } };
        inline for (fields) |x| x[1].* = try take(&w.ck, try std.fmt.bufPrint(&name, "text_embedder.blocks.{d}.{s}", .{ i, x[0] }), x[2]);
    }
    w.text_in = try take(&w.ck, "text_embedder.proj.weight", &.{ h, cfg.context });
    w.text_in_b = try take(&w.ck, "text_embedder.proj.bias", &.{h});
    w.video_in = try take(&w.ck, "video_embedder.weight", &.{ h, cfg.channels });
    w.video_in_b = try take(&w.ck, "video_embedder.bias", &.{h});
    w.final_norm = try take(&w.ck, "final_norm_video.weight", &.{h});
    w.final_out = try take(&w.ck, "final_linear_video.weight", &.{ cfg.channels, h });
    const bands = try typed(&w.ck, "rope.bands", .f32, &.{cfg.bands});
    w.bands = @as(*const [cfg.bands]f32, @ptrCast(@alignCast(bands.addr()))).*;
    const parts = .{ .{ "time_embedding.0.weight", [_]usize{ h, cfg.time_dim } }, .{ "time_embedding.0.bias", [_]usize{h} }, .{ "time_embedding.2.weight", [_]usize{ h, h } }, .{ "time_embedding.2.bias", [_]usize{h} }, .{ "shared_block_modulation.weight", [_]usize{ 6 * h, h } }, .{ "shared_block_modulation.bias", [_]usize{6 * h} }, .{ "video_head_modulation.weight", [_]usize{ 2 * h, h } }, .{ "video_head_modulation.bias", [_]usize{2 * h} } };
    var host_parts: [parts.len][]const u16 = undefined;
    inline for (parts, 0..) |p, i| {
        const shape: [p[1].len]usize = p[1];
        var n: usize = 1;
        for (shape) |s| n *= s;
        host_parts[i] = values(try take(&w.ck, p[0], &shape), n);
    }
    w.modulation = .{ .time_1 = host_parts[0], .time_1_b = host_parts[1], .time_2 = host_parts[2], .time_2_b = host_parts[3], .blocks = host_parts[4], .blocks_b = host_parts[5], .head = host_parts[6], .head_b = host_parts[7] };
    const read = 10 * cfg.layers + 14 * cfg.text_layers + 2 + 2 + 2 + 1 + parts.len;
    if (w.ck.tensors.count() != read) {
        std.log.err("moss image: the checkpoint has {d} transformer tensors; this family reads {d}", .{ w.ck.tensors.count(), read });
        return error.UnexpectedTensors;
    }
    return w;
}
