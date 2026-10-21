//! Qwen-Image-2.1's transformer tensors from a diffusers checkpoint folder, bf16 as stored, into shared Metal buffers.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const index_format = @import("../../core/checkpoint_index.zig");
const cfg = @import("config.zig");
const host = @import("host.zig");
const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

pub const Block = struct { to_q: Ref, to_k: Ref, to_v: Ref, to_out: Ref, norm_q: Ref, norm_k: Ref, gate: Ref, proj: Ref, out: Ref };

pub const Weights = struct {
    ck: ck.Checkpoint,
    blocks: [cfg.layers]Block,
    img_in: Ref,
    txt_norm: Ref,
    txt_in: Ref,
    txt_out: Ref,
    proj_out: Ref,
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

/// The bf16 tensor `name` of exactly `shape`.
fn take(c: *const ck.Checkpoint, name: []const u8, shape: []const usize) !Ref {
    const t = try c.get(name);
    if (t.dtype != .bf16 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
        std.log.err("qwen image: {s} is {t} {any}; the transformer reads bf16 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
        return error.UnexpectedTensor;
    }
    if (t.offset % 16 != 0) return error.UnalignedTensor;
    return .{ .buf = t.buffer, .off = t.offset };
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
    const m = cfg.mlp_width;
    var name: [96]u8 = undefined;
    for (&w.blocks, 0..) |*b, i| {
        const fields = .{ .{ "attn.to_q.weight", &b.to_q, [_]usize{ h, h } }, .{ "attn.to_k.weight", &b.to_k, [_]usize{ h, h } }, .{ "attn.to_v.weight", &b.to_v, [_]usize{ h, h } }, .{ "attn.to_out.0.weight", &b.to_out, [_]usize{ h, h } }, .{ "img_mlp.gate_layer.weight", &b.gate, [_]usize{ m, h } }, .{ "img_mlp.proj.weight", &b.proj, [_]usize{ m, h } }, .{ "img_mlp.out.weight", &b.out, [_]usize{ h, m } } };
        inline for (fields) |f| {
            const shape: [2]usize = f[2];
            f[1].* = try take(&w.ck, try std.fmt.bufPrint(&name, "transformer_blocks.{d}.{s}", .{ i, f[0] }), &shape);
        }
        b.norm_q = try take(&w.ck, try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.norm_q.weight", .{i}), &.{cfg.head_dim});
        b.norm_k = try take(&w.ck, try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.norm_k.weight", .{i}), &.{cfg.head_dim});
    }
    w.img_in = try take(&w.ck, "img_in.weight", &.{ h, cfg.channels });
    w.txt_norm = try take(&w.ck, "txt_in.text_norm.weight", &.{cfg.context});
    w.txt_in = try take(&w.ck, "txt_in.in_layer.weight", &.{ h, cfg.context });
    w.txt_out = try take(&w.ck, "txt_in.out_layer.weight", &.{ h, h });
    w.proj_out = try take(&w.ck, "proj_out.weight", &.{ cfg.channels, h });
    const t1 = try take(&w.ck, "time_text_embed.timestep_embedder.linear_1.weight", &.{ h, cfg.time_dim });
    const t2 = try take(&w.ck, "time_text_embed.timestep_embedder.linear_2.weight", &.{ h, h });
    const mod = try take(&w.ck, "modulation.1.weight", &.{ 4 * h, h });
    const out = try take(&w.ck, "norm_out.linear.weight", &.{ h, h });
    w.modulation = .{ .time_1 = values(t1, h * cfg.time_dim), .time_2 = values(t2, h * h), .modulation = values(mod, 4 * h * h), .norm_out = values(out, h * h) };
    const read = 9 * cfg.layers + 9;
    if (w.ck.tensors.count() != read) {
        std.log.err("qwen image: the checkpoint has {d} transformer tensors; this family reads {d}", .{ w.ck.tensors.count(), read });
        return error.UnexpectedTensors;
    }
    return w;
}
