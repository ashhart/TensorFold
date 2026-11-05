//! A Qwen3 dense checkpoint on the GPU: BF16 projections as stored, fp32 norms, and host rows of lm_head on request.
const std = @import("std");
const core = @import("core");
const hip = @import("hip");
const config = @import("config.zig");
const names = @import("names.zig");

const quant = core.quant;
const Checkpoint = core.checkpoint.Checkpoint;
const Allocator = std.mem.Allocator;

pub const Projection = quant.Projection;

/// One decoder layer: norms are fp32 device addresses, projections BF16 (N, K) as the checkpoint stores them.
pub const Layer = struct {
    input_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    gate: Projection,
    up: Projection,
    down: Projection,
};

pub const Model = struct {
    gpa: Allocator,
    spec: config.Spec,
    /// The activation dtype: BF16, as the checkpoint's weights.
    act: hip.ops.Kind = .bf16,
    embed: Projection,
    /// The stored name of the output head's rows for `readRows`: lm_head, or the embedding when the model ties them.
    head: []const u8,
    final_norm: u64,
    layers: []Layer,
    buffers: std.ArrayList(hip.memory.DeviceBuffer) = .empty,

    /// The checkpoint in `dir` on the device of `rt`; its mapped pages are dropped layer by layer.
    pub fn load(gpa: Allocator, io: std.Io, rt: *const hip.Runtime, dir: []const u8) !Model {
        const spec = try readSpec(gpa, io, dir);
        var ck = try Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        ck.scope = scopeOf(&ck);
        var m: Model = .{ .gpa = gpa, .spec = spec, .embed = undefined, .head = headKey(spec.tied, ck.scope), .final_norm = 0, .layers = &.{} };
        errdefer m.deinit();
        const up: hip.Upload = .{ .r = rt, .buffers = &m.buffers, .gpa = gpa };
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        var t: Table = .{ .ck = &ck };
        m.embed = try projection(scratch.allocator(), &t, &up, "embed_tokens", spec.vocab, spec.hidden);
        m.final_norm = try norm(scratch.allocator(), &t, &up, "norm.weight", spec.hidden);
        m.layers = try gpa.alloc(Layer, spec.layers);
        for (m.layers, 0..) |*layer, i| {
            _ = scratch.reset(.retain_capacity);
            // no ck.release() between layers: its fadvise would drop the shards from the page cache mid-load
            layer.* = try readLayer(scratch.allocator(), &t, &up, spec, i);
        }
        return m;
    }

    pub fn deinit(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.gpa.free(m.layers);
        m.* = undefined;
    }
};

/// `quant.read`'s table over a checkpoint: dense tensors only, so no format asks for an affine width.
const Table = struct {
    ck: *Checkpoint,

    pub fn get(t: *const Table, name: []const u8) !quant.Tensor {
        return t.ck.get(name);
    }

    pub fn has(t: *const Table, name: []const u8) bool {
        return t.ck.has(name);
    }

    pub fn width(_: *const Table, _: []const u8) error{UnsupportedQuantization}!quant.mlx.Width {
        return error.UnsupportedQuantization;
    }
};

/// A BF16 projection `key` of shape (n, k), refused in any other dtype or shape.
fn projection(a: Allocator, t: *const Table, up: *const hip.Upload, key: []const u8, n: usize, k: usize) !Projection {
    const host = try quant.readStored(a, t, key);
    const w = switch (host) {
        .dense => |d| d.weight,
        else => return refuse(key, "is not a dense tensor"),
    };
    if (w.dtype != .bf16 or w.rank != 2 or w.shape[0] != n or w.shape[1] != k) {
        std.log.err("{s}: {t} {any}, expected bf16 [{d}, {d}]", .{ key, w.dtype, w.shape[0..w.rank], n, k });
        return error.UnexpectedTensor;
    }
    return quant.view(try quant.upload(up.uploader(), host), false);
}

/// An RMS norm weight `key` of `n` values, widened to fp32 on the device.
fn norm(a: Allocator, t: *const Table, up: *const hip.Upload, key: []const u8, n: usize) !u64 {
    const src = try t.get(key);
    if (src.rank != 1 or src.shape[0] != n) {
        std.log.err("{s}: shape {any}, expected [{d}]", .{ key, src.shape[0..src.rank], n });
        return error.UnexpectedTensor;
    }
    const wide = try quant.convert.float32(a, src);
    return (try up.uploader().tensor(wide)).ptr;
}

fn readLayer(a: Allocator, t: *const Table, up: *const hip.Upload, s: config.Spec, i: usize) !Layer {
    var buf: [128]u8 = undefined;
    const q_width = s.heads * s.head_dim;
    const kv_width = s.kv_heads * s.head_dim;
    return .{
        .input_norm = try norm(a, t, up, try layerKey(&buf, i, "input_layernorm.weight"), s.hidden),
        .post_norm = try norm(a, t, up, try layerKey(&buf, i, "post_attention_layernorm.weight"), s.hidden),
        .q_norm = try norm(a, t, up, try layerKey(&buf, i, "self_attn.q_norm.weight"), s.head_dim),
        .k_norm = try norm(a, t, up, try layerKey(&buf, i, "self_attn.k_norm.weight"), s.head_dim),
        .q = try projection(a, t, up, try layerKey(&buf, i, "self_attn.q_proj"), q_width, s.hidden),
        .k = try projection(a, t, up, try layerKey(&buf, i, "self_attn.k_proj"), kv_width, s.hidden),
        .v = try projection(a, t, up, try layerKey(&buf, i, "self_attn.v_proj"), kv_width, s.hidden),
        .o = try projection(a, t, up, try layerKey(&buf, i, "self_attn.o_proj"), s.hidden, q_width),
        .gate = try projection(a, t, up, try layerKey(&buf, i, "mlp.gate_proj"), s.inter, s.hidden),
        .up = try projection(a, t, up, try layerKey(&buf, i, "mlp.up_proj"), s.inter, s.hidden),
        .down = try projection(a, t, up, try layerKey(&buf, i, "mlp.down_proj"), s.hidden, s.inter),
    };
}

fn layerKey(buf: *[128]u8, i: usize, rest: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ i, rest });
}

pub const scopeOf = names.scopeOf;
pub const headKey = names.headKey;
pub const readSpec = names.readSpec;
pub const scopedKey = names.scopedKey;
pub const readVector = names.readVector;
pub const readRows = names.readRows;
const refuse = names.refuse;
