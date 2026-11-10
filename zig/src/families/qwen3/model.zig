//! A Qwen3 dense checkpoint on the GPU: BF16 projections as stored, fp32 norms, and host rows of lm_head on request.
const std = @import("std");
const core = @import("core");
const hip = @import("hip");
const config = @import("config.zig");

const quant = core.quant;
const Checkpoint = core.checkpoint.Checkpoint;
const Allocator = std.mem.Allocator;

pub const Projection = quant.Projection;

/// How the MLP's gate and up weights sit on the device: as stored, or packed for the WMMA product with silu x up.
pub const Layout = enum { plain, interleaved };

/// The gate and up projections, or one [2 x inter, hidden] weight: 16 gate rows, then the same features' 16 up rows.
pub const Mlp = union(Layout) {
    plain: struct { gate: Projection, up: Projection },
    interleaved: Projection,
};

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
    mlp: Mlp,
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

    /// The checkpoint in `dir` on the device of `rt`, MLPs in `layout`; its mapped pages are dropped layer by layer.
    pub fn load(gpa: Allocator, io: std.Io, rt: *const hip.Runtime, dir: []const u8, layout: Layout) !Model {
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
            layer.* = try readLayer(scratch.allocator(), &t, &up, spec, i, layout);
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

/// The reranker's save keeps the base model under `model.` (beside `lm_head`); the embedding's has no prefix.
pub fn scopeOf(ck: *const Checkpoint) []const u8 {
    return if (ck.has("model.embed_tokens.weight")) "model." else "";
}

/// The head rows' stored name: `lm_head.weight`, or the embedding's (prefixed as the checkpoint stores it) when tied.
pub fn headKey(tied: bool, scope: []const u8) []const u8 {
    if (!tied) return "lm_head.weight";
    return if (scope.len > 0) "model.embed_tokens.weight" else "embed_tokens.weight";
}

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

pub fn readSpec(gpa: Allocator, io: std.Io, dir: []const u8) !config.Spec {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    return config.parse(gpa, bytes);
}

/// The BF16 weight `key` of shape (n, k) as stored, refused in any other dtype or shape.
fn bf16Weight(a: Allocator, t: *const Table, key: []const u8, n: usize, k: usize) !quant.Tensor {
    const w = switch (try quant.readStored(a, t, key)) {
        .dense => |d| d.weight,
        else => return refuse(key, "is not a dense tensor"),
    };
    if (w.dtype != .bf16 or w.rank != 2 or w.shape[0] != n or w.shape[1] != k) {
        std.log.err("{s}: {t} {any}, expected bf16 [{d}, {d}]", .{ key, w.dtype, w.shape[0..w.rank], n, k });
        return error.UnexpectedTensor;
    }
    return w;
}

fn uploadDense(up: *const hip.Upload, w: quant.Tensor) !Projection {
    return quant.view(try quant.upload(up.uploader(), .{ .dense = .{ .weight = w } }), false);
}

/// A BF16 projection `key` of shape (n, k), refused in any other dtype or shape.
fn projection(a: Allocator, t: *const Table, up: *const hip.Upload, key: []const u8, n: usize, k: usize) !Projection {
    return uploadDense(up, try bf16Weight(a, t, key, n, k));
}

/// Rows of a packed gate/up weight in one group: this many gate rows, then the same features' up rows.
pub const pack_group = 16;

/// `gate` and `up` (inter, k) rows interleaved by `pack_group` into `out` (2 x inter, k): the packed weight's bytes.
pub fn interleave(out: []u8, gate: []const u8, up: []const u8, inter: usize, row_bytes: usize) void {
    const block = pack_group * row_bytes;
    for (0..inter / pack_group) |g| {
        @memcpy(out[2 * g * block ..][0..block], gate[g * block ..][0..block]);
        @memcpy(out[(2 * g + 1) * block ..][0..block], up[g * block ..][0..block]);
    }
}

/// The layer's gate and up projections as one packed BF16 weight, interleaved on the host before its one upload.
fn packedMlp(a: Allocator, t: *const Table, up: *const hip.Upload, s: config.Spec, i: usize) !Projection {
    var buf: [128]u8 = undefined;
    if (s.inter % pack_group != 0) return refuse("intermediate_size", "is not whole 16-row groups: no packed MLP");
    const gate = try bf16Weight(a, t, try layerKey(&buf, i, "mlp.gate_proj"), s.inter, s.hidden);
    const upw = try bf16Weight(a, t, try layerKey(&buf, i, "mlp.up_proj"), s.inter, s.hidden);
    var w = gate;
    const bytes = try a.alloc(u8, 2 * gate.bytes.len);
    interleave(bytes, gate.bytes, upw.bytes, s.inter, s.hidden * 2);
    w.bytes = bytes;
    w.shape[0] = 2 * s.inter;
    return uploadDense(up, w);
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

fn readLayer(a: Allocator, t: *const Table, up: *const hip.Upload, s: config.Spec, i: usize, layout: Layout) !Layer {
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
        .mlp = switch (layout) {
            .plain => .{ .plain = .{
                .gate = try projection(a, t, up, try layerKey(&buf, i, "mlp.gate_proj"), s.inter, s.hidden),
                .up = try projection(a, t, up, try layerKey(&buf, i, "mlp.up_proj"), s.inter, s.hidden),
            } },
            .interleaved => .{ .interleaved = try packedMlp(a, t, up, s, i) },
        },
        .down = try projection(a, t, up, try layerKey(&buf, i, "mlp.down_proj"), s.hidden, s.inter),
    };
}

fn layerKey(buf: *[128]u8, i: usize, rest: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ i, rest });
}

fn refuse(key: []const u8, why: []const u8) error{UnexpectedTensor} {
    std.log.err("{s} {s}", .{ key, why });
    return error.UnexpectedTensor;
}

/// `name` (e.g. "layers.0.mlp.up_proj.weight") as this checkpoint stores it: under `model.` for the reranker's save.
pub fn scopedKey(gpa: Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    var ck = try Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    return std.mem.concat(gpa, u8, &.{ scopeOf(&ck), name });
}

/// The 1-D tensor `key` (named as stored) as fp32 host values.
pub fn readVector(gpa: Allocator, io: std.Io, dir: []const u8, key: []const u8) ![]f32 {
    var ck = try Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const t = try ck.get(key);
    if (t.rank != 1) return refuse(key, "is not a vector");
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const wide = try quant.convert.float32(scratch.allocator(), t);
    const out = try gpa.alloc(f32, t.shape[0]);
    for (out, 0..) |*x, j| x.* = @bitCast(std.mem.readInt(u32, wide.bytes[4 * j ..][0..4], .little));
    return out;
}

/// Rows `rows` of the 2-D tensor `key` (named as stored, no prefix) as fp32 host values, row after row.
pub fn readRows(gpa: Allocator, io: std.Io, dir: []const u8, key: []const u8, rows: []const u32) ![]f32 {
    var ck = try Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const t = try ck.get(key);
    if (t.rank != 2) return refuse(key, "is not a matrix");
    const width = t.shape[1];
    const out = try gpa.alloc(f32, rows.len * width);
    errdefer gpa.free(out);
    const size = t.dtype.size();
    for (rows, 0..) |r, i| {
        if (r >= t.shape[0]) {
            std.log.warn("{s} has no row {d} of {d}", .{ key, r, t.shape[0] });
            return error.UnexpectedTensor;
        }
        var shape = t.shape;
        shape[0] = width;
        const one: quant.Tensor = .{ .bytes = t.bytes[r * width * size ..][0 .. width * size], .dtype = t.dtype, .rank = 1, .shape = shape };
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const wide = try quant.convert.float32(scratch.allocator(), one);
        // little-endian fp32 words, read one by one: a mapped fp32 tensor need not be 4-byte aligned
        for (out[i * width ..][0..width], 0..) |*x, j| x.* = @bitCast(std.mem.readInt(u32, wide.bytes[4 * j ..][0..4], .little));
    }
    return out;
}

fn writeFixture(dir: std.Io.Dir, prefixed: bool) !void {
    const fixture = core.checkpoint_fixture;
    const a = std.testing.allocator;
    // lm_head: 3 rows of 2 bf16 values: row r is (r + 1, -(r + 1)) as 1.0, 2.0, 3.0 and their negatives
    const head = [_]u16{ 0x3f80, 0xbf80, 0x4000, 0xc000, 0x4040, 0xc040 };
    const embed = [_]u16{ 0, 0, 0, 0, 0, 0 };
    try fixture.write(a, std.testing.io, dir, "model.safetensors", &.{
        .{ .name = if (prefixed) "model.embed_tokens.weight" else "embed_tokens.weight", .dtype = "BF16", .shape = &.{ 3, 2 }, .bytes = std.mem.sliceAsBytes(&embed) },
        .{ .name = "lm_head.weight", .dtype = "BF16", .shape = &.{ 3, 2 }, .bytes = std.mem.sliceAsBytes(&head) },
    });
}

test "a packed MLP weight is 16 gate rows, then the same features' 16 up rows, group after group" {
    var gate: [32 * 2]u8 = undefined;
    var up: [32 * 2]u8 = undefined;
    for (&gate, &up, 0..) |*g, *u, j| {
        g.* = @intCast(j / 2); // row j / 2 of gate, one 2-byte value a row
        u.* = @intCast(100 + j / 2);
    }
    var out: [64 * 2]u8 = undefined;
    interleave(&out, &gate, &up, 32, 2);
    for (0..64) |r| {
        const group = r / 32;
        const within = r % 32;
        const want: u8 = if (within < 16) @intCast(group * 16 + within) else @intCast(100 + group * 16 + within - 16);
        try std.testing.expectEqual(want, out[2 * r]);
        try std.testing.expectEqual(want, out[2 * r + 1]);
    }
}

test "the reranker's model. prefix is found, the embedding's bare names are not, and lm_head rows read as fp32" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    for ([_]bool{ true, false }) |prefixed| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try writeFixture(tmp.dir, prefixed);
        const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        defer a.free(dir);
        var ck = try Checkpoint.openModel(a, io, dir);
        defer ck.close();
        try std.testing.expectEqualStrings(if (prefixed) "model." else "", scopeOf(&ck));
        const rows = try readRows(a, io, dir, "lm_head.weight", &.{ 2, 0 });
        defer a.free(rows);
        try std.testing.expectEqualSlices(f32, &.{ 3, -3, 1, -1 }, rows);
        try std.testing.expectError(error.UnexpectedTensor, readRows(a, io, dir, "lm_head.weight", &.{3}));
        // a tied model reads its head rows from the embedding, as stored
        const tied = try readRows(a, io, dir, headKey(true, scopeOf(&ck)), &.{1});
        defer a.free(tied);
        try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, tied);
        try std.testing.expectEqualStrings("lm_head.weight", headKey(false, scopeOf(&ck)));
    }
}
