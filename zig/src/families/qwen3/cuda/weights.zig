//! A Qwen3 dense checkpoint: BF16 tensors on the device as stored, norms widened to fp32, the head's stored name.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("../config.zig");
const names = @import("../names.zig");
const store = @import("store.zig");

const Checkpoint = core.checkpoint.Checkpoint;
const Allocator = std.mem.Allocator;

/// One decoder layer: fp32 norm weights, BF16 projections [n, k] as the checkpoint stores them.
pub const Layer = struct {
    input_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    gate: u64,
    up: u64,
    down: u64,
};

pub const Model = struct {
    gpa: Allocator,
    spec: config.Spec,
    /// The stored name of the output head's rows: a reranker scores them on the host (names.headKey).
    head: []const u8,
    /// The embedding table [vocab, hidden] the token rows come from.
    embed: u64,
    final_norm: u64,
    layers: []Layer,
    st: store.Store,

    /// The checkpoint in `dir` on `d`'s device, its shards mapped and its tensors uploaded.
    pub fn load(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Model {
        const spec = try names.readSpec(gpa, io, dir);
        var ck = try Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        ck.scope = names.scopeOf(&ck);
        var m = try fromCheckpoint(gpa, d, &ck, spec, names.headKey(spec.tied, ck.scope));
        errdefer m.deinit();
        _ = try ck.stored(m.head); // the head's rows are read on the host later (names.readRows); its tensor still counts
        try refuseLeftovers(&ck);
        return m;
    }

    /// `spec`'s tensors from an open checkpoint under its scope (a model whose Qwen3 sits beside other parts).
    pub fn fromCheckpoint(gpa: Allocator, d: *const cuda.Driver, ck: *Checkpoint, spec: config.Spec, head: []const u8) !Model {
        var m: Model = .{ .gpa = gpa, .spec = spec, .head = head, .embed = 0, .final_norm = 0, .layers = &.{}, .st = .{ .gpa = gpa, .d = d } };
        errdefer m.deinit();
        m.embed = try projection(ck, &m.st, "embed_tokens.weight", spec.vocab, spec.hidden);
        m.final_norm = try norm(ck, &m.st, "norm.weight", spec.hidden);
        m.layers = try gpa.alloc(Layer, spec.layers);
        for (m.layers, 0..) |*l, i| l.* = try layer(ck, &m.st, spec, i);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.st.deinit();
        m.gpa.free(m.layers);
    }
};

/// A direct load's unbound names, refused without changing core's test logging for other loaders.
fn refuseLeftovers(ck: *const Checkpoint) !void {
    var n: usize = 0;
    for (ck.files.items) |*f| for (f.names.keys()) |k| {
        if (ck.used.contains(k)) continue;
        if (n < 5 and !@import("builtin").is_test) std.log.err("unused checkpoint tensor {s}", .{k});
        n += 1;
    };
    if (n != 0) return error.UnusedQwen3Tensors;
}

/// A BF16 matrix `key` of shape (n, k), uploaded as stored.
fn projection(ck: *Checkpoint, st: *store.Store, name: []const u8, n: usize, k: usize) !u64 {
    return st.put((try ck.expect(name, .bf16, &.{ n, k })).bytes);
}

/// An RMS norm weight `key` of `n` values, widened to fp32 on the device.
fn norm(ck: *Checkpoint, st: *store.Store, name: []const u8, n: usize) !u64 {
    const src = try ck.expect(name, .bf16, &.{n});
    const wide = try st.gpa.alloc(f32, n);
    defer st.gpa.free(wide);
    for (wide, 0..) |*w, i| w.* = @bitCast(@as(u32, std.mem.readInt(u16, src.bytes[2 * i ..][0..2], .little)) << 16);
    return st.put(std.mem.sliceAsBytes(wide));
}

fn layer(ck: *Checkpoint, st: *store.Store, s: config.Spec, i: usize) !Layer {
    var buf: [128]u8 = undefined;
    const qw = s.heads * s.head_dim;
    const kvw = s.kv_heads * s.head_dim;
    return .{
        .input_norm = try norm(ck, st, try layerName(&buf, i, "input_layernorm.weight"), s.hidden),
        .post_norm = try norm(ck, st, try layerName(&buf, i, "post_attention_layernorm.weight"), s.hidden),
        .q_norm = try norm(ck, st, try layerName(&buf, i, "self_attn.q_norm.weight"), s.head_dim),
        .k_norm = try norm(ck, st, try layerName(&buf, i, "self_attn.k_norm.weight"), s.head_dim),
        .q = try projection(ck, st, try layerName(&buf, i, "self_attn.q_proj.weight"), qw, s.hidden),
        .k = try projection(ck, st, try layerName(&buf, i, "self_attn.k_proj.weight"), kvw, s.hidden),
        .v = try projection(ck, st, try layerName(&buf, i, "self_attn.v_proj.weight"), kvw, s.hidden),
        .o = try projection(ck, st, try layerName(&buf, i, "self_attn.o_proj.weight"), s.hidden, qw),
        .gate = try projection(ck, st, try layerName(&buf, i, "mlp.gate_proj.weight"), s.inter, s.hidden),
        .up = try projection(ck, st, try layerName(&buf, i, "mlp.up_proj.weight"), s.inter, s.hidden),
        .down = try projection(ck, st, try layerName(&buf, i, "mlp.down_proj.weight"), s.hidden, s.inter),
    };
}

/// The stored name of layer `i`'s tensor `rest`, in `buf`.
fn layerName(buf: *[128]u8, i: usize, rest: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ i, rest });
}

test "Qwen3 weights: every checkpoint tensor is bound once, and a stray one refuses" {
    const t = std.testing;
    const fixture = core.checkpoint_fixture;
    const a = t.allocator;
    const io = t.io;
    const json =
        \\{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":4,"num_attention_heads":1,
        \\"num_key_value_heads":1,"head_dim":4,"intermediate_size":8,"vocab_size":6,
        \\"rms_norm_eps":1e-6,"rope_theta":10000,"max_position_embeddings":64,
        \\"tie_word_embeddings":false,"hidden_act":"silu"}
    ;
    const mat: [16]u16 = @splat(0x3f80); // bf16 1.0s, the 4x4 tensors
    const wide: [32]u16 = @splat(0x3f80); // the 4x8 and 8x4 tensors
    const vec: [4]u16 = @splat(0x3f80);
    const vocab_mat: [24]u16 = @splat(0x3f80); // embed_tokens and lm_head
    const entries = [_]fixture.Entry{
        .{ .name = "embed_tokens.weight", .dtype = "BF16", .shape = &.{ 6, 4 }, .bytes = std.mem.sliceAsBytes(&vocab_mat) },
        .{ .name = "norm.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&vec) },
        .{ .name = "layers.0.input_layernorm.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&vec) },
        .{ .name = "layers.0.post_attention_layernorm.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&vec) },
        .{ .name = "layers.0.self_attn.q_norm.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&vec) },
        .{ .name = "layers.0.self_attn.k_norm.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&vec) },
        .{ .name = "layers.0.self_attn.q_proj.weight", .dtype = "BF16", .shape = &.{ 4, 4 }, .bytes = std.mem.sliceAsBytes(&mat) },
        .{ .name = "layers.0.self_attn.k_proj.weight", .dtype = "BF16", .shape = &.{ 4, 4 }, .bytes = std.mem.sliceAsBytes(&mat) },
        .{ .name = "layers.0.self_attn.v_proj.weight", .dtype = "BF16", .shape = &.{ 4, 4 }, .bytes = std.mem.sliceAsBytes(&mat) },
        .{ .name = "layers.0.self_attn.o_proj.weight", .dtype = "BF16", .shape = &.{ 4, 4 }, .bytes = std.mem.sliceAsBytes(&mat) },
        .{ .name = "layers.0.mlp.gate_proj.weight", .dtype = "BF16", .shape = &.{ 8, 4 }, .bytes = std.mem.sliceAsBytes(&wide) },
        .{ .name = "layers.0.mlp.up_proj.weight", .dtype = "BF16", .shape = &.{ 8, 4 }, .bytes = std.mem.sliceAsBytes(&wide) },
        .{ .name = "layers.0.mlp.down_proj.weight", .dtype = "BF16", .shape = &.{ 4, 8 }, .bytes = std.mem.sliceAsBytes(&wide) },
        .{ .name = "lm_head.weight", .dtype = "BF16", .shape = &.{ 6, 4 }, .bytes = std.mem.sliceAsBytes(&vocab_mat) },
        .{ .name = "model.stray.weight", .dtype = "BF16", .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&vec) },
    };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    try fixture.write(a, io, tmp.dir, "model.safetensors", &entries);
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    const Mock = struct {
        var next: u64 = 0x1000;
        fn alloc(out: *u64, _: usize) callconv(.c) c_int {
            out.* = next;
            next += 256;
            return 0;
        }
        fn free(_: u64) callconv(.c) c_int {
            return 0;
        }
        fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
            return 0;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }
    };
    d.api.cuMemAlloc_v2 = Mock.alloc;
    d.api.cuMemFree_v2 = Mock.free;
    d.api.cuMemcpyHtoD_v2 = Mock.copy;
    d.api.cuStreamSynchronize = Mock.sync;
    try t.expectError(error.UnusedQwen3Tensors, Model.load(a, io, &d, dir));
    try tmp.dir.deleteFile(io, "model.safetensors");
    try fixture.write(a, io, tmp.dir, "model.safetensors", entries[0 .. entries.len - 1]);
    var m = try Model.load(a, io, &d, dir);
    m.deinit();
}
