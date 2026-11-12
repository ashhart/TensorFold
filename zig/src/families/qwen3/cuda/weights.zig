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
        var m: Model = .{ .gpa = gpa, .spec = spec, .head = names.headKey(spec.tied, ck.scope), .embed = 0, .final_norm = 0, .layers = &.{}, .st = .{ .gpa = gpa, .d = d } };
        errdefer m.deinit();
        m.embed = try projection(&ck, &m.st, "embed_tokens.weight", spec.vocab, spec.hidden);
        m.final_norm = try norm(&ck, &m.st, "norm.weight", spec.hidden);
        m.layers = try gpa.alloc(Layer, spec.layers);
        for (m.layers, 0..) |*l, i| l.* = try layer(&ck, &m.st, spec, i);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.st.deinit();
        m.gpa.free(m.layers);
    }
};

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
