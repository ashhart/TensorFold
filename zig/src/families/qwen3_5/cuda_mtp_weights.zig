//! The Qwen3.6 MTP layer's weights (mtp-4bit.safetensors beside the checkpoint) onto the device, their own arena.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");

const Affine4 = cuda.qlinear.Affine4;
const Proj = engine.Proj;
const DeviceBuffer = cuda.DeviceBuffer;
const Engine = engine.Engine;
const aligned = engine.aligned;
const head_dim = geometry.head_dim;

pub const file = "mtp-4bit.safetensors";

pub const Weights = struct {
    arena: DeviceBuffer,
    arena_at: usize = 0,
    fc_e: Affine4, // fc's halves: the embedding's inputs, then the hidden state's
    fc_h: Affine4,
    norm_e: u64,
    norm_h: u64,
    input_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    norm: u64,
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    mlp: engine.Mlp,

    pub fn alloc(w: *Weights, bytes: usize) !u64 {
        const at = aligned(w.arena_at);
        if (at + bytes > w.arena.len) return error.ArenaExhausted;
        w.arena_at = at + bytes;
        return w.arena.ptr + at;
    }

    pub fn deinit(w: *Weights) void {
        w.arena.free();
    }
};

/// Device bytes of everything `load` packs.
pub fn total(g: geometry.Geometry) usize {
    var t = 2 * weights.packedBytes(g.hidden, g.hidden, 4) + 7 * aligned(@max(g.hidden, head_dim) * 2);
    t += weights.packedBytes(2 * g.attnWidth(), g.hidden, 4) + 2 * weights.packedBytes(g.kv_heads * head_dim, g.hidden, 4) + weights.packedBytes(g.hidden, g.attnWidth(), 4);
    t += weights.routedBytes(g.experts, g.intermediate, g.hidden, 4); // the MTP layer's experts are 4-bit
    return t;
}

/// The largest tensor `load` stages, as stored: fc's whole rows, an attention projection or an expert table.
fn stageBytes(g: geometry.Geometry) usize {
    const raw = weights.rawBytes;
    return @max(@max(raw(g.hidden, 2 * g.hidden, 4), raw(2 * g.attnWidth(), g.hidden, 4)), @max(raw(g.experts * g.intermediate, g.hidden, 4), raw(g.experts * g.hidden, g.intermediate, 4)));
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, e: *Engine, dir: []const u8) !Weights {
    const g = e.g;
    var ck: core.Checkpoint = .{ .gpa = gpa, .io = io };
    defer ck.close();
    try ck.add(dir, file);
    var w: Weights = .{ .arena = try DeviceBuffer.alloc(e.d, total(g)), .fc_e = undefined, .fc_h = undefined, .norm_e = 0, .norm_h = 0, .input_norm = 0, .post_norm = 0, .q_norm = 0, .k_norm = 0, .norm = 0, .q = undefined, .k = undefined, .v = undefined, .o = undefined, .mlp = undefined };
    errdefer w.arena.free();
    try w.arena.fill8(0, null);
    var stage = try DeviceBuffer.alloc(e.d, stageBytes(g)); // the load's staging, freed once the layer is packed
    defer stage.free();
    const m = "language_model.mtp.";
    const p = "language_model.mtp.layers.0.";
    w.fc_e = try weights.linearSlice(e, &w, &ck, &stage, m, "fc", g.hidden, 2 * g.hidden, 0, g.hidden);
    w.fc_h = try weights.linearSlice(e, &w, &ck, &stage, m, "fc", g.hidden, 2 * g.hidden, g.hidden, g.hidden);
    w.norm_e = try weights.tensor(&w, &ck, "{s}pre_fc_norm_embedding.weight", .{m}, .bf16, &.{g.hidden});
    w.norm_h = try weights.tensor(&w, &ck, "{s}pre_fc_norm_hidden.weight", .{m}, .bf16, &.{g.hidden});
    w.norm = try weights.tensor(&w, &ck, "{s}norm.weight", .{m}, .bf16, &.{g.hidden});
    w.input_norm = try weights.tensor(&w, &ck, "{s}input_layernorm.weight", .{p}, .bf16, &.{g.hidden});
    w.post_norm = try weights.tensor(&w, &ck, "{s}post_attention_layernorm.weight", .{p}, .bf16, &.{g.hidden});
    w.q_norm = try weights.tensor(&w, &ck, "{s}self_attn.q_norm.weight", .{p}, .bf16, &.{head_dim});
    w.k_norm = try weights.tensor(&w, &ck, "{s}self_attn.k_norm.weight", .{p}, .bf16, &.{head_dim});
    w.q = try weights.linear(e, &w, &ck, &stage, p, "self_attn.q_proj", 2 * g.attnWidth(), g.hidden, 4);
    w.k = try weights.linear(e, &w, &ck, &stage, p, "self_attn.k_proj", g.kv_heads * head_dim, g.hidden, 4);
    w.v = try weights.linear(e, &w, &ck, &stage, p, "self_attn.v_proj", g.kv_heads * head_dim, g.hidden, 4);
    w.o = try weights.linear(e, &w, &ck, &stage, p, "self_attn.o_proj", g.hidden, g.attnWidth(), 4);
    w.mlp = try weights.routed(e, &w, &ck, &stage, p, 4);
    return w;
}
