//! The prompt encoder's tensors from the checkpoint's text_encoder folder, bf16 as stored, its vision tower unread.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const index_format = @import("../../core/checkpoint_index.zig");
const cfg = @import("encoder_config.zig");
const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

/// A Gated DeltaNet layer's mixer: the q k v projection and its conv, z, b and a, the decay's terms, its norm, out.
pub const Delta = struct { qkv: Ref, z: Ref, b: Ref, a: Ref, conv: Ref, a_log: Ref, dt_bias: Ref, norm: Ref, out: Ref };

/// A full-attention layer's mixer: q (with its gate), k, v, o and the head norms.
pub const Attention = struct { q: Ref, k: Ref, v: Ref, o: Ref, q_norm: Ref, k_norm: Ref };

pub const Layer = struct {
    norm1: Ref,
    norm2: Ref,
    gate: Ref,
    up: Ref,
    down: Ref,
    mixer: union(enum) { delta: Delta, attention: Attention },
};

pub const Weights = struct {
    ck: ck.Checkpoint,
    layers: [cfg.layers]Layer,
    /// The embedding table [vocab, hidden] on the host (shared storage).
    embed: []const u16,
    norm: Ref,

    pub fn deinit(w: *Weights) void {
        w.ck.deinit();
    }
};

const prefix = "language_model.";

fn take(c: *const ck.Checkpoint, name: []const u8, shape: []const usize) !Ref {
    const t = try c.get(name);
    if (t.dtype != .bf16 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
        std.log.err("moss image encoder: {s} is {t} {any}; the encoder reads bf16 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
        return error.UnexpectedTensor;
    }
    if (t.offset % 16 != 0) return error.UnalignedTensor;
    return .{ .buf = t.buffer, .off = t.offset };
}

/// The folder `dir` (a checkpoint's text_encoder/): config checked, then every language-model tensor.
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
    try cfg.check(gpa, config);
    const index = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" }), arena, .limited(16 << 20));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, index, .{});
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    var it = (try index_format.weightMap(doc)).iterator();
    while (it.next()) |kv| if (std.mem.startsWith(u8, kv.key_ptr.*, prefix)) try files.put(arena, kv.value_ptr.string, {});
    var w: Weights = undefined;
    w.ck = ck.Checkpoint.init(gpa);
    errdefer w.ck.deinit();
    for (files.keys()) |file| {
        const path = try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ dir, file }, 0);
        try w.ck.addFileSelected(device, path, "", prefix);
    }
    const h = cfg.hidden;
    const vd = cfg.v_heads * cfg.v_dim;
    var name: [96]u8 = undefined;
    for (&w.layers, 0..) |*l, i| {
        const at = struct {
            fn n(buf: []u8, layer: usize, field: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, prefix ++ "layers.{d}.{s}", .{ layer, field });
            }
        };
        l.norm1 = try take(&w.ck, try at.n(&name, i, "input_layernorm.weight"), &.{h});
        l.norm2 = try take(&w.ck, try at.n(&name, i, "post_attention_layernorm.weight"), &.{h});
        l.gate = try take(&w.ck, try at.n(&name, i, "mlp.gate_proj.weight"), &.{ cfg.ffn, h });
        l.up = try take(&w.ck, try at.n(&name, i, "mlp.up_proj.weight"), &.{ cfg.ffn, h });
        l.down = try take(&w.ck, try at.n(&name, i, "mlp.down_proj.weight"), &.{ h, cfg.ffn });
        if (cfg.isFull(i)) {
            const d = cfg.head_dim;
            l.mixer = .{ .attention = .{
                .q = try take(&w.ck, try at.n(&name, i, "self_attn.q_proj.weight"), &.{ 2 * cfg.heads * d, h }),
                .k = try take(&w.ck, try at.n(&name, i, "self_attn.k_proj.weight"), &.{ cfg.kv_heads * d, h }),
                .v = try take(&w.ck, try at.n(&name, i, "self_attn.v_proj.weight"), &.{ cfg.kv_heads * d, h }),
                .o = try take(&w.ck, try at.n(&name, i, "self_attn.o_proj.weight"), &.{ h, cfg.heads * d }),
                .q_norm = try take(&w.ck, try at.n(&name, i, "self_attn.q_norm.weight"), &.{d}),
                .k_norm = try take(&w.ck, try at.n(&name, i, "self_attn.k_norm.weight"), &.{d}),
            } };
        } else {
            l.mixer = .{ .delta = .{
                .qkv = try take(&w.ck, try at.n(&name, i, "linear_attn.in_proj_qkv.weight"), &.{ cfg.mix, h }),
                .z = try take(&w.ck, try at.n(&name, i, "linear_attn.in_proj_z.weight"), &.{ vd, h }),
                .b = try take(&w.ck, try at.n(&name, i, "linear_attn.in_proj_b.weight"), &.{ cfg.v_heads, h }),
                .a = try take(&w.ck, try at.n(&name, i, "linear_attn.in_proj_a.weight"), &.{ cfg.v_heads, h }),
                .conv = try take(&w.ck, try at.n(&name, i, "linear_attn.conv1d.weight"), &.{ cfg.mix, 1, cfg.conv_taps }),
                .a_log = try take(&w.ck, try at.n(&name, i, "linear_attn.A_log"), &.{cfg.v_heads}),
                .dt_bias = try take(&w.ck, try at.n(&name, i, "linear_attn.dt_bias"), &.{cfg.v_heads}),
                .norm = try take(&w.ck, try at.n(&name, i, "linear_attn.norm.weight"), &.{cfg.v_dim}),
                .out = try take(&w.ck, try at.n(&name, i, "linear_attn.out_proj.weight"), &.{ h, vd }),
            } };
        }
    }
    const embed = try take(&w.ck, prefix ++ "embed_tokens.weight", &.{ cfg.vocab, h });
    w.embed = @as([*]const u16, @ptrCast(@alignCast(embed.addr())))[0 .. cfg.vocab * h];
    w.norm = try take(&w.ck, prefix ++ "norm.weight", &.{h});
    const full = cfg.layers / cfg.full_every;
    const read = 5 * cfg.layers + 6 * full + 9 * (cfg.layers - full) + 2;
    if (w.ck.tensors.count() != read) {
        std.log.err("moss image encoder: the checkpoint has {d} language-model tensors; the encoder reads {d}", .{ w.ck.tensors.count(), read });
        return error.UnexpectedTensors;
    }
    return w;
}
