//! Inkling 2 Mini's DSpark draft head (mtp.safetensors, 54 bf16 tensors): buffer views checked to the config's shapes.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const config = @import("config.zig");
const gm = @import("metal_vision_gemm.zig");

pub const Ref = gm.Ref;

/// The draft head's attention head width: the kernels run 128-wide heads.
pub const head_dim = 128;
/// Rows a block holds at most, and so the most queries an attention pass takes.
pub const max_block = 8;
/// The most draft layers (config.Mtp's local flags).
pub const max_layers = 8;

pub const Layer = struct {
    in_norm: Ref,
    post_norm: Ref,
    ctx_norm: Ref, // dflare_norm: the mixed target features feeding this layer's context keys and values
    q_norm: Ref,
    k_norm: Ref,
    k_norm_ctx: Ref,
    q: Ref,
    k: Ref,
    v: Ref,
    o: Ref,
    k_ctx: Ref,
    v_ctx: Ref,
    gate: Ref,
    up: Ref,
    down: Ref,
};

pub const Weights = struct {
    ckpt: ck.Checkpoint,
    fuse: Ref, // [layers + 1, taps]: tap-mixture logits, rows 0.. per layer, the last the Markov head's
    mask: Ref,
    norm: Ref,
    markov_norm: Ref,
    hidden_out_w: Ref, // [hidden, rank]
    hidden_out_b: Ref,
    w1: Ref, // [vocab, rank]: the previous token's embedding
    rnn_w: Ref, // [3 rank, 2 hidden + 2 rank]
    rnn_b: Ref,
    layers: [max_layers]Layer,

    /// `dir`/mtp.safetensors read whole into one shared buffer, every tensor checked for dtype and exact shape.
    pub fn load(gpa: std.mem.Allocator, dir: []const u8, device: mtl.Device, c: *const config.Config) !Weights {
        const m = c.mtp orelse return error.NoDraftHead;
        if (m.head_dim != head_dim or m.layers > max_layers or m.block < 2 or m.block > max_block) return error.UnsupportedDraftHead;
        if (m.q_heads / m.kv_heads > 8) return error.UnsupportedDraftHead;
        // taps are logical visit ids of the looped backbone (the last one its final norm), never physical layers
        const logical = c.text.executionLayers();
        for (m.taps[0..m.taps_n]) |tap| if (tap > logical) return error.UnsupportedDraftHead;
        var w: Weights = undefined;
        w.ckpt = ck.Checkpoint.init(gpa);
        errdefer w.ckpt.deinit();
        const joined = try std.fs.path.join(gpa, &.{ dir, "mtp.safetensors" });
        defer gpa.free(joined);
        const path = try gpa.dupeSentinel(u8, joined, 0);
        defer gpa.free(path);
        try w.ckpt.addFile(device, path, "");
        const h: usize = c.text.hidden;
        const r: usize = m.rank;
        const qw: usize = @as(usize, m.q_heads) * head_dim;
        const kvw: usize = @as(usize, m.kv_heads) * head_dim;
        const inter: usize = m.inter;
        w.fuse = try w.get("fuse_weight", &.{ m.layers + 1, m.taps_n });
        w.mask = try w.get("mask_embed", &.{h});
        w.norm = try w.get("norm.weight", &.{h});
        w.markov_norm = try w.get("markov_head.dflare_norm.weight", &.{h});
        w.hidden_out_w = try w.get("markov_head.hidden_out.weight", &.{ h, r });
        w.hidden_out_b = try w.get("markov_head.hidden_out.bias", &.{h});
        w.w1 = try w.get("markov_head.markov_w1.weight", &.{ c.text.vocab, r });
        w.rnn_w = try w.get("markov_head.rnn_proj.weight", &.{ 3 * r, 2 * h + 2 * r });
        w.rnn_b = try w.get("markov_head.rnn_proj.bias", &.{3 * r});
        for (w.layers[0..m.layers], 0..) |*l, i| {
            l.in_norm = try w.layer(i, "input_layernorm.weight", &.{h});
            l.post_norm = try w.layer(i, "post_attention_layernorm.weight", &.{h});
            l.ctx_norm = try w.layer(i, "dflare_norm.weight", &.{h});
            l.q_norm = try w.layer(i, "self_attn.q_norm.weight", &.{head_dim});
            l.k_norm = try w.layer(i, "self_attn.k_norm.weight", &.{head_dim});
            l.k_norm_ctx = try w.layer(i, "self_attn.k_norm_ctx.weight", &.{head_dim});
            l.q = try w.layer(i, "self_attn.q_proj.weight", &.{ qw, h });
            l.k = try w.layer(i, "self_attn.k_proj.weight", &.{ kvw, h });
            l.v = try w.layer(i, "self_attn.v_proj.weight", &.{ kvw, h });
            l.o = try w.layer(i, "self_attn.o_proj.weight", &.{ h, qw });
            l.k_ctx = try w.layer(i, "self_attn.k_proj_ctx.weight", &.{ kvw, h });
            l.v_ctx = try w.layer(i, "self_attn.v_proj_ctx.weight", &.{ kvw, h });
            l.gate = try w.layer(i, "mlp.gate_proj.weight", &.{ inter, h });
            l.up = try w.layer(i, "mlp.up_proj.weight", &.{ inter, h });
            l.down = try w.layer(i, "mlp.down_proj.weight", &.{ h, inter });
        }
        return w;
    }

    pub fn deinit(w: *Weights) void {
        w.ckpt.deinit();
    }

    fn get(w: *const Weights, name: []const u8, dims: []const usize) !Ref {
        const x = try w.ckpt.get(name);
        const ok = x.dtype == .bf16 and x.rank == dims.len and std.mem.eql(usize, x.shape[0..x.rank], dims);
        if (!ok) {
            std.log.err("inkling2 draft: {s} is not bf16 {any}", .{ name, dims });
            return error.UnexpectedDraftTensor;
        }
        return .{ .buf = x.buffer, .off = x.offset };
    }

    fn layer(w: *const Weights, i: usize, comptime tail: []const u8, dims: []const usize) !Ref {
        var buf: [96]u8 = undefined;
        return w.get(try std.fmt.bufPrint(&buf, "layers.{d}." ++ tail, .{i}), dims);
    }

    /// Bytes the weights occupy on the device.
    pub fn bytes(w: *const Weights) u64 {
        var total: u64 = 0;
        for (w.ckpt.shards.items) |s| total += s.buffer.length();
        return total;
    }
};
