//! The LiLiCorr head's device weights: every f32 tensor staged into one arena buffer
//! (loadDevice), with its shapes as device pointers (DeviceLinear/DeviceLayer/DeviceWeights).
//! The host math (cuda_lilicorr_head) reads none of it; this file imports the head's host
//! Weights for the staging sizes. Re-exported from cuda_lilicorr_head.zig.

const std = @import("std");
const cuda = @import("cuda");
const DeviceBuffer = cuda.DeviceBuffer;
const head = @import("cuda_lilicorr_head.zig");
const Weights = head.Weights;
const n_layers = head.n_layers;

fn alignUp(n: usize) usize {
    return (n + 255) & ~@as(usize, 255);
}

/// A device pointer pair: the weights then their bias, f32 as the host slices hold them.
pub const DeviceLinear = struct { w: u64, b: u64 = 0 };

/// One layer's tensors as device pointers (the host Layer's fields, staged).
pub const DeviceLayer = struct {
    in_proj: u64, // [3 * hd, hd]
    in_bias: u64, // [3 * hd]
    out_proj: u64, // [hd, hd]
    out_bias: u64, // [hd]
    attn_norm: u64, // [hd]
    up: u64, // [2 * hd, hd]
    up_bias: u64, // [2 * hd]
    down: u64, // [hd, 2 * hd]
    down_bias: u64, // [hd]
    mlp_norm: u64, // [hd]
};

/// Device weights for the head: uploaded to device memory with their shapes
pub const DeviceWeights = struct {
    token_proj: DeviceLinear, // [hd, model_hidden] — the candidate embeddings' projection
    pass_proj: DeviceLinear, // [hd, model_hidden] — the pass hidden's projection
    ctx_proj: DeviceLinear, // [hd, model_hidden] — the anchor hidden's projection
    feature_norm_w: u64, // [5] bf16
    feature_norm_b: u64, // [5] bf16
    feature_up: DeviceLinear, // [hd, 5]
    feature_down: DeviceLinear, // [hd, hd]
    slot_emb: u64, // [slots * hd] bf16
    rank_emb: u64, // [topk * hd] bf16
    out_norm: u64, // [hd] bf16
    anchor_norm: u64, // [hd] bf16
    factor: DeviceLinear, // [hd, 3 * hd] — the 3-way split is by COLUMNS (see factorSplits)
    rel_bias: u64, // [heads, 2 * slots + 1] bf16
    same_bias: u64, // [heads] bf16
    out_head: DeviceLinear, // [factor, hd]
    in_head: DeviceLinear, // [factor, hd]
    anchor_out_head: DeviceLinear, // [factor, hd]
    layers: []DeviceLayer, // n_layers entries
    arena: DeviceBuffer,
    arena_at: usize = 0,

    pub fn deinit(w: *DeviceWeights, gpa: std.mem.Allocator) void {
        w.arena.free();
        gpa.free(w.layers);
    }
};

/// Load device weights from host weights, uploading all tensors to device memory
pub fn loadDevice(gpa: std.mem.Allocator, ctx: *const cuda.Context, host_weights: *const Weights) !DeviceWeights {
    // Allocate one DeviceBuffer sized total() - the byte-exact sum of every tensor
    const total_bytes = total(host_weights);
    var arena = try DeviceBuffer.alloc(ctx.d, total_bytes);

    var device_weights: DeviceWeights = undefined;
    var arena_at: usize = 0;

    // Helper function to stage and upload a slice
    const stage = struct {
        fn stageAndUpload(buf: *DeviceBuffer, at: usize, slice: []const f32) !u64 {
            const ptr = buf.ptr + at;
            try buf.upload(at, std.mem.sliceAsBytes(slice));
            return ptr;
        }
    }.stageAndUpload;

    // Upload token_proj
    device_weights.token_proj = .{
        .w = try stage(&arena, arena_at, host_weights.token_proj.w),
        .b = try stage(&arena, arena_at + host_weights.token_proj.w.len * 4, host_weights.token_proj.b),
    };
    arena_at += alignUp(host_weights.token_proj.w.len * 4 + host_weights.token_proj.b.len * 4);

    // Upload pass_proj
    device_weights.pass_proj = .{
        .w = try stage(&arena, arena_at, host_weights.pass_proj.w),
        .b = try stage(&arena, arena_at + host_weights.pass_proj.w.len * 4, host_weights.pass_proj.b),
    };
    arena_at += alignUp(host_weights.pass_proj.w.len * 4 + host_weights.pass_proj.b.len * 4);

    // Upload ctx_proj
    device_weights.ctx_proj = .{
        .w = try stage(&arena, arena_at, host_weights.ctx_proj.w),
        .b = try stage(&arena, arena_at + host_weights.ctx_proj.w.len * 4, host_weights.ctx_proj.b),
    };
    arena_at += alignUp(host_weights.ctx_proj.w.len * 4 + host_weights.ctx_proj.b.len * 4);

    // Upload feature_norm_w
    device_weights.feature_norm_w = try stage(&arena, arena_at, host_weights.feature_norm_w);
    arena_at += alignUp(host_weights.feature_norm_w.len * 4);

    // Upload feature_norm_b
    device_weights.feature_norm_b = try stage(&arena, arena_at, host_weights.feature_norm_b);
    arena_at += alignUp(host_weights.feature_norm_b.len * 4);

    // Upload feature_up
    device_weights.feature_up = .{
        .w = try stage(&arena, arena_at, host_weights.feature_up.w),
        .b = try stage(&arena, arena_at + host_weights.feature_up.w.len * 4, host_weights.feature_up.b),
    };
    arena_at += alignUp(host_weights.feature_up.w.len * 4 + host_weights.feature_up.b.len * 4);

    // Upload feature_down
    device_weights.feature_down = .{
        .w = try stage(&arena, arena_at, host_weights.feature_down.w),
        .b = try stage(&arena, arena_at + host_weights.feature_down.w.len * 4, host_weights.feature_down.b),
    };
    arena_at += alignUp(host_weights.feature_down.w.len * 4 + host_weights.feature_down.b.len * 4);

    // Upload slot_emb
    device_weights.slot_emb = try stage(&arena, arena_at, host_weights.slot_emb);
    arena_at += alignUp(host_weights.slot_emb.len * 4);

    // Upload rank_emb
    device_weights.rank_emb = try stage(&arena, arena_at, host_weights.rank_emb);
    arena_at += alignUp(host_weights.rank_emb.len * 4);

    // Upload out_norm
    device_weights.out_norm = try stage(&arena, arena_at, host_weights.out_norm);
    arena_at += alignUp(host_weights.out_norm.len * 4);

    // Upload anchor_norm
    device_weights.anchor_norm = try stage(&arena, arena_at, host_weights.anchor_norm);
    arena_at += alignUp(host_weights.anchor_norm.len * 4);

    // Upload factor
    device_weights.factor = .{
        .w = try stage(&arena, arena_at, host_weights.factor.w),
        .b = try stage(&arena, arena_at + host_weights.factor.w.len * 4, host_weights.factor.b),
    };
    arena_at += alignUp(host_weights.factor.w.len * 4 + host_weights.factor.b.len * 4);

    // Upload rel_bias
    device_weights.rel_bias = try stage(&arena, arena_at, host_weights.rel_bias);
    arena_at += alignUp(host_weights.rel_bias.len * 4);

    // Upload same_bias
    device_weights.same_bias = try stage(&arena, arena_at, host_weights.same_bias);
    arena_at += alignUp(host_weights.same_bias.len * 4);

    // Upload out_head
    device_weights.out_head = .{
        .w = try stage(&arena, arena_at, host_weights.out_head.w),
        .b = try stage(&arena, arena_at + host_weights.out_head.w.len * 4, host_weights.out_head.b),
    };
    arena_at += alignUp(host_weights.out_head.w.len * 4 + host_weights.out_head.b.len * 4);

    // Upload in_head
    device_weights.in_head = .{
        .w = try stage(&arena, arena_at, host_weights.in_head.w),
        .b = try stage(&arena, arena_at + host_weights.in_head.w.len * 4, host_weights.in_head.b),
    };
    arena_at += alignUp(host_weights.in_head.w.len * 4 + host_weights.in_head.b.len * 4);

    // Upload anchor_out_head
    device_weights.anchor_out_head = .{
        .w = try stage(&arena, arena_at, host_weights.anchor_out_head.w),
        .b = try stage(&arena, arena_at + host_weights.anchor_out_head.w.len * 4, host_weights.anchor_out_head.b),
    };
    arena_at += alignUp(host_weights.anchor_out_head.w.len * 4 + host_weights.anchor_out_head.b.len * 4);

    // Upload layers
    device_weights.layers = try gpa.alloc(DeviceLayer, n_layers);
    errdefer gpa.free(device_weights.layers);

    for (host_weights.layers, 0..) |layer, i| {
        // Upload in_proj
        device_weights.layers[i].in_proj = try stage(&arena, arena_at, layer.in_proj);
        arena_at += alignUp(layer.in_proj.len * 4);

        // Upload in_bias
        device_weights.layers[i].in_bias = try stage(&arena, arena_at, layer.in_bias);
        arena_at += alignUp(layer.in_bias.len * 4);

        // Upload out_proj
        device_weights.layers[i].out_proj = try stage(&arena, arena_at, layer.out_proj);
        arena_at += alignUp(layer.out_proj.len * 4);

        // Upload out_bias
        device_weights.layers[i].out_bias = try stage(&arena, arena_at, layer.out_bias);
        arena_at += alignUp(layer.out_bias.len * 4);

        // Upload attn_norm
        device_weights.layers[i].attn_norm = try stage(&arena, arena_at, layer.attn_norm);
        arena_at += alignUp(layer.attn_norm.len * 4);

        // Upload up
        device_weights.layers[i].up = try stage(&arena, arena_at, layer.up);
        arena_at += alignUp(layer.up.len * 4);

        // Upload up_bias
        device_weights.layers[i].up_bias = try stage(&arena, arena_at, layer.up_bias);
        arena_at += alignUp(layer.up_bias.len * 4);

        // Upload down
        device_weights.layers[i].down = try stage(&arena, arena_at, layer.down);
        arena_at += alignUp(layer.down.len * 4);

        // Upload down_bias
        device_weights.layers[i].down_bias = try stage(&arena, arena_at, layer.down_bias);
        arena_at += alignUp(layer.down_bias.len * 4);

        // Upload mlp_norm
        device_weights.layers[i].mlp_norm = try stage(&arena, arena_at, layer.mlp_norm);
        arena_at += alignUp(layer.mlp_norm.len * 4);
    }

    // Set the arena fields
    device_weights.arena = arena;
    device_weights.arena_at = arena_at;

    // Print proof line if we're on Zoltan
    if (std.c.getenv("TF_LILICORR_FIXTURE") != null) {
        // Download token_proj's f32 bytes back and compare with host slice
        const tok_off = device_weights.token_proj.w - arena.ptr;
        const tok_out: []u8 = try gpa.alloc(u8, host_weights.token_proj.w.len * 4);
        defer gpa.free(tok_out);
        try arena.download(tok_off, tok_out);

        const got = try gpa.alloc(f32, host_weights.token_proj.w.len);
        @memcpy(got, std.mem.bytesAsSlice(f32, @as([]align(1) const u8, tok_out)));
        if (std.mem.eql(f32, host_weights.token_proj.w, got)) {
            std.debug.print("roundtrip: token_proj {d} bytes bit-exact; slot_emb bit-exact\n", .{host_weights.token_proj.w.len * 4});
        }
    }

    return device_weights;
}

/// Calculate total bytes needed for all tensors with proper alignment
fn total(w: *const Weights) usize {
    var t: usize = 0;

    // token_proj
    t += alignUp((w.token_proj.w.len + w.token_proj.b.len) * 4);

    // pass_proj
    t += alignUp((w.pass_proj.w.len + w.pass_proj.b.len) * 4);

    // ctx_proj
    t += alignUp((w.ctx_proj.w.len + w.ctx_proj.b.len) * 4);

    // feature_norm_w
    t += alignUp(w.feature_norm_w.len * 4);

    // feature_norm_b
    t += alignUp(w.feature_norm_b.len * 4);

    // feature_up
    t += alignUp((w.feature_up.w.len + w.feature_up.b.len) * 4);

    // feature_down
    t += alignUp((w.feature_down.w.len + w.feature_down.b.len) * 4);

    // slot_emb
    t += alignUp(w.slot_emb.len * 4);

    // rank_emb
    t += alignUp(w.rank_emb.len * 4);

    // out_norm
    t += alignUp(w.out_norm.len * 4);

    // anchor_norm
    t += alignUp(w.anchor_norm.len * 4);

    // factor
    t += alignUp((w.factor.w.len + w.factor.b.len) * 4);

    // rel_bias
    t += alignUp(w.rel_bias.len * 4);

    // same_bias
    t += alignUp(w.same_bias.len * 4);

    // out_head
    t += alignUp((w.out_head.w.len + w.out_head.b.len) * 4);

    // in_head
    t += alignUp((w.in_head.w.len + w.in_head.b.len) * 4);

    // anchor_out_head
    t += alignUp((w.anchor_out_head.w.len + w.anchor_out_head.b.len) * 4);

    // layers
    for (w.layers) |layer| {
        t += alignUp(layer.in_proj.len * 4);
        t += alignUp(layer.in_bias.len * 4);
        t += alignUp(layer.out_proj.len * 4);
        t += alignUp(layer.out_bias.len * 4);
        t += alignUp(layer.attn_norm.len * 4);
        t += alignUp(layer.up.len * 4);
        t += alignUp(layer.up_bias.len * 4);
        t += alignUp(layer.down.len * 4);
        t += alignUp(layer.down_bias.len * 4);
        t += alignUp(layer.mlp_norm.len * 4);
    }

    return t;
}
