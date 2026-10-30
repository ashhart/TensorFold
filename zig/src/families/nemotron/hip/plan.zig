//! Nemotron-H's bytes on one HIP rank: the weights as tensor parallelism cuts them, a token's keys, a stream's state.
const std = @import("std");
const config = @import("../config.zig");

const Config = config.Config;

/// Stored: the checkpoint's bytes. Device: as the loader places them, its bf16 vectors widened to fp32.
pub const Mode = enum { stored, device };

/// The MTP head a rank 0 holds: none, the head alone, or the head and `drafts` rows of the vocabulary head.
pub const Head = struct { mtp: bool = false, drafts: usize = 0 };

/// Where a checkpoint's weights live on a rank of `world`.
pub const Weights = struct {
    /// On every rank: the embedding, the norms, the routers.
    replicated: u64 = 0,
    /// This rank's share: Mamba and attention heads, routed experts, vocabulary rows of the head.
    split: u64 = 0,
    /// Rank 0 alone: the shared experts.
    shared: u64 = 0,
    /// Rank 0 alone: the MTP head, whole, and its draft rows.
    head: u64 = 0,

    pub fn device(w: Weights, rank: u32) u64 {
        return w.replicated + w.split + if (rank == 0) w.shared + w.head else 0;
    }

    pub fn total(w: Weights, world: u32) u64 {
        return w.replicated * world + w.split * world + w.shared + w.head;
    }
};

/// Bytes of an MLX affine 4-bit group-64 matrix: packed codes, then a bf16 scale and bias a group.
pub fn affine(c: *const Config, n: u64, k: u64) u64 {
    return n * k * c.bits / 8 + 4 * n * (k / c.group_size);
}

fn even(n: u64, world: u32) !u64 {
    if (world == 0 or n % world != 0) return error.UnevenSplit;
    return n / world;
}

/// Key and value heads a rank holds: its share, one head shared by `world / kv_heads` ranks past that.
pub fn kvHeads(c: *const Config, world: u32) !u64 {
    if (world > c.kv_heads) {
        if (world % c.kv_heads != 0) return error.UnevenSplit;
        return 1;
    }
    return even(c.kv_heads, world);
}

/// A stored bf16 vector's bytes in `mode`.
fn float(mode: Mode, n: u64) u64 {
    return n * @as(u64, if (mode == .device) 4 else 2);
}

/// A mixer's bytes into `w`; `whole` keeps it uncut (the MTP head's, on rank 0).
fn mixer(c: *const Config, kind: config.Kind, world: u32, mode: Mode, whole: bool, w: *Weights) !void {
    const d: u64 = c.hidden;
    const ws: u32 = if (whole) 1 else world;
    var cut: u64 = 0;
    var mine: u64 = 0;
    switch (kind) {
        .mamba => {
            const heads = try even(c.mamba_heads, ws);
            const inner = heads * c.mamba_head_dim;
            const groups = try even(c.groups, ws);
            const conv = inner + 2 * groups * c.state;
            cut += affine(c, inner + conv + heads, d) + affine(c, d, inner);
            cut += float(mode, conv * c.conv_kernel + conv + 3 * heads + inner);
        },
        .attention => {
            const heads = try even(c.heads, ws);
            const kv = if (whole) c.kv_heads else try kvHeads(c, world);
            cut += affine(
                c,
                heads * c.head_dim,
                d,
            ) + 2 * affine(c, kv * c.head_dim, d) + affine(c, d, heads * c.head_dim);
        },
        .moe => {
            mine += float(mode, c.experts * d) + c.experts * 4;
            cut += try even(c.experts, ws) * (affine(c, c.expert_width, d) + affine(c, d, c.expert_width));
            const shared = affine(c, c.shared_width, d) + affine(c, d, c.shared_width);
            if (whole) cut += shared else w.shared += shared;
        },
    }
    if (whole) w.head += cut + mine else {
        w.split += cut;
        w.replicated += mine;
    }
}

/// The weights of `c` on one rank of `world`, the MTP head as `head` says.
pub fn weights(c: *const Config, world: u32, mode: Mode, head: Head) !Weights {
    var w: Weights = .{};
    const d: u64 = c.hidden;
    w.replicated += affine(c, c.vocab, d) + float(mode, d);
    w.split += affine(c, try even(c.vocab, world), d);
    for (c.kinds[0..c.layers]) |kind| {
        w.replicated += float(mode, d);
        try mixer(c, kind, world, mode, false, &w);
    }
    if (head.mtp) {
        w.head += affine(c, d, 2 * d) + float(mode, 5 * d);
        try mixer(c, .attention, world, mode, true, &w);
        try mixer(c, .moe, world, mode, true, &w);
        if (mode == .device and head.drafts > 0) w.head += affine(c, head.drafts, d) + head.drafts * 4;
    }
    return w;
}

/// Attention layers a stream's keys and values fill: the model's, and the MTP head's one.
pub fn attentionLayers(c: *const Config, mtp: bool) u64 {
    return c.count(.attention) + @intFromBool(mtp);
}

/// A position's cache bytes on one rank: keys and values of every attention layer, the MTP head's whole.
pub fn tokenBytes(c: *const Config, world: u32, act: u64, mtp: bool) !u64 {
    const heads = c.count(.attention) * (try kvHeads(c, world)) + if (mtp) c.kv_heads else 0;
    return heads * 2 * c.head_dim * act;
}

/// One Mamba layer's per-stream state on a rank: the conv tail (kernel - 1 rows, act) and the SSM state (fp32).
pub const MambaState = struct {
    conv: u64,
    ssm: u64,

    pub fn of(c: *const Config, world: u32, act: u64) !MambaState {
        const heads = try even(c.mamba_heads, world);
        const conv = heads * c.mamba_head_dim + 2 * (try even(c.groups, world)) * c.state;
        return .{ .conv = (c.conv_kernel - 1) * conv * act, .ssm = heads * c.mamba_head_dim * c.state * 4 };
    }
};

/// A stream's Mamba state on one rank, at any length.
pub fn streamBytes(c: *const Config, world: u32, act: u64) !u64 {
    const s = try MambaState.of(c, world, act);
    return c.count(.mamba) * (s.conv + s.ssm);
}

/// Positions one stream may hold in what the weights and `scratch` leave of `room`, or 0.
pub fn window(room: u64, weights_bytes: u64, scratch: u64, token: u64, stream: u64) u64 {
    const fixed = weights_bytes + scratch + stream;
    return if (fixed >= room) 0 else (room - fixed) / token;
}

fn lightning() !Config {
    var why: config.Why = .{};
    return config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"), &why);
}

test "the plan counts every byte of the stock checkpoint and its MTP head, and splits them without loss" {
    const c = try lightning();
    // the stock safetensors headers: four text shards, then mtp-4bit.safetensors
    const text: u64 = 5118606464 + 5291542144 + 5291542144 + 2073648640;
    const mtp: u64 = 751635200;
    const one = try weights(&c, 1, .stored, .{});
    try std.testing.expectEqual(text, one.total(1));
    try std.testing.expectEqual(text + mtp, (try weights(&c, 1, .stored, .{ .mtp = true })).total(1));
    for ([_]u32{ 2, 4, 8 }) |world| {
        const w = try weights(&c, world, .stored, .{});
        // every cut part sums back; only the replicated parts and the kv heads past two ranks add
        try std.testing.expectEqual(
            one.split,
            w.split * world - if (world > 2) (world - 2) * 6 * 2 * affine(&c, 128, 2688) else 0,
        );
        try std.testing.expectEqual(one.shared, w.shared);
        try std.testing.expectEqual(one.replicated, w.replicated);
    }
    try std.testing.expectError(error.UnevenSplit, weights(&c, 3, .stored, .{}));
    try std.testing.expectError(error.UnevenSplit, weights(&c, 16, .stored, .{}));
}

test "device bytes: the bf16 vectors widened, the draft rows of the head on rank 0" {
    const c = try lightning();
    const stored = try weights(&c, 1, .stored, .{ .mtp = true });
    const device = try weights(&c, 1, .device, .{ .mtp = true, .drafts = 49152 });
    // 52 block norms, norm_f, 23 Mamba layers' vectors, 24 routers and the head's 5 norms, each 2 bytes more a value
    const widened = 2 * (53 * 2688 + 23 * (6144 * 4 + 6144 + 3 * 64 + 4096) + 24 * 128 * 2688 + 5 * 2688);
    try std.testing.expectEqual(stored.total(1) + widened + 49152 * (1344 + 168) + 49152 * 4, device.total(1));
    const two = try weights(&c, 2, .device, .{ .mtp = true, .drafts = 49152 });
    try std.testing.expect(two.device(0) > two.device(1));
    try std.testing.expectEqual(two.device(0) - two.device(1), two.shared + two.head);
}

test "a token's keys and values, a stream's Mamba state, and the window a V620 leaves" {
    const c = try lightning();
    // 6 attention layers and the MTP head's, 2 kv heads of 128, 16-bit; a rank's head keeps every kv head
    try std.testing.expectEqual(@as(u64, 7 * 2 * 2 * 128 * 2), try tokenBytes(&c, 1, 2, true));
    try std.testing.expectEqual(@as(u64, (6 * 1 + 2) * 2 * 128 * 2), try tokenBytes(&c, 2, 2, true));
    try std.testing.expectEqual(@as(u64, (6 * 1 + 2) * 2 * 128 * 2), try tokenBytes(&c, 4, 2, true));
    try std.testing.expectEqual(@as(u64, 6 * 1 * 2 * 128 * 2), try tokenBytes(&c, 2, 2, false));
    // 23 Mamba layers: 3 conv rows of 6144 channels (16-bit), 64 heads of 64 x 128 fp32
    try std.testing.expectEqual(@as(u64, 23 * (3 * 6144 * 2 + 64 * 64 * 128 * 4)), try streamBytes(&c, 1, 2));
    try std.testing.expectEqual(try streamBytes(&c, 1, 2), 2 * try streamBytes(&c, 2, 2));
    const gib: u64 = 1 << 30;
    const w = (try weights(&c, 1, .device, .{ .mtp = true, .drafts = 49152 })).device(0);
    const most = window(30 * gib, w, 2 * gib, try tokenBytes(&c, 1, 2, true), try streamBytes(&c, 1, 2));
    try std.testing.expect(most >= 262144);
    try std.testing.expectEqual(@as(u64, 0), window(16 * gib, w, 2 * gib, 7168, 1 << 20));
}
