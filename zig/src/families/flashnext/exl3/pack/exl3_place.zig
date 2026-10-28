//! Header-only home for one tensor name.

const std = @import("std");
const scan = @import("exl3_scan.zig");
const shape = @import("exl3_shape.zig");
const cache = @import("../model/exl3_cache.zig");
const tp = @import("../tp/exl3_tp.zig");

pub const Home = enum {
    ngram_host, // PLE rows, page cache or SSD, never a VRAM alloc
    routed_host, // routed expert bank, host RAM, cache policy decides GPU copies
    shared_resident, // shared expert, replicated, stays on device
    resident, // dense, router, HC, MTP head, unquantized fp16 linears
    centred_norm, // add centredOffset in fp32 at load
    mtp_patch, // the mixer file kept outside the index
    vision_skip, // vision encoder, not this milestone
    plain,
};

pub const Tensor = struct {
    name: []const u8,
    rank: usize,
    /// Filename of the shard, used only for the MTP mixer patch.
    file: []const u8 = "",
};

const centred_names = [_][]const u8{
    "hc_norm",
    "q_norm",
    "k_norm",
    "q_layernorm",
    "k_layernorm",
    "norm_key",
    "norm_query",
    "norm_conv",
    "pre_fc_norm_embedding",
    "pre_fc_norm_hidden",
};

const resident_names = [_][]const u8{
    "lm_head",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "router",
    "hyper_connection",
    "in_proj",
    "conv1d",
    "A_log",
    "dt_bias",
    "indexer",
    "mtp",
    // Router logits and the PLE pair. `k_proj` is not a substring of `key_proj`.
    "mlp.gate",
    "key_proj",
    "value_proj",
};

comptime {
    std.debug.assert(shape.experts == 512);
    std.debug.assert(shape.shared_expert == shape.experts);
}

pub fn home(t: Tensor) Home {
    if (scan.isMtpMixerPatch(t.file)) return .mtp_patch;

    const kind = scan.classifyName(t.name, t.rank);
    if (kind == .ngram_row) return .ngram_host;
    if (isVision(t.name)) return .vision_skip;
    if (contains(t.name, "shared_expert")) return .shared_resident;
    if (isRouted(t.name)) return .routed_host;

    // Gamma is a rank-1 `.plain` weight.
    if (isCentred(t.name)) {
        if (kind == .exl3_part) return .plain;
        return .centred_norm;
    }

    // `mtp` pins the MTP head.
    if (contains(t.name, "mtp.fc_embedding")) return .plain;

    if ((kind == .exl3_part or kind == .plain) and isResident(t.name)) return .resident;
    return .plain;
}

/// True when `which` is the cache's host place and both host banks refuse a resident copy.
pub fn hostBank(which: cache.Place) bool {
    const budget = cache.Budget{
        .device_bytes = 0,
        .resident_bytes = 0,
        .kv_bytes = 0,
        .routed_expert_bytes = 0,
        .ngram_bytes = 0,
    };
    const host = cache.refusesResident(.routed) and cache.refusesResident(.ngram) and cache.placeWeights(budget) == .host;
    return host and which == .host;
}

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

fn isVision(name: []const u8) bool {
    return contains(name, "visual") or std.mem.startsWith(u8, name, "vision") or contains(name, ".vision.");
}

/// `.experts.` / `experts.` gate, up, and down for ids 0..511. Any `shared` substring stays out.
fn isRouted(name: []const u8) bool {
    if (contains(name, "shared")) return false;
    return contains(name, ".experts.") or contains(name, "experts.");
}

/// Token is a full dot-component, or the name's suffix.
pub fn isCentred(name: []const u8) bool {
    for (centred_names) |token| {
        if (std.mem.endsWith(u8, name, token)) {
            if (name.len == token.len or name[name.len - token.len - 1] == '.') return true;
        }
        var it = std.mem.splitScalar(u8, name, '.');
        while (it.next()) |part| if (std.mem.eql(u8, part, token)) return true;
    }
    return false;
}

fn isResident(name: []const u8) bool {
    for (resident_names) |token| {
        if (!contains(name, token)) continue;
        // `mlp.switch_mlp.gate_proj` contains `mlp.gate` and is not the router logits.
        if (std.mem.eql(u8, token, "mlp.gate") and contains(name, "gate_proj")) continue;
        return true;
    }
    return false;
}

test "a tensor name picks one home and the host banks stay off device" {
    const ngram = "model.language_model.layers.2.ple.ngram_embedding.shard_0.trellis";
    try std.testing.expectEqual(Home.ngram_host, home(.{ .name = ngram, .rank = 2 }));

    const wide = home(.{ .name = ngram, .rank = 3 });
    try std.testing.expect(wide != .ngram_host);
    try std.testing.expectEqual(Home.plain, wide);

    try std.testing.expectEqual(Home.routed_host, home(.{
        .name = "model.layers.0.mlp.experts.0.up_proj.trellis",
        .rank = 3,
    }));
    try std.testing.expectEqual(Home.shared_resident, home(.{
        .name = "model.layers.0.mlp.shared_expert.up_proj.trellis",
        .rank = 3,
    }));
    try std.testing.expectEqual(Home.centred_norm, home(.{
        .name = "model.layers.0.self_attn.q_norm.weight",
        .rank = 1,
    }));
    try std.testing.expectEqual(Home.centred_norm, home(.{
        .name = "model.layers.0.attn_hyper_connection.hc_norm.weight",
        .rank = 1,
    }));
    try std.testing.expectEqual(Home.centred_norm, home(.{
        .name = "language_model.mtp.pre_fc_norm_embedding.weight",
        .rank = 1,
    }));
    try std.testing.expectEqual(Home.plain, home(.{
        .name = "model.layers.0.self_attn.q_norm.trellis",
        .rank = 3,
    }));
    try std.testing.expectEqual(Home.mtp_patch, home(.{
        .name = "weight",
        .rank = 1,
        .file = "mtp_hyper_connection_mixer_patch.safetensors",
    }));
    try std.testing.expectEqual(Home.vision_skip, home(.{
        .name = "model.visual.blocks.0.weight",
        .rank = 1,
    }));
    try std.testing.expectEqual(Home.plain, home(.{
        .name = "model.embed_tokens.weight",
        .rank = 2,
    }));
    try std.testing.expectEqual(Home.resident, home(.{
        .name = "model.layers.0.linear_attn.out_proj.weight",
        .rank = 2,
    }));
    try std.testing.expectEqual(Home.plain, home(.{ .name = "readme", .rank = 0 }));
    try std.testing.expect(hostBank(.host));
}

/// Where the bytes for this rank live.
pub const Stay = enum { vram_pinned, vram_learned, host, disk, other_rank, skip };

/// Integer after `.experts.`, or null when the name is not a routed expert.
pub fn expertId(name: []const u8) ?u16 {
    const key = ".experts.";
    const at = std.mem.indexOf(u8, name, key) orelse return null;
    var i = at + key.len;
    if (i >= name.len or name[i] < '0' or name[i] > '9') return null;
    var n: u32 = 0;
    while (i < name.len and name[i] >= '0' and name[i] <= '9') : (i += 1) {
        n = n * 10 + (name[i] - '0');
        if (n > std.math.maxInt(u16)) return null;
    }
    if (i >= name.len or name[i] != '.') return null;
    return @intCast(n);
}

pub fn stay(which: Home, id: ?u16, rank: usize, ranks: usize) error{ IllegalCut, RankOutOfRange }!Stay {
    if (ranks != 1 and ranks != 2) return error.IllegalCut;
    if (rank >= ranks) return error.RankOutOfRange;
    switch (which) {
        .ngram_host => return .disk,
        .vision_skip => return .skip,
        .routed_host => {
            const expert = id orelse return error.IllegalCut;
            const span = try tp.range(.routed_set, rank, ranks);
            if (@as(usize, expert) >= span.lo and @as(usize, expert) < span.hi) return .vram_learned;
            return .other_rank;
        },
        .plain => return .host,
        .shared_resident, .resident, .centred_norm, .mtp_patch => return .vram_pinned,
    }
}

pub fn stayOf(t: Tensor, rank: usize, ranks: usize) error{ IllegalCut, RankOutOfRange }!Stay {
    return stay(home(t), expertId(t.name), rank, ranks);
}

test "tp 2 pins the decode path, leaves the n-gram on disk, and learns only this rank's experts" {
    const ngram = "model.language_model.layers.2.ple.ngram_embedding.shard_0.trellis";
    try std.testing.expectEqual(Stay.disk, try stayOf(.{ .name = ngram, .rank = 2 }, 0, 2));
    try std.testing.expectEqual(Stay.disk, try stayOf(.{ .name = ngram, .rank = 2 }, 1, 2));

    const pinned = [_][]const u8{
        "model.layers.0.mlp.shared_expert.up_proj.trellis",
        "lm_head.weight",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp_hyper_connection.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.out_proj.weight",
        "model.layers.0.linear_attn.conv1d.weight",
        "model.layers.0.linear_attn.A_log",
        "model.layers.0.linear_attn.dt_bias",
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.1.ple.key_proj.weight",
        "model.language_model.layers.1.ple.value_proj.weight",
    };
    try std.testing.expectEqual(Stay.host, try stayOf(.{
        .name = "model.embed_tokens.weight",
        .rank = 2,
    }, 0, 2));
    try std.testing.expectEqual(Stay.host, try stayOf(.{ .name = "readme", .rank = 0 }, 1, 2));
    for (pinned) |name| {
        try std.testing.expectEqual(Stay.vram_pinned, try stayOf(.{ .name = name, .rank = 2 }, 0, 2));
        try std.testing.expectEqual(Stay.vram_pinned, try stayOf(.{ .name = name, .rank = 2 }, 1, 2));
    }
    try std.testing.expectEqual(Stay.vram_pinned, try stayOf(.{
        .name = "model.layers.0.self_attn.q_norm.weight",
        .rank = 1,
    }, 0, 2));
    try std.testing.expectEqual(Stay.vram_pinned, try stayOf(.{
        .name = "weight",
        .rank = 1,
        .file = "mtp_hyper_connection_mixer_patch.safetensors",
    }, 1, 2));
    try std.testing.expectEqual(Stay.skip, try stayOf(.{
        .name = "model.visual.blocks.0.weight",
        .rank = 1,
    }, 0, 2));

    try std.testing.expectEqual(@as(?u16, 0), expertId("model.layers.0.mlp.experts.0.up_proj.trellis"));
    try std.testing.expectEqual(@as(?u16, 255), expertId("model.layers.0.mlp.experts.255.down_proj.trellis"));
    try std.testing.expectEqual(@as(?u16, 511), expertId("model.layers.0.mlp.experts.511.gate_proj.trellis"));
    try std.testing.expectEqual(@as(?u16, null), expertId("model.layers.0.mlp.shared_expert.up_proj.trellis"));

    const low = "model.layers.0.mlp.experts.0.up_proj.trellis";
    const high = "model.layers.0.mlp.experts.400.up_proj.trellis";
    try std.testing.expectEqual(Stay.vram_learned, try stayOf(.{ .name = low, .rank = 3 }, 0, 2));
    try std.testing.expectEqual(Stay.other_rank, try stayOf(.{ .name = low, .rank = 3 }, 1, 2));
    try std.testing.expectEqual(Stay.other_rank, try stayOf(.{ .name = high, .rank = 3 }, 0, 2));
    try std.testing.expectEqual(Stay.vram_learned, try stayOf(.{ .name = high, .rank = 3 }, 1, 2));
    try std.testing.expectEqual(Stay.vram_learned, try stayOf(.{ .name = high, .rank = 3 }, 0, 1));

    try std.testing.expectError(error.IllegalCut, stayOf(.{ .name = low, .rank = 3 }, 0, 4));
    try std.testing.expectError(error.RankOutOfRange, stayOf(.{ .name = low, .rank = 3 }, 2, 2));
    try std.testing.expectError(error.IllegalCut, stay(.routed_host, null, 0, 2));
    try std.testing.expect(std.mem.indexOf(u8, "key_proj", "k_proj") == null);
    try std.testing.expect(std.mem.indexOf(u8, "value_proj", "v_proj") == null);
    try std.testing.expect(std.mem.indexOf(u8, "mlp.shared_expert_gate", "mlp.gate") == null);
}

test "mlp.gate, key_proj, and value_proj stay pinned, and so does mlp.router.weight" {
    const gate = "model.language_model.layers.0.mlp.gate.weight";
    const key = "model.language_model.layers.1.ple.key_proj.weight";
    const value = "model.language_model.layers.1.ple.value_proj.weight";
    const router = "model.layers.0.mlp.router.weight";
    const shared_gate = "model.layers.0.mlp.shared_expert_gate.weight";
    for ([_][]const u8{ gate, key, value, router }) |name| {
        try std.testing.expectEqual(Home.resident, home(.{ .name = name, .rank = 2 }));
        try std.testing.expectEqual(Stay.vram_pinned, try stayOf(.{ .name = name, .rank = 2 }, 0, 2));
        try std.testing.expectEqual(Stay.vram_pinned, try stayOf(.{ .name = name, .rank = 2 }, 1, 2));
    }
    // `shared_expert` matches before the resident tokens. This is not the 2560×640 expert.
    try std.testing.expectEqual(Home.shared_resident, home(.{ .name = shared_gate, .rank = 2 }));
    try std.testing.expectEqual(Stay.vram_pinned, try stayOf(.{ .name = shared_gate, .rank = 1 }, 0, 2));

    const switch_gate = "model.layers.0.mlp.switch_mlp.gate_proj.weight";
    try std.testing.expectEqual(Home.plain, home(.{ .name = switch_gate, .rank = 2 }));
    try std.testing.expectEqual(Stay.host, try stayOf(.{ .name = switch_gate, .rank = 2 }, 0, 2));
    try std.testing.expectEqual(Home.routed_host, home(.{
        .name = "model.layers.0.mlp.experts.0.gate_proj.trellis",
        .rank = 3,
    }));
    try std.testing.expectEqual(Home.plain, home(.{ .name = "mtp.fc_embedding.weight", .rank = 2 }));
    try std.testing.expectEqual(Stay.host, try stayOf(.{ .name = "mtp.fc_embedding.weight", .rank = 2 }, 0, 2));
    try std.testing.expectEqual(Home.resident, home(.{ .name = "language_model.mtp.fc_hidden.weight", .rank = 2 }));
    try std.testing.expectEqual(Stay.vram_pinned, try stayOf(.{ .name = "language_model.mtp.fc_hidden.weight", .rank = 2 }, 0, 2));
}
