//! Public index admission proves required tensor names and safe shard basenames before payload headers are available.
const std = @import("std");
const Config = @import("config.zig").Config;

const Check = struct {
    map: std.json.ObjectMap,
    names: usize = 0,

    fn take(c: *Check, name: []const u8) !void {
        if (c.map.get(name) == null) return error.MissingFlashTensor;
        c.names += 1;
    }

    fn part(c: *Check, stem: []const u8, suffix: []const u8) !void {
        var buf: [256]u8 = undefined;
        try c.take(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, suffix }));
    }

    fn matrix(c: *Check, stem: []const u8) !void {
        try c.part(stem, "weight");
        var buf: [256]u8 = undefined;
        const scales = c.map.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{stem})) != null;
        const biases = c.map.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{stem})) != null;
        if (scales != biases) return error.IncompleteFlashAffine;
        if (scales) {
            try c.part(stem, "scales");
            try c.part(stem, "biases");
        }
    }

    fn proj(c: *Check, stem: []const u8, suffix: []const u8) !void {
        var buf: [256]u8 = undefined;
        try c.matrix(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, suffix }));
    }

    fn hc(c: *Check, stem: []const u8, inject: bool) !void {
        try c.part(stem, "hc_norm.weight");
        for ([_][]const u8{ "input_mix_weight_down", "input_mix_weight_up" }) |suffix| try c.proj(stem, suffix);
        if (inject) try c.proj(stem, "block_inject_weight");
    }

    fn layer(c: *Check, stem: []const u8, kind: @import("config.zig").Kind) !void {
        var buf: [256]u8 = undefined;
        for ([_][]const u8{ "attn_hyper_connection", "mlp_hyper_connection" }) |suffix| try c.hc(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, suffix }), true);
        for ([_][]const u8{ "mlp.gate", "mlp.shared_expert_gate", "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj", "mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj" }) |suffix| try c.proj(stem, suffix);
        if (kind == .linear_attention) {
            for ([_][]const u8{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.out_proj" }) |suffix| try c.proj(stem, suffix);
            for ([_][]const u8{ "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.conv1d.weight", "linear_attn.norm.weight" }) |suffix| try c.part(stem, suffix);
        } else {
            for ([_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.indexer.index_qk_proj" }) |suffix| try c.proj(stem, suffix);
            for ([_][]const u8{ "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.indexer.q_layernorm.weight", "self_attn.indexer.k_layernorm.weight" }) |suffix| try c.part(stem, suffix);
        }
    }

    fn ple(c: *Check, stem: []const u8, shards: usize, spelling: []const u8) !void {
        for ([_][]const u8{ "conv1d.weight", "norm_conv.weight", "norm_key.weight", "norm_query.weight", "ple_embedding.layer_multipliers", "ple_embedding.ngram_heads_offsets", "ple_embedding.ngram_heads_vocab_sizes" }) |suffix| try c.part(stem, suffix);
        try c.proj(stem, "key_proj");
        try c.proj(stem, "value_proj");
        var buf: [256]u8 = undefined;
        for (0..shards) |i| try c.proj(stem, try std.fmt.bufPrint(&buf, "ple_embedding.ngram_embedding.{s}{d}", .{ spelling, i }));
    }
};

/// The index's spelling of its n-gram table shards, `shard_N` when a `shard_0` is listed, else oMLX's `shards.N`.
pub fn ngramSpelling(map: std.json.ObjectMap) []const u8 {
    for (map.keys()) |name| if (std.mem.endsWith(u8, name, ".ngram_embedding.shard_0.weight")) return "shard_";
    return "shards.";
}

pub const Inventory = struct { tensor_names: usize, required_names: usize, shards: usize, headers_verified: bool = false };

pub fn admit(gpa: std.mem.Allocator, bytes: []const u8, config: *const Config) !Inventory {
    const p = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer p.deinit();
    if (p.value != .object) return error.InvalidFlashIndex;
    const map = p.value.object.get("weight_map") orelse return error.InvalidFlashIndex;
    if (map != .object or map.object.count() == 0) return error.InvalidFlashIndex;
    var files: std.StringHashMapUnmanaged(void) = .empty;
    defer files.deinit(gpa);
    var it = map.object.iterator();
    while (it.next()) |item| {
        const file = item.value_ptr.*;
        if (file != .string or file.string.len == 0 or file.string[0] == '.' or std.mem.indexOfAny(u8, file.string, "/\\") != null or !std.mem.endsWith(u8, file.string, ".safetensors")) return error.UnsafeFlashShard;
        try files.put(gpa, file.string, {});
    }
    var c = Check{ .map = map.object };
    const spelling = ngramSpelling(map.object); // every shard is checked under this one, so an index mixing the two is refused
    try c.matrix("language_model.model.embed_tokens");
    try c.matrix("language_model.lm_head");
    try c.hc("language_model.model.hyper_connection_mixer", false);
    var stem: [128]u8 = undefined;
    for (0..config.layers) |i| {
        const layer = try std.fmt.bufPrint(&stem, "language_model.model.layers.{d}", .{i});
        try c.layer(layer, try config.kind(i));
        if (config.ple[i]) {
            var buf: [160]u8 = undefined;
            try c.ple(try std.fmt.bufPrint(&buf, "{s}.ple", .{layer}), config.ngram_shards, spelling);
        }
    }
    if (config.mtp.layers > 0) {
        for ([_][]const u8{ "fc_embedding", "fc_hidden" }) |suffix| try c.proj("language_model.mtp", suffix);
        for ([_][]const u8{ "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight" }) |suffix| try c.part("language_model.mtp", suffix);
        try c.hc("language_model.mtp.hyper_connection_mixer", false);
        for (0..config.mtp.layers) |i| try c.layer(try std.fmt.bufPrint(&stem, "language_model.mtp.layers.{d}", .{i}), config.mtp.kinds[i]);
    }
    return .{ .tensor_names = map.object.count(), .required_names = c.names, .shards = files.count() };
}
