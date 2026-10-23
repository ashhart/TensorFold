//! Public index admission proves required tensor names and safe shard names before payload headers are available.
//! A shard may sit in a folder inside the model folder (`tensors/model-…safetensors`); names that leave it are refused.
const std = @import("std");
const Config = @import("config.zig").Config;

/// A weight map's shard, as core/checkpoint_index.zig's `shardName` takes it (kept here so this file imports no core):
/// plain names joined by `/`, no part hidden or reserved, and the file must be a safetensors one.
pub fn shardName(name: []const u8) bool {
    if (!std.mem.endsWith(u8, name, ".safetensors")) return false;
    var parts = std.mem.splitScalar(u8, name, '/');
    while (parts.next()) |part| if (!plainPart(part)) return false;
    return true;
}

fn plainPart(part: []const u8) bool {
    if (part.len == 0 or part[0] == '.' or part[part.len - 1] == '.' or part[part.len - 1] == ' ') return false;
    for (part) |c| if (c < 0x20 or c == 0x7f) return false;
    if (std.mem.indexOfAny(u8, part, "\\:<>\"|?*") != null) return false;
    const stem = part[0 .. std.mem.indexOfScalar(u8, part, '.') orelse part.len];
    for ([_][]const u8{ "CON", "PRN", "AUX", "NUL" }) |reserved|
        if (std.ascii.eqlIgnoreCase(stem, reserved)) return false;
    if (stem.len == 4 and stem[3] >= '1' and stem[3] <= '9' and
        (std.ascii.eqlIgnoreCase(stem[0..3], "COM") or std.ascii.eqlIgnoreCase(stem[0..3], "LPT"))) return false;
    return true;
}

/// A declared width, the fields of core/quant's mlx.Width, so this file imports no core.
pub const Width = struct { bits: u8, group: u16 };

/// The first required matrix the device kernels cannot read yet, by tensor stem, with its declared width.
pub const Refusal = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,
    width: ?Width = null,

    pub fn name(r: *const Refusal) []const u8 {
        return r.buf[0..r.len];
    }

    fn set(r: *Refusal, stem: []const u8, width: anytype) void {
        r.len = @min(stem.len, r.buf.len);
        @memcpy(r.buf[0..r.len], stem[0..r.len]);
        r.width = if (width) |w| .{ .bits = w.bits, .group = w.group } else null;
    }
};

/// The index walk; `Runtime` is widths.Widths when each matrix's width is checked, void when only names are.
fn Check(comptime Runtime: type) type {
    return struct {
        const C = @This();
        map: std.json.ObjectMap,
        names: usize = 0,
        runtime: ?*const Runtime = null,
        refusal: ?*Refusal = null,

        fn take(c: *C, name: []const u8) !void {
            if (c.map.get(name) == null) return error.MissingFlashTensor;
            c.names += 1;
        }

        fn part(c: *C, stem: []const u8, suffix: []const u8) !void {
            var buf: [256]u8 = undefined;
            try c.take(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, suffix }));
        }

        fn matrix(c: *C, stem: []const u8) !void {
            try c.part(stem, "weight");
            var buf: [256]u8 = undefined;
            const scales = c.map.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{stem})) != null;
            const biases = c.map.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{stem})) != null;
            if (scales != biases) return error.IncompleteFlashAffine;
            if (scales) {
                if (Runtime != void) if (c.runtime) |w| {
                    const width = try w.of(stem);
                    if (width == null or !w.reads(width.?)) {
                        if (c.refusal) |r| r.set(stem, width);
                        return error.UnsupportedFlashAffineKernel;
                    }
                };
                try c.part(stem, "scales");
                try c.part(stem, "biases");
            }
        }

        fn proj(c: *C, stem: []const u8, suffix: []const u8) !void {
            var buf: [256]u8 = undefined;
            try c.matrix(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, suffix }));
        }

        fn hc(c: *C, stem: []const u8, inject: bool) !void {
            try c.part(stem, "hc_norm.weight");
            for ([_][]const u8{ "input_mix_weight_down", "input_mix_weight_up" }) |suffix| try c.proj(stem, suffix);
            if (inject) try c.proj(stem, "block_inject_weight");
        }

        fn layer(c: *C, stem: []const u8, kind: @import("config.zig").Kind) !void {
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

        /// An n-gram table shard: a lookup reads a row of `row` values a group at a time, so its group divides the row.
        fn table(c: *C, stem: []const u8, row: usize) !void {
            try c.matrix(stem);
            if (Runtime == void) return;
            const w = c.runtime orelse return;
            const declared = try w.of(stem);
            const width = declared orelse return;
            if (row % width.group == 0) return;
            if (c.refusal) |r| r.set(stem, declared);
            return error.UnsupportedFlashAffineKernel;
        }

        fn ple(c: *C, stem: []const u8, shards: usize, spelling: []const u8, row: usize) !void {
            for ([_][]const u8{ "conv1d.weight", "norm_conv.weight", "norm_key.weight", "norm_query.weight", "ple_embedding.layer_multipliers", "ple_embedding.ngram_heads_offsets", "ple_embedding.ngram_heads_vocab_sizes" }) |suffix| try c.part(stem, suffix);
            try c.proj(stem, "key_proj");
            try c.proj(stem, "value_proj");
            var buf: [256]u8 = undefined;
            var name: [256]u8 = undefined;
            for (0..shards) |i| try c.table(try std.fmt.bufPrint(&name, "{s}.{s}", .{ stem, try std.fmt.bufPrint(&buf, "ple_embedding.ngram_embedding.{s}{d}", .{ spelling, i }) }), row);
        }
    };
}

/// Where the index keeps the MTP head: `language_model.mtp` (mlx-lm), else oMLX's top-level `mtp`.
pub fn mtpRoot(map: std.json.ObjectMap) []const u8 {
    for (map.keys()) |name| if (std.mem.startsWith(u8, name, "language_model.mtp.")) return "language_model.mtp";
    for (map.keys()) |name| if (std.mem.startsWith(u8, name, "mtp.")) return "mtp";
    return "language_model.mtp";
}

/// The index's spelling of its n-gram table shards, `shard_N` when a `shard_0` is listed, else oMLX's `shards.N`.
pub fn ngramSpelling(map: std.json.ObjectMap) []const u8 {
    for (map.keys()) |name| if (std.mem.endsWith(u8, name, ".ngram_embedding.shard_0.weight")) return "shard_";
    return "shards.";
}

pub const Inventory = struct { tensor_names: usize, required_names: usize, shards: usize, headers_verified: bool = false };

pub fn admit(gpa: std.mem.Allocator, bytes: []const u8, config: *const Config) !Inventory {
    return admitWith(void, gpa, bytes, config, null, null);
}

/// Each required matrix at its own module width, refused by name when no device kernel reads that width yet.
pub fn admitRuntime(gpa: std.mem.Allocator, bytes: []const u8, config: *const Config, w: anytype, refusal: *Refusal) !Inventory {
    return admitWith(@TypeOf(w.*), gpa, bytes, config, w, refusal);
}

fn admitWith(comptime Runtime: type, gpa: std.mem.Allocator, bytes: []const u8, config: *const Config, runtime: ?*const Runtime, refusal: ?*Refusal) !Inventory {
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
        if (file != .string or !shardName(file.string)) return error.UnsafeFlashShard;
        try files.put(gpa, file.string, {});
    }
    var c = Check(Runtime){ .map = map.object, .runtime = runtime, .refusal = refusal };
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
            try c.ple(try std.fmt.bufPrint(&buf, "{s}.ple", .{layer}), config.ngram_shards, spelling, config.hidden / try config.ngramHeadCount());
        }
    }
    if (config.mtp.layers > 0) {
        const root = mtpRoot(map.object);
        for ([_][]const u8{ "fc_embedding", "fc_hidden" }) |suffix| try c.proj(root, suffix);
        for ([_][]const u8{ "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight" }) |suffix| try c.part(root, suffix);
        var mixer: [64]u8 = undefined;
        try c.hc(try std.fmt.bufPrint(&mixer, "{s}.hyper_connection_mixer", .{root}), false);
        for (0..config.mtp.layers) |i| try c.layer(try std.fmt.bufPrint(&stem, "{s}.layers.{d}", .{ root, i }), config.mtp.kinds[i]);
    }
    return .{ .tensor_names = map.object.count(), .required_names = c.names, .shards = files.count() };
}

test "admission takes NVIDIA's tensors/ shard folder and keeps the escape refusal" {
    const a = std.testing.allocator;
    var c = try @import("config.zig").parse(a, @embedFile("fixtures/config.json"));
    defer c.deinit();
    const bytes = @embedFile("fixtures/index.json");
    const flat = try admit(a, bytes, &c);
    try std.testing.expect(flat.shards > 1);
    // The publisher's own export maps every tensor to `tensors/model-000NN-of-00021.safetensors`.
    const nested = try std.mem.replaceOwned(u8, a, bytes, "\":\"model-", "\":\"tensors/model-");
    defer a.free(nested);
    try std.testing.expectEqual(flat, try admit(a, nested, &c));
    const escaped = try std.mem.replaceOwned(u8, a, bytes, "\":\"model-", "\":\"../model-");
    defer a.free(escaped);
    try std.testing.expectError(error.UnsafeFlashShard, admit(a, escaped, &c));
}

test "a shard may sit in a folder inside the model folder; escapes keep their refusal" {
    // NVIDIA's own export writes `tensors/model-000NN-of-00021.safetensors`.
    for ([_][]const u8{
        "model-00001-of-00021.safetensors",
        "tensors/model-00001-of-00021.safetensors",
        "a/b/c.safetensors",
    }) |name| try std.testing.expect(shardName(name));
    for ([_][]const u8{
        "", // nothing to open
        "model.bin", // not a safetensors file
        "/model.safetensors", // absolute
        "../model.safetensors", // leaves the folder
        "tensors/../model.safetensors", // leaves it through a subfolder
        "tensors//model.safetensors", // an empty part
        ".cache/model.safetensors", // hidden folder
        "tensors/.hidden.safetensors", // hidden file
        "C:/model.safetensors", // a drive name
        "tensors\\model.safetensors", // a Windows separator
        "tensors/nul.safetensors", // a reserved device stem
        "tensors/model.safetensors ", // trailing space
    }) |name| try std.testing.expect(!shardName(name));
}
