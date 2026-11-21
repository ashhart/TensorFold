//! ZONOS3 Mini's geometry (params.json "model"): load() refuses a checkpoint unlike the one the kernels are built for.
const std = @import("std");

pub const dim = 1024;
pub const layers = 11;
pub const heads = 16;
pub const kv_heads = 4;
pub const head_dim = 32;
pub const rope_theta: f32 = 10000.0;
pub const eps: f32 = 1.0e-5;
pub const softcap: f32 = 15.0;
pub const max_seqlen = 32768;
// every gdn_period-th layer is a global attention layer, the rest are Gated DeltaNet
pub const gdn_period = 4;
pub const gdn_heads = 4;
pub const gdn_dim = 128;
pub const gdn_inner = gdn_heads * gdn_dim; // 512
pub const gdn_conv = 3 * gdn_inner; // q | k | v
pub const gdn_kernel = 4;
pub const gdn_proj = 4 * gdn_inner + 2 * gdn_heads; // qkv | z | a | b = 2056
pub const attn_inner = heads * head_dim; // 512
pub const kv_width = kv_heads * head_dim; // 128
// feed-forward: layer 0 dense, the rest a latent mixture of experts
pub const ffn_hidden = 2048;
pub const experts = 64;
pub const top_k = 2;
pub const moe_hidden = 1536;
pub const moe_latent = 256;
// frames: [code 0 .. code 9, text]
pub const codebooks = 10;
pub const codebook_size = 4096;
pub const eoa_id = 4096;
pub const audio_pad_id = 4097;
pub const audio_vocab = codebook_size + 2;
pub const span_end_id = audio_pad_id + 1; // an edit fill's end, a row the speech-edit adapter adds
pub const frame_width = codebooks + 1;
pub const text_vocab = 8196; // SuperBPE 8192 + 4; also the text column's padding row
pub const head_rows = codebooks * audio_vocab; // 40980
pub const sources = 2 * layers + 1; // Full AttnRes bank: the embedding and two a layer
pub const speaker_dim = 192;
pub const conditions = 3; // loudness, fmax, speaking rate
pub const max_buckets = 30;
/// The condition rows in prompt order, as params.json's condition_names and a request name them.
pub const condition_names = [conditions][]const u8{ "loudness", "fmax", "speaking_rate" };

pub fn usesGdn(layer: usize) bool {
    return (layer + 1) % gdn_period != 0;
}

pub const Config = struct {
    /// Buckets of each condition table; id == count is the table's unknown row.
    buckets: [conditions]u32,
    /// Each table's bucket edges, rising: bucket i holds [edge i-1, edge i), one edge fewer than buckets.
    edges: [conditions][max_buckets - 1]f64,

    /// plan.resolve_condition_buckets: a raw value's bucket, the training loader's bisect_right over the edges.
    pub fn bucketOf(c: *const Config, condition: usize, value: f64) u32 {
        var n: u32 = 0;
        for (c.edges[condition][0 .. c.buckets[condition] - 1]) |edge| n += @intFromBool(edge <= value);
        return n;
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "params.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        return parse(gpa, text);
    }

    pub fn parse(gpa: std.mem.Allocator, text: []const u8) !Config {
        var tree = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer tree.deinit();
        const root = try object(tree.value);
        const m = if (root.get("model")) |v| try object(v) else root;
        const ints = .{
            .{ "n_layers", layers },                   .{ "dim", dim },                     .{ "head_dim", head_dim },
            .{ "n_heads", heads },                     .{ "n_kv_heads", kv_heads },         .{ "max_seqlen", max_seqlen },
            .{ "n_codebooks", codebooks },             .{ "codebook_size", codebook_size }, .{ "eoa_id", eoa_id },
            .{ "audio_pad_id", audio_pad_id },         .{ "text_vocab", text_vocab },       .{ "moe_n_experts", experts },
            .{ "moe_router_topk", top_k },             .{ "moe_start_from_layer", 1 },      .{ "moe_end_from_layer", 0 },
            .{ "speaker_embedding_dim", speaker_dim }, .{ "condition_count", conditions },
        };
        inline for (ints) |e| if (try int(m, e[0]) != e[1]) return refuse(e[0]);
        const strings = .{
            .{ "attention_pattern", "gdn3_global1" }, .{ "residual_type", "full_attnres" },
            .{ "moe_router_type", "linear" },         .{ "moe_balancing_strategy", "quantile" },
            .{ "dtype", "bfloat16" },
        };
        inline for (strings) |e| {
            const v = m.get(e[0]) orelse return refuse(e[0]);
            if (v != .string or !std.mem.eql(u8, v.string, e[1])) return refuse(e[0]);
        }
        const floats = .{
            .{ "norm_eps", eps },           .{ "rope_theta", rope_theta },      .{ "loss_softcap", softcap },
            .{ "ffn_dim_multiplier", 2.0 }, .{ "moe_ffn_dim_multiplier", 1.5 }, .{ "moe_alpha", 0.25 },
        };
        inline for (floats) |e| if (@as(f32, @floatCast(try float(m, e[0]))) != @as(f32, e[1])) return refuse(e[0]);
        const off = .{ "use_xsa", "moe_router_use_eda" };
        inline for (off) |name| {
            const v = m.get(name) orelse return refuse(name);
            if (v != .bool or v.bool) return refuse(name);
        }
        var c: Config = undefined;
        const counts = m.get("condition_bucket_counts") orelse return refuse("condition_bucket_counts");
        if (counts != .array or counts.array.items.len != conditions) return refuse("condition_bucket_counts");
        for (counts.array.items, 0..) |v, i| {
            if (v != .integer or v.integer < 1 or v.integer > max_buckets) return refuse("condition_bucket_counts");
            c.buckets[i] = @intCast(v.integer);
        }
        const names = m.get("condition_names") orelse return refuse("condition_names");
        if (names != .array or names.array.items.len != conditions) return refuse("condition_names");
        const all = m.get("condition_edges") orelse return refuse("condition_edges");
        if (all != .object) return refuse("condition_edges");
        for (names.array.items, condition_names, 0..) |v, name, i| {
            if (v != .string or !std.mem.eql(u8, v.string, name)) return refuse("condition_names");
            const edges = all.object.get(name) orelse return refuse("condition_edges");
            if (edges != .array or edges.array.items.len != c.buckets[i] - 1) return refuse("condition_edges");
            c.edges[i] = @splat(std.math.inf(f64));
            for (edges.array.items, 0..) |e, j| {
                c.edges[i][j] = switch (e) {
                    .float => |f| f,
                    .integer => |n| @floatFromInt(n),
                    else => return refuse("condition_edges"),
                };
                if (!std.math.isFinite(c.edges[i][j]) or (j > 0 and c.edges[i][j] <= c.edges[i][j - 1])) return refuse("condition_edges");
            }
        }
        return c;
    }

    fn refuse(name: []const u8) error{UnsupportedCheckpoint} {
        std.log.err("zonos3: params.json {s} is not ZONOS3 Mini's", .{name});
        return error.UnsupportedCheckpoint;
    }

    fn object(v: std.json.Value) !std.json.ObjectMap {
        return if (v == .object) v.object else error.BadConfig;
    }

    fn int(m: std.json.ObjectMap, name: []const u8) !i64 {
        const v = m.get(name) orelse return refuse(name);
        return if (v == .integer) v.integer else refuse(name);
    }

    fn float(m: std.json.ObjectMap, name: []const u8) !f64 {
        const v = m.get(name) orelse return refuse(name);
        return switch (v) {
            .float => |f| f,
            .integer => |n| @floatFromInt(n),
            else => refuse(name),
        };
    }
};

test "layer kinds: one global attention layer in four" {
    var gdn: usize = 0;
    for (0..layers) |l| gdn += @intFromBool(usesGdn(l));
    try std.testing.expectEqual(@as(usize, 9), gdn);
    try std.testing.expect(!usesGdn(3) and !usesGdn(7) and usesGdn(10));
}

test "conditions: a raw value's bucket is the count of edges at or below it" {
    const t = std.testing;
    var c: Config = .{ .buckets = .{ 10, 10, 30 }, .edges = @splat(@splat(std.math.inf(f64))) };
    const loudness = [_]f64{ -29.778, -26.947, -25.047, -23.3, -21.743, -20.386, -19.186, -17.851, -15.967 };
    @memcpy(c.edges[0][0..loudness.len], &loudness);
    for (c.edges[2][0..29], 0..) |*e, i| e.* = @floatFromInt(i + 1);
    // the reference resolves loudness -20 to bucket 6
    try t.expectEqual(@as(u32, 6), c.bucketOf(0, -20.0));
    try t.expectEqual(@as(u32, 0), c.bucketOf(0, -40.0));
    try t.expectEqual(@as(u32, 9), c.bucketOf(0, 0.0));
    // bisect_right: a value on an edge belongs to the bucket above it
    try t.expectEqual(@as(u32, 4), c.bucketOf(0, -23.3));
    try t.expectEqual(@as(u32, 29), c.bucketOf(2, 1000.0));
    try t.expectEqual(@as(u32, 3), c.bucketOf(2, 3.5));
}
