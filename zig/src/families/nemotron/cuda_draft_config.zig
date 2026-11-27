//! Nemotron CUDA DFlash2 drafter config: qwen3_5's parser plus the drafter checkpoints' declaration shape (A5, A3).
const std = @import("std");

pub const tap_window = 2048; // qwen3_5's drafter ring-row cap; Nemotron's drafters (window 1024) fit

/// The target's geometry the taps and widths are checked against.
pub const Target = struct { hidden: usize, vocab: usize, layers: usize };

pub const Kind = enum { dflash2, lilicorr };

pub const Config = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block: usize,
    mask: u32,
    /// Frame row-0 token: a config that DECLARES parallel_drafting_token_id wins; a silent config keeps pending.
    parallel: u32 = 0,
    parallel_declared: bool = false,
    window: usize,
    group: usize,
    rank: usize,
    topk: usize,
    eps: f32,
    theta: f32,
    taps: [8]u32,
    n_taps: usize,
    /// dflash2 ranks codebooks with a candidate_selector; lilicorr runs the NVIDIA correlator head (its own topk).
    kind: Kind = .dflash2,
    /// The draft attention's learned per-head SWA sinks (dflash_config.attention_sink_bias).
    sinks: bool = false,
    /// The block input embeddings' scale (top-level input_embedding_scale; the reference's default).
    embed_scale: f32 = 1.0,

    pub fn qWidth(c: Config) usize {
        return c.heads * c.head_dim;
    }
    pub fn kvWidth(c: Config) usize {
        return c.kv_heads * c.head_dim;
    }
    pub fn dynamic(c: Config) usize {
        return 4 * (c.hidden / c.group);
    }
    /// The ring's rows: the window and a block more (qwen3_5's layout).
    pub fn ringRows(c: Config) usize {
        return c.window + 16;
    }
    pub fn ringBytes(c: Config) usize {
        return c.layers * 2 * c.ringRows() * c.kvWidth() * 2;
    }
    /// Attention's 256-key chunks a row spans: its window and the block.
    pub fn chunks(c: Config) usize {
        return (c.ringRows() + 255) / 256;
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, target: Target) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        return parse(gpa, bytes, target);
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, target: Target) !Config {
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer doc.deinit();
        if (doc.value != .object) return error.BadDraftConfig;
        const o = doc.value.object;
        const d = (o.get("dflash_config") orelse return error.BadDraftConfig);
        if (d != .object) return error.BadDraftConfig;
        var c: Config = .{
            .hidden = try int(o, "hidden_size"),
            .intermediate = try int(o, "intermediate_size"),
            .layers = try int(o, "num_hidden_layers"),
            .vocab = try int(o, "vocab_size"),
            .heads = try int(o, "num_attention_heads"),
            .kv_heads = try int(o, "num_key_value_heads"),
            .head_dim = try int(o, "head_dim"),
            // the reference reads dflash_config's block_size, else the top-level one
            .block = try int(if (d.object.get("block_size")) |_| d.object else o, "block_size"),
            .mask = @intCast(try whole(d.object, "mask_token_id")),
            .parallel = @intCast(if (d.object.get("parallel_drafting_token_id")) |v| (if (v == .integer) v.integer else 0) else 0),
            .parallel_declared = d.object.get("parallel_drafting_token_id") != null,
            .window = try int(o, "sliding_window"),
            .group = try int(d.object, "conv_group_size"),
            .rank = 0,
            .topk = 0,
            .eps = @floatCast(try num(o, "rms_norm_eps")),
            .theta = @floatCast(try num(try plainRope(o), "rope_theta")),
            .taps = @splat(0),
            .n_taps = 0,
        };
        // dflash_config.causal: the vendored default decides the block's mask; only the

        // declared-false shape (the NVIDIA drafter's) is accepted, a true one refuses.
        if (d.object.get("causal")) |causal| {
            if (causal != .bool) return error.BadDraftConfig;
            if (causal.bool) return error.UnsupportedDraftConfig;
        }
        if (d.object.get("attention_sink_bias")) |sinks| {
            if (sinks != .bool) return error.BadDraftConfig;
            c.sinks = sinks.bool;
        }
        if (o.get("input_embedding_scale")) |scale| {
            if (scale != .float and scale != .integer) return error.BadDraftConfig;
            c.embed_scale = @floatCast(try num(o, "input_embedding_scale"));
        }
        // The selector: a DFlash2 config declares rank and top-k; a LiLiCorr config drops

        // both and enables the correlator head (its candidate_topk, no rank).
        const has_rank = d.object.get("selector_rank") != null;
        const has_topk = d.object.get("selector_top_k") != null;
        if (has_rank != has_topk) return error.BadDraftConfig;
        if (has_rank) {
            c.rank = try int(d.object, "selector_rank");
            c.topk = try int(d.object, "selector_top_k");
        } else {
            const enabled = d.object.get("lilicorr_enabled") orelse return error.BadDraftConfig;
            if (enabled != .bool or !enabled.bool) return error.UnsupportedDraftConfig;
            c.kind = .lilicorr;
            c.rank = 0;
            c.topk = try correlatorTopk(d.object, "lilicorr_candidate_topk");
        }
        try slidingLayers(o, c.layers);
        const taps = d.object.get("target_layer_ids") orelse return error.BadDraftConfig;
        if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > c.taps.len) return error.BadDraftConfig;
        for (taps.array.items, 0..) |t, i| {
            if (t != .integer or t.integer < 0 or t.integer >= target.layers) return error.BadDraftConfig;
            c.taps[i] = @intCast(t.integer);
            // one layer twice would tap one residual into two ring slots
            if (std.mem.indexOfScalar(u32, c.taps[0..i], c.taps[i]) != null) return error.BadDraftConfig;
        }
        c.n_taps = taps.array.items.len;
        // what the kernels fix: the head width, two-tap convolutions, a block of 8, the selector's top-k within a row
        if (c.head_dim != 128 or try int(d.object, "conv_kernel_size") != 2 or c.block != 8 or c.topk > lp_top or c.mask >= c.vocab) return error.UnsupportedDraftConfig;
        if (c.hidden != target.hidden or c.vocab != target.vocab or c.hidden % 256 != 0 or c.hidden % c.group != 0 or c.heads % c.kv_heads != 0 or c.layers > 8) return error.UnsupportedDraftConfig;
        _ = std.math.mul(usize, c.heads, c.head_dim) catch return error.UnsupportedDraftConfig;
        _ = std.math.mul(usize, c.kv_heads, c.head_dim) catch return error.UnsupportedDraftConfig;
        if (c.window < 16 or c.window > tap_window) return error.UnsupportedDraftConfig;
        // a DFlash2 drafter holds past the trained block (the reference's 16-row blocks); a

        // LiLiCorr one cannot — its correlator scores at most block - 1 slots (7), so the

        // trained block stays and the walk is a chain of 7.
        if (c.kind == .dflash2) c.block = 16;
        return c;
    }
};

const lp_top = 20;

/// The owner A5 row-0 rule: the frame's row-0 token is CONFIG-BACKED (declared wins, silent keeps pending).
pub fn row0Token(c: *const Config, pending: u32) u32 {
    return if (c.parallel_declared) c.parallel else pending;
}

/// The correlator's fixed geometry (A3): scores at most 8 candidate slots, zero keeps nothing; DraftTopkGeometry.
const correlator_topk_max = 8;

/// lilicorr candidate_topk: an integer in 1..8 - above eight or below one refuses as error.DraftTopkGeometry.
fn correlatorTopk(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer <= 0 or v.integer > correlator_topk_max) return error.DraftTopkGeometry;
    return @intCast(v.integer);
}

/// The draft's window: use_sliding_window, when present, must be true; layer_types must name every layer sliding.
fn slidingLayers(o: std.json.ObjectMap, layers: usize) !void {
    if (o.get("use_sliding_window")) |enabled| {
        if (enabled != .bool) return error.BadDraftConfig;
        if (!enabled.bool) return error.UnsupportedDraftConfig;
    }
    if (o.get("layer_types")) |types| {
        if (types != .array or types.array.items.len != layers) return error.BadDraftConfig;
        for (types.array.items) |kind| {
            if (kind != .string) return error.BadDraftConfig;
            if (!std.mem.eql(u8, kind.string, "sliding_attention")) return error.UnsupportedDraftConfig;
        }
    }
}

/// The draft's rope, refused unless unscaled: its kernels rope with theta alone (rope_scaling, if any, defaulted).
fn plainRope(o: std.json.ObjectMap) !std.json.ObjectMap {
    if (o.get("rope_parameters")) |rope| {
        if (rope != .object) return error.BadDraftConfig;
        for ([_][]const u8{ "rope_type", "type" }) |key| if (rope.object.get(key)) |v| {
            if (v != .string or !std.mem.eql(u8, v.string, "default")) return error.UnsupportedDraftConfig;
        };
        if (o.get("rope_scaling")) |v| if (v != .null) return error.UnsupportedDraftConfig;
        return rope.object;
    }
    if (o.get("rope_scaling")) |v| if (v != .null) return error.UnsupportedDraftConfig;
    return o;
}

/// Whether `dir`'s config.json declares a DFlash2 draft (a top-level dflash_config); no readable object: not one.
pub fn declared(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, "config.json" }) catch return false;
    defer gpa.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20)) catch return false;
    defer gpa.free(bytes);
    const doc = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch return false;
    defer doc.deinit();
    return doc.value == .object and doc.value.object.get("dflash_config") != null;
}

/// A size, count or divisor: positive, so a zero can't reach a remainder or an empty allocation.
fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try whole(o, key);
    if (v == 0) return error.BadDraftConfig;
    return v;
}

/// A token id: any whole number.
fn whole(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer < 0) return error.BadDraftConfig;
    return @intCast(v.integer);
}

fn num(o: std.json.ObjectMap, key: []const u8) !f64 {
    return switch (o.get(key) orelse return error.BadDraftConfig) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

const bolt_drafter_config =
    \\{
    \\ "architectures": ["DFlash2DraftModel"],
    \\ "attention_bias": false,
    \\ "attention_sink_bias": true,
    \\ "block_size": 8,
    \\ "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
    \\  "mask_token_id": 990, "selector_rank": 256, "selector_top_k": 16,
    \\  "target_layer_ids": [1, 18, 35, 51, 68, 85]},
    \\ "head_dim": 128,
    \\ "hidden_act": "silu",
    \\ "hidden_size": 4096,
    \\ "intermediate_size": 12288,
    \\ "model_type": "qwen3",
    \\ "num_attention_heads": 32,
    \\ "num_hidden_layers": 5,
    \\ "num_key_value_heads": 2,
    \\ "rms_norm_eps": 1e-06,
    \\ "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    \\ "sliding_window": 1024,
    \\ "use_sliding_window": true,
    \\ "vocab_size": 131072
    \\}
;

const bolt_target = Target{ .hidden = 4096, .vocab = 131072, .layers = 86 };

test "the Nemotron Bolt DFlash2 drafter's config parses to the kernel widths" {
    const a = std.testing.allocator;
    const c = try Config.parse(a, bolt_drafter_config, bolt_target);
    try std.testing.expectEqual(@as(usize, 4096), c.hidden);
    try std.testing.expectEqual(@as(usize, 5), c.layers);
    try std.testing.expectEqual(@as(usize, 1024), c.window);
    try std.testing.expectEqual(@as(usize, 256), c.rank);
    try std.testing.expectEqual(@as(usize, 16), c.topk);
    try std.testing.expectEqual(@as(usize, 6), c.n_taps);
    try std.testing.expectEqual(@as(u32, 85), c.taps[5]);
    try std.testing.expectEqual(@as(usize, 16), c.block); // past the trained 8, the hold block
    try std.testing.expectEqual(@as(usize, 1040), c.ringRows());
}

test "the same config without the sliding-window declarations still parses (Nemotron's shape)" {
    const a = std.testing.allocator;
    const stripped = try std.mem.replaceOwned(u8, a, bolt_drafter_config, ",\n \\ \"use_sliding_window\": true", "");
    defer a.free(stripped);
    _ = try Config.parse(a, stripped, bolt_target);
}

test "a wrong target width, a duplicated tap and a non-sliding declaration each refuse loudly" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.UnsupportedDraftConfig, Config.parse(a, bolt_drafter_config, .{ .hidden = 8192, .vocab = 131072, .layers = 86 }));
    const dup = try std.mem.replaceOwned(u8, a, bolt_drafter_config, "[1, 18, 35, 51, 68, 85]", "[1, 18, 35, 51, 68, 68]");
    defer a.free(dup);
    try std.testing.expectError(error.BadDraftConfig, Config.parse(a, dup, bolt_target));
    const full = try std.mem.replaceOwned(u8, a, bolt_drafter_config, "\"use_sliding_window\": true", "\"use_sliding_window\": false");
    defer a.free(full);
    try std.testing.expectError(error.UnsupportedDraftConfig, Config.parse(a, full, bolt_target));
}

test "a tap past the target's layers refuses, and so does an out-of-range window" {
    const a = std.testing.allocator;
    const deep = try std.mem.replaceOwned(u8, a, bolt_drafter_config, "[1, 18, 35, 51, 68, 85]", "[1, 18, 35, 51, 68, 96]");
    defer a.free(deep);
    try std.testing.expectError(error.BadDraftConfig, Config.parse(a, deep, bolt_target));
    const wide = try std.mem.replaceOwned(u8, a, bolt_drafter_config, "\"sliding_window\": 1024", "\"sliding_window\": 4096");
    defer a.free(wide);
    try std.testing.expectError(error.UnsupportedDraftConfig, Config.parse(a, wide, bolt_target));
}

test "a drafter directory is DFlash2's by its config's dflash_config; a target's own config or none is not" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    try std.testing.expect(!declared(a, io, root));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"NemotronH_Omni_Reasoning_V3\"}" });
    try std.testing.expect(!declared(a, io, root));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3\",\"dflash_config\":{\"block_size\":8}}" });
    try std.testing.expect(declared(a, io, root));
}

const lilicorr_config =
    \\{"architectures":["LiLiCorrDraftModel"],"model_type":"qwen3","block_size":8,
    \\ "dflash_config":{"mask_token_id":990,"target_layer_ids":[1,18,35,51,68,85],"use_swa":true,
    \\ "swa_window_size":1024,"causal":false,"attention_sink_bias":true,"projector_type":"lilicorr",
    \\ "lilicorr_enabled":true,"lilicorr_candidate_topk":8,"lilicorr_hidden_size":1024,
    \\ "lilicorr_num_layers":2,"lilicorr_num_heads":8,"lilicorr_mlp_ratio":2.0,
    \\ "lilicorr_factor_dim":1024,"lilicorr_vector_eps":0.0001,"lilicorr_logit_scale":8.0,
    \\ "conv_kernel_size":2,"conv_group_size":16},
    \\ "hidden_size":4096,"num_hidden_layers":5,"num_attention_heads":32,"num_key_value_heads":2,
    \\ "head_dim":128,"intermediate_size":12288,"hidden_act":"silu","rms_norm_eps":1e-06,
    \\ "vocab_size":131072,"max_position_embeddings":262144,"attention_bias":false,
    \\ "attention_dropout":0.0,"rope_theta":10000.0,"rope_scaling":null,"tie_word_embeddings":false,
    \\ "torch_dtype":"bfloat16","layer_types":["sliding_attention","sliding_attention",
    \\ "sliding_attention","sliding_attention","sliding_attention"],"sliding_window":1024,
    \\ "attention_sink_bias":true,"use_sliding_window":true,"max_window_layers":0,
    \\ "quantization_config":{"quant_algo":"W4A16_NVFP4","group_size":16,"quant_method":"modelopt_fp4"}}
;

test "the NVIDIA LiLiCorr drafter's config parses as the lilicorr kind with the correlator's top-k" {
    const a = std.testing.allocator;
    const c = try Config.parse(a, lilicorr_config, bolt_target);
    try std.testing.expectEqual(Kind.lilicorr, c.kind);
    try std.testing.expectEqual(@as(usize, 0), c.rank); // no selector rank: the correlator replaces it
    try std.testing.expectEqual(@as(usize, 8), c.topk); // lilicorr_candidate_topk
    try std.testing.expectEqual(@as(usize, 8), c.block); // the trained block stays: 7 correlator slots
    try std.testing.expectEqual(@as(usize, 6), c.n_taps);
    try std.testing.expectEqual(@as(u32, 1), c.taps[0]);
    try std.testing.expectEqual(@as(u32, 85), c.taps[5]);
    try std.testing.expect(c.sinks); // the draft attention's learned per-head SWA sinks
    try std.testing.expectEqual(@as(f32, 1.0), c.embed_scale); // the reference's default
    try std.testing.expectEqual(@as(usize, 1024), c.dynamic()); // the kernel projections' width
}

test "the LiLiCorr config's deviations refuse loudly: a causal block, the correlator disabled, a half selector" {
    const a = std.testing.allocator;
    const causal = try std.mem.replaceOwned(u8, a, lilicorr_config, "\"causal\":false", "\"causal\":true");
    defer a.free(causal);
    try std.testing.expectError(error.UnsupportedDraftConfig, Config.parse(a, causal, bolt_target));
    const disabled = try std.mem.replaceOwned(u8, a, lilicorr_config, "\"lilicorr_enabled\":true", "\"lilicorr_enabled\":false");
    defer a.free(disabled);
    try std.testing.expectError(error.UnsupportedDraftConfig, Config.parse(a, disabled, bolt_target));
    // neither a selector nor the correlator: no candidate source at all
    const neither = try std.mem.replaceOwned(u8, a, lilicorr_config, "\"lilicorr_enabled\":true,", "");
    defer a.free(neither);
    try std.testing.expectError(error.BadDraftConfig, Config.parse(a, neither, bolt_target));
    // a rank without a top-k (and the reverse) is a malformed selector
    const half = try std.mem.replaceOwned(u8, a, lilicorr_config, "\"conv_kernel_size\":2", "\"conv_kernel_size\":2,\"selector_rank\":256");
    defer a.free(half);
    try std.testing.expectError(error.BadDraftConfig, Config.parse(a, half, bolt_target));
}

test "the LiLiCorr config with a DFlash2 selector becomes the dflash2 kind and takes the hold block" {
    const a = std.testing.allocator;
    const both = try std.mem.replaceOwned(u8, a, lilicorr_config, "\"conv_kernel_size\":2", "\"conv_kernel_size\":2,\"selector_rank\":256,\"selector_top_k\":16");
    defer a.free(both);
    const c = try Config.parse(a, both, bolt_target);
    try std.testing.expectEqual(Kind.dflash2, c.kind);
    try std.testing.expectEqual(@as(usize, 16), c.block); // the extended hold block, chains and trees
    try std.testing.expectEqual(@as(f32, 3.5), blk: {
        const scaled = try std.mem.replaceOwned(u8, a, both, "\"rope_theta\":10000.0", "\"rope_theta\":10000.0,\"input_embedding_scale\":3.5");
        defer a.free(scaled);
        break :blk (try Config.parse(a, scaled, bolt_target)).embed_scale;
    });
}
