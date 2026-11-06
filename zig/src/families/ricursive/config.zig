//! Ricursive (model_type ricursive, Ripple-65B) admission: sliding and global attention, gated, with value residuals.
const std = @import("std");

pub const head_dim = 128;
pub const max_layers = 128;
pub const max_eos = 4;
/// Query heads a key head at most: a decode attention threadgroup holds a simdgroup each.
pub const max_group = 32;
/// The longest context taken: the longest checked (a needle at 258,039 tokens); longer waits for a qualifying run.
pub const max_context = 262144;

pub const Error = error{
    BadRicursiveConfig,
    BadRicursiveEos,
    UnsupportedRicursiveGeometry,
    UnsupportedRicursiveRotary,
    UnsupportedRicursivePrecision,
    UnqualifiedRicursiveCheckpoint,
} || std.json.ParseError(std.json.Scanner);

/// A checkpoint checked on the device, by the geometry its config names.
pub const Qualified = struct {
    name: []const u8,
    hidden: usize,
    intermediate: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    window: usize,
    theta: f64,
};

/// The checkpoints the family serves; another `ricursive` checkpoint is refused until it has been checked the same way.
pub const qualified = [_]Qualified{
    .{ .name = "Ripple-65B-Preview", .hidden = 5120, .intermediate = 40960, .layers = 80, .heads = 80, .kv_heads = 8, .vocab = 100352, .window = 1024, .theta = 10000 },
};

/// Attention's score scale, 1 / sqrt(head_dim).
pub fn headScale() f32 {
    return 1.0 / @sqrt(@as(f32, head_dim));
}

pub const Config = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    eps: f32,
    theta: f64,
    /// Keys a sliding layer's row sees: itself and the window - 1 before it.
    window: usize,
    /// The embeddings' scale, as the model multiplies its bf16 rows (fp32 opmath).
    embed_scale: f32,
    /// The FFN's clamp: gate at most this, up within plus or minus it.
    clamp: f32,
    context: usize,
    /// The MLP's projections are the publisher's NVFP4 (ModelOpt: E2M1 in blocks of 16, E4M3 scales, an fp32 global).
    fp4_mlp: bool = false,
    /// A global layer: full attention without RoPE; the others slide with RoPE.
    global: [max_layers]bool = @splat(false),
    eos: [max_eos]u32 = @splat(0),
    eos_count: usize = 0,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        const c = try parse(gpa, bytes);
        _ = try c.qualification();
        return c;
    }

    /// The qualified checkpoint this config names.
    pub fn qualification(c: Config) Error!*const Qualified {
        for (&qualified) |*q| {
            if (c.hidden == q.hidden and c.intermediate == q.intermediate and c.layers == q.layers and c.heads == q.heads and
                c.kv_heads == q.kv_heads and c.vocab == q.vocab and c.window == q.window and c.theta == q.theta and
                c.fifthGlobal()) return q;
        }
        return error.UnqualifiedRicursiveCheckpoint;
    }

    /// The release's layer kinds: every fifth layer global, the others sliding.
    fn fifthGlobal(c: Config) bool {
        for (c.global[0..c.layers], 0..) |g, i| if (g != (i % 5 == 4)) return false;
        return true;
    }

    pub fn queryWidth(c: Config) usize {
        return c.heads * head_dim;
    }

    pub fn kvWidth(c: Config) usize {
        return c.kv_heads * head_dim;
    }

    /// The q, k and v projections' outputs side by side in a row.
    pub fn qkvWidth(c: Config) usize {
        return c.queryWidth() + 2 * c.kvWidth();
    }

    /// Query heads a key head.
    pub fn group(c: Config) usize {
        return c.heads / c.kv_heads;
    }

    /// 1 / sqrt(head_dim), the attention scale.
    pub fn scale(_: Config) f32 {
        return headScale();
    }

    /// Global layers: every position's keys kept.
    pub fn globals(c: Config) usize {
        var n: usize = 0;
        for (c.global[0..c.layers]) |g| n += @intFromBool(g);
        return n;
    }

    pub fn isEos(c: Config, token: u32) bool {
        for (c.eos[0..c.eos_count]) |e| if (e == token) return true;
        return false;
    }

    /// RoPE inverse frequencies [head_dim / 2] as the release makes them: 1 / theta^(2i / head_dim), both steps fp32.
    pub fn inverseFrequencies(c: Config, out: []f32) void {
        for (out, 0..) |*f, i| {
            const e: f64 = @as(f64, @floatFromInt(2 * i)) / @as(f64, head_dim);
            const p: f32 = @floatCast(std.math.pow(f64, c.theta, e)); // powf, correctly rounded
            f.* = 1.0 / p;
        }
    }
};

fn object(v: std.json.Value) Error!std.json.ObjectMap {
    return if (v == .object) v.object else error.BadRicursiveConfig;
}

fn number(v: std.json.Value) Error!f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadRicursiveConfig,
    };
}

fn integer(o: std.json.ObjectMap, key: []const u8) Error!i64 {
    const v = o.get(key) orelse return error.BadRicursiveConfig;
    return if (v == .integer) v.integer else error.BadRicursiveConfig;
}

fn isString(v: ?std.json.Value, expected: []const u8) bool {
    const s = v orelse return false;
    return s == .string and std.mem.eql(u8, s.string, expected);
}

/// `key` is the JSON boolean `want`.
fn isBool(o: std.json.ObjectMap, key: []const u8, want: bool) bool {
    const v = o.get(key) orelse return false;
    return v == .bool and v.bool == want;
}

/// A positive dimension.
fn size(o: std.json.ObjectMap, key: []const u8) Error!usize {
    const v = try integer(o, key);
    return if (v > 0) @intCast(v) else error.UnsupportedRicursiveGeometry;
}

fn eosIds(v: std.json.Value, c: *Config) Error!void {
    switch (v) {
        .integer => |i| {
            if (i < 0 or i >= c.vocab or c.eos_count == c.eos.len) return error.BadRicursiveEos;
            c.eos[c.eos_count] = @intCast(i);
            c.eos_count += 1;
        },
        .array => |a| for (a.items) |x| try eosIds(x, c),
        else => return error.BadRicursiveEos,
    }
}

/// layer_types and hybrid_layer_pattern (1 sliding, 0 global) name the same layers, one a layer.
fn layerKinds(root: std.json.ObjectMap, c: *Config) Error!void {
    const types = root.get("layer_types") orelse return error.BadRicursiveConfig;
    const pattern = root.get("hybrid_layer_pattern") orelse return error.BadRicursiveConfig;
    if (types != .array or pattern != .array) return error.BadRicursiveConfig;
    if (types.array.items.len != c.layers or pattern.array.items.len != c.layers) return error.UnsupportedRicursiveGeometry;
    for (types.array.items, pattern.array.items, 0..) |t, p, i| {
        const global = isString(t, "full_attention");
        if (!global and !isString(t, "sliding_attention")) return error.UnsupportedRicursiveGeometry;
        if (p != .integer or p.integer != @intFromBool(!global)) return error.UnsupportedRicursiveGeometry;
        c.global[i] = global;
    }
}

/// ModelOpt NVFP4, groups of 16, no cache scheme, ignoring exactly the embeddings, head and every layer's attention.
fn nvfp4Mlp(q: std.json.Value, layers: usize) Error!bool {
    const o = try object(q);
    if (!isString(o.get("quant_method"), "modelopt") or !isString(o.get("quant_algo"), "NVFP4")) return error.UnsupportedRicursivePrecision;
    for ([_][]const u8{ "kv_cache_quant_algo", "kv_cache_scheme" }) |key| if (o.get(key)) |v| if (v != .null) return error.UnsupportedRicursivePrecision;
    const groups = try object(o.get("config_groups") orelse return error.UnsupportedRicursivePrecision);
    if (groups.count() != 1) return error.UnsupportedRicursivePrecision;
    const g = try object(groups.values()[0]);
    const w = try object(g.get("weights") orelse return error.UnsupportedRicursivePrecision);
    if (!isString(w.get("type"), "float") or (number(w.get("num_bits") orelse return error.UnsupportedRicursivePrecision) catch 0) != 4 or
        (number(w.get("group_size") orelse return error.UnsupportedRicursivePrecision) catch 0) != 16) return error.UnsupportedRicursivePrecision;
    const ignore = switch (o.get("ignore") orelse return error.UnsupportedRicursivePrecision) {
        .array => |a| a.items,
        else => return error.UnsupportedRicursivePrecision,
    };
    var attention: [max_layers]bool = @splat(false);
    for (ignore) |item| {
        const name = switch (item) {
            .string => |t| t,
            else => return error.UnsupportedRicursivePrecision,
        };
        if (std.mem.eql(u8, name, "lm_head") or std.mem.eql(u8, name, "model.embed_tokens")) continue;
        const prefix = "model.layers.";
        const suffix = ".self_attn*";
        if (!std.mem.startsWith(u8, name, prefix) or !std.mem.endsWith(u8, name, suffix) or name.len <= prefix.len + suffix.len) return error.UnsupportedRicursivePrecision;
        const l = std.fmt.parseInt(usize, name[prefix.len .. name.len - suffix.len], 10) catch return error.UnsupportedRicursivePrecision;
        if (l >= layers or l >= max_layers) return error.UnsupportedRicursivePrecision;
        attention[l] = true;
    }
    // every layer's attention stays bf16, as the kernels read it
    for (attention[0..@min(layers, max_layers)]) |kept| if (!kept) return error.UnsupportedRicursivePrecision;
    return true;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) Error!Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    if (!isString(root.get("model_type"), "ricursive")) return error.BadRicursiveConfig;
    // the release's layer: gated attention (an fp32 gate), value residuals, sliding and global layers, an untied head
    for ([_][]const u8{ "use_gated_attention", "gate_full_precision", "use_valres", "is_hybrid_swa" }) |key| {
        if (!isBool(root, key, true)) return error.UnsupportedRicursiveGeometry;
    }
    if (!isBool(root, "tie_word_embeddings", false)) return error.UnsupportedRicursiveGeometry;
    if (!isString(root.get("hidden_act"), "silu")) return error.UnsupportedRicursiveGeometry;
    for ([_][]const u8{ "torch_dtype", "dtype" }) |key| if (root.get(key)) |d| if (!isString(d, "bfloat16")) return error.UnsupportedRicursivePrecision;
    if (root.get("quantization") != null) return error.UnsupportedRicursivePrecision;
    if (root.get("rope_scaling")) |r| if (r != .null) return error.UnsupportedRicursiveRotary;
    var c: Config = .{
        .hidden = try size(root, "hidden_size"),
        .intermediate = try size(root, "intermediate_size"),
        .layers = try size(root, "num_hidden_layers"),
        .heads = try size(root, "num_attention_heads"),
        .kv_heads = try size(root, "num_key_value_heads"),
        .vocab = try size(root, "vocab_size"),
        .eps = @floatCast(try number(root.get("rms_norm_eps") orelse return error.BadRicursiveConfig)),
        .theta = try number(root.get("rope_theta") orelse return error.BadRicursiveConfig),
        .window = try size(root, "sliding_window"),
        .embed_scale = @floatCast(try number(root.get("embed_scale") orelse return error.BadRicursiveConfig)),
        .clamp = @floatCast(try number(root.get("ffn_clamp_threshold") orelse return error.BadRicursiveConfig)),
        .context = 0,
    };
    if (root.get("quantization_config")) |q| c.fp4_mlp = try nvfp4Mlp(q, c.layers);
    if (try size(root, "head_dim") != head_dim) return error.UnsupportedRicursiveGeometry;
    if (c.layers > max_layers or c.heads % c.kv_heads != 0 or c.group() > max_group) return error.UnsupportedRicursiveGeometry;
    // the tensor-unit GEMM steps k by 256s; the row kernels read eight values a lane over 256
    if (c.hidden % 256 != 0 or c.intermediate % 256 != 0 or c.queryWidth() % 256 != 0) return error.UnsupportedRicursiveGeometry;
    if (!(c.eps > 0) or !(c.theta > 0) or !(c.embed_scale > 0) or !(c.clamp > 0)) return error.BadRicursiveConfig;
    try layerKinds(root, &c);
    c.context = @min(try size(root, "max_position_embeddings"), max_context);
    try eosIds(root.get("eos_token_id") orelse return error.BadRicursiveEos, &c);
    if (c.eos_count == 0) return error.BadRicursiveEos;
    return c;
}

const fixture = @embedFile("fixtures/config.json");

test "Ripple-65B-Preview's config is admitted with its geometry, layer kinds and end tokens" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqualStrings("Ripple-65B-Preview", (try c.qualification()).name);
    try std.testing.expectEqual(@as(usize, 5120), c.hidden);
    try std.testing.expectEqual(@as(usize, 40960), c.intermediate);
    try std.testing.expectEqual(@as(usize, 80), c.layers);
    try std.testing.expectEqual(@as(usize, 10240), c.queryWidth());
    try std.testing.expectEqual(@as(usize, 12288), c.qkvWidth());
    try std.testing.expectEqual(@as(usize, 10), c.group());
    try std.testing.expectEqual(@as(usize, 1024), c.window);
    try std.testing.expectEqual(@as(usize, 16), c.globals());
    for (0..c.layers) |i| try std.testing.expectEqual(i % 5 == 4, c.global[i]);
    try std.testing.expectEqual(@as(f32, 20), c.clamp);
    try std.testing.expectEqual(@as(f32, @floatCast(71.55417527999327)), c.embed_scale);
    try std.testing.expectEqual(@as(usize, max_context), c.context);
    try std.testing.expectEqualSlices(u32, &.{ 100265, 100257 }, c.eos[0..c.eos_count]);
}

test "ricursive config: another geometry or layer is refused" {
    const gpa = std.testing.allocator;
    const refused = .{
        .{ "\"use_valres\": true", "\"use_valres\": false", error.UnsupportedRicursiveGeometry },
        .{ "\"gate_full_precision\": true", "\"gate_full_precision\": false", error.UnsupportedRicursiveGeometry },
        .{ "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true", error.UnsupportedRicursiveGeometry },
        .{ "\"head_dim\": 128", "\"head_dim\": 64", error.UnsupportedRicursiveGeometry },
        .{ "\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"", error.UnsupportedRicursiveGeometry },
        .{ "\"model_type\": \"ricursive\"", "\"model_type\": \"llama\"", error.BadRicursiveConfig },
        .{ "\"rope_theta\": 10000.0", "\"rope_theta\": 10000.0, \"rope_scaling\": {\"rope_type\": \"yarn\"}", error.UnsupportedRicursiveRotary },
        .{ "\"layer_types\": [\n    \"sliding_attention\"", "\"layer_types\": [\n    \"full_attention\"", error.UnsupportedRicursiveGeometry },
    };
    inline for (refused) |k| {
        const bytes = try std.mem.replaceOwned(u8, gpa, fixture, k[0], k[1]);
        defer gpa.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, fixture));
        try std.testing.expectError(k[2], parse(gpa, bytes));
    }
    const other = try std.mem.replaceOwned(u8, gpa, fixture, "\"sliding_window\": 1024", "\"sliding_window\": 4096");
    defer gpa.free(other);
    try std.testing.expectError(error.UnqualifiedRicursiveCheckpoint, (try parse(gpa, other)).qualification());
}

/// The fixture with an NVFP4 quantization_config: `extra` in it, every layer's attention ignored but `skip`'s.
fn withNvfp4(gpa: std.mem.Allocator, extra: []const u8, skip: ?usize) ![]u8 {
    var ignore: std.ArrayList(u8) = .empty;
    defer ignore.deinit(gpa);
    try ignore.appendSlice(gpa, "\"lm_head\", \"model.embed_tokens\"");
    for (0..80) |l| if (l != skip) try ignore.print(gpa, ", \"model.layers.{d}.self_attn*\"", .{l});
    const q = try std.fmt.allocPrint(gpa, "\"model_type\": \"ricursive\", \"quantization_config\": {{{s}\"quant_method\": \"modelopt\", \"quant_algo\": \"NVFP4\", \"config_groups\": {{\"group_0\": {{\"weights\": {{\"type\": \"float\", \"num_bits\": 4, \"group_size\": 16}}}}}}, \"ignore\": [{s}]}}", .{ extra, ignore.items });
    defer gpa.free(q);
    return std.mem.replaceOwned(u8, gpa, fixture, "\"model_type\": \"ricursive\"", q);
}

test "ricursive config: the NVFP4 release's scheme is admitted, a cache scheme or a quantized attention refused" {
    const gpa = std.testing.allocator;
    const ok = try withNvfp4(gpa, "", null);
    defer gpa.free(ok);
    try std.testing.expect((try parse(gpa, ok)).fp4_mlp);
    const cache = try withNvfp4(gpa, "\"kv_cache_quant_algo\": \"FP8\", ", null);
    defer gpa.free(cache);
    try std.testing.expectError(error.UnsupportedRicursivePrecision, parse(gpa, cache));
    const attention = try withNvfp4(gpa, "", 41);
    defer gpa.free(attention);
    try std.testing.expectError(error.UnsupportedRicursivePrecision, parse(gpa, attention));
}

test "ricursive config: another layer pattern with the release's geometry is not qualified" {
    const gpa = std.testing.allocator;
    const types = try std.mem.replaceOwned(u8, gpa, fixture, "\"layer_types\": [\n    \"sliding_attention\"", "\"layer_types\": [\n    \"full_attention\"");
    defer gpa.free(types);
    const both = try std.mem.replaceOwned(u8, gpa, types, "\"hybrid_layer_pattern\": [\n    1,", "\"hybrid_layer_pattern\": [\n    0,");
    defer gpa.free(both);
    try std.testing.expectError(error.UnqualifiedRicursiveCheckpoint, (try parse(gpa, both)).qualification());
}
