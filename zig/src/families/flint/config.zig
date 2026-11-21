//! Flint's shape from config.json: per-layer attention kind (sliding with RoPE, full without), dense or routed MLP.
const std = @import("std");

pub const max_layers = 128;
pub const Kind = enum { sliding, full };

/// A layer's attention kind, whether it rotates q and k (sliding layers do, full layers are NoPE), and its MLP.
pub const Layer = struct { kind: Kind, rope: bool, moe: bool };

pub const Config = struct {
    hidden: u32 = 2048,
    layers: u32 = 49,
    vocab: u32 = 150144,
    eps: f32 = 1e-6,
    heads: u32 = 32,
    kv_heads: u32 = 4,
    head_dim: u32 = 128,
    intermediate: u32 = 6144,
    moe_intermediate: u32 = 512,
    shared_intermediate: u32 = 1024,
    experts: u32 = 256,
    topk: u32 = 6,
    theta: f32 = 10000,
    /// Keys a sliding row sees: itself and window - 1 before it.
    window: u32 = 513,
    context: u32 = 262144,
    layer: [max_layers]Layer = undefined,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    /// Key and value width of one token in one layer.
    pub fn kvWidth(c: *const Config) u32 {
        return c.kv_heads * c.head_dim;
    }
};

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse {
        std.log.err("flint: config.json lacks {s}", .{key});
        return error.BadFlintConfig;
    };
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadFlintConfig,
    };
}

fn int(o: std.json.ObjectMap, key: []const u8) !u32 {
    const v = try field(o, key);
    if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.BadFlintConfig;
    return @intCast(v.integer);
}

/// Layer ids listed under `key`, as a set.
fn ids(o: std.json.ObjectMap, key: []const u8) !std.StaticBitSet(max_layers) {
    var out: std.StaticBitSet(max_layers) = .empty;
    const v = try field(o, key);
    if (v != .array) return error.BadFlintConfig;
    for (v.array.items) |x| {
        if (x != .integer or x.integer < 0 or x.integer >= max_layers) return error.BadFlintConfig;
        out.set(@intCast(x.integer));
    }
    return out;
}

/// `key` names `want` where present (absent keys take the release's meaning).
fn says(o: std.json.ObjectMap, key: []const u8, want: []const u8) bool {
    const v = o.get(key) orelse return true;
    return v == .string and std.mem.eql(u8, v.string, want);
}

/// The router, indexed experts and shared scratch must all admit the same shape.
pub fn validateExperts(c: *const Config) error{UnsupportedFlintExperts}!void {
    if (c.hidden == 0 or c.hidden % 1024 != 0 or c.experts == 0 or c.experts > 256 or c.experts % 4 != 0 or
        c.topk == 0 or c.topk > c.experts or c.topk > 16 or c.intermediate == 0 or c.intermediate % 16 != 0 or
        c.moe_intermediate == 0 or c.moe_intermediate % 16 != 0 or c.shared_intermediate == 0 or
        c.shared_intermediate % 16 != 0 or c.shared_intermediate > c.intermediate) return error.UnsupportedFlintExperts;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    var doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadFlintConfig;
    const t = doc.value.object;
    var c = Config{};
    c.hidden = try int(t, "hidden_size");
    c.layers = try int(t, "num_hidden_layers");
    c.vocab = try int(t, "vocab_size");
    c.eps = @floatCast(try number(try field(t, "rms_norm_eps")));
    c.heads = try int(t, "num_attention_heads");
    c.kv_heads = try int(t, "num_key_value_heads");
    c.head_dim = try int(t, "head_dim");
    c.intermediate = try int(t, "intermediate_size");
    c.moe_intermediate = try int(t, "moe_intermediate_size");
    c.shared_intermediate = try int(t, "shared_expert_intermediate_size");
    c.experts = try int(t, "num_experts");
    c.topk = try int(t, "num_experts_per_tok");
    c.window = try int(t, "sliding_window");
    c.context = try int(t, "max_position_embeddings");
    c.theta = @floatCast(try number(try field(t, "rope_theta")));
    try validateExperts(&c);
    // the attention kernels hold a head in four lanes' registers a simdgroup: at most 256 dims, 32 at a time
    if (c.layers > max_layers or c.heads % c.kv_heads != 0 or c.head_dim % 32 != 0 or c.head_dim > 256 or c.window < 2) return error.BadFlintConfig;
    // sigmoid scores, top-k renormalized to 1, no score correction bias: the release's router
    const renorm = try field(t, "norm_topk_prob");
    if (renorm != .bool or !renorm.bool) return error.UnsupportedFlintRouter;
    if (!says(t, "router_score_function", "sigmoid") or !says(t, "scoring_func", "sigmoid")) return error.UnsupportedFlintRouter;
    if (t.get("router_expert_bias")) |v| if (v != .bool or v.bool) return error.UnsupportedFlintRouter;
    if (!says(t, "softmax_type", "learnable")) return error.UnsupportedFlintAttention;
    if (t.get("tie_word_embeddings")) |v| if (v != .bool or v.bool) return error.BadFlintConfig;
    const kinds = try field(t, "layer_types");
    if (kinds != .array or kinds.array.items.len != c.layers) return error.BadFlintConfig;
    const nope = try ids(t, "no_rope_layer_indices");
    const dense = try ids(t, "mlp_only_layers");
    for (0..c.layers) |i| {
        const k = kinds.array.items[i];
        if (k != .string) return error.BadFlintConfig;
        const kind: Kind = if (std.mem.eql(u8, k.string, "sliding_attention")) .sliding else if (std.mem.eql(u8, k.string, "full_attention")) .full else return error.BadFlintConfig;
        c.layer[i] = .{ .kind = kind, .rope = !nope.isSet(i), .moe = !dense.isSet(i) };
    }
    return c;
}

test "flint config: the release's kinds, NoPE layers, dense layers and router" {
    const c = try parse(std.testing.allocator, @embedFile("config_fixture.json"));
    try std.testing.expectEqual(@as(u32, 49), c.layers);
    try std.testing.expectEqual(Kind.full, c.layer[0].kind);
    try std.testing.expect(!c.layer[0].rope and !c.layer[0].moe);
    try std.testing.expect(c.layer[2].kind == .sliding and c.layer[2].rope and c.layer[2].moe);
    try std.testing.expect(c.layer[48].kind == .full and !c.layer[48].rope);
    try std.testing.expectEqual(@as(u32, 513), c.window);
    try std.testing.expectEqual(@as(u32, 512), c.kvWidth());
    var full: usize = 0;
    for (c.layer[0..c.layers]) |l| full += @intFromBool(l.kind == .full);
    try std.testing.expectEqual(@as(usize, 14), full);
}

test "flint config: a softmax router, a score bias or plain softmax attention is refused" {
    const fixture = @embedFile("config_fixture.json");
    const Case = struct { from: []const u8, to: []const u8, err: anyerror };
    const cases = [_]Case{
        .{ .from = "\"scoring_func\": \"sigmoid\"", .to = "\"scoring_func\": \"softmax\"", .err = error.UnsupportedFlintRouter },
        .{ .from = "\"router_expert_bias\": false", .to = "\"router_expert_bias\": true", .err = error.UnsupportedFlintRouter },
        .{ .from = "\"norm_topk_prob\": true", .to = "\"norm_topk_prob\": false", .err = error.UnsupportedFlintRouter },
        .{ .from = "\"softmax_type\": \"learnable\"", .to = "\"softmax_type\": \"vanilla\"", .err = error.UnsupportedFlintAttention },
        .{ .from = "\"num_experts_per_tok\": 6", .to = "\"num_experts_per_tok\": 17", .err = error.UnsupportedFlintExperts },
        .{ .from = "\"head_dim\": 128", .to = "\"head_dim\": 100", .err = error.BadFlintConfig },
    };
    for (cases) |x| {
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, fixture, x.from, x.to);
        defer std.testing.allocator.free(bytes);
        try std.testing.expectError(x.err, parse(std.testing.allocator, bytes));
    }
}
