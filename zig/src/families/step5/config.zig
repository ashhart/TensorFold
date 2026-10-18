//! Step-5's text model shape from config.json's text_config: per-layer attention kind, rotary span, theta and limits.
const std = @import("std");

pub const max_layers = 128;
pub const Kind = enum { sliding, full };

pub const Layer = struct { kind: Kind, span: u32, theta: f32, limit: f32, shared_limit: f32, moe: bool };

pub const Config = struct {
    hidden: u32 = 4096,
    layers: u32 = 92,
    /// MTP layers after the main stack (their per-layer entries follow the main stack's).
    mtp: u32 = 0,
    vocab: u32 = 128896,
    eps: f32 = 1e-5,
    heads: u32 = 64,
    kv_heads: u32 = 4,
    head_dim: u32 = 192,
    intermediate: u32 = 13824,
    moe_intermediate: u32 = 1536,
    shared_intermediate: u32 = 1536,
    experts: u32 = 352,
    topk: u32 = 8,
    routed_scale: f32 = 3.0,
    window: u32 = 512,
    /// The sparse indexer: heads (4 a KV group), proxy width, rotary span, regions kept, tokens a region.
    index_heads: u32 = 16,
    proxy: u32 = 256,
    index_span: u32 = 32,
    index_topk: u32 = 512,
    region: u32 = 8,
    context: u32 = 1048576,
    eos: [2]u32 = .{ 1, 2 },
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
        std.log.err("step5: config.json's text_config lacks {s}", .{key});
        return error.BadStep5Config;
    };
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadStep5Config,
    };
}

fn int(o: std.json.ObjectMap, key: []const u8) !u32 {
    const v = try field(o, key);
    if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.BadStep5Config;
    return @intCast(v.integer);
}

/// Indexer heads: sparse_indexer_num_heads, provider groups being the KV groups.
fn indexHeads(sp: std.json.ObjectMap, kv: u32) !u32 {
    const groups = try int(sp, "num_provider_groups");
    if (groups != kv) return error.UnsupportedStep5Indexer;
    return int(sp, "sparse_indexer_num_heads");
}

fn list(o: std.json.ObjectMap, key: []const u8, n: u32) ![]const std.json.Value {
    const v = try field(o, key);
    if (v != .array or v.array.items.len < n) return error.BadStep5Config;
    return v.array.items;
}

/// The router, indexed experts and shared scratch must all admit the same shape.
pub fn validateExperts(c: *const Config) error{UnsupportedStep5Experts}!void {
    if (c.hidden == 0 or c.hidden % 64 != 0 or c.experts == 0 or c.experts > 512 or c.experts % 4 != 0 or
        c.topk == 0 or c.topk > c.experts or c.topk > 16 or c.intermediate == 0 or c.intermediate % 16 != 0 or
        c.moe_intermediate == 0 or c.moe_intermediate % 16 != 0 or c.shared_intermediate == 0 or
        c.shared_intermediate % 16 != 0 or c.shared_intermediate > c.intermediate) return error.UnsupportedStep5Experts;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    var doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadStep5Config;
    const top = doc.value.object;
    const tv = top.get("text_config") orelse doc.value;
    if (tv != .object) return error.BadStep5Config;
    const t = tv.object;
    var c = Config{};
    c.hidden = try int(t, "hidden_size");
    c.layers = try int(t, "num_hidden_layers");
    if (t.get("num_nextn_predict_layers")) |v| if (v == .integer and v.integer > 0) {
        c.mtp = @intCast(v.integer);
    };
    c.vocab = try int(t, "vocab_size");
    c.eps = @floatCast(try number(try field(t, "rms_norm_eps")));
    c.heads = try int(t, "num_attention_heads");
    c.kv_heads = try int(t, "num_attention_groups");
    c.head_dim = try int(t, "head_dim");
    c.intermediate = try int(t, "intermediate_size");
    c.moe_intermediate = try int(t, "moe_intermediate_size");
    c.shared_intermediate = try int(t, "share_expert_dim");
    c.experts = try int(t, "moe_num_experts");
    c.topk = try int(t, "moe_top_k");
    c.routed_scale = @floatCast(try number(try field(t, "moe_router_scaling_factor")));
    c.window = try int(t, "sliding_window");
    c.context = try int(t, "max_position_embeddings");
    try validateExperts(&c);
    if (c.layers + c.mtp > max_layers or c.heads % c.kv_heads != 0 or c.head_dim % 32 != 0 or c.head_dim > 256) return error.BadStep5Config;
    const norm = try field(t, "norm_expert_weight");
    if (norm != .bool or !norm.bool) return error.UnsupportedStep5Router;
    const sparse = try field(t, "sparse_config");
    if (sparse != .object) return error.BadStep5Config;
    const sp = sparse.object;
    const on = try field(sp, "enabled");
    if (on != .bool or !on.bool) return error.UnsupportedStep5Attention;
    c.region = try int(sp, "region_block_size");
    c.index_topk = try int(sp, "topk");
    c.proxy = try int(sp, "proxy_dim");
    c.index_span = try int(sp, "sparse_indexer_rope_dim");
    c.index_heads = try indexHeads(sp, c.kv_heads);
    if (c.region != 8 or c.proxy != 256 or c.index_heads != 4 * c.kv_heads or c.index_span % 2 != 0 or c.index_span > c.proxy) return error.UnsupportedStep5Indexer;
    const eos = try field(t, "eos_token_id");
    if (eos == .array and eos.array.items.len == 2) {
        for (&c.eos, eos.array.items) |*e, v| e.* = @intCast(v.integer);
    }
    const all = c.layers + c.mtp;
    const kinds = try list(t, "layer_types", all);
    const factors = try list(t, "partial_rotary_factors", all);
    const thetas = try list(t, "rope_theta", all);
    const limits = try list(t, "swiglu_limits", all);
    const shared = try list(t, "swiglu_limits_shared", all);
    var moe: std.StaticBitSet(max_layers) = .empty;
    for ((try field(t, "moe_layer_list")).array.items) |v| {
        if (v != .integer or v.integer < 0 or v.integer >= max_layers) return error.BadStep5Config;
        moe.set(@intCast(v.integer));
    }
    for (0..all) |i| {
        if (kinds[i] != .string) return error.BadStep5Config;
        const kind: Kind = if (std.mem.eql(u8, kinds[i].string, "sliding_attention")) .sliding else if (std.mem.eql(u8, kinds[i].string, "full_attention")) .full else return error.BadStep5Config;
        // the rotated span stays inside the head, which the qk kernel's 256-wide norm array holds
        const factor = try number(factors[i]);
        if (!(factor >= 0 and factor <= 1)) return error.BadStep5Config;
        var span: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(c.head_dim)) * factor));
        span -= span % 2;
        c.layer[i] = .{ .kind = kind, .span = span, .theta = @floatCast(try number(thetas[i])), .limit = @floatCast(try number(limits[i])), .shared_limit = @floatCast(try number(shared[i])), .moe = moe.isSet(i) };
    }
    return c;
}

test "a sliced Step-5 config: kinds, spans, thetas and the dense limit" {
    const text =
        \\{"text_config": {"hidden_size": 4096, "num_hidden_layers": 4, "vocab_size": 128896, "rms_norm_eps": 1e-05,
        \\ "num_attention_heads": 64, "num_attention_groups": 4, "head_dim": 192, "intermediate_size": 13824,
        \\ "moe_intermediate_size": 1536, "share_expert_dim": 1536, "moe_num_experts": 352, "moe_top_k": 8,
        \\ "moe_router_scaling_factor": 3.0, "sliding_window": 512, "max_position_embeddings": 1048576,
        \\ "norm_expert_weight": true, "eos_token_id": [1, 2], "moe_layer_list": [3],
        \\ "sparse_config": {"enabled": true, "region_block_size": 8, "topk": 512, "proxy_dim": 256, "sparse_indexer_rope_dim": 32,
        \\  "sparse_indexer_num_heads": 16, "num_provider_groups": 4},
        \\ "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        \\ "partial_rotary_factors": [1.0, 1.0, 1.0, 0.3333333333333333],
        \\ "rope_theta": [10000.0, 10000.0, 10000.0, 10000000.0],
        \\ "swiglu_limits": [7.0, 7.0, 7.0, 7.0], "swiglu_limits_shared": [7.0, 7.0, 7.0, 7.0]}}
    ;
    const c = try parse(std.testing.allocator, text);
    try std.testing.expectEqual(@as(u32, 4), c.layers);
    try std.testing.expectEqual(Kind.full, c.layer[3].kind);
    try std.testing.expectEqual(@as(u32, 64), c.layer[3].span);
    try std.testing.expectEqual(@as(u32, 192), c.layer[0].span);
    try std.testing.expect(c.layer[3].moe and !c.layer[2].moe);
    try std.testing.expectEqual(@as(f32, 1e7), c.layer[3].theta);
    try std.testing.expectEqual(@as(u32, 512), c.index_topk);
    try std.testing.expectEqual(@as(u32, 16), c.index_heads);
    try std.testing.expectEqual(@as(u32, 768), c.kvWidth());
    // a rotary factor past 1 (or below 0) would rotate past the head
    for ([_][]const u8{ "1.5", "-0.5" }) |bad| {
        const sliced = try std.mem.replaceOwned(u8, std.testing.allocator, text, "0.3333333333333333", bad);
        defer std.testing.allocator.free(sliced);
        try std.testing.expectError(error.BadStep5Config, parse(std.testing.allocator, sliced));
    }
}

test "step5 host: expert geometry is refused by the serialized config before loading" {
    const fixture = @embedFile("config_fixture.json");
    const Case = struct { from: []const u8, to: []const u8 };
    const cases = [_]Case{
        .{ .from = "\"moe_top_k\": 8", .to = "\"moe_top_k\": 17" },
        .{ .from = "\"moe_top_k\": 8", .to = "\"moe_top_k\": 353" },
        .{ .from = "\"moe_num_experts\": 352", .to = "\"moe_num_experts\": 4" },
        .{ .from = "\"moe_num_experts\": 352", .to = "\"moe_num_experts\": 350" },
        .{ .from = "\"moe_num_experts\": 352", .to = "\"moe_num_experts\": 516" },
        .{ .from = "\"share_expert_dim\": 1536", .to = "\"share_expert_dim\": 13840" },
    };
    for (cases) |c| {
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, fixture, c.from, c.to);
        defer std.testing.allocator.free(bytes);
        try std.testing.expectError(error.UnsupportedStep5Experts, parse(std.testing.allocator, bytes));
    }
}

test "step5 host: expert endpoints and released rotary refusals stay admitted or refused" {
    var c = Config{};
    try validateExperts(&c);
    for ([_]u32{ 4, 512 }) |experts| for ([_]u32{ 1, @min(experts, 16) }) |topk| {
        c.experts = experts;
        c.topk = topk;
        try validateExperts(&c);
    };
    const fields = [_][]const u8{ "hidden", "intermediate", "moe_intermediate", "shared_intermediate" };
    inline for (fields) |field_name| {
        c = .{};
        @field(c, field_name) += 1;
        try std.testing.expectError(error.UnsupportedStep5Experts, validateExperts(&c));
    }
    for ([_][]const u8{ "1.5", "-0.5" }) |value| {
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, @embedFile("config_fixture.json"), "0.3333333333333333", value);
        defer std.testing.allocator.free(bytes);
        try std.testing.expectError(error.BadStep5Config, parse(std.testing.allocator, bytes));
    }
}
