//! Independently qualified prompt operators; decode keeps its existing path.
const std = @import("std");
const cuda = @import("cuda");

pub const Config = struct {
    gdn: bool,
    attention: cuda.fp8_attention_prompt.Mode,
    exact: cuda.attention_tuned.Mode = .disabled,
    expand: bool = false,

    pub fn key(c: Config) u8 {
        return @as(u8, @intFromBool(c.gdn)) | (@as(u8, @intCast(@backingInt(c.attention))) << 1) |
            (@as(u8, @intCast(@backingInt(c.exact))) << 3) | (@as(u8, @intFromBool(c.expand)) << 6);
    }
};

pub fn requested() !Config {
    const gdn = try @import("sys").environment.given("TF_QWEN_GDN_CHUNKED");
    const attention_raw = try @import("sys").environment.given("TF_QWEN_FP8_ATTENTION");
    const exact_raw = try @import("sys").environment.given("TF_QWEN_ATTN_EXACT");
    const expand_raw = try @import("sys").environment.given("TF_QWEN_EXPAND_KV");
    return selectExpanded(gdn, attention_raw, exact_raw, expand_raw);
}

pub fn select(gdn: ?[]const u8, attention_raw: ?[]const u8, exact_raw: ?[]const u8) !Config {
    return selectExpanded(gdn, attention_raw, exact_raw, null);
}

pub fn selectExpanded(gdn: ?[]const u8, attention_raw: ?[]const u8, exact_raw: ?[]const u8, expand_raw: ?[]const u8) !Config {
    const selected_attention = try cuda.fp8_attention_prompt.selected(attention_raw);
    const selected_exact = try cuda.attention_tuned.selected(exact_raw);
    const expand = try cuda.kv_expand.selected(expand_raw);
    if ((selected_attention != .disabled and selected_exact != .disabled) or
        (expand and (selected_attention != .disabled or selected_exact != .disabled))) return error.ConflictingPrefillAttention;
    return .{ .gdn = try cuda.fp8_prefill.selected(gdn), .attention = selected_attention, .exact = selected_exact, .expand = expand };
}

fn expansionGeometry(g: anytype, fp8_cache: bool) !void {
    if (!fp8_cache) return error.Fp8AttentionNeedsFp8Cache;
    if (!@import("cuda_geometry.zig").dense27(g) or g.q_heads != 24 or g.kv_heads != 4)
        return error.UnsupportedExpandKVGeometry;
}

pub const Paths = struct {
    config: Config = .{ .gdn = false, .attention = .disabled },
    gdn: ?cuda.gdn_chunked.Chunked = null,
    attention: ?cuda.fp8_attention_prompt.Native = null,
    exact: ?cuda.attention_tuned.Tuned = null,
    expand: ?cuda.kv_expand.Expanded = null,

    pub fn load(e: anytype, capability: u32) !Paths {
        var paths: Paths = .{ .config = try requested() };
        errdefer paths.deinit();
        if (paths.config.key() != 0 and (capability != 120 and capability != 121))
            return error.UnsupportedFastPrefillDevice;
        if (paths.config.gdn) paths.gdn = try cuda.gdn_chunked.Chunked.load(e.d, cuda.kernels.gdn_chunked, .{ .rows = @import("cuda_engine.zig").max_rows, .hk = e.g.k_heads, .hv = e.g.v_heads });
        if (paths.config.attention != .disabled) {
            if (!e.kv8) return error.Fp8AttentionNeedsFp8Cache;
            paths.attention = try cuda.fp8_attention_prompt.Native.init(e.d, cuda.kernels.fp8_attention_prompt, capability);
        }
        if (paths.config.exact != .disabled) {
            if (!e.kv8) return error.Fp8AttentionNeedsFp8Cache;
            paths.exact = if (paths.config.exact == .compact_w4h2s1)
                try cuda.attention_tuned.Tuned.borrowCompact(e.mods[16])
            else
                try cuda.attention_tuned.Tuned.init(e.d, cuda.kernels.prefill_attention_optimized, capability);
        }
        if (paths.config.expand) {
            try expansionGeometry(e.g, e.kv8);
            paths.expand = try cuda.kv_expand.Expanded.borrow(e.mods[16], e.prompt_attn.bf16_whole);
        }
        return paths;
    }

    pub fn deinit(paths: *Paths) void {
        if (paths.gdn) |*gdn| gdn.unload();
        if (paths.attention) |*op| op.deinit();
        if (paths.exact) |*op| op.deinit();
        if (paths.expand) |*op| op.deinit();
        paths.gdn = null;
        paths.attention = null;
        paths.exact = null;
        paths.expand = null;
    }
};

pub fn delta(e: anytype, layer: usize, rows: usize, sg: anytype) !bool {
    if (e.learning != null) return false;
    const gdn = if (e.prefill_paths.gdn) |*gdn| gdn else return false;
    const state = sg.rec + layer * e.g.v_heads * 128 * 128 * 4;
    try gdn.run(e.stream, .{ .rows = rows, .hk = e.g.k_heads, .hv = e.g.v_heads }, .{ .q = e.q, .k = e.k, .v = e.v, .decay = e.g_, .beta = e.beta, .initial = state, .state = state, .output = e.y });
    return true;
}

pub fn attention(e: anytype, layer: usize, from: usize, end: usize, sg: anytype) !bool {
    if (e.learning != null or !e.kv8) return false;
    if (e.prefill_paths.attention == null and e.prefill_paths.exact == null and e.prefill_paths.expand == null) return false;
    const cap: usize = @intCast(sg.cap);
    const count = e.g.kv_heads * (if (sg.pt != 0) @as(usize, @intCast(sg.ptok)) else cap);
    const base = sg.kv + layer * 2 * count * (256 + 4);
    const offset = from * e.g.q_heads * 256 * 2;
    const input: cuda.fp8_attention_prompt.Input = .{
        .q = e.qn + offset,
        .k = base,
        .v = base + count * 256,
        .k_scales = base + count * 512,
        .v_scales = base + count * 516,
        .output = e.y + offset,
        .pages = sg.pt,
        .shape = .{ .rows = @intCast(end - from), .prefix = @intCast(sg.len), .heads = @intCast(e.g.q_heads), .kv_heads = @intCast(e.g.kv_heads), .capacity = @intCast(cap), .scale = 1.0 / 16.0, .page_bytes = if (sg.pt != 0) @intCast(sg.ps) else 0, .page_tokens = if (sg.pt != 0) @intCast(sg.ptok) else 0 },
    };
    if (e.prefill_paths.expand) |*op| return op.run(e.stream, input);
    if (e.prefill_paths.exact) |*op| try op.run(e.stream, e.prefill_paths.config.exact, input) else if (e.prefill_paths.attention) |*op| try op.run(e.stream, e.prefill_paths.config.attention, input);
    return true;
}

test "experimental prompt identity separates each arithmetic mode" {
    const base: Config = .{ .gdn = false, .attention = .disabled };
    try std.testing.expectEqual(@as(u8, 0), base.key());
    var seen: u128 = 0;
    for ([_]bool{ false, true }) |gdn| for (std.enums.values(cuda.fp8_attention_prompt.Mode)) |mode|
        for (std.enums.values(cuda.attention_tuned.Mode)) |exact| for ([_]bool{ false, true }) |expand| {
            const key = (Config{ .gdn = gdn, .attention = mode, .exact = exact, .expand = expand }).key();
            try std.testing.expect(seen & (@as(u128, 1) << @intCast(key)) == 0);
            seen |= @as(u128, 1) << @intCast(key);
        };
}

test "experimental attention selection refuses conflicting and ambiguous flags" {
    try std.testing.expectEqual(@as(u8, 0), (try select(null, null, null)).key());
    try std.testing.expectError(error.ConflictingPrefillAttention, select("0", "2", "1"));
    try std.testing.expectError(error.InvalidExactAttentionMode, select("0", "0", "true"));
    try std.testing.expectError(error.InvalidFp8AttentionMode, select("0", "3", "0"));
    try std.testing.expectError(error.InvalidFastPrefill, select("yes", "0", "0"));
    try std.testing.expectEqual(@as(u8, 64), (try selectExpanded(null, null, null, "1")).key());
    try std.testing.expectError(error.ConflictingPrefillAttention, selectExpanded("0", "1", "0", "1"));
    try std.testing.expectError(error.ConflictingPrefillAttention, selectExpanded("0", "0", "6", "1"));
    try std.testing.expectError(error.InvalidExpandKVFlag, selectExpanded(null, null, null, "true"));
}

test "KV expansion is restricted to dense27 geometry and the original FP8 cache" {
    const Geometry = struct { hidden: usize = 5120, layers: usize = 64, experts: usize = 0, q_heads: usize = 24, kv_heads: usize = 4 };
    var g: Geometry = .{};
    try expansionGeometry(g, true);
    try std.testing.expectError(error.Fp8AttentionNeedsFp8Cache, expansionGeometry(g, false));
    g.kv_heads = 8;
    try std.testing.expectError(error.UnsupportedExpandKVGeometry, expansionGeometry(g, true));
    g.kv_heads = 4;
    g.layers = 63;
    try std.testing.expectError(error.UnsupportedExpandKVGeometry, expansionGeometry(g, true));
}
