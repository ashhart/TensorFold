//! DFlash2 configuration and validated kernel widths.
const std = @import("std");
const engine = @import("cuda_engine.zig");
const max_rows = engine.max_rows;
const lp_top = 20;
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
    window: usize,
    group: usize,
    rank: usize,
    topk: usize,
    eps: f32,
    theta: f32,
    taps: [8]u32,
    n_taps: usize,
    kind: Kind = .dflash2,
    // DSpine's code bridge: code width (rank is its message width) and its norm epsilon
    code: usize = 0,
    bridge_eps: f32 = 0,
    copy_match: i64 = 0,

    pub const Kind = enum { dflash2, dspine };

    pub fn qWidth(c: Config) usize {
        return c.heads * c.head_dim;
    }
    pub fn kvWidth(c: Config) usize {
        return c.kv_heads * c.head_dim;
    }
    pub fn dynamic(c: Config) usize {
        return 4 * (c.hidden / c.group);
    }
    /// The ring's rows: the window and a block more.
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

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, target: engine.Geometry) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        var config = try parse(gpa, bytes, target);
        if (config.kind != .dflash2) return config;
        if (try @import("sys").environment.given("TF_DRAFT_BLOCK")) |raw| {
            const block = try std.fmt.parseInt(usize, raw, 10);
            if (block != 8 and block != 16) return error.UnsupportedDraftConfig;
            config.block = block;
        }
        if (try @import("sys").environment.given("TF_DRAFT_COPY_MATCH")) |raw| {
            const count = try std.fmt.parseInt(i64, raw, 10);
            if (count < 2 or count > 20) return error.UnsupportedDraftConfig;
            config.copy_match = count;
        }
        return config;
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, target: engine.Geometry) !Config {
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer doc.deinit();
        if (doc.value != .object) return error.BadDraftConfig;
        const o = doc.value.object;
        const d = (o.get("dflash_config") orelse return error.BadDraftConfig);
        if (d != .object) return error.BadDraftConfig;
        if (@import("core").dspine.declared(doc.value)) return @import("cuda_dspine_config.zig").parse(doc.value, target);
        var c: Config = .{
            .hidden = try int(o, "hidden_size"),
            .intermediate = try int(o, "intermediate_size"),
            .layers = try int(o, "num_hidden_layers"),
            .vocab = try int(o, "vocab_size"),
            .heads = try int(o, "num_attention_heads"),
            .kv_heads = try int(o, "num_key_value_heads"),
            .head_dim = try int(o, "head_dim"),
            .block = try int(d.object, "block_size"),
            .mask = @intCast(try whole(d.object, "mask_token_id")),
            .window = try int(o, "sliding_window"),
            .group = try int(d.object, "conv_group_size"),
            .rank = try int(d.object, "selector_rank"),
            .topk = try int(d.object, "selector_top_k"),
            .eps = @floatCast(try num(o, "rms_norm_eps")),
            .theta = @floatCast(try num(try plainRope(o), "rope_theta")),
            .taps = @splat(0),
            .n_taps = 0,
        };
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
        if (c.window < 16 or c.window > engine.tap_rows - max_rows) return error.UnsupportedDraftConfig;
        c.block = 16;
        return c;
    }
};

/// CUDA's drafter runs every layer over a sliding ring; no declaration may silently select full attention.
pub fn slidingLayers(o: std.json.ObjectMap, layers: usize) !void {
    const enabled = o.get("use_sliding_window") orelse return error.BadDraftConfig;
    if (enabled != .bool) return error.BadDraftConfig;
    if (!enabled.bool) return error.UnsupportedDraftConfig;
    const types = o.get("layer_types") orelse return error.BadDraftConfig;
    if (types != .array or types.array.items.len != layers) return error.BadDraftConfig;
    for (types.array.items) |kind| {
        if (kind != .string) return error.BadDraftConfig;
        if (!std.mem.eql(u8, kind.string, "sliding_attention")) return error.UnsupportedDraftConfig;
    }
}

/// The draft's rope_parameters, refused unless unscaled: its kernels rope with theta alone.
pub fn plainRope(o: std.json.ObjectMap) !std.json.ObjectMap {
    const rope = o.get("rope_parameters") orelse return error.BadDraftConfig;
    if (rope != .object) return error.BadDraftConfig;
    for ([_][]const u8{ "rope_type", "type" }) |key| if (rope.object.get(key)) |v| {
        if (v != .string or !std.mem.eql(u8, v.string, "default")) return error.UnsupportedDraftConfig;
    };
    if (o.get("rope_scaling")) |v| if (v != .null) return error.UnsupportedDraftConfig;
    return rope.object;
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
pub fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try whole(o, key);
    if (v == 0) return error.BadDraftConfig;
    return v;
}

/// A token id: any whole number.
pub fn whole(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer < 0) return error.BadDraftConfig;
    return @intCast(v.integer);
}

pub fn num(o: std.json.ObjectMap, key: []const u8) !f64 {
    return switch (o.get(key) orelse return error.BadDraftConfig) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

test "a drafter directory is DFlash2's by its config's dflash_config; a checkpoint's own config or none is not" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    try std.testing.expect(!declared(a, io, root));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3_5_moe\",\"text_config\":{}}" });
    try std.testing.expect(!declared(a, io, root));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3\",\"dflash_config\":{\"block_size\":8}}" });
    try std.testing.expect(declared(a, io, root));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "[1]" });
    try std.testing.expect(!declared(a, io, root));
}

test {
    _ = @import("cuda_draft_config_test.zig");
    _ = @import("cuda_dspine_config.zig");
}

test "a drafter config with a zero divisor is refused, and any admitted rank's selector rows fit the staging area" {
    const Draft = @import("cuda_draft_frame.zig");
    const gpa = std.testing.allocator;
    const target: engine.Geometry = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 262144, .linear = @splat(false) };
    const text =
        \\{"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":248070,"selector_rank":256,
        \\ "selector_top_k":16,"target_layer_ids":[5,19,33,47,61]},"head_dim":128,"hidden_size":5120,"intermediate_size":17408,
        \\ "num_attention_heads":32,"num_hidden_layers":5,"num_key_value_heads":8,"rms_norm_eps":1e-06,
        \\ "rope_parameters":{"rope_theta":10000000},"sliding_window":2048,"vocab_size":248320,"use_sliding_window":true,
        \\ "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"]}
    ;
    try std.testing.expectEqual(@as(usize, 256), (try Config.parse(gpa, text, target)).rank);
    for ([_][2][]const u8{ .{ "\"conv_group_size\":16", "\"conv_group_size\":0" }, .{ "\"num_key_value_heads\":8", "\"num_key_value_heads\":0" } }) |r| {
        const bad = try std.mem.replaceOwned(u8, gpa, text, r[0], r[1]);
        defer gpa.free(bad);
        try std.testing.expectError(error.BadDraftConfig, Config.parse(gpa, bad, target));
    }
    // the kernels rope with theta alone: a scaled rope is refused, an explicit default one is read
    for ([_][2][]const u8{ .{ "{\"rope_theta\"", "{\"rope_type\":\"yarn\",\"factor\":4.0,\"rope_theta\"" }, .{ "{\"rope_theta\"", "{\"type\":\"yarn\",\"rope_theta\"" }, .{ "\"sliding_window\"", "\"rope_scaling\":{\"type\":\"yarn\"},\"sliding_window\"" } }) |r| {
        const scaled = try std.mem.replaceOwned(u8, gpa, text, r[0], r[1]);
        defer gpa.free(scaled);
        try std.testing.expectError(error.UnsupportedDraftConfig, Config.parse(gpa, scaled, target));
    }
    const plain = try std.mem.replaceOwned(u8, gpa, text, "{\"rope_theta\"", "{\"rope_type\":\"default\",\"rope_theta\"");
    defer gpa.free(plain);
    try std.testing.expectEqual(@as(usize, 256), (try Config.parse(gpa, plain, target)).rank);
    const wide = try std.mem.replaceOwned(u8, gpa, text, "\"selector_rank\":256", "\"selector_rank\":1024");
    defer gpa.free(wide);
    const c = try Config.parse(gpa, wide, target);
    try std.testing.expect(Draft.stagedLen(c.rank) >= Draft.hp_at + (c.block - 1) * c.rank * 2);
}
