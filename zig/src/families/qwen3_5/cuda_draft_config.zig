//! DFlash2 configuration and validated kernel widths.
const std = @import("std");
const engine = @import("cuda_engine.zig");
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
        return parse(gpa, bytes, target);
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
        if (c.window < 16 or c.window > @import("cuda_cache.zig").tap_window) return error.UnsupportedDraftConfig;
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
