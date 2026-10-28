//! DFlash2's config.json for the HIP engine, validated against the target and the HIP tap ring.
const std = @import("std");
const engine = @import("../engine/engine.zig");

pub const Spec = @import("../model/view.zig").Spec;

/// Rows behind a stream's newest chunk the target's tap ring keeps: the drafter's window fits in them.
pub const tap_window = engine.tap_window;
/// The selector's candidates a row keeps at most: its top-k must fit.
pub const max_top = 20;
/// Most drafter layers and taps the engine runs.
pub const max_layers = 8;

/// What the drafter is checked against: the target's residual width, layers and vocabulary.
pub const Target = struct {
    hidden: usize,
    layers: usize,
    vocab: usize,

    pub fn of(s: Spec) Target {
        return .{ .hidden = s.hidden, .layers = s.n_layers, .vocab = s.vocab };
    }
};

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
    taps: [max_layers]u32,
    n_taps: usize,

    pub fn qWidth(c: Config) usize {
        return c.heads * c.head_dim;
    }
    pub fn kvWidth(c: Config) usize {
        return c.kv_heads * c.head_dim;
    }
    pub fn dynamic(c: Config) usize {
        return 4 * (c.hidden / c.group);
    }
    /// The target layers the drafter reads, in its fc's order.
    pub fn tapLayers(c: *const Config) []const u32 {
        return c.taps[0..c.n_taps];
    }
    /// The drafter's own key/value ring rows: the window and a padded block more.
    pub fn ringRows(c: Config) usize {
        return c.window + 16;
    }
    pub fn ringBytes(c: Config) usize {
        return c.layers * 2 * c.ringRows() * c.kvWidth() * 2;
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, target: Target) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        return parse(gpa, bytes, target);
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, target: Target) !Config {
        const doc = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch |e| switch (e) {
            error.OutOfMemory => return e,
            else => return error.BadDraftConfig,
        };
        defer doc.deinit();
        if (doc.value != .object) return error.BadDraftConfig;
        const o = doc.value.object;
        const d = (o.get("dflash_config") orelse return error.BadDraftConfig);
        if (d != .object) return error.BadDraftConfig;
        const rope = o.get("rope_parameters") orelse return error.BadDraftConfig;
        if (rope != .object) return error.BadDraftConfig;
        var c: Config = .{
            .hidden = try int(o, "hidden_size"),
            .intermediate = try int(o, "intermediate_size"),
            .layers = try int(o, "num_hidden_layers"),
            .vocab = try int(o, "vocab_size"),
            .heads = try int(o, "num_attention_heads"),
            .kv_heads = try int(o, "num_key_value_heads"),
            .head_dim = try int(o, "head_dim"),
            .block = try int(d.object, "block_size"),
            .mask = std.math.cast(u32, try whole(d.object, "mask_token_id")) orelse return error.BadDraftConfig,
            .window = try int(o, "sliding_window"),
            .group = try int(d.object, "conv_group_size"),
            .rank = try int(d.object, "selector_rank"),
            .topk = try int(d.object, "selector_top_k"),
            .eps = try positive(o, "rms_norm_eps"),
            .theta = try positive(rope.object, "rope_theta"),
            .taps = @splat(0),
            .n_taps = 0,
        };
        const taps = d.object.get("target_layer_ids") orelse return error.BadDraftConfig;
        if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > c.taps.len)
            return error.BadDraftConfig;
        for (taps.array.items, 0..) |t, i| {
            if (t != .integer or t.integer < 0 or t.integer >= target.layers) return error.BadDraftConfig;
            c.taps[i] = @intCast(t.integer);
            // one layer twice would tap one residual into two ring slots
            if (std.mem.indexOfScalar(u32, c.taps[0..i], c.taps[i]) != null) return error.BadDraftConfig;
        }
        c.n_taps = taps.array.items.len;
        // what the kernels fix: the head width, two-tap convolutions, a block of 8, the selector's top-k within a row
        const kernel = try int(d.object, "conv_kernel_size");
        if (c.head_dim != 128 or kernel != 2 or c.block != 8 or c.topk > max_top or c.mask >= c.vocab)
            return error.UnsupportedDraftConfig;
        if (c.hidden != target.hidden or c.vocab != target.vocab or c.hidden % 256 != 0 or c.hidden % c.group != 0)
            return error.UnsupportedDraftConfig;
        if (c.heads % c.kv_heads != 0 or c.layers > max_layers) return error.UnsupportedDraftConfig;
        // every projection's inputs are whole affine groups of 64
        if (c.intermediate % 64 != 0) return error.UnsupportedDraftConfig;
        _ = std.math.mul(usize, c.heads, c.head_dim) catch return error.UnsupportedDraftConfig;
        _ = std.math.mul(usize, c.kv_heads, c.head_dim) catch return error.UnsupportedDraftConfig;
        if (c.window < 16 or c.window > tap_window) return error.UnsupportedDraftConfig;
        return c;
    }
};

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

/// A norm epsilon or a RoPE base: positive and nonzero as an f32, so no zero, sign or infinity reaches the kernels.
fn positive(o: std.json.ObjectMap, key: []const u8) !f32 {
    const v = try num(o, key);
    if (!(v > 0 and v <= std.math.floatMax(f32))) return error.BadDraftConfig;
    const f: f32 = @floatCast(v);
    if (f == 0) return error.BadDraftConfig;
    return f;
}

fn num(o: std.json.ObjectMap, key: []const u8) !f64 {
    return switch (o.get(key) orelse return error.BadDraftConfig) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}
