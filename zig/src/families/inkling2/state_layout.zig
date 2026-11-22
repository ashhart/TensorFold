//! Inkling 2 Mini's stream state as bytes: KV rows and conv-history rings laid out so a rollback is a length reset.
const std = @import("std");
const config = @import("config.zig");

/// The most rows a `keep` may drop (a verify window: the pending token and its drafts).
pub const max_rollback = 16;
/// Slots of every convolution history: the 3 rows a window reads plus `max_rollback` plus slack, a power of two.
pub const hist_ring = 32;
/// Global layers' slots come in whole blocks of this many positions (the attention kernels' key blocks).
pub const global_block = 256;

pub const Geometry = struct {
    /// Positions a stream holds (global layers' keys); a stream's context limit.
    capacity: u32,
    /// The most rows of one forward over every item (Options.rows); sliding rings are sized by it.
    rows: u32,
};

/// Slots of a sliding layer's ring: its window, one forward's rows and a window's rollback, in whole 64s.
pub fn ringSlots(c: *const config.Text, rows: u32) u32 {
    return std.mem.alignForward(u32, c.window + @max(rows, max_rollback) + max_rollback, 64);
}

/// Slots of layer `i`'s keys: a ring (sliding) or one per position (global).
pub fn slots(c: *const config.Text, i: u32, g: Geometry) u32 {
    return if (config.isGlobal(i)) std.mem.alignForward(u32, g.capacity, global_block) else ringSlots(c, g.rows);
}

/// 0 where slots are positions, else the ring's slot count.
pub fn ring(c: *const config.Text, i: u32, g: Geometry) u32 {
    return if (config.isGlobal(i)) 0 else ringSlots(c, g.rows);
}

/// Width of one KV row of layer `i` in elements.
pub fn kvWidth(c: *const config.Text, i: u32) usize {
    const h = c.heads(i);
    return @as(usize, h.kv) * h.dim;
}

/// Byte offsets of layer `i`'s four histories in its history buffer, and its size.
pub const Hist = struct {
    k: usize,
    v: usize,
    attn: usize,
    mlp: usize,
    bytes: usize,
};

pub fn hist(c: *const config.Text, i: u32) Hist {
    const kv = kvWidth(c, i) * 2 * hist_ring;
    const full = @as(usize, c.hidden) * 2 * hist_ring;
    return .{ .k = 0, .v = kv, .attn = 2 * kv, .mlp = 2 * kv + full, .bytes = 2 * kv + 2 * full };
}

/// Bytes of layer `i`'s keys, and of its values (the same).
pub fn kvBytes(c: *const config.Text, i: u32, g: Geometry) usize {
    return @as(usize, slots(c, i, g)) * kvWidth(c, i) * 2;
}

/// Bytes a stream holds: every layer's keys and values and its histories.
pub fn bytes(cfg: *const config.Config, g: Geometry) u64 {
    var total: u64 = 0;
    for (0..cfg.text.executionLayers()) |l| {
        const i = cfg.text.physicalLayer(@intCast(l));
        total += 2 * kvBytes(&cfg.text, i, g) + hist(&cfg.text, i).bytes;
    }
    return total;
}

/// What a lane holds: its state, one logits row, and with DSpark the taps of `window_rows` rows.
pub fn laneBytes(cfg: *const config.Config, g: Geometry, drafts: bool, window_rows: u32) u64 {
    const taps: u64 = if (drafts) @as(u64, cfg.mtp.?.taps_n) * cfg.text.hidden * @sizeOf(u16) else 0;
    return bytes(cfg, g) + @as(u64, cfg.text.unpadded_vocab) * @sizeOf(f32) + window_rows * taps;
}

test "inkling2 state: rings, histories and a stream's bytes follow the geometry" {
    var t: config.Text = std.mem.zeroes(config.Text);
    t.window = 1024;
    t.layers = 48;
    t.hidden = 4608;
    t.local = .{ .q = 48, .kv = 8, .dim = 128 };
    t.global = .{ .q = 48, .kv = 8, .dim = 128 };
    const g: Geometry = .{ .capacity = 5000, .rows = 128 };
    // window + one forward's rows + a window's rollback, in whole 64s
    try std.testing.expectEqual(@as(u32, 1216), ringSlots(&t, g.rows));
    try std.testing.expectEqual(@as(u32, 5120), slots(&t, 5, g));
    try std.testing.expectEqual(@as(u32, 0), ring(&t, 5, g));
    try std.testing.expectEqual(@as(u32, 1216), ring(&t, 0, g));
    // a decode-sized forward still keeps the rollback slack
    try std.testing.expectEqual(@as(u32, 1088), ringSlots(&t, 4));
    // the convolution histories: 2 KV inputs and 2 hidden inputs a layer, 32 slots each
    const h = hist(&t, 0);
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 2 * 32 + 2 * 4608 * 2 * 32), h.bytes);
    try std.testing.expectEqual(h.bytes - 4608 * 2 * 32, h.mlp);
    var total: u64 = 0;
    for (0..48) |l| total += 2 * kvBytes(&t, @intCast(l), g) + hist(&t, @intCast(l)).bytes;
    const cfg: config.Config = .{ .text = t, .vision = undefined, .mtp = null, .eos = 0, .image_token = 0 };
    try std.testing.expectEqual(total, bytes(&cfg, g));
    // Mini's looped geometry: 24 physical layers run as 48 visits, 8 of them global (physical 5, 11, 17 and 23).
    var mini: config.Text = std.mem.zeroes(config.Text);
    mini.window = 1024;
    mini.layers = 24;
    mini.hidden = 4608;
    mini.local = .{ .q = 48, .kv = 8, .dim = 128 };
    mini.global = .{ .q = 48, .kv = 8, .dim = 128 };
    mini.loop = .{ .start = 6, .end = 18, .passes = 3, .rank = 64 };
    const mc: config.Config = .{ .text = mini, .vision = undefined, .mtp = null, .eos = 0, .image_token = 0 };
    try std.testing.expectEqual(@as(u32, 48), mini.executionLayers());
    var mtotal: u64 = 0;
    var global_visits: u32 = 0;
    for (0..mini.executionLayers()) |l| {
        const i = mini.physicalLayer(@intCast(l));
        if (config.isGlobal(i)) global_visits += 1;
        mtotal += 2 * kvBytes(&mini, i, g) + hist(&mini, i).bytes;
    }
    try std.testing.expectEqual(@as(u32, 8), global_visits);
    try std.testing.expectEqual(mtotal, bytes(&mc, g));
    // visit 41 is physical 17, global; visit 40 is physical 16, sliding
    try std.testing.expectEqual(@as(u32, 17), mini.physicalLayer(41));
    try std.testing.expectEqual(@as(u32, 0), ring(&mini, mini.physicalLayer(41), g));
    try std.testing.expectEqual(@as(u32, 1216), ring(&mini, mini.physicalLayer(40), g));
}
