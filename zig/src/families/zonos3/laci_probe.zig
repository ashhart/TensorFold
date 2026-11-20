//! LACI's alignment probe (zonos3.tts.probe): a decode row's reading position from its attention over the text.
const std = @import("std");
const cfg = @import("config.zig");
const laci = @import("laci.zig");

/// Attention layers in layer order: every gdn_period-th layer.
pub const layers = cfg.layers / cfg.gdn_period;
/// A cache row's keys: kv_heads of head_dim.
pub const key_width = cfg.kv_width;

/// A masked head's window of text positions, inclusive and inside the text.
pub const Mask = struct { layer: u32, head: u32, lo: usize, hi: usize };

/// What a pass does for a probed row: its queries out, and the heads it recomputes over a window of the text.
pub const Probed = struct {
    /// The row's queries at the attention layers, normed, scaled and rotated as its attention took them.
    q: [layers][cfg.attn_inner]f32 = undefined,
    /// ACI: these heads see the text keys inside their window only (every other key as before).
    masks: []const Mask = &.{},
    /// The prompt's BOS..EOS text rows.
    text_start: usize = 0,
    text_len: usize = 0,
};

/// probe.clip_window: a window inside a text of `text_len` positions.
pub fn clip(lo: i64, hi: i64, text_len: usize) [2]usize {
    const last: i64 = @intCast(text_len - 1);
    const l = @max(0, @min(lo, last));
    return .{ @intCast(l), @intCast(@min(last, @max(hi, l))) };
}

/// The session's windows around `center`, clipped, for a pass; `out` holds a mask a head.
pub fn masks(heads: laci.Heads, center: ?usize, text_len: usize, out: []Mask) []const Mask {
    const c = center orelse return &.{};
    var buf: [cfg.heads * layers]laci.Window = undefined;
    for (heads.windows(c, &buf), out[0..heads.heads.len]) |w, *m| {
        const r = clip(w.lo, w.hi, text_len);
        m.* = .{ .layer = w.layer, .head = w.head, .lo = r[0], .hi = r[1] };
    }
    return out[0..heads.heads.len];
}

pub const Reading = struct { pos: usize, m: f64 };

/// probe.readout: the probed heads' softmax over the text keys (a masked head's over its window), summed.
pub fn readout(heads: laci.Heads, windows: []const Mask, q: *const [layers][cfg.attn_inner]f32, keys: [layers][]const f32, text_len: usize, scratch: []f32) Reading {
    const scale: f32 = 0.17677669529663688; // head_dim ** -0.5
    const acc = scratch[0..text_len];
    const layer_sum = scratch[text_len..][0..text_len];
    const p = scratch[2 * text_len ..][0..text_len];
    @memset(acc, 0);
    for (0..layers) |s| {
        const layer = (s + 1) * cfg.gdn_period - 1;
        // a layer's heads in head order, summed before they join the other layers'
        @memset(layer_sum, 0);
        for (0..cfg.heads) |h| {
            const probed = for (heads.heads) |head| {
                if (head.layer == layer and head.head == h) break true;
            } else false;
            if (!probed) continue;
            var lo: usize = 0;
            var hi: usize = text_len - 1;
            for (windows) |w| if (w.layer == layer and w.head == h) {
                lo = w.lo;
                hi = w.hi;
            };
            const qh = q[s][h * cfg.head_dim ..][0..cfg.head_dim];
            const kh = (h * cfg.kv_heads / cfg.heads) * cfg.head_dim;
            var top = -std.math.inf(f32);
            for (lo..hi + 1) |l| {
                var d: f32 = 0;
                for (qh, keys[s][l * key_width + kh ..][0..cfg.head_dim]) |x, y| d += x * y;
                p[l] = d * scale;
                top = @max(top, p[l]);
            }
            var sum: f32 = 0;
            for (p[lo .. hi + 1]) |*x| {
                x.* = @exp(x.* - top);
                sum += x.*;
            }
            for (layer_sum[lo .. hi + 1], p[lo .. hi + 1]) |*o, x| o.* += x / sum;
        }
        for (acc, layer_sum) |*o, x| o.* += x;
    }
    var best: usize = 0;
    var total: f32 = 0;
    var weighted: f32 = 0;
    for (acc, 0..) |x, l| {
        if (x > acc[best]) best = l;
        total += x;
        weighted += x * @as(f32, @floatFromInt(l));
    }
    return .{ .pos = best, .m = weighted / @max(total, 1e-30) };
}

test "readout: the heads' attention over the text keys, its argmax and centroid; a window cuts the rest off" {
    const t = std.testing;
    const heads: laci.Heads = .{ .heads = &.{ .{ .layer = 7, .head = 4, .radius = 2 }, .{ .layer = 3, .head = 2, .radius = 2 } }, .frames_per_token = 1 };
    const T = 6;
    var q: [layers][cfg.attn_inner]f32 = @splat(@splat(0));
    var k3: [T * key_width]f32 = @splat(0);
    var k7: [T * key_width]f32 = @splat(0);
    // layer 3's head 2 (key head 0) looks at text position 1, layer 7's head 4 (key head 1) at position 4
    q[0][2 * cfg.head_dim] = 100;
    k3[1 * key_width] = 1;
    q[1][4 * cfg.head_dim] = 100;
    k7[4 * key_width + cfg.head_dim] = 1;
    var scratch: [3 * T]f32 = undefined;
    const open = readout(heads, &.{}, &q, .{ &k3, &k7 }, T, &scratch);
    try t.expectEqual(@as(usize, 1), open.pos);
    try t.expectApproxEqAbs(@as(f64, 2.5), open.m, 1e-3);
    // layer 7's head held to positions 0..2: it spreads evenly there
    const held = readout(heads, &.{.{ .layer = 7, .head = 4, .lo = 0, .hi = 2 }}, &q, .{ &k3, &k7 }, T, &scratch);
    try t.expectEqual(@as(usize, 1), held.pos);
    try t.expectApproxEqAbs(@as(f64, 1.0), held.m, 1e-3);
    try t.expectEqual([2]usize{ 0, 3 }, clip(-2, 3, 6));
    try t.expectEqual([2]usize{ 5, 5 }, clip(9, 12, 6));
    var buf: [2]Mask = undefined;
    try t.expectEqualSlices(Mask, &.{ .{ .layer = 7, .head = 4, .lo = 3, .hi = 5 }, .{ .layer = 3, .head = 2, .lo = 3, .hi = 5 } }, masks(heads, 4, T, &buf));
}
