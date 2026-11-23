//! A draft stream's context at a length, device-copied: the keys and values its next block attends over.
const std = @import("std");
const mtl = @import("metal");
const config = @import("config.zig");
const core_mod = @import("metal_draft_core.zig");
const wt = @import("metal_draft_weights.zig");
const runs = @import("metal_snapshot.zig").runs;

const Core = core_mod.Core;
const Lane = core_mod.Lane;

/// Positions of draft layer `li` a snapshot at `len` keeps: a sliding layer's last window, a full layer's all.
fn live(m: config.Mtp, li: usize, len: u32) struct { start: u32, count: u32 } {
    const start: u32 = if (m.local[li]) len -| m.window else 0;
    return .{ .start = start, .count = len - start };
}

/// Bytes of one position's key (or value) in a layer.
fn row(m: config.Mtp) usize {
    return @as(usize, m.kv_heads) * wt.head_dim * @sizeOf(u16);
}

pub const Snapshot = struct {
    len: u32,
    layers: u32,
    /// Per layer: its live keys then its live values, in position order.
    kv: [wt.max_layers]mtl.Buffer,

    pub fn deinit(s: *Snapshot) void {
        for (s.kv[0..s.layers]) |b| b.deinit();
    }

    /// Bytes a snapshot at `len` holds: every position of a full layer and the last window of a sliding one.
    pub fn bytes(m: config.Mtp, len: u32) u64 {
        var total: u64 = 0;
        for (0..m.layers) |li| total += 2 * @as(u64, live(m, li, len).count) * row(m);
        return total;
    }
};

/// A copy of `l`'s context after `len` rows, all absorbed (none left on the host), in new device storage.
pub fn save(k: *const Core, l: *const Lane, len: u32) !Snapshot {
    var s: Snapshot = .{ .len = len, .layers = k.m.layers, .kv = undefined };
    var made: usize = 0;
    errdefer for (s.kv[0..made]) |b| b.deinit();
    for (0..k.m.layers) |li| {
        s.kv[li] = try k.device.buffer(@max(2 * @as(usize, live(k.m, li, len).count) * row(k.m), 16), mtl.ResourceOptions.private);
        made += 1;
    }
    try copy(k, l, &s, true);
    return s;
}

/// Writes `s` into `l`'s caches, each position in the slot absorbing it would have used.
pub fn restore(k: *const Core, l: *const Lane, s: *const Snapshot) !void {
    try copy(k, l, s, false);
}

/// One blit pass between a lane's caches and a snapshot: `out` copies the lane's rows into it, else back.
fn copy(k: *const Core, l: *const Lane, s: *const Snapshot, out: bool) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = k.queue.commandBuffer();
    const blit = cb.blit();
    const w = row(k.m);
    for (0..k.m.layers) |li| {
        const lv = live(k.m, li, s.len);
        const ring: u32 = if (k.m.local[li]) Core.ring(k.m) else 0; // a full layer: slot = position
        var at: usize = 0;
        for ([_]mtl.Buffer{ l.k[li], l.v[li] }) |side| for (runs(lv.start, lv.count, ring)) |run| {
            if (run.count == 0) continue;
            if (out) blit.copy(side, run.slot * w, s.kv[li], at, run.count * w) else blit.copy(s.kv[li], at, side, run.slot * w, run.count * w);
            at += run.count * w;
        };
    }
    blit.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| return Core.failed(msg);
}
