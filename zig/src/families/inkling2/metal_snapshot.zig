//! A stream's exact state at a length, device-copied: keys and values next rows can read, every convolution history.
const std = @import("std");
const mtl = @import("metal");
const config = @import("config.zig");
const st = @import("metal_state.zig");
const layout = @import("state_layout.zig");
const Model = @import("metal_model.zig").Model;

/// Positions of layer `i` a snapshot at `len` keeps: the first of them and their count.
fn live(t: *const config.Text, i: u32, len: u32) struct { start: u32, count: u32 } {
    if (config.isGlobal(i)) return .{ .start = 0, .count = len };
    const start = len -| t.window;
    return .{ .start = start, .count = len - start };
}

/// `count` positions from `start` in a ring of `ring` slots (0: slot = position) as at most two runs of slots.
pub fn runs(start: u32, count: u32, ring: u32) [2]struct { slot: u32, count: u32 } {
    if (ring == 0 or count == 0) return .{ .{ .slot = start, .count = count }, .{ .slot = 0, .count = 0 } };
    const first = start % ring;
    const head = @min(count, ring - first);
    return .{ .{ .slot = first, .count = head }, .{ .slot = 0, .count = count - head } };
}

pub const Snapshot = struct {
    gpa: std.mem.Allocator,
    geometry: layout.Geometry,
    len: u32,
    /// Per layer: its live keys then its live values, in position order.
    kv: []mtl.Buffer,
    /// Every layer's convolution histories, as the state lays them out.
    hist: []u8,

    pub fn deinit(s: *Snapshot) void {
        for (s.kv) |b| b.deinit();
        s.gpa.free(s.kv);
        s.gpa.free(s.hist);
    }

    /// Bytes a snapshot of a stream at `len` holds (device and host).
    pub fn bytes(cfg: *const config.Config, len: u32) u64 {
        var total: u64 = 0;
        for (0..cfg.text.executionLayers()) |l| {
            const i = cfg.text.physicalLayer(@intCast(l));
            total += 2 * @as(u64, live(&cfg.text, i, len).count) * layout.kvWidth(&cfg.text, i) * 2 + layout.hist(&cfg.text, i).bytes;
        }
        return total;
    }
};

fn failed(cb: mtl.CommandBuffer) !void {
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2 snapshot: {s}", .{msg});
        return error.GpuFailed;
    }
}

/// Copy `state`'s exact contents (the model's queue; no forward in flight).
pub fn save(m: *Model, state: *const st.State) !Snapshot {
    const t = &m.config.text;
    const g = state.geometry;
    var s: Snapshot = .{ .gpa = m.gpa, .geometry = g, .len = state.len, .kv = undefined, .hist = undefined };
    s.kv = try m.gpa.alloc(mtl.Buffer, state.layers);
    errdefer m.gpa.free(s.kv);
    var made: usize = 0;
    errdefer for (s.kv[0..made]) |b| b.deinit();
    var hist_bytes: usize = 0;
    for (0..state.layers) |l| {
        const i = t.physicalLayer(@intCast(l));
        const kvw = layout.kvWidth(t, i);
        s.kv[l] = try m.device.buffer(@max(2 * @as(usize, live(t, i, state.len).count) * kvw * 2, 16), mtl.ResourceOptions.private);
        made += 1;
        hist_bytes += state.hist[l].length();
    }
    s.hist = try m.gpa.alloc(u8, hist_bytes);
    errdefer m.gpa.free(s.hist);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    var at: usize = 0;
    for (0..state.layers) |l| {
        const i = t.physicalLayer(@intCast(l));
        const kvw: usize = layout.kvWidth(t, i) * 2;
        const lv = live(t, i, state.len);
        const slot_bytes = layout.kvBytes(t, i, g);
        var packed_at: usize = 0;
        for ([_]usize{ 0, slot_bytes }) |side| {
            for (runs(lv.start, lv.count, layout.ring(t, i, g))) |run| {
                if (run.count == 0) continue;
                blit.copy(state.kv[l], side + run.slot * kvw, s.kv[l], packed_at, run.count * kvw);
                packed_at += run.count * kvw;
            }
        }
        @memcpy(s.hist[at..][0..state.hist[l].length()], state.hist[l].contents()[0..state.hist[l].length()]);
        at += state.hist[l].length();
    }
    blit.end();
    cb.commit();
    try failed(cb);
    return s;
}

/// Write `snap` back into `state` (same geometry): its length, keys, values and histories exactly as saved.
pub fn restore(m: *Model, state: *st.State, snap: *const Snapshot) !void {
    const t = &m.config.text;
    const g = state.geometry;
    if (snap.geometry.capacity != g.capacity or snap.geometry.rows != g.rows or snap.kv.len != state.layers or state.layers != t.executionLayers()) return error.SnapshotGeometry;
    errdefer state.failed = true;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const blit = cb.blit();
    var at: usize = 0;
    for (0..state.layers) |l| {
        const i = t.physicalLayer(@intCast(l));
        const kvw: usize = layout.kvWidth(t, i) * 2;
        const lv = live(t, i, snap.len);
        const slot_bytes = layout.kvBytes(t, i, g);
        var packed_at: usize = 0;
        for ([_]usize{ 0, slot_bytes }) |side| {
            for (runs(lv.start, lv.count, layout.ring(t, i, g))) |run| {
                if (run.count == 0) continue;
                blit.copy(snap.kv[l], packed_at, state.kv[l], side + run.slot * kvw, run.count * kvw);
                packed_at += run.count * kvw;
            }
        }
        const n = state.hist[l].length();
        @memcpy(state.hist[l].contents()[0..n], snap.hist[at..][0..n]);
        at += n;
    }
    blit.end();
    cb.commit();
    try failed(cb);
    state.len = snap.len;
    state.floor = snap.len;
    state.failed = false;
}

test "inkling2 snapshot: live positions wrap a ring in at most two runs" {
    var t: config.Text = std.mem.zeroes(config.Text);
    t.window = 1024;
    const l = live(&t, 0, 5000);
    try std.testing.expectEqual(@as(u32, 3976), l.start);
    try std.testing.expectEqual(@as(u32, 1024), l.count);
    const g = live(&t, 5, 5000);
    try std.testing.expectEqual(@as(u32, 0), g.start);
    try std.testing.expectEqual(@as(u32, 5000), g.count);
    const r = runs(3976, 1024, 1216); // slots 3976 % 1216 = 328 .. 1215, then 0 .. 135
    try std.testing.expectEqual(@as(u32, 328), r[0].slot);
    try std.testing.expectEqual(@as(u32, 888), r[0].count);
    try std.testing.expectEqual(@as(u32, 0), r[1].slot);
    try std.testing.expectEqual(@as(u32, 136), r[1].count);
    try std.testing.expectEqual(@as(u32, 1024), r[0].count + r[1].count);
    const whole = runs(0, 77, 0);
    try std.testing.expectEqual(@as(u32, 77), whole[0].count);
    try std.testing.expectEqual(@as(u32, 0), whole[1].count);
}
