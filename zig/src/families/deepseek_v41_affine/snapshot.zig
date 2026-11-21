//! Kept prompt states on the host: engram history, last 128 window keys, sources' groups, a filling pair.
const std = @import("std");
const c = @import("config.zig");
const st = @import("state.zig");

pub const Snap = struct { at: u32, bytes: []u8 };

const key_row = c.head_dim * 2;
const index_row = c.index_dim * 2;
const proj_row = c.head_dim * 4;
/// The ratio-2 projections a state keeps: a pair's first half when `at` is odd (two rows, the simplest bound).
const proj_rows = 2;

fn windowRows(at: usize) usize {
    return @min(c.window, at);
}

pub fn bytes(at: u32, drafts: bool) u64 {
    const a: u64 = at;
    var n: u64 = 4 * a + @as(u64, c.layers) * windowRows(at) * key_row;
    if (drafts) n += @as(u64, c.stages) * windowRows(at) * key_row;
    for (c.kv_sources) |layer| {
        n += (a / c.ratios[layer]) * (key_row + index_row);
        if (c.ratios[layer] > 1) n += 2 * proj_rows * proj_row;
    }
    return n;
}

const Copy = enum { save, restore };

/// Rows `first..first + n` of a ring buffer, by position, to or from `out` at `off`.
fn ring(comptime dir: Copy, buf: anytype, row: usize, first: usize, n: usize, out: []u8, off: *usize) void {
    for (first..first + n) |p| {
        const r = (buf.contents() + (p % st.ring) * row)[0..row];
        switch (dir) {
            .save => @memcpy(out[off.*..][0..row], r),
            .restore => @memcpy(r, out[off.*..][0..row]),
        }
        off.* += row;
    }
}

fn span(comptime dir: Copy, buf: anytype, n: usize, out: []u8, off: *usize) void {
    switch (dir) {
        .save => @memcpy(out[off.*..][0..n], buf.contents()[0..n]),
        .restore => @memcpy(buf.contents()[0..n], out[off.*..][0..n]),
    }
    off.* += n;
}

fn copy(comptime dir: Copy, cache: anytype, at: usize, out: []u8) void {
    var off: usize = 0;
    const ids = std.mem.sliceAsBytes(cache.engram_ids.items[0..at]);
    switch (dir) {
        .save => @memcpy(out[0..ids.len], ids),
        .restore => @memcpy(@constCast(ids), out[0..ids.len]),
    }
    off += ids.len;
    const w = windowRows(at);
    for (cache.windows) |b| ring(dir, b, key_row, at - w, w, out, &off);
    if (cache.stage_rings) |rings| for (rings) |b| ring(dir, b, key_row, at - w, w, out, &off);
    for (cache.sources, c.kv_sources, 0..) |src, layer, i| {
        const groups = at / c.ratios[layer];
        span(dir, src.latents, groups * key_row, out, &off);
        span(dir, src.keys, groups * index_row, out, &off);
        if (c.ratios[layer] > 1) {
            const first = at -| proj_rows;
            const n = at - first;
            ring(dir, cache.proj_kv[i].?, proj_row, first, n, out, &off);
            ring(dir, cache.proj_score[i].?, proj_row, first, n, out, &off);
            const pad = (proj_rows - n) * 2 * proj_row;
            if (dir == .save) @memset(out[off..][0..pad], 0);
            off += pad;
        }
    }
}

/// Copy `cache`'s state after `at` tokens (cache.len == at, nothing in flight).
pub fn save(gpa: std.mem.Allocator, cache: *const st.Cache, at: u32) !*Snap {
    if (cache.len != at) return error.DeepseekSnapshotPosition;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .at = at, .bytes = try gpa.alloc(u8, bytes(at, cache.stage_rings != null)) };
    copy(.save, cache, at, snap.bytes);
    return snap;
}

/// Make a fresh `cache` hold `snap`'s state (cache.len becomes snap.at).
pub fn restore(cache: *st.Cache, snap: *const Snap) !void {
    if (cache.len != 0 or snap.at > cache.capacity or snap.bytes.len != bytes(snap.at, cache.stage_rings != null)) return error.DeepseekSnapshotPosition;
    cache.engram_ids.items.len = snap.at;
    copy(.restore, cache, snap.at, snap.bytes);
    cache.len = snap.at;
    cache.held_n = 0;
}

pub fn drop(gpa: std.mem.Allocator, snap: *Snap) void {
    gpa.free(snap.bytes);
    gpa.destroy(snap);
}

test "a state's bytes grow by the engram id and the sources' groups a token, past the windows" {
    const a = bytes(1000, false);
    const b = bytes(1002, false);
    // two tokens: 8 bytes of ids, ratio-2 sources one group each (3), the ratio-1 source two
    try std.testing.expectEqual(@as(u64, 8 + 5 * (key_row + index_row)), b - a);
    try std.testing.expect(bytes(10, true) > bytes(10, false));
}
