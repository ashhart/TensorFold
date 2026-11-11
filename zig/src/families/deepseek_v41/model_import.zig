//! A blank host Snapshot an importer writes into, and the seal that makes it restorable by `snapshotRestore`.
const std = @import("std");
const kv = @import("kv.zig");
const engram = @import("engram.zig");
const model = @import("model.zig");
const Model = model.Model;
const Snapshot = Model.Snapshot;

/// A zeroed Snapshot shaped for `m`'s slots (capacity-sized `storage`, mutable `live`, the taps ring); position 0.
pub fn blank(m: *Model) !*Snapshot {
    const plan = m.caches[0].plan;
    const snap = try m.gpa.create(Snapshot);
    errdefer m.gpa.destroy(snap);
    const storage = try m.gpa.alloc(u8, plan.packedBytes());
    errdefer m.gpa.free(storage);
    const live = try m.gpa.alloc(u8, plan.mutableBytes());
    errdefer m.gpa.free(live);
    const taps = try m.gpa.alloc(u8, m.taps.length());
    errdefer m.gpa.free(taps);
    @memset(storage, 0);
    @memset(live, 0);
    @memset(taps, 0);
    snap.* = .{ .storage = storage, .live = live, .taps = taps, .position = 0, .history = @splat(0), .taps_row = 0, .taps_n = 0, .taps_rows_last = 0, .taps_rows_valid = 0, .taps_base = 0 };
    return snap;
}

/// Stands the imported state after `upto` tokens: position, Engram record, and taps without rows.
pub fn seal(m: *const Model, snap: *Snapshot, tokens: []const u32, upto: u32) !void {
    if (upto == 0 or upto > tokens.len) return error.InvalidPrefix;
    if (upto > m.caches[0].plan.capacity) return error.SnapshotOutOfRange;
    const tm = m.workspace.view(.token_map);
    const map = engram.hash.TokenMap{ .bytes = (tm.buffer.contents() + tm.offset)[0 .. engram.hash.raw_vocab * 4] };
    snap.history = try engram.hash.resumeRecord(map, tokens, upto);
    snap.position = upto;
    snap.taps_row = upto % kv.cache.window;
    snap.taps_base = snap.taps_row;
    snap.taps_n = 0;
    snap.taps_rows_last = 0;
    snap.taps_rows_valid = 0;
}
