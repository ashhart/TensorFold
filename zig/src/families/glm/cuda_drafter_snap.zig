//! DFlash2's ring in kept states (rank 0's TF_GLM_REQUEST_STARTS, told to every rank): resumes draft as fresh passes.
const std = @import("std");
const dflash = @import("dflash.zig");

/// The ring bytes a kept state adds (cuda_snapshots.bytes): none unless the ranks keep rings and a drafter runs.
pub fn bytes(e: anytype) usize {
    if (comptime !@hasField(@TypeOf(e.*), "drafter")) return 0;
    return if (e.snap_drafter and e.drafter != null) dflash.ring_bytes else 0;
}

/// A keep's plan gains the bound slot's ring; the first position it holds of this sequence (null: no ring kept).
pub fn append(e: anytype, plan: anytype) !?i64 {
    if (comptime !@hasField(@TypeOf(e.*), "drafter")) return null;
    if (!e.snap_drafter or e.drafter == null) return null;
    const from = try span(e, plan);
    // a pass over pass_len ids taps only its last WIN rows (dflash.promptRun): below them the ring holds stale rows
    return @max(from, @as(i64, @intCast(e.pass_len)) - dflash.WIN);
}

/// The ring as a direct span after the target's MLA, indexer and MTP spans; a restore adds it as its keep did.
pub fn span(e: anytype, plan: anytype) !i64 {
    const dr = e.drafter orelse return error.SnapshotShape;
    const s = try dr.snapshotSlot(e.slot);
    if (plan.count == plan.parts.len) return error.SnapshotShape;
    plan.len = try std.math.add(usize, plan.len, dflash.ring_bytes);
    plan.parts[plan.count] = .{ .live = s.ring.?.ptr, .len = dflash.ring_bytes };
    plan.count += 1;
    return s.from;
}

/// After a restore whose state kept a ring: its context origin (a state without one restarts at its position instead).
pub fn restore(e: anytype, from: i64) !void {
    const dr = e.drafter orelse return error.SnapshotShape;
    const s = try dr.snapshotSlot(e.slot);
    s.from = from;
    s.v_rows = 0; // a fresh prompt pass leaves no verify taps to ingest either
}

const Fake = struct {
    const Slot = struct { ring: ?struct { ptr: u64 } = .{ .ptr = 0x1000 }, from: i64 = 7, v_rows: usize = 3 };
    const Drafter = struct {
        state: Slot = .{},
        fn snapshotSlot(d: *@This(), _: u32) !*Slot {
            return &d.state;
        }
    };
    const Part = struct { live: u64, len: usize };
    const Plan = struct { parts: [3]Part = undefined, count: usize = 0, len: usize = 123 };
    drafter: ?*Drafter,
    slot: u32 = 2,
    snap_drafter: bool = false,
    pass_len: usize = 100,
};

test "GLM warm turns: snapshots: rank 0's flag adds the ring once, from where the pass tapped, and a restore takes it" {
    var d: Fake.Drafter = .{};
    var e: Fake = .{ .drafter = &d };
    var plan: Fake.Plan = .{};
    try std.testing.expectEqual(@as(usize, 0), bytes(&e)); // no flag: main's states (a resume restarts the context)
    try std.testing.expectEqual(@as(?i64, null), try append(&e, &plan));
    try std.testing.expectEqual(@as(usize, 0), plan.count);
    e.snap_drafter = true;
    try std.testing.expectEqual(@as(usize, dflash.ring_bytes), bytes(&e));
    const from = try append(&e, &plan); // a short pass tapped every row from its origin
    try std.testing.expectEqual(@as(?i64, 7), from);
    try std.testing.expectEqual(@as(usize, 123 + dflash.ring_bytes), plan.len);
    try std.testing.expectEqual(@as(usize, 1), plan.count);
    try std.testing.expectEqual(@as(u64, 0x1000), plan.parts[0].live);
    e.pass_len = 5000; // a long pass tapped from 5000 - WIN only
    try std.testing.expectEqual(@as(?i64, 5000 - dflash.WIN), try append(&e, &plan));
    d.state = .{ .from = 99 };
    try restore(&e, from.?);
    try std.testing.expectEqual(@as(i64, 7), d.state.from);
    try std.testing.expectEqual(@as(usize, 0), d.state.v_rows);
    plan.count = plan.parts.len;
    try std.testing.expectError(error.SnapshotShape, append(&e, &plan));
    e.drafter = null;
    try std.testing.expectEqual(@as(usize, 0), bytes(&e));
    try std.testing.expectEqual(@as(?i64, null), try append(&e, &plan));
    try std.testing.expectError(error.SnapshotShape, span(&e, &plan));
    try std.testing.expectError(error.SnapshotShape, restore(&e, 7));
}
