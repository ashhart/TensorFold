//! Prompt-sized snapshots on each rank; the existing chunk grid and the live sequence's layout do not change.
const std = @import("std");
const cuda = @import("cuda");
const types = @import("cuda_engine_types.zig");
pub const pages = @import("cuda_snapshot_pages.zig");
pub const layout = @import("cuda_snapshot_layout.zig");

pub const Setup = struct { ptr: *anyopaque, run: *const fn (*anyopaque, u64) anyerror!void };

pub fn prepare(e: anytype, budget: u64) !void {
    if (e.snap_pages != null or e.snaps.count() != 0) return error.SnapshotCacheMade;
    const count = std.math.cast(u32, budget / pages.page) orelse return error.SnapshotBudget;
    try e.snaps.ensureTotalCapacity(e.gpa, count);
    try e.dropped.ensureTotalCapacity(e.gpa, @as(usize, count) + 1);
    e.snap_pages = try pages.Pool.init(e.gpa, e.d, budget);
}

/// A failed allocation still joins the startup agreement, so a peer never waits for a missing participant.
pub fn agree(e: anytype, budget: u64) !void {
    var failed: ?anyerror = null;
    e.prepareSnapshots(budget) catch |err| {
        failed = err;
    };
    if (try e.shared(if (failed != null) 1 else 0) > 0) return failed orelse error.PeerSnapshotCache;
}

pub fn bytes(e: anytype, at: usize) u64 {
    const head: ?usize = if (e.mtp) @min(e.capacity, std.math.add(usize, at, @import("cuda_state.zig").window_rows) catch return std.math.maxInt(u64)) else null;
    // a paged sequence holds its request's rows: its prefix is counted to the capacity, in a window's bytes
    const n = layout.Plan.size(&e.cur.st, &e.c, e.capacity, at, head) catch return std.math.maxInt(u64);
    return pages.rounded(n) catch std.math.maxInt(u64);
}

pub fn keep(e: anytype) !u64 {
    if (e.next_snap > std.math.maxInt(u32)) return error.SnapshotIdLimit;
    const id = e.next_snap;
    try keepAs(e, id);
    e.next_snap += 1;
    e.announce = id;
    return id;
}

pub fn keepAs(e: anytype, id: u64) !void {
    if (e.snaps.contains(id)) return error.StateKept;
    if (e.cur.st.pending != 0 or e.cur.st.mtp_len > e.cur.st.pos + @import("cuda_state.zig").window_rows)
        return error.NotAtMark;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const plan = try layout.Plan.init(&e.cur.st, &e.c, e.cur.st.pos, if (e.mtp) e.cur.st.mtp_len else null);
    var lease = try pool.take(plan.len);
    errdefer {
        cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot copy did not drain");
        pool.give(&lease);
    }
    const cur = try e.gpa.dupe(u1, e.cur.st.cur);
    errdefer e.gpa.free(cur);
    try pool.move(e.stream, &lease, &plan, true);
    try e.stream.synchronize();
    e.snaps.putAssumeCapacityNoClobber(id, .{ .lease = lease, .len = plan.len, .pos = e.cur.st.pos, .mtp_len = e.cur.st.mtp_len, .cur = cur, .bases = .{ e.cur.st.base, e.cur.st.mtp_base } });
}

pub fn restore(e: anytype, id: u64) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const plan = try layout.Plan.init(&e.cur.st, &e.c, s.pos, if (e.mtp) s.mtp_len else null);
    if (plan.len != s.len) return error.SnapshotSize;
    errdefer cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot restore did not drain");
    try pool.move(e.stream, &s.lease, &plan, false);
    try e.cur.st.restoreHost(e.stream, s.pos, s.mtp_len, s.cur, s.bases);
    e.cur.in_head = 0;
    try e.stream.synchronize();
}

pub fn holds(e: anytype, id: u64) bool {
    return e.snaps.contains(id);
}

pub fn drop(e: anytype, id: u64) void {
    var kv = e.snaps.fetchRemove(id) orelse return;
    free(e, &kv.value);
    if (e.share.rank == 0 and e.announce == id) {
        e.announce = 0; // the store failed to publish this keep, so no follower has it yet
        return;
    }
    if (e.share.rank == 0 and e.share.world > 1) e.dropped.appendAssumeCapacity(@intCast(id));
}

pub fn free(e: anytype, s: *types.Snap) void {
    e.snap_pages.?.give(&s.lease);
    e.gpa.free(s.cur);
}

/// Odd words release an evicted id before an even word announces its replacement; an idle gate still sends one word.
pub fn gate(e: anytype) !void {
    const leader = e.share.rank == 0;
    var dropped: usize = 0;
    while (true) {
        const word: u64 = if (!leader) 0 else if (dropped < e.dropped.items.len)
            2 * @as(u64, e.dropped.items[dropped]) + 1
        else
            2 * e.announce;
        const shared = try e.shared(@floatFromInt(word));
        if (!std.math.isFinite(shared) or shared < 0 or shared > 2 * @as(f64, std.math.maxInt(u32)) + 1 or @floor(shared) != shared)
            return error.BadSnapshotControl;
        const got: u64 = @intFromFloat(shared);
        if (got & 1 != 0) {
            if (leader) dropped += 1 else e.dropState(got / 2);
            continue;
        }
        if (leader) {
            e.dropped.clearRetainingCapacity();
            e.announce = 0;
        } else if (got > 0) try e.keepAs(got / 2);
        return;
    }
}

/// After all ranks agree to cancel, publish the last completed state before any rank releases or rebinds its slot.
pub fn interrupted(e: anytype, marks: ?types.Marks, at: u32) !void {
    if (marks) |m| if (m.interrupted) |save_prefix| save_prefix(m.ctx, at);
    try gate(e);
}
