//! Shared request-start snapshots own their actual prefix pages and survive deferred restore or cancellation.
const std = @import("std");

pub fn active(e: anytype) bool {
    return e.share_pages and @import("cuda_env.zig").requestStarts();
}

/// Only rank 0 owns pool references; followers keep the same ids for the restore's mapping check.
pub fn keep(e: anytype, count: usize) ![]u32 {
    if (!active(e)) return &.{};
    if (count > e.cur.st.pages.len) return error.SnapshotPages;
    const ids = try e.gpa.dupe(u32, e.cur.st.pages[0..count]);
    if (e.share.rank == 0) for (ids) |id| e.pages.?.pool.retain(id);
    return ids;
}

pub fn give(e: anytype, ids: []u32) void {
    if (e.share.rank == 0) for (ids) |id| e.pages.?.pool.release(id);
    e.gpa.free(ids);
}

/// A borrowed snapshot cannot free its pages; otherwise its sole-held prefix page is immediately reclaimable.
pub fn reclaimable(e: anytype, id: u64) bool {
    if (!active(e) or e.share.rank != 0) return false;
    const s = e.snaps.get(id) orelse return false;
    if (s.borrows != 0 or s.retired) return false;
    for (s.shared_ids) |page| if (e.pages.?.pool.holders(page) == 1) return true;
    return false;
}

pub fn pin(e: anytype, id: u64) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    if (s.len == 0 or s.retired) return error.NoKeptState;
    s.borrows = try std.math.add(u32, s.borrows, 1);
}

pub fn unpin(e: anytype, id: u64) void {
    const s = e.snaps.getPtr(id) orelse unreachable;
    std.debug.assert(s.borrows > 0);
    s.borrows -= 1;
    if (s.borrows == 0 and s.retired) e.dropState(id);
}

pub fn release(e: anytype, q: anytype) void {
    const id = q.resume_pin orelse return;
    q.resume_pin = null;
    unpin(e, id);
}

/// Token-identical tree pages may come from another cut schedule; adopt this snapshot's actual bytes instead.
pub fn adopt(e: anytype, id: u64, out: *std.ArrayList(u32)) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    if (s.shared_ids.len != s.shared) return error.SnapshotPages;
    try out.ensureTotalCapacity(e.gpa, s.shared_ids.len);
    for (s.shared_ids) |page| e.pages.?.pool.retain(page);
    e.pages.?.give(out.items);
    out.clearRetainingCapacity();
    out.appendSliceAssumeCapacity(s.shared_ids);
}

/// A restore writes only the private tail, after the original complete pages are mapped on every rank.
pub fn mapped(e: anytype, s: anytype) !void {
    if (!active(e)) return;
    if (s.shared_ids.len != s.shared or s.shared > e.cur.st.pages.len or
        !std.mem.eql(u32, s.shared_ids, e.cur.st.pages[0..s.shared])) return error.SnapshotPages;
}
