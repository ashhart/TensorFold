// Portable RAM resume exercises the actual page pool, partial-page copy and bounded owner.
const std = @import("std");
const sessions = @import("sessions");
const t = std.testing;
const fams = [_]sessions.Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 4 },
    .{ .name = "index_k.2", .ratio = 2, .row_bytes = 2 },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 4 },
};
const Bound = struct {
    ring: [32]u8,
    carry: [8]u8,
    released: bool = false,
    fn bytes(_: *anyopaque) u64 {
        return 40;
    }
    fn blobs(_: *anyopaque, _: std.mem.Allocator) ![]sessions.disk.Blob {
        return error.Unused;
    }
    fn release(ptr: *anyopaque) void {
        @as(*Bound, @ptrCast(@alignCast(ptr))).released = true;
    }
    fn bounded(b: *Bound) sessions.store.Bounded {
        return .{ .ptr = b, .vtable = &.{ .nbytes = bytes, .blobs = blobs, .release = release } };
    }
};
fn row(h: *sessions.HostPool, slot: *sessions.Slot, fam: u32, pos: u32) []u8 {
    const f = fams[fam];
    const pg = slot.tableAt(pos / h.page);
    const off = (pos % h.page) / f.ratio * f.row_bytes;
    return h.pageOf(fam, pg)[off..][0..f.row_bytes];
}
fn fill(h: *sessions.HostPool, slot: *sessions.Slot, from: u32, end: u32) void {
    for (fams, 0..) |f, fi| {
        var pos = (from + f.ratio - 1) / f.ratio * f.ratio;
        while (pos < end) : (pos += f.ratio) for (row(h, slot, @intCast(fi), pos), 0..) |*b, j| {
            b.* = @truncate(pos * 7 + fi * 31 + j);
        };
    }
}
test "Snapshot state: actual RAM session resumes full and partial pages with separate ring and carry ownership" {
    const a = t.allocator;
    for ([_]u32{ 1, 7, 8, 9, 15, 16, 17 }) |seam| {
        var pool = try sessions.Pool.init(a, .{ .families = &fams, .page = 8 }, 128, 1, 0);
        defer pool.deinit();
        var host = try sessions.HostPool.init(a, &pool);
        defer host.deinit();
        var store = try sessions.Store.init(a, t.io, &pool, host.store(), null, .{});
        defer store.deinit();
        const live = try pool.newSlot(48);
        const resumed = try pool.newSlot(48);
        const fresh = try pool.newSlot(48);
        try live.ensure(seam);
        fill(&host, live, 0, seam);
        var snapshot: Bound = .{ .ring = undefined, .carry = undefined };
        for (&snapshot.ring, 0..) |*b, j| b.* = @truncate(seam + j);
        for (&snapshot.carry, 0..) |*b, j| b.* = @truncate(seam * 3 + j);
        var ids: [40]i32 = undefined;
        for (&ids, 0..) |*id, j| id.* = @intCast(j + 1);
        const id = (try store.save(live, snapshot.bounded(), ids[0..seam], 41, 0)).?;
        _ = try live.releaseAll();
        const restored = try store.restoreRam(id, resumed);
        const bounded: *Bound = @ptrCast(@alignCast(restored.ptr));
        try t.expectEqualSlices(u8, &snapshot.ring, &bounded.ring);
        try t.expectEqualSlices(u8, &snapshot.carry, &bounded.carry);
        try resumed.ensure(40);
        try fresh.ensure(40);
        fill(&host, resumed, seam, 40);
        fill(&host, fresh, 0, 40);
        for (fams, 0..) |f, fi| {
            var pos: u32 = 0;
            while (pos < 40) : (pos += f.ratio) try t.expectEqualSlices(u8, row(&host, fresh, @intCast(fi), pos), row(&host, resumed, @intCast(fi), pos));
        }
        if (seam % 8 != 0) try t.expect(live.len == 0 and resumed.tableAt(seam / 8) != store.entry(id).extra.pages[seam / 8]);
        _ = try resumed.releaseAll();
        _ = try fresh.releaseAll();
        try store.drop(id);
        try t.expect(snapshot.released);
    }
}
