//! 2 x 1M, 4 x 740K push the first chat to NVMe; its resend takes the prompt as history.

const std = @import("std");
const testing = std.testing;
const sessions = @import("sessions");
const sess = @import("sess.zig");

const Pool = sessions.Pool;
const Store = sessions.Store;
const Bounded = sessions.store.Bounded;

/// The pool's families, a few bytes a row: the served token counts and page, its pages small enough for a host test
const fams = [_]sessions.Family{
    .{ .name = "comp.4", .ratio = 4, .row_bytes = 8 },
    .{ .name = "index_k.4", .ratio = 4, .row_bytes = 4 },
};

const Blob = struct {
    gpa: std.mem.Allocator,
    pos: u64,
    const vt: Bounded.VTable = .{ .nbytes = nbytes, .blobs = blobs, .release = release };
    fn make(gpa: std.mem.Allocator, pos: u64) !Bounded {
        const b = try gpa.create(Blob);
        b.* = .{ .gpa = gpa, .pos = pos };
        return .{ .ptr = b, .vtable = &vt };
    }
    fn nbytes(_: *anyopaque) u64 {
        return 64 << 20; // the bounded state of a long chat (rings, carries, stash)
    }
    fn blobs(p: *anyopaque, gpa: std.mem.Allocator) anyerror![]sessions.disk.Blob {
        const b: *Blob = @ptrCast(@alignCast(p));
        const out = try gpa.alloc(sessions.disk.Blob, 1);
        out[0] = .{ .name = try gpa.dupe(u8, "slot"), .bytes = try gpa.dupe(u8, std.mem.asBytes(&b.pos)) };
        return out;
    }
    fn release(p: *anyopaque) void {
        const b: *Blob = @ptrCast(@alignCast(p));
        b.gpa.destroy(b);
    }
};

fn chat(ids: []i32, salt: i32) void {
    for (ids, 0..) |*t, i| t.* = salt * 7_000_000 + @as(i32, @intCast(i));
}

/// A turn under the served admission: spills parked, pages reserved, the prompt saved, the slot free.
fn turn(gpa: std.mem.Allocator, st: *Store, slot: *sessions.Slot, ids: []const i32, tag: u32) !void {
    var spills: std.ArrayList(u32) = .empty;
    defer spills.deinit(gpa);
    const plan = try sess.admit(st, null, ids.len, 512, slot.capacity, &.{}, 0, &spills);
    try testing.expect(!plan.wait);
    for (spills.items) |id| _ = try st.evict(id);
    try st.settleAll();
    slot.reserve(plan.need);
    try slot.ensure(ids.len);
    _ = try st.save(slot, try Blob.make(gpa, ids.len), ids, tag, 0);
    _ = try slot.releaseAll();
    try st.settleAll();
}

test "a 1M chat pushed to NVMe by 4 x 740K resumes with the prompt's ids as its history (a served NotInRam)" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "tier");
    const root = try tmp.dir.realPathFileAlloc(testing.io, "tier", gpa);
    defer gpa.free(root);
    const page = 256;
    var pool = try Pool.init(gpa, .{ .families = &fams, .page = page }, 11_718 * page, 1, 0); // a 3.0M pool
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    const disk = try sessions.Disk.open(gpa, testing.io, .{ .root = root, .compat = @splat(3), .min_tokens = 1024, .chunk = 1 << 20, .staging = 4, .lanes = 2, .direct = false });
    defer disk.close();
    var st = try Store.init(gpa, testing.io, &pool, host.store(), disk, .{ .ram_bytes = 256 << 20, .skip_covered = false, .delta = false });
    defer st.deinit();
    var slots: [4]*sessions.Slot = undefined;
    for (&slots) |*s| s.* = try pool.newSlot(1 << 20);
    const tag = 7;
    const first = try gpa.alloc(i32, 1_000_000);
    defer gpa.free(first);
    chat(first, 1);
    const second = try gpa.alloc(i32, 1_000_000);
    defer gpa.free(second);
    chat(second, 2);
    try turn(gpa, &st, slots[0], first, tag);
    try turn(gpa, &st, slots[1], second, tag);
    const long = try gpa.alloc(i32, 740_000);
    defer gpa.free(long);
    for (0..4) |k| {
        chat(long, @intCast(3 + k));
        try turn(gpa, &st, slots[k], long, tag);
    }
    try testing.expect(st.stats.parks >= 1);
    // the first chat's prompt again, plus a new turn: found on NVMe, its tokens no longer indexed
    const again = try gpa.alloc(i32, 1_000_100);
    defer gpa.free(again);
    @memcpy(again[0..first.len], first);
    for (again[first.len..], 0..) |*t, i| t.* = 900 + @as(i32, @intCast(i));
    const hit = (try st.find(tag, again)).?;
    try testing.expect(!hit.ram and st.entry(hit.id).extra.file);
    try testing.expectEqual(@as(u64, first.len), hit.pos);
    const hist = try gpa.alloc(i32, hit.pos);
    defer gpa.free(hist);
    try testing.expectError(error.NotInRam, st.index.idsOf(hit.id, hist)); // what the restore read before
    // the admission for it: the hit held, the coldest RAM entries spilled, the pages reserved, then the restore
    var spills: std.ArrayList(u32) = .empty;
    defer spills.deinit(gpa);
    const plan = try sess.admit(&st, hit, again.len, 512, slots[0].capacity, &.{hit.id}, 0, &spills);
    try testing.expect(!plan.wait and spills.items.len > 0);
    for (spills.items) |id| _ = try st.evict(id);
    try st.settleAll();
    slots[0].reserve(plan.need);
    const j = try st.beginRestore(hit.id, slots[0]);
    var got = (try st.settle(j)).?;
    defer got.deinit(gpa);
    try st.historyOf(hit.id, again, hist);
    try testing.expectEqualSlices(i32, first, hist);
    // the leader's ids are the entry's: another prompt's are refused, and none at all is the old failure
    try testing.expectError(error.NotTheEntry, st.historyOf(hit.id, second, hist));
    try testing.expectError(error.NotInRam, st.historyOf(hit.id, null, hist));
}
