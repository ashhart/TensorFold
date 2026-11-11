//! Qwen27 snapshot-cache adapter and its independent immutable KV page store.
const pc = @import("engine_api").prompt_cache;
const q = @import("tensorfold").qwen27;
const lanes = @import("tensorfold").lanes;
const served = @import("qwen27_served.zig");
const Host = served.Host;
const room = served.room;
pub const Snaps = struct {
    pub fn host(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    pub fn bytes(ptr: *anyopaque, at: u32) u64 {
        return incremental(ptr, null, at);
    }
    pub fn charged(_: *anyopaque, saved: pc.Saved) u64 {
        const st: *q.snapshot.State = @ptrCast(@alignCast(saved));
        return st.charged();
    }
    pub fn canShare(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved, at: u32) bool {
        const h = host(ptr);
        const stream: *lanes.Stream = @ptrCast(@alignCast(owner orelse return false));
        const slot = h.back.slotOf(stream) orelse return false;
        const st: *q.snapshot.State = @ptrCast(@alignCast(saved));
        const p = st.paged orelse return false;
        return p.store == h.snapshot_pages.? and !st.failed and h.runner.snapshot_lineage[slot].allows(st.provenance, st.at, at);
    }
    pub fn incremental(ptr: *anyopaque, parent: ?pc.Saved, at: u32) u64 {
        const h = host(ptr);
        const p = h.snapshot_pages.?;
        const old: ?*q.snapshot.State = if (parent) |v| @ptrCast(@alignCast(v)) else null;
        return q.snapshot.privateBytes(&h.runner, h.back.ringOf(0), at) + p.pages.added(q.snapshot.imageOf(old, p), at);
    }
    pub fn shared(ptr: *anyopaque) u64 {
        const p = host(ptr).snapshot_pages.?;
        return p.pages.sharedBytes();
    }
    pub fn sharedFreed(ptr: *anyopaque, set: pc.SavedSet) u64 {
        const store = host(ptr).snapshot_pages.?;
        const C = struct {
            set: pc.SavedSet,
            store: *q.snapshot.page_store.Store,
            fn image(c: @This(), i: usize) ?q.snapshot.page_store.Image {
                const st: *q.snapshot.State = @ptrCast(@alignCast(c.set.get(c.set.ctx, i) orelse return null));
                return q.snapshot.imageOf(st, c.store);
            }
        };
        return store.pages.sharedFreed(C{ .set = set, .store = store }, set.len, C.image);
    }
    pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        return saveFrom(ptr, owner, at, null);
    }
    pub fn saveFrom(ptr: *anyopaque, owner: ?*anyopaque, at: u32, parent: ?pc.Saved) anyerror!pc.Saved {
        const h = host(ptr);
        if (parent) |p| if (!canShare(ptr, owner, p, at)) return error.SnapshotLineage;
        const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const slot = h.back.slotOf(s) orelse return error.UnknownStream;
        // Scratch already occupies its bytes; do not let cache growth spend that same free room again.
        if (h.runner.gdn.marks != null) {
            const need = incremental(ptr, parent, at);
            if (need > room(h.model) -| h.runner.gdn.missingBeforeBytes()) return error.LaneMemory;
        }
        if (h.draft) |d| try d.generation.finishKeep();
        const old: ?*q.snapshot.State = if (parent) |v| @ptrCast(@alignCast(v)) else null;
        const st = try q.snapshot.saveSlotFrom(h.gpa, &h.runner, slot, h.back.ringOf(slot), at, null, h.snapshot_pages, old);
        return @ptrCast(st);
    }
    pub fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.RestoredInPrefill; // the backend restores a stream's kept state in its prompt pass
    }
    pub fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const h = host(ptr);
        const st: *q.snapshot.State = @ptrCast(@alignCast(saved));
        q.snapshot.drop(h.gpa, st);
    }
};

test "qwen27 share: actual native owner mapping and lineage reject other slots, stale runs and flat states" {
    const std = @import("std");
    var h: Host = undefined;
    h.runner.snapshot_ids = .{};
    h.runner.snapshot_lineage = @splat(.{});
    var back: @TypeOf(h.back.*) = undefined;
    back.owners = @splat(null);
    back.runner = &h.runner;
    h.runner.slots = 2;
    h.back = &back;
    var left: lanes.Stream = undefined;
    var right: lanes.Stream = undefined;
    back.owners[0] = &left;
    back.owners[1] = &right;
    var store: q.snapshot.page_store.Store = undefined;
    h.snapshot_pages = &store;
    h.runner.snapshot_lineage[0].fresh(&h.runner.snapshot_ids);
    h.runner.snapshot_lineage[1].fresh(&h.runner.snapshot_ids);
    const first = h.runner.snapshot_lineage[0].stamp(&h.runner.snapshot_ids);
    const sibling = h.runner.snapshot_lineage[0].stamp(&h.runner.snapshot_ids);
    var st: q.snapshot.State = .{ .buf = undefined, .at = 4, .bytes = 16, .draft_end = 0, .provenance = first, .paged = .{ .store = &store, .image = .{ .ids = &.{}, .at = 4, .shared_pages = 0 }, .layout = .{ .fixed = 0, .groups = 1, .row = 4, .tokens = 256 } } };
    try std.testing.expect(Snaps.canShare(&h, &left, &st, 8));
    try std.testing.expect(!Snaps.canShare(&h, &right, &st, 8));
    try std.testing.expect(!Snaps.canShare(&h, null, &st, 8));
    h.runner.snapshot_lineage[0].fresh(&h.runner.snapshot_ids);
    try std.testing.expect(!Snaps.canShare(&h, &left, &st, 8));
    h.runner.snapshot_lineage[0].restoredFrom(&h.runner.snapshot_ids, first, 4);
    try std.testing.expect(Snaps.canShare(&h, &left, &st, 8));
    st.provenance = sibling;
    try std.testing.expect(!Snaps.canShare(&h, &left, &st, 8));
    st.provenance = first;
    st.paged = null;
    try std.testing.expect(!Snaps.canShare(&h, &left, &st, 8));
}
