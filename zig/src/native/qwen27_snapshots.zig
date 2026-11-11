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
        const h = host(ptr);
        return if (h.snapshot_pages != null) incremental(ptr, null, at) else q.snapshot.Spares.size(q.snapshot.slotBytes(&h.runner, h.back.ringOf(0), at));
    }
    pub fn charged(_: *anyopaque, saved: pc.Saved) u64 {
        const st: *q.snapshot.State = @ptrCast(@alignCast(saved));
        return st.charged();
    }
    pub fn spare(ptr: *anyopaque) u64 {
        return host(ptr).spares.spare();
    }
    pub fn trim(ptr: *anyopaque, left: u64) void {
        host(ptr).spares.trim(left);
    }
    pub fn reuses(ptr: *anyopaque, at: u32) bool {
        const h = host(ptr);
        return h.snapshot_pages == null and h.spares.fits(q.snapshot.slotBytes(&h.runner, h.back.ringOf(0), at));
    }
    pub fn incremental(ptr: *anyopaque, parent: ?pc.Saved, at: u32) u64 {
        const h = host(ptr);
        const p = h.snapshot_pages orelse return bytes(ptr, at);
        const old: ?*q.snapshot.State = if (parent) |v| @ptrCast(@alignCast(v)) else null;
        return q.snapshot.privateBytes(&h.runner, h.back.ringOf(0), at) + p.pages.added(q.snapshot.imageOf(old, p), at);
    }
    pub fn shared(ptr: *anyopaque) u64 {
        const p = host(ptr).snapshot_pages orelse return 0;
        return p.pages.sharedBytes();
    }
    pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        return saveFrom(ptr, owner, at, null);
    }
    pub fn saveFrom(ptr: *anyopaque, owner: ?*anyopaque, at: u32, parent: ?pc.Saved) anyerror!pc.Saved {
        const h = host(ptr);
        const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const slot = h.back.slotOf(s) orelse return error.UnknownStream;
        // Scratch already occupies its bytes; do not let cache growth spend that same free room again.
        if (h.runner.gdn.marks != null) {
            const need = incremental(ptr, parent, at);
            if (need > room(h.model) -| h.runner.gdn.missingBeforeBytes()) return error.LaneMemory;
        }
        if (h.draft) |d| try d.generation.finishKeep();
        h.spares.follow = h.runner.residency;
        const old: ?*q.snapshot.State = if (parent) |v| @ptrCast(@alignCast(v)) else null;
        const st = try q.snapshot.saveSlotFrom(h.gpa, &h.runner, slot, h.back.ringOf(slot), at, &h.spares, h.snapshot_pages, old);
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
