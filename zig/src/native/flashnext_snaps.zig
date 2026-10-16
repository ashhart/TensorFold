//! Flash Next's prompt-cache copies of the engine's state, made and restored on the engine's thread.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fx = tf.flashnext_engine;
const snap = tf.flashnext_snapshot;
const learned = tf.flashnext_learned;
const pc = api.prompt_cache;
const disk = @import("flashnext_disk.zig");
const Host = @import("flashnext_host.zig").Host;

/// The prompt cache's copies of the engine's state (snapshot.zig), on the engine's thread.
pub const Snaps = struct {
    pub fn bytes(ptr: *anyopaque, at: u32) u64 {
        const h: *Host = @ptrCast(@alignCast(ptr));
        return h.eng.snap_pool.size(snap.bytes(at)); // a new buffer's size; a free one in the pool may take the state instead
    }
    pub fn charged(_: *anyopaque, saved: pc.Saved) u64 {
        const k: *Kept = @ptrCast(@alignCast(saved));
        return k.st.cap;
    }
    pub fn spare(ptr: *anyopaque) u64 {
        const h: *Host = @ptrCast(@alignCast(ptr));
        return h.eng.snap_pool.spare();
    }
    pub fn trim(ptr: *anyopaque, room: u64) void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        h.eng.snap_pool.trim(room);
    }
    pub fn reuses(ptr: *anyopaque, at: u32) bool {
        const h: *Host = @ptrCast(@alignCast(ptr));
        return h.eng.snap_pool.fits(snap.bytes(at));
    }
    /// A kept state and its name on both Macs (speed-up mode's rank 1 keeps its own under the same name).
    const Kept = disk.Kept;
    pub fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const h: *Host = @ptrCast(@alignCast(ptr));
        const k = try h.gpa.create(Kept);
        errdefer h.gpa.destroy(k);
        k.* = .{ .st = try snap.save(h.eng, h.gpa, at), .key = fx.keyOf(h.prompt[0 .. at + 1]) };
        if (std.mem.indexOfScalar(u64, h.eng.peer_drops.items, k.key)) |i| _ = h.eng.peer_drops.swapRemove(i); // rank 1's copy is this new one
        return k;
    }
    pub fn restore(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        const k: *Kept = @ptrCast(@alignCast(saved));
        try snap.restore(h.eng, k.st);
    }
    pub fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        const k: *Kept = @ptrCast(@alignCast(saved));
        if (h.eng.r.tp != null) h.eng.peer_drops.append(h.gpa, k.key) catch {};
        snap.drop(h.gpa, k.st);
        h.gpa.destroy(k);
    }
    /// --learn: a kept state to its file under `dir`.
    pub fn write(_: *anyopaque, saved: pc.Saved, dir: [:0]const u8, key: u64) anyerror!void {
        const k: *Kept = @ptrCast(@alignCast(saved));
        var buf: [1100]u8 = undefined;
        try learned.writeFile(k.st, try learned.path(&buf, dir, key));
    }
    /// --learn: learned state `key` read back into a pool buffer, named by the request's tokens as a saved one is.
    pub fn read(ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!pc.Saved {
        const h: *Host = @ptrCast(@alignCast(ptr));
        if (at >= h.prompt.len) return error.SnapshotRead;
        var buf: [1100]u8 = undefined;
        const k = try h.gpa.create(Kept);
        errdefer h.gpa.destroy(k);
        k.* = .{ .st = try learned.readFile(h.eng, h.gpa, at, try learned.path(&buf, dir, key)), .key = fx.keyOf(h.prompt[0 .. at + 1]) };
        return k;
    }
    /// --learn: learned state `key`'s file removed.
    pub fn forget(_: *anyopaque, dir: [:0]const u8, key: u64) void {
        var buf: [1100]u8 = undefined;
        _ = std.c.unlink(learned.path(&buf, dir, key) catch return);
    }
};
