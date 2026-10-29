//! Shared prompt-cache hooks retain the actual native state and its final logits on the same inference queue.
const std = @import("std");
pub fn Hooks(comptime A: type, comptime PC: type) type {
    return struct {
        const H = @This();
        const Snap = A.Snap;
        adapter: *A,
        retained: ?*Snap = null,
        fn retain(h: *H, s: *Snap) void {
            s.retained_next = h.retained;
            h.retained = s;
            h.adapter.e.fail();
        }
        pub fn reclaimDrained(h: *H) void {
            while (h.retained) |s| {
                h.retained = s.retained_next;
                s.deinitDrained();
                h.adapter.a.destroy(s);
            }
        }
        pub fn snapshots(h: *H) PC.Snapshots {
            return .{ .ptr = h, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .charged = charged } };
        }
        fn self(p: *anyopaque) *H {
            return @ptrCast(@alignCast(p));
        }
        fn bytes(p: *anyopaque, at: u32) u64 {
            const l = self(p).adapter;
            if (at == 0 or at > l.e.o.capacity) return std.math.maxInt(u64);
            var state = l.e.slots[0].state;
            state.pos = at;
            return Snap.charge(&state) catch std.math.maxInt(u64);
        }
        fn save(p: *anyopaque, owner: ?*anyopaque, at: u32) !*anyopaque {
            const l = self(p).adapter;
            const stream: *A.Stream = @ptrCast(@alignCast(owner orelse return error.Qwen4SnapshotOwner));
            const lane = try l.find(stream);
            const slot = lane.slot.?;
            if (!l.e.healthy()) return error.Qwen4EngineFailed;
            if (!slot.has_logits or slot.state.pos != at) return error.Qwen4SnapshotPhase;
            const snap = try l.a.create(Snap);
            snap.* = Snap.open(l.a, &slot.state, l.key, bytes(p, at)) catch |err| {
                l.a.destroy(snap);
                return err;
            };
            errdefer self(p).retain(snap);
            try l.e.gpu.begin();
            errdefer l.e.gpu.discard();
            try snap.captureWithLogits(&slot.state, l.key, l.last(lane));
            try l.e.gpu.submit();
            // The cache publishes only completed states; failures leave the destination with this owner.
            try l.e.backend.drain();
            return snap;
        }
        fn restore(p: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) !void {
            const l = self(p).adapter;
            const stream: *A.Stream = @ptrCast(@alignCast(owner orelse return error.Qwen4SnapshotOwner));
            try l.restore(try l.find(stream), @ptrCast(@alignCast(saved)));
        }
        fn charged(_: *anyopaque, saved: *anyopaque) u64 {
            const snap: *Snap = @ptrCast(@alignCast(saved));
            return snap.charged;
        }
        fn drop(p: *anyopaque, saved: *anyopaque) void {
            const l = self(p).adapter;
            const snap: *Snap = @ptrCast(@alignCast(saved));
            snap.close() catch {
                self(p).retain(snap);
                return;
            };
            l.a.destroy(snap);
        }
    };
}
