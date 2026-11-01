//! The prompt cache's copies of a Flash Next lane's state (cuda_snap.zig), called with the lane namespace's target.
const std = @import("std");
const lanes = @import("lanes");
const snap = @import("cuda_snap.zig");
const parts = @import("cuda_batch_state.zig");

pub fn Snaps(comptime Cuda: type) type {
    return struct {
        /// Prompt cache marks inside a chunk kept without cutting the pass (TF_FLASHNEXT_KEEP_INPASS=0: cut there).
        pub fn keepInPass() bool {
            const v = std.c.getenv("TF_FLASHNEXT_KEEP_INPASS") orelse return true;
            return !std.mem.eql(u8, std.mem.span(v), "0");
        }

        /// The n-gram history (the last two tokens) at position `to` of a pass from `from` that began at `old`.
        pub fn historyAt(old: [2]i64, ids: []const u32, from: usize, to: usize) [2]i64 {
            if (to - from >= 2) return .{ ids[to - 2], ids[to - 1] };
            if (to - from == 1) return .{ old[1], ids[to - 1] };
            return old;
        }

        /// The chunk's stop: the first cache mark in [at, stop0) it cannot keep on the way; kept ones into `marks`.
        pub fn cut(self: *Cuda, s: *lanes.Stream, l: anytype, ids: []const u32, at: usize, stop0: usize, marks: *[parts.max_marks]parts.Mark) struct { stop: usize, kept: usize } {
            var n: usize = 0;
            const inpass = keepInPass() and self.ranks == null and l.seq.media == null and @import("cuda_gdn_fused.zig").on();
            for (s.reuse.marks) |m| if (m > at and m < stop0) {
                if (inpass and n < parts.max_marks) {
                    if (snap.prepare(self.gpa, self.engine.driver, m, historyAt(l.seq.history, ids, at, m), l.draft != null)) |p| {
                        l.prepared[n] = p;
                        marks[n] = .{ .at = m, .base = p.buf.ptr };
                        n += 1;
                        continue;
                    } else |_| {} // no room for it now: the pass stops at the mark as before
                }
                return .{ .stop = m, .kept = n };
            };
            return .{ .stop = stop0, .kept = n };
        }

        /// After the pass: each kept state finished (its attention rows), then offered to the cache (save takes it).
        pub fn handOver(self: *Cuda, s: *lanes.Stream, l: anytype, marks: []const parts.Mark) !void {
            for (l.prepared[0..marks.len]) |p| try snap.finish(self.stream, &l.seq, if (l.draft) |*d| d else null, p.?);
            try self.stream.synchronize();
            if (s.reuse.hook) |k| for (marks) |m| k.at(k.ptr, s, @intCast(m.at));
        }

        /// The kept states a chunk prepared that the cache did not take.
        pub fn dropPrepared(self: *Cuda, l: anytype) void {
            for (&l.prepared) |*p| if (p.*) |x| {
                snap.drop(self.gpa, self.stream, x);
                p.* = null;
            };
        }

        fn of(ptr: *anyopaque) *Cuda {
            return @ptrCast(@alignCast(ptr));
        }

        pub fn bytes(ptr: *anyopaque, at: u32) u64 {
            return snap.bytes(at, of(ptr).head != null);
        }

        /// The owner stream's state, its prompt pass standing at `at` (a drafting stream's with the head's rows).
        pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = try self.lane(s);
            // a state its chunk kept on the way (TF_FLASHNEXT_KEEP_INPASS): the cache takes it as it is
            for (&l.prepared) |*p| if (p.*) |x| if (x.at == at) {
                p.* = null;
                return x;
            };
            if (l.seq.fed != at) return error.NotAtMark;
            if (self.ranks) |r| {
                if (l.draft != null) return error.TensorRankDraftSnapshotUnavailable;
                return try r.save(&l.seq);
            }
            // pending DeltaNet rows into the states first: a snapshot holds them as they stand
            if (l.draft) |*d| try self.ops().flushLate(&l.seq, d);
            return try snap.save(self.gpa, self.engine.driver, self.stream, &l.seq, if (l.draft) |*d| d else null);
        }

        pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = try self.lane(s);
            if (self.ranks) |r| {
                if (l.draft != null) return error.NoHeadState;
                return r.restore(&l.seq, @ptrCast(@alignCast(saved)));
            }
            try snap.restore(self.stream, &l.seq, if (l.draft) |*d| d else null, @ptrCast(@alignCast(saved)));
        }

        pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
            const self = of(ptr);
            if (self.ranks) |r| return r.dropSaved(@ptrCast(@alignCast(saved)));
            snap.drop(self.gpa, self.stream, @ptrCast(@alignCast(saved)));
        }
    };
}
