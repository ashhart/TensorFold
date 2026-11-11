//! The prompt cache's copies of a Flash Next lane's state (cuda_snap.zig), called with the lane namespace's target.
const std = @import("std");
const lanes = @import("lanes");
const snap = @import("cuda_snap.zig");
const parts = @import("cuda_batch_state.zig");

/// Marks inside a chunk kept without cutting the pass (TF_FLASHNEXT_KEEP_INPASS=0: cut there), read once at load.
pub fn keepInPass() bool {
    // the startup check refused every malformed value already, so a plain default stands here
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_KEEP_INPASS", true) catch true;
}

/// Size, save, restore and drop over the lane owner `of` finds behind a target pointer.
pub fn Copies(comptime of: anytype) type {
    return struct {
        /// The n-gram history (the last two tokens) at position `to` of a pass from `from` that began at `old`.
        pub fn historyAt(old: [2]i64, ids: []const u32, from: usize, to: usize) [2]i64 {
            if (to - from >= 2) return .{ ids[to - 2], ids[to - 1] };
            if (to - from == 1) return .{ old[1], ids[to - 1] };
            return old;
        }

        /// The chunk's stop: the first cache mark in [at, stop0) it cannot keep on the way; kept ones into `marks`.
        pub fn cut(self: anytype, s: *lanes.Stream, l: anytype, ids: []const u32, at: usize, stop0: usize, marks: *[parts.max_marks]parts.Mark) struct { stop: usize, kept: usize } {
            var n: usize = 0;
            const forms = self.engine.batch_k.forms; // TF_FLASHNEXT_KEEP_INPASS and _GDN_FUSED as the load read them
            const inpass = forms.in_pass and self.engine.store.exl3_ops == null and self.ranks == null and l.seq.media == null and forms.gdn_fused;
            for (s.reuse.marks) |m| if (m > at and m < stop0) {
                if (inpass and n < parts.max_marks) {
                    if (snap.prepare(self.gpa, self.engine.driver, m, historyAt(l.seq.history, ids, at, m), l.draft != null, l.seq.heads)) |p| {
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
        pub fn handOver(self: anytype, s: *lanes.Stream, l: anytype, marks: []const parts.Mark) !void {
            for (l.prepared[0..marks.len]) |p| try snap.finish(self.stream, &l.seq, if (l.draft) |*d| d else null, p.?);
            try self.stream.synchronize();
            if (s.reuse.hook) |k| for (marks) |m| k.at(k.ptr, s, @intCast(m.at));
        }

        /// The kept states a chunk prepared that the cache did not take.
        pub fn dropPrepared(self: anytype, l: anytype) void {
            for (&l.prepared) |*p| if (p.*) |x| {
                snap.drop(self.gpa, self.stream, x);
                p.* = null;
            };
        }

        pub fn bytes(ptr: *anyopaque, at: u32) u64 {
            return snap.bytes(at, of(ptr).head != null, of(ptr).engine.heads);
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

            // pending DeltaNet rows into the states first: a snapshot holds them as they stand
            if (l.draft) |*d| try self.ops().flushLate(&l.seq, d);
            if (self.ranks) |r| return try r.save(&l.seq, if (l.draft) |*d| d else null);
            return try snap.save(self.gpa, self.engine.driver, self.stream, &l.seq, if (l.draft) |*d| d else null);
        }

        pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = self.lane(s) catch |err| {
                std.log.warn("flashnext: a kept state was not restored: no lane ({t})", .{err});
                return err;
            };
            if (self.ranks) |r| return r.restore(&l.seq, if (l.draft) |*d| d else null, @ptrCast(@alignCast(saved)));
            self.engine.round_graph.clear();
            snap.restore(self.stream, &l.seq, if (l.draft) |*d| d else null, @ptrCast(@alignCast(saved))) catch |err| {
                std.log.warn("flashnext: a kept state was not restored ({t})", .{err});
                return err;
            };
        }

        pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
            const self = of(ptr);
            if (self.ranks) |r| return r.dropSaved(@ptrCast(@alignCast(saved)));
            snap.drop(self.gpa, self.stream, @ptrCast(@alignCast(saved)));
        }
    };
}

/// Restore the stream's kept prompt state when its cache mark can be reused.
pub fn reuse(self: anytype, l: anytype, s: *lanes.Stream, ids: []const u32) !void {
    s.cached = 0;
    const saved = s.reuse.saved orelse return;
    const kept: *snap.Snap = @ptrCast(@alignCast(saved));
    if (kept.at != s.reuse.at or kept.at == 0 or kept.at >= ids.len) {
        std.log.warn("flashnext: kept state at {d} not restored for reuse at {d} of {d} ids", .{ kept.at, s.reuse.at, ids.len });
        s.reuse_failed = true;
    } else if (if (self.ranks) |r| r.restore(&l.seq, if (l.draft) |*d| d else null, kept) else snap.restore(self.stream, &l.seq, if (l.draft) |*d| d else null, kept)) |_| {
        l.filled = kept.at;
        s.cached = @intCast(kept.at);
    } else |err| switch (err) {
        error.SequenceTooShort, error.NoHeadState => {
            std.log.warn("flashnext: kept state at {d} not restored ({t})", .{ kept.at, err });
            s.reuse_failed = true;
        },
        else => return err,
    }
}
