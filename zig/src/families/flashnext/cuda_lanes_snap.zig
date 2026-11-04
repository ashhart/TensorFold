//! The prompt cache's copies of a Flash Next stream (cuda_snap.zig), called with `Loaded.target`.

const std = @import("std");
const lanes = @import("lanes");
const snap = @import("cuda_snap.zig");

/// Size, save, restore and drop over the lane owner `of` finds behind a target pointer.
pub fn Copies(comptime of: anytype) type {
    return struct {
        pub fn bytes(ptr: *anyopaque, at: u32) u64 {
            return snap.bytes(at, of(ptr).head != null);
        }

        /// The owner stream's state, its prompt pass standing at `at` (a drafting stream's with the head's rows).
        pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = try self.lane(s);
            if (l.seq.fed != at) return error.NotAtMark;
            if (l.draft) |*d| try self.ops().flushStash(d); // the head's deferred rounds are part of its rows
            if (self.ranks) |r| {
                if (l.draft != null and !self.exl3) return error.TensorRankDraftSnapshotUnavailable;
                // an EXL3 pack's state carries the head's rows on rank 0
                return try r.save(&l.seq, if (l.draft) |*d| d else null);
            }
            // pending DeltaNet rows into the states first: a snapshot holds them as they stand
            if (l.draft) |*d| try self.ops().flushLate(&l.seq, d);
            return try snap.save(self.gpa, self.engine.driver, self.stream, &l.seq, if (l.draft) |*d| d else null);
        }

        pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = self.lane(s) catch |err| {
                std.log.warn("flashnext: a kept state was not restored: no lane ({t})", .{err});
                return err;
            };
            if (self.ranks) |r| {
                if (l.draft != null and !self.exl3) return error.NoHeadState;
                return r.restore(&l.seq, if (l.draft) |*d| d else null, @ptrCast(@alignCast(saved)));
            }
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
    } else if (self.ranks != null and l.draft != null and !self.exl3) {
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
