//! The prompt cache's copies of a Flash Next lane's state (cuda_snap.zig), called with the lane namespace's target.
const lanes = @import("lanes");
const snap = @import("cuda_snap.zig");

pub fn Snaps(comptime Cuda: type) type {
    return struct {
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
