//! The prompt cache's copies of a Flash Next stream (cuda_snap.zig), called with `Loaded.target`.

const std = @import("std");
const lanes = @import("lanes");
const snap = @import("cuda_snap.zig");
const mtp = @import("cuda_mtp.zig");
const prompt = @import("cuda_prompt.zig");

/// Size, save, restore and drop over the lane owner `of` finds behind a target pointer.
pub fn Copies(comptime of: anytype) type {
    return struct {
        pub fn bytes(ptr: *anyopaque, at: u32) u64 {
            const self = of(ptr);
            if (self.engine.page_config) |cfg| return snap.pagedBytes(@min(self.context + mtp.max_rows, prompt.max_tokens), cfg.forChunk(self.engine.chunk), at, self.head != null) catch std.math.maxInt(u64);
            return snap.bytesKv(at, of(ptr).head != null, @import("exl3/kv/cuda_exl3_kv.zig").of(of(ptr).engine).rows());
        }

        /// The owner stream's state, its prompt pass standing at `at` (a drafting stream's with the head's rows).
        pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = try self.lane(s);
            if (l.seq.fed != at) return error.NotAtMark;
            // ours: a drafting stream's state carries the head's rows on rank 0 (cuda_rank_engine.Session.save)
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
