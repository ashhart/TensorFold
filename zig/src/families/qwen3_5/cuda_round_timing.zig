//! Enabled-only numeric frame receipts join actual CUDA forward buckets to the shared core's accepted paths.
const std = @import("std");
const be = @import("lanes").backend;
pub const Shape = @import("cuda_shapes.zig").Shape;
pub const Trace = struct {
    const Frame = struct { first: usize, windows: usize, offered: usize, bucket: usize, ms: f64 };
    frames: [@import("cuda_round.zig").max_segs]Frame = undefined,
    len: usize = 0,

    pub fn enabled() bool {
        return std.c.getenv("TF_TIMING") != null;
    }

    pub fn create(a: std.mem.Allocator, on: bool) !?*Trace {
        if (!on) return null;
        const trace = try a.create(Trace);
        trace.* = .{};
        return trace;
    }

    pub fn observer(t: *Trace) be.RoundObserver {
        return .{ .ptr = t, .done = done };
    }

    pub fn reset(t: *Trace) void {
        t.len = 0;
    }

    pub fn frame(t: *Trace, first: usize, windows: usize, offered: usize, bucket: usize, ms: f64) void {
        std.debug.assert(t.len < t.frames.len);
        t.frames[t.len] = .{ .first = first, .windows = windows, .offered = offered, .bucket = bucket, .ms = ms };
        t.len += 1;
    }

    fn done(ptr: *anyopaque, paths: []const []const u32) void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        for (t.frames[0..t.len]) |frame_| {
            var kept: usize = 0;
            for (paths[frame_.first..][0..frame_.windows]) |path| kept += path.len;
            std.debug.print("[timing] verify_round offered={d} kept={d} bucket={d} ms={d:.3}\n", .{ frame_.offered, kept, frame_.bucket, frame_.ms });
        }
        t.reset();
    }
};
