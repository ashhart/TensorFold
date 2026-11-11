//! Recording byte copies and lifetimes only; this facade never executes a GPU command.
const std = @import("std");
const base = @import("qwen27_speed_metal.zig");
pub const Buffer = base.Buffer;
pub const Allocations = base.Allocations;
pub const ResidencySet = base.ResidencySet;
pub const ResourceOptions = struct {
    pub const shared = 0;
    pub const private = 1;
    pub const untracked = 2;
};
pub const Submission = struct { waits: usize = 0, releases: usize = 0, failed: bool = false };
pub const Device = struct {
    allocations: *Allocations,
    before: ?*Submission = null,
    pub fn buffer(d: Device, bytes: usize, options: usize) !Buffer {
        if (d.before) |s| if (s.waits == 0) return error.ParentNotLanded;
        return (base.Device{ .allocations = d.allocations }).buffer(bytes, options);
    }
};
pub const CommandBuffer = struct {
    submission: *Submission,
    pub fn wait(c: CommandBuffer) void {
        c.submission.waits += 1;
    }
    pub fn release(c: CommandBuffer) void {
        c.submission.releases += 1;
    }
    pub fn failure(c: CommandBuffer) ?[]const u8 {
        return if (c.submission.failed) "copy failed" else null;
    }
};
pub const BlitEncoder = struct {
    copied: *usize,
    pub fn copy(e: BlitEncoder, source: Buffer, off: usize, dest: Buffer, to: usize, len: usize) void {
        std.debug.assert(off % 4 == 0 and to % 4 == 0 and len % 4 == 0);
        @memcpy(dest.data[to..][0..len], source.data[off..][0..len]);
        e.copied.* += len;
    }
};
