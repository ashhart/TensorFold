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
pub const objc = struct {
    pub const Id = usize;
    pub const Pool = struct {
        pub fn push() @This() {
            return .{};
        }
        pub fn pop(_: @This()) void {}
    };
};
pub const Submission = struct { waits: usize = 0, releases: usize = 0, commits: usize = 0, failed: bool = false };
pub const Queue = struct {
    submission: *Submission,
    copied: *usize,
    before: ?*Submission = null,
    pub fn commandBuffer(q: Queue) CommandBuffer {
        return .{ .submission = q.submission, .copied = q.copied, .before = q.before };
    }
};
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
    copied: ?*usize = null,
    before: ?*Submission = null,
    pub fn blit(c: CommandBuffer) BlitEncoder {
        if (c.before) |b| std.debug.assert(b.waits > 0);
        return .{ .copied = c.copied.? };
    }
    pub fn commit(c: CommandBuffer) void {
        c.submission.commits += 1;
    }
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
    pub fn end(_: BlitEncoder) void {}
    pub fn copy(e: BlitEncoder, source: Buffer, off: usize, dest: Buffer, to: usize, len: usize) void {
        std.debug.assert(off % 4 == 0 and to % 4 == 0 and len % 4 == 0);
        @memcpy(dest.data[to..][0..len], source.data[off..][0..len]);
        e.copied.* += len;
    }
};
