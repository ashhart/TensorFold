//! The process closes an engine before releasing that reload's scratch owner.
const std = @import("std");
const api = @import("engine_api");

pub const Closer = struct {
    opened: api.Opened,
    closed: bool = false,
    scratch: ?*std.heap.ArenaAllocator = null,
    pub fn adopt(c: *Closer, opened: api.Opened, scratch: *std.heap.ArenaAllocator) void {
        std.debug.assert(c.closed);
        if (c.scratch) |a| destroyScratch(a);
        c.opened = opened;
        c.scratch = scratch;
        c.closed = false;
    }
    pub fn closeOnce(c: *Closer) void {
        if (c.closed) return;
        c.closed = true;
        c.opened.close(c.opened.ctx);
    }
    /// Final process teardown follows server teardown, after the last parked metadata reader is gone.
    pub fn deinit(c: *Closer) void {
        c.closeOnce();
        if (c.scratch) |a| destroyScratch(a);
        c.scratch = null;
    }
    pub fn close(p: *anyopaque) void {
        const c: *Closer = @ptrCast(@alignCast(p));
        c.closeOnce();
    }
    pub fn halt(p: *anyopaque, reason: []const u8) void {
        const c: *Closer = @ptrCast(@alignCast(p));
        if (c.closed) return;
        if (comptime @hasField(api.Opened, "halt")) {
            if (c.opened.halt) |f| f(c.opened.ctx, reason);
        }
    }
};

/// An arena's stable heap header stays alive as long as any allocator it handed the engine.
pub fn destroyScratch(a: *std.heap.ArenaAllocator) void {
    const gpa = a.child_allocator;
    a.deinit();
    gpa.destroy(a);
}
