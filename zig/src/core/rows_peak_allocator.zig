//! Requested live/peak host allocation bytes, excluding allocator bookkeeping and the filesystem page cache.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

pub const Peak = struct {
    child: Allocator,
    live: usize = 0,
    peak: usize = 0,

    pub fn allocator(p: *Peak) Allocator {
        return .{ .ptr = p, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn of(ctx: *anyopaque) *Peak {
        return @ptrCast(@alignCast(ctx));
    }

    fn change(p: *Peak, old: usize, new: usize) void {
        p.live = p.live - old + new;
        p.peak = @max(p.peak, p.live);
    }

    fn alloc(ctx: *anyopaque, n: usize, alignment: Alignment, ra: usize) ?[*]u8 {
        const p = of(ctx);
        const out = p.child.rawAlloc(n, alignment, ra) orelse return null;
        p.change(0, n);
        return out;
    }

    fn resize(ctx: *anyopaque, old: []u8, alignment: Alignment, n: usize, ra: usize) bool {
        const p = of(ctx);
        if (!p.child.rawResize(old, alignment, n, ra)) return false;
        p.change(old.len, n);
        return true;
    }

    fn remap(ctx: *anyopaque, old: []u8, alignment: Alignment, n: usize, ra: usize) ?[*]u8 {
        const p = of(ctx);
        const out = p.child.rawRemap(old, alignment, n, ra) orelse return null;
        p.change(old.len, n);
        return out;
    }

    fn free(ctx: *anyopaque, old: []u8, alignment: Alignment, ra: usize) void {
        const p = of(ctx);
        p.child.rawFree(old, alignment, ra);
        p.change(old.len, 0);
    }
};
