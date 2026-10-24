//! An arena that asks its child for each allocation's exact size: no node grown 1.5 times past a large array.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

/// Every allocation goes to the child as asked and is recorded; deinit frees them all (an arena's one free).
pub const Exact = struct {
    child: Allocator,
    held: std.ArrayList(Block) = .empty,

    pub const Block = struct { ptr: [*]u8, len: usize, alignment: Alignment };

    pub fn init(child: Allocator) Exact {
        return .{ .child = child };
    }

    pub fn allocator(e: *Exact) Allocator {
        return .{ .ptr = e, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = Allocator.noRemap, .free = free } };
    }

    pub fn deinit(e: *Exact) void {
        for (e.held.items) |b| e.child.rawFree(b.ptr[0..b.len], b.alignment, @returnAddress());
        e.held.deinit(e.child);
    }

    /// The bytes asked for, the record list's own included (what a counting allocator under the child sees).
    pub fn bytes(e: *const Exact) usize {
        var n: usize = e.held.capacity * @sizeOf(Block);
        for (e.held.items) |b| n += b.len;
        return n;
    }

    fn self(ctx: *anyopaque) *Exact {
        return @ptrCast(@alignCast(ctx));
    }

    fn find(e: *Exact, ptr: [*]u8) ?usize {
        var k = e.held.items.len;
        while (k > 0) {
            k -= 1;
            if (e.held.items[k].ptr == ptr) return k;
        }
        return null;
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: Alignment, ret: usize) ?[*]u8 {
        const e = self(ctx);
        e.held.ensureUnusedCapacity(e.child, 1) catch return null;
        const p = e.child.rawAlloc(len, alignment, ret) orelse return null;
        e.held.appendAssumeCapacity(.{ .ptr = p, .len = len, .alignment = alignment });
        return p;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret: usize) bool {
        const e = self(ctx);
        const k = e.find(memory.ptr) orelse return false;
        if (!e.child.rawResize(memory, alignment, new_len, ret)) return false;
        e.held.items[k].len = new_len;
        return true;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: Alignment, ret: usize) void {
        const e = self(ctx);
        const k = e.find(memory.ptr) orelse return;
        _ = e.held.swapRemove(k);
        e.child.rawFree(memory, alignment, ret);
    }
};

test "each allocation is the child's exact size, a free gives it back, and deinit frees the rest" {
    var c = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    var e = Exact.init(c.allocator());
    const a = e.allocator();
    const big = try a.alloc(u32, 1_000_003);
    const small = try a.alloc(u8, 5);
    try std.testing.expectEqual(@as(usize, 4_000_012 + 5), e.bytes() - e.held.capacity * @sizeOf(Exact.Block));
    a.free(small);
    try std.testing.expectEqual(@as(usize, 1), e.held.items.len);
    _ = big;
    e.deinit();
    try std.testing.expectEqual(c.allocated_bytes, c.freed_bytes);
}
