//! An allocator that zeroes memory before giving it back: a note's text and token ids outlive no free.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

pub const Zeroing = struct {
    child: Allocator,

    pub fn allocator(z: *Zeroing) Allocator {
        return .{ .ptr = z, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn self(ctx: *anyopaque) *Zeroing {
        return @ptrCast(@alignCast(ctx));
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: Alignment, ret: usize) ?[*]u8 {
        return self(ctx).child.rawAlloc(len, alignment, ret);
    }

    /// In place only; a shrink zeroes the tail first (the caller keeps only new_len bytes either way).
    fn resize(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret: usize) bool {
        if (new_len < memory.len) std.crypto.secureZero(u8, memory[new_len..]);
        return self(ctx).child.rawResize(memory, alignment, new_len, ret);
    }

    /// Never moves: a move would free the old bytes unzeroed, so the caller copies through alloc and free instead.
    fn remap(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret: usize) ?[*]u8 {
        if (resize(ctx, memory, alignment, new_len, ret)) return memory.ptr;
        return null;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: Alignment, ret: usize) void {
        std.crypto.secureZero(u8, memory);
        self(ctx).child.rawFree(memory, alignment, ret);
    }
};

test "freed and shrunk memory is zeroed before the child sees it" {
    const Spy = struct {
        buf: [64]u8 align(16) = undefined,
        seen_free: [64]u8 = undefined,
        freed: usize = 0,
        fn a(s: *@This()) Allocator {
            return .{ .ptr = s, .vtable = &.{ .alloc = al, .resize = Allocator.noResize, .remap = Allocator.noRemap, .free = fr } };
        }
        fn al(ctx: *anyopaque, len: usize, _: Alignment, _: usize) ?[*]u8 {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            return if (len <= s.buf.len) &s.buf else null;
        }
        fn fr(ctx: *anyopaque, memory: []u8, _: Alignment, _: usize) void {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            @memcpy(s.seen_free[0..memory.len], memory);
            s.freed = memory.len;
        }
    };
    var spy: Spy = .{};
    var z: Zeroing = .{ .child = spy.a() };
    const g = z.allocator();
    const text = try g.dupe(u8, "My favourite colour is green.");
    g.free(text);
    try std.testing.expectEqual(@as(usize, 29), spy.freed);
    for (spy.seen_free[0..spy.freed]) |b| try std.testing.expectEqual(@as(u8, 0), b);
}
