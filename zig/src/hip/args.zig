//! C argument values for hipModuleLaunchKernel; pointers are rebuilt after the pack moves.
const std = @import("std");

pub const Args = struct {
    pub const max_args = 48;
    pub const max_bytes = 2048;
    storage: [max_bytes]u8 align(16) = undefined,
    offsets: [max_args]u16 = undefined,
    ptrs: [max_args]?*anyopaque = undefined,
    used: usize = 0,
    count: usize = 0,

    /// Values must match the kernel's C parameter types; capacity failures leave the pack unchanged.
    pub fn add(self: *Args, value: anytype) error{Invalid}!void {
        const T = @TypeOf(value);
        comptime {
            if (@sizeOf(T) == 0 or @alignOf(T) > 16)
                @compileError("HIP arguments require nonzero size and alignment at most 16");
        }
        const at = std.mem.alignForward(usize, self.used, @alignOf(T));
        if (self.count == max_args or at > max_bytes or @sizeOf(T) > max_bytes - at)
            return error.Invalid;
        @memcpy(self.storage[at..][0..@sizeOf(T)], std.mem.asBytes(&value));
        self.offsets[self.count] = @intCast(at);
        self.count += 1;
        self.used = at + @sizeOf(T);
    }

    pub fn pointers(self: *Args) ?[*]?*anyopaque {
        if (self.count == 0) return null;
        for (self.offsets[0..self.count], 0..) |off, i| self.ptrs[i] = &self.storage[off];
        return &self.ptrs;
    }
};

test "mixed C values preserve alignment and moved packs rebuild pointers" {
    var original: Args = .{};
    try std.testing.expect(original.pointers() == null);
    try original.add(@as(u8, 7));
    try original.add(@as(u64, 0x1122334455667788));
    try original.add(@as(c_int, -3));
    const Pair = extern struct { n: c_int, scale: f32 };
    try original.add(Pair{ .n = 9, .scale = 2.5 });
    _ = original.pointers();
    var moved = original;
    const p = moved.pointers().?;
    try std.testing.expectEqualSlices(u16, &.{ 0, 8, 16, 20 }, moved.offsets[0..moved.count]);
    try std.testing.expectEqual(@intFromPtr(&moved.storage[8]), @intFromPtr(p[1].?));
    try std.testing.expectEqual(@as(u64, 0x1122334455667788), @as(*align(1) const u64, @ptrCast(p[1].?)).*);
    try std.testing.expectEqual(@as(c_int, -3), @as(*align(1) const c_int, @ptrCast(p[2].?)).*);
    try std.testing.expectEqual(@as(f32, 2.5), @as(*align(1) const Pair, @ptrCast(p[3].?)).scale);
}

test "argument count overflow is checked without mutation" {
    var a: Args = .{};
    for (0..Args.max_args) |_| try a.add(@as(u8, 1));
    try std.testing.expectError(error.Invalid, a.add(@as(u64, 2)));
    try std.testing.expectEqual(Args.max_args, a.count);
    try std.testing.expectEqual(Args.max_args, a.used);
}

test "storage overflow is checked without mutation" {
    var a: Args = .{};
    const Block = extern struct { bytes: [Args.max_bytes]u8 };
    try a.add(Block{ .bytes = @splat(0) });
    try std.testing.expectError(error.Invalid, a.add(@as(u64, 1)));
    try std.testing.expectEqual(@as(usize, 1), a.count);
    try std.testing.expectEqual(Args.max_bytes, a.used);
}
