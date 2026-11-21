//! Device memory for the Qwen3 CUDA engine: the weights held for its life, the pass's scratch in one carved block.
const std = @import("std");
const cuda = @import("cuda");

/// Bytes between two scratch sections: every kernel's 16-byte loads and the warp forms need this much.
pub const unit = 256;

/// `n` bytes rounded up to the sections' `unit`.
pub fn up(n: usize) usize {
    return std.mem.alignForward(usize, n, unit);
}

/// A span another value owns, for one copy into or out of it.
pub fn view(d: *const cuda.Driver, ptr: u64, len: usize) cuda.DeviceBuffer {
    return .{ .d = d, .ptr = ptr, .len = len, .borrowed = true };
}

/// Every weight buffer the model holds, freed together.
pub const Store = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn deinit(st: *Store) void {
        for (st.list.items) |*b| b.free();
        st.list.deinit(st.gpa);
    }

    /// `n` bytes at a fresh address (`unit` at least, so a zero-length tensor still has one).
    pub fn alloc(st: *Store, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(st.d, @max(n, unit));
        errdefer b.free();
        try st.list.append(st.gpa, b);
        return b.ptr;
    }

    /// `bytes` uploaded at a fresh address.
    pub fn put(st: *Store, bytes: []const u8) !u64 {
        const p = try st.alloc(bytes.len);
        try view(st.d, p, bytes.len).upload(0, bytes);
        return p;
    }
};

test "the sections sit a unit apart, so a shifted address stays 16-byte aligned" {
    try std.testing.expectEqual(@as(usize, 256), up(1));
    try std.testing.expectEqual(@as(usize, 256), up(256));
    try std.testing.expectEqual(@as(usize, 512), up(257));
    try std.testing.expectEqual(@as(usize, 0), up(0));
}
