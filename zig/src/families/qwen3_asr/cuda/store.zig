//! Device memory for the shared schedules: `put` and `taps` place tensors, `alloc` hands out scratch rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

/// A span another value owns, for one copy into or out of it.
pub fn view(d: *const cuda.Driver, ptr: u64, len: usize) cuda.DeviceBuffer {
    return .{ .d = d, .ptr = ptr, .len = len, .borrowed = true };
}

/// The bf16 bytes of a checkpoint's tensors by name and shape, for tower.Weights.load and lm.Weights.load.
pub const Source = struct {
    ck: *core.Checkpoint,

    pub fn expect(s: Source, name: []const u8, shape: []const usize) ![]const u8 {
        return (try s.ck.expect(name, .bf16, shape)).bytes;
    }
};

/// Every allocation the model holds, freed together.
pub const Store = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0,

    pub fn deinit(st: *Store) void {
        for (st.list.items) |*b| b.free();
        st.list.deinit(st.gpa);
    }

    pub fn alloc(st: *Store, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(st.d, @max(n, 256));
        errdefer b.free();
        try st.list.append(st.gpa, b);
        st.bytes += b.len;
        return b.ptr;
    }

    pub fn put(st: *Store, bytes: []const u8) !u64 {
        const p = try st.alloc(bytes.len);
        try view(st.d, p, bytes.len).upload(0, bytes);
        return p;
    }

    /// A conv's [co, k] taps with each row padded with zeros to `ld` (k to a multiple of 8: 16-byte rows for cuBLASLt).
    pub fn taps(st: *Store, bytes: []const u8, co: usize, k: usize) !u64 {
        const ld = tapsLd(k);
        if (ld == k) return st.put(bytes);
        const padded = try st.gpa.alloc(u8, co * ld * 2);
        defer st.gpa.free(padded);
        @memset(padded, 0);
        for (0..co) |o| @memcpy(padded[o * ld * 2 ..][0 .. k * 2], bytes[o * k * 2 ..][0 .. k * 2]);
        return st.put(padded);
    }
};

/// The taps' row width a conv's columns use: k rounded up to 8 values.
pub fn tapsLd(k: usize) usize {
    return std.mem.alignForward(usize, k, 8);
}
