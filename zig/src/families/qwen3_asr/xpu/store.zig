//! Device memory for the shared schedules on Level Zero: `put` and `taps` place tensors, `alloc` gives scratch.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rt = xpu.rt;

/// A span another value owns (a device address from the schedules), for one launch argument or copy.
pub fn view(r: *rt.Runtime, ptr: u64, len: usize) rt.Buffer {
    return .{ .rt = r, .ptr = @ptrFromInt(ptr), .len = len };
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
    r: *rt.Runtime,
    list: std.ArrayList(rt.Buffer) = .empty,
    bytes: u64 = 0,

    pub fn deinit(st: *Store) void {
        for (st.list.items) |*b| b.free();
        st.list.deinit(st.gpa);
    }

    pub fn alloc(st: *Store, n: usize) !u64 {
        var b = try st.r.alloc(@max(n, 256));
        errdefer b.free();
        try st.list.append(st.gpa, b);
        st.bytes += b.len;
        return @intFromPtr(b.ptr.?);
    }

    /// The bytes copied before returning: the caller may free them (tower.joined does).
    pub fn put(st: *Store, bytes: []const u8) !u64 {
        const p = try st.alloc(bytes.len);
        try st.r.upload(view(st.r, p, bytes.len), bytes);
        try st.r.sync();
        return p;
    }

    /// A conv's [co, k] taps as stored: asr_conv reads rows k values apart.
    pub fn taps(st: *Store, bytes: []const u8, co: usize, k: usize) !u64 {
        if (bytes.len != co * k * 2) return error.TapsShape;
        return st.put(bytes);
    }
};
