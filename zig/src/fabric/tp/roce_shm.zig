//! The RoCE protocol's shm wire: both ranks' regions in one shared mapping, posts through the guard's stripes.
const std = @import("std");
const Bootstrap = @import("bootstrap.zig").Bootstrap;
const sock = @import("sock.zig");
const guard = @import("roce_guard.zig");
const posix = std.posix;
const page = std.heap.page_size_min;

/// A test's view of each post: slot, padded bytes and the stripes the wire is about to move.
pub const PostTap = struct {
    ctx: *anyopaque,
    seen: *const fn (ctx: *anyopaque, l: guard.Layout, slot: u32, padded: u32, s: []const guard.Stripe) void,
};

/// The peer's region is mapped here; a post copies each stripe, then stores its flag with release order.
pub const ShmWire = struct {
    peer: []u8,
    layout: guard.Layout,
    local: []u8,
    /// Stripes a post makes, as HCAs would: the verbs wire's split, flags and bounds on one host.
    stripes: u32 = 1,
    tap: ?*PostTap = null,

    pub fn post(w: *ShmWire, slot: u32, seq: u32, padded: u32) guard.Error!void {
        var plan: [guard.max_hcas]guard.Stripe = undefined;
        try guard.stripes(w.layout, w.stripes, slot, padded, &plan);
        if (w.tap) |t| t.seen(t.ctx, w.layout, slot, padded, plan[0..w.stripes]);
        for (plan[0..w.stripes]) |s| {
            @memcpy(w.peer[s.remote..][0..s.len], w.local[s.local..][0..s.len]);
            const flag: *std.atomic.Value(u32) = @ptrCast(@alignCast(w.peer[s.flag..].ptr));
            flag.store(seq, .release);
        }
    }
};

/// One shared mapping holding both ranks' regions (rank 0 creates and sizes the file before anyone learns its name).
pub fn sharedPair(b: *Bootstrap, len: usize) ![]align(page) u8 {
    var name: [64]u8 = undefined;
    var name_len: u8 = 0;
    var fd: std.c.fd_t = -1;
    defer if (fd >= 0) {
        _ = std.c.close(fd);
    };
    if (b.rank == 0) blk: {
        const s = std.fmt.bufPrint(&name, "/dev/shm/tf-tp-roce-{d}-{x}", .{ std.c.getpid(), sock.nowNs() }) catch break :blk;
        name[s.len] = 0;
        fd = std.c.open(name[0..s.len :0], .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (fd >= 0 and std.c.ftruncate(fd, @intCast(2 * len)) == 0) name_len = @intCast(s.len);
    }
    // an empty name tells the other rank that rank 0 failed (it must not wait for a name that never comes)
    try b.broadcast(std.mem.asBytes(&name_len), std.mem.asBytes(&name_len));
    if (name_len == 0) return error.SharedMemoryUnavailable;
    try b.broadcast(name[0..name_len], name[0..name_len]);
    name[name_len] = 0;
    if (b.rank != 0) {
        fd = std.c.open(name[0..name_len :0], .{ .ACCMODE = .RDWR, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.SharedMemoryUnavailable;
    }
    const m = try posix.mmap(null, 2 * len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
    try b.agree("mapped", "RoCE shm wire");
    if (b.rank == 0) _ = std.c.unlink(name[0..name_len :0]);
    return m;
}
