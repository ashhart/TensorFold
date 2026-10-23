//! The RoCE protocol's shm wire: every rank's region in one shared mapping, posts through the guard's stripes.
const std = @import("std");
const Bootstrap = @import("bootstrap.zig").Bootstrap;
const sock = @import("sock.zig");
const guard = @import("roce_guard.zig");
const posix = std.posix;
const page = std.heap.page_size_min;

/// A test's view of each post: the destination rank, slot, padded bytes and the stripes the wire is about to move.
pub const PostTap = struct {
    ctx: *anyopaque,
    seen: *const fn (ctx: *anyopaque, l: guard.Layout, dst: u32, slot: u32, padded: u32, s: []const guard.Stripe) void,
};

/// Every peer's region is mapped here; a post copies each stripe to each peer, then stores its flag (release).
pub const ShmWire = struct {
    /// regions[p]: rank p's region (this rank's own at `me`).
    regions: [guard.max_peers][]u8,
    me: u32,
    layout: guard.Layout,
    local: []u8,
    /// Stripes a post makes, as HCAs would: the verbs wire's split, flags and bounds on one host.
    stripes: u32 = 1,
    tap: ?*PostTap = null,

    pub fn post(w: *ShmWire, slot: u32, seq: u32, padded: u32) guard.Error!void {
        const l = w.layout;
        var plan: [guard.max_hcas]guard.Stripe = undefined;
        for (0..l.world - 1) |k| {
            const p = guard.peerAt(w.me, l.world, @intCast(k));
            try guard.stripes(l, w.me, p, w.stripes, slot, padded, &plan);
            if (w.tap) |t| t.seen(t.ctx, l, p, slot, padded, plan[0..w.stripes]);
            for (plan[0..w.stripes]) |s| {
                @memcpy(w.regions[p][s.remote..][0..s.len], w.local[s.local..][0..s.len]);
                const flag: *std.atomic.Value(u32) = @ptrCast(@alignCast(w.regions[p][s.flag..].ptr));
                flag.store(seq, .release);
            }
        }
    }
};

/// One shared mapping holding every rank's region, rank order (rank 0 sizes the file before anyone learns its name).
pub fn sharedAll(b: *Bootstrap, len: usize) ![]align(page) u8 {
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
        if (fd >= 0 and std.c.ftruncate(fd, @intCast(b.world * len)) == 0) name_len = @intCast(s.len);
    }
    // an empty name tells the other ranks that rank 0 failed (they must not wait for a name that never comes)
    try b.broadcast(std.mem.asBytes(&name_len), std.mem.asBytes(&name_len));
    if (name_len == 0) return error.SharedMemoryUnavailable;
    try b.broadcast(name[0..name_len], name[0..name_len]);
    name[name_len] = 0;
    if (b.rank != 0) {
        fd = std.c.open(name[0..name_len :0], .{ .ACCMODE = .RDWR, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.SharedMemoryUnavailable;
    }
    const m = try posix.mmap(null, b.world * len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
    try b.agree("mapped", "RoCE shm wire");
    if (b.rank == 0) _ = std.c.unlink(name[0..name_len :0]);
    return m;
}
