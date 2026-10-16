//! Groups of threads joined over 127.0.0.1 for host tests: each rank holds the collective.Link a served rank has.

const std = @import("std");
const posix = std.posix;
const collective = @import("../collective.zig");
const link = @import("link.zig");
const group = @import("group.zig");
const Fault = @import("../agree.zig").Fault;
const Policy = @import("../policy.zig").Policy;

/// A port nothing listens on now: bound once, read back and released.
pub fn freePort() !u16 {
    const fd = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(fd) != .SUCCESS) return error.SocketFailed;
    defer _ = posix.system.close(@intCast(fd));
    var addr: posix.sockaddr.in = .{ .port = 0, .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    if (posix.errno(posix.system.bind(
        @intCast(fd),
        @ptrCast(&addr),
        @sizeOf(posix.sockaddr.in),
    )) != .SUCCESS) return error.BindFailed;
    var len: posix.socklen_t = @sizeOf(posix.sockaddr.in);
    if (posix.errno(posix.system.getsockname(@intCast(fd), @ptrCast(&addr), &len)) != .SUCCESS) return error.BindFailed;
    return std.mem.bigToNative(u16, addr.port);
}

/// Rank `rank` of `world`'s hello in a test group: one GPU kind, RCCL, build and checkpoint for all.
pub fn hello(rank: u32, world: u32) link.Hello {
    return .{
        .rank = rank,
        .world = world,
        .gpu = 11,
        .rccl = 22203,
        .ident = .{ .program = @splat(1), .config = @splat(2), .weights = @splat(3) },
    };
}

/// A write to a peer that left fails with an error instead of ending the test process.
pub fn ignorePipe() void {
    const ignore: posix.Sigaction = .{
        .handler = .{ .handler = posix.SIG.IGN },
        .mask = posix.sigemptyset(),
        .flags = 0,
    };
    posix.sigaction(.PIPE, &ignore, null);
}

/// `body(ctx, rank, its link)` on `world` threads, each joined over loopback first; returns once all have.
pub fn run(world: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), usize, collective.Link) void) !void {
    const Rank = struct {
        fn go(c: @TypeOf(ctx), rank: usize, size: usize, port: u16, failed: *?link.Error) void {
            var fault: Fault = .{};
            const policy: [Policy.word_count]u32 = @splat(0);
            const me = hello(@intCast(rank), @intCast(size));
            const id: @import("rccl.zig").UniqueId = .{ .internal = @splat(7) };
            const out = link.Link.open(std.testing.io, "127.0.0.1", port, id, me, policy, &fault) catch |err| {
                failed.* = err;
                return;
            };
            var l = out[0];
            defer l.close();
            body(c, rank, group.linkOver(&l));
        }
    };
    if (world < 2 or world > link.max_world) return error.BadRank;
    ignorePipe();
    const port = try freePort();
    var threads: [link.max_world]std.Thread = undefined;
    var failed: [link.max_world]?link.Error = @splat(null);
    for (0..world) |r| threads[r] = try std.Thread.spawn(.{}, Rank.go, .{ ctx, r, world, port, &failed[r] });
    for (threads[0..world]) |t| t.join();
    for (failed[0..world]) |f| if (f) |err| return err;
}
