//! A rank pair whose image broadcast fails after it was enqueued: the features stay owned until the stream drains.
const std = @import("std");
const cuda = @import("cuda");
const shared = @import("qwen_vision");
const prompt = @import("cuda_prompt.zig");
const ranks = @import("cuda_rank_engine.zig");
const wire = @import("cuda_rank_wire.zig");
const images = @import("cuda_images.zig");
const media = @import("cuda_rank_media.zig");
const nccl = cuda.nccl;

// Virtual device memory, a broadcast the peer abandons as soon as it is enqueued, and waits that can fail.
const Fake = struct {
    var pending: bool = false;
    var bad_free: usize = 0;
    var fail_waits: usize = 0;
    var live: usize = 0;
    var next: u64 = 0x100000;
    var peer: c_int = -1;

    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        out.* = next;
        next += n + 256;
        live += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        if (pending) bad_free += 1;
        live -= 1;
        return 0;
    }
    fn upload(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        if (pending and fail_waits > 0) {
            fail_waits -= 1;
            return cuda.abi.error_not_ready;
        }
        pending = false;
        return 0;
    }
    fn context() callconv(.c) c_int {
        pending = false;
        return 0;
    }
    fn broadcast(_: cuda.abi.DevicePtr, _: cuda.abi.DevicePtr, _: usize, _: nccl.DataType, _: c_int, _: nccl.Comm, _: cuda.abi.Stream) callconv(.c) c_int {
        pending = true;
        if (peer >= 0) {
            _ = std.c.close(peer);
            peer = -1;
        }
        return 0;
    }
    fn asyncError(_: nccl.Comm, state: *c_int) callconv(.c) c_int {
        state.* = 0;
        return 0;
    }
    fn abort(_: nccl.Comm) callconv(.c) c_int {
        return 0;
    }
    fn driver() cuda.Driver {
        pending = false;
        bad_free = 0;
        live = 0;
        fail_waits = 1; // the drain's stream wait fails once; the context's completes
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyHtoD_v2 = upload;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = context;
        return d;
    }
};

const config: shared.Config = .{ .hidden = 1024, .heads = 16, .depth = 24, .intermediate = 4096, .out_hidden = 2560, .image_token = 50, .vision_start = 52, .vision_end = 53 };

fn encoder(d: *const cuda.Driver, s: cuda.Stream) shared.media.Encoder {
    return .{ .ctx = undefined, .config = config, .driver = d, .stream = s, .splice = .{ .d = d, .handle = null }, .encode = undefined };
}

/// A rank of the pair over one end of a socket pair, its collective faked; the other end is the peer that leaves.
fn session(engine: *prompt.Engine, rank: usize, pair: *[2]c_int) !ranks.Session {
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, pair) != 0) return error.SocketPair;
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
    for (pair) |fd| {
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
    }
    Fake.peer = pair[1];
    var s: ranks.Session = .{ .gpa = std.testing.allocator, .rank = rank, .link = .{ .fd = pair[0] }, .collective = undefined, .engine = engine };
    s.collective.failed = false;
    s.collective.comm = @ptrFromInt(0x1000);
    s.collective.link = .{ .fd = pair[0] };
    s.collective.timeout_ms = 1000;
    s.collective.lib.api.ncclBroadcast = Fake.broadcast;
    s.collective.api.ncclCommGetAsyncError = Fake.asyncError;
    s.collective.api.ncclCommAbort = Fake.abort;
    return s;
}

fn close(pair: [2]c_int) void {
    _ = std.c.close(pair[0]);
    if (Fake.peer >= 0) _ = std.c.close(Fake.peer);
    Fake.peer = -1;
}

test "a leader whose peer leaves after the image broadcast is enqueued drains before its features can be freed" {
    const a = std.testing.allocator;
    var d = Fake.driver();
    var stream: cuda.Stream = .{ .d = &d, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.driver = &d;
    engine.stream = &stream;
    var pair: [2]c_int = undefined;
    var s = try session(&engine, 0, &pair);
    defer close(pair);
    const plan = try shared.media.Plan.init(a, 12, &.{.{ .at = 2, .gh = 4, .gw = 6 }}, .{});
    var q: prompt.Seq = undefined;
    q.rank_id = 1;
    q.media = .{ .image = try shared.media.Media.allocate(plan, encoder(&d, stream)), .prep = .{ .d = &d, .handle = null }, .pool = .{ .d = &d, .handle = null } };
    try std.testing.expectError(error.RankDisconnected, media.send(&s, &q));
    try std.testing.expect(!Fake.pending);
    q.media.?.deinit();
    try std.testing.expectEqual(@as(usize, 0), Fake.bad_free);
    try std.testing.expectEqual(@as(usize, 0), Fake.live);
}

test "a follower whose image broadcast fails after it is enqueued keeps the features until the stream drains" {
    const a = std.testing.allocator;
    var d = Fake.driver();
    var stream: cuda.Stream = .{ .d = &d, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.driver = &d;
    engine.stream = &stream;
    var v: images.Vision = undefined;
    v.a = a;
    v.limits = .{};
    v.encoder = encoder(&d, stream);
    v.prep = .{ .d = &d, .handle = null };
    v.pool = .{ .d = &d, .handle = null };
    engine.image_factory = &v;
    var pair: [2]c_int = undefined;
    var s = try session(&engine, 1, &pair);
    defer close(pair);
    var q: prompt.Seq = undefined;
    q.media = null;
    q.fed = 0;
    q.cap = 64;
    var raw: [12]u8 = undefined;
    for ([_]u32{ 2, 4, 6 }, 0..) |x, i| std.mem.writeInt(u32, raw[i * 4 ..][0..4], x, .little);
    const h: wire.Header = .{ .kind = .media, .object = 1, .arg = 12, .bytes = raw.len };
    try std.testing.expectError(error.RankDisconnected, media.receive(&s, &q, h, &raw));
    try std.testing.expect(!Fake.pending);
    try std.testing.expect(q.media != null);
    q.media.?.deinit();
    try std.testing.expectEqual(@as(usize, 0), Fake.bad_free);
    try std.testing.expectEqual(@as(usize, 0), Fake.live);
}
