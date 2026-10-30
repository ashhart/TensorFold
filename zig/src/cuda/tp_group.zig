//! A tensor-parallel group over NCCL (NCCL_COMM_ID; rows in rank order); two ranks add RoCE small gathers (roce.zig).
const std = @import("std");
const nccl = @import("nccl.zig");
const driver = @import("driver.zig");
const stream_ = @import("stream.zig");
const memory = @import("memory.zig");
const roce = @import("roce.zig");
const p2p_report = @import("p2p_report.zig");

const Stream = stream_.Stream;

pub const Group = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: usize,
    world: usize,
    aborted: std.atomic.Value(bool) = .init(false),
    fast: ?*roce.Gather = null, // two ranks' small all-gathers by RoCE writes (openRoce), when both ranks brought it up

    /// Rank `rank` of `world` (NCCL opened at run time; NCCL_COMM_ID names rank 0's address, every rank the same).
    pub fn open(d: *const driver.Driver, rank: usize, world: usize) !Group {
        if (std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
        var lib = try nccl.Library.open();
        errdefer lib.close();
        try lib.requireSupportedVersion();
        var diag = p2p_report.begin(rank, world);
        defer diag.end();
        var id: nccl.UniqueId = undefined;
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        p2p_report.afterInit(&diag, .{ .lib = &lib, .comm = comm, .d = d, .later = true });
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    pub fn close(g: *Group) void {
        p2p_report.ready(); // a transport line still owed is reported before the communicator goes
        if (g.fast) |f| f.deinit();
        g.destroyComm();
        g.lib.close();
        g.* = undefined;
    }

    fn destroyComm(g: *Group) void {
        if (!g.aborted.swap(true, .acq_rel)) _ = g.lib.api.ncclCommDestroy(g.comm);
    }

    pub fn abort(g: *Group) void {
        if (!g.aborted.swap(true, .acq_rel)) _ = g.lib.api.ncclCommAbort(g.comm);
    }

    /// Two ranks: gathers up to `max_bytes` a rank go by RoCE writes if both ranks bring it up; else NCCL keeps them.
    pub fn openRoce(g: *Group, gpa: std.mem.Allocator, io: std.Io, d: *const driver.Driver, s: Stream, max_bytes: usize) !void {
        if (g.world != 2 or g.fast != null) return;
        g.fast = try roce.Gather.init(gpa, io, d, g, s, roce.wanted(), max_bytes);
    }

    /// A failed RoCE gather's error (its rows hold nothing valid): call it after a synchronize that reads a gather.
    pub fn check(g: *const Group) !void {
        if (g.fast) |f| try f.check();
    }

    /// `bytes` at `mine` from every rank into `all` in rank order (world * bytes there), on `s`.
    pub fn allGather(g: *const Group, s: Stream, mine: u64, all: u64, bytes: usize) !void {
        if (g.fast) |f| if (f.fits(bytes)) return f.gather(s.handle, mine, all, bytes);
        try g.lib.check(g.lib.api.ncclAllGather(mine, all, bytes, .u8, g.comm, s.handle), "ncclAllGather");
    }

    /// `bytes` at `ptr` to rank `peer`, on `s` (its matching recv takes the same count).
    pub fn send(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclSend(ptr, bytes, .u8, @intCast(peer), g.comm, s.handle), "ncclSend");
    }

    /// `bytes` from rank `peer` into `ptr`, on `s`.
    pub fn recv(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclRecv(ptr, bytes, .u8, @intCast(peer), g.comm, s.handle), "ncclRecv");
    }

    /// Rank 0 and rank `peer` connect both ways with one small exchange (`ptr`: 128 bytes, 64 out and 64 in).
    pub fn connect(g: *const Group, s: Stream, ptr: u64, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
        const posted: anyerror!void = post: {
            g.send(s, ptr, 64, peer) catch |err| break :post err;
            g.recv(s, ptr + 64, 64, peer) catch |err| break :post err;
            break :post {};
        };
        try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
        try posted;
        try s.synchronize();
    }

    /// Every rank's `mine` in rank order into `out` (world of them), waited for.
    pub fn gather(g: *const Group, comptime T: type, d: *const driver.Driver, s: Stream, mine: T, out: []T) !void {
        if (out.len != g.world) return error.BadWorld;
        const size = @sizeOf(T);
        var dev = try memory.DeviceBuffer.alloc(d, size * g.world);
        defer dev.free();
        try dev.uploadAsync(g.rank * size, std.mem.asBytes(&mine), s.handle);
        try g.allGather(s, dev.ptr + g.rank * size, dev.ptr, size);
        try s.synchronize();
        try g.check();
        try dev.download(0, std.mem.sliceAsBytes(out));
    }
};

test "an aborted two-rank communicator is not destroyed a second time" {
    const Fake = struct {
        var aborts: usize = 0;
        var destroys: usize = 0;
        fn abort(_: nccl.Comm) callconv(.c) nccl.Result {
            aborts += 1;
            return 0;
        }
        fn destroy(_: nccl.Comm) callconv(.c) nccl.Result {
            destroys += 1;
            return 0;
        }
    };
    var group: Group = .{ .lib = undefined, .comm = null, .rank = 0, .world = 2 };
    group.lib.api.ncclCommAbort = Fake.abort;
    group.lib.api.ncclCommDestroy = Fake.destroy;
    group.abort();
    group.abort();
    group.destroyComm();
    try std.testing.expectEqual(@as(usize, 1), Fake.aborts);
    try std.testing.expectEqual(@as(usize, 0), Fake.destroys);
}
