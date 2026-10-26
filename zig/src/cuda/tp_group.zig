//! A tensor-parallel group over NCCL (NCCL_COMM_ID; rows in rank order); two ranks add RoCE small gathers (roce.zig).
const std = @import("std");
const nccl = @import("nccl.zig");
const driver = @import("driver.zig");
const stream_ = @import("stream.zig");
const memory = @import("memory.zig");
const roce = @import("roce.zig");
const p2p_report = @import("p2p_report.zig");
const abi = @import("abi.zig");

const Stream = stream_.Stream;

/// Three ranks or more: a wire a family borrows for small all-gathers (fabric/tp's multirank RoCE), every rank or none.
pub const Multi = struct {
    ctx: *anyopaque,
    fits: *const fn (ctx: *anyopaque, bytes: usize) bool,
    gather: *const fn (ctx: *anyopaque, s: abi.Stream, mine: abi.DevicePtr, all: abi.DevicePtr, bytes: usize) anyerror!void,
    check: *const fn (ctx: *anyopaque) anyerror!void, // a failed gather's error, as Group.check reports it
    abort: *const fn (ctx: *anyopaque) void, // its device waits give up, as NCCL's do on Group.abort
    deinit: *const fn (ctx: *anyopaque) void,
};

pub const Group = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: usize,
    world: usize,
    aborted: std.atomic.Value(bool) = .init(false),
    fast: ?*roce.Gather = null, // two ranks' small all-gathers by RoCE writes (openRoce), when both ranks brought it up
    multi: ?Multi = null, // three ranks or more: small all-gathers on a borrowed wire, set once every rank's probe agreed

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
        g.closeWires();
        g.destroyComm();
        g.lib.close();
        g.* = undefined;
    }

    /// The RoCE wires beside NCCL: two ranks' gathers, or a borrowed one.
    fn closeWires(g: *Group) void {
        if (g.fast) |f| f.deinit();
        g.fast = null;
        if (g.multi) |m| m.deinit(m.ctx);
        g.multi = null;
    }

    fn destroyComm(g: *Group) void {
        if (!g.aborted.swap(true, .acq_rel)) _ = g.lib.api.ncclCommDestroy(g.comm);
    }

    pub fn abort(g: *Group) void {
        if (g.multi) |m| m.abort(m.ctx);
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
        if (g.multi) |m| try m.check(m.ctx);
    }

    /// `bytes` at `mine` from every rank into `all` in rank order (world * bytes there), on `s`.
    pub fn allGather(g: *const Group, s: Stream, mine: u64, all: u64, bytes: usize) !void {
        if (g.fast) |f| if (f.fits(bytes)) return f.gather(s.handle, mine, all, bytes);
        if (g.multi) |m| if (m.fits(m.ctx, bytes)) return m.gather(m.ctx, s.handle, mine, all, bytes);
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

test "three ranks send a gather the borrowed wire fits down it and the rest to NCCL; its check, abort and close follow" {
    const Fake = struct {
        var gathers: usize = 0;
        var nccl_gathers: usize = 0;
        var checks: usize = 0;
        var aborts: usize = 0;
        var deinits: usize = 0;
        var failing = false;
        fn fits(_: *anyopaque, bytes: usize) bool {
            return bytes <= 8192;
        }
        fn gather(_: *anyopaque, _: abi.Stream, _: abi.DevicePtr, _: abi.DevicePtr, _: usize) anyerror!void {
            gathers += 1;
        }
        fn check(_: *anyopaque) anyerror!void {
            checks += 1;
            if (failing) return error.PeerFailed;
        }
        fn abort(_: *anyopaque) void {
            aborts += 1;
        }
        fn deinit(_: *anyopaque) void {
            deinits += 1;
        }
        fn allGather(_: abi.DevicePtr, _: abi.DevicePtr, _: usize, _: nccl.DataType, _: nccl.Comm, _: abi.Stream) callconv(.c) nccl.Result {
            nccl_gathers += 1;
            return 0;
        }
        fn commAbort(_: nccl.Comm) callconv(.c) nccl.Result {
            return 0;
        }
    };
    var ctx: u8 = 0;
    var group: Group = .{ .lib = undefined, .comm = null, .rank = 0, .world = 3, .multi = .{ .ctx = &ctx, .fits = Fake.fits, .gather = Fake.gather, .check = Fake.check, .abort = Fake.abort, .deinit = Fake.deinit } };
    group.lib.api.ncclAllGather = Fake.allGather;
    group.lib.api.ncclCommAbort = Fake.commAbort;
    const s: Stream = .{ .d = undefined, .handle = null };
    try group.allGather(s, 0, 0, 8192);
    try group.allGather(s, 0, 0, 8193);
    try std.testing.expectEqual(@as(usize, 1), Fake.gathers);
    try std.testing.expectEqual(@as(usize, 1), Fake.nccl_gathers);
    try group.check();
    Fake.failing = true;
    try std.testing.expectError(error.PeerFailed, group.check());
    group.abort();
    try std.testing.expectEqual(@as(usize, 1), Fake.aborts);
    group.closeWires();
    try std.testing.expectEqual(@as(usize, 1), Fake.deinits);
    try std.testing.expect(group.multi == null);
}

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
