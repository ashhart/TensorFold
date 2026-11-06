//! The real init wrapper prints before and after a fake communicator call, with only a digest of its id.
const std = @import("std");
const log = @import("startup_log.zig");
const nccl = @import("nccl.zig");
const cudap = @import("p2p_cuda.zig");
const t = std.testing;
const Fake = struct {
    var buf: [4096]u8 = undefined;
    var len: usize = 0;
    var ms: u64 = 0;
    var fail = false;
    var no_version = false;
    var calls: usize = 0;
    pub fn now(_: @This()) u64 {
        return ms;
    }
    fn sink(line: []const u8) void {
        const s = std.fmt.bufPrint(buf[len..], "{s}\n", .{line}) catch unreachable;
        len += s.len;
    }
    fn text() []const u8 {
        return buf[0..len];
    }
    fn version(v: *c_int) callconv(.c) c_int {
        v.* = 23007;
        return if (no_version) 1 else 0;
    }
    fn init(c: *nccl.Comm, world: c_int, id: nccl.UniqueId, rank: c_int) callconv(.c) c_int {
        t.expectEqual(@as(c_int, 2), world) catch unreachable;
        t.expectEqual(@as(c_int, 1), rank) catch unreachable;
        t.expectEqual(@as([128]u8, @splat(0)), id.internal) catch unreachable;
        sink("fake ncclCommInitRank");
        ms += 47;
        calls += 1;
        c.* = @ptrFromInt(0x1000);
        return if (fail) 1 else 0;
    }
    fn library() nccl.Library {
        var l: nccl.Library = undefined;
        l.api.ncclGetVersion = version;
        l.api.ncclCommInitRank = init;
        return l;
    }
    fn reset() void {
        len = 0;
        ms = 900;
        fail = false;
        no_version = false;
        calls = 0;
    }
};

test "startup log: liveness, sent and taken id hashes, init entry, call and elapsed completion print in order" {
    Fake.reset();
    const id: nccl.UniqueId = .{ .internal = @splat(0) };
    try t.expectEqualStrings("38723a2e", &log.idHash(&id.internal));
    log.connected(Fake.sink, 0, 1, 23457, 0);
    log.connected(Fake.sink, 1, 1, 0, 3);
    log.handoff(Fake.sink, 0, &id.internal, true);
    log.handoff(Fake.sink, 1, &id.internal, false);
    const lib = Fake.library();
    var dev: cudap.Dev = .{ .ordinal = 0 };
    dev.bus = .ofC("0000:01:00.0");
    const comm = try log.initWith(&lib, id, 1, 2, &dev, Fake{}, Fake.sink);
    try t.expectEqual(@as(usize, 0x1000), @intFromPtr(comm.?));
    try t.expectEqual(@as(usize, 1), Fake.calls);
    try t.expectEqualStrings(
        "rank 0: liveness lines from 1 ranks on port 23457\n" ++
            "rank 1: liveness line to rank 0 connected after 3 tries\n" ++
            "rank 0: NCCL id 38723a2e sent\n" ++
            "rank 1: NCCL id 38723a2e taken\n" ++
            "rank 1 startup: entering NCCL init (2 ranks, id 38723a2e, NCCL 2.30.7, CUDA device 0 0000:01:00.0)\n" ++
            "fake ncclCommInitRank\nrank 1 startup: NCCL init done in 47 ms\n",
        Fake.text(),
    );
    var changed = id;
    changed.internal[127] = 1;
    try t.expect(!std.mem.eql(u8, &log.idHash(&id.internal), &log.idHash(&changed.internal)));
}

test "startup log: missing diagnostic facts never refuse init, a failed init never claims done" {
    for ([_]bool{ false, true }) |fail| {
        Fake.reset();
        Fake.fail = fail;
        Fake.no_version = true;
        const lib = Fake.library();
        const result = log.initWith(&lib, .{ .internal = @splat(0) }, 1, 2, null, Fake{}, Fake.sink);
        if (fail) try t.expectError(error.NcclFailed, result) else _ = try result;
        try t.expectEqual(@as(usize, 1), Fake.calls);
        try t.expect(std.mem.indexOf(u8, Fake.text(), "NCCL ?, CUDA device ? ?") != null);
        try t.expectEqual(!fail, std.mem.indexOf(u8, Fake.text(), "done in 47 ms") != null);
    }
}

const DeviceFake = struct {
    var missing = false;
    var unknown_bus = false;
    fn current(d: *c_int) callconv(.c) c_int {
        d.* = 9;
        return if (missing) 1 else 0;
    }
    fn count(n: *c_int) callconv(.c) c_int {
        n.* = 2;
        return 0;
    }
    fn get(d: *c_int, ordinal: c_int) callconv(.c) c_int {
        d.* = if (ordinal == 0) 7 else 9;
        return 0;
    }
    fn bus(buf: [*]u8, _: c_int, handle: c_int) callconv(.c) c_int {
        t.expectEqual(@as(c_int, 9), handle) catch unreachable;
        const text = "0000:02:00.0";
        @memcpy(buf[0..text.len], text);
        buf[text.len] = 0;
        return if (unknown_bus) 1 else 0;
    }
};
test "startup log: current handle maps to its visible ordinal and bus, query failures stay unknown" {
    var api: cudap.Api = undefined;
    api.cuCtxGetDevice = DeviceFake.current;
    api.cuDeviceGetCount = DeviceFake.count;
    api.cuDeviceGet = DeviceFake.get;
    api.cuDeviceGetPCIBusId = DeviceFake.bus;
    DeviceFake.missing = false;
    DeviceFake.unknown_bus = false;
    const d = log.deviceWith(&api).?;
    try t.expectEqual(@as(u32, 1), d.ordinal);
    try t.expectEqualStrings("0000:02:00.0", d.bus.slice());
    DeviceFake.unknown_bus = true;
    try t.expectEqual(@as(usize, 0), log.deviceWith(&api).?.bus.len);
    DeviceFake.missing = true;
    try t.expect(log.deviceWith(&api) == null);
}
