//! Host tests of the tensor-parallel EXL3 graph step: dispatch, eviction and update agreements.

const std = @import("std");
const windows = @import("builtin").os.tag == .windows;
const cuda = @import("cuda");
const collective = @import("../../cuda_collective.zig");
const wire = @import("../../cuda_rank_wire.zig");
const state_binding = @import("../kv/exl3_state_binding.zig");
const memory = @import("cuda_exl3_graph_memory.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const gpending = @import("cuda_exl3_graph_pending.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const gbudget = @import("cuda_exl3_graph_budget.zig");
const grun = @import("cuda_exl3_graph_run.zig");
const gtogether = @import("cuda_exl3_graph_together.zig");
const gprefill = @import("cuda_exl3_graph_prefill.zig");
const gdriver = @import("cuda_exl3_graph_test_driver.zig");
const Key = gkey.Key;
const PKey = gprefill.PKey;
const GraphTestDriver = gdriver.GraphTestDriver;

test "exl3: TP eviction proposes the least recently used cohort and ignores local graphs" {
    gcache.entries = @splat(null);
    defer gcache.entries = @splat(null);
    const key: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    gcache.entries[0] = .{ .key = key, .used = 30, .cohort = 3 };
    gcache.entries[1] = .{ .key = key, .used = 10, .cohort = 7 };
    gcache.entries[2] = .{ .key = key, .used = 0, .cohort = 0 };
    try std.testing.expectEqual(@as(u64, 7), gbudget.oldestCohort());
    gcache.entries[0].?.used = 10;
    try std.testing.expectEqual(@as(u64, 3), gbudget.oldestCohort());
}

test "exl3: TP decode dispatch stays eager while its peer instantiates and replays only after agreement" {
    if (windows) return error.SkipZigTest; // The rank wire explicitly refuses Windows.
    gcache.clear();
    defer gcache.clear();
    var pair: [2]std.posix.socket_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
    for (pair) |fd| {
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
    }
    const key: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    const Peer = struct {
        fn main(link: wire.Link, signature: u64) void {
            var c = GraphTestDriver.control(link);
            const first = c.graphAgree(signature, .{ .state = .pending, .cohort = 7 }) catch @panic("first peer agreement failed");
            std.debug.assert(first.choice == .eager);
            const second = c.graphAgree(signature, .{ .state = .ready, .cohort = 7 }) catch @panic("second peer agreement failed");
            std.debug.assert(second.choice == .replay);
        }
    };
    const thread = try std.Thread.spawn(.{}, Peer.main, .{ wire.Link{ .fd = pair[1] }, gkey.graphSignature(key, 0) });
    defer thread.join();
    var c = GraphTestDriver.control(.{ .fd = pair[0] });
    var driver = GraphTestDriver.driver();
    GraphTestDriver.launches = 0;
    gcache.entries[0] = .{ .key = key, .used = 1, .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .cohort = 7 };
    const Pass = struct {
        io: std.Io = std.testing.io,
        stream: cuda.Stream,
        m: struct { collective: ?*collective.Collective },
        prepared: bool = false,
        chunks: usize = 0,

        pub fn chunk(p: *@This(), _: []const u32, _: usize, _: ?[]u8) !void {
            p.chunks += 1;
        }
    };
    var pass: Pass = .{ .stream = cuda.Stream.borrowed(&driver, null), .m = .{ .collective = &c } };
    try grun.run(key, &pass, &.{1}, 0);
    try std.testing.expectEqual(@as(usize, 1), pass.chunks);
    try std.testing.expectEqual(@as(usize, 0), GraphTestDriver.launches);
    try grun.run(key, &pass, &.{1}, 0);
    try std.testing.expectEqual(@as(usize, 1), pass.chunks);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.launches);
}

test "exl3: TP pressure on either rank retires cohorts before retrying admission" {
    if (windows) return error.SkipZigTest; // The rank wire explicitly refuses Windows.
    gcache.clear();
    defer gcache.clear();
    for ([_]bool{ true, false }) |local_pressure| {
        try memory.configure(.{ .reservation_bytes = 100, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 }, .prefill_charge = .{ .exec = 1, .template = 0, .transient = 0 } });
        var driver = GraphTestDriver.driver();
        GraphTestDriver.exec_drops = 0;
        const local = try memory.admit(.prefill, 0, 100);
        try memory.installed(local);
        gprefill.pentries[0] = .{ .key = std.mem.zeroes(PKey), .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 0, .admission = local };
        if (local_pressure) for (0..2) |index| {
            const ticket = try memory.admit(.decode, index + 7, 100);
            try memory.installed(ticket);
            gcache.entries[index] = .{ .key = .{ .kind = 1, .rows = index + 1, .scratch = 1, .scores = 2, .output = 3 }, .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = index + 1, .cohort = index + 7, .admission = ticket };
        };
        var pair: [2]std.posix.socket_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        defer for (pair) |fd| {
            _ = std.c.close(fd);
        };
        const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
        for (pair) |fd| {
            const flags = std.c.fcntl(fd, std.c.F.GETFL);
            if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
        }
        const Peer = struct {
            fn run(link: wire.Link, pressure: bool) void {
                var control = GraphTestDriver.control(link);
                const first = control.graphAgree(42 ^ 0x61646d6974, .{ .state = if (pressure) .ready else .eager, .cohort = 11 }) catch @panic("pressure agreement failed");
                std.debug.assert(first.choice == .eager);
                const victim = control.graphVictim(42 ^ 0x76696374696d, if (pressure) 8 else 7) catch @panic("victim agreement failed");
                std.debug.assert(victim == 7);
                gbudget.retirementBarrier(&control, 42 ^ 0x71756965736365, victim) catch @panic("quiescence agreement failed");
                gbudget.retirementBarrier(&control, 42 ^ 0x72657469726564, victim) catch @panic("retirement agreement failed");
                const admitted = control.graphAgree(42 ^ 0x61646d6974, .{ .state = .ready, .cohort = 11 }) catch @panic("retry agreement failed");
                std.debug.assert(admitted.choice == .replay);
            }
        };
        const peer = try std.Thread.spawn(.{}, Peer.run, .{ wire.Link{ .fd = pair[1] }, local_pressure });
        defer peer.join();
        var control = GraphTestDriver.control(.{ .fd = pair[0] });
        const ticket = try gbudget.reserveTogether(&control, cuda.Stream.borrowed(&driver, null), 42, 11);
        try std.testing.expectEqual(@as(u64, if (local_pressure) 71 else 41), memory.charged());
        try std.testing.expectEqual(@as(usize, if (local_pressure) 1 else 0), GraphTestDriver.exec_drops);
        try std.testing.expect(gprefill.pentries[0] != null);
        if (local_pressure) try std.testing.expectEqual(@as(u64, 8), gcache.entries[1].?.cohort);
        try memory.release(ticket);
        gcache.clear();
    }
}

test "exl3: exhausted TP victims reclaim locals incrementally with asymmetric peer caches" {
    if (windows) return error.SkipZigTest; // The rank wire explicitly refuses Windows.
    gcache.clear();
    defer gcache.clear();
    for ([_]bool{ false, true }) |local_cached| for ([_]bool{ false, true }) |peer_cached| {
        try memory.configure(.{ .reservation_bytes = 100, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 } });
        var driver = GraphTestDriver.driver();
        GraphTestDriver.exec_drops = 0;
        if (local_cached) {
            const prefill = try memory.admit(.prefill, 0, 100);
            try memory.installed(prefill);
            gprefill.pentries[0] = .{ .key = std.mem.zeroes(PKey), .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 1, .admission = prefill };
            const local = try memory.admit(.decode, 0, 100);
            try memory.installed(local);
            gcache.entries[0] = .{ .key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 }, .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 2, .admission = local };
        }
        var pair: [2]std.posix.socket_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        defer for (pair) |fd| {
            _ = std.c.close(fd);
        };
        const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
        for (pair) |fd| {
            const flags = std.c.fcntl(fd, std.c.F.GETFL);
            if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
        }
        const Peer = struct {
            fn run(link: wire.Link, local_cached_peer: bool, cached: bool) void {
                var control = GraphTestDriver.control(link);
                const first = control.graphAgree(42 ^ 0x61646d6974, .{ .state = .eager, .cohort = 11 }) catch @panic("pressure agreement failed");
                std.debug.assert(first.choice == .eager);
                const victim = control.graphVictim(42 ^ 0x76696374696d, 0) catch @panic("victim agreement failed");
                std.debug.assert(victim == 0);
                gbudget.retirementBarrier(&control, 42 ^ 0x6c6f63616c7175, 11) catch @panic("local quiescence failed");
                const reclaimed = control.graphVictim(42 ^ 0x6c6f63616c646f, @intFromBool(cached)) catch @panic("local destruction acknowledgment failed");
                std.debug.assert(reclaimed == @as(u64, @intFromBool(local_cached_peer or cached)));
                if (reclaimed == 0) return;
                const admitted = control.graphAgree(42 ^ 0x61646d6974, .{ .state = .ready, .cohort = 11 }) catch @panic("retry agreement failed");
                std.debug.assert(admitted.choice == .replay);
            }
        };
        const peer = try std.Thread.spawn(.{}, Peer.run, .{ wire.Link{ .fd = pair[1] }, local_cached, peer_cached });
        defer peer.join();
        var control = GraphTestDriver.control(.{ .fd = pair[0] });
        const result = gbudget.reserveTogether(&control, cuda.Stream.borrowed(&driver, null), 42, 11);
        if (local_cached or peer_cached) {
            const ticket = try result;
            try std.testing.expectEqual(@as(u64, if (local_cached) 70 else 40), memory.charged());
            try memory.release(ticket);
        } else {
            try std.testing.expectError(error.GraphBudgetExceeded, result);
            try std.testing.expectEqual(@as(u64, 0), memory.charged());
        }
        try std.testing.expectEqual(@as(usize, @intFromBool(local_cached)), GraphTestDriver.exec_drops);
        try std.testing.expect(gprefill.pentries[0] == null);
        try std.testing.expectEqual(local_cached, gcache.entries[0] != null);
        gcache.clear();
    };
}

test "exl3: TP updates agree the old cohort and transfer template and admission together" {
    if (windows) return error.SkipZigTest; // The rank wire explicitly refuses Windows.
    try testTPUpdate(true);
}

test "exl3: asymmetric TP update rejection retires both execs before re-instantiating" {
    if (windows) return error.SkipZigTest; // The rank wire explicitly refuses Windows.
    try testTPUpdate(false);
}

fn testTPUpdate(peer_accept: bool) !void {
    if (windows) return error.SkipZigTest;
    gcache.clear();
    defer gcache.clear();
    const previous_api = gpending.ctx_api;
    defer gpending.ctx_api = previous_api;
    gpending.ctx_api = .{ .get = GraphTestDriver.noContext, .set = GraphTestDriver.setContext };
    try memory.configure(.{ .reservation_bytes = 120, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 } });
    var driver = GraphTestDriver.driver();
    GraphTestDriver.exec_updates = 0;
    GraphTestDriver.exec_drops = 0;
    GraphTestDriver.graph_drops = 0;
    const old_ticket = try memory.admit(.decode, 7, 100);
    try memory.installed(old_ticket);
    const key: Key = .{ .kind = 1, .rows = 1, .scratch = 2, .scores = 2, .output = 3 };
    var old_key = key;
    old_key.scratch = 1;
    gcache.entries[0] = .{ .key = old_key, .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .template = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 1, .cohort = 7, .admission = old_ticket };
    var old_binding_request: state_binding.Request = .{
        .width = 2,
        .chain = 10,
        .chain_to = 11,
        .geometry = .{ .grid = .{ 16, 8, 1 }, .block = .{ 128, 1, 1 } },
        .nonstate = .{ .gq = 1, .gk = 2, .gv = 3, .ggb = 4, .rows = 2, .ys = 5 },
        .count = 1,
    };
    old_binding_request.layers[0] = .{ .layer = 0, .state_in = 100, .state_out = 200 };
    gcache.entries[0].?.binding = try state_binding.Binding.init(1, old_binding_request);
    gcache.seen[0] = .{ .key = key, .n = std.math.maxInt(u32) - 1, .used = 1 };
    const previous_clock = gcache.clock;
    defer gcache.clock = previous_clock;
    gcache.clock = 1024;
    var pair: [2]std.posix.socket_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
    for (pair) |fd| {
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
    }
    const Peer = struct {
        fn run(link: wire.Link, signature: u64, accept: bool) void {
            var control = GraphTestDriver.control(link);
            control.graph_step = 10;
            const capture = control.graphAgree(signature, .{ .state = .capture }) catch @panic("capture agreement failed");
            std.debug.assert(capture.choice == .capture and capture.cohort == 11);
            gbudget.retirementBarrier(&control, signature ^ 0x61646d6974, 11) catch @panic("admission failed");
            const captured = control.graphAgree(signature ^ 0x63617074757265, .{ .state = .ready, .cohort = 11 }) catch @panic("capture outcome agreement failed");
            std.debug.assert(captured.choice == .replay);
            gbudget.retirementBarrier(&control, signature ^ 0x757064617465, 7) catch @panic("old cohort agreement failed");
            const result = control.graphAgree(signature ^ 0x75706461746564, .{ .state = if (accept) .ready else .capture, .cohort = 11 }) catch @panic("update result agreement failed");
            std.debug.assert(result.choice == if (accept) collective.GraphChoice.replay else collective.GraphChoice.capture);
            if (!accept) gbudget.retirementBarrier(&control, signature ^ 0x72656275696c64, 11) catch @panic("rebuild retirement agreement failed");
            gbudget.retirementBarrier(&control, signature ^ 0x696e7374616c6c, 11) catch @panic("installation agreement failed");
        }
    };
    const peer = try std.Thread.spawn(.{}, Peer.run, .{ wire.Link{ .fd = pair[1] }, gkey.graphSignature(key, 0), peer_accept });
    defer peer.join();
    var control = GraphTestDriver.control(.{ .fd = pair[0] });
    control.graph_step = 10;
    const Pass = struct {
        io: std.Io = std.testing.io,
        stream: cuda.Stream,
        prepared: bool = false,
        chunks: usize = 0,
        pub fn chunk(pass: *@This(), _: []const u32, _: usize, _: ?[]u8) !void {
            pass.chunks += 1;
        }
    };
    var pass: Pass = .{ .stream = cuda.Stream.borrowed(&driver, null) };
    try gtogether.runTogether(&control, key, &pass, &.{1}, 0);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.exec_updates);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.graph_drops);
    try std.testing.expectEqual(@as(usize, if (peer_accept) 0 else 1), GraphTestDriver.exec_drops);
    try std.testing.expectEqual(@as(u64, 30), memory.charged());
    try std.testing.expectEqual(@as(u64, 11), gcache.find(key).?.cohort);
    try std.testing.expect(gcache.find(key).?.binding == null);
    try std.testing.expectError(error.StaleGraphTicket, memory.release(old_ticket));
    try std.testing.expectEqual(@as(usize, 1), pass.chunks);
}
