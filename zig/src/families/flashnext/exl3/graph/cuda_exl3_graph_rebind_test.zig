//! Host tests of rebinding a cached EXL3 window graph to another sequence's state.

const std = @import("std");
const windows = @import("builtin").os.tag == .windows;
const wire = @import("../../cuda_rank_wire.zig");
const state_binding = @import("../kv/exl3_state_binding.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const grebind = @import("cuda_exl3_graph_rebind.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const gpending = @import("cuda_exl3_graph_pending.zig");
const gdriver = @import("cuda_exl3_graph_test_driver.zig");
const Key = gkey.Key;
const Pending = gpending.Pending;
const GraphTestDriver = gdriver.GraphTestDriver;

test "exl3: state binding candidate excludes pending poisoned retired and incompatible generations" {
    gcache.entries = @splat(null);
    defer gcache.entries = @splat(null);
    var request: state_binding.Request = .{
        .width = 2,
        .chain = 1,
        .chain_to = 2,
        .geometry = .{ .grid = .{ 16, 8, 1 }, .block = .{ 128, 1, 1 } },
        .nonstate = .{ .gq = 1, .gk = 2, .gv = 3, .ggb = 4, .rows = 2, .ys = 5 },
        .count = 1,
    };
    request.layers[0] = .{ .layer = 0, .state_in = 100, .state_out = 200 };
    for (2..state_binding.max_width + 1) |width| {
        request.width = width;
        request.nonstate.rows = @intCast(width);
        const key: Key = .{ .kind = 2, .rows = width, .scratch = 1, .scores = 2, .output = 3, .keep = 4, .keep_nonstate = 5 };
        gcache.entries[0] = .{
            .key = key,
            .used = 1,
            .cohort = 7,
            .exec = .{ .d = undefined, .handle = @ptrFromInt(1) },
            .template = .{ .d = undefined, .handle = @ptrFromInt(2) },
            .binding = try state_binding.Binding.init(2, request),
        };
        var target = request;
        std.mem.swap(u64, &target.layers[0].state_in, &target.layers[0].state_out);
        var flipped = key;
        flipped.seq[0][4] = 200;
        flipped.keep = 9;
        try std.testing.expect(gcache.find(flipped) == null);
        try std.testing.expect(grebind.rebindCandidate(flipped, target, true) == &gcache.entries[0].?);
        gcache.entries[0].?.pending = @ptrFromInt(@alignOf(Pending));
        try std.testing.expect(grebind.rebindCandidate(flipped, target, true) == null);
        gcache.entries[0].?.pending = null;
        gcache.entries[0].?.poisoned = true;
        try std.testing.expect(grebind.rebindCandidate(flipped, target, true) == null);
        gcache.entries[0].?.poisoned = false;
        target.layers[0].state_in = 300;
        try std.testing.expect(grebind.rebindCandidate(flipped, target, true) == null);
        target = request;
        flipped.output += 1;
        try std.testing.expect(grebind.rebindCandidate(flipped, target, true) == null);
        gcache.entries[0].?.binding.?.invalidate();
        try std.testing.expect(grebind.rebindCandidate(key, request, true) == null);
    }
}

test "exl3: state binding transaction uses real graphAgree before publication" {
    if (windows) return error.SkipZigTest; // The rank wire explicitly refuses Windows.
    const Fake = struct {
        calls: usize = 0,
        fail: bool = false,

        pub fn set(self: *@This(), _: usize, _: usize, _: state_binding.Geometry, _: state_binding.Parameters) !void {
            self.calls += 1;
            if (self.fail and self.calls == 2) return error.SetterFailed;
        }
    };
    var request: state_binding.Request = .{
        .width = 2,
        .chain = 10,
        .chain_to = 11,
        .geometry = .{ .grid = .{ 16, 8, 1 }, .block = .{ 128, 1, 1 } },
        .nonstate = .{ .gq = 1, .gk = 2, .gv = 3, .ggb = 4, .rows = 2, .ys = 5 },
        .count = 2,
    };
    request.layers[0] = .{ .layer = 0, .state_in = 100, .state_out = 200 };
    request.layers[1] = .{ .layer = 2, .state_in = 300, .state_out = 400 };
    var target = request;
    for (target.layers[0..target.count]) |*mapping| std.mem.swap(u64, &mapping.state_in, &mapping.state_out);
    for (2..state_binding.max_width + 1) |width| for (0..6) |scenario| {
        request.width = width;
        request.nonstate.rows = @intCast(width);
        target.width = width;
        target.nonstate.rows = @intCast(width);
        var pair: [2]std.posix.socket_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        defer for (pair) |descriptor| {
            _ = std.c.close(descriptor);
        };
        const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
        for (pair) |descriptor| {
            const flags = std.c.fcntl(descriptor, std.c.F.GETFL);
            if (flags < 0 or std.c.fcntl(descriptor, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
        }
        const Peer = struct {
            fn run(link: wire.Link, current: usize) void {
                var control = GraphTestDriver.control(link);
                const selected = control.graphAgree(51, .{
                    .state = if (current == 2) .pending else if (current == 3) .capture else .ready,
                    .cohort = if (current == 1) 8 else 7,
                }) catch @panic("binding selection failed");
                if (selected.choice != .replay) return;
                const result = control.graphAgree(52, .{ .state = if (current == 5) .failed else .ready, .cohort = 7 });
                if (current == 4 or current == 5) {
                    if (result) |_| @panic("binding failure accepted") else |err| std.debug.assert(err == error.GraphInstantiationFailed);
                } else {
                    std.debug.assert((result catch @panic("binding commit failed")).choice == .replay);
                }
            }
        };
        const peer = try std.Thread.spawn(.{}, Peer.run, .{ wire.Link{ .fd = pair[1] }, scenario });
        defer peer.join();
        var control = GraphTestDriver.control(.{ .fd = pair[0] });
        var state = try state_binding.Binding.init(42, request);
        for (request.layers[0..request.count], 0..) |mapping, index| {
            state.nodes[index] = index;
            state.parameters[index] = .{ .nonstate = request.nonstate, .state_in = mapping.state_in, .state_out = mapping.state_out };
        }
        state.loaded = true;
        var fake: Fake = .{ .fail = scenario == 4 };
        const selected = try control.graphAgree(51, .{ .state = .ready, .cohort = 7 });
        var published = false;
        if (selected.choice == .replay) {
            if (scenario == 4 or scenario == 5) {
                try std.testing.expectError(error.GraphInstantiationFailed, state.applyTogether(&fake, target, &control, 52, selected.cohort));
                try std.testing.expect(state.poisoned);
            } else {
                try state.applyTogether(&fake, target, &control, 52, selected.cohort);
                published = true;
            }
        }
        try std.testing.expectEqual(scenario == 0, published);
        try std.testing.expectEqual(@as(usize, if (scenario == 0 or scenario >= 4) 2 else 0), fake.calls);
    };
}

test "exl3: state binding current pass supports exact window layouts through the MTP row bound" {
    try @import("../kv/cuda_exl3_state_request_test.zig").currentPass(Key, grebind.windowStateRequest);
}
