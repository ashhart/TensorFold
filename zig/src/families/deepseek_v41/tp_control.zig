//! Rank zero orders lane operations; both ranks acknowledge the resulting cursor.
const std = @import("std");
const wire = @import("../glm/ep_control.zig");
pub const BYTES = wire.BYTES;
pub const Kind = enum(u32) { reset = 1, begin, commit, rollback, forward };
pub const Command = struct {
    kind: Kind,
    slot: u32,
    position: u32,
    logits: u32 = 0,
    tokens: []const u32 = &.{},
};
pub const Limits = struct { lanes: u32, rows: u32, capacity: u32 };
pub const Executor = struct {
    ptr: *anyopaque,
    /// Refuse an unexpected starting cursor before touching state; return the resulting cursor.
    apply: *const fn (*anyopaque, Command) anyerror!u32,
};

pub fn decode(kind: u32, words: []const u32, limits: Limits) !Command {
    const k: Kind = switch (kind) {
        1...5 => @fromBackingInt(@intCast(kind)),
        else => return error.InvalidTensorParallelCommand,
    };
    if (words.len < 2 or words[0] >= limits.lanes or words[1] > limits.capacity)
        return error.InvalidTensorParallelCommand;
    var c: Command = .{ .kind = k, .slot = words[0], .position = words[1] };
    if (k == .forward) {
        if (words.len < 4 or words[2] > 2 or words.len - 3 > limits.rows or words.len - 3 > limits.capacity - c.position)
            return error.InvalidTensorParallelCommand;
        c.logits = words[2];
        c.tokens = words[3..];
        for (c.tokens) |id| if (id >= 129280) return error.InvalidTensorParallelCommand;
    } else if (words.len != 2) return error.InvalidTensorParallelCommand;
    return c;
}

pub const Control = struct {
    wire: *wire.Control,
    failed: *std.atomic.Value(bool),
    limits: Limits,
    executor: Executor,
    scratch: []u32,

    pub fn run(c: *Control, command: Command) !void {
        if (c.failed.load(.acquire)) return error.InvalidTensorParallelState;
        if (c.wire.rank != 0) return error.FollowsPeer;
        errdefer c.failed.store(true, .release);
        const n: usize = if (command.kind == .forward) 3 + command.tokens.len else 2;
        if (n > c.scratch.len) return error.InvalidTensorParallelCommand;
        const words = c.scratch[0..n];
        words[0] = command.slot;
        words[1] = command.position;
        if (command.kind == .forward) {
            words[2] = command.logits;
            @memcpy(words[3..], command.tokens);
        } else if (command.tokens.len != 0 or command.logits != 0) return error.InvalidTensorParallelCommand;
        const checked = try decode(@backingInt(command.kind), words, c.limits);
        try c.wire.sendCommand(@backingInt(command.kind), words);
        const position = try c.executor.apply(c.executor.ptr, checked);
        if (try c.wire.waitReplyValue() != position) return error.TensorParallelHistoryMismatch;
    }

    pub fn follow(c: *Control) !void {
        if (c.wire.rank != 1) return error.InvalidTensorParallelRank;
        errdefer c.failed.store(true, .release);
        while (try c.wire.waitCommand()) |request| {
            const command = switch (request) {
                .lanes => |v| decode(v.kind, v.words, c.limits) catch |err| {
                    c.wire.reply(false) catch {};
                    return err;
                },
                .request => {
                    c.wire.reply(false) catch {};
                    return error.InvalidTensorParallelCommand;
                },
            };
            const position = c.executor.apply(c.executor.ptr, command) catch |err| {
                c.wire.reply(false) catch {};
                return err;
            };
            try c.wire.replyValue(position);
        }
    }
};

test "TP control validates the complete command before model mutation" {
    const limits: Limits = .{ .lanes = 2, .rows = 3, .capacity = 8 };
    for ([_][]const u32{ &.{}, &.{0}, &.{ 2, 0 }, &.{ 0, 9 }, &.{ 0, 0, 0 } }) |bad|
        try std.testing.expectError(error.InvalidTensorParallelCommand, decode(1, bad, limits));
    for ([_][]const u32{ &.{ 0, 0, 0 }, &.{ 0, 0, 3, 1 }, &.{ 0, 0, 0, 129280 }, &.{ 0, 8, 0, 1 }, &.{ 0, 0, 0, 1, 2, 3, 4 } }) |bad|
        try std.testing.expectError(error.InvalidTensorParallelCommand, decode(5, bad, limits));
    try std.testing.expectError(error.InvalidTensorParallelCommand, decode(0, &.{ 0, 0 }, limits));
    const c = try decode(5, &.{ 1, 5, 2, 1, 2, 3 }, limits);
    try std.testing.expectEqual(@as(u32, 5), c.position);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, c.tokens);
}

test "TP control mirrors transactions and rejects a divergent peer cursor" {
    const fabric = @import("fabric");
    const a = std.testing.allocator;
    const cluster = try fabric.fake.Cluster.init(a, 2, BYTES, .immediate, null);
    defer cluster.deinit();
    var failed = [_]std.atomic.Value(bool){ .init(false), .init(false) };
    var wires = [_]wire.Control{
        try wire.Control.init(a, cluster.endpoint(0), 0, &failed[0]),
        try wire.Control.init(a, cluster.endpoint(1), 0, &failed[1]),
    };
    defer for (&wires) |*w| w.deinit(a);
    const State = struct {
        position: u32 = 0,
        saved: ?u32 = null,
        calls: u32 = 0,
        fn apply(ptr: *anyopaque, command: Command) !u32 {
            const s: *@This() = @ptrCast(@alignCast(ptr));
            if (s.position != command.position) return error.TensorParallelHistoryMismatch;
            switch (command.kind) {
                .reset => {
                    s.position = 0;
                    s.saved = null;
                },
                .begin => s.saved = s.position,
                .commit => {},
                .rollback => s.position = s.saved orelse return error.NoTransaction,
                .forward => s.position += @intCast(command.tokens.len),
            }
            s.calls += 1;
            return s.position;
        }
    };
    var states = [_]State{ .{}, .{} };
    var scratch: [2][6]u32 = undefined;
    var controls: [2]Control = undefined;
    for (0..2) |i| controls[i] = .{ .wire = &wires[i], .failed = &failed[i], .limits = .{ .lanes = 1, .rows = 3, .capacity = 16 }, .executor = .{ .ptr = &states[i], .apply = State.apply }, .scratch = &scratch[i] };
    const Worker = struct {
        control: *Control,
        result: ?anyerror = null,
        fn run(w: *@This()) void {
            w.control.follow() catch |err| {
                w.result = err;
            };
        }
    };
    var worker: Worker = .{ .control = &controls[1] };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    defer {
        wires[1].stop.store(true, .release);
        thread.join();
    }
    try controls[0].run(.{ .kind = .reset, .slot = 0, .position = 0 });
    try controls[0].run(.{ .kind = .begin, .slot = 0, .position = 0 });
    try controls[0].run(.{ .kind = .forward, .slot = 0, .position = 0, .tokens = &.{ 1, 2, 3 }, .logits = 2 });
    try controls[0].run(.{ .kind = .commit, .slot = 0, .position = 3 });
    try controls[0].run(.{ .kind = .rollback, .slot = 0, .position = 3 });
    try std.testing.expectEqual(@as(u32, 0), states[0].position);
    try std.testing.expectEqual(states[0].calls, states[1].calls);
    // Only rank zero's cursor is changed. Rank one must refuse before applying another operation.
    states[0].position = 1;
    try std.testing.expectError(error.EpOutOfStep, controls[0].run(.{ .kind = .reset, .slot = 0, .position = 1 }));
    try std.testing.expectError(error.InvalidTensorParallelState, controls[0].run(.{ .kind = .reset, .slot = 0, .position = 0 }));
    try std.testing.expectEqual(@as(u32, 5), states[1].calls);
}

test "TP control refuses reuse after an absent peer fails to acknowledge execution" {
    const fabric = @import("fabric");
    const a = std.testing.allocator;
    const cluster = try fabric.fake.Cluster.init(a, 2, BYTES, .immediate, null);
    defer cluster.deinit();
    var failed: std.atomic.Value(bool) = .init(false);
    var w = try wire.Control.init(a, cluster.endpoint(0), 0, &failed);
    defer w.deinit(a);
    w.limit = 24_000; // 1 ms on Apple silicon, the wire's monotonic clock.
    const State = struct {
        calls: usize = 0,
        fn apply(ptr: *anyopaque, _: Command) !u32 {
            const s: *@This() = @ptrCast(@alignCast(ptr));
            s.calls += 1;
            return 0;
        }
    };
    var state: State = .{};
    var scratch: [4]u32 = undefined;
    var c: Control = .{ .wire = &w, .failed = &failed, .limits = .{ .lanes = 1, .rows = 1, .capacity = 8 }, .executor = .{ .ptr = &state, .apply = State.apply }, .scratch = &scratch };
    try std.testing.expectError(error.EpPeerSilent, c.run(.{ .kind = .reset, .slot = 0, .position = 0 }));
    try std.testing.expectEqual(@as(usize, 1), state.calls);
    try std.testing.expectError(error.InvalidTensorParallelState, c.run(.{ .kind = .reset, .slot = 0, .position = 0 }));
    try std.testing.expectEqual(@as(usize, 1), state.calls);
}
