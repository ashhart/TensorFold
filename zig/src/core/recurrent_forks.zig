//! Recurrent state blocks forked from the prompt; each update sees one lane's bytes and rolls back on error.
const std = @import("std");

/// Copy one state into N disjoint blocks. No source/destination aliasing and no per-token prompt copy.
pub fn copyInto(prompt: []const u8, destination: []u8, lanes: usize, stride: usize) !void {
    if (lanes == 0 or lanes > 16 or prompt.len == 0 or stride < prompt.len or
        destination.len != try std.math.mul(usize, lanes, stride)) return error.InvalidForkShape;
    const aa = @intFromPtr(prompt.ptr);
    const bb = @intFromPtr(destination.ptr);
    const overlaps = if (aa <= bb) bb - aa < prompt.len else aa - bb < destination.len;
    if (overlaps) return error.AliasedForkStorage;
    @memset(destination, 0);
    for (0..lanes) |i| @memcpy(destination[i * stride ..][0..prompt.len], prompt);
}

pub const LaneStatus = enum { active, finished, canceled };

pub const Forks = struct {
    gpa: std.mem.Allocator,
    bytes: []align(64) u8,
    snapshots: []align(64) u8,
    statuses: [16]LaneStatus = @splat(.active),
    in_flight: u16 = 0,
    lanes: usize,
    state_bytes: usize,
    stride: usize,
    steps: [16]u64 = @splat(0),

    pub fn init(gpa: std.mem.Allocator, prompt: []const u8, lanes: usize) !Forks {
        if (lanes == 0 or lanes > 16 or prompt.len == 0) return error.InvalidForkShape;
        const stride = (try std.math.add(usize, prompt.len, 63)) / 64 * 64;
        const bytes = try gpa.alignedAlloc(u8, .@"64", try std.math.mul(usize, stride, lanes));
        errdefer gpa.free(bytes);
        const snapshots = try gpa.alignedAlloc(u8, .@"64", bytes.len);
        errdefer gpa.free(snapshots);
        try copyInto(prompt, bytes, lanes, stride);
        return .{ .gpa = gpa, .bytes = bytes, .snapshots = snapshots, .lanes = lanes, .state_bytes = prompt.len, .stride = stride };
    }
    pub fn deinit(f: *Forks) void {
        f.gpa.free(f.bytes);
        f.gpa.free(f.snapshots);
        f.* = undefined;
    }
    pub fn lane(f: *Forks, index: usize) ![]u8 {
        if (index >= f.lanes) return error.InvalidLane;
        return f.bytes[index * f.stride ..][0..f.state_bytes];
    }
    pub fn state(f: *Forks, comptime T: type, index: usize) !*T {
        if (@sizeOf(T) != f.state_bytes or @alignOf(T) > 64) return error.InvalidStateType;
        return @ptrCast(@alignCast((try f.lane(index)).ptr));
    }
    pub fn status(f: *const Forks, index: usize) !LaneStatus {
        if (index >= f.lanes) return error.InvalidLane;
        return f.statuses[index];
    }
    pub fn finish(f: *Forks, index: usize) !void {
        if (try f.status(index) == .canceled) return error.LaneCanceled;
        f.statuses[index] = .finished;
    }
    pub fn cancel(f: *Forks, index: usize) !void {
        if (try f.status(index) == .finished) return error.LaneFinished;
        f.statuses[index] = .canceled;
    }
    pub fn cancelAll(f: *Forks) void {
        for (f.statuses[0..f.lanes]) |*s| if (s.* == .active) {
            s.* = .canceled;
        };
    }

    /// A synchronous family update. Failed or canceled work restores bytes under the old step count.
    pub fn advance(f: *Forks, index: usize, token: u32, context: anytype, apply: anytype) !void {
        const data = try f.lane(index);
        switch (f.statuses[index]) {
            .finished => return error.LaneFinished,
            .canceled => return error.LaneCanceled,
            .active => {},
        }
        const bit = @as(u16, 1) << @intCast(index);
        if (f.in_flight & bit != 0) return error.LaneInFlight;
        if (f.steps[index] == std.math.maxInt(u64)) return error.StepCountExhausted;
        const old_steps = f.steps[index];
        const saved = f.snapshots[index * f.stride ..][0..f.state_bytes];
        @memcpy(saved, data);
        f.in_flight |= bit;
        defer f.in_flight &= ~bit;
        errdefer {
            @memcpy(data, saved);
            f.steps[index] = old_steps;
        }
        try apply(context, data, token);
        switch (f.statuses[index]) {
            .finished => return error.LaneFinished,
            .canceled => return error.LaneCanceled,
            .active => {},
        }
        f.steps[index] = old_steps + 1;
    }
};

const State = extern struct { conv: [3]f32, ssm: [4]f32 };
fn step(s: *State, token: u32) void {
    const x: f32 = @floatFromInt(token);
    const h = s.conv[0] * 0.25 + s.conv[1] * -0.5 + s.conv[2] * 0.125 + x;
    s.conv = .{ s.conv[1], s.conv[2], x };
    for (&s.ssm, 0..) |*v, i| v.* = @mulAdd(f32, 0.75, v.*, h * @as(f32, @floatFromInt(i + 1)));
}
fn update(_: void, bytes: []u8, token: u32) !void {
    step(@ptrCast(@alignCast(bytes.ptr)), token);
}

test "each fork is byte-equal to one-sequence recurrence for its own tokens" {
    var prompt = State{ .conv = .{ 0, 0, 0 }, .ssm = .{ 0, 0, 0, 0 } };
    for ([_]u32{ 1, 2, 3, 4, 5 }) |token| step(&prompt, token);
    const saved = prompt;
    for (1..17) |n| {
        var forks = try Forks.init(std.testing.allocator, std.mem.asBytes(&prompt), n);
        defer forks.deinit();
        var ordinary: [16]State = @splat(prompt);
        for (0..23) |round| for (0..n) |lane| {
            const token: u32 = @intCast(7 + round * 19 + lane * 31);
            step(&ordinary[lane], token);
            try forks.advance(lane, token, {}, update);
            for (0..n) |i| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&ordinary[i]), try forks.lane(i));
        };
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&saved), std.mem.asBytes(&prompt));
        for (0..n) |i| try std.testing.expectEqual(@as(u64, 23), forks.steps[i]);
        try std.testing.expectError(error.InvalidLane, forks.lane(n));
    }
}

test "caller-owned recurrent blocks support per-layer destinations and reject aliasing" {
    const prompt = [_]u8{ 1, 2, 3, 4, 5 };
    var out: [24]u8 = undefined;
    try copyInto(&prompt, &out, 3, 8);
    for (0..3) |i| {
        try std.testing.expectEqualSlices(u8, &prompt, out[i * 8 ..][0..5]);
        try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0 }, out[i * 8 + 5 ..][0..3]);
    }
    try std.testing.expectError(error.AliasedForkStorage, copyInto(out[0..5], &out, 3, 8));
    try std.testing.expectError(error.InvalidForkShape, copyInto(&prompt, &out, 3, 4));
}

fn dirtyFailure(_: void, bytes: []u8, _: u32) !void {
    @memset(bytes, 0xee);
    return error.UpdateFailed;
}
const CancelContext = struct { forks: *Forks, lane: usize };
fn dirtyCancel(ctx: CancelContext, bytes: []u8, _: u32) !void {
    @memset(bytes, 0xdd);
    try ctx.forks.cancel(ctx.lane);
}
fn reenter(ctx: CancelContext, _: []u8, token: u32) !void {
    try ctx.forks.advance(ctx.lane, token, {}, dirtyFailure);
}

test "failed updates restore bytes and allow the next valid token without changing other lanes" {
    var prompt = State{ .conv = .{ 1, 2, 3 }, .ssm = .{ 4, 5, 6, 7 } };
    var f = try Forks.init(std.testing.allocator, std.mem.asBytes(&prompt), 16);
    defer f.deinit();
    const original = try std.testing.allocator.dupe(u8, f.bytes);
    defer std.testing.allocator.free(original);
    try std.testing.expectError(error.UpdateFailed, f.advance(7, 99, {}, dirtyFailure));
    try std.testing.expectEqualSlices(u8, original, f.bytes);
    try std.testing.expectEqual(@as(u64, 0), f.steps[7]);
    try std.testing.expectEqual(@as(u16, 0), f.in_flight);
    var expected = prompt;
    step(&expected, 23);
    try f.advance(7, 23, {}, update);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&expected), try f.lane(7));
    try std.testing.expectEqual(@as(u64, 1), f.steps[7]);
    try std.testing.expectEqualSlices(u8, original[0 .. f.stride * 7], f.bytes[0 .. f.stride * 7]);
    try std.testing.expectEqualSlices(u8, original[f.stride * 8 ..], f.bytes[f.stride * 8 ..]);
}

test "finished and canceled lanes stay unchanged while other lanes advance" {
    var prompt = State{ .conv = .{ 0, 0, 0 }, .ssm = .{ 0, 0, 0, 0 } };
    var f = try Forks.init(std.testing.allocator, std.mem.asBytes(&prompt), 3);
    defer f.deinit();
    try f.advance(0, 11, {}, update);
    const stopped = (try f.state(State, 0)).*;
    try f.finish(0);
    try f.finish(0);
    try std.testing.expectError(error.LaneFinished, f.advance(0, 12, {}, update));
    try f.cancel(1);
    try std.testing.expectError(error.LaneCanceled, f.advance(1, 13, {}, update));
    for ([_]u32{ 14, 15, 16 }) |t| try f.advance(2, t, {}, update);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&stopped), try f.lane(0));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&prompt), try f.lane(1));
    try std.testing.expectEqualSlices(u64, &.{ 1, 0, 3 }, f.steps[0..3]);
    f.cancelAll();
    try std.testing.expectEqual(LaneStatus.finished, try f.status(0));
    try std.testing.expectError(error.LaneCanceled, f.advance(2, 17, {}, update));
    try std.testing.expectError(error.InvalidLane, f.finish(3));
    try std.testing.expectError(error.InvalidLane, f.cancel(3));
}

test "cancel during a dirty callback and reentry both roll back without counting a token" {
    const prompt = [_]u8{ 1, 2, 3, 4 };
    var f = try Forks.init(std.testing.allocator, &prompt, 2);
    defer f.deinit();
    try std.testing.expectError(error.LaneCanceled, f.advance(0, 1, CancelContext{ .forks = &f, .lane = 0 }, dirtyCancel));
    try std.testing.expectEqualSlices(u8, &prompt, try f.lane(0));
    try std.testing.expectEqual(@as(u64, 0), f.steps[0]);
    try std.testing.expectError(error.LaneInFlight, f.advance(1, 2, CancelContext{ .forks = &f, .lane = 1 }, reenter));
    try std.testing.expectEqualSlices(u8, &prompt, try f.lane(1));
    try std.testing.expectEqual(@as(u16, 0), f.in_flight);
    f.steps[1] = std.math.maxInt(u64);
    try std.testing.expectError(error.StepCountExhausted, f.advance(1, 3, {}, dirtyFailure));
    try std.testing.expectEqualSlices(u8, &prompt, try f.lane(1));
}
