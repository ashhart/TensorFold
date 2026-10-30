//! The real frame owner is exercised with a queued-copy recorder over owned memory, never a model fallback.
const std = @import("std");
const cfg = @import("config.zig");
const a = std.testing.allocator;
pub const Backend = struct {
    pub const Buffer = []align(16) u8;
    live: usize = 0,
    allocs: usize = 0,
    copies: usize = 0,
    fail_copy: bool = false,
    fail_drain: bool = false,
    pub fn allocationBytes(n: usize) !u64 {
        return n;
    }
    pub fn allocate(b: *Backend, n: usize) !Buffer {
        const out = try a.alignedAlloc(u8, .@"16", n);
        b.live += 1;
        b.allocs += 1;
        @memset(out, 0xcc);
        return out;
    }
    pub fn release(b: *Backend, buffer: Buffer) void {
        a.free(buffer);
        b.live -= 1;
    }
    pub fn zero(_: *Backend, buffer: Buffer, offset: usize, n: usize) !void {
        @memset(buffer[offset..][0..n], 0);
    }
    pub fn copy(b: *Backend, to: Buffer, to_offset: usize, from: Buffer, from_offset: usize, n: usize) !void {
        b.copies += 1;
        if (b.fail_copy) return error.PlantedCopy;
        @memcpy(to[to_offset..][0..n], from[from_offset..][0..n]);
    }
    pub fn drain(b: *Backend) !void {
        if (b.fail_drain) return error.PlantedDrain;
    }
};
const State = @import("state.zig").State(Backend);
pub fn complete(s: *State, tokens: []const u32) !void {
    for (0..s.rows) |row| {
        const view = try s.row(row, .{ .bytes = s.layout.state_bytes });
        @memset(view.buffer[view.offset..][0..view.len], @intCast(17 + row));
    }
    for (0..s.layers()) |layer| try s.captured(layer);
    try s.advanced(tokens);
}
fn stateIs(s: *State, value: u8) !void {
    const v = try s.current(.{ .bytes = s.layout.state_bytes });
    for (v.buffer[v.offset..][0..v.len]) |byte| try std.testing.expectEqual(value, byte);
}
test "Qwen4 frames: every partial keep and rejection picks its exact frame; another stream remains unchanged" {
    var problem: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &problem);
    defer c.deinit();
    var b: Backend = .{};
    var s = try State.open(a, &b, &c, 32, 4, std.math.maxInt(u64));
    defer s.close() catch unreachable;
    var other = try State.open(a, &b, &c, 32, 4, std.math.maxInt(u64));
    defer other.close() catch unreachable;
    try other.reset();
    for (0..5) |keep| {
        try s.reset();
        try stateIs(&s, 0);
        try s.begin(4);
        try complete(&s, &.{ 1, 2, 3, 4 });
        try s.keep(keep);
        try std.testing.expectEqual(keep, s.pos);
        try stateIs(&s, if (keep == 0) 0 else @intCast(16 + keep));
        try std.testing.expectEqualSlices(u32, (&[_]u32{ 1, 2, 3, 4 })[0..keep], s.history());
        try stateIs(&other, 0);
        try std.testing.expectEqual(@as(usize, 0), other.pos);
    }
    try s.begin(2);
    try complete(&s, &.{ 5, 6 });
    try s.abort();
    try std.testing.expectEqual(@as(usize, 4), s.pos);
    try stateIs(&s, 20);
}
test "Qwen4 frames: missing rows, cancellation, phase, capacity and failed drains preserve ownership" {
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/tiny-dense.json"), &p);
    defer c.deinit();
    var b: Backend = .{};
    try std.testing.expectError(error.Qwen4StateBudget, State.open(a, &b, &c, 32, 4, 0));
    try std.testing.expectEqual(@as(usize, 0), b.allocs);
    var s = try State.open(a, &b, &c, 32, 4, std.math.maxInt(u64));
    try std.testing.expectError(error.Qwen4StatePhase, s.begin(1));
    try s.reset();
    const n = b.copies;
    try std.testing.expectError(error.Qwen4StateCapacity, s.begin(5));
    try std.testing.expectEqual(n, b.copies);
    try s.begin(1);
    try std.testing.expectError(error.Qwen4StateMissingRow, s.advanced(&.{1}));
    try std.testing.expectEqual(@as(usize, 0), s.pos);
    s.poison();
    try s.abort();
    try stateIs(&s, 0);
    b.fail_copy = true;
    try std.testing.expectError(error.PlantedCopy, s.begin(1));
    try std.testing.expectError(error.Qwen4StateMissingBase, s.abort());
    b.fail_copy = false;
    try s.reset();
    try s.begin(1);
    try complete(&s, &.{7});
    b.fail_copy = true;
    try std.testing.expectError(error.PlantedCopy, s.keep(1));
    try std.testing.expectError(error.Qwen4StatePhase, s.begin(1));
    b.fail_copy = false;
    try s.abort();
    try stateIs(&s, 0);
    b.fail_drain = true;
    try std.testing.expectError(error.PlantedDrain, s.close());
    try std.testing.expectEqual(@as(usize, 1), b.live);
    b.fail_drain = false;
    try s.close();
    try std.testing.expectEqual(@as(usize, 0), b.live);
}
