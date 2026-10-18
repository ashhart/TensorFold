//! Constraint mask ownership survives matcher failures on the caller's long-lived allocator.
const std = @import("std");
const grammar = @import("grammar.zig");
const words = 7760;
const Mode = enum { failure, empty, success };
fn fillError(_: *anyopaque, bits: []u32) anyerror!void {
    @memset(bits, 1);
    return error.InjectedFill;
}
fn fillEmpty(_: *anyopaque, bits: []u32) anyerror!void {
    @memset(bits, 0);
}
fn fillSuccess(_: *anyopaque, bits: []u32) anyerror!void {
    @memset(bits, 1);
}
fn check(mode: Mode) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var allocator = std.testing.FailingAllocator.init(arena.allocator(), .{});
    const a = allocator.allocator();
    var seq: grammar.Sequence = .{ .gpa = a, .want = &.{5}, .stop = 2 };
    defer seq.log.deinit(a);
    var vt = seq.rules().vtable.*;
    vt.fill = switch (mode) {
        .failure => fillError,
        .empty => fillEmpty,
        .success => fillSuccess,
    };
    var c = grammar.Constraint.init(.{ .ptr = &seq, .vtable = &vt }, words, null);
    for (0..4) |_| {
        switch (mode) {
            .failure => try std.testing.expectError(error.InjectedFill, c.next(a)),
            .empty => try std.testing.expectError(error.GrammarDeadEnd, c.next(a)),
            .success => {
                const bits = (try c.next(a)).?;
                try std.testing.expectEqual(@as(usize, words), bits.len);
                try std.testing.expectEqual(@as(usize, words * 4), allocator.allocated_bytes - allocator.freed_bytes);
                a.free(bits);
            },
        }
        try std.testing.expectEqual(allocator.allocated_bytes, allocator.freed_bytes);
        try std.testing.expectEqual(allocator.allocations, allocator.deallocations);
        try std.testing.expectEqual(@as(usize, 0), seq.at);
    }
    try std.testing.expectEqual(@as(usize, 4 * 31040), allocator.allocated_bytes);
    c.active = false;
    try std.testing.expect((try c.next(a)) == null);
    try std.testing.expectEqual(@as(usize, 4), allocator.allocations);
}

test "grammar errors: a failed matcher fill releases its whole 27B mask on every attempt" {
    try check(.failure);
}
test "grammar errors: a dead grammar releases its whole 27B mask on every attempt" {
    try check(.empty);
}
test "grammar errors: a successful mask transfers ownership to the caller and inactive grammars allocate nothing" {
    try check(.success);
}
