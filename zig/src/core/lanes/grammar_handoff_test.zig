//! A grammar compiled once (the server's) gives the engine's constraint without a second compile, mask for mask.
const std = @import("std");
const grammar = @import("grammar.zig");
const fake = @import("grammar_fake.zig");

/// The masks of `steps` tokens, each step following the first token its mask allows.
fn walk(a: std.mem.Allocator, c: *grammar.Constraint, steps: usize) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    for (0..steps) |_| {
        const bits = (try c.next(a)).?;
        try out.appendSlice(a, bits);
        const t: u32 = for (0..64) |t| {
            if (bits[t / 32] >> @intCast(t % 32) & 1 == 1) break @intCast(t);
        } else unreachable;
        try std.testing.expect(try c.follow(t));
    }
    return out.toOwnedSlice(a);
}

test "a compiled grammar's constraint compiles nothing and walks the masks a fresh compile walks" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const old = fake.install();
    defer _ = grammar.swapLibrary(old);
    var c = fake.compiler();
    var err: [512]u8 = undefined;
    const g = try c.compile(std.testing.io, .json_schema, "{\"type\":\"object\"}", &err);
    try std.testing.expectEqual(@as(usize, 1), fake.compiles);
    const handed = try c.constrained(gpa, g, null);
    defer handed.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), fake.compiles); // no second compile, so no lock either
    const fresh = try c.constraint(gpa, std.testing.io, .json_schema, "{\"type\":\"object\"}", null);
    defer fresh.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), fake.compiles);
    try std.testing.expectEqual(@as(usize, 1), fake.frees); // the fresh compile is freed, the handed one stays the caller's
    try std.testing.expectEqualSlices(u32, try walk(arena.allocator(), fresh, 7), try walk(arena.allocator(), handed, 7));
    g.free();
}

test "another grammar's masks differ, so equal walks are not vacuous" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const old = fake.install();
    defer _ = grammar.swapLibrary(old);
    var c = fake.compiler();
    const x = try c.constraint(gpa, std.testing.io, .regex, "a+", null);
    defer x.deinit(gpa);
    const y = try c.constraint(gpa, std.testing.io, .regex, "ab+", null);
    defer y.deinit(gpa);
    const wx = try walk(arena.allocator(), x, 3);
    const wy = try walk(arena.allocator(), y, 3);
    try std.testing.expect(!std.mem.eql(u32, wx, wy));
}
