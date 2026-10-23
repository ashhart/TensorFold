//! Kernel enumeration must not change a named fixture's seed.
const std = @import("std");
const g = @import("gates.zig");
const c = @import("contract.zig");
const Probe = struct {
    seed: u64 = 0,
    fn prepare(p: *anyopaque, _: c.Shape, _: g.Fill, s: u64) anyerror!bool {
        const self: *Probe = @ptrCast(@alignCast(p));
        self.seed = s;
        return false;
    }
    fn run(_: *anyopaque) anyerror!void {
        unreachable;
    }
    fn bytes(_: *anyopaque) anyerror![]const u8 {
        unreachable;
    }
    fn intact(_: *anyopaque) anyerror!bool {
        unreachable;
    }
    fn overlap(_: *anyopaque) bool {
        unreachable;
    }
    fn count(_: *anyopaque) usize {
        unreachable;
    }
    fn reference(_: *anyopaque, _: []f64, _: []f64) anyerror!void {
        unreachable;
    }
    fn depth(_: c.Shape) usize {
        return 0;
    }
    fn kernel(p: *Probe, name: []const u8) g.Kernel {
        return .{ .spec = .{ .name = name, .accum = .none, .out = .raw, .depth = depth, .served = &.{.{ .rows = 1, .n = 1, .k = 1 }}, .tiles = .{ .{}, .{}, .{} } }, .ctx = p, .vt = &.{ .prepare = prepare, .run = run, .output = bytes, .inputs = bytes, .intact = intact, .overlap = overlap, .count = count, .reference = reference } };
    }
};
test "contract seed: an inserted kernel cannot change another named fixture's seed" {
    const was = g.quiet;
    g.quiet = true;
    defer g.quiet = was;
    var target: Probe = .{};
    var inserted: Probe = .{};
    var tally: g.Tally = .{};
    try g.runAll(std.testing.allocator, &.{target.kernel("unchanged fixture")}, 0x6b63, &tally);
    const before = target.seed;
    try g.runAll(std.testing.allocator, &.{ inserted.kernel("new fixture"), target.kernel("unchanged fixture") }, 0x6b63, &tally);
    try std.testing.expectEqual(before, target.seed);
    try std.testing.expect(inserted.seed != target.seed);
}
