//! The drift gate must be explicit and paired with Living Weights.
const std = @import("std");
const cli = @import("cli.zig");
const t = std.testing;

test "Slide metrics: CLI keeps the gate off unless explicitly paired with slide" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: cli.Usage = .{};
    try t.expect(!(try cli.parse(a, &.{"m"}, &usage)).slide_drift_gate);
    try t.expect(!(try cli.parse(a, &.{ "m", "--slide" }, &usage)).slide_drift_gate);
    try t.expect((try cli.parse(a, &.{ "m", "--slide", "--slide-drift-gate" }, &usage)).slide_drift_gate);
    try t.expectError(error.Usage, cli.parse(a, &.{ "m", "--slide-drift-gate" }, &usage));
    try t.expect(std.mem.indexOf(u8, usage.message, "needs --slide") != null);
    var output: std.Io.Writer.Allocating = .init(a);
    try cli.capabilities(&output.writer, .{ .version = "test" });
    try t.expect(std.mem.indexOf(u8, output.written(), "\"--slide-drift-gate\"") != null);
}
