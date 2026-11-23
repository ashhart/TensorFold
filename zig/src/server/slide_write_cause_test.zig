//! A fact graph write the OS refuses names its cause in the client's message and the log line.
const std = @import("std");
const errors = @import("errors.zig");
const slide = @import("slide.zig");
const slide_forget = @import("slide_forget.zig");
const Conn = @import("http_conn.zig").Conn;
const fixture = @import("slide_forget_test.zig");

test "a graph write failure names the OS cause in the client's message" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [160]u8 = undefined;
    const path = try fixture.graphPath(&tmp, &buf, "cause");
    // The staged temporary cannot be renamed onto a directory: a real OS failure, not a planted fault.
    try std.Io.Dir.cwd().createDirPath(std.testing.io, path);
    var text: fixture.ByteText = .{};
    var m: fixture.Model = .{};
    const t = try fixture.taughtAt(&text, &m, path);
    defer t.srv.deinit();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Capture = struct {
        bytes: std.ArrayList(u8) = .empty,
        fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            c.bytes.appendSlice(std.testing.allocator, bytes) catch return error.Closed;
        }
    };
    var capture: Capture = .{};
    defer capture.bytes.deinit(std.testing.allocator);
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .write_hooks = .{ .ctx = &capture, .write = Capture.write } };
    var out: slide.Out = .{ .conn = &conn, .a = a };
    var cx: errors.Cx = .{ .a = a };
    try slide_forget.delete(t.srv, &cx, &out, try t.srv.slide.snapshot(a), 1, "a test", fixture.Stays{});

    const cause = @import("sys").atomic_file.causeName();
    try std.testing.expect(!std.mem.eql(u8, cause, "unknown"));
    const named = try std.fmt.allocPrint(a, "ShardWrite ({s})", .{cause});
    const wire = capture.bytes.items;
    try std.testing.expect(std.mem.indexOf(u8, wire, named) != null);
    try std.testing.expect(std.mem.indexOf(u8, wire, "the graph file could not be written") != null);
}
