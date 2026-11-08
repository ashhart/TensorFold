//! A request's fault from an engine refusal, and the JSON reply with extra headers the dialects send it with.
const std = @import("std");
const types = @import("types.zig");
const errors = @import("../errors.zig");
const http = @import("../http_conn.zig");
const t = std.testing;

test "ocr fault: refusals map to classes, messages copied to the request arena" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { errors.Kind, types.Class, ?u32 }{
        .{ .request, .bad_input, null },
        .{ .context_length, .bad_input, null },
        .{ .unimplemented, .bad_input, null },
        .{ .capacity, .capacity, 1 },
        .{ .other, .internal, null },
        .{ .server, .internal, null },
    };
    for (cases) |c| {
        var page_text = "engine said no".*;
        const cx: errors.Cx = .{ .a = a, .kind = c[0], .message = &page_text };
        var fault: types.Fault = undefined;
        try t.expectEqual(error.Faulted, types.Fault.fromCx(a, &cx, &fault));
        page_text[0] = 'X'; // the page arena's bytes go away; the fault keeps its own copy
        try t.expectEqual(c[1], fault.class);
        try t.expectEqual(c[2], fault.retry_after);
        try t.expectEqualStrings("engine said no", fault.message);
        try t.expectEqual(@as(usize, 0), fault.loc.len);
    }
    var fault: types.Fault = undefined;
    try t.expectEqual(error.Faulted, fault.fail(a, .unreadable, "page {d}: cut", .{3}));
    try t.expectEqual(types.Class.unreadable, fault.class);
    try t.expectEqualStrings("page 3: cut", fault.message);
}

const Wire = struct {
    a: std.mem.Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

fn reply(a: std.mem.Allocator, close: bool, extra: ?[]const [2][]const u8) ![]const u8 {
    var conn = try http.Conn.init(a, -1, "fixture");
    defer conn.deinit();
    var wire: Wire = .{ .a = a };
    conn.write_hooks = .{ .ctx = &wire, .write = Wire.write };
    conn.method = "POST";
    conn.version = "HTTP/1.1";
    conn.close = close;
    if (extra) |e| conn.sendJsonWith(503, "{}", e) else conn.sendJson(503, "{}");
    return wire.bytes.items;
}

test "ocr http: sendJsonWith adds its headers before the body, sendJson none" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const with = try reply(a, true, &.{.{ "Retry-After", "1" }});
    const head = with[0 .. std.mem.indexOf(u8, with, "\r\n\r\n").? + 4];
    try t.expect(std.mem.startsWith(u8, head, "HTTP/1.1 503"));
    try t.expect(std.mem.indexOf(u8, head, "\r\nContent-Type: application/json\r\n") != null);
    try t.expect(std.mem.indexOf(u8, head, "\r\nContent-Length: 2\r\n") != null);
    try t.expect(std.mem.indexOf(u8, head, "\r\nRetry-After: 1\r\n") != null);
    try t.expect(std.mem.indexOf(u8, head, "\r\nConnection: close\r\n") != null);
    try t.expectEqualStrings("{}", with[head.len..]);
    const plain = try reply(a, false, null);
    try t.expect(std.mem.indexOf(u8, plain, "Retry-After") == null);
    try t.expect(std.mem.indexOf(u8, plain, "Connection") == null);
    try t.expect(std.mem.endsWith(u8, plain, "\r\n\r\n{}"));
}
