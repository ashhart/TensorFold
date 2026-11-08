//! The /ocr dialect choice (docs/api.md OCR) and the refusals sent before a dialect is known.
const std = @import("std");
const route = @import("route.zig");
const routes = @import("../routes.zig");
const auth_scope = @import("../auth_scope.zig");
const harness = @import("harness_test.zig");
const t = std.testing;

const form = "multipart/form-data; boundary=XyZ";

fn parts(a: std.mem.Allocator, names: []const []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (names) |n| try out.print(a, "--XyZ\r\nContent-Disposition: form-data; name=\"{s}\"; filename=\"p.png\"\r\nContent-Type: image/png\r\n\r\nbytes\r\n", .{n});
    try out.appendSlice(a, "--XyZ--\r\n");
    return out.items;
}

test "ocr route: the format query, then the media type, then the part names choose the dialect" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { path: []const u8 = "/ocr", ct: ?[]const u8, body: []const u8 = "", want: ?route.Dialect, status: u16 = 0 };
    const cases = [_]Case{
        .{ .path = "/ocr?format=liteparse", .ct = "application/json", .want = .liteparse },
        .{ .path = "/ocr?x=1&format=doctr", .ct = null, .want = .doctr },
        .{ .path = "/ocr?format=tesseract", .ct = "application/json", .want = null, .status = 400 },
        .{ .path = "/ocr?format=%zz", .ct = "application/json", .want = null, .status = 400 },
        .{ .ct = "application/json", .want = null, .status = 415 },
        .{ .ct = null, .body = "  {\"file\": \"x\"}", .want = null, .status = 415 },
        .{ .ct = null, .body = "file=x", .want = null, .status = 415 },
        .{ .ct = "text/plain", .want = null, .status = 415 },
        .{ .ct = form, .body = try parts(a, &.{ "files", "files" }), .want = .doctr },
        .{ .ct = form, .body = try parts(a, &.{ "file", "files" }), .want = null, .status = 400 },
        .{ .ct = form, .body = try parts(a, &.{"file"}), .want = .liteparse },
        .{ .ct = form, .body = try parts(a, &.{ "file", "file" }), .want = null, .status = 400 },
        .{ .ct = form, .body = try parts(a, &.{"image"}), .want = null, .status = 400 },
        .{ .ct = "multipart/form-data", .body = try parts(a, &.{"file"}), .want = null, .status = 400 },
        .{ .ct = form, .body = "not multipart", .want = null, .status = 400 },
    };
    for (cases) |c| switch (try route.select(a, c.path, c.ct, c.body)) {
        .ok => |s| {
            try t.expectEqual(c.want, s.dialect);
            if (c.status == 0 and c.body.len > 0) try t.expect(s.parts != null);
        },
        .refused => |r| {
            try t.expectEqual(@as(?route.Dialect, null), c.want);
            try t.expectEqual(c.status, r.status);
        },
    };
}

test "ocr route: files and file parts together are refused on every path, the format query included" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const both = try parts(a, &.{ "files", "file" });
    switch (try route.select(a, "/ocr", form, both)) {
        .ok => return error.TestUnexpectedResult,
        .refused => |r| {
            try t.expectEqual(@as(u16, 400), r.status);
            try t.expectEqualStrings(route.mixed_parts, r.message);
        },
    }
    // the format query picks the dialect, which refuses the same body in its own envelope rather than drop parts
    var h: harness.Harness = .{};
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false });
    defer h.stop();
    const cases = [_]struct { []const u8, []const u8 }{ .{ "/ocr", "error" }, .{ "/ocr?format=doctr", "detail" }, .{ "/ocr?format=liteparse", "error" } };
    for (cases) |c| {
        var wire: harness.Wire = .{ .a = a };
        routes.dispatch(h.srv, try harness.request(a, &wire, c[0], form, both), a);
        try t.expectEqual(@as(u16, 400), wire.status());
        const got = (try wire.parsed()).get(c[1]).?;
        try t.expectEqualStrings(route.mixed_parts, if (got == .string) got.string else got.strField("message").?);
    }
    try t.expectEqual(@as(usize, 0), h.next);
}

test "ocr route: /ocr answers in the OpenAI shape until a dialect is chosen; both paths need the inference scope" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: harness.Harness = .{};
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false });
    defer h.stop();
    const cases = [_]struct { path: []const u8, ct: ?[]const u8, status: u16, says: []const u8 }{
        .{ .path = "/ocr", .ct = "text/csv", .status = 415, .says = "unsupported Content-Type text/csv" },
        .{ .path = "/ocr/", .ct = form, .status = 400, .says = "multipart body is malformed" },
    };
    for (cases) |c| {
        var wire: harness.Wire = .{ .a = a };
        routes.dispatch(h.srv, try harness.request(a, &wire, c.path, c.ct, "{}"), a);
        try t.expectEqual(c.status, wire.status());
        const message = (try wire.parsed()).get("error").?.strField("message").?;
        try t.expect(std.mem.indexOf(u8, message, c.says) != null);
    }
    const infer: auth_scope.Scopes = .{ .inference = true };
    const read: auth_scope.Scopes = .{ .read = true };
    for ([_][]const u8{ "/ocr", "/ocr/", "/v1/ocr", "/ocr?format=doctr" }) |p| {
        try t.expect(auth_scope.allows(infer, "POST", p));
        try t.expect(!auth_scope.allows(read, "POST", p));
    }
}
