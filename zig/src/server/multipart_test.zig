//! multipart/form-data reader: boundaries, part headers, line-break edges and limits.
const std = @import("std");
const mp = @import("multipart.zig");
const t = std.testing;

fn parse(arena: *std.heap.ArenaAllocator, body: []const u8) ![]mp.Part {
    return mp.parse(arena.allocator(), body, "XyZ");
}

test "ocr multipart: boundary from quoted and token parameters" {
    try t.expectEqualStrings("abc-123", mp.boundary("multipart/form-data; boundary=abc-123").?);
    try t.expectEqualStrings("abc-123", mp.boundary("multipart/form-data;boundary=abc-123 ; charset=x").?);
    try t.expectEqualStrings("a b:c", mp.boundary("multipart/form-data; BOUNDARY=\"a b:c\"").?);
    try t.expectEqualStrings("q", mp.boundary("multipart/form-data; x=\"y;boundary=z\"; boundary=q").?);
    const seventy: [70]u8 = @splat('b');
    try t.expectEqualStrings(&seventy, mp.boundary("multipart/form-data; boundary=" ++ seventy).?);
}

test "ocr multipart: boundary refusals" {
    const too_long: [71]u8 = @splat('b');
    try t.expect(mp.boundary("multipart/form-data") == null);
    try t.expect(mp.boundary("multipart/form-data; charset=utf-8") == null);
    try t.expect(mp.boundary("multipart/form-data; boundary=" ++ too_long) == null);
    try t.expect(mp.boundary("multipart/form-data; boundary=") == null);
    try t.expect(mp.boundary("multipart/form-data; boundary=\"\"") == null);
    try t.expect(mp.boundary("multipart/form-data; boundary=\"ends in space \"") == null);
    try t.expect(mp.boundary("multipart/form-data; boundary=\"unterminated") == null);
    try t.expect(mp.boundary("multipart/form-data; boundary=a\"b") == null);
    try t.expect(mp.boundary("boundary=abc") == null);
}

test "ocr multipart: CRLF parts with a preamble and an epilogue" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const body = "preamble line\r\n--XyZ\r\n" ++
        "Content-Disposition: form-data; name=\"files\"; filename=\"a.jpg\"\r\n" ++
        "Content-Type: image/jpeg\r\n\r\n" ++
        "\xff\xd8\r\nbinary\r\n" ++
        "\r\n--XyZ\r\n" ++
        "Content-Disposition: form-data; name=language\r\n\r\n" ++
        "eng\r\n--XyZ--\r\nepilogue --XyZ\r\n";
    const parts = try parse(&arena, body);
    try t.expectEqual(@as(usize, 2), parts.len);
    try t.expectEqualStrings("files", parts[0].name);
    try t.expectEqualStrings("a.jpg", parts[0].filename.?);
    try t.expectEqualStrings("image/jpeg", parts[0].content_type.?);
    try t.expectEqualStrings("image/jpeg", parts[0].media.?);
    try t.expectEqualStrings("\xff\xd8\r\nbinary\r\n", parts[0].body);
    try t.expectEqualStrings("language", parts[1].name);
    try t.expect(parts[1].filename == null);
    try t.expect(parts[1].content_type == null and parts[1].media == null);
    try t.expectEqualStrings("eng", parts[1].body);
    const base = @intFromPtr(body.ptr);
    try t.expect(@intFromPtr(parts[0].body.ptr) > base and @intFromPtr(parts[0].body.ptr) < base + body.len);
}

test "ocr multipart: bare LF delimiters and headers" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const body = "--XyZ\ncontent-disposition: form-data; name=\"file\"; filename=\"p.png\"\n" ++
        "CONTENT-TYPE: image/png\n\nPNG\n--XyZ \t\ncontent-disposition: form-data; name=empty\n\n\n--XyZ--";
    const parts = try parse(&arena, body);
    try t.expectEqual(@as(usize, 2), parts.len);
    try t.expectEqualStrings("PNG", parts[0].body);
    try t.expectEqualStrings("image/png", parts[0].media.?);
    try t.expectEqualStrings("empty", parts[1].name);
    try t.expectEqualStrings("", parts[1].body);
}

test "ocr multipart: an empty body right after the header block" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const parts = try parse(&arena, "--XyZ\r\nContent-Disposition: form-data; name=a\r\n\r\n--XyZ--");
    try t.expectEqual(@as(usize, 1), parts.len);
    try t.expectEqualStrings("", parts[0].body);
    try t.expectEqual(@as(usize, 0), (try parse(&arena, "--XyZ--\r\n")).len);
}

test "ocr multipart: boundary-like text inside a body stays in the body" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const inner = "a --XyZ\r\n mid\r\n--XyZW not a line\r\n--XyZ- one dash\r\n-XyZ\r\ntail--XyZ";
    const body = "--XyZ\r\nContent-Disposition: form-data; name=a\r\n\r\n" ++ inner ++ "\r\n--XyZ--";
    const parts = try parse(&arena, body);
    try t.expectEqual(@as(usize, 1), parts.len);
    try t.expectEqualStrings(inner, parts[0].body);
}

test "ocr multipart: missing delimiters are malformed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const head = "--XyZ\r\nContent-Disposition: form-data; name=a\r\n\r\n";
    try t.expectError(error.Malformed, parse(&arena, ""));
    try t.expectError(error.Malformed, parse(&arena, "no delimiter at all"));
    try t.expectError(error.Malformed, parse(&arena, head ++ "body without a close"));
    try t.expectError(error.Malformed, parse(&arena, head ++ "body\r\n--XyZ"));
    try t.expectError(error.Malformed, parse(&arena, head ++ "body\r\n--XyZ\r\n"));
    try t.expectError(error.Malformed, parse(&arena, "--XyZ\r\nContent-Disposition: form-data; name=a\r\n"));
    try t.expectError(error.Malformed, mp.parse(arena.allocator(), head ++ "x\r\n--XyZ--", ""));
}

test "ocr multipart: Content-Disposition is required with form-data and a name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const cases = [_][]const u8{
        "Content-Type: text/plain\r\n",
        "Content-Disposition: attachment; name=a\r\n",
        "Content-Disposition: form-data; filename=\"a.png\"\r\n",
        "Content-Disposition: form-data; name=\"a\r\n",
        "Content-Disposition: form-data; name\r\n",
        "Content-Disposition: form-data; name=a; name=b\r\n",
        "Content-Disposition: form-data; name=a\r\nContent-Disposition: form-data; name=b\r\n",
        "Content-Disposition: form-data; name=a\r\nContent-Type: a/b\r\ncontent-type: c/d\r\n",
        "Content-Disposition: form-data;\r\n name=a\r\n",
        "Content-Disposition form-data; name=a\r\n",
    };
    for (cases) |hdr| {
        const body = try std.mem.concat(arena.allocator(), u8, &.{ "--XyZ\r\n", hdr, "\r\nx\r\n--XyZ--" });
        try t.expectError(error.Malformed, parse(&arena, body));
    }
}

test "ocr multipart: quoted parameters with escapes, tokens and filename star" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const body = "--XyZ\r\nContent-Disposition: Form-Data ; name = \"a \\\"q\\\" ; b\" ; " ++
        "filename*=UTF-8''x.png; filename=\"C:\\dir\\\\f.png\"; other=1\r\n\r\nx\r\n--XyZ--";
    const parts = try parse(&arena, body);
    try t.expectEqualStrings("a \"q\" ; b", parts[0].name);
    try t.expectEqualStrings("C:\\dir\\f.png", parts[0].filename.?);
    const blank = "--XyZ\r\nContent-Disposition: form-data; name=\"\"; filename=\"\"\r\n\r\nx\r\n--XyZ--";
    const empty = try parse(&arena, blank);
    try t.expectEqualStrings("", empty[0].name);
    try t.expectEqualStrings("", empty[0].filename.?);
}

test "ocr multipart: Content-Type keeps the raw value and a lower-cased media type" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const body = "--XyZ\r\nContent-Disposition: form-data; name=files\r\n" ++
        "Content-Type:  Image/PNG; charset=binary \r\n\r\nx\r\n" ++
        "--XyZ\r\nContent-Disposition: form-data; name=files\r\ncontent-type:\r\n\r\ny\r\n--XyZ--";
    const parts = try parse(&arena, body);
    try t.expectEqualStrings("Image/PNG; charset=binary", parts[0].content_type.?);
    try t.expectEqualStrings("image/png", parts[0].media.?);
    try t.expectEqualStrings("", parts[1].content_type.?);
    try t.expect(parts[1].media == null);
}

test "ocr multipart: part headers stop at 8 KiB" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const disposition = "Content-Disposition: form-data; name=a\r\n";
    const filler_head = "X-Pad: ";
    // the header block (with its blank line) at exactly the limit passes; one byte more is malformed
    const fit = mp.max_header_bytes - disposition.len - filler_head.len - 4;
    for ([_]usize{ fit, fit + 1 }, [_]bool{ true, false }) |pad, ok| {
        const filler = try a.alloc(u8, pad);
        @memset(filler, 'p');
        const tail = "\r\n\r\nx\r\n--XyZ--";
        const body = try std.mem.concat(a, u8, &.{ "--XyZ\r\n", disposition, filler_head, filler, tail });
        if (ok) {
            try t.expectEqualStrings("x", (try parse(&arena, body))[0].body);
        } else try t.expectError(error.Malformed, parse(&arena, body));
    }
}

test "ocr multipart: at most 256 parts" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const one = "--XyZ\r\nContent-Disposition: form-data; name=files\r\n\r\nx\r\n";
    var body: std.ArrayList(u8) = .empty;
    for (0..mp.max_parts) |_| try body.appendSlice(a, one);
    const fits = try std.mem.concat(a, u8, &.{ body.items, "--XyZ--" });
    try t.expectEqual(@as(usize, mp.max_parts), (try parse(&arena, fits)).len);
    try body.appendSlice(a, one);
    try body.appendSlice(a, "--XyZ--");
    try t.expectError(error.Malformed, parse(&arena, body.items));
}

test "ocr multipart: other part headers and transfer encodings are not interpreted" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const body = "--XyZ\r\nContent-Transfer-Encoding: base64\r\nContent-Disposition: form-data; name=a\r\n" ++
        "Content-Type: multipart/mixed; boundary=inner\r\n\r\naGk=\r\n--XyZ--";
    const parts = try parse(&arena, body);
    try t.expectEqualStrings("aGk=", parts[0].body);
    try t.expectEqualStrings("multipart/mixed", parts[0].media.?);
}
