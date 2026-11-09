//! Mistral /v1/ocr request decoding (field refusals), table forms and the error envelope.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const picture = @import("core").picture;
const mistral = @import("mistral.zig");
const reply = @import("mistral_reply.zig");
const types = @import("types.zig");
const harness = @import("harness_test.zig");
const t = std.testing;
const Allocator = std.mem.Allocator;

fn value(a: Allocator, text: []const u8) !json.Value {
    return switch (try json.parse(a, text)) {
        .ok => |v| v,
        .err => error.TestUnexpectedResult,
    };
}

const facts: mistral.Facts = .{ .fetch = null };

/// A request body around ``doc`` (a document object) and ``extra`` fields.
fn body(a: Allocator, doc: []const u8, extra: []const u8) !json.Value {
    return value(a, try std.fmt.allocPrint(a, "{{\"model\": \"m\", \"document\": {s}{s}}}", .{ doc, extra }));
}

fn imageDoc(a: Allocator) ![]const u8 {
    const png = try harness.png(a, 4, 3);
    const enc = std.base64.standard.Encoder;
    const b64 = try a.alloc(u8, enc.calcSize(png.len));
    return std.fmt.allocPrint(a, "{{\"type\": \"image_url\", \"image_url\": \"data:image/png;base64,{s}\"}}", .{enc.encode(b64, png)});
}

test "ocr mistral: explicit nulls mean absent, and each document form decodes" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fault: types.Fault = undefined;
    const nulls = ", \"pages\": null, \"include_image_base64\": null, \"image_limit\": null, \"image_min_size\": null, \"table_format\": null, " ++
        "\"extract_header\": null, \"extract_footer\": null, \"include_blocks\": null, \"confidence_scores_granularity\": null, " ++
        "\"bbox_annotation_format\": null, \"document_annotation_format\": null, \"document_annotation_prompt\": null, \"id\": \"x\"";
    const doc = try imageDoc(a);
    const r = try mistral.decode(a, try body(a, doc, nulls), facts, &fault);
    try t.expectEqual(mistral.Options{}, r.options);
    try t.expectEqual(r.source.bytes.len, r.doc_size);
    // image_url as {url, detail}, and document_url with an image data URL
    const url = doc[std.mem.indexOf(u8, doc, "data:").? .. doc.len - 2];
    for ([_][]const u8{
        try std.fmt.allocPrint(a, "{{\"type\": \"image_url\", \"image_url\": {{\"url\": \"{s}\", \"detail\": \"high\"}}}}", .{url}),
        try std.fmt.allocPrint(a, "{{\"type\": \"document_url\", \"document_url\": \"{s}\", \"document_name\": \"scan\"}}", .{url}),
    }) |d| {
        const got = try mistral.decode(a, try body(a, d, ""), facts, &fault);
        try t.expectEqualSlices(u8, r.source.bytes, got.source.bytes);
    }
    const set = try mistral.decode(a, try body(a, doc, ", \"table_format\": \"html\", \"extract_header\": true, \"include_blocks\": false, \"image_limit\": 2, \"image_min_size\": 9, \"confidence_scores_granularity\": \"word\""), facts, &fault);
    try t.expectEqual(mistral.Options{ .tables = .html, .header = true, .blocks = false, .image_limit = 2, .image_min_size = 9 }, set.options);
}

test "ocr mistral: content-changing and malformed fields are refused with their loc" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = try imageDoc(a);
    const pdf_doc = "{\"type\": \"document_url\", \"document_url\": \"data:application/pdf;base64,JVBERi0xLjQK\"}";
    const cases = [_]struct { doc: []const u8, extra: []const u8, field: []const u8, says: []const u8 }{
        .{ .doc = doc, .extra = ", \"bbox_annotation_format\": {\"type\": \"json_schema\"}", .field = "bbox_annotation_format", .says = "annotations are not supported" },
        .{ .doc = doc, .extra = ", \"document_annotation_prompt\": \"x\"", .field = "document_annotation_prompt", .says = "annotations are not supported" },
        .{ .doc = doc, .extra = ", \"table_format\": \"csv\"", .field = "table_format", .says = "table_format must be" },
        .{ .doc = doc, .extra = ", \"include_image_base64\": \"yes\"", .field = "include_image_base64", .says = "must be a boolean" },
        .{ .doc = doc, .extra = ", \"image_limit\": -1", .field = "image_limit", .says = "non-negative integer" },
        .{ .doc = doc, .extra = ", \"confidence_scores_granularity\": \"line\"", .field = "confidence_scores_granularity", .says = "must be" },
        .{ .doc = doc, .extra = ", \"pages\": [0]", .field = "pages", .says = "pages selects pages of a PDF document; POST /v1/ocr reads one page image" },
        .{ .doc = doc, .extra = ", \"pages\": \"0-2\"", .field = "pages", .says = "pages selects pages of a PDF document" },
        .{ .doc = doc, .extra = ", \"model\": 3", .field = "model", .says = "model must be" },
        .{ .doc = "{\"type\": \"file\", \"file_id\": \"f\"}", .extra = "", .field = "document", .says = "file documents are not supported; send document_url with a data URL" },
        .{ .doc = "{\"type\": \"document_url\", \"document_url\": \"https://example.com/a.pdf\"}", .extra = "", .field = "document", .says = "--vision-urls" },
        .{ .doc = "{\"type\": \"image_url\", \"image_url\": \"http://example.com/a.png\"}", .extra = "", .field = "document", .says = "public HTTPS URLs" },
        .{ .doc = "{\"type\": \"image_url\", \"image_url\": \"data:image/png;base64,!!\"}", .extra = "", .field = "document", .says = "invalid document data URL encoding" },
        .{ .doc = "{\"type\": \"image_url\", \"image_url\": \"data:text/plain,hello\"}", .extra = "", .field = "document", .says = "the document is not a PNG, JPEG or WebP image" },
        .{ .doc = "{\"type\": \"video_url\"}", .extra = "", .field = "document", .says = "document.type must be" },
        .{ .doc = pdf_doc, .extra = "", .field = "document", .says = "PDF documents are not read on POST /v1/ocr; send each page as a PNG, JPEG or WebP image" },
    };
    for (cases) |c| {
        var fault: types.Fault = undefined;
        try t.expectError(error.Faulted, mistral.decode(a, try body(a, c.doc, c.extra), facts, &fault));
        try t.expectEqual(types.Class.bad_input, fault.class);
        try t.expectEqualStrings("body", fault.loc[0]);
        try t.expectEqualStrings(c.field, fault.loc[1]);
        if (std.mem.indexOf(u8, fault.message, c.says) == null) {
            std.debug.print("{s}: {s}\n", .{ c.field, fault.message });
            return error.TestUnexpectedResult;
        }
    }
    var fault: types.Fault = undefined;
    try t.expectError(error.Faulted, mistral.decode(a, try value(a, "[1]"), facts, &fault));
    try t.expectError(error.Faulted, mistral.decode(a, try value(a, "{\"model\": null}"), facts, &fault));
    try t.expectEqualStrings("document is required", fault.message);
}

test "ocr mistral: tables per table_format, a spanned HTML table kept as HTML only when it leaves the Markdown" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const spanned = "<table><tr><th>a</th><th>b</th></tr><tr><td colspan=\"2\">wide</td></tr></table>";
    const plain = "<table><tr><td>a</td><td>b</td></tr></table>";
    const pipe = "| x | y |\n| --- | --- |\n| 1 | 2 |";
    const inline_spanned = try reply.tableText(a, null, spanned, true);
    try t.expectEqual(reply.TableFormat.markdown, inline_spanned.format);
    try t.expect(std.mem.startsWith(u8, inline_spanned.text, "| a | b |\n| --- | --- |\n| wide |"));
    try t.expectEqualStrings(spanned, (try reply.tableText(a, .markdown, spanned, true)).text);
    try t.expectEqual(reply.TableFormat.html, (try reply.tableText(a, .markdown, spanned, true)).format);
    try t.expectEqualStrings("| a | b |\n| --- | --- |", (try reply.tableText(a, .markdown, plain, true)).text);
    try t.expectEqualStrings(pipe, (try reply.tableText(a, .markdown, pipe, false)).text);
    const as_html = try reply.tableText(a, .html, pipe, false);
    try t.expectEqual(reply.TableFormat.html, as_html.format);
    try t.expectEqualStrings("<table><tr><th>x</th><th>y</th></tr><tr><td>1</td><td>2</td></tr></table>", as_html.text);
    // a table no reader parses keeps its text and its own format
    const broken = "<table><tr><td>a</td>";
    try t.expectEqual(reply.TableFormat.html, (try reply.tableText(a, .markdown, broken, true)).format);
}

test "ocr mistral: every error is an HTTPValidationError body with its status, capacity with Retry-After: 1" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { types.Class, u16, []const u8 }{
        .{ .bad_input, 400, "bad_input" },
        .{ .unreadable, 422, "unreadable_page" },
        .{ .capacity, 503, "capacity" },
        .{ .internal, 500, "internal" },
    };
    for (cases) |c| {
        var wire: harness.Wire = .{ .a = a };
        const conn = try harness.request(a, &wire, "/v1/ocr", null, "");
        const fault: types.Fault = .{ .class = c[0], .message = "why", .loc = if (c[0] == .bad_input) &.{ "body", "pages" } else &.{}, .retry_after = if (c[0] == .capacity) 1 else null };
        reply.sendFault(conn, a, fault);
        try t.expectEqual(c[1], wire.status());
        try t.expect(std.mem.indexOf(u8, wire.head(), "\r\nContent-Type: application/json\r\n") != null);
        try t.expectEqual(c[0] == .capacity, std.mem.indexOf(u8, wire.head(), "\r\nRetry-After: 1\r\n") != null);
        const detail = (try wire.parsed()).get("detail").?.array;
        try t.expectEqual(@as(usize, 1), detail.len);
        try t.expectEqualStrings("why", detail[0].strField("msg").?);
        try t.expectEqualStrings(c[2], detail[0].strField("type").?);
        const loc = detail[0].get("loc").?.array;
        try t.expectEqualStrings("body", loc[0].string);
        try t.expectEqual(@as(usize, if (c[0] == .bad_input) 2 else 1), loc.len);
    }
}

test "ocr mistral: reader kinds map to the block types in docs/api.md OCR: Mistral" {
    try t.expectEqualStrings("title", reply.blockType(.section_header));
    try t.expectEqualStrings("equation", reply.blockType(.formula));
    try t.expectEqualStrings("text", reply.blockType(.footnote));
    try t.expectEqualStrings("header", reply.blockType(.page_header));
    try t.expectEqualStrings("footer", reply.blockType(.page_footer));
    try t.expectEqualStrings("image", reply.blockType(api.ocr.Kind.picture));
}

test "ocr mistral: a PNG whose text chunk says %PDF- is an image, not a PDF" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const png = try harness.pngWithPdfText(a, 4, 3);
    const rgb = try picture.decodeWithLimits(t.io, a, png, .{ .pixels = 1 << 20 });
    try t.expectEqual(@as(usize, 4), rgb.width);
    const enc = std.base64.standard.Encoder;
    const b64 = try a.alloc(u8, enc.calcSize(png.len));
    const doc = try std.fmt.allocPrint(a, "{{\"type\": \"image_url\", \"image_url\": \"data:image/png;base64,{s}\"}}", .{enc.encode(b64, png)});
    var fault: types.Fault = undefined;
    // a PDF document would be refused; the image is read
    const req = try mistral.decode(a, try body(a, doc, ""), facts, &fault);
    try t.expectEqual(png.len, req.source.bytes.len);
}
