//! The Mistral Python SDK's own request, replayed byte for byte; its reply checked against the SDK's models.
const std = @import("std");
const json = @import("json");
const routes = @import("../routes.zig");
const harness = @import("harness_test.zig");
const t = std.testing;
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// `mistralai` 3.7.0's `ocr.process` body for an image data URL with `include_image_base64=True` (fixtures/README.md).
const sdk_body = @embedFile("fixtures/mistral-python-3.7.0-ocr-image.json");

// The keys mistralai 3.7.0's models define (mistralai/client/models/ocr*.py: response, page, image, table and blocks).
const response_keys = [_][]const u8{ "pages", "model", "usage_info", "document_annotation" };
const response_required = [_][]const u8{ "pages", "model", "usage_info" };
const page_keys = [_][]const u8{ "index", "markdown", "images", "dimensions", "tables", "hyperlinks", "header", "footer", "confidence_scores", "blocks" };
const page_required = [_][]const u8{ "index", "markdown", "images", "dimensions" };
const image_keys = [_][]const u8{ "id", "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y", "image_base64", "image_annotation" };
const table_keys = [_][]const u8{ "id", "content", "format", "word_confidence_scores" };
const dimension_keys = [_][]const u8{ "dpi", "height", "width" };
const usage_keys = [_][]const u8{ "pages_processed", "doc_size_bytes" };
const box_keys = [_][]const u8{ "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y" };
const block_keys = box_keys ++ [_][]const u8{ "content", "confidence_scores", "type" };
/// ocrpageobject.py `_BLOCK_VARIANTS`: any other `type` parses as `UnknownBlock`.
const block_types = [_][]const u8{ "aside_text", "caption", "code", "equation", "footer", "header", "image", "list", "references", "signature", "table", "text", "title" };

fn oneOf(s: []const u8, set: []const []const u8) bool {
    for (set) |x| if (std.mem.eql(u8, s, x)) return true;
    return false;
}

/// Every key of ``o`` is in ``allowed``, and every key of ``required`` is present.
fn keys(o: Value, allowed: []const []const u8, required: []const []const u8) !void {
    try t.expect(o == .object);
    for (o.object.keys()) |k| if (!oneOf(k, allowed)) {
        std.debug.print("key {s} is not in the SDK's model\n", .{k});
        return error.TestUnexpectedResult;
    };
    for (required) |k| try t.expect(o.object.contains(k));
}

/// The value an SDK field accepts: a bad value in an optional field makes the SDK fail or return UnknownBlock.
const Accepts = enum { string, nullable_string, nullable_int, absent };
const Field = struct { []const u8, Accepts };
const response_values = [_]Field{.{ "document_annotation", .nullable_string }};
const usage_values = [_]Field{.{ "doc_size_bytes", .nullable_int }};
const page_values = [_]Field{ .{ "header", .nullable_string }, .{ "footer", .nullable_string }, .{ "confidence_scores", .absent }, .{ "hyperlinks", .absent } };
const image_values = [_]Field{ .{ "id", .string }, .{ "image_base64", .nullable_string }, .{ "image_annotation", .nullable_string } };
const table_values = [_]Field{ .{ "id", .string }, .{ "content", .string }, .{ "format", .string }, .{ "word_confidence_scores", .absent } };
const block_values = [_]Field{ .{ "type", .string }, .{ "content", .string }, .{ "confidence_scores", .absent }, .{ "image_id", .nullable_string }, .{ "table_id", .nullable_string } };

/// Each field of ``fields`` that ``o`` holds has a value its SDK model accepts; an ``absent`` field is not there.
fn values(o: Value, fields: []const Field) !void {
    for (fields) |f| {
        const v = o.get(f[0]) orelse continue;
        const ok = switch (f[1]) {
            .string => v == .string,
            .nullable_string => v == .string or v == .null,
            .nullable_int => v == .int or v == .null,
            .absent => false,
        };
        if (!ok) std.debug.print("{s} has a value the SDK's model refuses\n", .{f[0]});
        try t.expect(ok);
    }
}

fn int(v: ?Value) !void {
    try t.expect(v != null and v.? == .int);
}

/// A page of every dots.mocr kind in the harness reader's grammar (frame 100 x 50 over the SDK's 48 x 24 PNG).
const page_text =
    "page_header 0 0 100 4\nReport head\n;;\n" ++
    "title 4 4 96 8\n# Title\n;;\n" ++
    "section_header 4 8 96 11\n## Section\n;;\n" ++
    "text 4 11 96 15\nBody **text**.\n;;\n" ++
    "list 4 15 96 19\n- one\n- two\n;;\n" ++
    "table 4 19 96 27\n<table><tr><th>a</th><th>b</th></tr><tr><td colspan=\"2\">w</td></tr></table>\n;;\n" ++
    "formula 4 27 96 31\n$$\nE=mc^2\n$$\n;;\n" ++
    "code 4 31 96 35\n```\nx = 1\n```\n;;\n" ++
    "caption 4 35 50 38\nFigure 1\n;;\n" ++
    "footnote 4 38 50 41\n1. A note\n;;\n" ++
    "picture 60 30 96 46\n\n;;\n" ++
    "page_footer 0 46 100 50\nPage 1";

test "ocr mistral sdk: the Python SDK's own request gets a reply its models parse, with no unknown block" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: harness.Harness = .{ .replies = &.{.{ .text = page_text }} };
    try h.start(.{ .served_name = "m", .model_ids = &.{ "m", "mistral-ocr-latest" }, .enable_thinking = false, .use_drafts = false });
    defer h.stop();
    var wire: harness.Wire = .{ .a = a };
    routes.dispatch(h.srv, try harness.request(a, &wire, "/v1/ocr", "application/json", sdk_body), a);
    const head = wire.head();
    if (wire.status() == 400) { // only a host without libjpeg-turbo, which include_image_base64 needs, skips
        try t.expect(std.mem.indexOf(u8, wire.bytes.items[head.len..], "libjpeg-turbo") != null);
        return error.SkipZigTest;
    }
    try t.expectEqual(@as(u16, 200), wire.status());
    // the SDKs parse a 200 only as application/json
    try t.expect(std.mem.indexOf(u8, head, "\r\nContent-Type: application/json\r\n") != null);
    const r = try wire.parsed();
    try keys(r, &response_keys, &response_required);
    try values(r, &response_values);
    try t.expectEqualStrings("mistral-ocr-latest", r.strField("model").?);
    try t.expect(r.get("document_annotation").? == .null);
    try keys(r.get("usage_info").?, &usage_keys, &.{"pages_processed"});
    try values(r.get("usage_info").?, &usage_values);
    try int(r.get("usage_info").?.get("pages_processed"));
    const pages = r.get("pages").?.array;
    try t.expectEqual(@as(usize, 1), pages.len);
    const page = pages[0];
    try keys(page, &page_keys, &page_required);
    try values(page, &page_values);
    try int(page.get("index"));
    try t.expect(page.get("markdown").? == .string);
    // the TS SDK refuses null for tables and hyperlinks; images is required and never null
    try t.expect(page.get("images").? == .array and page.get("tables").? == .array and page.get("hyperlinks") == null);
    const dims = page.get("dimensions").?;
    try keys(dims, &dimension_keys, &dimension_keys);
    for (dimension_keys) |k| try int(dims.get(k));
    try t.expectEqualStrings("48", dims.get("width").?.int);
    const images = page.get("images").?.array;
    try t.expectEqual(@as(usize, 1), images.len);
    try keys(images[0], &image_keys, &image_keys[0..5].*);
    try values(images[0], &image_values);
    for (box_keys) |k| try int(images[0].get(k));
    try t.expect(std.mem.startsWith(u8, images[0].strField("image_base64").?, "data:image/jpeg;base64,"));
    for (page.get("tables").?.array) |tb| {
        try keys(tb, &table_keys, &table_keys[0..3].*);
        try values(tb, &table_values);
    }
    const blocks = page.get("blocks").?.array;
    try t.expectEqual(@as(usize, 12), blocks.len);
    var seen: std.ArrayList([]const u8) = .empty;
    for (blocks) |b| {
        const kind = b.strField("type") orelse return error.TestUnexpectedResult; // a block without type fails the whole Python parse
        if (!oneOf(kind, &block_types)) {
            std.debug.print("block type {s} would parse as UnknownBlock\n", .{kind});
            return error.TestUnexpectedResult;
        }
        const extra: []const []const u8 = if (std.mem.eql(u8, kind, "image")) &.{"image_id"} else if (std.mem.eql(u8, kind, "table")) &.{"table_id"} else &.{};
        try keys(b, &(block_keys ++ [_][]const u8{ "image_id", "table_id" }), &(box_keys ++ [_][]const u8{ "content", "type" }));
        for (b.object.keys()) |k| if (std.mem.eql(u8, k, "image_id") or std.mem.eql(u8, k, "table_id")) try t.expect(oneOf(k, extra));
        for (box_keys) |k| try int(b.get(k));
        try values(b, &block_values);
        if (std.mem.eql(u8, kind, "image")) try t.expectEqualStrings(images[0].strField("id").?, b.strField("image_id").?);
        if (!oneOf(kind, seen.items)) try seen.append(a, kind);
    }
    // ten distinct SDK block types from the twelve dots.mocr kinds (section headers are titles, footnotes text)
    try t.expectEqual(@as(usize, 10), seen.items.len);
}
