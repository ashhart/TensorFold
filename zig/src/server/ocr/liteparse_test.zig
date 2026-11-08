//! LiteParse on POST /ocr: results as its published OCR API spec gives them, the file part, replies and errors.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const multipart = @import("../multipart.zig");
const routes = @import("../routes.zig");
const types = @import("types.zig");
const budget = @import("budget.zig");
const run = @import("run.zig");
const lp = @import("liteparse.zig");
const harness = @import("harness_test.zig");
const Harness = harness.Harness;
const t = std.testing;
const Allocator = std.mem.Allocator;
const Quad = api.ocr.Quad;

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) Quad {
    return .{ .{ x0, y0 }, .{ x1, y0 }, .{ x1, y1 }, .{ x0, y1 } };
}

test "ocr liteparse: a result has text, a float min/max bbox and confidence 1.0, and never a polygon" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try lp.result(a, "Total\t12", .{ .{ 140, 45 }, .{ 30, 45 }, .{ 30, 15 }, .{ 140, 15 } }, 200, 100);
    try t.expectEqualStrings("Total\t12", r.strField("text").?);
    const bbox = r.get("bbox").?.array;
    for (bbox, [_]f64{ 30, 15, 140, 45 }) |v, want| try t.expectEqual(want, v.float);
    try t.expectEqual(@as(f64, 1.0), r.get("confidence").?.float);
    const text = try json.stringify(a, r, .{});
    // a 180-degree quad and a tall band alike: bbox only, as the spec allows (polygon is optional)
    try t.expectEqualStrings("{\"text\": \"Total\\t12\", \"bbox\": [30.0, 15.0, 140.0, 45.0], \"confidence\": 1.0}", text);
    try t.expect((try lp.result(a, "tall", rect(10, 10, 20, 90), 200, 100)).get("polygon") == null);
    // a coordinate is written as its shortest f32 decimal
    const third = try lp.result(a, "y", rect(0, 0, 100.0 / 3.0, 5), 200, 100);
    try t.expectEqual(@as(f64, 33.333332), third.get("bbox").?.array[2].float);
}

fn part(name: []const u8, body: []const u8) multipart.Part {
    return .{ .name = name, .filename = "image.png", .content_type = "image/png", .media = "image/png", .body = body };
}

test "ocr liteparse: the one file part is the page; a PDF, an empty or unknown file and two files are refused" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fault: types.Fault = undefined;
    const png = try harness.png(a, 4, 3);
    const s = try lp.decode(a, &.{ part("language", "en"), part("file", png), part("other", "x") }, &fault);
    try t.expectEqual(.image, s.kind);
    try t.expectEqualStrings("image.png", s.name);
    try t.expectEqual(png.len, s.bytes.len);
    const cases = [_]struct { []const multipart.Part, []const u8 }{
        .{ &.{part("file", "%PDF-1.7\n...")}, lp.no_pdf },
        .{ &.{part("file", "")}, "the file part is empty" },
        .{ &.{part("file", "GIF89a....")}, "the file is not a PNG, JPEG or WebP image" },
        .{ &.{ part("file", png), part("file", png) }, "a LiteParse request has exactly one file part" },
        .{ &.{part("files", png)}, "a LiteParse request has exactly one file part" },
    };
    for (cases) |c| {
        try t.expectError(error.Faulted, lp.decode(a, c[0], &fault));
        try t.expectEqual(types.Class.bad_input, fault.class);
        try t.expectEqualStrings(c[1], fault.message);
    }
}

test "ocr liteparse: a page without text gives an empty results list, and faults map to 400, 422, 503 and 500" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var enc: lp.Encoder = .{ .geometry = .block };
    const sink = enc.sink();
    var pixels: [3]u8 = .{ 0, 0, 0 };
    var blocks = [_]api.ocr.OcrBlock{.{ .content = "", .quad = rect(0, 0, 1, 1), .kind = .picture }};
    const page: types.OcrPage = .{ .index = 0, .width = 1, .height = 1, .dpi = 72, .blocks = &blocks };
    var budget_ = budget.Encodes.init(1, 1);
    var fault: types.Fault = undefined;
    try sink.page(sink.ctx, a, 0, &page, .{ .width = 1, .height = 1, .pixels = &pixels }, &budget_, &fault);
    try t.expectEqualStrings("{\"results\": []}", try enc.reply(a));
    const want = [_]struct { types.Class, u16 }{ .{ .bad_input, 400 }, .{ .unreadable, 422 }, .{ .capacity, 503 }, .{ .internal, 500 } };
    for (want) |w| try t.expectEqual(w[1], lp.status(w[0]));
    try t.expectEqualStrings("{\"error\": \"no \\\"boxes\\\"\"}", try lp.errorBody(a, "no \"boxes\""));
}

const boundary = "XyZ";
const form = "multipart/form-data; boundary=" ++ boundary;

/// LiteParse's request: one ``file`` part (image.png) and a ``language`` part.
fn liteBody(a: Allocator, file: []const u8) ![]const u8 {
    return std.mem.concat(a, u8, &.{
        "--" ++ boundary ++ "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"image.png\"\r\nContent-Type: image/png\r\n\r\n",
        file,
        "\r\n--" ++ boundary ++ "\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\neng\r\n--" ++ boundary ++ "--\r\n",
    });
}

/// A whole POST through the dispatcher; the wire holds the reply.
fn post(a: Allocator, h: *Harness, path: []const u8, ct: []const u8, body: []const u8) !harness.Wire {
    var wire: harness.Wire = .{ .a = a };
    routes.dispatch(h.srv, try harness.request(a, &wire, path, ct, body), a);
    try t.expect(std.mem.indexOf(u8, wire.head(), "\r\nContent-Type: application/json\r\n") != null);
    return wire;
}

fn serve(h: *Harness, replies: []const harness.Script) !void {
    h.* = .{ .replies = replies };
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false });
}

test "ocr liteparse: a whole request gives one result a line in image pixels, bbox only" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = undefined;
    // the frame is 100 x 50 and the page 200 x 100: the reader's boxes scale by 2
    const page_text = "title 10 5 90 10\n# Report\n;;\ntext 10 20 90 40\nline **one**\nline two\n;;\npicture 10 40 50 48\n\n;;\ntext 60 5 65 45\nnarrow";
    try serve(&h, &.{ .{ .text = page_text }, .{ .text = page_text } });
    defer h.stop();
    const body = try liteBody(a, try harness.png(a, 200, 100));
    for ([_][]const u8{ "/ocr", "/ocr?format=liteparse" }) |path| {
        const wire = try post(a, &h, path, form, body);
        try t.expectEqual(@as(u16, 200), wire.status());
        const results = (try wire.parsed()).get("results").?.array;
        const Want = struct { text: []const u8, bbox: [4]f64 };
        const want = [_]Want{
            .{ .text = "Report", .bbox = .{ 20, 10, 180, 20 } },
            .{ .text = "line one", .bbox = .{ 20, 40, 180, 60 } },
            .{ .text = "line two", .bbox = .{ 20, 60, 180, 80 } },
            .{ .text = "narrow", .bbox = .{ 120, 10, 130, 90 } },
        };
        try t.expectEqual(want.len, results.len);
        for (results, want) |r, w| {
            try t.expectEqualStrings(w.text, r.strField("text").?);
            for (r.get("bbox").?.array, w.bbox) |v, x| try t.expectEqual(x, v.float64().?);
            try t.expectEqual(@as(f64, 1.0), r.get("confidence").?.float64().?);
            try t.expectEqual(@as(usize, 3), r.object.count()); // text, bbox, confidence
        }
    }
}

test "ocr liteparse: refusals answer in {\"error\"} with LiteParse's statuses" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = undefined;
    try serve(&h, &.{ .{ .text = "!not the grammar" }, .{ .text = "text 0 0 1 1\nhalf", .reason = .length }, .{ .text = "", .reason = .cancelled } });
    defer h.stop();
    const png = try harness.png(a, 20, 10);
    const body = try liteBody(a, png);
    const Case = struct { path: []const u8 = "/ocr", ct: []const u8 = form, body: []const u8, status: u16, says: []const u8 };
    const cases = [_]Case{
        .{ .body = body, .status = 422, .says = "page 0: the model's output does not follow its format" },
        .{ .body = body, .status = 422, .says = "page 0: the model's output was cut at the context limit" },
        .{ .body = body, .status = 503, .says = "the engine stopped the page; retry" },
        .{ .body = try liteBody(a, "%PDF-1.4\n%fake"), .status = 400, .says = lp.no_pdf },
        .{ .path = "/ocr?format=liteparse", .ct = "application/json", .body = "{}", .status = 400, .says = "LiteParse sends multipart/form-data with one file part" },
    };
    for (cases) |c| {
        const wire = try post(a, &h, c.path, c.ct, c.body);
        try t.expectEqual(c.status, wire.status());
        try t.expectEqualStrings(c.says, (try wire.parsed()).strField("error").?);
        if (c.status == 503) try t.expect(std.mem.indexOf(u8, wire.head(), "\r\nRetry-After: 1\r\n") != null);
    }
    // a reader without boxes, then no reader, then no --vision: 400 before any generation
    h.srv.info.ocr = .{ .prompt = "Text Recognition:", .geometry = .none, .parse = h.reader.?.parse };
    var wire = try post(a, &h, "/ocr", form, body);
    try t.expectEqual(@as(u16, 400), wire.status());
    try t.expectEqualStrings(lp.no_boxes, (try wire.parsed()).strField("error").?);
    h.srv.info.ocr = null;
    wire = try post(a, &h, "/ocr", form, body);
    try t.expectEqualStrings(run.no_reader, (try wire.parsed()).strField("error").?);
    h.srv.info.ocr = h.reader;
    h.srv.info.vision = null;
    wire = try post(a, &h, "/ocr", form, body);
    try t.expectEqual(@as(u16, 400), wire.status());
    try t.expectEqualStrings(run.no_vision, (try wire.parsed()).strField("error").?);
    try t.expectEqual(@as(usize, 3), h.next); // the refusals reached no engine
}

test "ocr liteparse: a PNG whose text chunk says %PDF- is read as the image, not refused as a PDF" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const png = try harness.pngWithPdfText(a, 200, 100);
    var fault: types.Fault = undefined;
    try t.expectEqual(.image, (try lp.decode(a, &.{part("file", png)}, &fault)).kind);
    var h: Harness = undefined;
    try serve(&h, &.{.{ .text = "text 10 5 90 10\nhello" }});
    defer h.stop();
    const wire = try post(a, &h, "/ocr", form, try liteBody(a, png));
    try t.expectEqual(@as(u16, 200), wire.status());
    const results = (try wire.parsed()).get("results").?.array;
    try t.expectEqual(@as(usize, 1), results.len);
    try t.expectEqualStrings("hello", results[0].strField("text").?);
}

/// LiteParse 2514f8e's own request for a 48 x 24 PNG page, byte for byte (fixtures/README.md), and its boundary.
const lit_body = @embedFile("fixtures/lit-2514f8e-ocr.multipart");
const lit_form = "multipart/form-data; boundary=58d2094fd455366b-9209308a2fdfd403-4cb870f78f225bb5-840e3d54fef15603";

fn isNumber(v: json.Value) bool {
    return v == .int or v == .float;
}

test "ocr liteparse: lit's own request gets the reply LiteParse's OCR API spec defines" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = undefined;
    try serve(&h, &.{.{ .text = "text 4 4 50 20\nHello\n;;\ntext 4 25 96 40\nWorld" }});
    defer h.stop();
    const wire = try post(a, &h, "/ocr", lit_form, lit_body);
    try t.expectEqual(@as(u16, 200), wire.status());
    const r = try wire.parsed();
    // OCR_API_SPEC.md (LiteParse 2514f8e): {"results": [{"text", "bbox": [x1, y1, x2, y2], "confidence", "polygon"?}]}
    try t.expectEqual(@as(usize, 1), r.object.count());
    const results = r.get("results").?.array;
    try t.expectEqual(@as(usize, 2), results.len);
    for (results, [_][]const u8{ "Hello", "World" }) |item, text| {
        for (item.object.keys()) |k| try t.expect(std.mem.eql(u8, k, "text") or std.mem.eql(u8, k, "bbox") or std.mem.eql(u8, k, "confidence"));
        try t.expectEqualStrings(text, item.strField("text").?);
        const bbox = item.get("bbox").?.array;
        try t.expectEqual(@as(usize, 4), bbox.len);
        for (bbox) |v| try t.expect(isNumber(v));
        try t.expect(isNumber(item.get("confidence").?));
        // the spec's bbox: x1 < x2 and y1 < y2
        try t.expect(bbox[0].float64().? < bbox[2].float64().? and bbox[1].float64().? < bbox[3].float64().?);
    }
    // boxes are in pixels of the 48 x 24 page LiteParse sent (the harness frame is 100 x 50)
    try t.expect(results[1].get("bbox").?.array[2].float64().? <= 48);
}
