//! PaddleX on POST /ocr: lenient base64, schema and knob refusals, prunedResult's keys and lists, and whole requests.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const picture = @import("core").picture;
const routes = @import("../routes.zig");
const paddlex = @import("paddlex.zig");
const reply = @import("paddlex_reply.zig");
const types = @import("types.zig");
const budget = @import("budget.zig");
const geometry = @import("geometry.zig");
const harness = @import("harness_test.zig");
const t = std.testing;
const Allocator = std.mem.Allocator;
const Value = json.Value;

fn value(a: Allocator, text: []const u8) !Value {
    return switch (try json.parse(a, text)) {
        .ok => |v| v,
        .err => error.TestUnexpectedResult,
    };
}

test "ocr paddlex: base64 skips bytes outside the alphabet and repairs its padding" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectEqualStrings("hello!", (try paddlex.base64(a, "aGVs\nbG8h")).?);
    try t.expectEqualStrings("hello", (try paddlex.base64(a, " aGVs bG8 ")).?);
    try t.expectEqualStrings("hello", (try paddlex.base64(a, "aGVsbG8=*=")).?);
    try t.expectEqualStrings("hi", (try paddlex.base64(a, "aGk")).?);
    try t.expect(try paddlex.base64(a, "aGVsb") == null); // one byte past a group cannot be data
    try t.expect(try paddlex.base64(a, "") == null);
    try t.expect(try paddlex.base64(a, "!!**") == null);
    // the last character's unused bits are ignored, as Python ignores them (PNG's IEND CRC)
    try t.expectEqualSlices(u8, &.{ 0xae, 0x42, 0x60, 0x82 }, (try paddlex.base64(a, "rkJggg==")).?);
    try t.expectEqualSlices(u8, &.{ 0xae, 0x42, 0x60, 0x82 }, (try paddlex.base64(a, "rkJggh==")).?);
    try t.expectEqualStrings("hi", (try paddlex.base64(a, "aGl=")).?);
}

test "ocr paddlex: fields convert as pydantic's lax mode converts them, and numbers must be finite" {
    const Case = struct { []const u8, paddlex.Kind, ?[]const u8 }; // the JSON, the field's kind, the error type (null: accepted)
    const cases = [_]Case{
        .{ "1.0", .integer, null },
        .{ "1.5", .integer, "int_from_float" },
        .{ "\"1\"", .integer, null },
        .{ "\"x\"", .integer, "int_parsing" },
        .{ "true", .integer, null },
        .{ "1e999", .integer, "finite_number" },
        .{ "\"1.0\"", .integer, null },
        .{ "\"-3.00\"", .integer, null },
        .{ "\"1.5\"", .integer, "int_parsing" },
        .{ "\"9223372036854775808\"", .integer, null },
        .{ "\"\\u00a01\\u00a0\"", .integer, null },
        .{ "[]", .integer, "int_type" },
        .{ "0", .boolean, null },
        .{ "1.0", .boolean, null },
        .{ "2", .boolean, "bool_parsing" },
        .{ "2.0", .boolean, "bool_parsing" },
        .{ "1.5", .boolean, "bool_type" },
        .{ "1e999", .boolean, "bool_type" },
        .{ "\"OFF\"", .boolean, null },
        .{ "\"y\"", .boolean, null },
        .{ "\"maybe\"", .boolean, "bool_parsing" },
        .{ "[]", .boolean, "bool_type" },
        .{ "\"0.5\"", .number, null },
        .{ "\"\\u00a00.5\\u2003\"", .number, null },
        .{ "\".5e-1\"", .number, null },
        .{ "\"x\"", .number, "float_parsing" },
        .{ "\"0x1p0\"", .number, "float_parsing" },
        .{ "\"1e\"", .number, "float_parsing" },
        .{ "3", .number, null },
        .{ "true", .number, null },
        .{ "1e999", .number, "finite_number" },
        .{ "\"inf\"", .number, "finite_number" },
        .{ "\"-Infinity\"", .number, "finite_number" },
        .{ "\"NaN\"", .number, "finite_number" },
        .{ "[]", .number, "float_type" },
        .{ "1", .string, "string_type" },
    };
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (cases) |c| switch (paddlex.lax(try value(a, c[0]), c[1])) {
        .ok => if (c[2]) |want| {
            std.debug.print("{s} accepted, want {s}\n", .{ c[0], want });
            return error.TestUnexpectedResult;
        },
        .bad => |b| try t.expectEqualStrings(c[2] orelse return error.TestUnexpectedResult, b[0]),
    };
    try t.expectEqual(@as(f64, 0.5), paddlex.lax(try value(a, "\"0.5\""), .number).ok.float);
    try t.expectEqual(false, paddlex.lax(try value(a, "\"OFF\""), .boolean).ok.bool);
    try t.expectEqual(true, paddlex.lax(try value(a, "1"), .boolean).ok.bool);
    try t.expectEqual(@as(f64, 1), paddlex.lax(try value(a, "true"), .number).ok.float);
    try t.expectEqual(@as(f64, 0.5), paddlex.lax(try value(a, "\"\\u00a00.5\\u2003\""), .number).ok.float);
    try t.expectEqualStrings("1", paddlex.lax(try value(a, "true"), .integer).ok.int);
    // converted values drive the knob checks and the options; a non-finite threshold is refused before any page
    var fault: types.Fault = undefined;
    const ok = try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"useDocUnwarping\": 0, \"useTextlineOrientation\": \"false\", \"visualize\": \"off\", " ++
        "\"textRecScoreThresh\": \"0.5\", \"returnWordBox\": 1}}", .{try pngField(a)});
    try t.expectEqual(paddlex.Options{ .thresh = 0.5, .word_box = true, .visualize = false }, (try paddlex.decode(a, try value(a, ok), facts, &fault)).options);
    const inf = try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"textRecScoreThresh\": 1e999}}", .{try pngField(a)});
    try t.expectError(error.Faulted, paddlex.decode(a, try value(a, inf), facts, &fault));
    try t.expectEqualStrings("finite_number", (try issues(a, fault))[0].strField("type").?);
    const on = try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"useDocUnwarping\": \"yes\"}}", .{try pngField(a)});
    try t.expectError(error.Faulted, paddlex.decode(a, try value(a, on), facts, &fault));
    try t.expectEqualStrings("unsupported", (try issues(a, fault))[0].strField("type").?);
}

test "ocr paddlex: polygon corners round half to even, as Python's round does" {
    try t.expectEqual(@as(f32, 10), geometry.roundEven(10.5));
    try t.expectEqual(@as(f32, 4), geometry.roundEven(4.5));
    try t.expectEqual(@as(f32, 12), geometry.roundEven(11.5));
    try t.expectEqual(@as(f32, 2), geometry.roundEven(2.4999));
    try t.expectEqual(@as(f32, 3), geometry.roundEven(2.5001));
    const q: api.ocr.Quad = .{ .{ 10.5, 4.5 }, .{ 21.5, 4.5 }, .{ 21.5, 14.5 }, .{ 10.5, 14.5 } };
    try t.expectEqual([4][2]i32{ .{ 10, 4 }, .{ 22, 4 }, .{ 22, 14 }, .{ 10, 14 } }, geometry.paddlePoly(q, 100, 100));
    // 10.4 and 10.5 both round to 10: the polygon collapses and is repaired into a 1 px wide box
    const thin: api.ocr.Quad = .{ .{ 10.4, 4.5 }, .{ 10.5, 4.5 }, .{ 10.5, 9.5 }, .{ 10.4, 9.5 } };
    try t.expectEqual([4][2]i32{ .{ 10, 4 }, .{ 11, 4 }, .{ 11, 10 }, .{ 10, 10 } }, geometry.paddlePoly(thin, 100, 100));
}

const facts: paddlex.Facts = .{ .pdf = false, .fetch = null };

fn pngField(a: Allocator) ![]const u8 {
    const png = try harness.png(a, 4, 3);
    const enc = std.base64.standard.Encoder;
    return enc.encode(try a.alloc(u8, enc.calcSize(png.len)), png);
}

/// The fault's errorMsg as the issue list a schema refusal writes.
fn issues(a: Allocator, fault: types.Fault) ![]Value {
    return (try value(a, fault.message)).array;
}

test "ocr paddlex: schema errors list each field with a dot-joined loc; nulls and false switches pass" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fault: types.Fault = undefined;
    try t.expectError(error.Faulted, paddlex.decode(a, try value(a, "{\"fileType\": 2, \"visualize\": \"maybe\", \"textRecScoreThresh\": \"x\"}"), facts, &fault));
    try t.expectEqual(types.Class.bad_input, fault.class);
    const got = try issues(a, fault);
    const want = [_][2][]const u8{ .{ "missing", "body.file" }, .{ "literal_error", "body.fileType" }, .{ "float_parsing", "body.textRecScoreThresh" }, .{ "bool_parsing", "body.visualize" } };
    try t.expectEqual(want.len, got.len);
    for (got, want) |g, w| {
        try t.expectEqualStrings(w[0], g.strField("type").?);
        try t.expectEqualStrings(w[1], g.strField("loc").?);
        try t.expect(g.strField("msg").?.len > 0);
    }
    try t.expectError(error.Faulted, paddlex.decode(a, try value(a, "[1]"), facts, &fault));
    try t.expectEqualStrings("body", (try issues(a, fault))[0].strField("loc").?);
    try t.expectError(error.Faulted, paddlex.decode(a, try value(a, "{\"file\": null}"), facts, &fault));
    try t.expectEqualStrings("string_type", (try issues(a, fault))[0].strField("type").?);
    // content-changing knobs name their fields; the switches pass when false, every knob when null
    const knobs = try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"useDocUnwarping\": true, \"textDetBoxThresh\": 0.5, \"useTextlineOrientation\": false}}", .{try pngField(a)});
    try t.expectError(error.Faulted, paddlex.decode(a, try value(a, knobs), facts, &fault));
    const named = try issues(a, fault);
    try t.expectEqual(@as(usize, 2), named.len);
    try t.expectEqualStrings("body.useDocUnwarping", named[0].strField("loc").?);
    try t.expectEqualStrings("body.textDetBoxThresh", named[1].strField("loc").?);
    const nulls = try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"fileType\": 1, \"logId\": null, \"useDocOrientationClassify\": false, \"useDocUnwarping\": false, " ++
        "\"useTextlineOrientation\": null, \"textDetLimitSideLen\": null, \"textDetLimitType\": null, \"textDetThresh\": null, \"textDetBoxThresh\": null, " ++
        "\"textDetUnclipRatio\": null, \"textRecScoreThresh\": 1, \"returnWordBox\": true, \"visualize\": false, \"unknown\": 3}}", .{try pngField(a)});
    const r = try paddlex.decode(a, try value(a, nulls), facts, &fault);
    try t.expectEqual(paddlex.Options{ .thresh = 1, .word_box = true, .visualize = false }, r.options);
    try t.expect(r.source.kind == .image);
    try t.expectEqual(paddlex.Options{}, (try paddlex.decode(a, try value(a, try std.fmt.allocPrint(a, "{{\"file\": \"{s}\"}}", .{try pngField(a)})), facts, &fault)).options);
}

test "ocr paddlex: the file's kind, URLs and PDFs are checked with PaddleX's words where they fit" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const png = try pngField(a);
    const cases = [_]struct { file: []const u8, extra: []const u8 = "", says: []const u8 }{
        .{ .file = "!!!", .says = "Invalid input file" },
        .{ .file = "R0lGODlhAQABAAAAACw=", .says = "Unsupported file type" },
        .{ .file = png, .extra = ", \"fileType\": 0", .says = "fileType 0 names a PDF" },
        .{ .file = "JVBERi0xLjQK", .extra = ", \"fileType\": 1", .says = "fileType 1 names an image" },
        .{ .file = "JVBERi0xLjQK", .says = "--vision-pdf" },
        .{ .file = "http://example.com/a.png", .says = "public HTTPS URLs" },
        .{ .file = "https://example.com/a.png", .says = "--vision-urls" },
        .{ .file = "data:image/png;base64,!!", .says = "invalid file data URL encoding" },
    };
    for (cases) |c| {
        var fault: types.Fault = undefined;
        const body = try std.fmt.allocPrint(a, "{{\"file\": \"{s}\"{s}}}", .{ c.file, c.extra });
        try t.expectError(error.Faulted, paddlex.decode(a, try value(a, body), facts, &fault));
        try t.expectEqual(types.Class.bad_input, fault.class);
        if (std.mem.indexOf(u8, fault.message, c.says) == null) {
            std.debug.print("{s}: {s}\n", .{ c.file, fault.message });
            return error.TestUnexpectedResult;
        }
    }
    // a PNG with "%PDF-1.7" in an early text chunk is an image: the signature is read first
    var fault: types.Fault = undefined;
    const real = try harness.png(a, 4, 3);
    var tagged: std.ArrayList(u8) = .empty;
    try tagged.appendSlice(a, real[0..33]); // the signature and IHDR
    try harness.chunk(a, &tagged, "tEXt", "Comment\x00%PDF-1.7");
    try tagged.appendSlice(a, real[33..]);
    const enc = std.base64.standard.Encoder;
    const field = enc.encode(try a.alloc(u8, enc.calcSize(tagged.items.len)), tagged.items);
    for ([_][]const u8{ "", ", \"fileType\": 1" }) |extra| {
        const got = try paddlex.decode(a, try value(a, try std.fmt.allocPrint(a, "{{\"file\": \"{s}\"{s}}}", .{ field, extra })), .{ .pdf = true, .fetch = null }, &fault);
        try t.expect(got.source.kind == .image);
    }
    try t.expectError(error.Faulted, paddlex.decode(a, try value(a, try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"fileType\": 0}}", .{field})), .{ .pdf = true, .fetch = null }, &fault));
    try t.expect(std.mem.indexOf(u8, fault.message, "fileType 0 names a PDF") != null);
    // fileType is Literal[0, 1] in lax mode: a bool or a number equal to 0 or 1 counts; a string does not
    for ([_][]const u8{ "1", "1.0", "true" }) |ft| {
        const got = try paddlex.decode(a, try value(a, try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"fileType\": {s}}}", .{ png, ft })), facts, &fault);
        try t.expect(got.source.kind == .image);
    }
    for ([_][]const u8{ "0", "0.0", "false" }) |ft| {
        try t.expectError(error.Faulted, paddlex.decode(a, try value(a, try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"fileType\": {s}}}", .{ png, ft })), facts, &fault));
        try t.expect(std.mem.indexOf(u8, fault.message, "fileType 0 names a PDF") != null);
    }
    for ([_][]const u8{ "\"1\"", "2.0", "0.5" }) |ft| {
        try t.expectError(error.Faulted, paddlex.decode(a, try value(a, try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"fileType\": {s}}}", .{ png, ft })), facts, &fault));
        try t.expectEqualStrings("literal_error", (try issues(a, fault))[0].strField("type").?);
    }
    // a PDF is read to the page budget, never refused for its length
    const r = try paddlex.decode(a, try value(a, "{\"file\": \"JVBERi0xLjQK\", \"fileType\": 0}"), .{ .pdf = true, .fetch = null }, &fault);
    try t.expect(r.source.kind == .pdf);
    try t.expectEqual(@as(u32, std.math.maxInt(u32)), r.source.sel.first);
}

/// One page through the encoder as the page loop hands it over.
fn feed(e: *reply.Encoder, a: Allocator, blocks: []api.ocr.OcrBlock, w: u32, h: u32) !void {
    const page: types.OcrPage = .{ .index = 0, .width = w, .height = h, .dpi = 72, .blocks = blocks };
    const rgb: picture.Rgb = .{ .width = w, .height = h, .pixels = try a.alloc(u8, @as(usize, w) * h * 3) };
    var enc = budget.Encodes.init(@as(u64, w) * h, 1);
    var fault: types.Fault = undefined;
    const s = e.sink();
    try s.page(s.ctx, a, 0, &page, rgb, &enc, &fault);
}

test "ocr paddlex: prunedResult keys in PaddleX's order, rec_* filtered by the threshold, boxes from the polygons" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var blocks = [_]api.ocr.OcrBlock{
        .{ .content = "# Title", .quad = .{ .{ 10.4, 5 }, .{ 90.6, 5 }, .{ 90.6, 15 }, .{ 10.4, 15 } }, .kind = .title },
        .{ .content = "two words\nthird", .quad = .{ .{ 10, 20 }, .{ 90, 20 }, .{ 90, 40 }, .{ 10, 40 } }, .kind = .text },
        .{ .content = "", .quad = .{ .{ 0, 0 }, .{ 5, 0 }, .{ 5, 5 }, .{ 0, 5 } }, .kind = .picture },
    };
    var e: reply.Encoder = .{ .o = .{ .word_box = true, .visualize = false }, .geometry = .block, .gpa = t.allocator };
    try feed(&e, a, &blocks, 100, 50);
    const pruned = e.results.items[0].get("prunedResult").?;
    const order = [_][]const u8{ "model_settings", "dt_polys", "text_type", "textline_orientation_angles", "text_rec_score_thresh", "return_word_box", "rec_texts", "rec_scores", "rec_polys", "rec_boxes", "text_word_boxes", "text_word" };
    try t.expectEqual(order.len, pruned.object.count());
    for (pruned.object.keys(), order) |k, w| try t.expectEqualStrings(w, k);
    try t.expect(e.results.items[0].get("ocrImage") == null and e.results.items[0].get("inputImage") == null);
    const texts = pruned.get("rec_texts").?.array;
    try t.expectEqual(@as(usize, 3), texts.len);
    try t.expectEqualStrings("Title", texts[0].string);
    try t.expectEqualStrings("two words", texts[1].string);
    try t.expectEqual(@as(f64, 1.0), pruned.get("rec_scores").?.array[0].float);
    // the corners round to nearest; rec_boxes is the min/max of the integer polygon
    const out = try json.stringify(a, pruned.get("rec_polys").?.array[0], .{ .compact = true });
    try t.expectEqualStrings("[[10,5],[91,5],[91,15],[10,15]]", out);
    try t.expectEqualStrings("[10,5,91,15]", try json.stringify(a, pruned.get("rec_boxes").?.array[0], .{ .compact = true }));
    try t.expectEqualStrings("[[\"two\",\"words\"]]", try json.stringify(a, .{ .array = pruned.get("text_word").?.array[1..2] }, .{ .compact = true }));
    try t.expectEqual(@as(usize, 2), pruned.get("text_word_boxes").?.array[1].array.len);
    // a threshold above every score empties rec_* but keeps every detected line
    var high: reply.Encoder = .{ .o = .{ .thresh = 1.5, .visualize = false }, .geometry = .block, .gpa = t.allocator };
    try feed(&high, a, &blocks, 100, 50);
    const hp = high.results.items[0].get("prunedResult").?;
    try t.expectEqual(@as(usize, 3), hp.get("dt_polys").?.array.len);
    try t.expectEqual(@as(usize, 0), hp.get("rec_polys").?.array.len);
    try t.expectEqual(@as(usize, 3), hp.get("textline_orientation_angles").?.array.len);
    try t.expect(hp.get("text_word") == null and hp.get("dt_scores") == null);
    // a page with no lines has empty lists and no orientation angles
    var none: reply.Encoder = .{ .o = .{ .visualize = false }, .geometry = .block, .gpa = t.allocator };
    try feed(&none, a, blocks[2..], 100, 50);
    const np = none.results.items[0].get("prunedResult").?;
    try t.expect(np.get("textline_orientation_angles") == null);
    try t.expectEqual(@as(usize, 0), np.get("rec_boxes").?.array.len);
    // two PDF pages: dataInfo counts them and gives each size
    try feed(&none, a, blocks[2..], 30, 20);
    const pages = try value(a, try none.reply(a, "log", true));
    const info = pages.get("result").?.get("dataInfo").?;
    try t.expectEqualStrings("2", info.get("numPages").?.int);
    try t.expectEqualStrings("20", info.get("pages").?.array[1].get("height").?.int);
    try t.expectEqualStrings("pdf", info.strField("type").?);
}

fn serve(h: *harness.Harness, replies: []const harness.Script) !void {
    h.* = .{ .replies = replies };
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false });
}

/// A POST /ocr of ``body`` as PaddleX's clients send it; the reply as JSON, its status in ``status``.
fn post(a: Allocator, h: *harness.Harness, content_type: ?[]const u8, body: []const u8, status: *u16) !Value {
    var wire: harness.Wire = .{ .a = a };
    routes.dispatch(h.srv, try harness.request(a, &wire, "/ocr", content_type, body), a);
    status.* = wire.status();
    try t.expect(std.mem.indexOf(u8, wire.head(), "\r\nContent-Type: application/json\r\n") != null);
    return wire.parsed();
}

test "ocr paddlex: a whole request answers with logId, errorCode 0 and the page's result" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: harness.Harness = undefined;
    try serve(&h, &.{ .{ .text = "title 10 5 90 10\n# Report\n;;\ntext 10 20 90 40\nBody" }, .{ .text = "text 0 0 10 10\nx" } });
    defer h.stop();
    const png = try harness.png(a, 200, 100);
    const enc = std.base64.standard.Encoder;
    const b64 = enc.encode(try a.alloc(u8, enc.calcSize(png.len)), png);
    var status: u16 = 0;
    const body = try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"fileType\": 1, \"visualize\": false, \"logId\": \"abc\"}}", .{b64});
    const r = try post(a, &h, "application/json; charset=utf-8", body, &status);
    try t.expectEqual(@as(u16, 200), status);
    try t.expectEqualStrings("abc", r.strField("logId").?);
    try t.expectEqualStrings("0", r.get("errorCode").?.int);
    try t.expectEqualStrings("Success", r.strField("errorMsg").?);
    const result = r.get("result").?;
    const pruned = result.get("ocrResults").?.array[0].get("prunedResult").?;
    try t.expectEqualStrings("Report", pruned.get("rec_texts").?.array[0].string);
    try t.expectEqualStrings("[20,10,180,20]", try json.stringify(a, pruned.get("rec_boxes").?.array[0], .{ .compact = true }));
    const info = result.get("dataInfo").?;
    try t.expectEqualStrings("200", info.get("width").?.int);
    try t.expectEqualStrings("image", info.strField("type").?);
    // an empty logId gets a fresh UUID v4; no Content-Type with a JSON body is PaddleX too
    const again = try post(a, &h, null, try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"visualize\": false, \"logId\": \"\"}}", .{b64}), &status);
    try t.expectEqual(@as(u16, 200), status);
    const id = again.strField("logId").?;
    try t.expectEqual(@as(usize, 36), id.len);
    try t.expectEqual(@as(u8, '4'), id[14]);
}

test "ocr paddlex: every failure is PaddleX's envelope: 422 for input, unreadable pages and the gate" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: harness.Harness = undefined;
    try serve(&h, &.{.{ .text = "!garbage" }});
    defer h.stop();
    const b64 = try pngField(a);
    var status: u16 = 0;
    const bad_json = try post(a, &h, "application/json", "{\"file\": ", &status);
    try t.expectEqual(@as(u16, 422), status);
    try t.expectEqualStrings("422", bad_json.get("errorCode").?.int);
    try t.expectEqual(@as(usize, 36), bad_json.strField("logId").?.len);
    const listed = try value(a, bad_json.strField("errorMsg").?);
    try t.expectEqualStrings("json_invalid", listed.array[0].strField("type").?);
    const knob = try post(a, &h, "application/json", try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"logId\": \"L\", \"useDocUnwarping\": true}}", .{b64}), &status);
    try t.expectEqual(@as(u16, 422), status);
    try t.expectEqualStrings("L", knob.strField("logId").?);
    try t.expectEqualStrings("body.useDocUnwarping", (try value(a, knob.strField("errorMsg").?)).array[0].strField("loc").?);
    const page = try post(a, &h, "application/json", try std.fmt.allocPrint(a, "{{\"file\": \"{s}\", \"logId\": \"P\"}}", .{b64}), &status);
    try t.expectEqual(@as(u16, 422), status);
    try t.expectEqualStrings("page 0: the model's output does not follow its format", page.strField("errorMsg").?);
    h.srv.info.ocr = null;
    const gated = try post(a, &h, "application/json", "{\"logId\": \"G\"}", &status);
    try t.expectEqual(@as(u16, 422), status);
    try t.expectEqualStrings("G", gated.strField("logId").?);
    try t.expectEqualStrings("this model has no OCR reader; use POST /v1/chat/completions", gated.strField("errorMsg").?);
}

test "ocr paddlex: a capacity fault is 503 with Retry-After, an internal one 500" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]struct { types.Class, u16 }{ .{ .capacity, 503 }, .{ .internal, 500 } }) |c| {
        var wire: harness.Wire = .{ .a = a };
        reply.sendFault(try harness.request(a, &wire, "/ocr", null, ""), a, "id", .{ .class = c[0], .message = "m", .retry_after = 1 });
        try t.expectEqual(c[1], wire.status());
        try t.expectEqual(c[0] == .capacity, std.mem.indexOf(u8, wire.head(), "\r\nRetry-After: 1\r\n") != null);
        try t.expectEqualStrings(try std.fmt.allocPrint(a, "{d}", .{c[1]}), (try wire.parsed()).get("errorCode").?.int);
    }
}

test "ocr paddlex: by default ocrImage and inputImage are base64 JPEGs of the page" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: harness.Harness = undefined;
    try serve(&h, &.{.{ .text = "text 10 5 90 10\nLine" }});
    defer h.stop();
    var status: u16 = 0;
    const r = try post(a, &h, "application/json", try std.fmt.allocPrint(a, "{{\"file\": \"{s}\"}}", .{try pngField(a)}), &status);
    if (status == 500) { // only a host without the pinned libjpeg-turbo skips
        try t.expect(std.mem.indexOf(u8, r.strField("errorMsg").?, "libjpeg-turbo") != null);
        return error.SkipZigTest;
    }
    try t.expectEqual(@as(u16, 200), status);
    const res = r.get("result").?.get("ocrResults").?.array[0];
    try t.expectEqualStrings("prunedResult", res.object.keys()[0]);
    for ([_][]const u8{ "ocrImage", "inputImage" }) |k| {
        const b64 = res.strField(k).?;
        const bytes = try a.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(b64));
        try std.base64.standard.Decoder.decode(bytes, b64);
        try t.expectEqual(picture.Format.jpeg, picture.sniff(bytes));
    }
}
