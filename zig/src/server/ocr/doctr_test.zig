//! docTR's dialect: options, files, OCROut pages in box and polygon geometry, and whole requests through the route.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const doctr = @import("doctr.zig");
const multipart = @import("../multipart.zig");
const routes = @import("../routes.zig");
const types = @import("types.zig");
const run = @import("run.zig");
const harness = @import("harness_test.zig");
const Harness = harness.Harness;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const t = std.testing;

fn optionsOf(a: Allocator, path: []const u8, fault: *types.Fault) !doctr.Options {
    return doctr.options(a, path, fault);
}

test "ocr doctr: options read pydantic booleans, ignore model choices and refuse content options off their defaults" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fault: types.Fault = undefined;
    try t.expectEqualDeep(doctr.Options{}, try optionsOf(a, "/ocr?det_arch=db_resnet50&reco_arch=crnn_vgg16_bn&det_bs=9&x=1", &fault));
    try t.expectEqualDeep(doctr.Options{ .straight = false, .blocks = true, .lines = false }, try optionsOf(a, "/ocr?assume_straight_pages=OFF&resolve_blocks=Yes&resolve_lines=0", &fault));
    const ok = "/ocr?detect_orientation=true&detect_language=1&detect_tables=false&straighten_pages=no&bin_thresh=0.1&box_thresh=0.10&paragraph_break=0.0035";
    try t.expectEqualDeep(doctr.Options{}, try optionsOf(a, ok, &fault));
    const Case = struct { []const u8, []const u8 };
    for ([_]Case{
        .{ "/ocr?detect_tables=true", "detect_tables=true is not supported" },
        .{ "/ocr?detect_layout=1", "detect_layout=true is not supported" },
        .{ "/ocr?straighten_pages=on", "straighten_pages=true is not supported" },
        .{ "/ocr?paragraph_break=0.01", "paragraph_break=0.01 is not supported" },
        .{ "/ocr?bin_thresh=0.3", "bin_thresh=0.3 is not supported" },
        .{ "/ocr?box_thresh=x", "box_thresh must be a number" },
        .{ "/ocr?resolve_blocks=maybe", "resolve_blocks must be a boolean" },
        .{ "/ocr?detect_language=2", "detect_language must be a boolean" },
        .{ "/ocr?resolve_lines=%zz", "the resolve_lines query parameter is malformed" },
    }) |c| {
        try t.expectError(error.Faulted, optionsOf(a, c[0], &fault));
        try t.expectEqual(types.Class.bad_input, fault.class);
        try t.expect(std.mem.indexOf(u8, fault.message, c[1]) != null);
    }
}

fn part(name: []const u8, filename: ?[]const u8, content_type: ?[]const u8) multipart.Part {
    return .{ .name = name, .filename = filename, .content_type = content_type, .media = null, .body = "bytes" };
}

test "ocr doctr: files keep part order and their names; other part types are refused as docTR words them" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fault: types.Fault = undefined;
    const parts = [_]multipart.Part{ part("files", "b.png", "image/png"), part("other", "x", "text/plain"), part("files", "a.pdf", "application/pdf"), part("files", null, "image/jpeg") };
    const got = try doctr.files(a, &parts, true, &fault);
    try t.expectEqual(@as(usize, 3), got.len);
    try t.expectEqualStrings("b.png", got[0].name);
    try t.expect(got[0].kind == .image and got[0].sel.all_within == 1);
    try t.expectEqualStrings("a.pdf", got[1].name);
    try t.expect(got[1].kind == .pdf and got[1].sel.all_within == std.math.maxInt(u32));
    try t.expectEqualStrings("", got[2].name);
    const Case = struct { []const multipart.Part, []const u8 };
    for ([_]Case{
        .{ &.{part("files", "x.txt", "text/plain")}, "Unsupported file format: text/plain for file x.txt" },
        .{ &.{part("files", "y.png", null)}, "Unsupported file format: None for file y.png" },
        .{ &.{part("files", "z.png", "image/png; charset=binary")}, "Unsupported file format: image/png; charset=binary for file z.png" },
        .{ &.{part("files", "w.webp", "image/webp")}, "Unsupported file format: image/webp for file w.webp" },
        .{ &.{part("file", "v.png", "image/png")}, "a docTR request needs one or more files parts" },
    }) |c| {
        try t.expectError(error.Faulted, doctr.files(a, c[0], true, &fault));
        try t.expectEqualStrings(c[1], fault.message);
    }
    try t.expectError(error.Faulted, doctr.files(a, &.{part("files", "a.pdf", "application/pdf")}, false, &fault));
    try t.expect(std.mem.indexOf(u8, fault.message, "--vision-pdf") != null);
}

fn page(a: Allocator, w: u32, h: u32, blocks: []const api.ocr.OcrBlock) !types.OcrPage {
    return .{ .index = 0, .width = w, .height = h, .dpi = 72, .blocks = try a.dupe(api.ocr.OcrBlock, blocks) };
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) api.ocr.Quad {
    return .{ .{ x0, y0 }, .{ x1, y0 }, .{ x1, y1 }, .{ x0, y1 } };
}

fn encode(a: Allocator, o: doctr.Options, geo: api.ocr.Geometry, p: *const types.OcrPage) !Value {
    const e: doctr.Encoder = .{ .o = o, .geometry = geo, .sources = &.{} };
    return e.encodePage(a, "page.png", p);
}

fn expectFloats(want: []const f64, got: Value) !void {
    try t.expectEqual(want.len, got.array.len);
    for (want, got.array) |w, g| try t.expectEqual(w, g.float);
}

fn blocksOf(v: Value) []Value {
    return v.get("items").?.array[0].get("blocks").?.array;
}

const two_blocks = [_]api.ocr.OcrBlock{
    .{ .content = "Hello world", .quad = rect(10, 10, 60, 20), .kind = .text },
    .{ .content = "", .quad = rect(0, 0, 5, 5), .kind = .picture },
    .{ .content = "one\ntwo", .quad = rect(10, 30, 90, 40), .kind = .text },
};

test "ocr doctr: one block a page by default, source blocks with resolve_blocks, one line with resolve_lines=false" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try page(a, 100, 50, &two_blocks);
    const v = try encode(a, .{}, .block, &p);
    try t.expectEqualStrings("page.png", v.strField("name").?);
    try t.expect(v.get("orientation").?.get("value").? == .null and v.get("language").?.get("confidence").? == .null);
    try t.expectEqualStrings("50", v.get("dimensions").?.array[0].int);
    try t.expectEqualStrings("100", v.get("dimensions").?.array[1].int);
    try t.expect(v.get("layout").?.array.len == 0 and v.get("tables").?.array.len == 0);
    const one = blocksOf(v);
    try t.expectEqual(@as(usize, 1), one.len);
    try expectFloats(&.{ 0.1, 0.2, 0.9, 0.8 }, one[0].get("geometry").?);
    try t.expectEqual(@as(f64, 1.0), one[0].get("objectness_score").?.float);
    const ls = one[0].get("lines").?.array;
    try t.expectEqual(@as(usize, 3), ls.len);
    try expectFloats(&.{ 0.1, 0.2, 0.6, 0.4 }, ls[0].get("geometry").?);
    try expectFloats(&.{ 0.1, 0.7, 0.9, 0.8 }, ls[2].get("geometry").?);
    const word = ls[0].get("words").?.array[0];
    try t.expectEqualStrings("Hello", word.strField("value").?);
    try expectFloats(&.{ 0.1, 0.2, 0.35, 0.4 }, word.get("geometry").?);
    try t.expectEqual(@as(f64, 1.0), word.get("confidence").?.float);
    try t.expectEqual(@as(f64, 1.0), word.get("objectness_score").?.float);
    try t.expectEqualStrings("0", word.get("crop_orientation").?.get("value").?.int);
    try t.expect(word.get("crop_orientation").?.get("confidence").? == .null);
    const grouped = blocksOf(try encode(a, .{ .blocks = true }, .block, &p));
    try t.expectEqual(@as(usize, 2), grouped.len);
    try t.expectEqual(@as(usize, 1), grouped[0].get("lines").?.array.len);
    try t.expectEqual(@as(usize, 2), grouped[1].get("lines").?.array.len);
    try expectFloats(&.{ 0.1, 0.6, 0.9, 0.8 }, grouped[1].get("geometry").?);
    for ([_]doctr.Options{ .{ .lines = false }, .{ .lines = false, .blocks = true } }) |o| {
        const merged = blocksOf(try encode(a, o, .block, &p));
        try t.expectEqual(@as(usize, 1), merged.len);
        const line = merged[0].get("lines").?.array;
        try t.expectEqual(@as(usize, 1), line.len);
        const ws = line[0].get("words").?.array;
        try t.expectEqual(@as(usize, 4), ws.len);
        for (ws, [_][]const u8{ "Hello", "world", "one", "two" }) |w, s| try t.expectEqualStrings(s, w.strField("value").?);
        try expectFloats(&.{ 0.1, 0.2, 0.9, 0.8 }, line[0].get("geometry").?);
    }
}

test "ocr doctr: polygon geometry is 8 relative floats in reading order, never rounded" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // a 180-degree line: TL is at the right, so the first word starts there
    const turned: api.ocr.Quad = .{ .{ 60, 20 }, .{ 10, 20 }, .{ 10, 10 }, .{ 60, 10 } };
    const p = try page(a, 100, 50, &.{.{ .content = "ab cd", .quad = turned, .kind = .text }});
    const block = blocksOf(try encode(a, .{ .straight = false }, .block, &p))[0];
    try expectFloats(&.{ 0.1, 0.2, 0.6, 0.2, 0.6, 0.4, 0.1, 0.4 }, block.get("geometry").?);
    const line = block.get("lines").?.array[0];
    try expectFloats(&.{ 0.6, 0.4, 0.1, 0.4, 0.1, 0.2, 0.6, 0.2 }, line.get("geometry").?);
    try expectFloats(&.{ 0.6, 0.4, 0.35, 0.4, 0.35, 0.2, 0.6, 0.2 }, line.get("words").?.array[0].get("geometry").?);
    // a merged line has no quad of its own: its union's corners
    const merged = blocksOf(try encode(a, .{ .straight = false, .lines = false }, .block, &p))[0].get("lines").?.array[0];
    try expectFloats(&.{ 0.1, 0.2, 0.6, 0.2, 0.6, 0.4, 0.1, 0.4 }, merged.get("geometry").?);
    const third = try page(a, 30, 10, &.{.{ .content = "x", .quad = rect(10, 0, 20, 10), .kind = .text }});
    const g = blocksOf(try encode(a, .{}, .block, &third))[0].get("geometry").?;
    try expectFloats(&.{ 10.0 / 30.0, 0, 20.0 / 30.0, 1 }, g);
    const text = try json.stringify(a, g, .{});
    try t.expect(std.mem.indexOf(u8, text, "0.3333333333333333") != null);
}

test "ocr doctr: an empty page has no blocks, and a reader without geometry gives whole-page boxes" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const empty = try page(a, 100, 50, &.{.{ .content = "", .quad = rect(0, 0, 5, 5), .kind = .picture }});
    for ([_]doctr.Options{ .{}, .{ .lines = false }, .{ .blocks = true } }) |o| try t.expectEqual(@as(usize, 0), blocksOf(try encode(a, o, .block, &empty)).len);
    const whole = try page(a, 100, 50, &.{.{ .content = "# Title\n\nbody text", .quad = null, .kind = .page }});
    const b = blocksOf(try encode(a, .{}, .none, &whole))[0];
    try expectFloats(&.{ 0, 0, 1, 1 }, b.get("geometry").?);
    const ls = b.get("lines").?.array;
    try t.expectEqual(@as(usize, 2), ls.len);
    try expectFloats(&.{ 0, 0, 1, 1 }, ls[1].get("words").?.array[1].get("geometry").?);
    const poly = blocksOf(try encode(a, .{ .straight = false }, .none, &whole))[0];
    try expectFloats(&.{ 0, 0, 1, 0, 1, 1, 0, 1 }, poly.get("lines").?.array[0].get("geometry").?);
}

fn noInfinity(a: Allocator, v: Value) !void {
    const text = try json.stringify(a, v, .{});
    try t.expect(std.mem.indexOf(u8, text, "inf") == null and std.mem.indexOf(u8, text, "Inf") == null);
}

test "ocr doctr: lines of only Unicode whitespace are left out, and so are blocks left without lines" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const blank = try page(a, 100, 50, &.{
        .{ .content = "\u{00A0}", .quad = rect(10, 10, 60, 20), .kind = .text },
        .{ .content = "\u{3000}", .quad = rect(10, 30, 60, 40), .kind = .text },
    });
    const all = [_]doctr.Options{ .{}, .{ .straight = false }, .{ .lines = false }, .{ .straight = false, .lines = false }, .{ .blocks = true } };
    for (all) |o| {
        const v = try encode(a, o, .block, &blank);
        try t.expectEqual(@as(usize, 0), blocksOf(v).len);
        try noInfinity(a, v);
    }
    const mixed = try page(a, 100, 50, &.{
        .{ .content = "Hello", .quad = rect(10, 10, 60, 20), .kind = .text },
        .{ .content = "\u{3000}", .quad = rect(10, 20, 60, 30), .kind = .text },
        .{ .content = "world", .quad = rect(10, 30, 60, 40), .kind = .text },
    });
    for ([_]bool{ true, false }) |straight| {
        const one = try encode(a, .{ .straight = straight }, .block, &mixed);
        try noInfinity(a, one);
        const ls = blocksOf(one)[0].get("lines").?.array;
        try t.expectEqual(@as(usize, 2), ls.len);
        try t.expectEqualStrings("world", ls[1].get("words").?.array[0].strField("value").?);
        if (straight) try expectFloats(&.{ 0.1, 0.2, 0.6, 0.8 }, blocksOf(one)[0].get("geometry").?);
        const grouped = try encode(a, .{ .straight = straight, .blocks = true }, .block, &mixed);
        try noInfinity(a, grouped);
        try t.expectEqual(@as(usize, 2), blocksOf(grouped).len);
        const merged = try encode(a, .{ .straight = straight, .lines = false }, .block, &mixed);
        try noInfinity(a, merged);
        try t.expectEqual(@as(usize, 2), blocksOf(merged)[0].get("lines").?.array[0].get("words").?.array.len);
    }
}

test "ocr doctr: a line polygon encloses its words, or the words' union is written instead" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // two words under 1 px² each are widened by validation past the half-pixel-wide line quad
    const thin = try page(a, 100, 50, &.{.{ .content = "a b", .quad = rect(10, 10, 10.5, 12), .kind = .text }});
    const line = blocksOf(try encode(a, .{ .straight = false }, .block, &thin))[0].get("lines").?.array[0];
    const g = line.get("geometry").?.array;
    var lo: [2]f64 = .{ std.math.inf(f64), std.math.inf(f64) };
    var hi: [2]f64 = .{ -std.math.inf(f64), -std.math.inf(f64) };
    for (line.get("words").?.array) |w| for (w.get("geometry").?.array, 0..) |x, i| {
        lo[i % 2] = @min(lo[i % 2], x.float);
        hi[i % 2] = @max(hi[i % 2], x.float);
    };
    try t.expect(hi[0] > 10.5 / 100.0);
    const want = [_]f64{ lo[0], lo[1], hi[0], lo[1], hi[0], hi[1], lo[0], hi[1] };
    for (want, g) |w, x| try t.expectEqual(w, x.float);
    // a line whose words lie inside it keeps its own quad
    const wide = try page(a, 100, 50, &.{.{ .content = "a b", .quad = rect(10, 10, 60, 20), .kind = .text }});
    const kept = blocksOf(try encode(a, .{ .straight = false }, .block, &wide))[0].get("lines").?.array[0];
    try expectFloats(&.{ 0.1, 0.2, 0.6, 0.2, 0.6, 0.4, 0.1, 0.4 }, kept.get("geometry").?);
}

const File = struct { filename: ?[]const u8 = "page.png", ct: ?[]const u8 = "image/png", bytes: []const u8 };

fn form(a: Allocator, files: []const File) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (files) |f| {
        try out.appendSlice(a, "--XyZ\r\nContent-Disposition: form-data; name=\"files\"");
        if (f.filename) |n| try out.print(a, "; filename=\"{s}\"", .{n});
        try out.appendSlice(a, "\r\n");
        if (f.ct) |ct| try out.print(a, "Content-Type: {s}\r\n", .{ct});
        try out.print(a, "\r\n{s}\r\n", .{f.bytes});
    }
    try out.appendSlice(a, "--XyZ--\r\n");
    return out.items;
}

/// A whole POST through the dispatcher; the reply as JSON, its status in ``status``.
fn post(a: Allocator, h: *Harness, path: []const u8, body: []const u8, status: *u16) !Value {
    var wire: harness.Wire = .{ .a = a };
    routes.dispatch(h.srv, try harness.request(a, &wire, path, "multipart/form-data; boundary=XyZ", body), a);
    status.* = wire.status();
    try t.expect(std.mem.indexOf(u8, wire.head(), "\r\nContent-Type: application/json\r\n") != null);
    return wire.parsed();
}

fn serve(h: *Harness, replies: []const harness.Script) !void {
    h.* = .{ .replies = replies };
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false });
}

test "ocr doctr: the README call on /ocr and /ocr/ gives one OCROut a file, in order, with docTR's field types" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = undefined;
    const first = "text 10 10 60 20\nHello world";
    const second = "title 0 0 100 50\n# Second";
    try serve(&h, &.{ .{ .text = first }, .{ .text = second }, .{ .text = first }, .{ .text = second } });
    defer h.stop();
    const png = try harness.png(a, 100, 50);
    const body = try form(a, &.{ .{ .filename = "a.png", .bytes = png }, .{ .filename = "b.png", .bytes = png } });
    for ([_][]const u8{ "/ocr?det_arch=db_resnet50&reco_arch=crnn_vgg16_bn", "/ocr/?det_arch=db_resnet50&reco_arch=crnn_vgg16_bn" }) |path| {
        var status: u16 = 0;
        const r = try post(a, &h, path, body, &status);
        try t.expectEqual(@as(u16, 200), status);
        try t.expectEqual(@as(usize, 2), r.array.len);
        try t.expectEqualStrings("a.png", r.array[0].strField("name").?);
        try t.expectEqualStrings("b.png", r.array[1].strField("name").?);
        try t.expectEqualStrings("50", r.array[0].get("dimensions").?.array[0].int);
        const word = blocksOf(r.array[0])[0].get("lines").?.array[0].get("words").?.array[1];
        try t.expectEqualStrings("world", word.strField("value").?);
        try expectFloats(&.{ 0.35, 0.2, 0.6, 0.4 }, word.get("geometry").?);
        try t.expectEqualStrings("0", word.get("crop_orientation").?.get("value").?.int);
        try t.expectEqualStrings("Second", blocksOf(r.array[1])[0].get("lines").?.array[0].get("words").?.array[0].strField("value").?);
    }
}

test "ocr doctr: ?format=doctr reads the parts itself, and refusals and failures answer {detail} with their status" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = undefined;
    try serve(&h, &.{ .{ .text = "text 0 0 10 10\nok" }, .{ .text = "!not the grammar" }, .{ .text = "text 0 0 1 1\nhalf", .reason = .length } });
    defer h.stop();
    const png = try harness.png(a, 100, 50);
    const one = try form(a, &.{.{ .bytes = png }});
    var status: u16 = 0;
    try t.expectEqual(@as(usize, 1), (try post(a, &h, "/ocr?format=doctr", one, &status)).array.len);
    try t.expectEqual(@as(u16, 200), status);
    var many: [17]File = undefined;
    const small = try harness.png(a, 8, 8);
    for (&many) |*f| f.* = .{ .bytes = small };
    const Case = struct { []const u8, []const u8, u16, []const u8 };
    for ([_]Case{
        .{ "/ocr", one, 422, "page 0: the model's output does not follow its format" },
        .{ "/ocr", one, 422, "page 0: the model's output was cut at the context limit" },
        .{ "/ocr", try form(a, &.{.{ .filename = "n.txt", .ct = "text/plain", .bytes = "x" }}), 400, "Unsupported file format: text/plain for file n.txt" },
        .{ "/ocr", try form(a, &.{.{ .ct = null, .bytes = png }}), 400, "Unsupported file format: None for file page.png" },
        .{ "/ocr?detect_tables=true", one, 400, "detect_tables=true is not supported; TensorFold reads pages with detect_tables=false" },
        .{ "/ocr", try form(a, &many), 400, "a request may read at most 16 pages (TENSORFOLD_MAX_PDF_PAGES); this one selects more" },
        .{ "/ocr?format=doctr", "{}", 400, "a docTR request is multipart/form-data with one or more files parts" },
    }) |c| {
        var wire: harness.Wire = .{ .a = a };
        const ct: []const u8 = if (c[1][0] == '{') "application/json" else "multipart/form-data; boundary=XyZ";
        routes.dispatch(h.srv, try harness.request(a, &wire, c[0], ct, c[1]), a);
        try t.expectEqual(c[2], wire.status());
        try t.expectEqualStrings(c[3], (try wire.parsed()).strField("detail").?);
    }
    h.srv.info.ocr = null;
    _ = try post(a, &h, "/ocr", one, &status);
    try t.expectEqual(@as(u16, 400), status);
    var wire: harness.Wire = .{ .a = a };
    routes.dispatch(h.srv, try harness.request(a, &wire, "/ocr", "multipart/form-data; boundary=XyZ", one), a);
    try t.expectEqualStrings(run.no_reader, (try wire.parsed()).strField("detail").?);
}

test "ocr doctr: a capacity fault sends 503 with Retry-After, other classes their status, all as {detail}" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { types.Class, u16 };
    for ([_]Case{ .{ .bad_input, 400 }, .{ .unreadable, 422 }, .{ .capacity, 503 }, .{ .internal, 500 } }) |c| {
        var wire: harness.Wire = .{ .a = a };
        const conn = try harness.request(a, &wire, "/ocr", null, "");
        doctr.sendFault(conn, a, .{ .class = c[0], .message = "why", .retry_after = if (c[0] == .capacity) 1 else null });
        try t.expectEqual(c[1], wire.status());
        try t.expectEqual(c[0] == .capacity, std.mem.indexOf(u8, wire.head(), "\r\nRetry-After: 1\r\n") != null);
        try t.expectEqualStrings("why", (try wire.parsed()).strField("detail").?);
    }
}
