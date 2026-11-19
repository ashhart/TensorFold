//! Measured PDF pages (Job.measure): each selected page's size and DPI without a bitmap or a JPEG, as OCR plans them.
const std = @import("std");
const builtin = @import("builtin");
const errors = @import("errors.zig");
const jpeg_out = @import("jpeg_out.zig");
const log = @import("log.zig");
const pdf = @import("pdf.zig");
const worker = @import("pdf_worker.zig");
const Cx = errors.Cx;
const t = std.testing;

/// `sel` of the two-page fixture in this process, measured or rendered; skips without PDFium or libjpeg-turbo.
fn fixture(cx: *Cx, sel: pdf.Selection, caps: pdf.Caps, budget: *pdf.Budget, measure: bool) !pdf.Rendered {
    log.init(t.io, false);
    const jf = jpeg_out.get(cx.a) orelse return error.SkipZigTest;
    const f = jpeg_out.loadDynLib(pdf.PdfiumFns, .pdfium, cx.a) orelse return error.SkipZigTest;
    f.FPDF_InitLibrary(); // a test runner cannot be the worker (pdf_worker_test.zig measures through it)
    var pages: std.ArrayList(pdf.Page) = .empty;
    const job: pdf.Job = .{ .pdf_bytes = pdf.minimal_two_page_pdf, .sel = sel, .caps = caps, .deadline_ns = std.Io.Clock.awake.now(log.io()).toNanoseconds() + pdf.render_timeout_ns, .measure = measure };
    const total = try pdf.renderInProcess(cx, f, jf, job, budget, &pages);
    return .{ .pages = pages.items, .total = total };
}

test "pdf measure: both pages' sizes and DPI equal a real render's, with no JPEG bytes and no bytes charged" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    var measured_budget: pdf.Budget = .{};
    var rendered_budget: pdf.Budget = .{};
    const measured = try fixture(&cx, .{ .all_within = 16 }, pdf.ocrCaps(16), &measured_budget, true);
    const rendered = try fixture(&cx, .{ .all_within = 16 }, pdf.ocrCaps(16), &rendered_budget, false);
    try t.expectEqual(@as(u32, 2), measured.total);
    try t.expectEqual(@as(usize, 2), measured.pages.len);
    for (measured.pages, rendered.pages) |m, r| {
        try t.expectEqual(@as(usize, 0), m.jpeg.len);
        try t.expectEqual(pdf.Page{ .index = r.index, .jpeg = m.jpeg, .width = r.width, .height = r.height, .dpi = r.dpi }, m);
        try t.expectEqual([2]u32{ m.width, m.height }, pdf.jpegSize(r.jpeg).?);
    }
    try t.expectEqual(pdf.Budget{ .images_used = 2, .pdf_pages_used = 2, .encoded_bytes = 0, .pixels_used = rendered_budget.pixels_used }, measured_budget);
}

/// Expects `sel` of the fixture refused, measured or rendered; a missing library still skips the test.
fn expectRefused(cx: *Cx, sel: pdf.Selection, budget: *pdf.Budget, measure: bool) !void {
    if (fixture(cx, sel, pdf.ocrCaps(16), budget, measure)) |_| return error.TestUnexpectedResult else |e| switch (e) {
        error.Refused => {},
        else => return e,
    }
}

test "pdf measure: a selection over the page budget is refused whole, as a render refuses it" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ true, false }) |measure| {
        var cx: Cx = .{ .a = a };
        var budget: pdf.Budget = .{};
        try expectRefused(&cx, .{ .all_within = 1 }, &budget, measure);
        try t.expectEqualStrings("a PDF may have at most 1 pages (--vision-max-images, TENSORFOLD_MAX_PDF_PAGES); this request has 2", cx.message);
        try t.expectEqual(pdf.Budget{}, budget);
        var out_cx: Cx = .{ .a = a };
        try expectRefused(&out_cx, .{ .list = &.{ 0, 2 } }, &budget, measure);
        try t.expectEqualStrings("page 2 is out of range; the document has 2 pages", out_cx.message);
    }
}

test "pdf measure: a worker frame without JPEG bytes passes only for a measure, and one with them only for a render" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest; // the worker is POSIX only; Windows renders in process
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // first(1) of a two-page document: page 0 at 400 x 400 and 144 DPI, its JPEG length 0
    const sized = [_]u8{ 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0x90, 1, 0, 0, 0x90, 1, 0, 0, 144, 0, 0, 0, 0, 0, 0, 0 };
    var budget: pdf.Budget = .{};
    var pages: std.ArrayList(pdf.Page) = .empty;
    var cx: Cx = .{ .a = a };
    try t.expectEqual(@as(u32, 2), try worker.parseChildReply(&cx, &sized, .{ .first = 1 }, true, pdf.ocrCaps(4), &budget, &pages));
    try t.expectEqual(pdf.Page{ .index = 0, .jpeg = pages.items[0].jpeg, .width = 400, .height = 400, .dpi = 144 }, pages.items[0]);
    try t.expectEqual(@as(usize, 0), pages.items[0].jpeg.len);
    try t.expectEqual(pdf.Budget{ .images_used = 1, .pdf_pages_used = 1, .pixels_used = 400 * 400 }, budget);
    // the same frame for a render is a renderer failure of the caller's kind, and leaves the budget alone
    for ([_]pdf.Caps{ pdf.ocrCaps(4), pdf.chatCaps(4) }) |caps| {
        var render_cx: Cx = .{ .a = a };
        var fresh: pdf.Budget = .{};
        try t.expectError(error.Refused, worker.parseChildReply(&render_cx, &sized, .{ .first = 1 }, false, caps, &fresh, &pages));
        try t.expectEqual(caps.infra, render_cx.kind);
        try t.expectEqual(pdf.Budget{}, fresh);
    }
    // a page with bytes in a measure's frame is refused too
    const with_bytes = [_]u8{ 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0x90, 1, 0, 0, 0x90, 1, 0, 0, 144, 0, 0, 0, 1, 0, 0, 0, 0xFF };
    var bytes_cx: Cx = .{ .a = a };
    var fresh: pdf.Budget = .{};
    try t.expectError(error.Refused, worker.parseChildReply(&bytes_cx, &with_bytes, .{ .first = 1 }, true, pdf.ocrCaps(4), &fresh, &pages));
    try t.expectEqual(errors.Kind.server, bytes_cx.kind);
}
