//! The PDF render worker spawned from the built tensorfold-native, as the server starts it, with or without libpdfium.
const std = @import("std");
const sys = @import("sys");
const errors = @import("errors.zig");
const images = @import("images.zig");
const log = @import("log.zig");
const pdf = @import("pdf.zig");
const worker = @import("pdf_worker.zig");
const Value = @import("json").Value;
const Cx = errors.Cx;

const text_pdf = "%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >> endobj\n4 0 obj << /Length 40 >>\nstream\nBT /F1 24 Tf 20 90 Td (TensorFold) Tj ET\nendstream endobj\n5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000241 00000 n \n0000000331 00000 n \ntrailer << /Size 6 /Root 1 0 R >>\nstartxref\n401\n%%EOF\n";

fn useServer() void {
    log.init(std.testing.io, false);
    worker.server_exe = @import("pdf_worker_exe").path;
}

/// One render of `pdf_bytes` by the render deadline.
fn job(pdf_bytes: []const u8, sel: pdf.Selection, caps: pdf.Caps) pdf.Job {
    return .{ .pdf_bytes = pdf_bytes, .sel = sel, .caps = caps, .deadline_ns = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + pdf.render_timeout_ns };
}

fn expectMissingLibrary(cx: Cx) !void {
    const libs = sys.optional_libraries;
    try std.testing.expectEqual(errors.Kind.unimplemented, cx.kind);
    try std.testing.expect(std.mem.indexOf(u8, cx.message, libs.spec(.pdfium).names[0]) != null or std.mem.indexOf(u8, cx.message, libs.spec(.jpeg).names[0]) != null);
}

test "tensorfold-native PDF render worker: renders a text PDF in its spawned worker, or the worker names the missing library" {
    useServer();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var budget: pdf.Budget = .{};
    var pages: std.ArrayList(pdf.Page) = .empty;
    const total = worker.render(&cx, try worker.argv(a), job(text_pdf, .{ .all_within = 4 }, pdf.chatCaps(4)), &budget, &pages) catch |err| {
        try std.testing.expectEqual(error.Refused, err);
        return expectMissingLibrary(cx);
    };
    try std.testing.expectEqual(@as(u32, 1), total);
    try std.testing.expectEqual(@as(usize, 1), pages.items.len);
    try std.testing.expectEqual(pdf.Budget{ .images_used = 1, .pdf_pages_used = 1, .encoded_bytes = pages.items[0].jpeg.len, .pixels_used = 400 * 400 }, budget);
    try std.testing.expectEqual([2]u32{ 400, 400 }, pdf.jpegSize(pages.items[0].jpeg).?);
}

test "tensorfold-native PDF render worker: libpdfium renders in the spawned worker and enforces request-wide page, byte and pixel caps" {
    useServer();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var parts: std.ArrayList(Value) = .empty;
    var budget: pdf.Budget = .{};
    pdf.expandBytes(&cx, pdf.minimal_two_page_pdf, "image_url", 4, 4, &budget, &parts) catch |err| {
        if (err == error.Refused and cx.kind == .unimplemented) return expectMissingLibrary(cx);
        return err;
    };
    try std.testing.expectEqual(@as(usize, 2), parts.items.len);
    try std.testing.expectEqual(@as(u32, 2), budget.pdf_pages_used);
    try std.testing.expectEqual(@as(u32, 2), budget.images_used);
    try std.testing.expect(budget.encoded_bytes > 0 and budget.pixels_used == 2 * 400 * 400);
    for (parts.items) |p| {
        const img_bytes = try images.dataBytes(&cx, p.get("image_url").?.get("url").?.string);
        try std.testing.expectEqual(images.Format.jpeg, images.sniff(img_bytes));
        try std.testing.expectEqual([2]u32{ 400, 400 }, pdf.jpegSize(img_bytes).?);
    }
    var second_cx: Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, pdf.expandBytes(&second_cx, pdf.minimal_two_page_pdf, "image_url", 4, 3, &budget, &parts));
    try std.testing.expect(std.mem.indexOf(u8, second_cx.message, "at most 3 pages") != null);
}

test "tensorfold-native PDF render worker: a list selection renders only its pages, with their indices, size and DPI" {
    useServer();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var budget: pdf.Budget = .{};
    var pages: std.ArrayList(pdf.Page) = .empty;
    const total = worker.render(&cx, try worker.argv(a), job(pdf.minimal_two_page_pdf, .{ .list = &.{1} }, pdf.ocrCaps(16)), &budget, &pages) catch |err| {
        try std.testing.expectEqual(error.Refused, err);
        return expectMissingLibrary(cx);
    };
    try std.testing.expectEqual(@as(u32, 2), total);
    try std.testing.expectEqual(@as(usize, 1), pages.items.len);
    const p = pages.items[0];
    try std.testing.expectEqual(pdf.Page{ .index = 1, .jpeg = p.jpeg, .width = 400, .height = 400, .dpi = 144 }, p);
    var out_cx: Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, worker.render(&out_cx, try worker.argv(a), job(pdf.minimal_two_page_pdf, .{ .list = &.{ 0, 2 } }, pdf.ocrCaps(16)), &budget, &pages));
    try std.testing.expectEqualStrings("page 2 is out of range; the document has 2 pages", out_cx.message);
    try std.testing.expectEqual(errors.Kind.request, out_cx.kind);
}

test "tensorfold-native PDF render worker: two chat PDFs share one request budget" {
    useServer();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var parts: std.ArrayList(Value) = .empty;
    var budget: pdf.Budget = .{};
    for (0..2) |_| {
        var cx: Cx = .{ .a = a };
        pdf.expandBytes(&cx, pdf.minimal_two_page_pdf, "image_url", 4, 4, &budget, &parts) catch |err| {
            if (err == error.Refused and cx.kind == .unimplemented) return expectMissingLibrary(cx);
            return err;
        };
    }
    try std.testing.expectEqual(@as(usize, 4), parts.items.len);
    try std.testing.expectEqual(@as(u32, 4), budget.pdf_pages_used);
    try std.testing.expectEqual(@as(u64, 4 * 400 * 400), budget.pixels_used);
    var pages_cx: Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, pdf.expandBytes(&pages_cx, pdf.minimal_two_page_pdf, "image_url", 4, 4, &budget, &parts));
    try std.testing.expectEqualStrings("a PDF may have at most 4 pages (--vision-max-images, TENSORFOLD_MAX_PDF_PAGES); this request has 6", pages_cx.message);
    var images_cx: Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, pdf.expandBytes(&images_cx, pdf.minimal_two_page_pdf, "image_url", 4, 8, &budget, &parts));
    try std.testing.expectEqualStrings("a request may hold at most 4 images (--vision-max-images); it has 6", images_cx.message);
    try std.testing.expectEqual(@as(usize, 4), parts.items.len);
}

test "tensorfold-native PDF render worker: a measure returns each page's size and DPI with no JPEG bytes" {
    useServer();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var budget: pdf.Budget = .{};
    var pages: std.ArrayList(pdf.Page) = .empty;
    var measure = job(pdf.minimal_two_page_pdf, .{ .all_within = 16 }, pdf.ocrCaps(16));
    measure.measure = true;
    const total = worker.render(&cx, try worker.argv(a), measure, &budget, &pages) catch |err| {
        try std.testing.expectEqual(error.Refused, err);
        return expectMissingLibrary(cx);
    };
    try std.testing.expectEqual(@as(u32, 2), total);
    try std.testing.expectEqual(@as(usize, 2), pages.items.len);
    for (pages.items, 0..) |p, i| {
        try std.testing.expectEqual(@as(usize, 0), p.jpeg.len);
        try std.testing.expectEqual(pdf.Page{ .index = @intCast(i), .jpeg = p.jpeg, .width = 400, .height = 400, .dpi = 144 }, p);
    }
    try std.testing.expectEqual(pdf.Budget{ .images_used = 2, .pdf_pages_used = 2, .pixels_used = 2 * 400 * 400 }, budget);
}
