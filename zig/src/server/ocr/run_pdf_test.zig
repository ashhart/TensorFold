//! One rendered PDF page at a time: a scripted render worker answers each call in turn, the sink counts them per page.
const std = @import("std");
const builtin = @import("builtin");
const picture = @import("core").picture;
const jpeg_out = @import("../jpeg_out.zig");
const log = @import("../log.zig");
const pdf = @import("../pdf.zig");
const worker = @import("../pdf_worker.zig");
const run = @import("run.zig");
const types = @import("types.zig");
const budget = @import("budget.zig");
const harness = @import("harness_test.zig");
const t = std.testing;
const Allocator = std.mem.Allocator;

const Gone = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

/// The script the server starts as its render worker: it reads the request and answers the n-th call with reply<n>.bin.
const script =
    \\#!/bin/sh
    \\cd "$(dirname "$0")" || exit 1
    \\cat >/dev/null
    \\n=$(cat calls 2>/dev/null || echo 0)
    \\echo $((n + 1)) >calls
    \\exec cat "reply$n.bin"
;

/// A worker reply frame for a two-page document: each page at 400 x 400 and 144 DPI with `jpeg` (empty: measured).
fn frame(a: Allocator, indices: []const u32, jpeg: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.append(a, 0);
    for ([_]u32{ @intCast(indices.len), 2 }) |v| try out.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
    for (indices) |i| {
        for ([_]u32{ i, 400, 400, 144, @intCast(jpeg.len) }) |v| try out.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
        try out.appendSlice(a, jpeg);
    }
    return out.items;
}

/// Records, as each page reaches the sink, how many times the render worker had been started.
const Order = struct {
    dir: std.Io.Dir,
    calls: std.ArrayList(u32) = .empty,
    indices: std.ArrayList(u32) = .empty,
    widths: std.ArrayList(u32) = .empty,
    fn sink(o: *Order) run.PageSink {
        return .{ .ctx = o, .page = page };
    }
    fn page(ctx: *anyopaque, a: Allocator, p: *const types.OcrPage, rgb: picture.Rgb, _: *budget.Encodes, _: *types.Fault) run.Step!void {
        const o: *Order = @ptrCast(@alignCast(ctx));
        const calls: u32 = blk: { // 0 when the count cannot be read, which the test then reports
            const text = o.dir.readFileAlloc(t.io, "calls", a, .limited(64)) catch break :blk 0;
            break :blk std.fmt.parseInt(u32, std.mem.trim(u8, text, " \n"), 10) catch 0;
        };
        try o.calls.append(a, calls);
        try o.indices.append(a, p.index);
        try o.widths.append(a, @intCast(rgb.width));
    }
};

test "ocr run: PDF pages are measured first, and page k renders only after page k-1's sink returned" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest; // the worker is POSIX only; Windows renders in process
    log.init(t.io, false);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // the parent opens PDFium before it starts a worker; the pages decode through the pinned libjpeg-turbo
    if (jpeg_out.loadDynLib(pdf.PdfiumFns, .pdfium, a) == null) return error.SkipZigTest;
    const rgb = try a.alloc(u8, 400 * 400 * 3);
    @memset(rgb, 0xC0);
    const jpeg = jpeg_out.encodeRgb(a, rgb, 400, 400, 85) catch |e| return if (e == error.NoLibrary) error.SkipZigTest else e;
    _ = picture.decode(t.io, a, jpeg) catch |e| return switch (e) {
        error.NoJpegLibrary, error.JpegLibraryVersion => error.SkipZigTest,
        else => e,
    };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    // call 0 measures both pages (no JPEG bytes); calls 1 and 2 render page 0 and page 1, one page each
    try tmp.dir.writeFile(t.io, .{ .sub_path = "reply0.bin", .data = try frame(a, &.{ 0, 1 }, "") });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "reply1.bin", .data = try frame(a, &.{0}, jpeg) });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "reply2.bin", .data = try frame(a, &.{1}, jpeg) });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "worker", .data = script, .flags = .{ .permissions = .fromMode(0o755) } });
    const saved = worker.server_exe;
    defer worker.server_exe = saved;
    worker.server_exe = try std.fs.path.join(a, &.{ try tmp.dir.realPathFileAlloc(t.io, ".", a), "worker" });
    var h: harness.Harness = .{ .replies = &.{ .{ .text = "text 0 0 10 10\na" }, .{ .text = "text 0 0 10 10\nb" } } };
    try h.start(.{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .vision_pdf = true });
    defer h.stop();
    var order: Order = .{ .dir = tmp.dir };
    var fault: types.Fault = undefined;
    const doc: types.Source = .{ .bytes = pdf.minimal_two_page_pdf, .kind = .pdf, .sel = .{ .all_within = 16 } };
    run.run(h.srv, a, &.{doc}, order.sink(), &fault, Gone{}) catch |e| {
        if (e == error.Faulted) std.debug.print("ocr run: {s}\n", .{fault.message});
        return e;
    };
    try t.expectEqualSlices(u32, &.{ 0, 1 }, order.indices.items);
    try t.expectEqualSlices(u32, &.{ 400, 400 }, order.widths.items);
    // page 0's sink saw the measure and its own render only; page 1 rendered after it returned
    try t.expectEqualSlices(u32, &.{ 2, 3 }, order.calls.items);
    try t.expectEqual(@as(usize, 2), h.next);
}
