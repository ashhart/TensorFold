//! PDF document input (--vision): renders PDF pages via PDFium and libjpeg-turbo into image_url parts.
const std = @import("std");
const sys = @import("sys");
const json = @import("json");
const errors = @import("errors.zig");
const images = @import("images.zig");
const log = @import("log.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const env = "TENSORFOLD_LIBPDFIUM";
pub const max_pdf_bytes: usize = 32 << 20;
pub const default_max_pages: u32 = 16;

const PdfiumFns = struct {
    FPDF_InitLibrary: *const fn () callconv(.c) void,
    FPDF_LoadMemDocument: *const fn ([*]const u8, c_int, ?[*:0]const u8) callconv(.c) ?*anyopaque,
    FPDF_CloseDocument: *const fn (?*anyopaque) callconv(.c) void,
    FPDF_GetPageCount: *const fn (?*anyopaque) callconv(.c) c_int,
    FPDF_LoadPage: *const fn (?*anyopaque, c_int) callconv(.c) ?*anyopaque,
    FPDF_ClosePage: *const fn (?*anyopaque) callconv(.c) void,
    FPDF_GetPageWidthF: *const fn (?*anyopaque) callconv(.c) f32,
    FPDF_GetPageHeightF: *const fn (?*anyopaque) callconv(.c) f32,
    FPDFBitmap_Create: *const fn (c_int, c_int, c_int) callconv(.c) ?*anyopaque,
    FPDFBitmap_Destroy: *const fn (?*anyopaque) callconv(.c) void,
    FPDFBitmap_GetBuffer: *const fn (?*anyopaque) callconv(.c) ?*anyopaque,
    FPDFBitmap_GetStride: *const fn (?*anyopaque) callconv(.c) c_int,
    FPDF_RenderPageBitmap: *const fn (?*anyopaque, ?*anyopaque, c_int, c_int, c_int, c_int, c_int, c_int) callconv(.c) void,
};

const JpegFns = struct {
    tj3Init: *const fn (c_int) callconv(.c) ?*anyopaque,
    tj3Destroy: *const fn (?*anyopaque) callconv(.c) void,
    tj3Set: *const fn (?*anyopaque, c_int, c_int) callconv(.c) c_int,
    tj3Compress8: *const fn (?*anyopaque, [*]const u8, c_int, c_int, c_int, c_int, *?[*]u8, *usize) callconv(.c) c_int,
    tj3Free: *const fn (?*anyopaque) callconv(.c) void,
};

const PdfiumLoaded = union(enum) { unknown, missing, ready: PdfiumFns };
const JpegLoaded = union(enum) { unknown, missing, ready: JpegFns };

var pdfium_state: PdfiumLoaded = .unknown;
var jpeg_state: JpegLoaded = .unknown;
var pdfium_mutex: std.Io.Mutex = .init;
var jpeg_mutex: std.Io.Mutex = .init;

fn findPdfium(a: Allocator) PdfiumLoaded {
    const exe = std.process.executableDirPathAlloc(log.io(), a) catch null;
    defer if (exe) |p| a.free(p);
    var dl = sys.optional_libraries.open(sys.optional_libraries.spec(.pdfium), a, sys, exe) orelse return .missing;
    var f: PdfiumFns = undefined;
    const info = @typeInfo(PdfiumFns).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(f, name) = dl.lookup(T, name) orelse {
            dl.close();
            return .missing;
        };
    }
    f.FPDF_InitLibrary();
    return .{ .ready = f };
}

fn findJpeg(a: Allocator) JpegLoaded {
    const exe = std.process.executableDirPathAlloc(log.io(), a) catch null;
    defer if (exe) |p| a.free(p);
    var dl = sys.optional_libraries.open(sys.optional_libraries.spec(.jpeg), a, sys, exe) orelse return .missing;
    var f: JpegFns = undefined;
    const info = @typeInfo(JpegFns).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(f, name) = dl.lookup(T, name) orelse {
            dl.close();
            return .missing;
        };
    }
    return .{ .ready = f };
}

fn getJpeg(a: Allocator) ?JpegFns {
    const io = log.io();
    jpeg_mutex.lockUncancelable(io);
    defer jpeg_mutex.unlock(io);
    if (jpeg_state == .unknown) jpeg_state = findJpeg(a);
    return switch (jpeg_state) {
        .ready => |f| f,
        else => null,
    };
}

/// Configured page limit per PDF document (default 16, capped by the engine's max_images).
pub fn maxPagesLimit(max_images: u32) u32 {
    const cap = if (max_images > 0) max_images else default_max_pages;
    if (sys.getenv("TENSORFOLD_MAX_PDF_PAGES")) |raw| {
        if (std.fmt.parseInt(u32, raw, 10) catch null) |n| {
            if (n > 0) return @min(n, cap);
        }
    }
    return @min(default_max_pages, cap);
}

/// True when an image_url content part carries a data:application/pdf URL.
pub fn isPdfImagePart(part: Value) bool {
    const iu = part.get("image_url") orelse return false;
    const url = if (iu == .string) iu.string else if (iu == .object) (iu.strField("url") orelse return false) else return false;
    return std.ascii.startsWithIgnoreCase(url, "data:application/pdf");
}

/// Expands a `file`, `input_file` or PDF `image_url` content part into one or more `image_url` parts.
pub fn expandPart(cx: *Cx, part: Value, image_type: []const u8, max_images: u32, parts: *std.ArrayList(Value)) errors.Refused!void {
    const pdf_bytes = try extractPartBytes(cx, part);
    var encoded_bytes: usize = 0;
    try expandBytes(cx, pdf_bytes, image_type, maxPagesLimit(max_images), &encoded_bytes, parts);
}

fn extractPartBytes(cx: *Cx, part: Value) errors.Refused![]const u8 {
    if (part.get("image_url")) |iu| {
        const url = if (iu == .string) iu.string else if (iu == .object) (iu.strField("url") orelse return cx.refuse("image_url must contain a url string")) else return cx.refuse("image_url must be a url string or an object with one");
        return decodeFileData(cx, url);
    }
    const container: Value = if (part.get("file")) |f| (if (f == .object) f else return cx.refuse("file content part must contain a file object")) else part;
    if (container.get("file_id") != null and container.get("file_data") == null and container.get("data") == null and container.get("url") == null and container.get("file_url") == null)
        return cx.refuse("file parts require inline file_data (base64 or data:application/pdf;base64,...); file_id is not supported");
    if (container.strField("file_url") orelse container.strField("url")) |u| {
        const trimmed = std.mem.trim(u8, u, " \t\r\n");
        if (trimmed.len > 0 and !std.ascii.startsWithIgnoreCase(trimmed, "data:"))
            return cx.refuse("URLs are not accepted for PDF inputs; send inline base64 or a data:application/pdf;base64,... URL");
    }
    const raw = container.strField("file_data") orelse container.strField("data") orelse container.strField("url") orelse container.strField("file_url") orelse
        return cx.refuse("file content part must contain file_data");
    if (raw.len == 0) return cx.refuse("file_data must be a non-empty base64 string or data URL");
    return decodeFileData(cx, raw);
}

pub fn decodeFileData(cx: *Cx, raw: []const u8) errors.Refused![]const u8 {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (!std.ascii.startsWithIgnoreCase(trimmed, "data:") and (std.ascii.startsWithIgnoreCase(trimmed, "http://") or std.ascii.startsWithIgnoreCase(trimmed, "https://") or std.mem.indexOf(u8, trimmed, "://") != null))
        return cx.refuse("URLs are not accepted for PDF inputs; send inline base64 or a data:application/pdf;base64,... URL");
    const bytes = if (std.ascii.startsWithIgnoreCase(trimmed, "data:")) blk: {
        const comma = std.mem.indexOfScalar(u8, trimmed, ',') orelse return cx.refuse("invalid PDF data URL");
        const header = trimmed[5..comma];
        var it = std.mem.splitScalar(u8, header, ';');
        const mime = it.next() orelse "";
        if (mime.len > 0 and !std.ascii.eqlIgnoreCase(mime, "application/pdf") and !std.ascii.eqlIgnoreCase(mime, "application/octet-stream") and !std.ascii.eqlIgnoreCase(mime, "application/x-pdf"))
            return cx.refuse("file parts support application/pdf only");
        break :blk try images.dataBytesOf(cx, trimmed, max_pdf_bytes, "PDF");
    } else blk: {
        var clean: std.ArrayList(u8) = .empty;
        for (trimmed) |ch| {
            if (!std.ascii.isWhitespace(ch)) try clean.append(cx.a, ch);
        }
        if (clean.items.len == 0) return cx.refuse("PDF is empty or exceeds the encoded byte limit");
        if (clean.items.len > 4 * ((max_pdf_bytes + 2) / 3)) return cx.refuse("PDF exceeds the encoded byte limit");
        const d = std.base64.standard.Decoder;
        const n = d.calcSizeForSlice(clean.items) catch return cx.refuse("invalid PDF base64 encoding");
        if (n == 0 or n > max_pdf_bytes) return cx.refuse("PDF is empty or exceeds the encoded byte limit");
        const out = try cx.a.alloc(u8, n);
        d.decode(out, clean.items) catch return cx.refuse("invalid PDF base64 encoding");
        break :blk out;
    };
    const head = bytes[0..@min(bytes.len, 1024)];
    if (std.mem.indexOf(u8, head, "%PDF-") == null) return cx.refuse("file_data is not a valid PDF document");
    return bytes;
}

/// Renders all pages of `pdf_bytes` (up to `max_pages`, refusing if more) into `image_url` parts appended to `parts`.
pub fn expandBytes(cx: *Cx, pdf_bytes: []const u8, image_type: []const u8, max_pages: u32, encoded_bytes: *usize, parts: *std.ArrayList(Value)) errors.Refused!void {
    const jf = getJpeg(cx.a) orelse return cx.refuse(sys.optional_libraries.spec(.jpeg).message);
    const io = log.io();
    pdfium_mutex.lockUncancelable(io);
    defer pdfium_mutex.unlock(io);
    if (pdfium_state == .unknown) pdfium_state = findPdfium(cx.a);
    return switch (pdfium_state) {
        .ready => |f| renderWithPdfium(cx, f, jf, pdf_bytes, image_type, max_pages, encoded_bytes, parts),
        .missing, .unknown => cx.refuse(sys.optional_libraries.spec(.pdfium).message),
    };
}

fn renderWithPdfium(cx: *Cx, f: PdfiumFns, jf: JpegFns, pdf_bytes: []const u8, image_type: []const u8, max_pages: u32, encoded_bytes: *usize, parts: *std.ArrayList(Value)) errors.Refused!void {
    if (pdf_bytes.len > std.math.maxInt(c_int)) return cx.refuse("PDF exceeds the encoded byte limit");
    const doc = f.FPDF_LoadMemDocument(pdf_bytes.ptr, @intCast(pdf_bytes.len), null) orelse
        return cx.refuse("PDF document could not be opened (corrupt or password-protected)");
    defer f.FPDF_CloseDocument(doc);
    const total_pages = f.FPDF_GetPageCount(doc);
    if (total_pages <= 0) return cx.refuse("PDF document contains no pages");
    const pages: u32 = @intCast(total_pages);
    const limit: u32 = @max(@as(u32, 1), max_pages);
    if (pages > limit) {
        return cx.refuse(try std.fmt.allocPrint(
            cx.a,
            "a PDF may have at most {d} pages (--vision-max-images, TENSORFOLD_MAX_PDF_PAGES); this one has {d}",
            .{ limit, pages },
        ));
    }
    var rendered_any = false;
    for (0..pages) |idx| {
        const page = f.FPDF_LoadPage(doc, @intCast(idx)) orelse continue;
        defer f.FPDF_ClosePage(page);
        const pw_raw = f.FPDF_GetPageWidthF(page);
        const ph_raw = f.FPDF_GetPageHeightF(page);
        const pw: f32 = if (std.math.isFinite(pw_raw) and pw_raw > 1.0) pw_raw else 595.0;
        const ph: f32 = if (std.math.isFinite(ph_raw) and ph_raw > 1.0) ph_raw else 842.0;
        var scale: f32 = 2.0; // 144 DPI baseline
        const max_side: f32 = 2048.0;
        const max_area: f32 = 3_600_000.0;
        if (pw * scale > max_side) scale = max_side / pw;
        if (ph * scale > max_side) scale = max_side / ph;
        if ((pw * scale) * (ph * scale) > max_area) scale = @sqrt(max_area / (pw * ph));
        const w: c_int = @max(28, @min(4096, @as(c_int, @intFromFloat(@round(pw * scale)))));
        const h: c_int = @max(28, @min(4096, @as(c_int, @intFromFloat(@round(ph * scale)))));
        const bmp = f.FPDFBitmap_Create(w, h, 0) orelse return error.OutOfMemory;
        defer f.FPDFBitmap_Destroy(bmp);
        const stride = f.FPDFBitmap_GetStride(bmp);
        const buf_ptr = f.FPDFBitmap_GetBuffer(bmp) orelse return error.OutOfMemory;
        if (stride < w * 4) return cx.refuse("PDF page bitmap has invalid stride");
        const bgra: [*]u8 = @ptrCast(buf_ptr);
        const total_bytes = @as(usize, @intCast(stride)) * @as(usize, @intCast(h));
        @memset(bgra[0..total_bytes], 0xFF); // white background
        // 0x01 = FPDF_ANNOT, 0x10 = FPDF_PRINTING
        f.FPDF_RenderPageBitmap(bmp, page, 0, 0, w, h, 0, 0x01 | 0x10);
        const jpeg_bytes = encodeJpegBgra(cx.a, jf, bgra[0..total_bytes], @intCast(w), @intCast(h), @intCast(stride)) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.EncodeFailed => return cx.refuse("rendered PDF page could not be encoded as JPEG"),
        };
        if (jpeg_bytes.len > images.max_image_bytes) return cx.refuse("rendered PDF page exceeds the 10 MB image byte limit");
        if (encoded_bytes.* + jpeg_bytes.len > images.max_request_bytes) return cx.refuse("the request's images exceed 20 MB encoded");
        encoded_bytes.* += jpeg_bytes.len;
        const data_url = try makeDataUrl(cx.a, "image/jpeg", jpeg_bytes);
        try appendImageUrlPart(cx.a, parts, image_type, data_url);
        rendered_any = true;
    }
    if (!rendered_any) return cx.refuse("PDF document pages could not be rendered");
}

fn encodeJpegBgra(a: Allocator, jf: JpegFns, bgra: []const u8, width: usize, height: usize, stride: usize) error{ OutOfMemory, EncodeFailed }![]const u8 {
    const handle = jf.tj3Init(0) orelse return error.EncodeFailed; // 0 = TJINIT_COMPRESS
    defer jf.tj3Destroy(handle);
    _ = jf.tj3Set(handle, 3, 85); // TJPARAM_QUALITY = 3
    _ = jf.tj3Set(handle, 4, 2); // TJPARAM_SUBSAMP = 4, TJSAMP_420 = 2
    var out_buf: ?[*]u8 = null;
    var out_size: usize = 0;
    // TJPF_BGRX = 8 (matches FPDFBitmap_Create with alpha=0)
    const rc = jf.tj3Compress8(handle, bgra.ptr, @intCast(width), @intCast(stride), @intCast(height), 8, &out_buf, &out_size);
    defer if (out_buf) |p| jf.tj3Free(p);
    if (rc != 0 or out_buf == null or out_size == 0) return error.EncodeFailed;
    return a.dupe(u8, out_buf.?[0..out_size]);
}

pub fn makeDataUrl(a: Allocator, mime: []const u8, bytes: []const u8) Allocator.Error![]const u8 {
    const prefix = try std.fmt.allocPrint(a, "data:{s};base64,", .{mime});
    const enc = std.base64.standard.Encoder;
    const b64_len = enc.calcSize(bytes.len);
    const out = try a.alloc(u8, prefix.len + b64_len);
    @memcpy(out[0..prefix.len], prefix);
    _ = enc.encode(out[prefix.len..], bytes);
    return out;
}

fn appendImageUrlPart(a: Allocator, parts: *std.ArrayList(Value), image_type: []const u8, data_url: []const u8) Allocator.Error!void {
    const inner = try json.newObject(a);
    try inner.put(a, "url", .{ .string = data_url });
    const p = try json.newObject(a);
    try p.put(a, "type", .{ .string = image_type });
    try p.put(a, "image_url", .{ .object = inner });
    try parts.append(a, .{ .object = p });
}

pub const minimal_two_page_pdf =
    "%PDF-1.4\n" ++
    "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n" ++
    "2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj\n" ++
    "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj\n" ++
    "4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj\n" ++
    "xref\n0 5\n" ++
    "0000000000 65535 f \n" ++
    "0000000009 00000 n \n" ++
    "0000000058 00000 n \n" ++
    "0000000121 00000 n \n" ++
    "0000000190 00000 n \n" ++
    "trailer << /Size 5 /Root 1 0 R >>\n" ++
    "startxref\n259\n%%EOF\n";

test "decodeFileData validates PDF headers and refuses remote URLs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };

    const data_url = try makeDataUrl(a, "application/pdf", minimal_two_page_pdf);
    const decoded = try decodeFileData(&cx, data_url);
    try std.testing.expectEqualSlices(u8, minimal_two_page_pdf, decoded);

    var url_cx: Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, decodeFileData(&url_cx, "https://example.com/doc.pdf"));
    try std.testing.expect(std.mem.indexOf(u8, url_cx.message, "URLs are not accepted") != null);
}

test "libpdfium renders a multi-page PDF document and refuses when page count exceeds limit" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };

    var parts: std.ArrayList(Value) = .empty;
    var used: usize = 0;
    expandBytes(&cx, minimal_two_page_pdf, "image_url", 4, &used, &parts) catch |err| {
        if (err == error.Refused and (pdfium_state == .missing or jpeg_state == .missing)) {
            const want = if (jpeg_state == .missing) sys.optional_libraries.spec(.jpeg).names[0] else sys.optional_libraries.spec(.pdfium).names[0];
            try std.testing.expect(std.mem.indexOf(u8, cx.message, want) != null);
            return;
        }
        return err;
    };
    try std.testing.expectEqual(@as(usize, 2), parts.items.len);
    try std.testing.expect(used > 0 and used <= images.max_request_bytes);
    for (parts.items) |p| {
        const url = p.get("image_url").?.get("url").?.string;
        const img_bytes = try images.dataBytes(&cx, url);
        try std.testing.expectEqual(images.Format.jpeg, images.sniff(img_bytes));
    }

    // Refuse when total_pages (2) > max_pages (1), naming both numbers in the refusal
    var cap_cx: Cx = .{ .a = a };
    var cap_parts: std.ArrayList(Value) = .empty;
    var cap_used: usize = 0;
    try std.testing.expectError(error.Refused, expandBytes(&cap_cx, minimal_two_page_pdf, "image_url", 1, &cap_used, &cap_parts));
    try std.testing.expect(std.mem.indexOf(u8, cap_cx.message, "at most 1 pages") != null);
    try std.testing.expect(std.mem.indexOf(u8, cap_cx.message, "has 2") != null);
}
