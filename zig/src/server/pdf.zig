//! PDF document input (--vision --vision-pdf): renders PDF pages via PDFium and libjpeg-turbo in a bounded worker.
const std = @import("std");
const builtin = @import("builtin");
const sys = @import("sys");
const json = @import("json");
const errors = @import("errors.zig");
const images = @import("images.zig");
const jpeg_out = @import("jpeg_out.zig");
const log = @import("log.zig");
const worker = @import("pdf_worker.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const env = "TENSORFOLD_LIBPDFIUM";
pub const max_pdf_bytes: usize = 32 << 20;
pub const max_request_pixels: u64 = 32 << 20;
pub const default_max_pages: u32 = 16;
pub const render_timeout_ns: i96 = 10 * std.time.ns_per_s;
pub const max_child_heap_bytes: u64 = 1 << 30;
pub const off_message = "PDF inputs are off on this server; start the server with --vision --vision-pdf";

pub const Budget = struct {
    images_used: u32 = 0,
    pdf_pages_used: u32 = 0,
    encoded_bytes: usize = 0,
    pixels_used: u64 = 0,
    /// Counts one rendered page of `px` pixels and `len` JPEG bytes.
    pub fn add(b: *Budget, px: u64, len: usize) void {
        b.pdf_pages_used += 1;
        b.images_used += 1;
        b.pixels_used += px;
        b.encoded_bytes += len;
    }
};
pub const IfsdkPause = extern struct { version: c_int = 1, NeedToPauseNow: *const fn (*IfsdkPause) callconv(.c) c_int, user: ?*anyopaque = null };

pub const PdfiumFns = struct {
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
    FPDF_RenderPageBitmap_Start: ?*const fn (?*anyopaque, ?*anyopaque, c_int, c_int, c_int, c_int, c_int, c_int, *IfsdkPause) callconv(.c) c_int = null,
    FPDF_RenderPage_Continue: ?*const fn (?*anyopaque, *IfsdkPause) callconv(.c) c_int = null,
    FPDF_RenderPage_Close: ?*const fn (?*anyopaque) callconv(.c) void = null,
};

const DlInfo = extern struct { fname: ?[*:0]const u8, fbase: ?*anyopaque, sname: ?[*:0]const u8, saddr: ?*anyopaque };
extern "c" fn dladdr(addr: *const anyopaque, info: *DlInfo) c_int;

const PdfiumLoaded = union(enum) { unknown, missing, ready: PdfiumFns };

var pdfium_state: PdfiumLoaded = .unknown;
var pdfium_mutex: std.Io.Mutex = .init;

fn findPdfium(a: Allocator) PdfiumLoaded {
    const f = jpeg_out.loadDynLib(PdfiumFns, .pdfium, a) orelse return .missing;
    if (sys.windows) f.FPDF_InitLibrary();
    return .{ .ready = f };
}

/// Logs the loaded PDFium path and SHA-256 at startup when --vision-pdf is enabled, or warns if a library is missing.
pub fn logStartup(a: Allocator, io: std.Io) void {
    if (jpeg_out.get(a) == null) log.line("--vision-pdf: {s}", .{sys.optional_libraries.spec(.jpeg).message});
    pdfium_mutex.lockUncancelable(io);
    defer pdfium_mutex.unlock(io);
    if (pdfium_state == .unknown) pdfium_state = findPdfium(a);
    const f = if (pdfium_state == .ready) pdfium_state.ready else return log.line("--vision-pdf: {s}", .{sys.optional_libraries.spec(.pdfium).message});
    var path_buf: [1100]u8 = undefined;
    const symbol: *const anyopaque = @ptrCast(f.FPDF_InitLibrary);
    const file = if (comptime sys.windows) sys.libraryOf(symbol, &path_buf) orelse "pdfium.dll" else blk: {
        var info: DlInfo = undefined;
        break :blk if (dladdr(symbol, &info) != 0 and info.fname != null) std.mem.span(info.fname.?) else "libpdfium";
    };
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(128 << 20)) catch return log.line("PDF inputs (--vision-pdf): {s}", .{file});
    defer a.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    log.line("PDF inputs (--vision-pdf): {s} (sha256 {s})", .{ file, &hex });
}

/// Configured page limit per request's PDFs (default 16, capped by the engine's max_images).
pub fn maxPagesLimit(max_images: u32) u32 {
    const cap = if (max_images > 0) max_images else default_max_pages;
    if (sys.getenv("TENSORFOLD_MAX_PDF_PAGES")) |raw| if (std.fmt.parseInt(u32, raw, 10) catch null) |n| if (n > 0) return @min(n, cap);
    return @min(default_max_pages, cap);
}

/// True when an image_url content part carries a data:application/pdf URL.
pub fn isPdfImagePart(part: Value) bool {
    const iu = part.get("image_url") orelse return false;
    const url = if (iu == .string) iu.string else if (iu == .object) (iu.strField("url") orelse return false) else return false;
    return std.ascii.startsWithIgnoreCase(url, "data:application/pdf");
}

/// Accounts for a non-PDF image_url part in the request-wide budget so mixed PDF + image requests enforce one cap.
pub fn noteImagePart(cx: *Cx, part: Value, max_images: u32, budget: *Budget) errors.Refused!void {
    budget.images_used += 1;
    if (budget.pdf_pages_used > 0 and max_images > 0 and budget.images_used > max_images)
        return cx.refuse(try std.fmt.allocPrint(cx.a, "a request may hold at most {d} images (--vision-max-images); it has {d}", .{ max_images, budget.images_used }));
    const iu = part.get("image_url") orelse return;
    const url = if (iu == .string) iu.string else if (iu == .object) (iu.strField("url") orelse return) else return;
    if (!std.ascii.startsWithIgnoreCase(url, "data:")) return;
    var scratch: Cx = .{ .a = cx.a };
    const bytes = images.dataBytes(&scratch, url) catch return;
    if (budget.pdf_pages_used > 0 and budget.encoded_bytes + bytes.len > images.max_request_bytes)
        return cx.refuse("the request's images exceed 20 MB encoded");
    budget.encoded_bytes +|= bytes.len;
}

/// Expands a `file`, `input_file` or PDF `image_url` content part into one or more `image_url` parts.
pub fn expandPart(cx: *Cx, part: Value, image_type: []const u8, max_images: u32, budget: *Budget, parts: *std.ArrayList(Value)) errors.Refused!void {
    const pdf_bytes = try extractPartBytes(cx, part);
    try expandBytes(cx, pdf_bytes, image_type, max_images, maxPagesLimit(max_images), budget, parts);
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
        const max_b64_len = 4 * ((max_pdf_bytes + 2) / 3);
        if (trimmed.len == 0) return cx.refuse("PDF is empty or exceeds the encoded byte limit");
        if (trimmed.len > max_b64_len + (max_b64_len / 16)) return cx.refuse("PDF exceeds the encoded byte limit");
        var clean: std.ArrayList(u8) = .empty;
        for (trimmed) |ch| if (!std.ascii.isWhitespace(ch)) {
            if (clean.items.len >= max_b64_len) return cx.refuse("PDF exceeds the encoded byte limit");
            try clean.append(cx.a, ch);
        };
        if (clean.items.len == 0) return cx.refuse("PDF is empty or exceeds the encoded byte limit");
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

/// Which pages `renderPages` renders, by 0-based index.
pub const Selection = union(enum) {
    /// Every page; refused when the document has more (the page budget left).
    all_within: u32,
    /// The first n pages; the rest are dropped (PaddleX).
    first: u32,
    /// These indices, ascending and distinct; one past the last page is refused.
    list: []const u32,
};
/// One call's limits over the caller's running Budget; `infra` is the error kind of a renderer failure.
pub const Caps = struct { request_pixels: u64, request_bytes: usize, page_bytes: usize, max_images: u32, infra: errors.Kind };
/// A rendered page: its 0-based index, JPEG bytes, pixel size and the DPI it was rendered at.
pub const Page = struct { index: u32, jpeg: []const u8, width: u32, height: u32, dpi: u32 };
/// The rendered pages in selection order, and the document's page count.
pub const Rendered = struct { pages: []Page, total: u32 };
/// One render call as the in-process renderer and the POSIX worker (pdf_worker.zig) take it.
pub const Job = struct {
    pdf_bytes: []const u8,
    sel: Selection,
    caps: Caps,
    deadline_ns: i96,
    /// Each selected page's index, size and DPI from its geometry alone: no bitmap, no JPEG (`jpeg` empty).
    measure: bool = false,
};

/// A rendered page's shortest and longest side in pixels and its highest DPI, from pageDimensions.
pub const min_side: u32 = 28;
pub const max_side: u32 = 2048;
pub const max_dpi: u32 = 144;

/// Chat caps: 32 MP, 20 MB encoded, 10 MB a page and `max_images` images (0: no cap); a renderer failure is a 400.
pub fn chatCaps(max_images: u32) Caps {
    return .{ .request_pixels = max_request_pixels, .request_bytes = images.max_request_bytes, .page_bytes = images.max_image_bytes, .max_images = max_images, .infra = .request };
}

/// An OCR call's caps for `pages` pages: 2048 px a side and 10 MB each, no image cap; a renderer failure is a 500.
pub fn ocrCaps(pages: u32) Caps {
    return .{ .request_pixels = @as(u64, pages) * max_side * max_side, .request_bytes = @as(usize, pages) * images.max_image_bytes, .page_bytes = images.max_image_bytes, .max_images = 0, .infra = .server };
}

/// Renders all pages of `pdf_bytes` within the request-wide `budget` into `image_url` parts appended to `parts`.
pub fn expandBytes(cx: *Cx, pdf_bytes: []const u8, image_type: []const u8, max_images: u32, max_pages: u32, budget: *Budget, parts: *std.ArrayList(Value)) errors.Refused!void {
    const sel: Selection = .{ .all_within = @max(@as(u32, 1), max_pages) -| budget.pdf_pages_used };
    const r = try renderPages(cx, pdf_bytes, sel, chatCaps(max_images), budget, null);
    for (r.pages) |p| try appendImageUrlPart(cx.a, parts, image_type, try makeDataUrl(cx.a, "image/jpeg", p.jpeg));
}

/// renderPages' errors: a probe's `error.Cancelled` besides the refusals; without a probe (chat) the refusals alone.
pub fn RenderError(comptime Gone: type) type {
    return if (Gone == @TypeOf(null)) errors.Refused else errors.Refused || error{Cancelled};
}

/// Renders `sel` within `caps` and the running `budget`, one worker a call; `gone`: null or a probe, asked once locked.
pub fn renderPages(cx: *Cx, pdf_bytes: []const u8, sel: Selection, caps: Caps, budget: *Budget, gone: anytype) RenderError(@TypeOf(gone))!Rendered {
    return call(cx, pdf_bytes, sel, caps, budget, gone, false);
}

/// renderPages' admission, sizes and DPI with no page rendered: each page charged with no bytes, its `jpeg` empty.
pub fn measurePages(cx: *Cx, pdf_bytes: []const u8, sel: Selection, caps: Caps, budget: *Budget, gone: anytype) RenderError(@TypeOf(gone))!Rendered {
    return call(cx, pdf_bytes, sel, caps, budget, gone, true);
}

fn call(cx: *Cx, pdf_bytes: []const u8, sel: Selection, caps: Caps, budget: *Budget, gone: anytype, measure: bool) RenderError(@TypeOf(gone))!Rendered {
    const jf = jpeg_out.get(cx.a) orelse return cx.fail(.unimplemented, "{s}", .{sys.optional_libraries.spec(.jpeg).message});
    const io = log.io();
    pdfium_mutex.lockUncancelable(io);
    defer pdfium_mutex.unlock(io);
    // a client that left while queued here starts no worker with a fresh deadline ahead of live requests
    if (@TypeOf(gone) != @TypeOf(null) and gone.check()) return error.Cancelled;
    if (pdfium_state == .unknown) pdfium_state = findPdfium(cx.a);
    const f = if (pdfium_state == .ready) pdfium_state.ready else return cx.fail(.unimplemented, "{s}", .{sys.optional_libraries.spec(.pdfium).message});
    const job: Job = .{ .pdf_bytes = pdf_bytes, .sel = sel, .caps = caps, .deadline_ns = std.Io.Clock.awake.now(io).toNanoseconds() + render_timeout_ns, .measure = measure };
    var pages: std.ArrayList(Page) = .empty;
    const total = if (comptime sys.windows) try renderInProcess(cx, f, jf, job, budget, &pages) else blk: {
        const cmd = worker.argv(cx.a) catch return cx.fail(caps.infra, "PDF rendering worker process could not be started", .{});
        break :blk try worker.render(cx, cmd, job, budget, &pages);
    };
    return .{ .pages = pages.items, .total = total };
}

/// How many pages `sel` takes from `total`, refused over the page or image budget; pdf_worker.zig checks with it too.
pub fn admitCount(cx: *Cx, sel: Selection, caps: Caps, budget: *const Budget, total: u32) errors.Refused!u32 {
    if (total == 0) return cx.refuse("PDF document contains no pages");
    const count: u32 = switch (sel) {
        .all_within => |n| if (total <= n) total else return cx.fail(.request, "a PDF may have at most {d} pages (--vision-max-images, TENSORFOLD_MAX_PDF_PAGES); this request has {d}", .{ budget.pdf_pages_used +| n, budget.pdf_pages_used +| total }),
        .first => |n| @min(n, total),
        .list => |l| for (l, 0..) |i, k| {
            if (k > 0 and i <= l[k - 1]) return cx.refuse("PDF page indices must be ascending and distinct");
            if (i >= total) return cx.fail(.request, "page {d} is out of range; the document has {d} pages", .{ i, total });
        } else @intCast(l.len),
    };
    if (caps.max_images > 0 and budget.images_used +| count > caps.max_images)
        return cx.fail(.request, "a request may hold at most {d} images (--vision-max-images); it has {d}", .{ caps.max_images, budget.images_used +| count });
    return count;
}

/// The index of the selection's `k`-th page.
pub fn pageAt(sel: Selection, k: u32) u32 {
    return if (sel == .list) sel.list[k] else k;
}

/// Refuses a page of `px` pixels and `len` JPEG bytes (0 before the encode: some room must be left) over `caps`.
pub fn checkPage(cx: *Cx, caps: Caps, budget: *const Budget, px: u64, len: usize) errors.Refused!void {
    if (budget.pixels_used +| px > caps.request_pixels) return cx.fail(.request, "the request's images exceed the {d}-megapixel limit", .{caps.request_pixels >> 20});
    if (len > caps.page_bytes) return cx.fail(.request, "rendered PDF page exceeds the {d} MB image byte limit", .{caps.page_bytes >> 20});
    if (budget.encoded_bytes +| @max(len, 1) > caps.request_bytes) return cx.fail(.request, "the request's images exceed {d} MB encoded", .{caps.request_bytes >> 20});
}

const PauseState = struct {
    pause: IfsdkPause,
    deadline_ns: i96,
    timed_out: bool = false,
    fn check(p: *IfsdkPause) callconv(.c) c_int {
        const self: *PauseState = @alignCast(@fieldParentPtr("pause", p));
        if (std.Io.Clock.awake.now(log.io()).toNanoseconds() < self.deadline_ns) return 0;
        self.timed_out = true;
        return 1;
    }
};

fn expired(deadline_ns: i96) bool {
    return std.Io.Clock.awake.now(log.io()).toNanoseconds() >= deadline_ns;
}

pub const render_flags: c_int = 0x01; // FPDF_ANNOT; 0x10 is REVERSE_BYTE_ORDER, RGBA where the JPEG reads BGRA
fn renderPageWithDeadline(cx: *Cx, f: PdfiumFns, job: Job, bmp: ?*anyopaque, page: ?*anyopaque, w: c_int, h: c_int, idx: usize) errors.Refused!void {
    if (f.FPDF_RenderPageBitmap_Start != null and f.FPDF_RenderPage_Continue != null and f.FPDF_RenderPage_Close != null) {
        var state: PauseState = .{ .pause = .{ .version = 1, .NeedToPauseNow = PauseState.check }, .deadline_ns = job.deadline_ns };
        var status = f.FPDF_RenderPageBitmap_Start.?(bmp, page, 0, 0, w, h, 0, render_flags, &state.pause);
        while (status == 1) {
            if (state.timed_out or expired(job.deadline_ns)) {
                f.FPDF_RenderPage_Close.?(page);
                return cx.fail(job.caps.infra, "PDF rendering timed out", .{});
            }
            status = f.FPDF_RenderPage_Continue.?(page, &state.pause);
        }
        f.FPDF_RenderPage_Close.?(page);
        if (state.timed_out or expired(job.deadline_ns)) return cx.fail(job.caps.infra, "PDF rendering timed out", .{});
        if (status == 3) return cx.refuse(try std.fmt.allocPrint(cx.a, "PDF page {d} could not be rendered", .{idx + 1}));
        return;
    }
    f.FPDF_RenderPageBitmap(bmp, page, 0, 0, w, h, 0, render_flags);
    if (expired(job.deadline_ns)) return cx.fail(job.caps.infra, "PDF rendering timed out", .{});
}

/// Renders `job` in this process (Windows, and the POSIX worker); returns the document's page count.
pub fn renderInProcess(cx: *Cx, f: PdfiumFns, jf: jpeg_out.Fns, job: Job, budget: *Budget, out: *std.ArrayList(Page)) errors.Refused!u32 {
    if (job.pdf_bytes.len > std.math.maxInt(c_int)) return cx.refuse("PDF exceeds the encoded byte limit");
    if (expired(job.deadline_ns)) return cx.fail(job.caps.infra, "PDF rendering timed out", .{});
    const doc = f.FPDF_LoadMemDocument(job.pdf_bytes.ptr, @intCast(job.pdf_bytes.len), null) orelse
        return cx.refuse("PDF document could not be opened (corrupt or password-protected)");
    defer f.FPDF_CloseDocument(doc);
    const total: u32 = @intCast(@max(0, f.FPDF_GetPageCount(doc)));
    const count = try admitCount(cx, job.sel, job.caps, budget, total);
    for (0..count) |k| {
        const idx = pageAt(job.sel, @intCast(k));
        if (expired(job.deadline_ns)) return cx.fail(job.caps.infra, "PDF rendering timed out", .{});
        const page = f.FPDF_LoadPage(doc, @intCast(idx)) orelse
            return cx.refuse(try std.fmt.allocPrint(cx.a, "PDF page {d} could not be rendered", .{idx + 1}));
        defer f.FPDF_ClosePage(page);
        const dims = pageDimensions(f.FPDF_GetPageWidthF(page), f.FPDF_GetPageHeightF(page));
        const page_pixels = @as(u64, @intCast(dims.w)) * @as(u64, @intCast(dims.h));
        try checkPage(cx, job.caps, budget, page_pixels, 0);
        if (job.measure) {
            // the parent's deadline guards the POSIX worker; in process, a load past the deadline must not pass
            if (expired(job.deadline_ns)) return cx.fail(job.caps.infra, "PDF rendering timed out", .{});
            budget.add(page_pixels, 0);
            try out.append(cx.a, .{ .index = idx, .jpeg = &.{}, .width = @intCast(dims.w), .height = @intCast(dims.h), .dpi = pageDpi(dims) });
            continue;
        }
        const bmp = f.FPDFBitmap_Create(dims.w, dims.h, 0) orelse return error.OutOfMemory;
        defer f.FPDFBitmap_Destroy(bmp);
        const stride = f.FPDFBitmap_GetStride(bmp);
        const buf_ptr = f.FPDFBitmap_GetBuffer(bmp) orelse return error.OutOfMemory;
        if (stride < dims.w * 4) return cx.refuse("PDF page bitmap has invalid stride");
        const bgra: [*]u8 = @ptrCast(buf_ptr);
        const total_bytes = @as(usize, @intCast(stride)) * @as(usize, @intCast(dims.h));
        @memset(bgra[0..total_bytes], 0xFF);
        try renderPageWithDeadline(cx, f, job, bmp, page, dims.w, dims.h, idx);
        const jpeg_bytes = jpeg_out.encodeBgra(cx.a, jf, bgra[0..total_bytes], @intCast(dims.w), @intCast(dims.h), @intCast(stride)) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.EncodeFailed => return cx.refuse("rendered PDF page could not be encoded as JPEG"),
        };
        try checkPage(cx, job.caps, budget, page_pixels, jpeg_bytes.len);
        budget.add(page_pixels, jpeg_bytes.len);
        try out.append(cx.a, .{ .index = idx, .jpeg = jpeg_bytes, .width = @intCast(dims.w), .height = @intCast(dims.h), .dpi = pageDpi(dims) });
    }
    return total;
}

const Dims = struct { w: c_int, h: c_int, scale: f32 };
/// round(72 x scale), at least 1: a huge page's scale can round to 0 DPI.
fn pageDpi(dims: Dims) u32 {
    return @max(1, @as(u32, @intFromFloat(@round(72.0 * dims.scale))));
}

fn pageDimensions(pw_raw: f32, ph_raw: f32) Dims {
    const pw: f32 = if (std.math.isFinite(pw_raw) and pw_raw > 1.0) pw_raw else 595.0;
    const ph: f32 = if (std.math.isFinite(ph_raw) and ph_raw > 1.0) ph_raw else 842.0;
    var scale: f32 = @as(f32, @floatFromInt(max_dpi)) / 72.0; // 144 DPI baseline
    const side: f32 = @floatFromInt(max_side);
    const max_area: f32 = 3_600_000.0;
    if (pw * scale > side) scale = side / pw;
    if (ph * scale > side) scale = side / ph;
    if ((pw * scale) * (ph * scale) > max_area) scale = @sqrt(max_area / (pw * ph));
    const w: c_int = @max(min_side, @min(4096, @as(c_int, @intFromFloat(@round(pw * scale)))));
    const h: c_int = @max(min_side, @min(4096, @as(c_int, @intFromFloat(@round(ph * scale)))));
    return .{ .w = w, .h = h, .scale = scale };
}

/// The width and height a baseline or progressive JPEG's frame header names, or null when it has none.
pub fn jpegSize(bytes: []const u8) ?[2]u32 {
    if (bytes.len < 4 or bytes[0] != 0xFF or bytes[1] != 0xD8) return null;
    var i: usize = 2;
    while (i + 4 <= bytes.len) {
        if (bytes[i] != 0xFF) return null;
        while (i < bytes.len and bytes[i] == 0xFF) i += 1;
        if (i >= bytes.len) return null;
        const marker = bytes[i];
        i += 1;
        if (marker == 0xD8 or marker == 0xD9 or (marker >= 0xD0 and marker <= 0xD7) or marker == 0x01) continue;
        if (i + 2 > bytes.len) return null;
        const seg_len: usize = std.mem.readInt(u16, bytes[i..][0..2], .big);
        if (seg_len < 2 or i + seg_len > bytes.len) return null;
        if (marker == 0xC0 or marker == 0xC1 or marker == 0xC2) {
            if (seg_len < 7) return null;
            const h: u32 = std.mem.readInt(u16, bytes[i + 3 ..][0..2], .big);
            const w: u32 = std.mem.readInt(u16, bytes[i + 5 ..][0..2], .big);
            return if (w > 0 and h > 0) .{ w, h } else null;
        }
        i += seg_len;
    }
    return null;
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

pub const minimal_two_page_pdf = "%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj\n4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000121 00000 n \n0000000190 00000 n \ntrailer << /Size 5 /Root 1 0 R >>\nstartxref\n259\n%%EOF\n";

test "decodeFileData and parseChildReply validate untrusted input and recompute parent budgets" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const data_url = try makeDataUrl(a, "application/pdf", minimal_two_page_pdf);
    try std.testing.expectEqualSlices(u8, minimal_two_page_pdf, try decodeFileData(&cx, data_url));
    var url_cx: Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, decodeFileData(&url_cx, "https://example.com/doc.pdf"));
    try std.testing.expect(std.mem.indexOf(u8, url_cx.message, "URLs are not accepted") != null);
    var bad_reply_cx: Cx = .{ .a = a };
    var budget: Budget = .{};
    var pages: std.ArrayList(Page) = .empty;
    const reply = [_]u8{ 1, 0, 3, 0, 0, 0, 'b', 'a', 'd' };
    try std.testing.expectError(error.Refused, worker.parseChildReply(&bad_reply_cx, &reply, .{ .all_within = 4 }, false, chatCaps(4), &budget, &pages));
    try std.testing.expectEqual(errors.Kind.request, bad_reply_cx.kind);
    try std.testing.expectEqualStrings("bad", bad_reply_cx.message);
}

/// Renders `sel` of the two-page fixture in this process by `timeout_ns`; skips without PDFium or libjpeg-turbo.
fn renderFixture(cx: *Cx, sel: Selection, caps: Caps, budget: *Budget, timeout_ns: i96) !Rendered {
    const jf = jpeg_out.get(cx.a) orelse return error.SkipZigTest;
    const f = jpeg_out.loadDynLib(PdfiumFns, .pdfium, cx.a) orelse return error.SkipZigTest;
    f.FPDF_InitLibrary(); // a test runner cannot be the worker (pdf_worker_test.zig renders through it)
    var pages: std.ArrayList(Page) = .empty;
    const job: Job = .{ .pdf_bytes = minimal_two_page_pdf, .sel = sel, .caps = caps, .deadline_ns = std.Io.Clock.awake.now(log.io()).toNanoseconds() + timeout_ns };
    const total = try renderInProcess(cx, f, jf, job, budget, &pages);
    return .{ .pages = pages.items, .total = total };
}

test "pdf pages: a list renders only its pages, with their indices, size and DPI" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var budget: Budget = .{};
    const r = try renderFixture(&cx, .{ .list = &.{1} }, ocrCaps(16), &budget, render_timeout_ns);
    try std.testing.expectEqual(@as(u32, 2), r.total);
    try std.testing.expectEqual(@as(usize, 1), r.pages.len);
    const p = r.pages[0];
    try std.testing.expectEqual(Page{ .index = 1, .jpeg = p.jpeg, .width = 400, .height = 400, .dpi = 144 }, p);
    try std.testing.expectEqual([2]u32{ 400, 400 }, jpegSize(p.jpeg).?);
    try std.testing.expectEqual(Budget{ .images_used = 1, .pdf_pages_used = 1, .encoded_bytes = p.jpeg.len, .pixels_used = 400 * 400 }, budget);
    var out_cx: Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, renderFixture(&out_cx, .{ .list = &.{ 0, 2 } }, ocrCaps(16), &budget, render_timeout_ns));
    try std.testing.expectEqualStrings("page 2 is out of range; the document has 2 pages", out_cx.message);
    try std.testing.expectEqual(errors.Kind.request, out_cx.kind);
}

test "pdf pages: first(1) renders page 0 and reports the document's two pages" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    var budget: Budget = .{};
    const r = try renderFixture(&cx, .{ .first = 1 }, chatCaps(4), &budget, render_timeout_ns);
    try std.testing.expectEqual(@as(u32, 2), r.total);
    try std.testing.expectEqual(@as(usize, 1), r.pages.len);
    try std.testing.expectEqual(@as(u32, 0), r.pages[0].index);
    try std.testing.expectEqual(@as(u32, 1), budget.pdf_pages_used);
}

test "pdf pages: a renderer deadline is a server error under OCR caps and a request error under chat caps" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_]Caps{ ocrCaps(16), chatCaps(4) }) |caps| {
        var cx: Cx = .{ .a = arena.allocator() };
        var budget: Budget = .{};
        if (renderFixture(&cx, .{ .all_within = 16 }, caps, &budget, 0)) |_| return error.TestUnexpectedResult else |err| if (err != error.Refused) return err;
        try std.testing.expectEqual(caps.infra, cx.kind);
        try std.testing.expectEqualStrings("PDF rendering timed out", cx.message);
        try std.testing.expectEqual(Budget{}, budget);
    }
}

test "pdf pages: a huge page fits 2048 px a side and keeps a DPI of at least 1" {
    const dims = pageDimensions(400_000, 200_000);
    try std.testing.expectEqual(@as(c_int, 2048), dims.w);
    try std.testing.expectEqual(@as(c_int, 1024), dims.h);
    try std.testing.expectEqual(@as(u32, 0), @as(u32, @intFromFloat(@round(72.0 * dims.scale))));
    try std.testing.expectEqual(@as(u32, 1), pageDpi(dims));
    try std.testing.expectEqual(max_dpi, pageDpi(pageDimensions(200, 200)));
}

/// A cancellation probe that counts its calls, says whether the renderer was held then, and reports `left`.
const Probe = struct {
    left: bool,
    calls: u32 = 0,
    held: bool = false,
    pub fn check(p: *Probe) bool {
        p.calls += 1;
        p.held = !pdfium_mutex.tryLock();
        if (!p.held) pdfium_mutex.unlock(log.io());
        return p.left;
    }
};

test "pdf pages: a client that left while queued for the renderer gets Cancelled and starts no worker" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest; // Windows renders in process, with no worker to refuse
    log.init(std.testing.io, false);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if (jpeg_out.get(a) == null) return error.SkipZigTest; // without libjpeg-turbo the call refuses before the lock
    // a worker start would refuse with "could not be started", never Cancelled
    const saved = worker.server_exe;
    defer worker.server_exe = saved;
    worker.server_exe = "/nonexistent/tensorfold-native";
    var budget: Budget = .{};
    var cx: Cx = .{ .a = a };
    var left: Probe = .{ .left = true };
    try std.testing.expectError(error.Cancelled, renderPages(&cx, minimal_two_page_pdf, .{ .all_within = 16 }, ocrCaps(16), &budget, &left));
    try std.testing.expectEqual(@as(u32, 1), left.calls);
    try std.testing.expect(left.held); // asked with the renderer held: after the wait, before any worker
    try std.testing.expectEqual(Budget{}, budget);
    try std.testing.expectEqualStrings("", cx.message);
    // a client still there, and chat's null probe (whose error set stays Refused), take the same path past the check
    comptime std.debug.assert(RenderError(@TypeOf(null)) == errors.Refused);
    var here: Probe = .{ .left = false };
    var here_cx: Cx = .{ .a = a };
    var chat_cx: Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, renderPages(&here_cx, minimal_two_page_pdf, .{ .all_within = 16 }, chatCaps(4), &budget, &here));
    try std.testing.expectError(error.Refused, renderPages(&chat_cx, minimal_two_page_pdf, .{ .all_within = 16 }, chatCaps(4), &budget, null));
    try std.testing.expectEqual(@as(u32, 1), here.calls);
    try std.testing.expectEqualStrings(chat_cx.message, here_cx.message);
    try std.testing.expectEqual(chat_cx.kind, here_cx.kind);
    try std.testing.expectEqual(Budget{}, budget);
}
