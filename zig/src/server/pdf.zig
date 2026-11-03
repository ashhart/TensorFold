//! PDF document input (--vision --vision-pdf): renders PDF pages via PDFium and libjpeg-turbo in a bounded worker.
const std = @import("std");
const builtin = @import("builtin");
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
pub const max_request_pixels: u64 = 32 << 20;
pub const default_max_pages: u32 = 16;
pub const render_timeout_ns: i96 = 10 * std.time.ns_per_s;
pub const max_child_heap_bytes: u64 = 1 << 30;
pub const off_message = "PDF inputs are off on this server; start the server with --vision --vision-pdf";

pub const Budget = struct { images_used: u32 = 0, pdf_pages_used: u32 = 0, encoded_bytes: usize = 0, pixels_used: u64 = 0 };
pub const IfsdkPause = extern struct { version: c_int = 1, NeedToPauseNow: *const fn (*IfsdkPause) callconv(.c) c_int, user: ?*anyopaque = null };

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
    FPDF_RenderPageBitmap_Start: ?*const fn (?*anyopaque, ?*anyopaque, c_int, c_int, c_int, c_int, c_int, c_int, *IfsdkPause) callconv(.c) c_int = null,
    FPDF_RenderPage_Continue: ?*const fn (?*anyopaque, *IfsdkPause) callconv(.c) c_int = null,
    FPDF_RenderPage_Close: ?*const fn (?*anyopaque) callconv(.c) void = null,
};

const JpegFns = struct {
    tj3Init: *const fn (c_int) callconv(.c) ?*anyopaque,
    tj3Destroy: *const fn (?*anyopaque) callconv(.c) void,
    tj3Set: *const fn (?*anyopaque, c_int, c_int) callconv(.c) c_int,
    tj3Compress8: *const fn (?*anyopaque, [*]const u8, c_int, c_int, c_int, c_int, *?[*]u8, *usize) callconv(.c) c_int,
    tj3Free: *const fn (?*anyopaque) callconv(.c) void,
};

const DlInfo = extern struct { fname: ?[*:0]const u8, fbase: ?*anyopaque, sname: ?[*:0]const u8, saddr: ?*anyopaque };
extern "c" fn dladdr(addr: *const anyopaque, info: *DlInfo) c_int;

const PdfiumLoaded = union(enum) { unknown, missing, ready: PdfiumFns };
const JpegLoaded = union(enum) { unknown, missing, ready: JpegFns };

var pdfium_state: PdfiumLoaded = .unknown;
var jpeg_state: JpegLoaded = .unknown;
var pdfium_mutex: std.Io.Mutex = .init;
var jpeg_mutex: std.Io.Mutex = .init;

fn loadDynLib(comptime T: type, comptime id: sys.optional_libraries.Id, a: Allocator) ?T {
    const exe = std.process.executableDirPathAlloc(log.io(), a) catch null;
    defer if (exe) |p| a.free(p);
    var dl = sys.optional_libraries.open(sys.optional_libraries.spec(id), a, sys, exe) orelse return null;
    var f: T = undefined;
    const info = @typeInfo(T).@"struct";
    inline for (info.field_names, info.field_types) |name, FT| {
        if (@typeInfo(FT) == .optional) {
            @field(f, name) = dl.lookup(@typeInfo(FT).optional.child, name);
        } else {
            @field(f, name) = dl.lookup(FT, name) orelse {
                dl.close();
                return null;
            };
        }
    }
    return f;
}

fn findPdfium(a: Allocator) PdfiumLoaded {
    const f = loadDynLib(PdfiumFns, .pdfium, a) orelse return .missing;
    if (sys.windows) f.FPDF_InitLibrary();
    return .{ .ready = f };
}

fn findJpeg(a: Allocator) JpegLoaded {
    return if (loadDynLib(JpegFns, .jpeg, a)) |f| .{ .ready = f } else .missing;
}

fn getJpeg(a: Allocator) ?JpegFns {
    const io = log.io();
    jpeg_mutex.lockUncancelable(io);
    defer jpeg_mutex.unlock(io);
    if (jpeg_state == .unknown) jpeg_state = findJpeg(a);
    return if (jpeg_state == .ready) jpeg_state.ready else null;
}

/// Logs the loaded PDFium path and SHA-256 at startup when --vision-pdf is enabled, or warns if a library is missing.
pub fn logStartup(a: Allocator, io: std.Io) void {
    if (getJpeg(a) == null) log.line("--vision-pdf: {s}", .{sys.optional_libraries.spec(.jpeg).message});
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

/// Renders all pages of `pdf_bytes` within the request-wide `budget` into `image_url` parts appended to `parts`.
pub fn expandBytes(cx: *Cx, pdf_bytes: []const u8, image_type: []const u8, max_images: u32, max_pages: u32, budget: *Budget, parts: *std.ArrayList(Value)) errors.Refused!void {
    const jf = getJpeg(cx.a) orelse return cx.fail(.unimplemented, "{s}", .{sys.optional_libraries.spec(.jpeg).message});
    const io = log.io();
    pdfium_mutex.lockUncancelable(io);
    defer pdfium_mutex.unlock(io);
    if (pdfium_state == .unknown) pdfium_state = findPdfium(cx.a);
    const f = if (pdfium_state == .ready) pdfium_state.ready else return cx.fail(.unimplemented, "{s}", .{sys.optional_libraries.spec(.pdfium).message});
    const deadline_ns = std.Io.Clock.awake.now(io).toNanoseconds() + render_timeout_ns;
    var jpegs: std.ArrayList([]const u8) = .empty;
    if (comptime sys.windows) try renderInProcess(cx, f, jf, pdf_bytes, max_images, max_pages, deadline_ns, budget, &jpegs) else try renderInChild(cx, f, jf, pdf_bytes, max_images, max_pages, deadline_ns, budget, &jpegs);
    for (jpegs.items) |jb| try appendImageUrlPart(cx.a, parts, image_type, try makeDataUrl(cx.a, "image/jpeg", jb));
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

fn renderPageWithDeadline(cx: *Cx, f: PdfiumFns, bmp: ?*anyopaque, page: ?*anyopaque, w: c_int, h: c_int, idx: usize, deadline_ns: i96) errors.Refused!void {
    const flags: c_int = 0x01 | 0x10; // FPDF_ANNOT | FPDF_PRINTING
    if (f.FPDF_RenderPageBitmap_Start != null and f.FPDF_RenderPage_Continue != null and f.FPDF_RenderPage_Close != null) {
        var state: PauseState = .{ .pause = .{ .version = 1, .NeedToPauseNow = PauseState.check }, .deadline_ns = deadline_ns };
        var status = f.FPDF_RenderPageBitmap_Start.?(bmp, page, 0, 0, w, h, 0, flags, &state.pause);
        while (status == 1) {
            if (state.timed_out or std.Io.Clock.awake.now(log.io()).toNanoseconds() >= deadline_ns) {
                f.FPDF_RenderPage_Close.?(page);
                return cx.refuse("PDF rendering timed out");
            }
            status = f.FPDF_RenderPage_Continue.?(page, &state.pause);
        }
        f.FPDF_RenderPage_Close.?(page);
        if (state.timed_out or std.Io.Clock.awake.now(log.io()).toNanoseconds() >= deadline_ns) return cx.refuse("PDF rendering timed out");
        if (status == 3) return cx.refuse(try std.fmt.allocPrint(cx.a, "PDF page {d} could not be rendered", .{idx + 1}));
        return;
    }
    f.FPDF_RenderPageBitmap(bmp, page, 0, 0, w, h, 0, flags);
    if (std.Io.Clock.awake.now(log.io()).toNanoseconds() >= deadline_ns) return cx.refuse("PDF rendering timed out");
}

fn renderInProcess(cx: *Cx, f: PdfiumFns, jf: JpegFns, pdf_bytes: []const u8, max_images: u32, max_pages: u32, deadline_ns: i96, budget: *Budget, out_jpegs: *std.ArrayList([]const u8)) errors.Refused!void {
    if (pdf_bytes.len > std.math.maxInt(c_int)) return cx.refuse("PDF exceeds the encoded byte limit");
    if (std.Io.Clock.awake.now(log.io()).toNanoseconds() >= deadline_ns) return cx.refuse("PDF rendering timed out");
    const doc = f.FPDF_LoadMemDocument(pdf_bytes.ptr, @intCast(pdf_bytes.len), null) orelse
        return cx.refuse("PDF document could not be opened (corrupt or password-protected)");
    defer f.FPDF_CloseDocument(doc);
    const total_pages = f.FPDF_GetPageCount(doc);
    if (total_pages <= 0) return cx.refuse("PDF document contains no pages");
    const pages: u32 = @intCast(total_pages);
    const limit: u32 = @max(@as(u32, 1), max_pages);
    if (budget.pdf_pages_used + pages > limit)
        return cx.refuse(try std.fmt.allocPrint(cx.a, "a PDF may have at most {d} pages (--vision-max-images, TENSORFOLD_MAX_PDF_PAGES); this request has {d}", .{ limit, budget.pdf_pages_used + pages }));
    if (max_images > 0 and budget.images_used + pages > max_images)
        return cx.refuse(try std.fmt.allocPrint(cx.a, "a request may hold at most {d} images (--vision-max-images); it has {d}", .{ max_images, budget.images_used + pages }));
    for (0..pages) |idx| {
        if (std.Io.Clock.awake.now(log.io()).toNanoseconds() >= deadline_ns) return cx.refuse("PDF rendering timed out");
        const page = f.FPDF_LoadPage(doc, @intCast(idx)) orelse
            return cx.refuse(try std.fmt.allocPrint(cx.a, "PDF page {d} could not be rendered", .{idx + 1}));
        defer f.FPDF_ClosePage(page);
        const dims = pageDimensions(f.FPDF_GetPageWidthF(page), f.FPDF_GetPageHeightF(page));
        const page_pixels = @as(u64, @intCast(dims[0])) * @as(u64, @intCast(dims[1]));
        if (budget.pixels_used + page_pixels > max_request_pixels) return cx.refuse("the request's images exceed the 32-megapixel limit");
        if (budget.encoded_bytes >= images.max_request_bytes) return cx.refuse("the request's images exceed 20 MB encoded");
        const bmp = f.FPDFBitmap_Create(dims[0], dims[1], 0) orelse return error.OutOfMemory;
        defer f.FPDFBitmap_Destroy(bmp);
        const stride = f.FPDFBitmap_GetStride(bmp);
        const buf_ptr = f.FPDFBitmap_GetBuffer(bmp) orelse return error.OutOfMemory;
        if (stride < dims[0] * 4) return cx.refuse("PDF page bitmap has invalid stride");
        const bgra: [*]u8 = @ptrCast(buf_ptr);
        const total_bytes = @as(usize, @intCast(stride)) * @as(usize, @intCast(dims[1]));
        @memset(bgra[0..total_bytes], 0xFF);
        try renderPageWithDeadline(cx, f, bmp, page, dims[0], dims[1], idx, deadline_ns);
        const jpeg_bytes = encodeJpegBgra(cx.a, jf, bgra[0..total_bytes], @intCast(dims[0]), @intCast(dims[1]), @intCast(stride)) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.EncodeFailed => return cx.refuse("rendered PDF page could not be encoded as JPEG"),
        };
        if (jpeg_bytes.len > images.max_image_bytes) return cx.refuse("rendered PDF page exceeds the 10 MB image byte limit");
        if (budget.encoded_bytes + jpeg_bytes.len > images.max_request_bytes) return cx.refuse("the request's images exceed 20 MB encoded");
        budget.pdf_pages_used += 1;
        budget.images_used += 1;
        budget.pixels_used += page_pixels;
        budget.encoded_bytes += jpeg_bytes.len;
        try out_jpegs.append(cx.a, jpeg_bytes);
    }
}

fn pageDimensions(pw_raw: f32, ph_raw: f32) [2]c_int {
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
    return .{ w, h };
}

const PosixChild = struct { extern "c" fn fork() c_int; extern "c" fn kill(c_int, c_int) c_int; extern "c" fn pipe(*[2]c_int) c_int; };

fn applyChildLimits() void {
    if (comptime sys.windows) return;
    std.posix.setrlimit(.CPU, .{ .cur = 10, .max = 10 }) catch {};
    std.posix.setrlimit(.CORE, .{ .cur = 0, .max = 0 }) catch {};
    std.posix.setrlimit(.FSIZE, .{ .cur = 0, .max = 0 }) catch {};
    if (comptime builtin.os.tag == .linux) {
        const fd = std.c.open("/proc/self/statm", .{}, @as(c_uint, 0));
        if (fd >= 0) {
            defer _ = std.c.close(fd);
            var buf: [64]u8 = undefined;
            const n = std.c.read(fd, &buf, buf.len);
            if (n > 0) {
                const text = buf[0..@intCast(n)];
                const end = std.mem.indexOfAny(u8, text, " \t\n") orelse text.len;
                if (std.fmt.parseInt(u64, text[0..end], 10) catch null) |pages| {
                    const cap = pages * std.heap.page_size_min + max_child_heap_bytes;
                    std.posix.setrlimit(.AS, .{ .cur = cap, .max = cap }) catch {};
                }
            }
        }
    }
}

fn closeRange(first: c_uint, last: c_uint) void {
    if (first > last) return;
    if (comptime builtin.os.tag == .linux) if (std.os.linux.syscall3(.close_range, first, last, 0) == 0) return;
    const limit = std.posix.getrlimit(.NOFILE) catch std.posix.rlimit{ .cur = 1024, .max = 1024 };
    const top: c_uint = @intCast(@min(@as(u64, last), @min(@as(u64, limit.cur), 65536)));
    var d = first;
    while (d <= top) : (d += 1) _ = std.c.close(@intCast(d));
}

fn closeDescriptorsExcept(keep_fd: c_int) void {
    if (comptime sys.windows) return;
    const ufd: c_uint = @intCast(keep_fd);
    if (ufd > 3) closeRange(3, ufd - 1);
    if (ufd < ~@as(c_uint, 0)) closeRange(@max(3, ufd + 1), ~@as(c_uint, 0));
}

fn writeAll(fd: c_int, bytes: []const u8) bool {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.c.write(fd, bytes[off..].ptr, bytes.len - off);
        if (n <= 0) return false;
        off += @intCast(n);
    }
    return true;
}

fn writeInt(comptime T: type, fd: c_int, val: T) bool {
    var b: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &b, val, .little);
    return writeAll(fd, &b);
}

fn runChildWorker(fd: c_int, f: PdfiumFns, jf: JpegFns, pdf_bytes: []const u8, max_images: u32, max_pages: u32, deadline_ns: i96, budget: Budget) noreturn {
    applyChildLimits();
    closeDescriptorsExcept(fd);
    f.FPDF_InitLibrary();
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    var cx: Cx = .{ .a = arena.allocator() };
    var b = budget;
    var jpegs: std.ArrayList([]const u8) = .empty;
    if (renderInProcess(&cx, f, jf, pdf_bytes, max_images, max_pages, deadline_ns, &b, &jpegs)) |_| {
        _ = writeAll(fd, &.{0});
        _ = writeInt(u32, fd, @intCast(jpegs.items.len));
        for (jpegs.items) |jb| {
            _ = writeInt(u32, fd, @intCast(jb.len));
            _ = writeAll(fd, jb);
        }
    } else |err| switch (err) {
        error.OutOfMemory => _ = writeAll(fd, &.{2}),
        error.Refused => {
            _ = writeAll(fd, &.{ 1, @intFromEnum(cx.kind) });
            _ = writeInt(u32, fd, @intCast(cx.message.len));
            _ = writeAll(fd, cx.message);
        },
    }
    _ = std.c.close(fd);
    std.c._exit(0);
}

pub fn jpegPixels(bytes: []const u8) ?u64 {
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
            const h: u64 = std.mem.readInt(u16, bytes[i + 3 ..][0..2], .big);
            const w: u64 = std.mem.readInt(u16, bytes[i + 5 ..][0..2], .big);
            return if (w > 0 and h > 0) w * h else null;
        }
        i += seg_len;
    }
    return null;
}

fn readChildPipe(a: Allocator, fd: c_int, deadline_ns: i96, max_bytes: usize) error{ OutOfMemory, Timeout, Overflow }![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        const now = std.Io.Clock.awake.now(log.io()).toNanoseconds();
        if (now >= deadline_ns) return error.Timeout;
        const rem_ms: i32 = @intCast(@max(@as(i96, 1), @min(@as(i96, 10_000), @divTrunc(deadline_ns - now + 999_999, 1_000_000))));
        var pfd = [1]sys.posix.pollfd{.{ .fd = fd, .events = sys.posix.POLL.IN, .revents = 0 }};
        if ((sys.posix.poll(&pfd, rem_ms) catch return error.Timeout) == 0) return error.Timeout;
        const n = std.c.read(fd, &buf, buf.len);
        if (n == 0) break;
        if (n < 0) return error.Timeout;
        const count: usize = @intCast(n);
        if (out.items.len + count > max_bytes) return error.Overflow;
        try out.appendSlice(a, buf[0..count]);
    }
    return out.items;
}

fn parseChildReply(cx: *Cx, frame: []const u8, max_images: u32, max_pages: u32, budget: *Budget, out_jpegs: *std.ArrayList([]const u8)) errors.Refused!void {
    if (frame.len == 0) return cx.refuse("PDF rendering exceeded resource limits or failed");
    switch (frame[0]) {
        0 => {
            if (frame.len < 5) return cx.refuse("PDF rendering exceeded resource limits or failed");
            const count = std.mem.readInt(u32, frame[1..5], .little);
            if (count == 0) return cx.refuse("PDF document contains no pages");
            const limit: u32 = @max(@as(u32, 1), max_pages);
            if (budget.pdf_pages_used +| count > limit)
                return cx.refuse(try std.fmt.allocPrint(cx.a, "a PDF may have at most {d} pages (--vision-max-images, TENSORFOLD_MAX_PDF_PAGES); this request has {d}", .{ limit, budget.pdf_pages_used +| count }));
            if (max_images > 0 and budget.images_used +| count > max_images)
                return cx.refuse(try std.fmt.allocPrint(cx.a, "a request may hold at most {d} images (--vision-max-images); it has {d}", .{ max_images, budget.images_used +| count }));
            var pos: usize = 5;
            for (0..count) |_| {
                if (pos + 4 > frame.len) return cx.refuse("PDF rendering exceeded resource limits or failed");
                const len: usize = std.mem.readInt(u32, frame[pos..][0..4], .little);
                pos += 4;
                if (len == 0 or pos + len > frame.len) return cx.refuse("PDF rendering exceeded resource limits or failed");
                if (len > images.max_image_bytes) return cx.refuse("rendered PDF page exceeds the 10 MB image byte limit");
                if (budget.encoded_bytes + len > images.max_request_bytes) return cx.refuse("the request's images exceed 20 MB encoded");
                const jb = frame[pos .. pos + len];
                const px = jpegPixels(jb) orelse return cx.refuse("rendered PDF page could not be encoded as JPEG");
                if (budget.pixels_used + px > max_request_pixels) return cx.refuse("the request's images exceed the 32-megapixel limit");
                budget.pdf_pages_used += 1;
                budget.images_used += 1;
                budget.encoded_bytes += len;
                budget.pixels_used += px;
                try out_jpegs.append(cx.a, jb);
                pos += len;
            }
            if (pos != frame.len) return cx.refuse("PDF rendering exceeded resource limits or failed");
        },
        1 => {
            if (frame.len < 6) return cx.refuse("PDF rendering failed");
            cx.kind = std.enums.fromInt(errors.Kind, frame[1]) orelse .request;
            const len: usize = std.mem.readInt(u32, frame[2..6], .little);
            if (6 + len != frame.len) return cx.refuse("PDF rendering failed");
            cx.message = frame[6 .. 6 + len];
            return error.Refused;
        },
        2 => return error.OutOfMemory,
        else => return cx.refuse("PDF rendering exceeded resource limits or failed"),
    }
}

fn renderInChild(cx: *Cx, f: PdfiumFns, jf: JpegFns, pdf_bytes: []const u8, max_images: u32, max_pages: u32, deadline_ns: i96, budget: *Budget, out_jpegs: *std.ArrayList([]const u8)) errors.Refused!void {
    var fds: [2]c_int = undefined;
    if (PosixChild.pipe(&fds) != 0) return cx.refuse("PDF rendering worker pipe could not be opened");
    const pid = PosixChild.fork();
    if (pid < 0) {
        _ = std.c.close(fds[0]);
        _ = std.c.close(fds[1]);
        return cx.refuse("PDF rendering worker process could not be started");
    }
    if (pid == 0) {
        _ = std.c.close(fds[0]);
        runChildWorker(fds[1], f, jf, pdf_bytes, max_images, max_pages, deadline_ns, budget.*);
    }
    _ = std.c.close(fds[1]);
    var status: c_int = 0;
    var reaped = false;
    defer {
        _ = std.c.close(fds[0]);
        if (!reaped) {
            _ = PosixChild.kill(pid, 9);
            _ = std.c.waitpid(pid, &status, 0);
        }
    }
    const frame = readChildPipe(cx.a, fds[0], deadline_ns, images.max_request_bytes + (64 << 10)) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Timeout => return cx.refuse("PDF rendering timed out"),
        error.Overflow => return cx.refuse("the request's images exceed 20 MB encoded"),
    };
    _ = std.c.waitpid(pid, &status, 0);
    reaped = true;
    return parseChildReply(cx, frame, max_images, max_pages, budget, out_jpegs);
}

fn encodeJpegBgra(a: Allocator, jf: JpegFns, bgra: []const u8, width: usize, height: usize, stride: usize) error{ OutOfMemory, EncodeFailed }![]const u8 {
    const handle = jf.tj3Init(0) orelse return error.EncodeFailed; // 0 = TJINIT_COMPRESS
    defer jf.tj3Destroy(handle);
    _ = jf.tj3Set(handle, 3, 85); // TJPARAM_QUALITY = 3
    _ = jf.tj3Set(handle, 4, 2); // TJPARAM_SUBSAMP = 4, TJSAMP_420 = 2
    var out_buf: ?[*]u8 = null;
    var out_size: usize = 0;
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
    var jpegs: std.ArrayList([]const u8) = .empty;
    try std.testing.expectError(error.Refused, parseChildReply(&bad_reply_cx, &.{ 1, 0xFF, 3, 0, 0, 0, 'b', 'a', 'd' }, 4, 4, &budget, &jpegs));
    try std.testing.expectEqual(errors.Kind.request, bad_reply_cx.kind);
    try std.testing.expectEqualStrings("bad", bad_reply_cx.message);
}

test "libpdfium renders in a child worker and enforces request-wide page, byte and pixel caps" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    var parts: std.ArrayList(Value) = .empty;
    var budget: Budget = .{};
    expandBytes(&cx, minimal_two_page_pdf, "image_url", 4, 4, &budget, &parts) catch |err| {
        if (err == error.Refused and (pdfium_state == .missing or jpeg_state == .missing)) {
            const want = if (jpeg_state == .missing) sys.optional_libraries.spec(.jpeg).names[0] else sys.optional_libraries.spec(.pdfium).names[0];
            try std.testing.expect(std.mem.indexOf(u8, cx.message, want) != null);
            try std.testing.expectEqual(errors.Kind.unimplemented, cx.kind);
            return;
        }
        return err;
    };
    try std.testing.expectEqual(@as(usize, 2), parts.items.len);
    try std.testing.expectEqual(@as(u32, 2), budget.pdf_pages_used);
    try std.testing.expectEqual(@as(u32, 2), budget.images_used);
    try std.testing.expect(budget.encoded_bytes > 0 and budget.pixels_used == 2 * 400 * 400);
    for (parts.items) |p| {
        const img_bytes = try images.dataBytes(&cx, p.get("image_url").?.get("url").?.string);
        try std.testing.expectEqual(images.Format.jpeg, images.sniff(img_bytes));
        try std.testing.expectEqual(@as(?u64, 400 * 400), jpegPixels(img_bytes));
    }
    var second_cx: Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, expandBytes(&second_cx, minimal_two_page_pdf, "image_url", 4, 3, &budget, &parts));
    try std.testing.expect(std.mem.indexOf(u8, second_cx.message, "at most 3 pages") != null);
}
