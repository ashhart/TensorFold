//! PDF document input (--vision): renders PDF pages via libpdfium (or extracts embedded images) into image_url parts.
const std = @import("std");
const builtin = @import("builtin");
const sys = @import("sys");
const json = @import("json");
const errors = @import("errors.zig");
const images = @import("images.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const env = "TENSORFOLD_LIBPDFIUM";
pub const max_pdf_bytes: usize = 32 << 20;
pub const default_max_pages: u32 = 16;

const pdfium_lib = if (builtin.os.tag == .macos) "libpdfium.dylib" else if (builtin.os.tag == .windows) "pdfium.dll" else "libpdfium.so";
const jpeg_lib = if (builtin.os.tag == .macos) "libturbojpeg.0.dylib" else if (builtin.os.tag == .windows) "turbojpeg.dll" else "libturbojpeg.so.0";

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
var pdfium_mutex: std.atomic.Mutex = .unlocked;
var jpeg_mutex: std.atomic.Mutex = .unlocked;

fn lock(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) std.atomic.spinLoopHint();
}

fn openPdfium(path: []const u8) ?PdfiumFns {
    var dl = sys.DynLib.open(path) catch return null;
    var f: PdfiumFns = undefined;
    const info = @typeInfo(PdfiumFns).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(f, name) = dl.lookup(T, name) orelse {
            dl.close();
            return null;
        };
    }
    f.FPDF_InitLibrary();
    return f;
}

fn findPdfium(a: Allocator) PdfiumLoaded {
    if (sys.getenv(env)) |named| {
        if (named.len == 0) return .missing;
        if (openPdfium(named)) |f| return .{ .ready = f };
        return .missing;
    }
    if (comptime builtin.os.tag == .linux) {
        var exe_buf: [1024]u8 = undefined;
        if (sys.optional_libraries.video_library.executableDir(&exe_buf)) |dir| {
            for ([_][]const []const u8{ &.{ dir, "..", "lib", pdfium_lib }, &.{ dir, pdfium_lib } }) |parts| {
                const p = std.fs.path.join(a, parts) catch continue;
                defer a.free(p);
                if (openPdfium(p)) |f| return .{ .ready = f };
            }
        }
    }
    const candidates = [_][]const u8{
        "/opt/tensorfold/lib/" ++ pdfium_lib,
        "/opt/tensorfold/native/lib/" ++ pdfium_lib,
        "/opt/codecs/pdfium/lib/" ++ pdfium_lib,
        "/usr/local/lib/" ++ pdfium_lib,
        pdfium_lib,
    };
    for (candidates) |p| {
        if (openPdfium(p)) |f| return .{ .ready = f };
    }
    return .missing;
}

fn openJpeg(path: []const u8) ?JpegFns {
    var dl = sys.DynLib.open(path) catch return null;
    var f: JpegFns = undefined;
    const info = @typeInfo(JpegFns).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(f, name) = dl.lookup(T, name) orelse {
            dl.close();
            return null;
        };
    }
    return f;
}

fn findJpeg(a: Allocator) JpegLoaded {
    if (sys.getenv("TENSORFOLD_LIBTURBOJPEG")) |named| {
        if (named.len > 0) if (openJpeg(named)) |f| return .{ .ready = f };
    }
    if (comptime builtin.os.tag == .linux) {
        var exe_buf: [1024]u8 = undefined;
        if (sys.optional_libraries.video_library.executableDir(&exe_buf)) |dir| {
            for ([_][]const []const u8{ &.{ dir, "..", "lib", jpeg_lib }, &.{ dir, jpeg_lib } }) |parts| {
                const p = std.fs.path.join(a, parts) catch continue;
                defer a.free(p);
                if (openJpeg(p)) |f| return .{ .ready = f };
            }
        }
    }
    const candidates = if (builtin.os.tag == .macos) [_][]const u8{
        "/opt/homebrew/opt/jpeg-turbo/lib/" ++ jpeg_lib,
        "/usr/local/opt/jpeg-turbo/lib/" ++ jpeg_lib,
        jpeg_lib,
    } else [_][]const u8{
        "/opt/tensorfold/lib/" ++ jpeg_lib,
        "/opt/codecs/jpeg/lib/" ++ jpeg_lib,
        jpeg_lib,
    };
    for (candidates) |p| {
        if (openJpeg(p)) |f| return .{ .ready = f };
    }
    return .missing;
}

fn getJpeg(a: Allocator) ?JpegFns {
    lock(&jpeg_mutex);
    defer jpeg_mutex.unlock();
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
    try expandBytes(cx, pdf_bytes, image_type, maxPagesLimit(max_images), parts);
}

fn extractPartBytes(cx: *Cx, part: Value) errors.Refused![]const u8 {
    if (part.get("image_url")) |iu| {
        const url = if (iu == .string) iu.string else if (iu == .object) (iu.strField("url") orelse return cx.refuse("image_url must contain a url string")) else return cx.refuse("image_url must be a url string or an object with one");
        return decodeFileData(cx, url);
    }
    const container: Value = if (part.get("file")) |f| (if (f == .object) f else return cx.refuse("file content part must contain a file object")) else part;
    if (container.get("file_id") != null and container.get("file_data") == null and container.get("data") == null and container.get("url") == null and container.get("file_url") == null)
        return cx.refuse("file parts require inline file_data (base64 or data:application/pdf;base64,...); file_id is not supported");
    const raw = container.strField("file_data") orelse container.strField("data") orelse container.strField("url") orelse container.strField("file_url") orelse
        return cx.refuse("file content part must contain file_data");
    if (raw.len == 0) return cx.refuse("file_data must be a non-empty base64 string or data URL");
    return decodeFileData(cx, raw);
}

pub fn decodeFileData(cx: *Cx, raw: []const u8) errors.Refused![]const u8 {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
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

/// Renders up to `max_pages` pages of `pdf_bytes` into `image_url` content parts appended to `parts`.
pub fn expandBytes(cx: *Cx, pdf_bytes: []const u8, image_type: []const u8, max_pages: u32, parts: *std.ArrayList(Value)) errors.Refused!void {
    lock(&pdfium_mutex);
    defer pdfium_mutex.unlock();
    if (pdfium_state == .unknown) pdfium_state = findPdfium(cx.a);
    switch (pdfium_state) {
        .ready => |f| return renderWithPdfium(cx, f, pdf_bytes, image_type, max_pages, parts),
        .missing, .unknown => {
            if (try extractEmbeddedImages(cx, pdf_bytes, image_type, max_pages, parts)) return;
            return cx.refuse("PDF input requires libpdfium; install pypdfium2 and set TENSORFOLD_LIBPDFIUM to libpdfium.so");
        },
    }
}

fn renderWithPdfium(cx: *Cx, f: PdfiumFns, pdf_bytes: []const u8, image_type: []const u8, max_pages: u32, parts: *std.ArrayList(Value)) errors.Refused!void {
    if (pdf_bytes.len > std.math.maxInt(c_int)) return cx.refuse("PDF exceeds the encoded byte limit");
    const doc = f.FPDF_LoadMemDocument(pdf_bytes.ptr, @intCast(pdf_bytes.len), null) orelse {
        if (try extractEmbeddedImages(cx, pdf_bytes, image_type, max_pages, parts)) return;
        return cx.refuse("PDF document could not be opened (corrupt or password-protected)");
    };
    defer f.FPDF_CloseDocument(doc);
    const total_pages = f.FPDF_GetPageCount(doc);
    if (total_pages <= 0) {
        if (try extractEmbeddedImages(cx, pdf_bytes, image_type, max_pages, parts)) return;
        return cx.refuse("PDF document contains no pages");
    }
    const pages_to_render: u32 = @min(@as(u32, @intCast(total_pages)), @max(@as(u32, 1), max_pages));
    var rendered_any = false;
    for (0..pages_to_render) |idx| {
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
        if ((pw * scale) * (ph * scale) > max_area) {
            scale = @sqrt(max_area / (pw * ph));
        }
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
        const data_url = try encodeBgraDataUrl(cx.a, bgra[0..total_bytes], @intCast(w), @intCast(h), @intCast(stride));
        try appendImageUrlPart(cx.a, parts, image_type, data_url);
        rendered_any = true;
    }
    if (!rendered_any) return cx.refuse("PDF document pages could not be rendered");
}

fn encodeBgraDataUrl(a: Allocator, bgra: []const u8, width: usize, height: usize, stride: usize) Allocator.Error![]const u8 {
    if (getJpeg(a)) |jf| {
        if (encodeJpegBgra(a, jf, bgra, width, height, stride)) |jpeg_bytes| {
            return makeDataUrl(a, "image/jpeg", jpeg_bytes);
        } else |_| {}
    }
    const png_bytes = try encodePngBgra(a, bgra, width, height, stride);
    return makeDataUrl(a, "image/png", png_bytes);
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

/// Pure-Zig uncompressed RGB PNG encoder (fallback when libturbojpeg is not loaded).
pub fn encodePngBgra(a: Allocator, bgra: []const u8, width: usize, height: usize, stride: usize) Allocator.Error![]const u8 {
    const row_raw = 1 + width * 3;
    const raw_len = height * row_raw;
    const num_blocks = if (raw_len == 0) 1 else (raw_len + 65534) / 65535;
    const zlib_len = 2 + num_blocks * 5 + raw_len + 4;
    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(a, 8 + 25 + 12 + zlib_len + 12);
    try out.appendSlice(a, "\x89PNG\r\n\x1a\n");

    // IHDR
    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], @intCast(width), .big);
    std.mem.writeInt(u32, ihdr[4..8], @intCast(height), .big);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // color type: RGB
    ihdr[10] = 0; // compression
    ihdr[11] = 0; // filter
    ihdr[12] = 0; // interlace
    try appendPngChunk(a, &out, "IHDR", &ihdr);

    // Build raw scanlines (filter 0 + RGB)
    const raw = try a.alloc(u8, raw_len);
    defer a.free(raw);
    var s1: u32 = 1;
    var s2: u32 = 0;
    for (0..height) |y| {
        const src_row = bgra[y * stride ..];
        const dst_row = raw[y * row_raw ..][0..row_raw];
        dst_row[0] = 0;
        for (0..width) |x| {
            dst_row[1 + x * 3 + 0] = src_row[x * 4 + 2]; // R
            dst_row[1 + x * 3 + 1] = src_row[x * 4 + 1]; // G
            dst_row[1 + x * 3 + 2] = src_row[x * 4 + 0]; // B
        }
        for (dst_row) |b| {
            s1 = (s1 + b) % 65521;
            s2 = (s2 + s1) % 65521;
        }
    }

    // Wrap raw scanlines in an uncompressed zlib stream
    var zbuf: std.ArrayList(u8) = .empty;
    defer zbuf.deinit(a);
    try zbuf.ensureTotalCapacity(a, zlib_len);
    try zbuf.appendSlice(a, &.{ 0x78, 0x01 });
    var offset: usize = 0;
    while (true) {
        const rem = raw.len - offset;
        const chunk_len: u16 = @intCast(@min(rem, 65535));
        const final_block: u8 = if (offset + chunk_len >= raw.len) 1 else 0;
        try zbuf.append(a, final_block);
        var len_bytes: [4]u8 = undefined;
        std.mem.writeInt(u16, len_bytes[0..2], chunk_len, .little);
        std.mem.writeInt(u16, len_bytes[2..4], ~chunk_len, .little);
        try zbuf.appendSlice(a, &len_bytes);
        try zbuf.appendSlice(a, raw[offset .. offset + chunk_len]);
        offset += chunk_len;
        if (final_block == 1) break;
    }
    var adler_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &adler_bytes, (s2 << 16) | s1, .big);
    try zbuf.appendSlice(a, &adler_bytes);
    try appendPngChunk(a, &out, "IDAT", zbuf.items);
    try appendPngChunk(a, &out, "IEND", &.{});
    return out.items;
}

fn appendPngChunk(a: Allocator, out: *std.ArrayList(u8), kind: *const [4]u8, body: []const u8) Allocator.Error!void {
    var len_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &len_bytes, @intCast(body.len), .big);
    try out.appendSlice(a, &len_bytes);
    const start = out.items.len;
    try out.appendSlice(a, kind);
    try out.appendSlice(a, body);
    const crc = std.hash.Crc32.hash(out.items[start..]);
    var crc_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &crc_bytes, crc, .big);
    try out.appendSlice(a, &crc_bytes);
}

fn makeDataUrl(a: Allocator, mime: []const u8, bytes: []const u8) Allocator.Error![]const u8 {
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

/// Fallback when libpdfium is unavailable: extracts raw JPEG or PNG streams embedded in the PDF.
fn extractEmbeddedImages(cx: *Cx, pdf_bytes: []const u8, image_type: []const u8, max_pages: u32, parts: *std.ArrayList(Value)) errors.Refused!bool {
    var count: u32 = 0;
    const limit = @max(@as(u32, 1), max_pages);
    var pos: usize = 0;
    while (pos < pdf_bytes.len and count < limit) {
        if (std.mem.indexOfPos(u8, pdf_bytes, pos, "stream")) |s_pos| {
            var data_start = s_pos + 6;
            if (data_start < pdf_bytes.len and pdf_bytes[data_start] == '\r') data_start += 1;
            if (data_start < pdf_bytes.len and pdf_bytes[data_start] == '\n') {
                data_start += 1;
                if (std.mem.indexOfPos(u8, pdf_bytes, data_start, "endstream")) |e_pos| {
                    var data_end = e_pos;
                    while (data_end > data_start and (pdf_bytes[data_end - 1] == '\n' or pdf_bytes[data_end - 1] == '\r' or pdf_bytes[data_end - 1] == ' ')) {
                        data_end -= 1;
                    }
                    const stream = pdf_bytes[data_start..data_end];
                    if (stream.len > 4 and images.sniff(stream) == .jpeg and stream[stream.len - 2] == 0xFF and stream[stream.len - 1] == 0xD9) {
                        const url = try makeDataUrl(cx.a, "image/jpeg", stream);
                        try appendImageUrlPart(cx.a, parts, image_type, url);
                        count += 1;
                    } else if (stream.len > 16 and images.sniff(stream) == .png) {
                        const url = try makeDataUrl(cx.a, "image/png", stream);
                        try appendImageUrlPart(cx.a, parts, image_type, url);
                        count += 1;
                    }
                    pos = e_pos + 9;
                    continue;
                }
            }
            pos = s_pos + 6;
        } else break;
    }
    return count > 0;
}

test "pure-Zig PNG encoder produces valid PNG bytes and embedded-image PDF fallback expands to image_url parts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };

    // 2x2 BGRA image -> PNG
    const bgra = [_]u8{
        255, 0,   0,   255, 0,   255, 0,   255,
        0,   0,   255, 255, 255, 255, 255, 255,
    };
    const png = try encodePngBgra(a, &bgra, 2, 2, 8);
    try std.testing.expectEqual(images.Format.png, images.sniff(png));

    // Build a synthetic PDF wrapping that PNG stream
    var pdf_doc: std.ArrayList(u8) = .empty;
    try pdf_doc.appendSlice(a, "%PDF-1.4\n1 0 obj\n<< /Length ");
    try pdf_doc.appendSlice(a, try std.fmt.allocPrint(a, "{d}", .{png.len}));
    try pdf_doc.appendSlice(a, " >>\nstream\n");
    try pdf_doc.appendSlice(a, png);
    try pdf_doc.appendSlice(a, "\nendstream\nendobj\n%%EOF\n");

    const data_url = try makeDataUrl(a, "application/pdf", pdf_doc.items);
    const decoded = try decodeFileData(&cx, data_url);
    try std.testing.expectEqualSlices(u8, pdf_doc.items, decoded);

    var parts: std.ArrayList(Value) = .empty;
    try std.testing.expect(try extractEmbeddedImages(&cx, decoded, "image_url", 4, &parts));
    try std.testing.expectEqual(@as(usize, 1), parts.items.len);
    const url = parts.items[0].get("image_url").?.get("url").?.string;
    const img_bytes = try images.dataBytes(&cx, url);
    try std.testing.expectEqual(images.Format.png, images.sniff(img_bytes));
}

test "libpdfium renders a multi-page PDF document when available" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };

    const minimal_two_page_pdf =
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

    var parts: std.ArrayList(Value) = .empty;
    expandBytes(&cx, minimal_two_page_pdf, "image_url", 4, &parts) catch |err| {
        if (err == error.Refused and pdfium_state == .missing) return;
        return err;
    };
    try std.testing.expectEqual(@as(usize, 2), parts.items.len);
    for (parts.items) |p| {
        const url = p.get("image_url").?.get("url").?.string;
        const img_bytes = try images.dataBytes(&cx, url);
        const fmt = images.sniff(img_bytes);
        try std.testing.expect(fmt == .jpeg or fmt == .png);
    }
}
