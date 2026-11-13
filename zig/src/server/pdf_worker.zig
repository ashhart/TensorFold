//! The PDF render worker: the server's own executable, spawned fresh per document, renders under limits over pipes.
const std = @import("std");
const builtin = @import("builtin");
const sys = @import("sys");
const errors = @import("errors.zig");
const log = @import("log.zig");
const pdf = @import("pdf.zig");
const jpeg_out = @import("jpeg_out.zig");
const Cx = errors.Cx;
const Budget = pdf.Budget;
const Page = pdf.Page;
const Allocator = std.mem.Allocator;

/// The hidden argument that makes tensorfold-native run only the PDF render worker (main.zig).
pub const arg = "--pdf-render-worker";
/// The binary started as the worker when set: tests name the server binary, since a test runner cannot be the worker.
pub var server_exe: ?[]const u8 = null;
const failed = "PDF rendering exceeded resource limits or failed";
/// Reply: 0, count, total, per page index, width, height, dpi, length (u32s) and JPEG; 1, kind, length, message; 2.
const page_head = 5 * 4; // a page's five u32s
/// The most page indices a selection list may send the worker.
const max_list: u32 = 1 << 16;

/// The parent's request on the worker's stdin, ahead of a list selection's indices and the PDF bytes; same build.
const Request = extern struct {
    remaining_ns: u64,
    encoded_bytes: u64,
    pixels_used: u64,
    pdf_len: u64,
    request_pixels: u64,
    request_bytes: u64,
    page_bytes: u64,
    max_images: u32,
    images_used: u32,
    pdf_pages_used: u32,
    sel_kind: u32, // pdf.Selection's tag
    sel_n: u32, // all_within's or first's count, or the list's length
    infra: u32, // caps.infra
    measure: u32, // 1: job.measure, sizes without bitmaps or JPEGs
};

/// This executable and the hidden argument; Linux's /proc/self/exe still names the running file once it is replaced.
pub fn argv(a: Allocator) ![]const []const u8 {
    const exe = server_exe orelse if (comptime builtin.os.tag == .linux) "/proc/self/exe" else try std.process.executablePathAlloc(log.io(), a);
    return a.dupe([]const u8, &.{ exe, arg });
}

/// Renders `job` in a fresh worker started from `cmd`, killed at its deadline; returns the document's page count.
pub fn render(cx: *Cx, cmd: []const []const u8, job: pdf.Job, budget: *Budget, out: *std.ArrayList(Page)) errors.Refused!u32 {
    const infra = job.caps.infra;
    const list: []const u32 = if (job.sel == .list) job.sel.list else &.{};
    if (list.len > max_list) return cx.fail(.request, "a PDF page selection may name at most {d} pages", .{max_list});
    const io = log.io();
    var child = std.process.spawn(io, .{ .argv = cmd, .stdin = .pipe, .stdout = .pipe }) catch return cx.fail(infra, "PDF rendering worker process could not be started", .{});
    defer if (child.id) |pid| {
        std.posix.kill(pid, .KILL) catch {};
        _ = child.wait(io) catch {};
    };
    const left = job.deadline_ns - std.Io.Clock.awake.now(io).toNanoseconds();
    const n: u32 = switch (job.sel) {
        .all_within, .first => |k| k,
        .list => |l| @intCast(l.len),
    };
    const req: Request = .{
        .remaining_ns = @intCast(@max(0, left)),
        .encoded_bytes = budget.encoded_bytes,
        .pixels_used = budget.pixels_used,
        .pdf_len = job.pdf_bytes.len,
        .request_pixels = job.caps.request_pixels,
        .request_bytes = job.caps.request_bytes,
        .page_bytes = job.caps.page_bytes,
        .max_images = job.caps.max_images,
        .images_used = budget.images_used,
        .pdf_pages_used = budget.pdf_pages_used,
        .sel_kind = @backingInt(std.meta.activeTag(job.sel)),
        .sel_n = n,
        .infra = @backingInt(infra),
        .measure = @intFromBool(job.measure),
    };
    const in = child.stdin.?.handle;
    const nonblock: u32 = @bitCast(std.c.O{ .NONBLOCK = true });
    _ = std.c.fcntl(in, std.c.F.SETFL, std.c.fcntl(in, std.c.F.GETFL) | @as(c_int, @bitCast(nonblock)));
    _ = sendAll(in, std.mem.asBytes(&req), job.deadline_ns) and sendAll(in, std.mem.sliceAsBytes(list), job.deadline_ns) and sendAll(in, job.pdf_bytes, job.deadline_ns);
    child.stdin.?.close(io);
    child.stdin = null;
    const frame = readChildPipe(cx.a, child.stdout.?.handle, job.deadline_ns, job.caps.request_bytes +| (1 << 20)) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Timeout => return cx.fail(infra, "PDF rendering timed out", .{}),
        // chat keeps its byte-limit message; under OCR an oversized reply is a renderer failure
        error.Overflow => return if (infra == .request) cx.fail(infra, "the request's images exceed {d} MB encoded", .{job.caps.request_bytes >> 20}) else cx.fail(infra, failed, .{}),
    };
    const term = child.wait(io) catch return cx.fail(infra, failed, .{});
    return reply(cx, term, frame, job.sel, job.measure, job.caps, budget, out);
}

/// A worker the kernel stopped at a limit keeps the limit message; any other signal is named as a crash.
fn reply(cx: *Cx, term: std.process.Child.Term, frame: []const u8, sel: pdf.Selection, measure: bool, caps: pdf.Caps, budget: *Budget, out: *std.ArrayList(Page)) errors.Refused!u32 {
    if (term == .signal) switch (term.signal) {
        .KILL, .XCPU, .XFSZ => return cx.fail(caps.infra, failed, .{}),
        else => |sig| if (std.enums.tagName(std.posix.SIG, sig)) |name|
            return cx.fail(caps.infra, "the PDF render worker crashed (SIG{s})", .{name})
        else
            return cx.fail(caps.infra, "the PDF render worker crashed (signal {d})", .{@backingInt(sig)}),
    };
    return parseChildReply(cx, frame, sel, measure, caps, budget, out);
}

fn ready(fd: c_int, events: i16, deadline_ns: i96) bool {
    const now = std.Io.Clock.awake.now(log.io()).toNanoseconds();
    if (now >= deadline_ns) return false;
    const rem_ms: i32 = @intCast(@max(@as(i96, 1), @min(@as(i96, 10_000), @divTrunc(deadline_ns - now + 999_999, 1_000_000))));
    var pfd = [1]sys.posix.pollfd{.{ .fd = fd, .events = events, .revents = 0 }};
    return (sys.posix.poll(&pfd, rem_ms) catch return false) > 0;
}

fn sendAll(fd: c_int, bytes: []const u8, deadline_ns: i96) bool {
    var off: usize = 0;
    while (off < bytes.len) {
        if (!ready(fd, sys.posix.POLL.OUT, deadline_ns)) return false;
        const n = std.c.write(fd, bytes[off..].ptr, bytes.len - off);
        if (n < 0 and std.posix.errno(n) == .AGAIN) continue;
        if (n <= 0) return false;
        off += @intCast(n);
    }
    return true;
}

fn readChildPipe(a: Allocator, fd: c_int, deadline_ns: i96, max_bytes: usize) error{ OutOfMemory, Timeout, Overflow }![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        if (!ready(fd, sys.posix.POLL.IN, deadline_ns)) return error.Timeout;
        const n = std.c.read(fd, &buf, buf.len);
        if (n == 0) break;
        if (n < 0) return error.Timeout;
        if (out.items.len + @as(usize, @intCast(n)) > max_bytes) return error.Overflow;
        try out.appendSlice(a, buf[0..@intCast(n)]);
    }
    return out.items;
}

fn readU32(bytes: []const u8, at: usize) u32 {
    return std.mem.readInt(u32, bytes[at..][0..4], .little);
}

/// Checks the frame against `sel`, `caps` and the running `budget`, appends its pages; a measure's pages carry no JPEG.
pub fn parseChildReply(cx: *Cx, frame: []const u8, sel: pdf.Selection, measure: bool, caps: pdf.Caps, budget: *Budget, out: *std.ArrayList(Page)) errors.Refused!u32 {
    if (frame.len == 0) return cx.fail(caps.infra, failed, .{});
    switch (frame[0]) {
        0 => {
            if (frame.len < 9) return cx.fail(caps.infra, failed, .{});
            const count = readU32(frame, 1);
            const total = readU32(frame, 5);
            if (count > total or count != try pdf.admitCount(cx, sel, caps, budget, total)) return cx.fail(caps.infra, failed, .{});
            var pos: usize = 9;
            for (0..count) |k| {
                if (frame.len - pos < page_head) return cx.fail(caps.infra, failed, .{});
                const index, const width, const height, const dpi, const len = .{ readU32(frame, pos), readU32(frame, pos + 4), readU32(frame, pos + 8), readU32(frame, pos + 12), readU32(frame, pos + 16) };
                pos += page_head;
                if (index != pdf.pageAt(sel, @intCast(k)) or dpi == 0 or dpi > pdf.max_dpi or (len == 0) != measure or frame.len - pos < len)
                    return cx.fail(caps.infra, failed, .{});
                const jb = frame[pos .. pos + len];
                if (!measure) {
                    const size = pdf.jpegSize(jb) orelse return cx.fail(caps.infra, "rendered PDF page could not be encoded as JPEG", .{});
                    if (size[0] != width or size[1] != height) return cx.fail(caps.infra, failed, .{});
                }
                if (@min(width, height) < pdf.min_side or @max(width, height) > pdf.max_side) return cx.fail(caps.infra, failed, .{});
                const px = @as(u64, width) * height;
                try pdf.checkPage(cx, caps, budget, px, len);
                budget.add(px, len);
                try out.append(cx.a, .{ .index = index, .jpeg = jb, .width = width, .height = height, .dpi = dpi });
                pos += len;
            }
            if (pos != frame.len) return cx.fail(caps.infra, failed, .{});
            return total;
        },
        1 => {
            if (frame.len < 6) return cx.fail(caps.infra, "PDF rendering failed", .{});
            const kind = std.enums.fromInt(errors.Kind, frame[1]) orelse return cx.fail(caps.infra, "PDF rendering failed", .{});
            const len: usize = readU32(frame, 2);
            if (6 + len != frame.len) return cx.fail(caps.infra, "PDF rendering failed", .{});
            cx.kind = kind;
            cx.message = frame[6 .. 6 + len];
            return error.Refused;
        },
        2 => return error.OutOfMemory,
        else => return cx.fail(caps.infra, failed, .{}),
    }
}

/// The worker process: reads its request, applies the limits, opens PDFium, writes one reply frame and exits.
pub fn serve() noreturn {
    closeRange(3, ~@as(c_uint, 0));
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    var cx: Cx = .{ .a = arena.allocator() };
    var pages: std.ArrayList(Page) = .empty;
    if (work(&cx, &pages)) |total| {
        _ = writeAll(1, &.{0});
        _ = writeInt(u32, 1, @intCast(pages.items.len));
        _ = writeInt(u32, 1, total);
        for (pages.items) |p| {
            for ([_]u32{ p.index, p.width, p.height, p.dpi, @intCast(p.jpeg.len) }) |v| _ = writeInt(u32, 1, v);
            _ = writeAll(1, p.jpeg);
        }
    } else |err| switch (err) {
        error.OutOfMemory => _ = writeAll(1, &.{2}),
        error.Refused => {
            _ = writeAll(1, &.{ 1, @backingInt(cx.kind) });
            _ = writeInt(u32, 1, @intCast(cx.message.len));
            _ = writeAll(1, cx.message);
        },
    }
    std.c._exit(0);
}

fn work(cx: *Cx, pages: *std.ArrayList(Page)) errors.Refused!u32 {
    var req: Request = undefined;
    if (!readAll(0, std.mem.asBytes(&req)) or req.pdf_len > pdf.max_pdf_bytes) return cx.refuse(failed);
    const deadline_ns = std.Io.Clock.awake.now(log.io()).toNanoseconds() + req.remaining_ns;
    const infra = std.enums.fromInt(errors.Kind, req.infra) orelse return cx.refuse(failed);
    const Tag = std.meta.Tag(pdf.Selection);
    const sel: pdf.Selection = switch (req.sel_kind) {
        @backingInt(Tag.all_within) => .{ .all_within = req.sel_n },
        @backingInt(Tag.first) => .{ .first = req.sel_n },
        @backingInt(Tag.list) => blk: {
            if (req.sel_n > max_list) return cx.fail(infra, failed, .{});
            const list = try cx.a.alloc(u32, req.sel_n);
            if (!readAll(0, std.mem.sliceAsBytes(list))) return cx.fail(infra, failed, .{});
            break :blk .{ .list = list };
        },
        else => return cx.fail(infra, failed, .{}),
    };
    const pdf_bytes = try cx.a.alloc(u8, @intCast(req.pdf_len));
    if (!readAll(0, pdf_bytes)) return cx.fail(infra, failed, .{});
    applyChildLimits();
    const libs = sys.optional_libraries;
    const f = jpeg_out.loadDynLib(pdf.PdfiumFns, .pdfium, cx.a) orelse return cx.fail(.unimplemented, "{s}", .{libs.spec(.pdfium).message});
    const jf = jpeg_out.loadDynLib(jpeg_out.Fns, .jpeg, cx.a) orelse return cx.fail(.unimplemented, "{s}", .{libs.spec(.jpeg).message});
    f.FPDF_InitLibrary();
    const caps: pdf.Caps = .{ .request_pixels = req.request_pixels, .request_bytes = @intCast(req.request_bytes), .page_bytes = @intCast(req.page_bytes), .max_images = req.max_images, .infra = infra };
    var budget: Budget = .{ .images_used = req.images_used, .pdf_pages_used = req.pdf_pages_used, .encoded_bytes = @intCast(req.encoded_bytes), .pixels_used = req.pixels_used };
    if (req.measure > 1) return cx.fail(infra, failed, .{});
    return pdf.renderInProcess(cx, f, jf, .{ .pdf_bytes = pdf_bytes, .sel = sel, .caps = caps, .deadline_ns = deadline_ns, .measure = req.measure == 1 }, &budget, pages);
}

fn applyChildLimits() void {
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
                    const cap = pages * std.heap.page_size_min + pdf.max_child_heap_bytes;
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

fn readAll(fd: c_int, buf: []u8) bool {
    var off: usize = 0;
    while (off < buf.len) {
        const n = std.c.read(fd, buf[off..].ptr, buf.len - off);
        if (n <= 0) return false;
        off += @intCast(n);
    }
    return true;
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

/// A refused render of the two-page fixture by a `script` worker under `caps`, within `ms`; the budget stays untouched.
fn fakeWorkerCx(a: Allocator, script: []const u8, ms: i96, caps: pdf.Caps) !Cx {
    log.init(std.testing.io, false);
    var cx: Cx = .{ .a = a };
    var budget: Budget = .{};
    var pages: std.ArrayList(Page) = .empty;
    const deadline_ns = std.Io.Clock.awake.now(log.io()).toNanoseconds() + ms * std.time.ns_per_ms;
    const job: pdf.Job = .{ .pdf_bytes = pdf.minimal_two_page_pdf, .sel = .{ .all_within = 4 }, .caps = caps, .deadline_ns = deadline_ns };
    try std.testing.expectError(error.Refused, render(&cx, &.{ "/bin/sh", "-c", script }, job, &budget, &pages));
    try std.testing.expectEqual(Budget{}, budget);
    return cx;
}

fn fakeWorker(a: Allocator, script: []const u8, ms: i96) ![]const u8 {
    return (try fakeWorkerCx(a, script, ms, pdf.chatCaps(4))).message;
}

test "PDF render worker: a worker killed by a signal is named as a crash; a limit or a missing reply keeps today's message" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest; // the worker is POSIX only; Windows renders in process
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("the PDF render worker crashed (SIGSEGV)", try fakeWorker(a, "cat >/dev/null; kill -SEGV $$", 10_000));
    try std.testing.expectEqualStrings(failed, try fakeWorker(a, "cat >/dev/null; ulimit -t 1; exec /bin/sh -c 'while :; do :; done'", 10_000));
    try std.testing.expectEqualStrings(failed, try fakeWorker(a, "cat >/dev/null; exit 0", 10_000));
    try std.testing.expectEqualStrings(failed, try fakeWorker(a, "cat >/dev/null; exit 3", 10_000));
    for ([_]std.process.Child.Term{ .{ .signal = .KILL }, .{ .signal = .XCPU }, .{ .signal = .XFSZ }, .{ .signal = .ABRT } }) |term| {
        var cx: Cx = .{ .a = a };
        var budget: Budget = .{};
        var pages: std.ArrayList(Page) = .empty;
        try std.testing.expectError(error.Refused, reply(&cx, term, &.{ 0, 0, 0, 0, 0 }, .{ .all_within = 4 }, false, pdf.chatCaps(4), &budget, &pages));
        try std.testing.expectEqualStrings(if (term.signal == .ABRT) "the PDF render worker crashed (SIGABRT)" else failed, cx.message);
    }
}

test "PDF render worker: a worker past its deadline is killed and reported as timed out" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest; // the worker is POSIX only; Windows renders in process
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const started = std.Io.Clock.awake.now(log.io()).toNanoseconds();
    try std.testing.expectEqualStrings("PDF rendering timed out", try fakeWorker(arena.allocator(), "exec sleep 30", 300));
    try std.testing.expect(std.Io.Clock.awake.now(log.io()).toNanoseconds() - started < 10 * std.time.ns_per_s);
}

test "pdf pages: a malformed worker frame, a crash or an oversized reply is a renderer failure of the caller's kind" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest; // the worker is POSIX only; Windows renders in process
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var budget: Budget = .{};
    var pages: std.ArrayList(Page) = .empty;
    const ocr = pdf.ocrCaps(4);
    const index_wrong = [_]u8{ 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 28, 0, 0, 0, 28, 0, 0, 0, 144, 0, 0, 0, 1, 0, 0, 0, 0xFF };
    const sof = [_]u8{ 0xFF, 0xD8, 0xFF, 0xC0, 0, 11, 8, 0x10, 0, 0x10, 0, 1, 1, 0x11, 0 }; // a 4096 x 4096 frame header
    const too_big = [_]u8{ 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0, 0, 0, 0x10, 0, 0, 144, 0, 0, 0, sof.len, 0, 0, 0 } ++ sof;
    const bad_kind = [_]u8{ 1, 0xFF, 3, 0, 0, 0, 'b', 'a', 'd' };
    const frames = [_][]const u8{ &.{}, &.{ 0, 1 }, &.{ 1, 0 }, &.{ 0, 1, 0, 0, 0, 2, 0, 0, 0 }, &index_wrong, &.{9}, &bad_kind };
    for ([_]pdf.Caps{ ocr, pdf.chatCaps(4) }) |caps| {
        for (frames) |frame| {
            var cx: Cx = .{ .a = a };
            try std.testing.expectError(error.Refused, parseChildReply(&cx, frame, .{ .first = 2 }, false, caps, &budget, &pages));
            try std.testing.expectEqual(caps.infra, cx.kind);
        }
        var cx: Cx = .{ .a = a };
        try std.testing.expectError(error.Refused, parseChildReply(&cx, &too_big, .{ .first = 1 }, false, caps, &budget, &pages));
        try std.testing.expectEqual(caps.infra, cx.kind);
        // a crash, and a reply past the caps' encoded bytes, are the caller's kind too
        try std.testing.expectEqual(caps.infra, (try fakeWorkerCx(a, "cat >/dev/null; kill -SEGV $$", 10_000, caps)).kind);
    }
    const over = try fakeWorkerCx(a, "cat >/dev/null; head -c 2000000 /dev/zero", 10_000, pdf.ocrCaps(0));
    try std.testing.expectEqualStrings(failed, over.message);
    try std.testing.expectEqual(errors.Kind.server, over.kind);
    try std.testing.expectEqual(Budget{}, budget);
    try std.testing.expectEqual(@as(usize, 0), pages.items.len);
    var oom_cx: Cx = .{ .a = a };
    try std.testing.expectError(error.OutOfMemory, parseChildReply(&oom_cx, &.{2}, .{ .first = 2 }, false, ocr, &budget, &pages));
}
