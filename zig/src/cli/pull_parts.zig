//! Large files as fixed pieces on one shared queue; a finished piece leaves a marker, so a restart fetches the rest.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Files at least this large come down in pieces.
pub const threshold: u64 = 64 << 20;
/// The bytes one request asks for.
pub const piece: u64 = 32 << 20;
pub const workers = 16;
const attempts = 5;

/// One file: where its bytes come from, where they land, and how many there are.
pub const File = struct { url: []const u8, path: []const u8, size: u64 };

const Piece = struct { file: usize, first: u64, end: u64, marker: []const u8 };

/// How many pieces of `bytes` each a file of `size` takes.
pub fn count(size: u64, bytes: u64) u64 {
    return @max(1, (size + bytes - 1) / bytes);
}

fn marker(a: Allocator, path: []const u8, i: u64) ![]const u8 {
    return std.fmt.allocPrint(a, "{s}.part{d}", .{ path, i });
}

const Shared = struct {
    io: std.Io,
    files: []const File,
    handles: []const std.Io.File,
    todo: []const Piece,
    next: std.atomic.Value(usize) = .init(0),
    bytes: std.atomic.Value(u64) = .init(0),
    running: std.atomic.Value(u32) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
};

/// `files` as `bytes`-sized pieces over up to 16 connections; error.RangeUnsupported when a range gets the whole file.
pub fn fetch(a: Allocator, io: std.Io, files: []const File, bytes: u64, progress: ?*std.Io.Writer) !void {
    const w = std.Io.Dir.cwd();
    const handles = try a.alloc(std.Io.File, files.len);
    var opened: usize = 0;
    defer for (handles[0..opened]) |h| h.close(io);
    var todo: std.ArrayList(Piece) = .empty;
    var left: u64 = 0;
    for (files, 0..) |f, k| {
        handles[k] = try w.createFile(io, f.path, .{ .read = true, .truncate = false });
        opened += 1;
        try handles[k].setLength(io, f.size);
        for (0..count(f.size, bytes)) |i| {
            const m = try marker(a, f.path, i);
            if (w.access(io, m, .{})) |_| continue else |_| {}
            try todo.append(a, .{ .file = k, .first = i * bytes, .end = @min(f.size, (i + 1) * bytes), .marker = m });
            left += todo.getLast().end - todo.getLast().first;
        }
    }
    var s: Shared = .{ .io = io, .files = files, .handles = handles, .todo = todo.items };
    var failures: [workers]?anyerror = @splat(null);
    var threads: [workers]std.Thread = undefined;
    const n = @min(workers, todo.items.len);
    s.running.store(@intCast(n), .release);
    for (0..n) |i| threads[i] = std.Thread.spawn(.{}, worker, .{ &s, &failures[i] }) catch |e| {
        s.stop.store(true, .release);
        for (threads[0..i]) |t| t.join();
        return e;
    };
    const start = std.Io.Clock.awake.now(io).toNanoseconds();
    var shown = start;
    while (s.running.load(.acquire) > 0) {
        std.Io.sleep(io, .fromMilliseconds(200), .awake) catch {};
        const now = std.Io.Clock.awake.now(io).toNanoseconds();
        if (progress == null or now - shown < 10 * std.time.ns_per_s) continue;
        shown = now;
        const got: f64 = @floatFromInt(s.bytes.load(.monotonic));
        const secs = @as(f64, @floatFromInt(now - start)) / std.time.ns_per_s;
        progress.?.print("  {d:.2} of {d:.2} GiB, {d:.1} MiB/s\n", .{ got / (1 << 30), @as(f64, @floatFromInt(left)) / (1 << 30), got / (1 << 20) / secs }) catch {};
        progress.?.flush() catch {};
    }
    for (threads[0..n]) |t| t.join();
    for (failures[0..n]) |f| if (f) |e| return e;
    for (files) |f| for (0..count(f.size, bytes)) |i| w.deleteFile(io, try marker(a, f.path, i)) catch {};
}

/// Removes a file's bytes and piece markers, so the next fetch starts it over.
pub fn discard(a: Allocator, io: std.Io, f: File, bytes: u64) void {
    const w = std.Io.Dir.cwd();
    w.deleteFile(io, f.path) catch {};
    for (0..count(f.size, bytes)) |i| w.deleteFile(io, marker(a, f.path, i) catch continue) catch {};
}

fn worker(s: *Shared, failure: *?anyerror) void {
    defer _ = s.running.fetchSub(1, .acq_rel);
    var client: std.http.Client = .{ .allocator = std.heap.page_allocator, .io = s.io };
    defer client.deinit();
    while (!s.stop.load(.acquire)) {
        const i = s.next.fetchAdd(1, .monotonic);
        if (i >= s.todo.len) return;
        get(s, &client, s.todo[i]) catch |e| {
            failure.* = e;
            s.stop.store(true, .release);
            return;
        };
    }
}

/// One piece, resuming within it after a reset or stall; the marker goes down once every byte is written.
fn get(s: *Shared, client: *std.http.Client, p: Piece) !void {
    var done: u64 = 0;
    var last: anyerror = error.ReadFailed;
    for (0..attempts) |_| {
        range(s, client, p, &done) catch |e| {
            if (e == error.RangeUnsupported) return e;
            last = e;
            continue;
        };
        try std.Io.Dir.cwd().writeFile(s.io, .{ .sub_path = p.marker, .data = "" });
        return;
    }
    return last;
}

fn range(s: *Shared, client: *std.http.Client, p: Piece, done: *u64) !void {
    const from = p.first + done.*;
    if (from >= p.end) return;
    var range_buf: [64]u8 = undefined;
    const headers = [_]std.http.Header{.{ .name = "Range", .value = try std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ from, p.end - 1 }) }};
    var req = try client.request(.GET, try std.Uri.parse(s.files[p.file].url), .{ .extra_headers = &headers, .headers = .{ .accept_encoding = .{ .override = "identity" } } });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buf: [8 << 10]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);
    if (response.head.status != .partial_content or startOf(response.head) != from) return error.RangeUnsupported;
    var reader_buf: [64 << 10]u8 = undefined;
    var r = response.reader(&reader_buf);
    var chunk: [256 << 10]u8 = undefined;
    while (p.first + done.* < p.end) {
        const want: usize = @intCast(@min(chunk.len, p.end - p.first - done.*));
        const n = r.readSliceShort(chunk[0..want]) catch return error.ReadFailed;
        if (n == 0) return error.ShortRange;
        try s.handles[p.file].writePositionalAll(s.io, chunk[0..n], p.first + done.*);
        done.* += n;
        _ = s.bytes.fetchAdd(n, .monotonic);
    }
}

/// The first byte a 206 says it carries (Content-Range: bytes FIRST-LAST/SIZE).
fn startOf(head: std.http.Client.Response.Head) ?u64 {
    var it = head.iterateHeaders();
    while (it.next()) |h| {
        if (!std.ascii.eqlIgnoreCase(h.name, "content-range")) continue;
        const v = std.mem.trim(u8, h.value, " ");
        if (!std.mem.startsWith(u8, v, "bytes ")) return null;
        const dash = std.mem.indexOfScalar(u8, v, '-') orelse return null;
        return std.fmt.parseInt(u64, v[6..dash], 10) catch null;
    }
    return null;
}

/// The sha256 of `path`'s first `size` bytes.
pub fn digest(a: Allocator, io: std.Io, path: []const u8, size: u64) ![32]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const buf = try a.alloc(u8, 4 << 20);
    defer a.free(buf);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var at: u64 = 0;
    while (at < size) {
        const n = try file.readPositionalAll(io, buf[0..@intCast(@min(buf.len, size - at))], at);
        if (n == 0) return error.ShortFile;
        hash.update(buf[0..n]);
        at += n;
    }
    var out: [32]u8 = undefined;
    hash.final(&out);
    return out;
}

test "pieces cover every byte once" {
    try std.testing.expectEqual(@as(u64, 1), count(1, piece));
    try std.testing.expectEqual(@as(u64, 4), count(128 << 20, piece));
    try std.testing.expectEqual(@as(u64, 5), count((128 << 20) + 1, piece));
    try std.testing.expectEqual(@as(u64, 4), count(5000, 1250));
}
