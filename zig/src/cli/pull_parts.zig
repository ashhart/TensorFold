//! A large file as concurrent byte ranges written in place; each finished range leaves a marker, so a restart fetches only the rest.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Files at least this large come down in ranges.
pub const threshold: u64 = 64 << 20;
pub const most_parts = 16;
const attempts = 5;

/// How many ranges `size` takes: one per 32 MiB, at most 16.
pub fn count(size: u64) u32 {
    return @intCast(std.math.clamp(size / (32 << 20), 1, most_parts));
}

fn bounds(size: u64, parts: u32, i: u32) [2]u64 {
    const step = (size + parts - 1) / parts;
    const first = @min(size, step * i);
    return .{ first, @min(size, first + step) };
}

const Job = struct { url: []const u8, file: std.Io.File, io: std.Io, first: u64, end: u64, marker: []const u8, done: u64 = 0, failure: ?anyerror = null };

/// `url` into `path` (`size` bytes) as `parts` concurrent ranges; error.RangeUnsupported when the server answers a range with the whole file.
pub fn fetch(a: Allocator, io: std.Io, url: []const u8, path: []const u8, size: u64, parts: u32) !void {
    if (parts == 0 or parts > most_parts) return error.BadPartCount;
    const w = std.Io.Dir.cwd();
    const file = try w.createFile(io, path, .{ .read = true, .truncate = false });
    defer file.close(io);
    try file.setLength(io, size);
    var jobs: [most_parts]Job = undefined;
    var threads: [most_parts]?std.Thread = @splat(null);
    defer for (threads[0..parts]) |t| if (t) |x| x.join();
    for (0..parts) |i| {
        const b = bounds(size, parts, @intCast(i));
        jobs[i] = .{ .url = url, .file = file, .io = io, .first = b[0], .end = b[1], .marker = try std.fmt.allocPrint(a, "{s}.part{d}", .{ path, i }) };
        if (w.access(io, jobs[i].marker, .{})) |_| continue else |_| {}
        threads[i] = try std.Thread.spawn(.{}, worker, .{&jobs[i]});
    }
    for (threads[0..parts]) |*t| if (t.*) |x| {
        x.join();
        t.* = null;
    };
    for (jobs[0..parts]) |j| if (j.failure) |e| return e;
    for (jobs[0..parts]) |j| w.deleteFile(io, j.marker) catch {};
}

fn worker(j: *Job) void {
    var last: anyerror = error.ReadFailed;
    for (0..attempts) |_| {
        range(j) catch |e| {
            if (e == error.RangeUnsupported) break;
            last = e; // a reset or stall: the next attempt resumes where this one stopped
            continue;
        };
        std.Io.Dir.cwd().writeFile(j.io, .{ .sub_path = j.marker, .data = "" }) catch {};
        return;
    } else {
        j.failure = last;
        return;
    }
    j.failure = error.RangeUnsupported;
}

fn range(j: *Job) !void {
    if (j.first + j.done >= j.end) return;
    var client: std.http.Client = .{ .allocator = std.heap.page_allocator, .io = j.io };
    defer client.deinit();
    var range_buf: [64]u8 = undefined;
    const headers = [_]std.http.Header{.{ .name = "Range", .value = try std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ j.first + j.done, j.end - 1 }) }};
    var req = try client.request(.GET, try std.Uri.parse(j.url), .{ .extra_headers = &headers, .headers = .{ .accept_encoding = .{ .override = "identity" } } });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buf: [8 << 10]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);
    if (response.head.status != .partial_content) return error.RangeUnsupported;
    var reader_buf: [64 << 10]u8 = undefined;
    var r = response.reader(&reader_buf);
    var chunk: [256 << 10]u8 = undefined;
    while (j.first + j.done < j.end) {
        const want: usize = @intCast(@min(chunk.len, j.end - j.first - j.done));
        const n = r.readSliceShort(chunk[0..want]) catch return error.ReadFailed;
        if (n == 0) return error.ShortRange;
        try j.file.writePositionalAll(j.io, chunk[0..n], j.first + j.done);
        j.done += n;
    }
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

test "ranges cover every byte once, at most 16, at least 32 MiB each" {
    try std.testing.expectEqual(@as(u32, 1), count(10 << 20));
    try std.testing.expectEqual(@as(u32, 4), count(128 << 20));
    try std.testing.expectEqual(@as(u32, 16), count(5 << 30));
    for ([_]u64{ 1, 5000, 128 << 20, (5 << 30) + 3 }) |size| {
        const parts = count(size);
        var next: u64 = 0;
        for (0..parts) |i| {
            const b = bounds(size, parts, @intCast(i));
            try std.testing.expectEqual(next, b[0]);
            next = b[1];
        }
        try std.testing.expectEqual(size, next);
    }
}
