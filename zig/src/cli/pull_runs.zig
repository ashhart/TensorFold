//! Byte runs of large files on one shared queue: each is checked in memory before it is written to its file.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const workers = 16;
const attempts = 6;
/// A resolved location is used this long before the hub is asked again (its signed links last about an hour).
const fresh_ns: i96 = 30 * std.time.ns_per_min;

/// One file a node fills in part: its hub URL (resolved to the storage link once), where it lands, and its full size.
pub const Target = struct { url: []const u8, path: []const u8, size: u64 };

/// One run of bytes [first, end) of target `file`; `check` decides whether the fetched bytes are the release's.
pub const Run = struct {
    file: u32,
    first: u64,
    end: u64,
    /// The run's own checks, by index into the caller's table (none for a header run).
    units: [2]u32,
};

/// The caller's check: true when a run's `bytes` (from its `first`) match what the release publishes for them.
pub const Check = struct {
    ctx: *const anyopaque,
    run: *const fn (ctx: *const anyopaque, r: Run, bytes: []const u8) bool,
};

const Link = struct {
    mutex: std.Io.Mutex = .init,
    url: ?[]u8 = null,
    at: i96 = 0,
};

const Shared = struct {
    io: std.Io,
    targets: []const Target,
    links: []Link,
    runs: []const Run,
    check: Check,
    max: u64,
    next: std.atomic.Value(usize) = .init(0),
    bytes: std.atomic.Value(u64) = .init(0),
    running: std.atomic.Value(u32) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
};

/// Fetches every run over up to 16 connections; a run whose bytes fail `check` comes again, up to six times.
pub fn fetch(a: Allocator, io: std.Io, targets: []const Target, runs: []const Run, check: Check, progress: ?*std.Io.Writer) !void {
    if (runs.len == 0) return;
    const w = std.Io.Dir.cwd();
    for (targets) |t| {
        const f = try w.createFile(io, t.path, .{ .read = true, .truncate = false });
        defer f.close(io);
        // the file keeps its full size so the release's offsets hold; bytes no run brings stay holes on disk
        if ((try f.length(io)) != t.size) try f.setLength(io, t.size);
    }
    const links = try a.alloc(Link, targets.len);
    for (links) |*l| l.* = .{};
    var max: u64 = 0;
    var left: u64 = 0;
    for (runs) |r| {
        max = @max(max, r.end - r.first);
        left += r.end - r.first;
    }
    var s: Shared = .{ .io = io, .targets = targets, .links = links, .runs = runs, .check = check, .max = max };
    var failures: [workers]?anyerror = @splat(null);
    var threads: [workers]std.Thread = undefined;
    const n = @min(workers, runs.len);
    s.running.store(@intCast(n), .release);
    for (0..n) |i| threads[i] = std.Thread.spawn(.{}, worker, .{ &s, &failures[i] }) catch |e| {
        s.stop.store(true, .release);
        s.running.store(@intCast(i), .release);
        for (threads[0..i]) |t| t.join();
        return e;
    };
    const start = std.Io.Clock.awake.now(io).toNanoseconds();
    var shown = start;
    while (s.running.load(.acquire) > 0) {
        std.Io.sleep(io, .fromMilliseconds(200), .awake) catch {};
        const now = std.Io.Clock.awake.now(io).toNanoseconds();
        if (progress == null or now - shown < 15 * std.time.ns_per_s) continue;
        shown = now;
        const got: f64 = @floatFromInt(s.bytes.load(.monotonic));
        const secs = @as(f64, @floatFromInt(now - start)) / std.time.ns_per_s;
        progress.?.print("  {d:.2} of {d:.2} GiB, {d:.1} MiB/s\n", .{ got / (1 << 30), @as(f64, @floatFromInt(left)) / (1 << 30), got / (1 << 20) / secs }) catch {};
        progress.?.flush() catch {};
    }
    for (threads[0..n]) |t| t.join();
    for (failures[0..n]) |f| if (f) |e| return e;
}

fn worker(s: *Shared, failure: *?anyerror) void {
    defer _ = s.running.fetchSub(1, .acq_rel);
    var client = @import("pull_http.zig").client(std.heap.page_allocator, s.io);
    defer client.deinit();
    const buf = std.heap.page_allocator.alloc(u8, @intCast(s.max)) catch |e| {
        failure.* = e;
        s.stop.store(true, .release);
        return;
    };
    defer std.heap.page_allocator.free(buf);
    // one open file a worker: runs come in file order, so a handle serves many runs
    var open: Open = .{};
    defer open.close(s.io);
    while (!s.stop.load(.acquire)) {
        const i = s.next.fetchAdd(1, .monotonic);
        if (i >= s.runs.len) return;
        const r = s.runs[i];
        const file = open.get(s.io, r.file, s.targets[r.file].path) catch |e| {
            failure.* = e;
            s.stop.store(true, .release);
            return;
        };
        one(s, &client, file, r, buf[0..@intCast(r.end - r.first)]) catch |e| {
            failure.* = e;
            s.stop.store(true, .release);
            return;
        };
    }
}

const Open = struct {
    index: ?u32 = null,
    file: ?std.Io.File = null,

    fn get(o: *Open, io: std.Io, index: u32, path: []const u8) !std.Io.File {
        if (o.index == index) return o.file.?;
        o.close(io);
        o.file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write });
        o.index = index;
        return o.file.?;
    }

    fn close(o: *Open, io: std.Io) void {
        if (o.file) |f| f.close(io);
        o.* = .{};
    }
};

/// One run: fetched whole into `buf`, checked, then written; a stale link is resolved again.
fn one(s: *Shared, client: *std.http.Client, file: std.Io.File, r: Run, buf: []u8) !void {
    var last: anyerror = error.ReadFailed;
    var wait_ms: u64 = 250;
    for (0..attempts) |_| {
        const link = location(s, client, r.file) catch |e| {
            last = e;
            std.Io.sleep(s.io, .fromMilliseconds(@intCast(wait_ms)), .awake) catch {};
            wait_ms *= 2;
            continue;
        };
        get(client, link, r.first, buf) catch |e| {
            last = e;
            if (e == error.Forbidden) forget(s, r.file, link);
            std.Io.sleep(s.io, .fromMilliseconds(@intCast(wait_ms)), .awake) catch {};
            wait_ms *= 2;
            continue;
        };
        if (!s.check.run(s.check.ctx, r, buf)) {
            last = error.ChecksumMismatch;
            continue;
        }
        try file.writePositionalAll(s.io, buf, r.first);
        _ = s.bytes.fetchAdd(buf.len, .monotonic);
        return;
    }
    return last;
}

/// The storage link the hub redirects `file` to, asked once and again when it grows old or is refused.
fn location(s: *Shared, client: *std.http.Client, file: u32) ![]const u8 {
    const l = &s.links[file];
    try l.mutex.lock(s.io);
    defer l.mutex.unlock(s.io);
    const now = std.Io.Clock.awake.now(s.io).toNanoseconds();
    if (l.url) |u| if (now - l.at < fresh_ns) return u;
    // an older link may still be in use by another worker: it stays allocated (a few a file an hour)
    const u = try resolve(std.heap.page_allocator, client, s.targets[file].url);
    l.url = u;
    l.at = now;
    return u;
}

fn forget(s: *Shared, file: u32, link: []const u8) void {
    const l = &s.links[file];
    l.mutex.lockUncancelable(s.io);
    defer l.mutex.unlock(s.io);
    if (l.url != null and l.url.?.ptr == link.ptr) l.at = 0;
}

/// Where `url` lands: the Location of a redirect, else `url` itself.
pub fn resolve(a: Allocator, client: *std.http.Client, url: []const u8) ![]u8 {
    var req = try client.request(.HEAD, try std.Uri.parse(url), .{ .redirect_behavior = .unhandled, .headers = .{ .accept_encoding = .{ .override = "identity" } } });
    defer req.deinit();
    try req.sendBodiless();
    const response = try req.receiveHead(&.{});
    if (response.head.status.class() == .redirect) {
        const to = response.head.location orelse return error.HttpRedirectLocationMissing;
        if (!std.mem.startsWith(u8, to, "https://") and !std.mem.startsWith(u8, to, "http://")) return error.RelativeRedirect;
        return a.dupe(u8, to);
    }
    if (response.head.status != .ok) return error.HubStatus;
    return a.dupe(u8, url);
}

/// Bytes [first, first + buf.len) of `link` into `buf`; the storage must answer 206 for exactly that range.
fn get(client: *std.http.Client, link: []const u8, first: u64, buf: []u8) !void {
    var range_buf: [64]u8 = undefined;
    const headers = [_]std.http.Header{.{ .name = "Range", .value = try std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ first, first + buf.len - 1 }) }};
    var req = try client.request(.GET, try std.Uri.parse(link), .{ .redirect_behavior = .not_allowed, .extra_headers = &headers, .headers = .{ .accept_encoding = .{ .override = "identity" } } });
    defer req.deinit();
    try req.sendBodiless();
    var response = try req.receiveHead(&.{});
    if (response.head.status == .forbidden or response.head.status == .gone) return error.Forbidden;
    if (response.head.status != .partial_content) return error.RangeUnsupported;
    var reader_buf: [64 << 10]u8 = undefined;
    var rd = response.reader(&reader_buf);
    var done: usize = 0;
    while (done < buf.len) {
        const n = rd.readSliceShort(buf[done..]) catch return error.ReadFailed;
        if (n == 0) return error.ShortRange;
        done += n;
    }
}

/// Runs over sorted (offset, length) pieces of one file: adjacent pieces join while the run stays under `limit` bytes.
pub fn coalesce(a: Allocator, file: u32, offs: []const [2]u64, unit0: u32, limit: u64, out: *std.ArrayList(Run)) !void {
    var i: usize = 0;
    while (i < offs.len) {
        var j = i + 1;
        while (j < offs.len and offs[j][0] == offs[j - 1][0] + offs[j - 1][1] and offs[j][0] + offs[j][1] - offs[i][0] <= limit) j += 1;
        try out.append(a, .{ .file = file, .first = offs[i][0], .end = offs[j - 1][0] + offs[j - 1][1], .units = .{ unit0 + @as(u32, @intCast(i)), unit0 + @as(u32, @intCast(j)) } });
        i = j;
    }
}

test "adjacent pieces join into runs up to the limit, and a gap starts a new run" {
    var out: std.ArrayList(Run) = .empty;
    defer out.deinit(std.testing.allocator);
    const offs = [_][2]u64{ .{ 100, 10 }, .{ 110, 10 }, .{ 120, 10 }, .{ 140, 10 }, .{ 150, 10 } };
    try coalesce(std.testing.allocator, 3, &offs, 7, 25, &out);
    try std.testing.expectEqual(@as(usize, 3), out.items.len);
    try std.testing.expectEqual(Run{ .file = 3, .first = 100, .end = 120, .units = .{ 7, 9 } }, out.items[0]);
    try std.testing.expectEqual(Run{ .file = 3, .first = 120, .end = 130, .units = .{ 9, 10 } }, out.items[1]);
    try std.testing.expectEqual(Run{ .file = 3, .first = 140, .end = 160, .units = .{ 10, 12 } }, out.items[2]);
}
