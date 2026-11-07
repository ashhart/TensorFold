//! Borrowed views into mapped files read again past the page cache, by threads, and pointed at the copies.

const std = @import("std");
const direct_io = @import("direct_io.zig");

/// The bytes of one reader's read.
pub const chunk = 8 << 20;
pub const max_readers = 16;
/// Files one read may draw on (a loader's held shards).
pub const max_sources = 64;
/// Ranges closer than this in one file are read as one: the gap costs less than another request.
pub const join_gap = 1 << 20;

/// One mapped file: its path and the mapping its views point into.
pub const Source = struct { path: [:0]const u8, memory: []const u8 };

/// A piece: `len` bytes at file offset `lo` (4 KiB-aligned) of `file`, read to `dst` within the arena.
const Piece = struct { file: usize, lo: u64, len: usize, dst: usize };
const Range = struct { file: usize, lo: u64, hi: u64 };

/// TF_LOAD_DIRECT=0 leaves loaders on their mappings; unset or any other value, they read past the page cache.
pub fn enabled() bool {
    const v = std.c.getenv("TF_LOAD_DIRECT") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

pub const Gather = struct {
    gpa: std.mem.Allocator,
    readers: usize,
    arena: []align(direct_io.alignment) u8 = &.{},

    pub fn init(gpa: std.mem.Allocator, readers: usize) Gather {
        return .{ .gpa = gpa, .readers = @max(1, @min(readers, max_readers)) };
    }

    pub fn deinit(g: *Gather) void {
        if (g.arena.len != 0) g.gpa.free(g.arena);
        g.arena = &.{};
    }

    /// Each non-empty view re-read from its source's file into the arena and pointed there, until the next read.
    pub fn read(g: *Gather, sources: []const Source, views: []const *[]const u8) !void {
        if (sources.len > max_sources) return error.TooManySources;
        var ranges: std.ArrayList(Range) = .empty;
        defer ranges.deinit(g.gpa);
        for (views) |v| {
            if (v.len == 0) continue;
            const file = locate(sources, v.*) orelse return error.UnmappedView;
            const lo: u64 = @intFromPtr(v.ptr) - @intFromPtr(sources[file].memory.ptr);
            try ranges.append(g.gpa, .{ .file = file, .lo = lo, .hi = lo + v.len });
        }
        if (ranges.items.len == 0) return;
        std.mem.sort(Range, ranges.items, {}, lower);
        // joined ranges, each laid out in the arena from its 4 KiB-aligned start so a view keeps its offset in it
        var joined: std.ArrayList(Range) = .empty;
        defer joined.deinit(g.gpa);
        for (ranges.items) |r| {
            if (joined.items.len > 0) {
                const last = &joined.items[joined.items.len - 1];
                if (last.file == r.file and r.lo <= last.hi + join_gap) {
                    last.hi = @max(last.hi, r.hi);
                    continue;
                }
            }
            try joined.append(g.gpa, .{ .file = r.file, .lo = std.mem.alignBackward(u64, r.lo, direct_io.alignment), .hi = r.hi });
        }
        var pieces: std.ArrayList(Piece) = .empty;
        defer pieces.deinit(g.gpa);
        var bases = try g.gpa.alloc(usize, joined.items.len);
        defer g.gpa.free(bases);
        var at: usize = 0;
        for (joined.items, 0..) |j, k| {
            bases[k] = at;
            var lo = j.lo;
            while (lo < j.hi) : (lo = @min(lo + chunk, j.hi)) {
                const len: usize = @intCast(@min(lo + chunk, j.hi) - lo);
                try pieces.append(g.gpa, .{ .file = j.file, .lo = lo, .len = len, .dst = at + @as(usize, @intCast(lo - j.lo)) });
            }
            at += direct_io.span(j.lo, @intCast(j.hi - j.lo));
        }
        if (g.arena.len < at) {
            if (g.arena.len != 0) g.gpa.free(g.arena);
            g.arena = &.{};
            g.arena = try g.gpa.alignedAlloc(u8, .fromByteUnits(direct_io.alignment), at);
        }
        try readAll(g, sources, pieces.items);
        // every view into its joined range's copy: joined ranges are sorted like the views' ranges
        for (views) |v| {
            if (v.len == 0) continue;
            const file = locate(sources, v.*).?;
            const lo: u64 = @intFromPtr(v.ptr) - @intFromPtr(sources[file].memory.ptr);
            const k = find(joined.items, file, lo);
            const off = bases[k] + @as(usize, @intCast(lo - joined.items[k].lo));
            v.* = g.arena[off..][0..v.len];
        }
    }
};

pub const Error = @typeInfo(@typeInfo(@TypeOf(Gather.read)).@"fn".return_type.?).error_union.error_set;

/// A loader's hook, run on the reading thread just before a read `Pair.start` began.
pub const Hook = struct { ctx: *anyopaque, run: *const fn (*anyopaque) void };

/// Two arenas: the views one read pointed at stay valid while the next read fills the other, on its own thread.
pub const Pair = struct {
    gathers: [2]Gather,
    at: usize = 0, // the arena the next read fills
    thread: ?std.Thread = null,
    sources: []const Source = &.{},
    views: []const *[]const u8 = &.{},
    hook: ?Hook = null,
    failed: ?Error = null,

    pub fn init(gpa: std.mem.Allocator, readers: usize) Pair {
        return .{ .gathers = .{ .init(gpa, readers), .init(gpa, readers) } };
    }

    pub fn deinit(p: *Pair) void {
        p.wait() catch {};
        for (&p.gathers) |*g| g.deinit();
    }

    /// `views` read now (after any read in flight), into the arena the read before last used.
    pub fn read(p: *Pair, sources: []const Source, views: []const *[]const u8) Error!void {
        try p.wait();
        const g = &p.gathers[p.at];
        p.at = 1 - p.at;
        try g.read(sources, views);
    }

    /// `views` read on a thread of its own (inline if none starts) until `wait`; they must stay put until then.
    pub fn start(p: *Pair, sources: []const Source, views: []const *[]const u8, hook: ?Hook) Error!void {
        try p.wait();
        p.sources = sources;
        p.views = views;
        p.hook = hook;
        p.thread = std.Thread.spawn(.{}, readAhead, .{p}) catch blk: {
            readAhead(p);
            break :blk null;
        };
    }

    /// The read `start` began joined (idempotent), then its error if it failed.
    pub fn wait(p: *Pair) Error!void {
        if (p.thread) |t| t.join();
        p.thread = null;
        const failed = p.failed;
        p.failed = null;
        if (failed) |err| return err;
    }

    fn readAhead(p: *Pair) void {
        if (p.hook) |h| h.run(h.ctx);
        const g = &p.gathers[p.at];
        p.at = 1 - p.at;
        g.read(p.sources, p.views) catch |err| {
            p.failed = err;
        };
    }
};

fn lower(_: void, x: Range, y: Range) bool {
    return if (x.file != y.file) x.file < y.file else x.lo < y.lo;
}

fn locate(sources: []const Source, v: []const u8) ?usize {
    const p = @intFromPtr(v.ptr);
    for (sources, 0..) |s, i| {
        const b = @intFromPtr(s.memory.ptr);
        if (p >= b and p + v.len <= b + s.memory.len) return i;
    }
    return null;
}

/// The joined range of `file` that holds offset `lo` (one exists: every view's range was joined into one).
fn find(joined: []const Range, file: usize, lo: u64) usize {
    var a: usize = 0;
    var b: usize = joined.len;
    while (b - a > 1) {
        const m = (a + b) / 2;
        const j = joined[m];
        if (j.file < file or (j.file == file and j.lo <= lo)) a = m else b = m;
    }
    return a;
}

const Work = struct {
    sources: []const Source,
    pieces: []const Piece,
    arena: []align(direct_io.alignment) u8,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
};

fn readAll(g: *Gather, sources: []const Source, pieces: []const Piece) !void {
    var w: Work = .{ .sources = sources, .pieces = pieces, .arena = g.arena };
    var threads: [max_readers]?std.Thread = @splat(null);
    const n = @min(g.readers, pieces.len);
    for (threads[1..n]) |*t| t.* = std.Thread.spawn(.{}, run, .{&w}) catch null;
    run(&w);
    for (threads[1..n]) |t| if (t) |th| th.join();
    if (w.failed.load(.monotonic)) return error.ReadFailed;
}

/// One reader: pieces off the shared cursor, each file opened once by this reader (direct where it is allowed).
fn run(w: *Work) void {
    var files: [max_sources]?direct_io.File = @splat(null);
    defer for (&files) |*f| if (f.*) |*file| file.close();
    while (!w.failed.load(.monotonic)) {
        const i = w.next.fetchAdd(1, .monotonic);
        if (i >= w.pieces.len) return;
        const p = w.pieces[i];
        if (files[p.file] == null) files[p.file] = direct_io.File.open(w.sources[p.file].path) catch {
            w.failed.store(true, .monotonic);
            return;
        };
        const dst: direct_io.Buffer = @alignCast(w.arena[p.dst..][0..direct_io.span(p.lo, p.len)]);
        _ = files[p.file].?.read(dst, p.lo, p.len) catch {
            w.failed.store(true, .monotonic);
            return;
        };
    }
}

test "views read past the page cache hold their mapped bytes, joined, cut and in any order" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const n = 3 * chunk + 12345;
    const bytes = try gpa.alloc(u8, n);
    defer gpa.free(bytes);
    var prng = std.Random.DefaultPrng.init(7);
    prng.random().bytes(bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "a.bin", .data = bytes });
    const real = try tmp.dir.realPathFileAlloc(io, "a.bin", gpa);
    defer gpa.free(real);
    const path = try gpa.dupeSentinel(u8, real, 0);
    defer gpa.free(path);
    const sources = [_]Source{.{ .path = path, .memory = bytes }};
    // spans across pieces, a 1-byte view, an unaligned tail, two views far apart, and an empty one
    var views = [_][]const u8{ bytes[5 .. 2 * chunk + 9], bytes[n - 1 ..], bytes[chunk + 3 ..][0..1], bytes[3 * chunk ..][0..77], &.{} };
    var ptrs: [views.len]*[]const u8 = undefined;
    for (&views, 0..) |*v, i| ptrs[i] = v;
    var g = Gather.init(gpa, 4);
    defer g.deinit();
    try g.read(&sources, &ptrs);
    try std.testing.expectEqualSlices(u8, bytes[5 .. 2 * chunk + 9], views[0]);
    try std.testing.expectEqualSlices(u8, bytes[n - 1 ..], views[1]);
    try std.testing.expectEqualSlices(u8, bytes[chunk + 3 ..][0..1], views[2]);
    try std.testing.expectEqualSlices(u8, bytes[3 * chunk ..][0..77], views[3]);
    try std.testing.expectEqual(@as(usize, 0), views[4].len);
    for (views[0..4]) |v| {
        const p = @intFromPtr(v.ptr);
        try std.testing.expect(p >= @intFromPtr(g.arena.ptr) and p + v.len <= @intFromPtr(g.arena.ptr) + g.arena.len);
    }
}

test "a pair reads ahead into one arena while the other's views stay readable" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const bytes = try gpa.alloc(u8, 2 * chunk + 4096);
    defer gpa.free(bytes);
    var prng = std.Random.DefaultPrng.init(11);
    prng.random().bytes(bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "p.bin", .data = bytes });
    const real = try tmp.dir.realPathFileAlloc(io, "p.bin", gpa);
    defer gpa.free(real);
    const path = try gpa.dupeSentinel(u8, real, 0);
    defer gpa.free(path);
    const sources = [_]Source{.{ .path = path, .memory = bytes }};
    var p = Pair.init(gpa, 2);
    defer p.deinit();
    var a: []const u8 = bytes[3..][0..chunk];
    var b: []const u8 = bytes[chunk + 5 ..][0..chunk];
    var c: []const u8 = bytes[7..][0..100];
    try p.read(&sources, &.{&a});
    const Count = struct {
        n: usize = 0,
        fn bump(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.n += 1;
        }
    };
    var count: Count = .{};
    try p.start(&sources, &.{&b}, .{ .ctx = &count, .run = Count.bump });
    try std.testing.expectEqualSlices(u8, bytes[3..][0..chunk], a); // a's arena is not the one being filled
    try p.wait();
    try std.testing.expectEqualSlices(u8, bytes[chunk + 5 ..][0..chunk], b);
    try std.testing.expectEqual(@as(usize, 1), count.n);
    // the third read takes a's arena again: b's views stay where they were
    try p.read(&sources, &.{&c});
    try std.testing.expectEqualSlices(u8, bytes[7..][0..100], c);
    try std.testing.expectEqualSlices(u8, bytes[chunk + 5 ..][0..chunk], b);
    try std.testing.expect(@intFromPtr(c.ptr) >= @intFromPtr(p.gathers[0].arena.ptr));
    try std.testing.expect(@intFromPtr(c.ptr) < @intFromPtr(p.gathers[0].arena.ptr) + p.gathers[0].arena.len);
}

test "a view outside every mapping is refused" {
    const gpa = std.testing.allocator;
    const a: [16]u8 = @splat(1);
    const b: [16]u8 = @splat(2);
    const sources = [_]Source{.{ .path = "/nonexistent", .memory = &a }};
    var v: []const u8 = b[0..4];
    var g = Gather.init(gpa, 2);
    defer g.deinit();
    try std.testing.expectError(error.UnmappedView, g.read(&sources, &.{&v}));
}
