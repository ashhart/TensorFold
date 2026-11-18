//! Checked load-time readers extending checkpoint_prefetch; uploads and page-release policy stay with the loader.
const std = @import("std");
const prefetch = @import("checkpoint_prefetch.zig");
const Checkpoint = @import("checkpoint.zig").Checkpoint;
pub const Span = prefetch.Span;
pub const readerCount = prefetch.readerCount;
const none = std.math.maxInt(usize);

pub fn readers() !usize {
    return prefetch.configured() catch |err| {
        if (!@import("builtin").is_test) std.log.err("TF_LOAD_PREFETCH: expected 0 to {d} reader threads", .{prefetch.max_readers});
        return err;
    };
}

/// New CUDA ports stay opt-in until their checkpoint and topology have device memory/load receipts.
pub fn optInCount(value: ?[]const u8) !usize {
    return if (value) |v| readerCount(v) else 0;
}

pub fn optInReaders() !usize {
    return optInCount(if (std.c.getenv("TF_LOAD_PREFETCH")) |v| std.mem.span(v) else null) catch |err| {
        if (!@import("builtin").is_test) std.log.err("TF_LOAD_PREFETCH: expected 0 to {d} reader threads", .{prefetch.max_readers});
        return err;
    };
}

const Work = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    spans: []Span,
    next: std.atomic.Value(usize) = .init(0),
    bytes: std.atomic.Value(u64) = .init(0),
    completed: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(usize) = .init(none),
    reason: anyerror = error.Unexpected,
    buffers: [prefetch.max_readers]?[]u8 = @splat(null),

    fn fail(w: *Work, index: usize, reason: anyerror) void {
        if (w.failed.cmpxchgStrong(none, index, .monotonic, .monotonic) == null) w.reason = reason;
    }
};

pub const Problem = struct { path: [*:0]const u8, reason: anyerror };
pub const Reader = struct {
    work: ?*Work = null,
    threads: [prefetch.max_readers]?std.Thread = @splat(null),

    fn join(r: *Reader) void {
        for (&r.threads) |*t| if (t.*) |thread| {
            thread.join();
            t.* = null;
        };
    }
    /// Call only after check has joined the readers; names live until the scope is deinitialized.
    pub fn problem(r: *const Reader) ?Problem {
        const w = r.work orelse return null;
        const index = w.failed.load(.monotonic);
        return if (index == none) null else .{ .path = w.spans[index].path, .reason = w.reason };
    }
    pub fn check(r: *Reader) !u64 {
        r.join();
        if (r.problem()) |p| {
            if (!@import("builtin").is_test) std.log.err("TF_LOAD_PREFETCH: {s}: {t}", .{ p.path, p.reason });
            return error.LoadPrefetchRead;
        }
        return if (r.work) |w| w.bytes.load(.monotonic) else 0;
    }
    pub fn deinit(r: *Reader) void {
        r.join();
        const w = r.work orelse return;
        for (w.buffers) |buf| if (buf) |b| w.gpa.free(b);
        w.gpa.free(w.spans);
        const gpa = w.gpa;
        gpa.destroy(w);
        r.work = null;
    }
};

/// Owns `spans` from entry; their paths must outlive deinit. All scratch is allocated before its reader starts.
pub fn start(gpa: std.mem.Allocator, io: std.Io, spans: []Span, count: usize) !Reader {
    var owned = true;
    errdefer if (owned) gpa.free(spans);
    if (count > prefetch.max_readers) return error.BadLoadPrefetch;
    if (count == 0 or spans.len == 0) {
        gpa.free(spans);
        return .{};
    }
    const w = try gpa.create(Work);
    w.* = .{ .gpa = gpa, .io = io, .spans = spans };
    var r: Reader = .{ .work = w };
    owned = false;
    errdefer r.deinit();
    for (r.threads[0..@min(count, spans.len)], 0..) |*thread, i| {
        w.buffers[i] = try gpa.alloc(u8, prefetch.chunk);
        thread.* = try std.Thread.spawn(.{}, run, .{ w, w.buffers[i].? });
    }
    return r;
}

fn run(w: *Work, buf: []u8) void {
    var opened: ?std.Io.File = null;
    defer if (opened) |file| file.close(w.io);
    var path: []const u8 = "";
    while (true) {
        const index = w.next.fetchAdd(1, .monotonic);
        if (index >= w.spans.len) return;
        const span = w.spans[index];
        const name = std.mem.span(span.path);
        if (opened == null or !std.mem.eql(u8, path, name)) {
            if (opened) |file| file.close(w.io);
            opened = null;
            opened = std.Io.Dir.cwd().openFile(w.io, name, .{}) catch |err| {
                w.fail(index, err);
                return;
            };
            path = name;
        }
        if (span.hi < span.lo or span.hi - span.lo > buf.len) {
            w.fail(index, error.BadLoadPrefetchRange);
            return;
        }
        const len: usize = @intCast(span.hi - span.lo);
        const got = opened.?.readPositionalAll(w.io, buf[0..len], span.lo) catch |err| {
            w.fail(index, err);
            return;
        };
        if (got != len) {
            w.fail(index, error.EndOfStream);
            return;
        }
        _ = w.bytes.fetchAdd(len, .monotonic);
        _ = w.completed.fetchAdd(1, .monotonic);
    }
}

pub const Scope = struct {
    reader: Reader = .{},
    arena: ?std.heap.ArenaAllocator = null,
    pub fn check(s: *Scope) !u64 {
        return s.reader.check();
    }
    pub fn deinit(s: *Scope) void {
        s.reader.deinit();
        if (s.arena) |*a| a.deinit();
        s.arena = null;
    }
};

fn refusal(prefix: []const u8, err: anyerror) anyerror {
    if (!@import("builtin").is_test) std.log.err("TF_LOAD_PREFETCH: {s}: {t}", .{ prefix, err });
    return err;
}

pub fn checkpoint(ck: *const Checkpoint, prefix: []const u8, count: usize) !Scope {
    if (count == 0) return .{};
    const spans = prefetch.cut(ck, prefix) catch |err| return refusal(prefix, err);
    return .{ .reader = start(ck.gpa, ck.io, spans, count) catch |err| return refusal(prefix, err) };
}

/// A header-only pack supplies the same file spans without opening a second set of whole-shard mappings.
pub fn pack(gpa: std.mem.Allocator, io: std.Io, p: anytype, prefix: []const u8, count: usize) !Scope {
    if (count == 0) return .{};
    return packInner(gpa, io, p, prefix, count) catch |err| return refusal(prefix, err);
}

fn packInner(gpa: std.mem.Allocator, io: std.Io, p: anytype, prefix: []const u8, count: usize) !Scope {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var spans: std.ArrayList(Span) = .empty;
    errdefer spans.deinit(gpa);
    for (p.files.items, 0..) |file, i| {
        var path: ?[:0]u8 = null;
        for (p.names.keys(), p.names.values()) |name, info| {
            if (info.file != i or !std.mem.startsWith(u8, name, prefix) or info.nbytes == 0) continue;
            if (path == null) path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ p.dir, file }, 0);
            const end = try std.math.add(u64, info.start, info.nbytes);
            var lo = info.start;
            while (lo < end) {
                const hi = lo + @min(prefetch.chunk, end - lo);
                try spans.append(gpa, .{ .path = path.?.ptr, .lo = lo, .hi = hi });
                lo = hi;
            }
        }
    }
    const owned = try spans.toOwnedSlice(gpa);
    const reader = try start(gpa, io, owned, count);
    return .{ .reader = reader, .arena = arena };
}
