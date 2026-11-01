//! Tensors under a name prefix read into the page cache ahead of a loader on the mapping, by threads of their own.

const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const Checkpoint = @import("checkpoint.zig").Checkpoint;

/// The bytes of one reader's pread.
pub const chunk = 8 << 20;
pub const max_readers = 16;

/// A piece of one file: its path lives in the checkpoint's arena, which outlives the readers.
const Span = struct { path: [*:0]const u8, lo: u64, hi: u64 };

const Work = struct {
    gpa: std.mem.Allocator,
    spans: []Span,
    next: std.atomic.Value(usize) = .init(0),
    read: std.atomic.Value(u64) = .init(0),
};

/// Readers `start` began; a reader that fails stops, and the loader faults those bytes in itself.
pub const Ahead = struct {
    work: ?*Work = null,
    threads: [max_readers]?std.Thread = @splat(null),

    /// Waits for the readers and frees their work: the bytes they read.
    pub fn join(a: *Ahead) u64 {
        const w = a.work orelse return 0;
        for (&a.threads) |*t| if (t.*) |thread| {
            thread.join();
            t.* = null;
        };
        const n = w.read.load(.monotonic);
        w.gpa.free(w.spans);
        w.gpa.destroy(w);
        a.work = null;
        return n;
    }
};

/// `readers` threads over every tensor named `prefix`..., 8 MiB at a time in each file's order; none off Linux.
pub fn start(ck: *const Checkpoint, prefix: []const u8, readers: usize) Ahead {
    var a: Ahead = .{};
    if (builtin.os.tag != .linux or readers == 0) return a;
    const spans = cut(ck, prefix) catch return a;
    const w = ck.gpa.create(Work) catch {
        ck.gpa.free(spans);
        return a;
    };
    w.* = .{ .gpa = ck.gpa, .spans = spans };
    a.work = w;
    for (a.threads[0..@min(readers, max_readers, spans.len)]) |*t| t.* = std.Thread.spawn(.{}, run, .{w}) catch null;
    return a;
}

/// The byte ranges of `prefix`'s tensors in each mapped file, joined where they touch and cut into `chunk` pieces.
fn cut(ck: *const Checkpoint, prefix: []const u8) ![]Span {
    const gpa = ck.gpa;
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(gpa);
    var ranges: std.ArrayList([2]u64) = .empty;
    defer ranges.deinit(gpa);
    for (ck.files.items) |*f| {
        if (f.map.section == null) continue; // a heap copy (no mmap) holds its bytes already
        ranges.clearRetainingCapacity();
        var it = f.names.iterator();
        while (it.next()) |e| if (std.mem.startsWith(u8, e.key_ptr.*, prefix)) {
            try ranges.append(gpa, .{ f.data + e.value_ptr.begin, f.data + e.value_ptr.end });
        };
        std.mem.sort([2]u64, ranges.items, {}, lower);
        var k: usize = 0;
        while (k < ranges.items.len) {
            var lo, var hi = ranges.items[k];
            k += 1;
            while (k < ranges.items.len and ranges.items[k][0] <= hi) : (k += 1) hi = @max(hi, ranges.items[k][1]);
            while (lo < hi) : (lo = @min(lo + chunk, hi)) try out.append(gpa, .{ .path = f.path, .lo = lo, .hi = @min(lo + chunk, hi) });
        }
    }
    return out.toOwnedSlice(gpa);
}

fn lower(_: void, x: [2]u64, y: [2]u64) bool {
    return x[0] < y[0];
}

/// One reader: pieces off the shared cursor, each file on a descriptor of its own (its own readahead).
fn run(w: *Work) void {
    const buf = std.heap.page_allocator.alloc(u8, chunk) catch return;
    defer std.heap.page_allocator.free(buf);
    var path: [*:0]const u8 = "";
    var fd: platform.c.fd_t = -1;
    defer if (fd >= 0) close(fd);
    while (true) {
        const i = w.next.fetchAdd(1, .monotonic);
        if (i >= w.spans.len) return;
        const s = w.spans[i];
        if (s.path != path) {
            if (fd >= 0) close(fd);
            fd = platform.c.open(s.path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
            if (fd < 0) return;
            path = s.path;
        }
        var at = s.lo;
        while (at < s.hi) {
            const n = platform.c.pread(fd, buf.ptr, @intCast(s.hi - at), @intCast(at));
            if (n <= 0) return;
            at += @intCast(n);
        }
        _ = w.read.fetchAdd(s.hi - s.lo, .monotonic);
    }
}

fn close(fd: platform.c.fd_t) void {
    _ = platform.c.close(fd);
}

const fixture = @import("checkpoint_fixture.zig");

test "readers read every byte named under a prefix, a piece at a time, and none past it" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const big = try gpa.alloc(u8, chunk + 3); // two pieces
    defer gpa.free(big);
    @memset(big, 7);
    const small: [100]u8 = @splat(1);
    // layers.1.c sits apart from layers.1.a (layers.10.b between): two ranges, the second cut in two
    try fixture.write(gpa, io, tmp.dir, "model.safetensors", &.{
        .{ .name = "layers.1.a", .shape = &.{100}, .bytes = &small },
        .{ .name = "layers.10.b", .shape = &.{50}, .bytes = small[0..50] },
        .{ .name = "layers.1.c", .shape = &.{chunk + 3}, .bytes = big },
        .{ .name = "layers.2.d", .shape = &.{7}, .bytes = small[0..7] },
    });
    var path: [64]u8 = undefined;
    var ck = try Checkpoint.openModel(gpa, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    const spans = try cut(&ck, "layers.1.");
    defer gpa.free(spans);
    try std.testing.expectEqual(@as(usize, 3), spans.len); // layers.1.a whole, then layers.1.c as 8 MiB and 3 bytes
    try std.testing.expectEqual(@as(u64, chunk), spans[1].hi - spans[1].lo);
    const cases = [_]struct { []const u8, usize, u64 }{
        .{ "layers.1.", 3, 100 + chunk + 3 },
        .{ "layers.", 2, 100 + 50 + chunk + 3 + 7 },
        .{ "layers.1.", 1, 100 + chunk + 3 },
        .{ "layers.3.", 2, 0 },
        .{ "layers.1.", 0, 0 },
    };
    for (cases) |c| {
        var ahead = start(&ck, c[0], c[1]);
        try std.testing.expectEqual(c[2], ahead.join());
    }
    try std.testing.expectEqualSlices(u8, big, (try ck.get("layers.1.c")).bytes); // the loader's own bytes
}

test "a map that fell back to reads (no mmap) is not read again" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{ .disable_memory_mapping = true });
    defer threaded.deinit();
    const io = threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes: [64]u8 = @splat(3);
    try fixture.write(gpa, io, tmp.dir, "model.safetensors", &.{.{ .name = "layers.0.w", .shape = &.{64}, .bytes = &bytes }});
    var path: [64]u8 = undefined;
    var ck = try Checkpoint.openModel(gpa, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    var ahead = start(&ck, "layers.0.", 4);
    try std.testing.expectEqual(@as(u64, 0), ahead.join());
}
