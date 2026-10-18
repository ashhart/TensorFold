//! Checkpoint bytes onto the GPU in file order through two O_DIRECT slots, not page faults on a cold mapping.
const std = @import("std");
const cuda = @import("cuda");
const dio = @import("core").direct_io;

const slot_bytes = 64 << 20;
/// Unused file bytes one read spans between two copies rather than starting another read.
const max_gap = 1 << 20;

/// `len` bytes at `offset` in file `file`, for the device at `dst`.
const Copy = struct { dst: u64, file: usize, offset: u64, len: usize };
/// One direct read of `len` bytes at `lo`, serving `pieces[first..end]`.
const Read = struct { file: usize, lo: u64, len: usize, first: usize, end: usize };
const Mapping = struct { base: usize, len: usize, path: [:0]const u8 };

pub const Stager = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    files: std.ArrayList(Mapping) = .empty,
    copies: std.ArrayList(Copy) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream) Stager {
        return .{ .gpa = gpa, .d = d, .stream = stream };
    }

    pub fn deinit(st: *Stager) void {
        st.stream.synchronize() catch {};
        st.files.deinit(st.gpa);
        st.copies.deinit(st.gpa);
        st.* = undefined;
    }

    /// `memory`, the whole of `path` mapped from its start: bytes `put` from it are read from the file instead.
    pub fn addFile(st: *Stager, path: [:0]const u8, memory: []const u8) !void {
        try st.files.append(st.gpa, .{ .base = @intFromPtr(memory.ptr), .len = memory.len, .path = path });
    }

    /// Queues bytes for dst (one allocation): mapped bytes are read at finish, other bytes copied now and owned.
    pub fn put(st: *Stager, dst: u64, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        const p = @intFromPtr(bytes.ptr);
        for (st.files.items, 0..) |f, i| if (p >= f.base and p + bytes.len <= f.base + f.len)
            return st.copies.append(st.gpa, .{ .dst = dst, .file = i, .offset = p - f.base, .len = bytes.len });
        try cuda.copy_source.enqueue(st.d, dst, bytes, st.stream);
    }

    /// Zeroes `n` device bytes at `dst` ahead of every copy `finish` makes.
    pub fn zero(st: *Stager, dst: u64, n: usize) !void {
        try st.d.check(st.d.api.cuMemsetD8Async(dst, 0, n, st.stream.handle), "cuMemsetD8Async");
    }

    /// Reads and copies every `put`; returns once all of them have landed.
    pub fn finish(st: *Stager) !void {
        defer st.copies.clearRetainingCapacity();
        const p = try plan(st.gpa, st.copies.items, dio.fits(slot_bytes));
        defer p.deinit(st.gpa);
        const files = try st.gpa.alloc(dio.File, st.files.items.len);
        defer st.gpa.free(files);
        var opened: usize = 0;
        defer for (files[0..opened]) |*f| f.close();
        while (opened < files.len) : (opened += 1) files[opened] = try dio.File.open(st.files.items[opened].path);

        var sink: Device = .{ .d = st.d, .stream = st.stream };
        var made: usize = 0;
        // a failed read leaves copies queued from the slots: they finish before the slots are freed
        defer {
            st.stream.synchronize() catch {};
            for (sink.slots[0..made], sink.events[0..made]) |*s, *e| {
                s.free();
                e.deinit();
            }
        }
        while (made < sink.slots.len) : (made += 1) {
            sink.slots[made] = try cuda.HostBuffer.alloc(st.d, slot_bytes);
            sink.events[made] = cuda.Event.init(st.d, false) catch |err| {
                sink.slots[made].free();
                return err;
            };
        }
        for (sink.slots) |s| if (@intFromPtr(s.bytes.ptr) % dio.alignment != 0) return error.UnalignedPinnedSlot;
        try run(files, p, .{ @alignCast(sink.slots[0].bytes), @alignCast(sink.slots[1].bytes) }, &sink);
        try st.stream.synchronize();
    }
};

/// Page-locked slots copied on by the stream; a slot's event marks its last copy.
const Device = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    slots: [2]cuda.HostBuffer = undefined,
    events: [2]cuda.Event = undefined,

    fn copy(x: *Device, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(x.d, dst, bytes, x.stream);
    }

    fn done(x: *Device, slot: usize) !void {
        try x.events[slot].record(x.stream);
    }

    fn wait(x: *Device, slot: usize) !void {
        try x.events[slot].synchronize();
    }
};

const Plan = struct {
    pieces: []Copy,
    reads: []Read,

    fn deinit(p: Plan, gpa: std.mem.Allocator) void {
        gpa.free(p.pieces);
        gpa.free(p.reads);
    }
};

/// copies sorted by file offset and cut into reads of at most limit; a piece never spans two copies.
fn plan(gpa: std.mem.Allocator, copies: []Copy, limit: usize) !Plan {
    std.mem.sort(Copy, copies, {}, struct {
        fn lt(_: void, a: Copy, b: Copy) bool {
            return if (a.file != b.file) a.file < b.file else a.offset < b.offset;
        }
    }.lt);
    var pieces: std.ArrayList(Copy) = .empty;
    errdefer pieces.deinit(gpa);
    var reads: std.ArrayList(Read) = .empty;
    errdefer reads.deinit(gpa);
    for (copies) |c| {
        var done: usize = 0;
        while (done < c.len) {
            const at = c.offset + done;
            const joins = reads.items.len > 0 and blk: {
                const r = reads.items[reads.items.len - 1];
                break :blk r.file == c.file and at >= r.lo and at - r.lo < limit and at <= r.lo + r.len + max_gap;
            };
            if (!joins) try reads.append(gpa, .{ .file = c.file, .lo = at, .len = 0, .first = pieces.items.len, .end = pieces.items.len });
            const r = &reads.items[reads.items.len - 1];
            const n = @min(c.len - done, limit - @as(usize, @intCast(at - r.lo)));
            r.len = @max(r.len, @as(usize, @intCast(at + n - r.lo)));
            try pieces.append(gpa, .{ .dst = c.dst + done, .file = c.file, .offset = at, .len = n });
            r.end = pieces.items.len;
            done += n;
        }
    }
    const owned_pieces = try pieces.toOwnedSlice(gpa);
    errdefer gpa.free(owned_pieces);
    return .{ .pieces = owned_pieces, .reads = try reads.toOwnedSlice(gpa) };
}

/// Each read into slot i % 2 once sink.wait frees it, then its pieces to sink.copy and sink.done.
fn run(files: []const dio.File, p: Plan, slots: [2]dio.Buffer, sink: anytype) !void {
    for (p.reads, 0..) |r, i| {
        const k = i % 2;
        if (i >= 2) try sink.wait(k);
        const bytes = try files[r.file].read(slots[k], r.lo, r.len);
        for (p.pieces[r.first..r.end]) |c| try sink.copy(c.dst, bytes[@intCast(c.offset - r.lo)..][0..c.len]);
        try sink.done(k);
    }
}

/// A host stand-in for the device whose copies land only when their slot is waited for, as late as a stream may.
const Image = struct {
    gpa: std.mem.Allocator,
    bytes: []u8,
    allocations: []const Copy,
    queued: std.ArrayList(Landing) = .empty,
    pending: [2]std.ArrayList(Landing) = .{ .empty, .empty },

    const Landing = struct { dst: u64, src: []const u8 };

    fn deinit(x: *Image) void {
        x.queued.deinit(x.gpa);
        for (&x.pending) |*p| p.deinit(x.gpa);
    }

    fn copy(x: *Image, dst: u64, bytes: []const u8) !void {
        for (x.allocations) |a| {
            if (dst >= a.dst and dst + bytes.len <= a.dst + a.len) break;
        } else return error.CopySpansAllocations;
        try x.queued.append(x.gpa, .{ .dst = dst, .src = bytes });
    }

    fn done(x: *Image, slot: usize) !void {
        if (x.pending[slot].items.len != 0) return error.SlotReusedBeforeWait;
        std.mem.swap(std.ArrayList(Landing), &x.pending[slot], &x.queued);
    }

    fn wait(x: *Image, slot: usize) !void {
        x.land(slot);
    }

    fn land(x: *Image, slot: usize) void {
        for (x.pending[slot].items) |l| @memcpy(x.bytes[@intCast(l.dst)..][0..l.src.len], l.src);
        x.pending[slot].clearRetainingCapacity();
    }
};

test "plan cuts reads at the limit, spans small gaps and keeps every piece inside its copy" {
    const gpa = std.testing.allocator;
    var copies = [_]Copy{
        .{ .dst = 9000, .file = 0, .offset = 100, .len = 50 }, // abuts the dst 9050 copy in file and device: not joined
        .{ .dst = 1000, .file = 1, .offset = 0, .len = 10 },
        .{ .dst = 0, .file = 0, .offset = 0, .len = 100 },
        .{ .dst = 9050, .file = 0, .offset = 150, .len = 30 },
        .{ .dst = 5000, .file = 0, .offset = 180 + max_gap + 1, .len = 25 }, // past the gap: a read of its own
        .{ .dst = 2000, .file = 0, .offset = 180, .len = 1 },
    };
    const p = try plan(gpa, &copies, 64);
    defer p.deinit(gpa);
    try std.testing.expectEqualSlices(Read, &.{
        .{ .file = 0, .lo = 0, .len = 64, .first = 0, .end = 1 },
        .{ .file = 0, .lo = 64, .len = 64, .first = 1, .end = 3 },
        .{ .file = 0, .lo = 128, .len = 53, .first = 3, .end = 6 },
        .{ .file = 0, .lo = 180 + max_gap + 1, .len = 25, .first = 6, .end = 7 },
        .{ .file = 1, .lo = 0, .len = 10, .first = 7, .end = 8 },
    }, p.reads);
    try std.testing.expectEqualSlices(Copy, &.{
        .{ .dst = 0, .file = 0, .offset = 0, .len = 64 },
        .{ .dst = 64, .file = 0, .offset = 64, .len = 36 },
        .{ .dst = 9000, .file = 0, .offset = 100, .len = 28 },
        .{ .dst = 9028, .file = 0, .offset = 128, .len = 22 },
        .{ .dst = 9050, .file = 0, .offset = 150, .len = 30 },
        .{ .dst = 2000, .file = 0, .offset = 180, .len = 1 },
        .{ .dst = 5000, .file = 0, .offset = 180 + max_gap + 1, .len = 25 },
        .{ .dst = 1000, .file = 1, .offset = 0, .len = 10 },
    }, p.pieces);
}

test "every byte lands where the mapping would have put it, for any copies and read limit" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var prng: std.Random.DefaultPrng = .init(0x5747_4552);
    const random = prng.random();
    const sizes = [_]usize{ 3 * dio.alignment + 1234, 40000, 1 };
    var contents: [sizes.len][]u8 = undefined;
    var paths: [sizes.len][:0]u8 = undefined;
    var dir_buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    for (sizes, &contents, &paths, 0..) |size, *c, *path, i| {
        c.* = try gpa.alloc(u8, size);
        random.bytes(c.*);
        var name: [16]u8 = undefined;
        const base = try std.fmt.bufPrint(&name, "shard{d}", .{i});
        try tmp.dir.writeFile(io, .{ .sub_path = base, .data = c.* });
        path.* = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, base }, 0);
    }
    defer for (contents, paths) |c, path| {
        gpa.free(c);
        gpa.free(path);
    };
    var files: [sizes.len]dio.File = undefined;
    for (&files, paths) |*f, path| f.* = try dio.File.open(path);
    defer for (&files) |*f| f.close();
    const slots = [2]dio.Buffer{
        try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), 4 * dio.alignment),
        try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), 4 * dio.alignment),
    };
    defer for (slots) |s| gpa.free(s);

    for (0..200) |round| {
        // Random copies, overlapping, repeating or abutting, some longer than a read; expected is loader-order copying.
        const want = try gpa.alloc(u8, 1 << 18);
        defer gpa.free(want);
        const got = try gpa.alloc(u8, want.len);
        defer gpa.free(got);
        @memset(want, 0xA5);
        @memset(got, 0xA5);
        var copies: std.ArrayList(Copy) = .empty;
        defer copies.deinit(gpa);
        var dst: u64 = random.uintLessThan(u64, 64);
        for (0..1 + random.uintLessThan(usize, 40)) |_| {
            const next: ?Copy = if (copies.items.len > 0 and random.boolean()) blk: {
                const last = copies.items[copies.items.len - 1];
                break :blk if (last.offset + last.len < sizes[last.file]) last else null;
            } else null;
            const file = if (next) |c| c.file else random.uintLessThan(usize, sizes.len);
            const offset = if (next) |c| c.offset + c.len else random.uintLessThan(usize, sizes[file]);
            const len = 1 + random.uintLessThan(usize, @min(sizes[file] - offset, 3 * dio.alignment + 77));
            if (dst + len > want.len) break;
            try copies.append(gpa, .{ .dst = dst, .file = file, .offset = offset, .len = len });
            @memcpy(want[dst..][0..len], contents[file][offset..][0..len]);
            dst += len + if (random.boolean()) 0 else random.uintLessThan(u64, 100);
        }
        const limit = if (round % 2 == 0) dio.fits(4 * dio.alignment) else 1 + random.uintLessThan(usize, dio.fits(4 * dio.alignment));
        const p = try plan(gpa, copies.items, limit);
        defer p.deinit(gpa);
        for (p.reads) |r| try std.testing.expect(r.len <= limit and dio.span(r.lo, r.len) <= slots[0].len);
        var image: Image = .{ .gpa = gpa, .bytes = got, .allocations = copies.items };
        defer image.deinit();
        try run(&files, p, slots, &image);
        for (0..2) |k| image.land(k);
        try std.testing.expectEqualSlices(u8, want, got);
    }
}
