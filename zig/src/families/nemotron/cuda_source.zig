//! Checkpoint bytes read with O_DIRECT through io_uring (or on threads where it is blocked), in flight while the loader allocates, then copied to the GPU.

const std = @import("std");
const linux = std.os.linux;
const cuda = @import("cuda");
const core = @import("core");
const kern = @import("cuda_kernels.zig");
const dio = core.direct_io;

/// Bytes a slot holds: one read's span; a larger tensor goes as several reads.
pub const slot_bytes = 64 << 20;
/// Slots, and so reads in flight at once: enough to cover the loader's largest allocation.
pub const slot_count = 4;

const Mapped = struct { base: usize, len: usize, file: dio.File };
/// A slot's read in flight: `len` bytes from `offset`, landing `at` into the slot, for the GPU at `dst`.
const Pending = struct { dst: u64, file: dio.File, offset: u64, len: usize, at: usize };
const ReadError = error{ BufferTooSmall, ReadFailed, EndOfFile };

/// A slot's read on its own thread where io_uring is unavailable: its bytes or error once `done`.
const Job = struct {
    file: dio.File,
    buf: dio.Buffer,
    offset: u64,
    len: usize,
    got: ReadError![]u8 = error.ReadFailed,
    done: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    fn run(j: *Job) void {
        j.got = j.file.read(j.buf, j.offset, j.len);
        j.done.store(true, .release);
    }
};

pub const Source = struct {
    gpa: std.mem.Allocator,
    ops: kern.Ops,
    files: std.ArrayList(Mapped) = .empty,
    slots: [slot_count]dio.Buffer = undefined,
    pinned: [slot_count]?cuda.HostBuffer = @splat(null), // a discrete card's slots, page-locked: copies run at the link's speed
    copied: [slot_count]?cuda.Event = @splat(null), // a page-locked slot's last copy: its next read waits for that, not the stream
    pending: [slot_count]?Pending = @splat(null),
    in_flight: usize = 0,
    ring: ?linux.IoUring = null, // null where io_uring is unavailable: each read then runs on its own thread
    jobs: [slot_count]Job = undefined,
    asked: [slot_count]u64 = @splat(0), // the order reads were asked in: copies follow it without a ring
    next: u64 = 0,

    /// Page-aligned slots; on a discrete card page-locked, each with an event its copies record.
    pub fn init(gpa: std.mem.Allocator, ops: kern.Ops) !Source {
        var s: Source = .{ .gpa = gpa, .ops = ops };
        var made: usize = 0;
        errdefer s.freeSlots(made);
        while (made < slot_count) : (made += 1) {
            if (!ops.k.discrete) {
                s.slots[made] = try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), slot_bytes);
                continue;
            }
            var h = try cuda.HostBuffer.alloc(ops.k.d, slot_bytes);
            if (@intFromPtr(h.bytes.ptr) % dio.alignment != 0) {
                h.free();
                return error.UnalignedPinnedSlot;
            }
            s.copied[made] = cuda.Event.init(ops.k.d, false) catch |err| {
                h.free();
                return err;
            };
            s.pinned[made] = h;
            s.slots[made] = @alignCast(h.bytes);
        }
        s.ring = linux.IoUring.init(slot_count, 0) catch |err| blk: {
            std.log.info("io_uring unavailable ({t}): checkpoint reads run on threads", .{err});
            break :blk null;
        };
        return s;
    }

    /// Waits for reads still in flight, then frees the ring and slots and closes the files.
    pub fn deinit(s: *Source) void {
        while (s.in_flight > 0) s.reap(1) catch break;
        if (s.ring) |*r| r.deinit() else s.joinAll();
        s.freeSlots(slot_count);
        for (s.files.items) |*m| m.file.close();
        s.files.deinit(s.gpa);
        s.* = undefined;
    }

    /// Threads still reading after a failed copy finish before their slots are freed.
    fn joinAll(s: *Source) void {
        for (&s.jobs, s.pending) |*j, p| if (p != null) if (j.thread) |t| t.join();
    }

    fn freeSlots(s: *Source, made: usize) void {
        for (s.slots[0..made], s.pinned[0..made], s.copied[0..made]) |b, *p, *e| {
            if (e.*) |*x| {
                x.synchronize() catch {};
                x.deinit();
            }
            if (p.*) |*h| h.free() else s.gpa.free(b);
        }
    }

    /// Slot k's last copy has read it, so new bytes can land there.
    fn settle(s: *Source, k: usize) !void {
        if (s.copied[k]) |e| try e.synchronize();
    }

    /// Every file of `ck`, opened for direct reads beside its mapping (copies from the mapping crawl on GB10).
    pub fn add(s: *Source, ck: *const core.Checkpoint) !void {
        for (ck.files.items) |*f| {
            var file = try dio.File.open(f.path);
            errdefer file.close();
            try s.files.append(s.gpa, .{ .base = @intFromPtr(f.map.memory.ptr), .len = f.map.memory.len, .file = file });
        }
    }

    /// The file and offset of mapped bytes; null for bytes that live elsewhere (host staging).
    fn locate(s: *const Source, bytes: []const u8) ?struct { file: dio.File, offset: u64 } {
        const p = @intFromPtr(bytes.ptr);
        for (s.files.items) |m| if (p >= m.base and p + bytes.len <= m.base + m.len) return .{ .file = m.file, .offset = p - m.base };
        return null;
    }

    /// Mapped bytes read for the GPU at `dst`: their reads are in flight on return; other bytes go now, after a flush.
    pub fn upload(s: *Source, dst: u64, bytes: []const u8) !void {
        const at = s.locate(bytes) orelse {
            try s.flush();
            return s.ops.upload(dst, bytes);
        };
        var done: usize = 0;
        while (done < bytes.len) {
            const n = @min(bytes.len - done, dio.fits(slot_bytes));
            try s.start(try s.free(), .{ .dst = dst + done, .file = at.file, .offset = at.offset + done, .len = n, .at = 0 });
            done += n;
        }
    }

    /// Waits for every read in flight and queues its copy: stream work after this runs after them.
    pub fn flush(s: *Source) !void {
        while (s.in_flight > 0) try s.reap(1);
    }

    /// A slot with no read in flight, once finished reads have been reaped (waiting for one when all are busy).
    fn free(s: *Source) !usize {
        try s.reap(0);
        while (true) {
            for (s.pending, 0..) |p, k| if (p == null) return k;
            try s.reap(1);
        }
    }

    /// Puts slot `k`'s read in flight: on the ring, else on its own thread (or here when no thread can start).
    fn start(s: *Source, k: usize, p: Pending) !void {
        try s.settle(k);
        const lo = std.mem.alignBackward(u64, p.offset, dio.alignment);
        var q = p;
        q.at = @intCast(p.offset - lo);
        if (s.ring) |*r| {
            _ = try r.read(k, p.file.fd, .{ .buffer = s.slots[k][0..dio.span(p.offset, p.len)] }, lo);
            _ = try r.submit();
        } else {
            const j = &s.jobs[k];
            j.* = .{ .file = p.file, .buf = s.slots[k], .offset = p.offset, .len = p.len };
            j.thread = std.Thread.spawn(.{}, Job.run, .{j}) catch null;
            if (j.thread == null) j.run();
        }
        s.pending[k] = q;
        s.asked[k] = s.next;
        s.next += 1;
        s.in_flight += 1;
    }

    /// Finished thread reads, oldest first, copied on; waits for at least `wait` of them.
    fn reapThreads(s: *Source, wait: u32) !void {
        var reaped: u32 = 0;
        while (s.in_flight > 0) : (reaped += 1) {
            var k: usize = slot_count;
            for (s.pending, 0..) |p, i| if (p != null and (k == slot_count or s.asked[i] < s.asked[k])) {
                k = i;
            };
            const j = &s.jobs[k];
            if (reaped >= wait and !j.done.load(.acquire)) return;
            if (j.thread) |t| t.join();
            j.thread = null;
            const p = s.pending[k].?;
            s.pending[k] = null;
            s.in_flight -= 1;
            try s.finish(k, p, try j.got);
        }
    }

    /// Finished reads, copied on; waits for at least `wait` of them.
    fn reap(s: *Source, wait: u32) !void {
        const r = if (s.ring) |*ring| ring else return s.reapThreads(wait);
        var cqes: [slot_count]linux.io_uring_cqe = undefined;
        const n = try r.copy_cqes(&cqes, wait);
        for (cqes[0..n]) |c| {
            const k: usize = @intCast(c.user_data);
            const p = s.pending[k].?;
            s.pending[k] = null;
            s.in_flight -= 1;
            // a short or failed read is finished by a plain read of the same span
            const got = if (c.res >= @as(i32, @intCast(p.at + p.len))) s.slots[k][p.at..][0..p.len] else blk: {
                if (c.res < 0) std.log.warn("io_uring read at {d} failed ({t}); reading again", .{ p.offset, c.err() });
                break :blk try p.file.read(s.slots[k], p.offset, p.len);
            };
            try s.finish(k, p, got);
        }
    }

    /// Slot k's bytes copied on: a page-locked slot records its event; a pageable one waits until the driver has staged it.
    fn finish(s: *Source, k: usize, p: Pending, got: []u8) !void {
        if (got.len == 0) return;
        try s.ops.k.d.check(s.ops.k.d.api.cuMemcpyHtoDAsync_v2(p.dst, got.ptr, got.len, s.ops.s.handle), "cuMemcpyHtoDAsync");
        if (s.copied[k]) |e| return e.record(s.ops.s);
        try s.ops.s.synchronize();
    }

    /// Mapped bytes copied into host memory `out` (its length is theirs).
    pub fn read(s: *Source, out: []u8, bytes: []const u8) !void {
        const at = s.locate(bytes) orelse return @memcpy(out, bytes);
        try s.flush();
        try s.settle(0);
        var done: usize = 0;
        while (done < bytes.len) {
            const n = @min(bytes.len - done, dio.fits(slot_bytes));
            @memcpy(out[done..][0..n], try at.file.read(s.slots[0], at.offset + done, n));
            done += n;
        }
    }

    /// Small mapped bytes as a view into a slot, valid until the next call; larger than a slot is refused.
    pub fn view(s: *Source, bytes: []const u8) ![]const u8 {
        const at = s.locate(bytes) orelse return bytes;
        if (bytes.len > dio.fits(slot_bytes)) return error.TensorLargerThanSlot;
        try s.flush();
        try s.settle(0);
        return at.file.read(s.slots[0], at.offset, bytes.len);
    }
};
