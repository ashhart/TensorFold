//! Checkpoint bytes read with O_DIRECT through io_uring, in flight while the loader allocates, then copied to the GPU.

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

pub const Source = struct {
    gpa: std.mem.Allocator,
    ops: kern.Ops,
    files: std.ArrayList(Mapped) = .empty,
    slots: [slot_count]dio.Buffer = undefined,
    pending: [slot_count]?Pending = @splat(null),
    in_flight: usize = 0,
    ring: ?linux.IoUring = null, // null where io_uring is unavailable: each read then completes as it is asked for

    /// Page-aligned pageable slots: a copy from them returns once it has read the slot, so it is free again.
    pub fn init(gpa: std.mem.Allocator, ops: kern.Ops) !Source {
        var s: Source = .{ .gpa = gpa, .ops = ops };
        var made: usize = 0;
        errdefer for (s.slots[0..made]) |b| gpa.free(b);
        while (made < slot_count) : (made += 1) s.slots[made] = try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), slot_bytes);
        s.ring = linux.IoUring.init(slot_count, 0) catch |err| blk: {
            std.log.info("io_uring unavailable ({t}): checkpoint reads run one at a time", .{err});
            break :blk null;
        };
        return s;
    }

    /// Waits for reads still in flight, then frees the ring and slots and closes the files.
    pub fn deinit(s: *Source) void {
        while (s.in_flight > 0) s.reap(1) catch break;
        if (s.ring) |*r| r.deinit();
        for (s.slots) |b| s.gpa.free(b);
        for (s.files.items) |*m| m.file.close();
        s.files.deinit(s.gpa);
        s.* = undefined;
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

    /// Puts slot `k`'s read in flight (or, without a ring, reads and copies it now).
    fn start(s: *Source, k: usize, p: Pending) !void {
        const lo = std.mem.alignBackward(u64, p.offset, dio.alignment);
        var q = p;
        q.at = @intCast(p.offset - lo);
        const r = if (s.ring) |*ring| ring else return s.finish(q, try p.file.read(s.slots[k], p.offset, p.len));
        _ = try r.read(k, p.file.fd, .{ .buffer = s.slots[k][0..dio.span(p.offset, p.len)] }, lo);
        _ = try r.submit();
        s.pending[k] = q;
        s.in_flight += 1;
    }

    /// Finished reads, copied on; waits for at least `wait` of them.
    fn reap(s: *Source, wait: u32) !void {
        const r = if (s.ring) |*ring| ring else return;
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
            try s.finish(p, got);
        }
    }

    /// cuMemcpyHtoDAsync plus cuStreamSynchronize returns only after the pageable source has been staged.
    fn finish(s: *Source, p: Pending, got: []u8) !void {
        if (got.len == 0) return;
        try s.ops.k.d.check(s.ops.k.d.api.cuMemcpyHtoDAsync_v2(p.dst, got.ptr, got.len, s.ops.s.handle), "cuMemcpyHtoDAsync");
        try s.ops.s.synchronize();
    }

    /// Mapped bytes copied into host memory `out` (its length is theirs).
    pub fn read(s: *Source, out: []u8, bytes: []const u8) !void {
        const at = s.locate(bytes) orelse return @memcpy(out, bytes);
        try s.flush();
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
        return at.file.read(s.slots[0], at.offset, bytes.len);
    }
};
