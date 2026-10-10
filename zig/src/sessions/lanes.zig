//! The NVMe tier's lane threads, which hash and write (or read and check) 4 KiB chunks, and their staging buffers.

const std = @import("std");
const linux = std.os.linux;
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const block = 4096;

pub fn up(n: u64) u64 {
    return (n + block - 1) / block * block;
}

/// Bounded, aligned staging buffers (allocated once; `release` gives the memory back while idle).
pub const Staging = struct {
    io: Io,
    size: usize,
    count: u32,
    bufs: [][]align(block) u8,
    free_list: []u32,
    free_n: u32,
    mutex: Io.Mutex = .init,
    cond: Io.Condition = .init,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, io: Io, count: u32, size: usize) !Staging {
        if (count == 0 or size % block != 0) return error.BadStaging;
        const bufs = try gpa.alloc([]align(block) u8, count);
        errdefer gpa.free(bufs);
        const fl = try gpa.alloc(u32, count);
        errdefer gpa.free(fl);
        for (bufs, 0..) |*b, i| {
            b.* = &.{};
            fl[i] = @intCast(i);
        }
        return .{ .io = io, .size = size, .count = count, .bufs = bufs, .free_list = fl, .free_n = count, .gpa = gpa };
    }

    pub fn deinit(s: *Staging) void {
        s.release();
        s.gpa.free(s.bufs);
        s.gpa.free(s.free_list);
        s.* = undefined;
    }

    /// A free buffer's index, waiting for one; its memory is mapped on first use.
    pub fn get(s: *Staging) !u32 {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        while (s.free_n == 0) s.cond.waitUncancelable(s.io, &s.mutex);
        s.free_n -= 1;
        const i = s.free_list[s.free_n];
        if (s.bufs[i].len == 0) s.bufs[i] = try s.gpa.alignedAlloc(u8, .fromByteUnits(block), s.size);
        return i;
    }

    pub fn put(s: *Staging, i: u32) void {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.free_list[s.free_n] = i;
        s.free_n += 1;
        s.cond.signal(s.io);
    }

    pub fn buf(s: *Staging, i: u32) []align(block) u8 {
        return s.bufs[i];
    }

    /// Unmaps every buffer (all must be free): the memory goes back between operations.
    pub fn release(s: *Staging) void {
        std.debug.assert(s.free_n == s.count);
        for (s.bufs) |*b| if (b.len != 0) {
            s.gpa.free(b.*);
            b.* = &.{};
        };
    }

    pub fn bytes(s: *const Staging) u64 {
        return @as(u64, s.count) * s.size;
    }
};

pub const Kind = enum { write, read };

/// One chunk's I/O: buf[0 .. span] at file offset off (span: the chunk padded to 4 KiB; data: the chunk's own bytes).
pub const Job = struct {
    kind: Kind,
    fd: linux.fd_t,
    staging: u32,
    data: u32,
    span: u32,
    off: u64,
    /// write: the chunk's digest goes here; read: the digest it must have
    digest: *[32]u8,
    op: *Op,
    /// the pump's tag for a finished read
    tag: u32 = 0,
};

/// A park's or restore's chunks in flight: how many are pending, the first error, and finished reads for the pump.
pub const Op = struct {
    io: Io,
    mutex: Io.Mutex = .init,
    cond: Io.Condition = .init,
    pending: u32 = 0,
    err: ?anyerror = null,
    /// finished reads (their jobs), drained by the pump
    done: std.ArrayList(Job) = .empty,
    gpa: std.mem.Allocator,
    /// write jobs hand their staging back themselves
    staging: *Staging,

    pub fn init(gpa: std.mem.Allocator, io: Io, staging: *Staging) !Op {
        var o: Op = .{ .io = io, .gpa = gpa, .staging = staging };
        try o.done.ensureTotalCapacity(gpa, staging.count);
        return o;
    }

    pub fn deinit(o: *Op) void {
        o.done.deinit(o.gpa);
    }

    fn finish(o: *Op, j: Job, err: ?anyerror) void {
        o.mutex.lockUncancelable(o.io);
        defer o.mutex.unlock(o.io);
        o.pending -= 1;
        if (err) |e| {
            if (o.err == null) o.err = e;
        }
        if (j.kind == .read) o.done.appendAssumeCapacity(j) else o.staging.put(j.staging);
        o.cond.broadcast(o.io);
    }

    /// The next finished read, waiting for one; null when none is pending.
    pub fn next(o: *Op) ?Job {
        o.mutex.lockUncancelable(o.io);
        defer o.mutex.unlock(o.io);
        while (o.done.items.len == 0 and o.pending > 0) o.cond.waitUncancelable(o.io, &o.mutex);
        return o.done.pop();
    }

    /// Waits for every pending job; returns the first error.
    pub fn wait(o: *Op) !void {
        o.mutex.lockUncancelable(o.io);
        defer o.mutex.unlock(o.io);
        while (o.pending > 0) o.cond.waitUncancelable(o.io, &o.mutex);
        if (o.err) |e| return e;
    }

    pub fn failed(o: *Op) ?anyerror {
        o.mutex.lockUncancelable(o.io);
        defer o.mutex.unlock(o.io);
        return o.err;
    }
};

/// The lane threads and their queue.
pub const Lanes = struct {
    io: Io,
    gpa: std.mem.Allocator,
    staging: *Staging,
    threads: []std.Thread,
    /// queued jobs: a ring as long as the staging set (every queued job holds a buffer)
    ring: []Job,
    head: usize = 0,
    len: usize = 0,
    mutex: Io.Mutex = .init,
    cond: Io.Condition = .init,
    stop: bool = false,
    /// seconds the lanes spent hashing and in I/O (summed over lanes)
    hash_ns: std.atomic.Value(u64) = .init(0),
    io_ns: std.atomic.Value(u64) = .init(0),

    pub fn start(gpa: std.mem.Allocator, io: Io, staging: *Staging, n: u32) !*Lanes {
        const l = try gpa.create(Lanes);
        errdefer gpa.destroy(l);
        const ring = try gpa.alloc(Job, staging.count);
        errdefer gpa.free(ring);
        l.* = .{ .io = io, .gpa = gpa, .staging = staging, .threads = try gpa.alloc(std.Thread, n), .ring = ring };
        errdefer gpa.free(l.threads);
        var made: usize = 0;
        errdefer {
            l.halt();
            for (l.threads[0..made]) |t| t.join();
        }
        for (l.threads) |*t| {
            t.* = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, run, .{l});
            made += 1;
        }
        return l;
    }

    fn halt(l: *Lanes) void {
        l.mutex.lockUncancelable(l.io);
        l.stop = true;
        l.cond.broadcast(l.io);
        l.mutex.unlock(l.io);
    }

    pub fn shutdown(l: *Lanes) void {
        l.halt();
        for (l.threads) |t| t.join();
        l.gpa.free(l.ring);
        l.gpa.free(l.threads);
        l.gpa.destroy(l);
    }

    /// Queues a job (each holds a staging buffer, so the ring never overflows).
    pub fn submit(l: *Lanes, j: Job) void {
        {
            j.op.mutex.lockUncancelable(l.io);
            defer j.op.mutex.unlock(l.io);
            j.op.pending += 1;
        }
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        std.debug.assert(l.len < l.ring.len);
        l.ring[(l.head + l.len) % l.ring.len] = j;
        l.len += 1;
        l.cond.signal(l.io);
    }

    fn run(l: *Lanes) void {
        while (true) {
            const j = blk: {
                l.mutex.lockUncancelable(l.io);
                defer l.mutex.unlock(l.io);
                while (l.len == 0 and !l.stop) l.cond.waitUncancelable(l.io, &l.mutex);
                if (l.len == 0) return;
                const j = l.ring[l.head];
                l.head = (l.head + 1) % l.ring.len;
                l.len -= 1;
                break :blk j;
            };
            j.op.finish(j, l.work(j));
        }
    }

    fn work(l: *Lanes, j: Job) ?anyerror {
        const b = l.staging.buf(j.staging);
        var t0 = std.Io.Clock.awake.now(l.io).toNanoseconds();
        switch (j.kind) {
            .write => {
                Sha256.hash(b[0..j.data], j.digest, .{});
                const t1 = std.Io.Clock.awake.now(l.io).toNanoseconds();
                _ = l.hash_ns.fetchAdd(@intCast(t1 - t0), .monotonic);
                t0 = t1;
                @memset(b[j.data..j.span], 0);
                pwriteAll(j.fd, b[0..j.span], j.off) catch |e| return e;
                _ = l.io_ns.fetchAdd(@intCast(std.Io.Clock.awake.now(l.io).toNanoseconds() - t0), .monotonic);
            },
            .read => {
                preadAll(j.fd, b[0..j.span], j.off) catch |e| return e;
                const t1 = std.Io.Clock.awake.now(l.io).toNanoseconds();
                _ = l.io_ns.fetchAdd(@intCast(t1 - t0), .monotonic);
                var d: [32]u8 = undefined;
                Sha256.hash(b[0..j.data], &d, .{});
                _ = l.hash_ns.fetchAdd(@intCast(std.Io.Clock.awake.now(l.io).toNanoseconds() - t1), .monotonic);
                if (!std.mem.eql(u8, &d, j.digest)) return error.ChecksumMismatch;
            },
        }
        return null;
    }
};

pub fn pwriteAll(fd: linux.fd_t, data: []const u8, off: u64) !void {
    var done: usize = 0;
    while (done < data.len) {
        const rc = linux.pwrite(fd, data[done..].ptr, data.len - done, @intCast(off + done));
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .INTR => continue,
            .NOSPC => return error.NoSpaceLeft,
            else => return error.WriteFailed,
        }
        if (rc == 0) return error.WriteFailed;
        done += rc;
    }
}

pub fn preadAll(fd: linux.fd_t, out: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < out.len) {
        const rc = linux.pread(fd, out[done..].ptr, out.len - done, @intCast(off + done));
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .INTR => continue,
            else => return error.ReadFailed,
        }
        if (rc == 0) return error.ShortRead;
        done += rc;
    }
}

test "lanes write, read back and check chunks through bounded staging" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var st = try Staging.init(gpa, io, 3, 2 * block);
    defer st.deinit();
    const l = try Lanes.start(gpa, io, &st, 2);
    defer l.shutdown();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const f = try tmp.dir.createFile(io, "lanes.bin", .{ .read = true });
    defer f.close(io);
    const fd = f.handle;
    var digests: [7][32]u8 = undefined;
    var op = try Op.init(gpa, io, &st);
    defer op.deinit();
    for (0..7) |i| {
        const s = try st.get();
        const n: u32 = if (i == 6) 1000 else 2 * block - 7;
        for (st.buf(s)[0..n], 0..) |*b, k| b.* = @truncate(k * 31 + i);
        l.submit(.{ .kind = .write, .fd = fd, .staging = s, .data = n, .span = @intCast(up(n)), .off = i * 2 * block, .digest = &digests[i], .op = &op });
    }
    try op.wait();
    var rop = try Op.init(gpa, io, &st);
    defer rop.deinit();
    var seen: u32 = 0;
    var sent: u32 = 0;
    while (seen < 7) {
        while (sent < 7 and sent - seen < 3) : (sent += 1) {
            const n: u32 = if (sent == 6) 1000 else 2 * block - 7;
            l.submit(.{ .kind = .read, .fd = fd, .staging = try st.get(), .data = n, .span = @intCast(up(n)), .off = sent * 2 * block, .digest = &digests[sent], .op = &rop, .tag = sent });
        }
        const j = rop.next().?;
        for (st.buf(j.staging)[0..j.data], 0..) |b, k| try std.testing.expectEqual(@as(u8, @truncate(k * 31 + j.tag)), b);
        st.put(j.staging);
        seen += 1;
    }
    try rop.wait();
    digests[2][0] ^= 1; // a damaged chunk is refused
    l.submit(.{ .kind = .read, .fd = fd, .staging = try st.get(), .data = 2 * block - 7, .span = 2 * block, .off = 2 * 2 * block, .digest = &digests[2], .op = &rop });
    st.put(rop.next().?.staging);
    try std.testing.expectError(error.ChecksumMismatch, rop.wait());
}
