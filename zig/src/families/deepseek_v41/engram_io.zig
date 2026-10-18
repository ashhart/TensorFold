//! Engram shard reads: the batches, LRU tables and reader paths (io_uring, native AIO, a thread pool, serial).

const std = @import("std");
const linux = std.os.linux;
const engram_rows = @import("engram_rows.zig");
const Rows = engram_rows.Rows;
const Header = engram_rows.Header;
const header_bytes = engram_rows.header_bytes;
const sector = engram_rows.sector;
const row_bytes = engram_rows.row_bytes;
const merge_gap = engram_rows.merge_gap;
const page = engram_rows.page;

/// One extent of a batch: file bytes [off, off + len) into the batch buffer at `at`, its rows `first .. + count`.
const Extent = struct { off: u64, len: u64, at: usize, first: u32, count: u32 };

pub const Batch = struct {
    table: u32,
    /// sorted unique global rows
    rows: []u64,
    ext: []Extent,
    buf: []align(page) u8,
    left: u32,
    failed: bool = false,
    in_use: bool = true,
    /// a `rows` call took at least half its rows (a bulk batch is read through once read); `taken`: this call's count
    read: bool = false,
    taken: u32 = 0,
    unread: u8 = 0, // `rows` calls that settled it unread: a TF_DSV41_PREFETCH_AHEAD batch waits through one

    /// The record of `rows[k]` (k found by the caller).
    pub fn record(b: *const Batch, k: usize, lo: u64) []const u8 {
        var a: usize = 0;
        var z: usize = b.ext.len;
        while (z - a > 1) {
            const m = (a + z) / 2;
            if (b.ext[m].first <= k) a = m else z = m;
        }
        const e = b.ext[a];
        const at = e.at + @as(usize, @intCast(header_bytes + (b.rows[k] - lo) * row_bytes - e.off));
        return b.buf[at..][0..row_bytes];
    }
};

const Entry = union(enum) { ready: u32, pending: u32 };
const none: u32 = std.math.maxInt(u32);

/// One layer's shard and record cache.
pub const Table = struct {
    fd: linux.fd_t,
    direct: bool,
    h: Header,
    size: u64,
    map: std.AutoHashMapUnmanaged(u64, Entry) = .empty,
    /// cap records, their ids, and the LRU links (head: most recent)
    recs: []u8,
    ids: []u64,
    prev: []u32,
    next: []u32,
    used: u32 = 0,
    head: u32 = none,
    tail: u32 = none,
    /// keys removed since the map was last rehashed: each left a tombstone that a missing key's probe walks past
    dead: u32 = 0,

    pub fn unlink(t: *Table, s: u32) void {
        if (t.prev[s] != none) t.next[t.prev[s]] = t.next[s] else t.head = t.next[s];
        if (t.next[s] != none) t.prev[t.next[s]] = t.prev[s] else t.tail = t.prev[s];
    }

    pub fn front(t: *Table, s: u32) void {
        t.prev[s] = none;
        t.next[s] = t.head;
        if (t.head != none) t.prev[t.head] = s;
        t.head = s;
        if (t.tail == none) t.tail = s;
    }

    /// A slot for a new record: an unused one, else the least recent (its id leaves the map).
    pub fn take(t: *Table) u32 {
        if (t.used < t.ids.len) {
            t.used += 1;
            return t.used - 1;
        }
        const s = t.tail;
        t.unlink(s);
        t.drop(t.ids[s]);
        return s;
    }

    /// std's tombstones never clear and keys fill 80 %: rehash at a tenth, so a tenth of the map always stays free.
    pub fn drop(t: *Table, x: u64) void {
        if (!t.map.remove(x)) return;
        t.dead += 1;
        if (@as(u64, t.dead) * 10 < t.map.capacity()) return;
        t.map.rehash(std.hash_map.AutoContext(u64){});
        t.dead = 0;
    }
};

/// One extent for a pool thread: file bytes [off, off + buf.len) into `buf` (the batch's own buffer), batch `bi`.
const Job = struct { bi: u32, fd: linux.fd_t, size: u64, direct: bool, buf: []u8, off: u64 };

/// The reader threads without io_uring; `mu` guards the queue, every batch's `left` / `failed`, and the batch list.
pub const Pool = struct {
    mu: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    work: std.c.pthread_cond_t = .{},
    done: std.c.pthread_cond_t = .{},
    jobs: std.ArrayList(Job) = .empty,
    head: usize = 0,
    stop: bool = false,
    threads: []std.Thread = &.{},

    pub fn lock(p: *Pool) void {
        _ = std.c.pthread_mutex_lock(&p.mu);
    }

    pub fn unlock(p: *Pool) void {
        _ = std.c.pthread_mutex_unlock(&p.mu);
    }
};

/// Linux native AIO (TF_DSV41_ENGRAM_AIO=1): one io_submit a batch, reaped by the waiter; works without io_uring.
pub const Aio = struct {
    ctx: u64 = 0,
    in_flight: u32 = 0,
    poll: bool = false, // TF_DSV41_ENGRAM_POLL=1: a wait spins on io_getevents up to `poll_ns`, then sleeps

    pub const poll_ns: u64 = 20 * std.time.ns_per_ms;

    pub const depth: u32 = 1024;

    const Iocb = extern struct {
        data: u64,
        key: u32 = 0,
        rw_flags: i32 = 0,
        opcode: u16 = 0, // IOCB_CMD_PREAD
        reqprio: i16 = 0,
        fildes: u32,
        buf: u64,
        nbytes: u64,
        offset: i64,
        reserved2: u64 = 0,
        flags: u32 = 0,
        resfd: u32 = 0,
    };

    const Event = extern struct { data: u64, obj: u64, res: i64, res2: i64 };

    pub fn init() !Aio {
        var a: Aio = .{ .poll = pollEnv() };
        const rc = linux.syscall2(.io_setup, depth, @intFromPtr(&a.ctx));
        if (linux.errno(rc) != .SUCCESS) return error.AioRefused;
        return a;
    }

    pub fn deinit(a: *Aio) void {
        _ = linux.syscall1(.io_destroy, a.ctx);
    }
};

/// TF_DSV41_ENGRAM_POLL: 0 (default) | 1 (with TF_DSV41_ENGRAM_AIO=1).
pub fn pollEnv() bool {
    const v = std.c.getenv("TF_DSV41_ENGRAM_POLL") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_DSV41_ENGRAM_AIO: 0 (default) | 1.
pub fn aioEnv() bool {
    const v = std.c.getenv("TF_DSV41_ENGRAM_AIO") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// TF_DSV41_ENGRAM_QD: the reader threads without io_uring (default 64; 0: none, reads serial).
pub fn poolThreads() u32 {
    const v = std.c.getenv("TF_DSV41_ENGRAM_QD") orelse return 64;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch 64;
}

/// Native AIO in place of io_uring or the pool (tests, the probe): every read async through one context.
pub fn startAio(r: *Rows) !void {
    if (r.ring) |*g| g.deinit();
    r.ring = null;
    r.stopPool();
    if (r.aio == null) r.aio = try Aio.init();
}

/// The reader threads (no io_uring): `n` workers waiting for extents.
pub fn startPool(r: *Rows, n: u32) !void {
    if (r.pool != null or r.ring != null or n == 0) return error.EngramPool;
    const p = try r.gpa.create(Pool);
    errdefer r.gpa.destroy(p);
    p.* = .{};
    p.threads = try r.gpa.alloc(std.Thread, n);
    var made: usize = 0;
    errdefer {
        p.lock();
        p.stop = true;
        p.unlock();
        _ = std.c.pthread_cond_broadcast(&p.work);
        for (p.threads[0..made]) |t| t.join();
        r.gpa.free(p.threads);
    }
    r.pool = p;
    errdefer r.pool = null;
    for (p.threads) |*t| {
        t.* = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, worker, .{r});
        made += 1;
    }
}

/// The reader threads stopped (their queue drained); reads are serial afterwards.
pub fn stopPool(r: *Rows) void {
    const p = r.pool orelse return;
    p.lock();
    p.stop = true;
    p.unlock();
    _ = std.c.pthread_cond_broadcast(&p.work);
    for (p.threads) |t| t.join();
    r.gpa.free(p.threads);
    p.jobs.deinit(r.gpa);
    r.gpa.destroy(p);
    r.pool = null;
}

/// A reader thread: extents off the queue until `stop` (the queue drained first).
pub fn worker(r: *Rows) void {
    const p = r.pool.?;
    while (true) {
        p.lock();
        while (p.head == p.jobs.items.len and !p.stop) _ = std.c.pthread_cond_wait(&p.work, &p.mu);
        if (p.head == p.jobs.items.len) {
            p.unlock();
            return;
        }
        const j = p.jobs.items[p.head];
        p.head += 1;
        if (p.head == p.jobs.items.len) {
            p.jobs.clearRetainingCapacity();
            p.head = 0;
        }
        p.unlock();
        const ok = if (readAt(j.fd, j.size, j.direct, j.buf, j.off)) true else |_| false;
        p.lock();
        const b = &r.batches.items[j.bi];
        if (!ok) b.failed = true;
        b.left -= 1;
        const last = b.left == 0;
        p.unlock();
        if (last) _ = std.c.pthread_cond_broadcast(&p.done);
    }
}

pub fn openTable(gpa: std.mem.Allocator, path: [:0]const u8, cap: u32) !Table {
    var direct = true;
    var rc = linux.openat(linux.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .DIRECT = true }, 0);
    if (linux.errno(rc) != .SUCCESS) {
        direct = false;
        rc = linux.openat(linux.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
        if (linux.errno(rc) != .SUCCESS) {
            std.log.scoped(.dsv41).err("engram: cannot open {s}", .{path});
            return error.FileNotFound;
        }
    }
    const fd: linux.fd_t = @intCast(rc);
    errdefer _ = linux.close(fd);
    var st: linux.Statx = undefined;
    if (linux.errno(linux.statx(fd, "", linux.AT.EMPTY_PATH, .{ .SIZE = true }, &st)) != .SUCCESS) return error.StatFailed;
    const hb = try gpa.alignedAlloc(u8, .fromByteUnits(page), page);
    defer gpa.free(hb);
    try preadAll(fd, hb, 0);
    if (!direct) _ = linux.fadvise(fd, 0, page, linux.POSIX_FADV.DONTNEED);
    const h = try Header.parse(hb[0..48], st.size);
    return tableStorage(gpa, fd, direct, h, st.size, cap);
}
pub fn tableStorage(gpa: std.mem.Allocator, fd: linux.fd_t, direct: bool, h: Header, size: u64, cap: u32) !Table {
    const recs = try gpa.alloc(u8, try std.math.mul(usize, cap, row_bytes));
    errdefer gpa.free(recs);
    const ids = try gpa.alloc(u64, cap);
    errdefer gpa.free(ids);
    const prev = try gpa.alloc(u32, cap);
    errdefer gpa.free(prev);
    const next = try gpa.alloc(u32, cap);
    const t: Table = .{ .fd = fd, .direct = direct, .h = h, .size = size, .recs = recs, .ids = ids, .prev = prev, .next = next };
    return t;
}

pub fn closeTable(r: *Rows, t: *Table) void {
    _ = linux.close(t.fd);
    t.map.deinit(r.gpa);
    r.gpa.free(t.recs);
    r.gpa.free(t.ids);
    r.gpa.free(t.prev);
    r.gpa.free(t.next);
}

/// Recover failed destinations before callers classify cached/pending ids or remember reusable batch indices.
pub fn recover(r: anytype) !void {
    // A previously failed drain still owns its destinations and map entries until completion.
    if (r.pool == null) for (r.batches.items, 0..) |*b, i| if (b.in_use and b.failed) {
        try r.drain(@intCast(i));
        discard(r, @intCast(i));
    };
}

/// A batch for sorted unique `rows` of table `ti`, its extents submitted; `urgent`: runs before queued read-ahead.
pub fn start(r: anytype, ti: u32, rows_in: []const u64, urgent: bool) !u32 {
    try recover(r);
    const t = &r.tables[ti];
    const gpa = r.gpa;
    var ext: std.ArrayList(Extent) = .empty;
    errdefer ext.deinit(gpa);
    var total: usize = 0;
    for (rows_in, 0..) |x, k| {
        const s = header_bytes + (x - t.h.lo) * row_bytes;
        const s0 = s / sector * sector;
        const s1 = (s + row_bytes + sector - 1) / sector * sector;
        if (ext.items.len > 0) {
            const e = &ext.items[ext.items.len - 1];
            if (s0 <= e.off + e.len + merge_gap) {
                const end = @max(e.off + e.len, s1);
                total += @intCast(end - e.off - e.len);
                e.len = end - e.off;
                e.count += 1;
                continue;
            }
        }
        total = std.mem.alignForward(usize, total, page);
        try ext.append(gpa, .{ .off = s0, .len = s1 - s0, .at = total, .first = @intCast(k), .count = 1 });
        total += @intCast(s1 - s0);
    }
    // extents keep their buffer offsets page aligned (O_DIRECT)
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(page), std.mem.alignForward(usize, total, page));
    var published = false;
    errdefer if (!published) gpa.free(buf);
    const owned = try gpa.dupe(u64, rows_in);
    errdefer if (!published) gpa.free(owned);
    const extents = try ext.toOwnedSlice(gpa);
    errdefer if (!published) gpa.free(extents);
    const b: Batch = .{ .table = ti, .rows = owned, .ext = extents, .buf = buf, .left = 0 };
    try t.map.ensureUnusedCapacity(gpa, @intCast(owned.len));
    if (r.pool) |p| {
        // the batch's slot and its extents queued under the pool's lock (a worker settles batches.items[bi])
        p.lock();
        const queued = r.queue(p, t, b, urgent);
        p.unlock();
        const bi = try queued;
        published = true;
        _ = std.c.pthread_cond_broadcast(&p.work);
        for (owned) |x| t.map.putAssumeCapacity(x, .{ .pending = bi });
        return bi;
    }
    const bi = try r.slotFor(b);
    published = true;
    errdefer {
        r.batches.items[bi].failed = true;
        r.drain(bi) catch {};
        if (r.batches.items[bi].left == 0) discard(r, bi);
    }
    for (owned) |x| t.map.putAssumeCapacity(x, .{ .pending = bi });
    const bp = &r.batches.items[bi];
    r.stats.reads += bp.ext.len;
    for (bp.ext) |e| r.stats.bytes += e.len;
    if (@hasDecl(@TypeOf(r.*), "submitBatch")) try r.submitBatch(bi) else try submitBatch(r, bi);

    return bi;
}

fn submitBatch(r: *Rows, bi: u32) !void {
    const bp = &r.batches.items[bi];
    const t = &r.tables[bp.table];
    if (r.ring) |*g| {
        for (bp.ext, 0..) |e, k| {
            const sqe = g.read(@as(u64, bi) << 32 | k, t.fd, .{ .buffer = bp.buf[e.at..][0..@intCast(e.len)] }, e.off) catch blk: {
                // a full submission queue: submit what is queued, take finished reads, then the slot is free
                _ = try g.submit();
                try r.reap(0);
                break :blk try g.read(@as(u64, bi) << 32 | k, t.fd, .{ .buffer = bp.buf[e.at..][0..@intCast(e.len)] }, e.off);
            };
            _ = sqe;
            bp.left += 1;
        }
        _ = try g.submit();
    } else if (r.aio != null) {
        try r.aioSubmit(bi, t);
    } else {
        for (bp.ext) |e| try r.readExtent(t, bp.buf[e.at..][0..@intCast(e.len)], e.off);
    }
}

pub fn slotFor(r: anytype, b: Batch) !u32 {
    for (r.batches.items, 0..) |*x, i| if (!x.in_use) {
        x.* = b;
        return @intCast(i);
    };
    try r.batches.append(r.gpa, b);
    return @intCast(r.batches.items.len - 1);
}

/// A batch `b` in a free slot, its extents queued on `p` (locked): at the end, or `urgent` at the head.
pub fn queue(r: anytype, p: *Pool, t: *const Table, b: Batch, urgent: bool) !u32 {
    try p.jobs.ensureUnusedCapacity(r.gpa, b.ext.len);
    const bi = try r.slotFor(b);
    const bp = &r.batches.items[bi];
    r.stats.reads += bp.ext.len;
    const n = bp.ext.len;
    const at = if (urgent) p.head else p.jobs.items.len;
    if (urgent) {
        // one shift of the queued jobs past the head, then this batch's in their place
        const old = p.jobs.items.len;
        p.jobs.items.len += n;
        std.mem.copyBackwards(Job, p.jobs.items[at + n ..], p.jobs.items[at..old]);
    } else p.jobs.items.len += n;
    for (bp.ext, p.jobs.items[at..][0..n]) |e, *j| {
        r.stats.bytes += e.len;
        j.* = .{ .bi = bi, .fd = t.fd, .size = t.size, .direct = t.direct, .buf = bp.buf[e.at..][0..@intCast(e.len)], .off = e.off };
    }
    bp.left = @intCast(bp.ext.len);
    return bi;
}

pub fn readExtent(r: *Rows, t: *const Table, buf: []u8, off: u64) !void {
    _ = r;
    return readAt(t.fd, t.size, t.direct, buf, off);
}

/// Batch `bi`'s extents on the AIO context (all of them, reaping finished reads while the context is full).
pub fn aioSubmit(r: *Rows, bi: u32, t: *const Table) !void {
    const a = &r.aio.?;
    const bp = &r.batches.items[bi];
    var cbs: [64]Aio.Iocb = undefined;
    var ptrs: [64]*Aio.Iocb = undefined;
    var k: usize = 0;
    while (k < bp.ext.len) {
        if (a.in_flight == Aio.depth) try r.reap(1);
        const n = @min(bp.ext.len - k, cbs.len, Aio.depth - a.in_flight);
        for (cbs[0..n], ptrs[0..n], bp.ext[k..][0..n], 0..) |*cb, *pp, e, j| {
            cb.* = .{ .data = @as(u64, bi) << 32 | (k + j), .fildes = @intCast(t.fd), .buf = @intFromPtr(bp.buf[e.at..].ptr), .nbytes = e.len, .offset = @intCast(e.off) };
            pp.* = cb;
        }
        const rc = linux.syscall3(.io_submit, a.ctx, n, @intFromPtr(&ptrs));
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .AGAIN, .INTR => {
                try r.reap(if (a.in_flight > 0) 1 else 0);
                continue;
            },
            else => return error.EngramReadFailed,
        }
        bp.left += @intCast(rc);
        a.in_flight += @intCast(rc);
        k += rc;
    }
}

/// Takes finished AIO reads (waiting for at least `at_least`).
pub fn aioReap(r: *Rows, at_least: u32) !void {
    const a = &r.aio.?;
    var evs: [64]Aio.Event = undefined;
    var need = @min(at_least, a.in_flight);
    const zero: linux.timespec = .{ .sec = 0, .nsec = 0 };
    const t0 = if (a.poll and need > 0) nowNs() else 0;
    var polling = a.poll and need > 0;
    while (true) {
        // polling: whatever has landed, at once; else the kernel's wait for `need`
        if (polling and nowNs() - t0 > Aio.poll_ns) polling = false;
        const rc = if (polling)
            linux.syscall5(.io_getevents, a.ctx, 0, evs.len, @intFromPtr(&evs), @intFromPtr(&zero))
        else
            linux.syscall5(.io_getevents, a.ctx, need, evs.len, @intFromPtr(&evs), 0);
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .INTR => continue,
            else => return error.EngramReadFailed,
        }
        for (evs[0..rc]) |ev| {
            const b = &r.batches.items[@intCast(ev.data >> 32)];
            const e = b.ext[@intCast(ev.data & 0xFFFF_FFFF)];
            const t = &r.tables[b.table];
            const want = @min(e.len, t.size - e.off);
            if (ev.res < 0 or @as(u64, @intCast(ev.res)) < want) b.failed = true;
            if (!t.direct) _ = linux.fadvise(t.fd, @intCast(e.off), @intCast(e.len), linux.POSIX_FADV.DONTNEED);
            b.left -= 1;
            a.in_flight -= 1;
        }
        need -|= @intCast(rc);
        if (need == 0) return;
        if (polling and rc == 0) std.atomic.spinLoopHint();
    }
}

/// Takes finished reads off the ring (waiting for at least `at_least`).
pub fn reap(r: *Rows, at_least: u32) !void {
    if (r.aio != null) return r.aioReap(at_least);
    const g = &(r.ring orelse return);
    var cqes: [64]linux.io_uring_cqe = undefined;
    var need = at_least;
    while (true) {
        const n = try g.copy_cqes(&cqes, need);
        for (cqes[0..n]) |c| {
            const b = &r.batches.items[@intCast(c.user_data >> 32)];
            const e = b.ext[@intCast(c.user_data & 0xFFFF_FFFF)];
            const t = &r.tables[b.table];
            const want = @min(e.len, t.size - e.off);
            if (c.res < 0 or @as(u64, @intCast(c.res)) < want) b.failed = true;
            if (!t.direct) _ = linux.fadvise(t.fd, @intCast(e.off), @intCast(e.len), linux.POSIX_FADV.DONTNEED);
            b.left -= 1;
        }
        need -|= n;
        if (need == 0) return;
    }
}

/// Completes every accepted or queued request. Errors leave the full owner published for retry/close.
pub fn drain(r: *Rows, bi: u32) !void {
    if (r.pool) |p| {
        p.lock();
        defer p.unlock();
        while (r.batches.items[bi].left > 0) _ = std.c.pthread_cond_wait(&p.done, &p.mu);
        return;
    }
    if (r.ring) |*g| {
        // Include SQEs not accepted by a failed/partial submit in the completion barrier.
        while (g.sq_ready() > 0) _ = try g.submit();
    }
    while (r.batches.items[bi].left > 0) try r.reap(1);
}
/// Only a completed batch may relinquish its three arrays and pending map entries.
pub fn discard(r: anytype, bi: u32) void {
    const b = &r.batches.items[bi];
    std.debug.assert(b.in_use and b.left == 0);
    const t = &r.tables[b.table];
    for (b.rows) |x| if (t.map.get(x)) |entry| switch (entry) {
        .pending => |owner| if (owner == bi) {
            t.drop(x);
        },
        .ready => {},
    };
    r.gpa.free(b.rows);
    r.gpa.free(b.ext);
    r.gpa.free(b.buf);
    b.in_use = false;
}

/// Blocks until batch `bi`'s reads are done.
pub fn wait(r: *Rows, bi: u32) !void {
    if (r.pool) |p| {
        p.lock();
        defer p.unlock();
        while (r.batches.items[bi].left > 0) _ = std.c.pthread_cond_wait(&p.done, &p.mu);
        if (r.batches.items[bi].failed) return error.EngramReadFailed;
        return;
    }
    while (r.batches.items[bi].left > 0) try r.reap(1);
    if (r.batches.items[bi].failed) return error.EngramReadFailed;
}

/// An extent's bytes [off, off + buf.len) of a shard of `size` bytes (stops at its end), pages dropped if buffered.
pub fn readAt(fd: linux.fd_t, size: u64, direct: bool, buf: []u8, off: u64) !void {
    const want = @min(buf.len, size - off);
    var done: usize = 0;
    while (done < want) {
        const rc = linux.pread(fd, buf[done..].ptr, buf.len - done, @intCast(off + done));
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .INTR => continue,
            else => return error.EngramReadFailed,
        }
        if (rc == 0) return error.EngramShortRead;
        done += rc;
    }
    if (!direct) _ = linux.fadvise(fd, @intCast(off), @intCast(buf.len), linux.POSIX_FADV.DONTNEED);
}

pub fn preadAll(fd: linux.fd_t, out: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < out.len) {
        const rc = linux.pread(fd, out[done..].ptr, out.len - done, @intCast(off + done));
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .INTR => continue,
            else => return error.EngramReadFailed,
        }
        if (rc == 0) return error.EngramShortRead;
        done += rc;
    }
}

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
