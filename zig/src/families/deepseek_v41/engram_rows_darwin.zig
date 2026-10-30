//! Darwin physical reader: pread on N threads over merged extents, F_NOCACHE, no read-ahead.
const std = @import("std");
const c = std.c;
const posix = std.posix;
const linux = std.os.linux;
const quiet = @import("builtin").is_test;
const format = @import("engram_rows.zig");
const eh = @import("engram_host.zig");
const engram_io = @import("engram_io.zig");
const Batch = engram_io.Batch;
const Table = engram_io.Table;
const Pool = engram_io.Pool;
const Aio = engram_io.Aio;
const header_bytes = format.header_bytes;
const sector = format.sector;
const row_bytes = format.row_bytes;
const values = format.values;
const page = format.page;
const fillTable = format.fillTable;
const nowNs = engram_io.nowNs;

const F_NOCACHE: i32 = 48; // sys/fcntl.h: turn data caching off/on
const F_RDAHEAD: i32 = 45; // sys/fcntl.h: turn read-ahead off/on
const AT_FDCWD: i32 = -2; // fcntl.h: the process's working directory

pub const Stats = struct { hits: u64 = 0, waited: u64 = 0, misses: u64 = 0, issued: u64 = 0, reads: u64 = 0, bytes: u64 = 0, wait_ns: u64 = 0 };

pub const Rows = struct {
    gpa: std.mem.Allocator,
    tables: []Table,
    /// absent on Darwin: the field stays (engram_io and the probe reference it), always null
    ring: ?linux.IoUring = null,
    /// no io_uring: the reader threads (null: serial reads on the engine thread)
    pool: ?*Pool = null,
    /// no native AIO on Darwin: always null, startAio refuses
    aio: ?Aio = null,
    batches: std.ArrayList(Batch) = .empty,
    lut: *[65536]u16,
    cap: u32,
    stats: Stats = .{},
    /// scratch reused by `rows` (sorted unique ids, their records)
    uniq: std.ArrayList(u64) = .empty,
    recs: std.ArrayList([*]const u8) = .empty,
    want: std.ArrayList(u32) = .empty,

    /// Opens rank `rank` of `world`'s shard of every Engram layer in `layer_ids` under `dir`, `cap` records a layer.
    pub fn open(gpa: std.mem.Allocator, dir: []const u8, rank: u32, world: u32, layer_ids: []const u32, cap: u32) !*Rows {
        return openWithIo(gpa, std.Io.Threaded.global_single_threaded.io(), dir, rank, world, layer_ids, cap);
    }

    pub fn openWithIo(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, rank: u32, world: u32, layer_ids: []const u32, cap: u32) !*Rows {
        _ = io; // Darwin reads through its own pread on the reader threads, not std.Io
        if (cap < 4096) return error.EngramCacheTooSmall;
        const r = try gpa.create(Rows);
        errdefer gpa.destroy(r);
        const owned = try format.storage(Table, gpa, layer_ids.len);
        const tables = owned.tables;
        errdefer gpa.free(tables);
        const lut = owned.lut;
        errdefer gpa.destroy(lut);
        r.* = .{ .gpa = gpa, .tables = tables, .ring = null, .lut = lut, .cap = cap };
        fillTable(r.lut);
        var made: usize = 0;
        errdefer for (r.tables[0..made]) |*t| r.closeTable(t);
        for (layer_ids, r.tables) |l, *t| {
            var nb: [4096]u8 = undefined;
            const p = try std.fmt.bufPrint(nb[0 .. nb.len - 1], "{s}/engram-l{d}-r{d}of{d}.bin", .{ dir, l, rank, world });
            nb[p.len] = 0;
            const path = nb[0..p.len :0];
            t.* = try openTable(gpa, path, cap);
            made += 1;
            if (t.h.layer != l) return error.BadEngramShard;
        }
        const n = poolThreads();
        if (n > 0) {
            try r.startPool(n);
            if (!quiet) std.log.scoped(.dsv41).info("engram: rows read by {d} threads (TF_DSV41_ENGRAM_READERS)", .{n});
        } else if (!quiet) std.log.scoped(.dsv41).warn("engram: TF_DSV41_ENGRAM_READERS=0; rows are read when a window needs them", .{});
        return r;
    }

    pub fn close(r: *Rows) void {
        r.closeChecked() catch {};
    }
    pub fn closeChecked(r: *Rows) !void {
        // every read in flight lands before its buffer goes
        for (r.batches.items, 0..) |*b, i| if (b.in_use) {
            try r.drain(@intCast(i));
            b.read = true;
        };
        r.settle(null);
        r.stopPool();
        for (r.tables) |*t| r.closeTable(t);
        r.gpa.free(r.tables);
        r.batches.deinit(r.gpa);
        r.uniq.deinit(r.gpa);
        r.recs.deinit(r.gpa);
        r.want.deinit(r.gpa);
        r.gpa.destroy(r.lut);
        r.gpa.destroy(r);
    }

    /// engram_host's interface to these rows.
    pub fn source(r: *Rows) eh.Source {
        return .{ .ptr = r, .vtable = &.{ .issue = issueFn, .rows = rowsFn } };
    }
    fn issueFn(p: *anyopaque, li: usize, idx: []const i64) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.issue(li, idx);
    }
    fn rowsFn(p: *anyopaque, li: usize, idx: []const i64, dim: usize, out: []u16) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.rows(li, idx, dim, out);
    }

    /// Sorted unique `idx` into r.uniq (checked against the shard's range).
    pub fn unique(r: anytype, t: *const Table, idx: []const i64) !void {
        r.uniq.clearRetainingCapacity();
        try r.uniq.ensureTotalCapacity(r.gpa, idx.len);
        for (idx) |i| {
            if (i < 0 or @as(u64, @intCast(i)) < t.h.lo or @as(u64, @intCast(i)) >= t.h.hi) return error.EngramRowOutsideShard;
            r.uniq.appendAssumeCapacity(@intCast(i));
        }
        std.mem.sort(u64, r.uniq.items, {}, std.sort.asc(u64));
        var m: usize = 0;
        for (r.uniq.items) |x| if (m == 0 or r.uniq.items[m - 1] != x) {
            r.uniq.items[m] = x;
            m += 1;
        };
        r.uniq.shrinkRetainingCapacity(m);
    }

    /// Starts reading layer `li`'s rows `idx` that are neither cached nor in flight; returns at once.
    pub fn issue(r: *Rows, li: usize, idx: []const i64) !void {
        const t = &r.tables[li];
        try r.unique(t, idx);
        try engram_io.recover(r);
        var m: usize = 0;
        for (r.uniq.items) |x| if (!t.map.contains(x)) {
            r.uniq.items[m] = x;
            m += 1;
        };
        if (m == 0) return;
        r.stats.issued += m;
        _ = try r.start(@intCast(li), r.uniq.items[0..m], false);
    }

    // The batch / queue / LRU machinery is backend-neutral: the Linux reader's shared core.
    pub const start = engram_io.start;
    pub const slotFor = engram_io.slotFor;
    pub const queue = engram_io.queue;
    pub const stopPool = engram_io.stopPool;

    /// Blocks until batch `bi`'s reads are done (Darwin: only the reader threads complete them).
    pub fn wait(r: *Rows, bi: u32) !void {
        if (r.pool) |p| {
            p.lock();
            defer p.unlock();
            while (r.batches.items[bi].left > 0) _ = c.pthread_cond_wait(&p.done, &p.mu);
        }
        if (r.batches.items[bi].failed) return error.EngramReadFailed;
    }
    /// Completes batch `bi`'s reads (its buffer and map entries stay owned by the caller).
    pub fn drain(r: *Rows, bi: u32) !void {
        if (r.pool) |p| {
            p.lock();
            defer p.unlock();
            while (r.batches.items[bi].left > 0) _ = c.pthread_cond_wait(&p.done, &p.mu);
        }
    }
    /// Batch `bi`'s extents read serially on the calling thread (no pool: reads happen when a window needs them).
    pub fn submitBatch(r: *Rows, bi: u32) !void {
        const bp = &r.batches.items[bi];
        const t = &r.tables[bp.table];
        for (bp.ext) |e| try r.readExtent(t, bp.buf[e.at..][0..@intCast(e.len)], e.off);
    }

    /// Darwin has no native AIO or io_uring; only the reader threads (or serial) read rows.
    pub fn startAio(r: *Rows) !void {
        _ = r;
        return error.EngramAioUnavailable;
    }

    /// The reader threads: `n` workers each reading queued extents with pread.
    pub fn startPool(r: *Rows, n: u32) !void {
        if (r.pool != null or n == 0) return error.EngramPool;
        const p = try r.gpa.create(Pool);
        errdefer r.gpa.destroy(p);
        p.* = .{};
        p.threads = try r.gpa.alloc(std.Thread, n);
        var made: usize = 0;
        errdefer {
            p.lock();
            p.stop = true;
            p.unlock();
            _ = c.pthread_cond_broadcast(&p.work);
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
    /// A reader thread: one extent off the queue until `stop` (the queue drained first).
    fn worker(r: *Rows) void {
        const p = r.pool.?;
        while (true) {
            p.lock();
            while (p.head == p.jobs.items.len and !p.stop) _ = c.pthread_cond_wait(&p.work, &p.mu);
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
            const ok = readAt(j.fd, j.size, j.buf, j.off);
            p.lock();
            const b = &r.batches.items[j.bi];
            if (!ok) b.failed = true;
            b.left -= 1;
            const last = b.left == 0;
            p.unlock();
            if (last) _ = c.pthread_cond_broadcast(&p.done);
        }
    }

    pub fn openTable(gpa: std.mem.Allocator, path: [:0]const u8, cap: u32) !Table {
        const fd = c.openat(AT_FDCWD, path, c.O{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(u32, 0));
        if (fd < 0) {
            std.log.scoped(.dsv41).err("engram: cannot open {s}", .{path});
            return error.FileNotFound;
        }
        // uncached reads with read-ahead off: these are one-shot positional reads, not a stream
        _ = c.fcntl(fd, F_NOCACHE, @as(i32, 1));
        _ = c.fcntl(fd, F_RDAHEAD, @as(i32, 0));
        var st: posix.Stat = undefined;
        if (c.fstat(fd, &st) < 0) {
            _ = c.close(fd);
            return error.StatFailed;
        }
        const size = @as(u64, @intCast(st.size));
        const hb = try gpa.alignedAlloc(u8, .fromByteUnits(page), page);
        errdefer gpa.free(hb);
        try preadAll(fd, 48, hb[0..48], 0);
        const h = try format.Header.parse(hb[0..48], size);
        gpa.free(hb);
        return engram_io.tableStorage(gpa, fd, false, h, size, cap);
    }
    pub fn closeTable(r: *Rows, t: *Table) void {
        _ = c.close(t.fd);
        t.map.deinit(r.gpa);
        r.gpa.free(t.recs);
        r.gpa.free(t.ids);
        r.gpa.free(t.prev);
        r.gpa.free(t.next);
    }

    pub fn readExtent(r: *Rows, t: *const Table, buf: []u8, off: u64) !void {
        _ = r;
        if (!readAt(t.fd, t.size, buf, off)) return error.EngramReadFailed;
    }
    /// pread a shard's bytes [off, off+len); clamp to size (extents align past EOF) and a short read is false.
    fn readAt(fd: linux.fd_t, size: u64, buf: []u8, off: u64) bool {
        const want = @min(buf.len, @as(usize, @intCast(size - off)));
        var done: usize = 0;
        while (done < want) {
            const rc = c.pread(fd, buf[done..].ptr, want - done, @as(i64, @intCast(off + done)));
            if (rc < 0) {
                if (c.errno(rc) == .INTR) continue;
                return false;
            }
            if (rc == 0) return false;
            done += @as(usize, @intCast(rc));
        }
        return true;
    }
    fn preadAll(fd: linux.fd_t, size: u64, out: []u8, off: u64) !void {
        if (!readAt(fd, size, out, off)) return error.EngramShortRead;
    }

    /// Finished batches into the caches (bulk reads past cap / 4 read through).
    pub fn settle(r: anytype, li: ?usize) void {
        // the pool's workers write `left` / `failed`: settled under its lock (the batches' list stays put)
        if (r.pool) |p| p.lock();
        defer if (r.pool) |p| p.unlock();
        for (r.batches.items) |*b| {
            if (!b.in_use or b.left > 0) continue;
            const t = &r.tables[b.table];
            const keep = !b.failed and b.rows.len <= r.cap / 4;
            // a settle on another table never drops it: segments read layer 1 then 14, so the next 14 batch sees both
            if (!keep and !b.failed and !b.read and (b.unread < 1 or li != b.table)) {
                if (li == b.table) b.unread += 1;
                continue;
            }
            for (b.rows, 0..) |x, k| {
                if (!keep) {
                    t.drop(x);
                    continue;
                }
                const s = t.take();
                @memcpy(t.recs[s * row_bytes ..][0..row_bytes], b.record(k, t.h.lo));
                t.ids[s] = x;
                t.front(s);
                // the entry is this batch's (a row is in one batch at most; evictions only remove ready entries)
                t.map.getPtr(x).?.* = .{ .ready = s };
            }
            r.gpa.free(b.rows);
            r.gpa.free(b.ext);
            r.gpa.free(b.buf);
            b.in_use = false;
        }
    }

    /// Layer `li`'s rows `idx` (global row ids this rank owns, any order, duplicates allowed) as bf16 [idx.len, dim].
    pub fn rows(r: anytype, li: usize, idx: []const i64, dim: usize, out: []u16) !void {
        if (dim > values or dim % 32 != 0 or out.len != idx.len * dim) return error.BadEngramRows;
        const t = &r.tables[li];
        try r.unique(t, idx);
        try engram_io.recover(r);
        const u = r.uniq.items;
        try r.recs.resize(r.gpa, u.len);
        r.want.clearRetainingCapacity();
        // cached records now; in-flight ones after their batches; the rest read as one batch
        var miss: std.ArrayList(u64) = .empty;
        defer miss.deinit(r.gpa);
        for (u, 0..) |x, k| {
            if (t.map.get(x)) |e| switch (e) {
                .ready => |s| {
                    r.recs.items[k] = t.recs[s * row_bytes ..].ptr;
                    t.unlink(s);
                    t.front(s);
                    r.stats.hits += 1;
                },
                .pending => |bi| {
                    r.batches.items[bi].taken += 1;
                    if (std.mem.indexOfScalar(u32, r.want.items, bi) == null) try r.want.append(r.gpa, bi);
                    r.stats.waited += 1;
                },
            } else try miss.append(r.gpa, x);
        }
        if (miss.items.len > 0) {
            r.stats.misses += miss.items.len;
            const bi = try r.start(@intCast(li), miss.items, true);
            r.batches.items[bi].read = true;
            try r.want.append(r.gpa, bi);
        }
        const t0 = nowNs();
        for (r.want.items) |bi| {
            try r.wait(bi);
            const b = &r.batches.items[bi];
            if (2 * @as(usize, b.taken) >= b.rows.len) b.read = true;
            b.taken = 0;
        }
        r.stats.wait_ns += nowNs() - t0;
        for (u, 0..) |x, k| switch (t.map.get(x).?) {
            .ready => {},
            .pending => |bi| {
                const b = &r.batches.items[bi];
                r.recs.items[k] = b.record(std.sort.binarySearch(u64, b.rows, x, orderU64).?, t.h.lo).ptr;
            },
        };
        // every position: its record's values through the table (scale byte of each 32 values, then the e4m3 byte)
        for (idx, 0..) |i, p| {
            const rec = r.recs.items[std.sort.binarySearch(u64, u, @as(u64, @intCast(i)), orderU64).?];
            const o = out[p * dim ..][0..dim];
            for (o, 0..) |*v, j| v.* = r.lut[@as(usize, rec[values + j / 32]) << 8 | rec[j]];
        }
        r.settle(li);
    }
};

fn orderU64(a: u64, b: u64) std.math.Order {
    return std.math.order(a, b);
}

/// TF_DSV41_ENGRAM_READERS: the reader threads (default 16; 0: none, reads serial; at most 32).
pub fn poolThreads() u32 {
    const v = std.c.getenv("TF_DSV41_ENGRAM_READERS") orelse return 16;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch return 16;
    return @min(n, 32);
}
