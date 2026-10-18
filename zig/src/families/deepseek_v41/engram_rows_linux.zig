//! Linux physical reader from @jayleaton's 08b, over shared row format and math.
const std = @import("std");
const quiet = @import("builtin").is_test;
const linux = std.os.linux;
const eh = @import("engram_host.zig");
const format = @import("engram_rows.zig");
pub const Header = format.Header;
const header_bytes = format.header_bytes;
const sector = format.sector;
const row_bytes = format.row_bytes;
const values = format.values;
const merge_gap = format.merge_gap;
const page = format.page;
const fillTable = format.fillTable;
const engram_io = @import("engram_io.zig");
const Batch = engram_io.Batch;
const Table = engram_io.Table;
const Pool = engram_io.Pool;
const Aio = engram_io.Aio;
pub const pollEnv = engram_io.pollEnv;
pub const aioEnv = engram_io.aioEnv;
pub const poolThreads = engram_io.poolThreads;
const nowNs = engram_io.nowNs;

pub const Stats = struct { hits: u64 = 0, waited: u64 = 0, misses: u64 = 0, issued: u64 = 0, reads: u64 = 0, bytes: u64 = 0, wait_ns: u64 = 0 };

pub const Rows = struct {
    gpa: std.mem.Allocator,
    tables: []Table,
    ring: ?linux.IoUring,
    /// no io_uring: the reader threads (null: serial reads on the engine thread)
    pool: ?*Pool = null,
    /// no io_uring, TF_DSV41_ENGRAM_AIO=1: native AIO (in place of the pool)
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
        errdefer {
            for (r.tables[0..made]) |*t| r.closeTable(t);
        }
        for (layer_ids, r.tables) |l, *t| {
            var nb: [4096]u8 = undefined;
            const p = try std.fmt.bufPrint(nb[0 .. nb.len - 1], "{s}/engram-l{d}-r{d}of{d}.bin", .{ dir, l, rank, world });
            nb[p.len] = 0;
            const path = nb[0..p.len :0];
            t.* = try openTable(gpa, path, cap);
            made += 1;
            if (t.h.layer != l) return error.BadEngramShard;
        }
        // TF_DSV41_ENGRAM_URING=0: the reader threads even where io_uring works (a container that allows it, A/B)
        const uring_off = if (std.c.getenv("TF_DSV41_ENGRAM_URING")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
        r.ring = (if (uring_off) error.PermissionDenied else linux.IoUring.init(256, 0)) catch |e| blk: {
            if (aioEnv()) {
                if (Aio.init()) |a| {
                    r.aio = a;
                    std.log.scoped(.dsv41).info("engram: no io_uring ({t}); rows read by native AIO (TF_DSV41_ENGRAM_AIO, {d} in flight), waits {s}", .{ e, Aio.depth, if (a.poll) "polled (TF_DSV41_ENGRAM_POLL)" else "asleep in the kernel" });
                    break :blk null;
                } else |ae| if (!quiet) std.log.scoped(.dsv41).warn("engram: TF_DSV41_ENGRAM_AIO=1 but io_setup failed ({t}): the reader threads", .{ae});
            }
            const n = poolThreads();
            if (n > 0) {
                try r.startPool(n);
                std.log.scoped(.dsv41).info("engram: no io_uring ({t}); rows read by {d} threads (TF_DSV41_ENGRAM_QD)", .{ e, n });
            } else if (!quiet) std.log.scoped(.dsv41).warn("engram: no io_uring ({t}) and TF_DSV41_ENGRAM_QD=0; rows are read when a window needs them", .{e});
            break :blk null;
        };
        return r;
    }

    pub const startAio = engram_io.startAio;
    pub const startPool = engram_io.startPool;
    pub const stopPool = engram_io.stopPool;
    const worker = engram_io.worker;
    const openTable = engram_io.openTable;
    const closeTable = engram_io.closeTable;

    pub fn close(r: *Rows) void {
        r.closeChecked() catch {}; // Failed drains retain the Rows owner, tables, fd and every destination.
    }
    pub fn closeChecked(r: *Rows) !void {
        // every read in flight lands before its buffer goes
        for (r.batches.items, 0..) |*b, i| if (b.in_use) {
            try r.drain(@intCast(i));
            b.read = true;
        };
        r.settle(null);
        r.stopPool();
        if (r.ring) |*g| g.deinit();
        if (r.aio) |*a| a.deinit();
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
    const start = engram_io.start;
    pub const slotFor = engram_io.slotFor;
    pub const queue = engram_io.queue;
    pub const readExtent = engram_io.readExtent;
    pub const aioSubmit = engram_io.aioSubmit;
    pub const aioReap = engram_io.aioReap;
    pub const reap = engram_io.reap;
    const wait = engram_io.wait;
    pub const drain = engram_io.drain;

    /// Finished batches into the caches (bulk read through); `li`: the table whose `rows` call settles (null: none).
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
