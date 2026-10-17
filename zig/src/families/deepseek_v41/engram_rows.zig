//! Engram table rows from packed per-rank shards on local NVMe, through io_uring, AIO or reader threads, LRU-cached.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const linux = std.os.linux;
const eh = @import("engram_host.zig");

pub const magic: u64 = 0x31344E4531565344; // "DSV41EN1"; file `<dir>/engram-l<layer>-r<rank>of<world>.bin`
pub const header_bytes: u64 = 4096; // 6 little-endian u64: magic, layer, lo, hi, total rows, row bytes
pub const sector: u64 = 512;
pub const row_bytes: u64 = 264; // 256 e4m3 values and 8 UE8M0 scales (one a 32)
pub const values = 256;
pub const merge_gap: u64 = 4096; // holes up to this merged into one extent
pub const page = 4096;
const engram_io = @import("engram_io.zig");

pub const Header = struct {
    layer: u64,
    lo: u64,
    hi: u64,
    total: u64,
    row_bytes: u64,

    /// The header's six words, checked against the file's size.
    pub fn parse(b: *const [48]u8, size: u64) !Header {
        var w: [6]u64 = undefined;
        for (&w, 0..) |*x, i| x.* = std.mem.readInt(u64, b[8 * i ..][0..8], .little);
        if (w[0] != magic) return error.NotEngramShard;
        const h: Header = .{ .layer = w[1], .lo = w[2], .hi = w[3], .total = w[4], .row_bytes = w[5] };
        if (h.row_bytes != row_bytes or h.hi < h.lo or size != header_bytes + (h.hi - h.lo) * h.row_bytes) return error.BadEngramShard;
        return h;
    }
};

/// torch's float8_e4m3fn -> float32 (c10 fp8e4m3fn_to_fp32_value, bit for bit).
pub fn e4m3(b: u8) f32 {
    const w: u32 = @as(u32, b) << 24;
    const sign = w & 0x8000_0000;
    const nonsign = w & 0x7FFF_FFFF;
    var shift: u32 = @clz(nonsign);
    shift = if (shift > 4) shift - 4 else 0;
    const inf_nan: u32 = @bitCast((@as(i32, @bitCast(nonsign +% 0x0100_0000)) >> 8) & 0x7F80_0000);
    const zero: u32 = @bitCast(@as(i32, @bitCast(nonsign -% 1)) >> 31);
    const r = sign | ((((nonsign << @intCast(shift)) >> 4) +% ((0x78 -% shift) << 23) | inf_nan) & ~zero);
    return @bitCast(r);
}

/// float32 -> bfloat16 as torch's CPU vector path: round to nearest even, NaN -> 0xFFFF (no packed table holds a NaN).
pub fn bf16(x: f32) u16 {
    if (std.math.isNan(x)) return 0xFFFF;
    const u: u32 = @bitCast(x);
    return @intCast((u +% (0x7FFF + ((u >> 16) & 1))) >> 16);
}

/// The table [scale << 8 | byte] -> bf16 bits of e4m3(byte) x 2^(scale - 127) (scale 0: x 0).
pub fn fillTable(t: *[65536]u16) void {
    for (0..256) |s| {
        const mul: f32 = @bitCast(@as(u32, @intCast(s)) << 23);
        for (0..256) |b| t[s << 8 | b] = bf16(e4m3(@intCast(b)) * mul);
    }
}

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
        r.* = .{ .gpa = gpa, .tables = try gpa.alloc(Table, layer_ids.len), .ring = null, .lut = try gpa.create([65536]u16), .cap = cap };
        fillTable(r.lut);
        var made: usize = 0;
        errdefer {
            for (r.tables[0..made]) |*t| r.closeTable(t);
            gpa.free(r.tables);
            gpa.destroy(r.lut);
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
        // every read in flight lands before its buffer goes
        for (r.batches.items, 0..) |*b, i| if (b.in_use) {
            r.wait(@intCast(i)) catch {};
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
    fn unique(r: *Rows, t: *const Table, idx: []const i64) !void {
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

    /// Finished batches into the caches (bulk read through); `li`: the table whose `rows` call settles (null: none).
    fn settle(r: *Rows, li: ?usize) void {
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
    pub fn rows(r: *Rows, li: usize, idx: []const i64, dim: usize, out: []u16) !void {
        if (dim > values or dim % 32 != 0 or out.len != idx.len * dim) return error.BadEngramRows;
        const t = &r.tables[li];
        try r.unique(t, idx);
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
