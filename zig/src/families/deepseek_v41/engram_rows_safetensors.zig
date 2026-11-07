//! Engram rows from the release's safetensors shards, for hosts that cannot hold the tables resident.

// The release packs one Engram layer to a shard: F8_E4M3 weights and F8_E8M0 scales far apart in the file.

// A row is two positional reads, a bounded LRU keeps the window, and the mapped shard serves rows by address.

const std = @import("std");
const f = @import("engram_rows.zig");
const eh = @import("engram_host.zig");
const st = @import("core").safetensors;

/// One row's record: 256 E4M3 values and 8 UE8M0 scales, the portable reader's own layout.
const row_bytes: usize = @intCast(f.row_bytes);
const values: usize = @intCast(f.values);
const scales: usize = values / 32;

pub const Stats = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    issued: u64 = 0,
    /// The shard regions read: two a row (the weights and the scales).
    reads: u64 = 0,
    /// The positional reads actually issued: zero when the shard is mapped.
    syscalls: u64 = 0,
    bytes: u64 = 0,
};

const Entry = struct { id: u64 = std.math.maxInt(u64), age: u64 = 0, bytes: [row_bytes]u8 = undefined };

const Table = struct {
    file: std.Io.File,
    /// Row 0's first weight byte and first scale byte, absolute in the shard.
    weight_at: u64,
    scale_at: u64,
    rows: u64,
    entries: []Entry,
    /// The whole shard mapped read-only: the gather then reads the release's own bytes by address.
    map: ?std.Io.File.MemoryMap = null,
};

/// The e4m3 byte x UE8M0 scale -> bf16 table, the resident path's own `fillTable`.
const tab_entries: usize = 65536;

/// The release packs one Engram layer to a shard; this is the pair load.zig plans for (same names, same layers).
const Known = struct { layer: u32, file: []const u8 };
const known = [_]Known{
    .{ .layer = 1, .file = "model-00047-of-00048.safetensors" },
    .{ .layer = 14, .file = "model-00048-of-00048.safetensors" },
};

/// The shard's two Engram tensors: dtype, shape and where row 0 starts.
const Extent = struct { weight_at: u64, scale_at: u64, rows: u64 };

pub const Rows = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    tables: []Table,
    clock: u64 = 0,
    /// Rotates the residency probe's sample so it never re-samples pages it warmed itself.
    probe_seq: u64 = 0,
    stats: Stats = .{},
    /// TF_DSV41_ENGRAM_TRACE: one read line a window, for the read-path measurements.
    trace: bool = false,
    /// The e4m3 x scale -> bf16 table, built once: the scalar decode's own values, looked up.
    tab: []u16,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, rank: u32, world: u32, layers: []const u32, cap: u32) !*Rows {
        return openWithMap(a, io, dir, rank, world, layers, cap, map_enabled());
    }

    /// The same reader with the map chosen by the caller.
    pub fn openWithMap(a: std.mem.Allocator, io: std.Io, dir: []const u8, rank: u32, world: u32, layers: []const u32, cap: u32, use_map: bool) !*Rows {
        if (world == 0 or rank >= world or cap == 0 or layers.len == 0) return error.BadEngramShard;
        const r = try a.create(Rows);
        errdefer a.destroy(r);
        const tables = try a.alloc(Table, layers.len);
        errdefer a.free(tables);
        var made: usize = 0;
        errdefer for (tables[0..made]) |*t| {
            if (t.map) |*mm| mm.destroy(io);
            t.file.close(io);
            a.free(t.entries);
        };
        const tab = try a.alloc(u16, tab_entries);
        errdefer a.free(tab);
        f.fillTable(@as(*[tab_entries]u16, @ptrCast(tab.ptr)));
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        for (layers, tables) |layer, *t| {
            const shard_name = for (known) |k| {
                if (k.layer == layer) break k.file;
            } else return error.UnknownEngramLayer;
            const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, shard_name });
            defer a.free(path);
            const file = try std.Io.Dir.cwd().openFile(io, path, .{});
            errdefer file.close(io);
            const size = try file.length(io);
            const ext = try extent(a, arena.allocator(), file, io, size, layer);
            const entries = try a.alloc(Entry, cap);
            @memset(entries, .{});
            t.* = .{ .file = file, .weight_at = ext.weight_at, .scale_at = ext.scale_at, .rows = ext.rows, .entries = entries };
            // The shard mapped read-only: the gather reads the release's own bytes by address.
            if (use_map) {
                t.map = std.Io.File.MemoryMap.create(io, file, .{
                    .len = @intCast(size),
                    .protection = .{ .read = true, .write = false },
                    .populate = false,
                }) catch |e| blk: {
                    std.log.scoped(.dsv41).warn("engram: the shard could not be mapped ({t}); the positional reads stay", .{e});
                    break :blk null;
                };
            }
            made += 1;
        }
        r.* = .{ .gpa = a, .io = io, .tables = tables, .trace = std.c.getenv("TF_DSV41_ENGRAM_TRACE") != null, .tab = tab };
        return r;
    }

    /// Whether the shard is mapped: the gather then reads by address, with no syscall a row.
    pub fn mapped(r: *Rows) bool {
        for (r.tables) |t| if (t.map != null) return true;
        return false;
    }

    /// TF_DSV41_ENGRAM_MMAP=0 restores the positional reads; the map is the default.
    fn map_enabled() bool {
        const v = std.mem.span(std.c.getenv("TF_DSV41_ENGRAM_MMAP") orelse return true);
        return !(v.len > 0 and v[0] == '0');
    }

    /// F_NOCACHE keeps such a read out of the cache, so the forward does not pay it back.
    pub const nocache_ids: usize = 256 * 1024;
    fn noCache() bool {
        const v = std.mem.span(std.c.getenv("TF_DSV41_ENGRAM_NOCACHE") orelse return true);
        return !(v.len > 0 and v[0] == '0');
    }

    /// The positional gather's own worker count.
    fn posThreads() usize {
        var v: usize = 32;
        if (std.c.getenv("TF_DSV41_ENGRAM_POS_THREADS")) |e| {
            v = std.fmt.parseInt(usize, std.mem.span(e), 10) catch v;
            if (v == 0 or v > 64) v = 32;
        }
        return v;
    }

    /// Turn the shard's cache bypass on or off for the gather that is about to run.
    fn setNoCache(r: *Rows, li: usize, on: bool) void {
        if (!noCache()) return;
        if (li >= r.tables.len) return;
        _ = std.c.fcntl(r.tables[li].file.handle, std.c.F.NOCACHE, @as(c_int, if (on) 1 else 0));
    }

    fn extent(a: std.mem.Allocator, arena: std.mem.Allocator, file: std.Io.File, io: std.Io, size: u64, layer: u32) !Extent {
        if (size < 8) return error.BadSafetensors;
        var head: [8]u8 = undefined;
        if (try file.readPositionalAll(io, &head, 0) != 8) return error.ShortRead;
        const header_bytes: usize = @intCast(std.mem.readInt(u64, &head, .little));
        if (header_bytes > @min(@as(usize, @intCast(size - 8)), 64 << 20)) return error.BadSafetensors;
        const json = try a.alloc(u8, header_bytes);
        defer a.free(json);
        if (try file.readPositionalAll(io, json, 8) != header_bytes) return error.ShortRead;
        const data_at: u64 = 8 + header_bytes;
        const names = try st.parseHeaderPrefix(arena, json, @intCast(size - data_at), null);
        var buf: [128]u8 = undefined;
        const w = names.get(try std.fmt.bufPrint(&buf, "layers.{d}.engram.embed.weight", .{layer})) orelse return error.MissingTensor;
        const s = names.get(try std.fmt.bufPrint(&buf, "layers.{d}.engram.embed.scale", .{layer})) orelse return error.MissingTensor;
        if (w.dtype != .f8_e4m3 or w.rank != 2 or w.shape[1] != values) return error.UnexpectedEngramTensor;
        if (s.dtype != .f8_e8m0 or s.rank != 2 or s.shape[1] != scales) return error.UnexpectedEngramTensor;
        if (s.shape[0] != w.shape[0]) return error.UnexpectedEngramTensor;
        if (w.end - w.begin < w.shape[0] * values or s.end - s.begin < s.shape[0] * scales) return error.UnexpectedEngramTensor;
        return .{ .weight_at = data_at + w.begin, .scale_at = data_at + s.begin, .rows = w.shape[0] };
    }

    pub fn close(r: *Rows) void {
        for (r.tables) |*t| {
            if (t.map) |*mm| mm.destroy(r.io);
            t.file.close(r.io);
            r.gpa.free(t.entries);
        }
        r.gpa.free(r.tab);
        r.gpa.free(r.tables);
        r.gpa.destroy(r);
    }

    fn validate(r: *Rows, li: usize, ids: []const i64) !void {
        if (li >= r.tables.len) return error.BadEngramShard;
        const total = r.tables[li].rows;
        for (ids) |id| if (id < 0 or @as(u64, @intCast(id)) >= total) return error.EngramRowOutsideShard;
    }

    fn record(r: *Rows, li: usize, id: u64) !*[row_bytes]u8 {
        const table = &r.tables[li];
        r.clock +%= 1;
        var slot = &table.entries[0];
        for (table.entries) |*entry| {
            if (entry.id == id) {
                entry.age = r.clock;
                r.stats.hits += 1;
                return &entry.bytes;
            }
            if (entry.age < slot.age) slot = entry;
        }
        // Publish the cache identity only after both reads succeed: a torn row is never served.
        var bytes: [row_bytes]u8 = undefined;
        if (table.map) |*mm| {
            // The mapped shard: the release's own bytes, copied by address. No syscall a row.
            const base = mm.memory;
            @memcpy(bytes[0..values], base[table.weight_at + id * values ..][0..values]);
            @memcpy(bytes[values..][0..scales], base[table.scale_at + id * scales ..][0..scales]);
            r.stats.reads += 2;
        } else {
            if (try table.file.readPositionalAll(r.io, bytes[0..values], table.weight_at + id * values) != values) return error.ShortRead;
            if (try table.file.readPositionalAll(r.io, bytes[values..][0..scales], table.scale_at + id * scales) != scales) return error.ShortRead;
            r.stats.reads += 2;
            r.stats.syscalls += 2;
        }
        slot.* = .{ .id = id, .age = r.clock, .bytes = bytes };
        r.stats.misses += 1;
        r.stats.bytes += row_bytes;
        return &slot.bytes;
    }

    /// One row's bf16 values, decoded from the mapped shard through the table.
    fn decodeMapped(r: *Rows, li: usize, id: u64, out: []align(1) u16) void {
        const t = &r.tables[li];
        const base = t.map.?.memory;
        const w = base[t.weight_at + id * values ..][0..values];
        const s = base[t.scale_at + id * scales ..][0..scales];
        const tab = r.tab;
        for (0..values) |j| out[j] = tab[(@as(u16, s[j / 32]) << 8) | w[j]];
    }

    /// Start the page-in for every row of a window before any row is decoded.
    pub fn prefetch(r: *Rows, li: usize, ids: []const u32) void {
        if (li >= r.tables.len) return;
        const t = &r.tables[li];
        const mm = t.map orelse return;
        const base = mm.memory;
        const page = std.heap.page_size_min;
        const start = @intFromPtr(base.ptr);
        for (ids) |id| {
            for ([_]u64{ t.weight_at + @as(u64, id) * values, t.scale_at + @as(u64, id) * scales }) |off| {
                if (off >= base.len) continue;
                const aligned = (start + off) & ~(page - 1);
                const len = @min(page, start + base.len - aligned);
                const at: [*]align(page) u8 = @ptrFromInt(aligned);
                std.posix.madvise(at, len, std.c.MADV.WILLNEED) catch {};
            }
        }
    }

    /// The bulk gather over the mapped shard: ids ascending, so the shard is walked forward and repeats are adjacent.
    fn rowsMapped(r: *Rows, li: usize, ids: []const u32, out: []align(1) u16, threads: usize) !void {
        const gpa = r.gpa;
        const order = try gpa.alloc(u32, ids.len);
        defer gpa.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, order, ids, struct {
            fn lessThan(ctx: []const u32, a: u32, b: u32) bool {
                return ctx[a] < ctx[b];
            }
        }.lessThan);
        const Counters = struct { hits: u64 = 0, misses: u64 = 0, reads: u64 = 0, bytes: u64 = 0 };
        const workers = @max(1, @min(threads, ids.len));
        const per = (ids.len + workers - 1) / workers;
        const Ctx = struct {
            r: *Rows,
            li: usize,
            ids: []const u32,
            order: []const u32,
            out: []align(1) u16,
            fn run(c: *@This(), from: usize, to: usize, counters: *Counters) void {
                var prev: u64 = std.math.maxInt(u64);
                for (c.order[from..to]) |i| {
                    const id: u64 = c.ids[i];
                    if (id == prev) {
                        counters.hits += 1;
                    } else {
                        counters.misses += 1;
                        counters.reads += 2;
                        counters.bytes += row_bytes;
                        prev = id;
                    }
                    c.r.decodeMapped(c.li, id, c.out[i * values ..][0..values]);
                }
            }
        };
        var ctx = Ctx{ .r = r, .li = li, .ids = ids, .order = order, .out = out };
        var counters: [64]Counters = undefined;
        for (&counters) |*c| c.* = .{};
        if (workers == 1) {
            ctx.run(0, ids.len, &counters[0]);
        } else {
            var handles: [64]std.Thread = undefined;
            var spawned: usize = 0;
            var w: usize = 0;
            while (w < workers) : (w += 1) {
                const from = w * per;
                const to = @min(ids.len, from + per);
                if (from >= to) continue;
                handles[spawned] = std.Thread.spawn(.{}, Ctx.run, .{ &ctx, from, to, &counters[w] }) catch break;
                spawned += 1;
            }
            while (w < workers) : (w += 1) {
                const from = w * per;
                const to = @min(ids.len, from + per);
                if (from >= to) continue;
                ctx.run(from, to, &counters[w]);
            }
            for (handles[0..spawned]) |h| h.join();
        }
        for (counters[0..workers]) |c| {
            r.stats.hits += c.hits;
            r.stats.misses += c.misses;
            r.stats.reads += c.reads;
            r.stats.bytes += c.bytes;
        }
    }

    /// Synchronous prefetch: the Source contract permits completion before return.
    pub fn issue(r: *Rows, li: usize, ids: []const i64) !void {
        try r.validate(li, ids);
        for (ids) |id| {
            _ = try r.record(li, @intCast(id));
            r.stats.issued += 1;
        }
    }

    /// The GPU's own u32 row ids straight in: no widening copy a window.
    pub fn rowsU32(r: *Rows, li: usize, ids: []const u32, dim: usize, out: []align(1) u16) !void {
        if (li >= r.tables.len) return error.BadEngramShard;
        if (dim != values or out.len != try std.math.mul(usize, ids.len, values)) return error.BadEngramShard;
        const total = r.tables[li].rows;
        for (ids) |id| if (id >= total) return error.EngramRowOutsideShard;
        // The mapped shard: the same rows, read by address and decoded through the table.
        if (r.tables[li].map != null) return r.rowsMapped(li, ids, out, 1);
        r.setNoCache(li, ids.len >= nocache_ids);
        // A chunk's ids are scattered over the shard: read them ascending so it is walked forward, not seeked.
        if (ids.len >= 64) {
            const gpa = r.gpa;
            const order = try gpa.alloc(u32, ids.len);
            defer gpa.free(order);
            for (order, 0..) |*o, i| o.* = @intCast(i);
            std.mem.sort(u32, order, ids, struct {
                fn lessThan(ctx: []const u32, a: u32, b: u32) bool {
                    return ctx[a] < ctx[b];
                }
            }.lessThan);
            for (order) |i| {
                const b = try r.record(li, ids[i]);
                for (0..values) |j| out[i * values + j] = f.bf16(f.e4m3(b[j]) * f.scale(b[values + j / 32]));
            }
            return;
        }
        for (ids, 0..) |id, i| {
            const b = try r.record(li, id);
            for (0..values) |j| out[i * values + j] = f.bf16(f.e4m3(b[j]) * f.scale(b[values + j / 32]));
        }
    }

    /// Is this mapped gather cold? TF_DSV41_ENGRAM_LARGE=map|pos forces either path; unset is the probe.
    fn coldGather(r: *Rows, li: usize, ids: []const u32, order: []const u32) bool {
        const S = struct {
            var mode: i8 = 0;
        };
        if (S.mode == 0) S.mode = if (std.c.getenv("TF_DSV41_ENGRAM_LARGE")) |e| blk: {
            const v = std.mem.span(e);
            if (std.mem.eql(u8, v, "map")) break :blk -1;
            if (std.mem.eql(u8, v, "pos")) break :blk 2;
            break :blk 1;
        } else 1;
        if (ids.len < 512) return false; // the decode window keeps its own prefetch and the map
        if (S.mode < 0) return false;
        if (S.mode == 2) return true;
        if (ids.len < 64 * 1024) return false; // the rows can stay resident: the warm case is real
        const t = &r.tables[li];
        const mm = t.map orelse return false;
        const base = mm.memory;
        const n = ids.len;
        const step = @max(1, n / 64);
        // The signal is the sampled reads' own latency, not the kernel's fault classification.
        var ts0: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts0);
        const t0: u64 = @as(u64, @intCast(ts0.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts0.nsec));
        if (r.probe_seq == 0) {
            // Seed from the clock: the page cache outlives the process.
            var tss: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &tss);
            r.probe_seq = @as(u64, @intCast(tss.nsec)) *% 2654435761 +% @as(u64, @intCast(tss.sec));
        }
        r.probe_seq +%= 1;
        const start = (r.probe_seq *% 7919) % step;
        var j: usize = start;
        var touched: usize = 0;
        var slowest: u64 = 0;
        while (j < n and touched < 64) : (j += step) {
            const id: u64 = ids[order[j]];
            const off = t.weight_at + id * values;
            if (off >= base.len) continue;
            var a: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &a);
            const ta: u64 = @as(u64, @intCast(a.sec)) * std.time.ns_per_s + @as(u64, @intCast(a.nsec));
            const p: *const volatile u8 = @ptrCast(&base[off]);
            _ = p.*;
            var b: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &b);
            const tb: u64 = @as(u64, @intCast(b.sec)) * std.time.ns_per_s + @as(u64, @intCast(b.nsec));
            slowest = @max(slowest, tb - ta);
            touched += 1;
        }
        var ts1: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts1);
        const t1: u64 = @as(u64, @intCast(ts1.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts1.nsec));
        if (touched == 0) return false;
        const per: u64 = (t1 - t0) / touched;
        // Strict: a list is warm only if NO sample had to leave the CPU.
        const cold = slowest > 20_000;
        if (std.c.getenv("TF_DSV41_HOST_PROF") != null)
            std.log.info("engram large probe: li {d}, ids {d}, samples {d}, mean {d} ns, slowest {d} ns -> {s}", .{ li, ids.len, touched, per, slowest, if (cold) "positional" else "mapped" });
        return cold;
    }

    /// The same gather with the id list split across threads.
    pub fn rowsU32Parallel(r: *Rows, li: usize, ids: []const u32, dim: usize, out: []align(1) u16, threads: usize) !void {
        if (threads <= 1) return r.rowsU32(li, ids, dim, out);
        // The model decides by timing, so a warm replay stays single.
        if (ids.len < 512 and (li >= r.tables.len or r.tables[li].map == null)) return r.rowsU32(li, ids, dim, out);
        if (li >= r.tables.len) return error.BadEngramShard;
        if (dim != values or out.len != try std.math.mul(usize, ids.len, values)) return error.BadEngramShard;
        const total = r.tables[li].rows;
        for (ids) |id| if (id >= total) return error.EngramRowOutsideShard;
        const gpa = r.gpa;
        const order = try gpa.alloc(u32, ids.len);
        defer gpa.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, order, ids, struct {
            fn lessThan(ctx: []const u32, a: u32, b: u32) bool {
                return ctx[a] < ctx[b];
            }
        }.lessThan);
        r.setNoCache(li, ids.len >= nocache_ids);
        // The mapped shard: the threads read by address.
        if (r.tables[li].map != null and !r.coldGather(li, ids, order)) return r.rowsMapped(li, ids, out, threads);
        const workers = @min(@max(threads, posThreads()), ids.len);
        const per = (ids.len + workers - 1) / workers;
        const Ctx = struct {
            r: *Rows,
            li: usize,
            ids: []const u32,
            order: []const u32,
            out: []align(1) u16,
            failures: *std.atomic.Value(usize),
            fn run(c: *const @This(), from: usize, to: usize) void {
                var bytes: [row_bytes]u8 = undefined;
                for (c.order[from..to]) |i| {
                    const table = &c.r.tables[c.li];
                    const id: u64 = c.ids[i];
                    const got = table.file.readPositionalAll(c.r.io, bytes[0..values], table.weight_at + id * values) catch {
                        _ = c.failures.fetchAdd(1, .monotonic);
                        return;
                    };
                    if (got != values) {
                        _ = c.failures.fetchAdd(1, .monotonic);
                        return;
                    }
                    const got_scales = table.file.readPositionalAll(c.r.io, bytes[values..][0..scales], table.scale_at + id * scales) catch {
                        _ = c.failures.fetchAdd(1, .monotonic);
                        return;
                    };
                    if (got_scales != scales) {
                        _ = c.failures.fetchAdd(1, .monotonic);
                        return;
                    }
                    for (0..values) |j| c.out[i * values + j] = f.bf16(f.e4m3(bytes[j]) * f.scale(bytes[values + j / 32]));
                }
            }
        };
        var failures = std.atomic.Value(usize).init(0);
        const ctx = Ctx{ .r = r, .li = li, .ids = ids, .order = order, .out = out, .failures = &failures };
        var handles: [64]std.Thread = undefined;
        var spawned: usize = 0;
        var w: usize = 0;
        while (w < workers) : (w += 1) {
            const from = w * per;
            const to = @min(ids.len, from + per);
            if (from >= to) continue;
            handles[spawned] = std.Thread.spawn(.{}, Ctx.run, .{ &ctx, from, to }) catch break;
            spawned += 1;
        }
        // The calling thread takes the remaining ranges so a short spawn still completes the gather.
        while (w < workers) : (w += 1) {
            const from = w * per;
            const to = @min(ids.len, from + per);
            if (from >= to) continue;
            ctx.run(from, to);
        }
        for (handles[0..spawned]) |h| h.join();
        if (failures.load(.monotonic) != 0) return error.ShortRead;
        r.stats.reads += 2 * ids.len;
        r.stats.bytes += row_bytes * ids.len;
    }

    pub fn rows(r: *Rows, li: usize, ids: []const i64, dim: usize, out: []u16) !void {
        try r.validate(li, ids);
        if (dim != values or out.len != try std.math.mul(usize, ids.len, values)) return error.BadEngramShard;
        for (ids, 0..) |id, i| {
            const b = try r.record(li, @intCast(id));
            for (0..values) |j| out[i * values + j] = f.bf16(f.e4m3(b[j]) * f.scale(b[values + j / 32]));
        }
    }

    /// One line a window when TF_DSV41_ENGRAM_TRACE is set: what this window read and what it hit.
    pub fn recordWindow(r: *Rows, window: u32) void {
        if (!r.trace) return;
        std.log.scoped(.dsv41).info("engram stream: rows={d} hits={d} misses={d} reads={d} syscalls={d} bytes={d} mapped={}", .{ window, r.stats.hits, r.stats.misses, r.stats.reads, r.stats.syscalls, r.stats.bytes, r.mapped() });
    }

    pub fn source(r: *Rows) eh.Source {
        return .{ .ptr = r, .vtable = &.{ .issue = issueFn, .rows = rowsFn } };
    }
    fn issueFn(p: *anyopaque, li: usize, ids: []const i64) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.issue(li, ids);
    }
    fn rowsFn(p: *anyopaque, li: usize, ids: []const i64, dim: usize, out: []u16) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.rows(li, ids, dim, out);
    }
};
