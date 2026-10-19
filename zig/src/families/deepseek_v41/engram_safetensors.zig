//! Deduplicate and stage uncached Engram rows through Fetcher; contracts: docs/deepseek-execution.md.
const std = @import("std");
const paced = @import("../../core/paced_read.zig");

pub const weight_bytes = 256;
pub const scale_bytes = 8;

/// One table's place in its shard; offsets are relative to `data_start`, as in the safetensors header.
pub const File = struct {
    path: [:0]const u8,
    file_bytes: usize,
    data_start: usize,
    weight_begin: usize,
    scale_begin: usize,
    rows: usize,
};

pub const Options = struct {
    /// Most row IDs one stage call takes (chunk rows x 24 columns).
    max_ids: usize,
    /// Rows kept in host memory across calls; 0 turns the cache off.
    cache_rows: u32 = 0,
    /// The whole scale column read once at open (8 bytes a row), so each miss costs one read.
    scales_in_ram: bool = true,
    threads: u32 = 16,
};

/// Counters for the residency receipt: every byte read from the shard after open is counted here.
pub const Stats = struct {
    requested: u64 = 0,
    unique: u64 = 0,
    hits: u64 = 0,
    /// Fetcher calls (one per stage call with misses) and the time spent inside them.
    fetches: u64 = 0,
    fetch_ns: u64 = 0,
    reads: u64 = 0,
    read_bytes: u64 = 0,
    /// Bytes of the scale column read at open; not part of `read_bytes`.
    open_bytes: u64 = 0,
    wait_ns: u64 = 0,

    pub fn add(a: *Stats, b: Stats) void {
        a.requested += b.requested;
        a.unique += b.unique;
        a.hits += b.hits;
        a.fetches += b.fetches;
        a.fetch_ns += b.fetch_ns;
        a.reads += b.reads;
        a.read_bytes += b.read_bytes;
        a.open_bytes += b.open_bytes;
        a.wait_ns += b.wait_ns;
    }
};

const Cache = struct {
    cap: u32,
    len: u32 = 0,
    head: u32 = none,
    tail: u32 = none,
    keys: []u32,
    prev: []u32,
    next: []u32,
    bytes: []u8,
    map: std.AutoHashMapUnmanaged(u32, u32) = .empty,

    const none = std.math.maxInt(u32);
    const entry = weight_bytes + scale_bytes;

    fn init(gpa: std.mem.Allocator, cap: u32) !Cache {
        var c = Cache{ .cap = cap, .keys = &.{}, .prev = &.{}, .next = &.{}, .bytes = &.{} };
        if (cap == 0) return c;
        errdefer c.deinit(gpa);
        c.keys = try gpa.alloc(u32, cap);
        c.prev = try gpa.alloc(u32, cap);
        c.next = try gpa.alloc(u32, cap);
        c.bytes = try gpa.alloc(u8, @as(usize, cap) * entry);
        try c.map.ensureTotalCapacity(gpa, cap);
        return c;
    }

    fn deinit(c: *Cache, gpa: std.mem.Allocator) void {
        gpa.free(c.keys);
        gpa.free(c.prev);
        gpa.free(c.next);
        gpa.free(c.bytes);
        c.map.deinit(gpa);
    }

    fn unlink(c: *Cache, s: u32) void {
        if (c.prev[s] != none) c.next[c.prev[s]] = c.next[s] else c.head = c.next[s];
        if (c.next[s] != none) c.prev[c.next[s]] = c.prev[s] else c.tail = c.prev[s];
    }

    fn front(c: *Cache, s: u32) void {
        c.prev[s] = none;
        c.next[s] = c.head;
        if (c.head != none) c.prev[c.head] = s;
        c.head = s;
        if (c.tail == none) c.tail = s;
    }

    /// The row's 264 cached bytes (weight then scale), now the most recently used.
    fn get(c: *Cache, row: u32) ?[]const u8 {
        if (c.cap == 0) return null;
        const s = c.map.get(row) orelse return null;
        c.unlink(s);
        c.front(s);
        return c.bytes[@as(usize, s) * entry ..][0..entry];
    }

    fn put(c: *Cache, row: u32, w: []const u8, sc: []const u8) void {
        if (c.cap == 0 or c.map.contains(row)) return;
        const s = if (c.len < c.cap) blk: {
            c.len += 1;
            break :blk c.len - 1;
        } else blk: {
            const old = c.tail;
            c.unlink(old);
            _ = c.map.remove(c.keys[old]);
            break :blk old;
        };
        c.keys[s] = row;
        c.map.putAssumeCapacity(row, s);
        @memcpy(c.bytes[@as(usize, s) * entry ..][0..weight_bytes], w[0..weight_bytes]);
        @memcpy(c.bytes[@as(usize, s) * entry + weight_bytes ..][0..scale_bytes], sc[0..scale_bytes]);
        c.front(s);
    }
};

/// A worker's slice of the misses; rows ascend, so neighbours share one aligned block read.
const Part = struct {
    fd: std.c.fd_t,
    file: *const File,
    rows: []const u32,
    slots: []const u32,
    out_w: []u8,
    out_s: []u8,
    scales: ?[]const u8,
    block: paced.Stage,
    reads: u64 = 0,
    bytes: u64 = 0,
    failed: bool = false,

    fn run(p: *Part) void {
        p.fill() catch {
            p.failed = true;
        };
    }

    fn fill(p: *Part) !void {
        var w = Window{};
        for (p.rows, p.slots) |row, slot| {
            const r: usize = row;
            const wo = p.file.data_start + p.file.weight_begin + r * weight_bytes;
            try w.copy(p, wo, p.out_w[@as(usize, slot) * weight_bytes ..][0..weight_bytes]);
        }
        const out = p.out_s;
        if (p.scales) |all| {
            for (p.rows, p.slots) |row, slot| @memcpy(out[@as(usize, slot) * scale_bytes ..][0..scale_bytes], all[@as(usize, row) * scale_bytes ..][0..scale_bytes]);
        } else {
            var s = Window{};
            for (p.rows, p.slots) |row, slot| {
                const so = p.file.data_start + p.file.scale_begin + @as(usize, row) * scale_bytes;
                try s.copy(p, so, out[@as(usize, slot) * scale_bytes ..][0..scale_bytes]);
            }
        }
    }
};

/// The file range currently held in a worker's aligned block.
const Window = struct {
    lo: usize = 0,
    hi: usize = 0,

    fn copy(w: *Window, p: *Part, at: usize, dest: []u8) !void {
        if (at < w.lo or at + dest.len > w.hi) {
            const lo = std.mem.alignBackward(usize, at, paced.alignment);
            const span = std.mem.alignForward(usize, at + dest.len - lo, paced.alignment);
            if (span > p.block.len) return error.EngramRowSpan;
            var got: usize = 0;
            while (got < span and lo + got < p.file.file_bytes) {
                const n = std.c.pread(p.fd, p.block.ptr + got, span - got, @intCast(lo + got));
                if (n <= 0) break;
                got += @intCast(n);
            }
            p.reads += 1;
            p.bytes += got;
            if (lo + got < at + dest.len) return error.ShortRead;
            w.lo = lo;
            w.hi = lo + got;
        }
        @memcpy(dest, p.block[at - w.lo ..][0..dest.len]);
    }
};

/// Fetch all ascending missed IDs into slots: 256 weight bytes and 8 scale bytes per row; implemented by Pread.
pub const Fetcher = struct {
    ptr: *anyopaque,
    fetch_fn: *const fn (*anyopaque, rows: []const u32, slots: []const u32, out_w: []u8, out_s: []u8) anyerror!Fetched,
    deinit_fn: *const fn (*anyopaque) void,

    pub fn fetch(f: Fetcher, rows: []const u32, slots: []const u32, out_w: []u8, out_s: []u8) !Fetched {
        return f.fetch_fn(f.ptr, rows, slots, out_w, out_s);
    }
    pub fn deinit(f: Fetcher) void {
        f.deinit_fn(f.ptr);
    }
};
/// What one fetch cost: requests issued and bytes moved (disk bytes for `Pread`).
pub const Fetched = struct { reads: u64 = 0, bytes: u64 = 0 };

/// Rows from the shard with F_NOCACHE preads on up to `threads` threads; scales optionally held in RAM.
pub const Pread = struct {
    gpa: std.mem.Allocator,
    file: File,
    fd: std.c.fd_t,
    scales: ?[]u8 = null,
    blocks: []paced.Stage,
    /// Bytes of the scale column read at open.
    open_bytes: u64 = 0,

    pub fn open(gpa: std.mem.Allocator, file: File, options: Options) !*Pread {
        if (file.rows == 0 or file.rows > std.math.maxInt(u32) or options.threads == 0) return error.InvalidEngramSource;
        const weight_end = try std.math.add(usize, file.weight_begin, try std.math.mul(usize, file.rows, weight_bytes));
        const scale_end = try std.math.add(usize, file.scale_begin, try std.math.mul(usize, file.rows, scale_bytes));
        if (file.data_start + @max(weight_end, scale_end) > file.file_bytes) return error.InvalidEngramSource;
        const fd = std.c.open(file.path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        errdefer _ = std.c.close(fd);
        if (std.c.fcntl(fd, 48, @as(c_int, 1)) != 0) return error.UncachedReadUnavailable; // F_NOCACHE
        if (std.c.lseek(fd, 0, std.c.SEEK.END) != file.file_bytes) return error.CheckpointChanged;
        const self = try gpa.create(Pread);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .file = file, .fd = fd, .blocks = try gpa.alloc(paced.Stage, options.threads) };
        var made: usize = 0;
        errdefer {
            for (self.blocks[0..made]) |b| gpa.free(b);
            gpa.free(self.blocks);
        }
        for (self.blocks) |*b| {
            b.* = try paced.stage(gpa, 2 * paced.alignment);
            made += 1;
        }
        if (options.scales_in_ram) {
            const bytes = file.rows * scale_bytes;
            const all = try gpa.alloc(u8, bytes);
            errdefer gpa.free(all);
            const stage_buf = try paced.stage(gpa, 32 << 20);
            defer gpa.free(stage_buf);
            var done: usize = 0;
            while (done < bytes) {
                const n = @min(bytes - done, stage_buf.len - paced.alignment);
                var short: paced.Memory = undefined;
                try paced.read(fd, all[done..][0..n], file.data_start + file.scale_begin + done, stage_buf, &short);
                done += n;
            }
            self.scales = all;
            self.open_bytes = bytes;
        }
        return self;
    }

    pub fn fetcher(self: *Pread) Fetcher {
        return .{ .ptr = self, .fetch_fn = fetchErased, .deinit_fn = deinitErased };
    }

    fn deinitErased(ptr: *anyopaque) void {
        const self: *Pread = @ptrCast(@alignCast(ptr));
        _ = std.c.close(self.fd);
        if (self.scales) |all| self.gpa.free(all);
        for (self.blocks) |b| self.gpa.free(b);
        self.gpa.free(self.blocks);
        self.gpa.destroy(self);
    }

    fn fetchErased(ptr: *anyopaque, rows: []const u32, slots: []const u32, out_w: []u8, out_s: []u8) anyerror!Fetched {
        const self: *Pread = @ptrCast(@alignCast(ptr));
        const misses = rows.len;
        if (misses == 0) return .{};
        const workers = @min(self.blocks.len, (misses + 7) / 8);
        var parts: [256]Part = undefined;
        var threads: [256]?std.Thread = @splat(null);
        const count = @min(workers, parts.len);
        const per = (misses + count - 1) / count;
        for (0..count) |i| {
            const lo = @min(i * per, misses);
            const hi = @min(lo + per, misses);
            parts[i] = .{ .fd = self.fd, .file = &self.file, .rows = rows[lo..hi], .slots = slots[lo..hi], .out_w = out_w, .out_s = out_s, .scales = self.scales, .block = self.blocks[i] };
        }
        for (1..count) |i| threads[i] = std.Thread.spawn(.{}, Part.run, .{&parts[i]}) catch null;
        for (0..count) |i| if (threads[i] == null) parts[i].run();
        for (threads[0..count]) |t| if (t) |th| th.join();
        var out: Fetched = .{};
        for (parts[0..count]) |p| {
            if (p.failed) return error.EngramReadFailed;
            out.reads += p.reads;
            out.bytes += p.bytes;
        }
        return out;
    }
};

/// Dedup, row cache and staging for one table over any `Fetcher`; counts bytes and fetch latency per source.
pub const Source = struct {
    gpa: std.mem.Allocator,
    rows: usize,
    fetcher: Fetcher,
    cache: Cache,
    unique: []u32,
    miss_rows: []u32,
    miss_slots: []u32,
    stats: Stats = .{},

    /// The shard-backed source: a `Pread` fetcher over `file`.
    pub fn open(gpa: std.mem.Allocator, file: File, options: Options) !Source {
        const pread = try Pread.open(gpa, file, options);
        const f = pread.fetcher();
        errdefer f.deinit();
        var s = try openFetcher(gpa, file.rows, f, options);
        s.stats.open_bytes = pread.open_bytes;
        return s;
    }

    /// A source over `fetcher`, which it owns from here on (deinit releases it).
    pub fn openFetcher(gpa: std.mem.Allocator, row_count: usize, fetcher: Fetcher, options: Options) !Source {
        if (row_count == 0 or row_count > std.math.maxInt(u32) or options.max_ids == 0) return error.InvalidEngramSource;
        var self = Source{ .gpa = gpa, .rows = row_count, .fetcher = fetcher, .cache = undefined, .unique = &.{}, .miss_rows = &.{}, .miss_slots = &.{} };
        self.cache = try Cache.init(gpa, options.cache_rows);
        errdefer self.cache.deinit(gpa);
        self.unique = try gpa.alloc(u32, options.max_ids);
        errdefer gpa.free(self.unique);
        self.miss_rows = try gpa.alloc(u32, options.max_ids);
        errdefer gpa.free(self.miss_rows);
        self.miss_slots = try gpa.alloc(u32, options.max_ids);
        return self;
    }

    pub fn deinit(s: *Source) void {
        s.fetcher.deinit();
        s.cache.deinit(s.gpa);
        s.gpa.free(s.unique);
        s.gpa.free(s.miss_rows);
        s.gpa.free(s.miss_slots);
        s.* = undefined;
    }

    /// Host bytes a shard-backed source holds beyond the staging buffers: scales, cache and scratch.
    pub fn hostBytes(options: Options, row_count: usize) usize {
        const scales = if (options.scales_in_ram) row_count * scale_bytes else 0;
        return scales + @as(usize, options.cache_rows) * (Cache.entry + 16) + options.max_ids * 12 + options.threads * 2 * paced.alignment;
    }

    /// Stage distinct sorted IDs, return their count and fill local indices; reject invalid IDs before fetching.
    pub fn stage(s: *Source, ids: []const u32, out_w: []u8, out_s: []u8, local: []u32) !u32 {
        if (ids.len == 0 or ids.len > s.unique.len or local.len < ids.len) return error.InvalidEngramStage;
        if (out_w.len < ids.len * weight_bytes or out_s.len < ids.len * scale_bytes) return error.InvalidEngramStage;
        const started = now();
        for (ids) |id| if (id >= s.rows) return error.InvalidEngramRow;
        const unique = s.unique[0..ids.len];
        @memcpy(unique, ids);
        std.mem.sort(u32, unique, {}, std.sort.asc(u32));
        var n: usize = 0;
        for (unique) |id| {
            if (n == 0 or unique[n - 1] != id) {
                unique[n] = id;
                n += 1;
            }
        }
        const rows = unique[0..n];
        for (ids, local[0..ids.len]) |id, *l| l.* = @intCast(std.sort.binarySearch(u32, rows, id, order).?);
        var misses: usize = 0;
        var hits: u64 = 0;
        for (rows, 0..) |row, slot| {
            if (s.cache.get(row)) |bytes| {
                @memcpy(out_w[slot * weight_bytes ..][0..weight_bytes], bytes[0..weight_bytes]);
                @memcpy(out_s[slot * scale_bytes ..][0..scale_bytes], bytes[weight_bytes..][0..scale_bytes]);
                hits += 1;
            } else {
                s.miss_rows[misses] = row;
                s.miss_slots[misses] = @intCast(slot);
                misses += 1;
            }
        }
        var got: Fetched = .{};
        var fetch_ns: u64 = 0;
        if (misses > 0) {
            const before = now();
            got = try s.fetcher.fetch(s.miss_rows[0..misses], s.miss_slots[0..misses], out_w, out_s);
            fetch_ns = now() - before;
            for (s.miss_rows[0..misses], s.miss_slots[0..misses]) |row, slot|
                s.cache.put(row, out_w[@as(usize, slot) * weight_bytes ..][0..weight_bytes], out_s[@as(usize, slot) * scale_bytes ..][0..scale_bytes]);
        }
        s.stats.add(.{ .requested = ids.len, .unique = n, .hits = hits, .fetches = @intFromBool(misses > 0), .reads = got.reads, .read_bytes = got.bytes, .fetch_ns = fetch_ns, .wait_ns = now() - started });
        return @intCast(n);
    }
};

fn order(key: u32, item: u32) std.math.Order {
    return std.math.order(key, item);
}

fn now() u64 {
    var t: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t) != 0 or t.sec < 0) return 0;
    return @as(u64, @intCast(t.sec)) * std.time.ns_per_s + @as(u64, @intCast(t.nsec));
}

/// A synthetic two-tensor shard: row r's weight byte d is (r*7 + d) & 0xff, its scale byte g is (r + g) & 0xff.
pub fn writeFixture(io: std.Io, dir: std.Io.Dir, name: []const u8, rows: usize, pad: usize) !struct { data_start: usize, file_bytes: usize } {
    var header_buf: [512]u8 = undefined;
    const wb = rows * weight_bytes;
    const header = try std.fmt.bufPrint(&header_buf, "{{\"pad\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}},\"weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},256],\"data_offsets\":[{d},{d}]}},\"scale\":{{\"dtype\":\"F8_E8M0\",\"shape\":[{d},8],\"data_offsets\":[{d},{d}]}}}}", .{ pad, pad, rows, pad, pad + wb, rows, pad + wb, pad + wb + rows * scale_bytes });
    const data_start = 8 + header.len;
    const total = data_start + pad + wb + rows * scale_bytes;
    const a = std.testing.allocator;
    const data = try a.alloc(u8, total);
    defer a.free(data);
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    @memset(data[data_start..][0..pad], 0xee);
    for (0..rows) |r| {
        for (0..weight_bytes) |d| data[data_start + pad + r * weight_bytes + d] = @truncate(r * 7 + d);
        for (0..scale_bytes) |g| data[data_start + pad + wb + r * scale_bytes + g] = @truncate(r + g);
    }
    try dir.writeFile(io, .{ .sub_path = name, .data = data });
    return .{ .data_start = data_start, .file_bytes = total };
}

test "staged rows equal the shard's bytes, deduplicated, with local indices, hits and exact byte counts" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const rows: usize = 300;
    const pad: usize = 1234; // rows straddle 16 KiB blocks
    const fx = try writeFixture(io, tmp.dir, "t.safetensors", rows, pad);
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/t.safetensors", .{tmp.sub_path}, 0);
    defer a.free(path);
    for ([_]bool{ true, false }) |in_ram| {
        var src = try Source.open(a, .{ .path = path, .file_bytes = fx.file_bytes, .data_start = fx.data_start, .weight_begin = pad, .scale_begin = pad + rows * weight_bytes, .rows = rows }, .{ .max_ids = 64, .cache_rows = 4, .scales_in_ram = in_ram, .threads = 3 });
        defer src.deinit();
        const ids = [_]u32{ 299, 5, 5, 64, 0, 299, 63, 128, 127, 5, 200, 201, 202, 203, 204, 205, 206, 207, 208, 1, 2, 3, 4, 6 };
        var w: [ids.len * weight_bytes]u8 = undefined;
        var sc: [ids.len * scale_bytes]u8 = undefined;
        var local: [ids.len]u32 = undefined;
        var first_bytes: u64 = 0;
        for (0..2) |round| {
            const n = try src.stage(&ids, &w, &sc, &local);
            try std.testing.expectEqual(@as(u32, 21), n);
            for (ids, local) |id, l| {
                for (0..weight_bytes) |d| try std.testing.expectEqual(@as(u8, @truncate(id * 7 + d)), w[l * weight_bytes + d]);
                for (0..scale_bytes) |g| try std.testing.expectEqual(@as(u8, @truncate(id + g)), sc[l * scale_bytes + g]);
            }
            for (local) |l| try std.testing.expect(l < n);
            if (round == 0) {
                try std.testing.expectEqual(@as(u64, 0), src.stats.hits);
                first_bytes = src.stats.read_bytes;
                try std.testing.expect(first_bytes > 0);
            } else {
                try std.testing.expectEqual(@as(u64, 4), src.stats.hits);
                try std.testing.expect(src.stats.read_bytes > first_bytes);
            }
        }
        try std.testing.expectEqual(@as(u64, 2 * ids.len), src.stats.requested);
        try std.testing.expectEqual(@as(u64, 42), src.stats.unique);
        try std.testing.expectEqual(@as(u64, if (in_ram) rows * scale_bytes else 0), src.stats.open_bytes);
        var bad = [_]u32{ 1, @intCast(rows) };
        try std.testing.expectError(error.InvalidEngramRow, src.stage(&bad, &w, &sc, &local));
    }
}

test "the row cache evicts the least recently used row" {
    const a = std.testing.allocator;
    var c = try Cache.init(a, 2);
    defer c.deinit(a);
    var w: [weight_bytes]u8 = @splat(1);
    const s: [scale_bytes]u8 = @splat(2);
    c.put(10, &w, &s);
    w[0] = 3;
    c.put(11, &w, &s);
    try std.testing.expect(c.get(10) != null); // 11 is now the oldest
    c.put(12, &w, &s);
    try std.testing.expect(c.get(11) == null);
    try std.testing.expectEqual(@as(u8, 1), c.get(10).?[0]);
    try std.testing.expectEqual(@as(u8, 3), c.get(12).?[0]);
    try std.testing.expectEqual(@as(u8, 2), c.get(12).?[weight_bytes]);
}

test "any fetcher gets each stage call's misses in one ascending batch" {
    const a = std.testing.allocator;
    const Fake = struct {
        calls: u32 = 0,
        last: u32 = 0,
        freed: bool = false,
        fn fetch(ptr: *anyopaque, rows: []const u32, slots: []const u32, out_w: []u8, out_s: []u8) anyerror!Fetched {
            const f: *@This() = @ptrCast(@alignCast(ptr));
            f.calls += 1;
            f.last = @intCast(rows.len);
            for (1..rows.len) |i| if (rows[i - 1] >= rows[i]) return error.NotAscending;
            for (rows, slots) |r, slot| {
                @memset(out_w[@as(usize, slot) * weight_bytes ..][0..weight_bytes], @truncate(r));
                @memset(out_s[@as(usize, slot) * scale_bytes ..][0..scale_bytes], @truncate(r +% 1));
            }
            return .{ .reads = 1, .bytes = rows.len * (weight_bytes + scale_bytes) };
        }
        fn deinit(ptr: *anyopaque) void {
            const f: *@This() = @ptrCast(@alignCast(ptr));
            f.freed = true;
        }
    };
    var fake = Fake{};
    var src = try Source.openFetcher(a, 100, .{ .ptr = &fake, .fetch_fn = Fake.fetch, .deinit_fn = Fake.deinit }, .{ .max_ids = 24, .cache_rows = 8 });
    const ids = [_]u32{ 9, 3, 9, 50, 3, 7 };
    var w: [ids.len * weight_bytes]u8 = undefined;
    var sc: [ids.len * scale_bytes]u8 = undefined;
    var local: [ids.len]u32 = undefined;
    try std.testing.expectEqual(@as(u32, 4), try src.stage(&ids, &w, &sc, &local));
    for (ids, local) |id, l| try std.testing.expectEqual(@as(u8, @truncate(id)), w[l * weight_bytes]);
    try std.testing.expectEqual(@as(u32, 1), fake.calls);
    try std.testing.expectEqual(@as(u32, 4), fake.last);
    _ = try src.stage(&ids, &w, &sc, &local); // all cached: no fetch
    try std.testing.expectEqual(@as(u32, 1), fake.calls);
    try std.testing.expectEqual(@as(u64, 1), src.stats.fetches);
    try std.testing.expectEqual(@as(u64, 4 * (weight_bytes + scale_bytes)), src.stats.read_bytes);
    try std.testing.expectEqual(@as(u64, 4), src.stats.hits);
    src.deinit();
    try std.testing.expect(fake.freed);
}
