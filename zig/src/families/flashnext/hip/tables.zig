//! The n-gram tables in host memory (the checkpoint's mapped shards): a window's rows gathered for the device.
const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const ngram = @import("ngram.zig");
const host_pages = @import("../host_pages.zig");
/// The parked workers a gather spreads over: no backend in them, so the CUDA n-gram reader shares them.
const pool = @import("../host_pool.zig");

/// One shard's packed rows and their group tables, as mapped.
pub const Shard = struct { words: []const u8, scales: []const u8, biases: []const u8 };

/// A gather copies the checkpoint's rows in id order: a staging too small, or a row past the last shard.
const Error = error{ StagingTooSmall, NgramRowOutOfRange };

/// Parts a gather spreads over at most, and rows a part takes before the gather stops adding parts.
const max_parts = 32;
const rows_per_part = 2;

/// The shards' rows as mapped: what a gather reads, apart from the hasher that names the rows.
pub const Rows = struct {
    shards: []const Shard,
    /// The first row of each shard, and one past the last.
    starts: []const usize,
    /// The first `extra` shards hold `wide` rows and the rest `narrow`; `wide` is 0 when the sizes do not fit.
    wide: usize,
    narrow: usize,
    extra: usize,
    dims: usize,
    bits: u8,
    group: usize,

    /// Bytes of a row's packed words, and of its scales (as of its biases).
    pub fn wordBytes(r: *const Rows) usize {
        return r.dims * r.bits / 8;
    }

    pub fn tableBytes(r: *const Rows) usize {
        return r.dims / r.group * 2;
    }

    /// The shard sizes the reference splits rows into: `wide` rows for the first `extra` shards, then `narrow`.
    pub const Split = struct { wide: usize, narrow: usize, extra: usize };

    /// The sizes `starts` shows, or zeros when they follow no such run and a row must be searched for.
    pub fn split(starts: []const usize) Split {
        const n = starts.len - 1;
        if (n == 0 or starts[1] - starts[0] == 0) return .{ .wide = 0, .narrow = 0, .extra = 0 };
        const wide = starts[1] - starts[0];
        var extra: usize = 1;
        while (extra < n and starts[extra + 1] - starts[extra] == wide) extra += 1;
        const narrow = if (extra < n) starts[extra + 1] - starts[extra] else wide;
        if (narrow == 0) return .{ .wide = 0, .narrow = 0, .extra = 0 };
        for (extra..n) |i| {
            if (starts[i + 1] - starts[i] != narrow) return .{ .wide = 0, .narrow = 0, .extra = 0 };
        }
        return .{ .wide = wide, .narrow = narrow, .extra = extra };
    }

    /// Row `id`'s shard and its row in it: a shard's rows are a contiguous run, so the sizes need no search.
    pub fn locate(r: *const Rows, id: u32) Error!struct { shard: usize, row: usize } {
        const at: usize = id;
        if (at >= r.starts[r.shards.len]) return error.NgramRowOutOfRange;
        if (r.wide != 0) {
            const first = r.wide * r.extra;
            if (at < first) return .{ .shard = at / r.wide, .row = at % r.wide };
            return .{ .shard = r.extra + (at - first) / r.narrow, .row = (at - first) % r.narrow };
        }
        const s = std.sort.upperBound(usize, r.starts, at, struct {
            fn order(x: usize, y: usize) std.math.Order {
                return std.math.order(x, y);
            }
        }.order) - 1;
        return .{ .shard = s, .row = at - r.starts[s] };
    }

    /// Every byte a gather reads (a shard's three arrays): the pages a checkpoint release over the weights keeps.
    pub fn spans(r: *const Rows, a: std.mem.Allocator) ![]const []const u8 {
        const out = try a.alloc([]const u8, r.shards.len * 3);
        for (r.shards, 0..) |s, i| {
            out[i * 3] = s.words;
            out[i * 3 + 1] = s.scales;
            out[i * 3 + 2] = s.biases;
        }
        return out;
    }

    /// Each id's packed words, scales and biases into the staging slices in id order; `w`'s parts warm first.
    pub fn gather(r: *const Rows, w: *Workers, ids: []const u32, words: []u8, scales: []u8, biases: []u8) !void {
        const wb = r.wordBytes();
        const tb = r.tableBytes();
        if (words.len < ids.len * wb or scales.len < ids.len * tb or biases.len < ids.len * tb)
            return error.StagingTooSmall;
        const parts = @min(max_parts, @max(1, ids.len / rows_per_part));
        const call: Parts = .{
            .r = r,
            .w = w,
            .ids = ids,
            .words = words,
            .scales = scales,
            .biases = biases,
            .per = (ids.len + parts - 1) / parts,
        };
        // a gather of pages asked for before is a copy alone: the caller takes every part, waking no worker
        const p = if (w.cold(r, ids)) w.pool else null;
        const run = p orelse {
            for (0..parts) |i| try call.part(i);
            return;
        };
        try run.run(parts, .{ .ctx = &call, .run = Parts.task });
    }
};

/// The parked workers a gather spreads its page faults over (none: the caller gathers alone) and the pages it warms.
pub const Workers = struct {
    pool: ?*pool.Pool = null,
    warmer: host_pages.Warmer = .{},

    /// Whether a row of `ids` holds a page no gather asked for yet (a warm gather copies alone, waking no worker).
    pub fn cold(w: *Workers, r: *const Rows, ids: []const u32) bool {
        for (ids) |id| {
            const at = r.locate(id) catch continue;
            const sh = r.shards[at.shard];
            const wb = r.wordBytes();
            const tb = r.tableBytes();
            const region = at.shard * 3;
            if (w.warmer.cold(region, sh.words[at.row * wb ..][0..wb])) return true;
            if (w.warmer.cold(region + 1, sh.scales[at.row * tb ..][0..tb])) return true;
            if (w.warmer.cold(region + 2, sh.biases[at.row * tb ..][0..tb])) return true;
        }
        return false;
    }
};

/// One gather in parts of `per` ids: a part asks for its rows' pages, then copies them into its own staging rows.
const Parts = struct {
    r: *const Rows,
    w: *Workers,
    ids: []const u32,
    words: []u8,
    scales: []u8,
    biases: []u8,
    per: usize,

    fn task(ctx: *const anyopaque, i: usize) anyerror!void {
        const self: *const Parts = @ptrCast(@alignCast(ctx));
        return self.part(i);
    }

    fn part(self: *const Parts, i: usize) Error!void {
        const first = @min(self.ids.len, i * self.per);
        const last = @min(self.ids.len, first + self.per);
        const wb = self.r.wordBytes();
        const tb = self.r.tableBytes();
        // every page of the part is asked for before its first byte is read: the reads wait together, not one by one
        for (self.ids[first..last]) |id| self.ask(id);
        for (self.ids[first..last], first..) |id, n| {
            const at = try self.r.locate(id);
            const sh = self.r.shards[at.shard];
            @memcpy(self.words[n * wb ..][0..wb], sh.words[at.row * wb ..][0..wb]);
            @memcpy(self.scales[n * tb ..][0..tb], sh.scales[at.row * tb ..][0..tb]);
            @memcpy(self.biases[n * tb ..][0..tb], sh.biases[at.row * tb ..][0..tb]);
        }
    }

    /// Row `id`'s three pages, best effort: a row past the last shard is refused by the copy.
    fn ask(self: *const Parts, id: u32) void {
        const at = self.r.locate(id) catch return;
        const sh = self.r.shards[at.shard];
        const wb = self.r.wordBytes();
        const tb = self.r.tableBytes();
        // a shard's words, scales and biases are its three warmer regions, in this order
        const region = at.shard * 3;
        self.w.warmer.warm(region, sh.words[at.row * wb ..][0..wb]);
        self.w.warmer.warm(region + 1, sh.scales[at.row * tb ..][0..tb]);
        self.w.warmer.warm(region + 2, sh.biases[at.row * tb ..][0..tb]);
    }
};

pub const Tables = struct {
    rows: Rows,
    hasher: ngram.Hasher,
    /// The gather's parked workers and its pages' advice; freed on deinit.
    workers: Workers,
    /// The spans the warmer covers, on the arena `init` was given.
    live: []const []const u8,

    pub fn wordBytes(t: Tables) usize {
        return t.rows.wordBytes();
    }

    pub fn tableBytes(t: Tables) usize {
        return t.rows.tableBytes();
    }

    /// The tables of PLE layer `layer` as the reference splits them, with the gather's workers and page advice.
    pub fn init(
        a: std.mem.Allocator,
        ck: *core.checkpoint.Checkpoint,
        io: std.Io,
        c: *const config.Config,
        layer: usize,
        spelling: []const u8,
    ) !Tables {
        const heads = try c.ngramHeadCount();
        const ref = try config.pleRef(c);
        var total: usize = 0;
        for (ref.sizes) |s| total += @intCast(s);
        const rows = try std.math.divCeil(usize, total, c.ngram_divisor) * c.ngram_divisor;
        const base = rows / c.ngram_shards;
        const extra = rows % c.ngram_shards;
        const shards = try a.alloc(Shard, c.ngram_shards);
        const starts = try a.alloc(usize, c.ngram_shards + 1);
        starts[0] = 0;
        var spec: ?@import("../spec.zig").Spec = null;
        for (shards, 0..) |*s, i| {
            var lb: [128]u8 = undefined;
            var nb: [256]u8 = undefined;
            const stem = try a.dupe(u8, try tensors.ple.table(&nb, try tensors.layer(&lb, layer), spelling, i));
            const n = base + @intFromBool(i < extra);
            starts[i + 1] = starts[i] + n;
            const w = try ck.get(try std.fmt.allocPrint(a, "{s}.weight", .{stem}));
            const sc = try ck.get(try std.fmt.allocPrint(a, "{s}.scales", .{stem}));
            const bi = try ck.get(try std.fmt.allocPrint(a, "{s}.biases", .{stem}));
            if (spec == null)
                spec = (try c.declared(stem, core.quant.keys.canonical)) orelse return error.UnsupportedQuantization;
            if (w.shape[0] != n or sc.dtype != .bf16 or bi.dtype != .bf16) return error.UnexpectedNgramTable;
            // a gather reads these rows in no order: readahead per row would fetch a window for it
            for ([_][]const u8{ "weight", "scales", "biases" }) |part|
                try ck.adviseRandom(try std.fmt.allocPrint(a, "{s}.{s}", .{ stem, part }));
            s.* = .{ .words = w.bytes, .scales = sc.bytes, .biases = bi.bytes };
        }
        const dims = c.ple_dim / heads;
        if (shards[0].words.len != (starts[1] - starts[0]) * dims * spec.?.bits / 8) return error.UnexpectedNgramTable;
        const sizes = Rows.split(starts);
        var r: Rows = .{
            .shards = shards,
            .starts = starts,
            .wide = sizes.wide,
            .narrow = sizes.narrow,
            .extra = sizes.extra,
            .dims = dims,
            .bits = spec.?.bits,
            .group = spec.?.group,
        };
        const live = try r.spans(a);
        errdefer a.free(live);
        var t: Tables = .{
            .rows = r,
            .hasher = try ngram.Hasher.init(c),
            .workers = .{ .warmer = try host_pages.Warmer.init(a, live) },
            .live = live,
        };
        errdefer t.workers.warmer.deinit(a);
        t.workers.pool = try pool.Pool.create(a, io);
        return t;
    }

    /// The parked workers freed and the advice's pages forgotten.
    pub fn deinit(t: *Tables, a: std.mem.Allocator) void {
        if (t.workers.pool) |p| p.destroy();
        t.workers.warmer.deinit(a);
        a.free(t.live);
        t.* = undefined;
    }

    /// Each id's packed words, scales and biases copied into the three staging slices, in id order.
    pub fn gather(t: *Tables, ids: []const u32, words: []u8, scales: []u8, biases: []u8) !void {
        return t.rows.gather(&t.workers, ids, words, scales, biases);
    }

    /// Every byte a gather reads at runtime: the pages a checkpoint release must leave mapped.
    pub fn spans(t: *const Tables, a: std.mem.Allocator) ![]const []const u8 {
        return t.rows.spans(a);
    }
};

/// The gather before the sizes were read off `starts`: one row at a time, its shard searched for.
fn searchedGather(r: *const Rows, ids: []const u32, words: []u8, scales: []u8, biases: []u8) !void {
    const wb = r.wordBytes();
    const tb = r.tableBytes();
    if (words.len < ids.len * wb or scales.len < ids.len * tb or biases.len < ids.len * tb)
        return error.StagingTooSmall;
    for (ids, 0..) |id, i| {
        const s = std.sort.upperBound(usize, r.starts, @as(usize, id), struct {
            fn order(x: usize, y: usize) std.math.Order {
                return std.math.order(x, y);
            }
        }.order) - 1;
        if (s >= r.shards.len) return error.NgramRowOutOfRange;
        const row = id - r.starts[s];
        const sh = r.shards[s];
        @memcpy(words[i * wb ..][0..wb], sh.words[row * wb ..][0..wb]);
        @memcpy(scales[i * tb ..][0..tb], sh.scales[row * tb ..][0..tb]);
        @memcpy(biases[i * tb ..][0..tb], sh.biases[row * tb ..][0..tb]);
    }
}

/// Synthetic shards of `sizes` rows (`dims` values a row at `bits`, groups of `group`): bytes hashed from their offset.
const Fake = struct {
    shards: []Shard,
    starts: []usize,
    arrays: [3][]u8,
    rows: Rows,
    /// The gather's parked workers (a pool over `io`) and the page advice over `spans`.
    workers: Workers,
    spans: []const []const u8,

    fn init(a: std.mem.Allocator, sizes: []const usize, dims: usize, bits: u8, group: usize) !Fake {
        return initOn(a, sizes, dims, bits, group, null);
    }

    /// The same shards gathered by `threads` parked workers (null: the caller alone, as without a pool).
    fn initOn(a: std.mem.Allocator, sizes: []const usize, dims: usize, bits: u8, group: usize, threads: ?std.Io) !Fake {
        var f: Fake = .{
            .shards = try a.alloc(Shard, sizes.len),
            .starts = try a.alloc(usize, sizes.len + 1),
            .arrays = .{ &.{}, &.{}, &.{} },
            .rows = undefined,
            .workers = undefined,
            .spans = &.{},
        };
        const wb = dims * bits / 8;
        const tb = dims / group * 2;
        f.starts[0] = 0;
        for (sizes, 0..) |n, i| f.starts[i + 1] = f.starts[i] + n;
        const total = f.starts[sizes.len];
        for (&f.arrays, [_]usize{ wb, tb, tb }, [_]u8{ 7, 3, 11 }) |*arr, width, step| {
            arr.* = try a.alloc(u8, total * width);
            for (arr.*, 0..) |*b, i| b.* = @truncate(i *% step +% width);
        }
        for (sizes, 0..) |n, i| {
            const from = f.starts[i];
            f.shards[i] = .{
                .words = f.arrays[0][from * wb ..][0 .. n * wb],
                .scales = f.arrays[1][from * tb ..][0 .. n * tb],
                .biases = f.arrays[2][from * tb ..][0 .. n * tb],
            };
        }
        const split = Rows.split(f.starts);
        f.rows = .{
            .shards = f.shards,
            .starts = f.starts,
            .wide = split.wide,
            .narrow = split.narrow,
            .extra = split.extra,
            .dims = dims,
            .bits = bits,
            .group = group,
        };
        f.spans = try f.rows.spans(a);
        f.workers = .{ .warmer = try host_pages.Warmer.init(a, f.spans) };
        if (threads) |io| f.workers.pool = try pool.Pool.create(a, io);
        return f;
    }

    fn deinit(f: *Fake, a: std.mem.Allocator) void {
        if (f.workers.pool) |p| p.destroy();
        f.workers.warmer.deinit(a);
        for (f.arrays) |arr| a.free(arr);
        a.free(f.spans);
        a.free(f.shards);
        a.free(f.starts);
    }

    fn gathered(f: *Fake, a: std.mem.Allocator, ids: []const u32) !Staging {
        const s: Staging = .{
            .words = try a.alloc(u8, ids.len * f.rows.wordBytes()),
            .scales = try a.alloc(u8, ids.len * f.rows.tableBytes()),
            .biases = try a.alloc(u8, ids.len * f.rows.tableBytes()),
        };
        @memset(s.words, 0);
        @memset(s.scales, 0);
        @memset(s.biases, 0);
        try f.rows.gather(&f.workers, ids, s.words, s.scales, s.biases);
        return s;
    }

    const Staging = struct {
        words: []u8,
        scales: []u8,
        biases: []u8,
        fn deinit(s: *const Staging, a: std.mem.Allocator) void {
            a.free(s.words);
            a.free(s.scales);
            a.free(s.biases);
        }
    };
};

test "a row's shard comes from the sizes alone: every id of a remainder split finds the searched shard" {
    const a = std.testing.allocator;
    // 128 shards of 250 rows with 3 of 251, as a checkpoint's padded row count splits
    var sizes: [128]usize = @splat(250);
    for (sizes[0..3]) |*n| n.* = 251;
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(a);
    var total: usize = 0;
    for (sizes) |n| total += n;
    for (0..total) |id| try ids.append(a, @intCast(id));
    var f = try Fake.init(a, &sizes, 160, 4, 32);
    defer f.deinit(a);
    try std.testing.expectEqual(@as(usize, 251), f.rows.wide);
    try std.testing.expectEqual(@as(usize, 250), f.rows.narrow);
    try std.testing.expectEqual(@as(usize, 3), f.rows.extra);
    var shard: usize = 0;
    for (ids.items) |id| {
        const at = try f.rows.locate(id);
        if (id == f.starts[shard + 1]) shard += 1;
        try std.testing.expectEqual(shard, at.shard);
        try std.testing.expectEqual(@as(usize, id) - f.starts[shard], at.row);
    }
    try std.testing.expectError(error.NgramRowOutOfRange, f.rows.locate(@intCast(f.starts[128])));
}

test "gather: the sizes' fast path copies the same bytes as a shard search for every id" {
    const a = std.testing.allocator;
    const cases = [_][]const usize{
        &.{},
        &.{97},
        &.{ 40, 40, 40, 40 },
        &.{ 41, 41, 40, 40 },
        &.{ 40, 39, 41, 40 },
        &.{ 250, 250, 250 },
    };
    for (cases) |sizes| {
        var ids: [256]u32 = undefined;
        var n: usize = 0;
        var total: usize = 0;
        for (sizes) |s| total += s;
        while (n < ids.len and n < total) : (n += 1) ids[n] = @intCast((n * 37) % total);
        var f = try Fake.init(a, sizes, 160, 4, 32);
        defer f.deinit(a);
        const want = try a.alloc(u8, n * f.rows.wordBytes());
        defer a.free(want);
        const want_scales = try a.alloc(u8, n * f.rows.tableBytes());
        defer a.free(want_scales);
        const want_biases = try a.alloc(u8, n * f.rows.tableBytes());
        defer a.free(want_biases);
        try searchedGather(&f.rows, ids[0..n], want, want_scales, want_biases);
        const got = try f.gathered(a, ids[0..n]);
        defer got.deinit(a);
        try std.testing.expectEqualSlices(u8, want, got.words);
        try std.testing.expectEqualSlices(u8, want_scales, got.scales);
        try std.testing.expectEqualSlices(u8, want_biases, got.biases);
    }
}

test "gather: rows copied by several threads equal one thread's, and a bad row or a short staging is refused" {
    const a = std.testing.allocator;
    var sizes: [64]usize = @splat(300);
    for (sizes[0..8]) |*s| s.* = 301;
    var total: usize = 0;
    for (sizes) |n| total += n;
    var ids: [8192]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 7919) % total);
    var f = try Fake.init(a, &sizes, 160, 4, 32);
    defer f.deinit(a);
    const want = try a.alloc(u8, ids.len * f.rows.wordBytes());
    defer a.free(want);
    const want_scales = try a.alloc(u8, ids.len * f.rows.tableBytes());
    defer a.free(want_scales);
    const want_biases = try a.alloc(u8, ids.len * f.rows.tableBytes());
    defer a.free(want_biases);
    try searchedGather(&f.rows, &ids, want, want_scales, want_biases);
    const got = try f.gathered(a, &ids);
    defer got.deinit(a);
    try std.testing.expectEqualSlices(u8, want, got.words);
    try std.testing.expectEqualSlices(u8, want_scales, got.scales);
    try std.testing.expectEqualSlices(u8, want_biases, got.biases);
    // an id past the last row, and staging that cannot hold the rows the ids ask for
    var past = ids;
    past[0] = @intCast(f.starts[sizes.len]);
    const small = try a.alloc(u8, ids.len * f.rows.wordBytes());
    defer a.free(small);
    try std.testing.expectError(error.NgramRowOutOfRange, f.rows.gather(&f.workers, &past, small, want_scales, want_biases));
    const short = small[0 .. small.len - 1];
    try std.testing.expectError(error.StagingTooSmall, f.rows.gather(&f.workers, &ids, short, want_scales, want_biases));
}

test "gather: the parked pool's parts copy the same bytes as one thread for every id count" {
    const a = std.testing.allocator;
    var sizes: [48]usize = @splat(120);
    for (sizes[0..4]) |*s| s.* = 121;
    var total: usize = 0;
    for (sizes) |n| total += n;
    var all: [8193]u32 = undefined;
    for (&all, 0..) |*id, i| id.* = @intCast((i * 7919) % total);
    var one = try Fake.init(a, &sizes, 160, 4, 32);
    defer one.deinit(a);
    var many = try Fake.initOn(a, &sizes, 160, 4, 32, std.testing.io);
    defer many.deinit(a);
    for ([_]usize{ 0, 1, 16, 17, 256, 4096, 8192 }) |n| {
        const ids = all[0..n];
        // the pool's copy, with a fresh seen set each call so every row's pages are asked for
        many.workers.warmer.deinit(a);
        many.workers.warmer = try host_pages.Warmer.init(a, many.spans);
        const got = try many.gathered(a, ids);
        defer got.deinit(a);
        // one thread's, on a warmer that never asks
        const want = try a.alloc(u8, n * one.rows.wordBytes());
        defer a.free(want);
        const want_scales = try a.alloc(u8, n * one.rows.tableBytes());
        defer a.free(want_scales);
        const want_biases = try a.alloc(u8, n * one.rows.tableBytes());
        defer a.free(want_biases);
        try searchedGather(&one.rows, ids, want, want_scales, want_biases);
        try std.testing.expectEqualSlices(u8, want, got.words);
        try std.testing.expectEqualSlices(u8, want_scales, got.scales);
        try std.testing.expectEqualSlices(u8, want_biases, got.biases);
    }
    try std.testing.expectEqual(@as(usize, max_parts - 1), many.workers.pool.?.spawned);
}

test "spans: one byte range for each of the three tensors of every shard" {
    const a = std.testing.allocator;
    var ids: [4]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i);
    var f = try Fake.init(a, &.{ 2, 2, 2 }, 160, 4, 32);
    defer f.deinit(a);
    const ranges = try f.rows.spans(a);
    defer a.free(ranges);
    try std.testing.expectEqual(@as(usize, 9), ranges.len);
    for (ranges, 0..) |r, i| {
        const shard = f.shards[i / 3];
        try std.testing.expectEqual(r.ptr, switch (i % 3) {
            0 => shard.words.ptr,
            1 => shard.scales.ptr,
            else => shard.biases.ptr,
        });
    }
}
