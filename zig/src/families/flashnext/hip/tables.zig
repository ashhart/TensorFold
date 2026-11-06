//! The n-gram tables in host memory (the checkpoint's mapped shards): a window's rows gathered for the device.
const std = @import("std");
const core = @import("core");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const ngram = @import("ngram.zig");

/// One shard's packed rows and their group tables, as mapped.
pub const Shard = struct { words: []const u8, scales: []const u8, biases: []const u8 };

/// A gather copies the checkpoint's rows in id order: a staging too small, or a row past the last shard.
const Error = error{ StagingTooSmall, NgramRowOutOfRange };

/// Threads a gather takes at most, and rows a thread takes at least before the gather spreads.
const max_workers = 8;
const rows_per_worker = 256;

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

    /// Every byte a gather reads: the pages a checkpoint release over the weights must leave mapped.
    pub fn spans(r: *const Rows, a: std.mem.Allocator) ![]const []const u8 {
        const out = try a.alloc([]const u8, r.shards.len * 3);
        for (r.shards, 0..) |s, i| {
            out[i * 3] = s.words;
            out[i * 3 + 1] = s.scales;
            out[i * 3 + 2] = s.biases;
        }
        return out;
    }

    /// Each id's packed words, scales and biases copied into the three staging slices, in id order.
    pub fn gather(r: *const Rows, ids: []const u32, words: []u8, scales: []u8, biases: []u8) !void {
        const wb = r.wordBytes();
        const tb = r.tableBytes();
        if (words.len < ids.len * wb or scales.len < ids.len * tb or biases.len < ids.len * tb)
            return error.StagingTooSmall;
        const workers = @min(max_workers, @max(1, ids.len / rows_per_worker));
        const per = (ids.len + workers - 1) / workers;
        var results: [max_workers]Error!void = @splat({});
        var threads: [max_workers]?std.Thread = @splat(null);
        for (1..workers) |w| {
            const last = @min(ids.len, (w + 1) * per);
            const job = .{ &results[w], r, ids, words, scales, biases, w * per, last };
            threads[w] = std.Thread.spawn(.{}, run, job) catch null;
        }
        run(&results[0], r, ids, words, scales, biases, 0, @min(ids.len, per));
        for (1..workers) |w| {
            const last = @min(ids.len, (w + 1) * per);
            if (threads[w]) |th| th.join() else run(&results[w], r, ids, words, scales, biases, w * per, last);
        }
        for (results[0..workers]) |res| try res;
    }
};

/// Ids [first, last) into their staging rows: every thread copies its own rows, so no two write one.
fn run(
    out: *Error!void,
    r: *const Rows,
    ids: []const u32,
    words: []u8,
    scales: []u8,
    biases: []u8,
    first: usize,
    last: usize,
) void {
    out.* = gatherRows(r, ids, words, scales, biases, first, last);
}

fn gatherRows(
    r: *const Rows,
    ids: []const u32,
    words: []u8,
    scales: []u8,
    biases: []u8,
    first: usize,
    last: usize,
) Error!void {
    const wb = r.wordBytes();
    const tb = r.tableBytes();
    for (ids[first..last], first..) |id, i| {
        const at = try r.locate(id);
        const sh = r.shards[at.shard];
        @memcpy(words[i * wb ..][0..wb], sh.words[at.row * wb ..][0..wb]);
        @memcpy(scales[i * tb ..][0..tb], sh.scales[at.row * tb ..][0..tb]);
        @memcpy(biases[i * tb ..][0..tb], sh.biases[at.row * tb ..][0..tb]);
    }
}

pub const Tables = struct {
    rows: Rows,
    hasher: ngram.Hasher,

    pub fn wordBytes(t: Tables) usize {
        return t.rows.wordBytes();
    }

    pub fn tableBytes(t: Tables) usize {
        return t.rows.tableBytes();
    }

    /// The tables of the PLE layer `layer` from the mapped checkpoint; the shards' rows as the reference splits them.
    pub fn init(
        a: std.mem.Allocator,
        ck: *core.checkpoint.Checkpoint,
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
        return .{
            .rows = .{
                .shards = shards,
                .starts = starts,
                .wide = sizes.wide,
                .narrow = sizes.narrow,
                .extra = sizes.extra,
                .dims = dims,
                .bits = spec.?.bits,
                .group = spec.?.group,
            },
            .hasher = try ngram.Hasher.init(c),
        };
    }

    /// Each id's packed words, scales and biases copied into the three staging slices, in id order.
    pub fn gather(t: *const Tables, ids: []const u32, words: []u8, scales: []u8, biases: []u8) !void {
        return t.rows.gather(ids, words, scales, biases);
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

    fn init(a: std.mem.Allocator, sizes: []const usize, dims: usize, bits: u8, group: usize) !Fake {
        var f: Fake = .{
            .shards = try a.alloc(Shard, sizes.len),
            .starts = try a.alloc(usize, sizes.len + 1),
            .arrays = .{ &.{}, &.{}, &.{} },
            .rows = undefined,
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
        return f;
    }

    fn deinit(f: *Fake, a: std.mem.Allocator) void {
        for (f.arrays) |arr| a.free(arr);
        a.free(f.shards);
        a.free(f.starts);
    }

    fn gathered(f: *const Fake, a: std.mem.Allocator, ids: []const u32) !Staging {
        const s: Staging = .{
            .words = try a.alloc(u8, ids.len * f.rows.wordBytes()),
            .scales = try a.alloc(u8, ids.len * f.rows.tableBytes()),
            .biases = try a.alloc(u8, ids.len * f.rows.tableBytes()),
        };
        @memset(s.words, 0);
        @memset(s.scales, 0);
        @memset(s.biases, 0);
        try f.rows.gather(ids, s.words, s.scales, s.biases);
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
    try std.testing.expectError(error.NgramRowOutOfRange, f.rows.gather(&past, small, want_scales, want_biases));
    const short = small[0 .. small.len - 1];
    try std.testing.expectError(error.StagingTooSmall, f.rows.gather(&ids, short, want_scales, want_biases));
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
