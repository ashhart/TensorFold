//! Engram's host side: table layout, n-gram hashes of a window's tokens, a rank's heads, and the synthetic index rows.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel

pub const dead_id: i64 = -1;
pub const max_layers = 4;
pub const max_ngram = 8;
pub const max_cols = 64;

/// Where a window's table rows come from (local NVMe shards or a remote service); every source gives the same bytes.
pub const Source = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Starts fetching Engram layer index `li`'s rows `idx` (this rank's heads' global rows); returns at once.
        issue: *const fn (ptr: *anyopaque, li: usize, idx: []const i64) anyerror!void,
        /// Layer index `li`'s rows `idx` as bf16 bits [idx.len, dim] (waits for any still in flight).
        rows: *const fn (ptr: *anyopaque, li: usize, idx: []const i64, dim: usize, out: []u16) anyerror!void,
    };
};

pub const Host = struct {
    layer_ids: []const u32,
    max_ngram: u32,
    heads: u32,
    /// token id -> compressed id
    token_map: []const u32,
    pad_id: i64,
    map_bytes: ?[]const u8 = null,
    mult: [max_layers][max_ngram]i64 = undefined,
    primes: [max_layers][max_cols]i64 = undefined,
    offsets: [max_layers][max_cols]i64 = undefined,

    pub fn cols(h: *const Host) u32 {
        return (h.max_ngram - 1) * h.heads;
    }

    /// Prime bucket sizes: for each layer and order the next unused primes above `vocab_size - 1` (reference order).
    pub fn layout(h: *Host, vocab_size: u64) void {
        var seen: [max_layers * max_cols]i64 = undefined;
        var nseen: usize = 0;
        for (0..h.layer_ids.len) |li| {
            var col: usize = 0;
            for (0..h.max_ngram - 1) |_| {
                var cur: i64 = @intCast(vocab_size - 1);
                for (0..h.heads) |_| {
                    cur += 1;
                    while (!isPrime(@intCast(cur)) or std.mem.indexOfScalar(i64, seen[0..nseen], cur) != null) cur += 1;
                    seen[nseen] = cur;
                    nseen += 1;
                    h.primes[li][col] = cur;
                    col += 1;
                }
            }
            var acc: i64 = 0;
            for (0..col) |c| {
                h.offsets[li][c] = acc;
                acc += h.primes[li][c];
            }
        }
    }

    /// Table rows [t][layers][cols] of `ids` after `lookback` (oldest first; short: sequence start); past map: dead.
    pub fn hash(h: *const Host, ids: []const u32, lookback: []const u32, out: []i64) void {
        const nb = lookback.len;
        const L = h.layer_ids.len;
        const C = h.cols();
        std.debug.assert(out.len == ids.len * L * C);
        for (0..ids.len) |t| for (0..L) |li| {
            var blocked = false;
            var tokens: [max_ngram]u32 = undefined;
            for (0..h.max_ngram) |shift| {
                const j = @as(i64, @intCast(t + nb)) - @as(i64, @intCast(shift));
                const before = j < 0;
                const v: i64 = if (before) h.pad_id else h.src(if (j < @as(i64, @intCast(nb)))
                    lookback[@intCast(j)]
                else
                    ids[@as(usize, @intCast(j)) - nb]);
                blocked = blocked or before or v == dead_id;
                tokens[shift] = @intCast(if (blocked) h.pad_id else v);
            }
            @import("engram_hash.zig").hashValues(tokens[0..h.max_ngram], h.mult[li][0..h.max_ngram], h.primes[li][0..C], h.offsets[li][0..C], h.heads, out[(t * L + li) * C ..][0..C]);
        };
    }

    fn src(h: *const Host, id: u32) i64 {
        if (h.map_bytes) |bytes| {
            if (id >= bytes.len / 4) return dead_id;
            return std.mem.readInt(u32, bytes[@as(usize, id) * 4 ..][0..4], .little);
        }
        return if (id >= h.token_map.len) dead_id else h.token_map[id];
    }

    /// (first head, heads) rank `rank` of `world` owns: complete heads, TP-major (head_shard).
    pub fn headShard(h: *const Host, rank: u32, world: u32) [2]u32 {
        const n = h.cols();
        const part = (n + world - 1) / world;
        const h0 = @min(rank * part, n);
        return .{ h0, @min(h0 + part, n) - h0 };
    }

    /// This rank's heads' global rows of Engram layer `layer` for a window's `ids` after `lookback`: idx [n][heads].
    pub fn shardIndex(h: *const Host, a: std.mem.Allocator, ids: []const u32, lookback: []const u32, layer: u32, rank: u32, world: u32) ![]i64 {
        if (world == 0 or rank >= world) return error.BadEngramHost;
        const li = std.mem.indexOfScalar(u32, h.layer_ids, layer) orelse return error.NotEngram;
        const L = h.layer_ids.len;
        const C = h.cols();
        const sh = h.headShard(rank, world);
        const hashes = try a.alloc(i64, ids.len * L * C);
        defer a.free(hashes);
        h.hash(ids, lookback, hashes);
        const idx = try a.alloc(i64, ids.len * sh[1]);
        for (0..ids.len) |t| @memcpy(idx[t * sh[1] ..][0..sh[1]], hashes[(t * L + li) * C + sh[0] ..][0..sh[1]]);
        return idx;
    }

    /// Engram layer `layer`'s window rows, this rank's heads, bf16 [ids.len, heads x dim]: from `source` or index rows.
    pub fn windowRows(h: *const Host, a: std.mem.Allocator, source: ?Source, ids: []const u32, lookback: []const u32, layer: u32, rank: u32, world: u32, dim: usize, out: []u16) !void {
        const idx = try h.shardIndex(a, ids, lookback, layer, rank, world);
        defer a.free(idx);
        const s = source orelse return indexRows(idx, dim, out);
        const li = std.mem.indexOfScalar(u32, h.layer_ids, layer).?;
        return s.vtable.rows(s.ptr, li, idx, dim, out);
    }

    /// Starts reading a coming window's rows on every Engram layer; a wrong guess costs a read; errors only logged.
    pub fn prefetch(h: *const Host, a: std.mem.Allocator, source: ?Source, ids: []const u32, lookback: []const u32, rank: u32, world: u32) void {
        const s = source orelse return;
        for (h.layer_ids, 0..) |layer, li| {
            const idx = h.shardIndex(a, ids, lookback, layer, rank, world) catch return;
            defer a.free(idx);
            s.vtable.issue(s.ptr, li, idx) catch |e| if (!quiet) std.log.scoped(.dsv41).warn("engram: prefetch of layer {d} failed ({t})", .{ layer, e });
        }
    }

    pub fn fromTokenMap(map: @import("engram_hash.zig").TokenMap, layer_buf: []u32) !Host {
        const hash_api = @import("engram_hash.zig");
        if (layer_buf.len < hash_api.layers.len) return error.BadEngramHost;
        var h: Host = .{ .layer_ids = layer_buf[0..hash_api.layers.len], .max_ngram = hash_api.order, .heads = hash_api.heads, .token_map = &.{}, .map_bytes = map.bytes, .pad_id = hash_api.pad_id };
        for (hash_api.layers, 0..) |meta, i| {
            layer_buf[i] = meta.layer_id;
            for (meta.multipliers, 0..) |v, j| h.mult[i][j] = @intCast(v);
            for (meta.primes, 0..) |v, j| h.primes[i][j] = v;
            for (meta.offsets, 0..) |v, j| h.offsets[i][j] = v;
        }
        return h;
    }

    /// Loads recipe/dsv41_engram_host.py's file (the map borrowed from `bytes`, which must outlive the host).
    pub fn load(bytes: []const u8, heads: u32, pad_token: u32, vocab_size: u64, layer_buf: []u32) !Host {
        if (heads != 8 or pad_token != 2 or vocab_size != 16000000) return error.BadEngramHost;
        return fromTokenMap(try @import("engram_hash.zig").TokenMap.parse(bytes), layer_buf);
    }

    // Legacy fixture adapter only. Production load requires the pinned canonical map above.
    pub fn loadLegacyFixture(bytes: []const u8, heads: u32, pad_token: u32, vocab_size: u64, layer_buf: []u32) !Host {
        if (bytes.len < 24 or !std.mem.eql(u8, bytes[0..8], "DSV41EH1")) return error.BadEngramHost;
        const L = std.mem.readInt(u32, bytes[8..12], .little);
        const M = std.mem.readInt(u32, bytes[12..16], .little);
        const V = std.mem.readInt(u32, bytes[16..20], .little);
        if (L == 0 or L > max_layers or M < 2 or M > max_ngram or heads == 0 or heads > max_cols / (M - 1) or L > layer_buf.len or pad_token >= V or vocab_size < 2 or vocab_size > 16000000) return error.BadEngramHost;
        const expected = 24 + 4 * @as(usize, L) + 8 * @as(usize, L) * M + 4 * @as(usize, V);
        if (bytes.len != expected) return error.BadEngramHost;
        var at: usize = 24;
        for (0..L) |i| layer_buf[i] = std.mem.readInt(u32, bytes[at + 4 * i ..][0..4], .little);
        at += 4 * L;
        var h: Host = .{ .layer_ids = layer_buf[0..L], .max_ngram = M, .heads = heads, .token_map = &.{}, .pad_id = 0 };
        for (0..L) |li| for (0..M) |s| {
            h.mult[li][s] = std.mem.readInt(i64, bytes[at..][0..8], .little);
            if (h.mult[li][s] <= 0) return error.BadEngramHost;
            at += 8;
        };
        if (bytes.len - at < 4 * @as(usize, V)) return error.BadEngramHost;
        h.map_bytes = bytes[at..];
        for (0..V) |id| {
            if (std.mem.readInt(u32, bytes[at + 4 * id ..][0..4], .little) >= 99092) return error.BadEngramHost;
        }
        h.pad_id = h.src(pad_token);
        h.layout(vocab_size);
        return h;
    }
};

/// The committed ids before `start` (last `keep_n`): the tail, then kept leading rows of `pending`; null: unreachable.
pub fn lookbackAt(tail: []const u32, pos: u64, pending: ?[]const u32, start: u64, keep_n: usize, buf: []u32) ?[]const u32 {
    if (start < pos) return null;
    const k: usize = @intCast(start - pos);
    const extra: []const u32 = if (pending) |p| (if (k <= p.len) p[0..k] else return null) else if (k == 0) &.{} else return null;
    if (tail.len + extra.len > buf.len) return null;
    @memcpy(buf[0..tail.len], tail);
    @memcpy(buf[tail.len..][0..extra.len], extra);
    const m = tail.len + extra.len;
    const n = @min(m, keep_n);
    return buf[m - n .. m];
}

/// A slot's Engram tail after keeping `ids`: the last `keep_n` of tail + ids, into `tail`; returns its new length.
pub fn tailAfter(tail: []u32, len: usize, keep_n: usize, ids: []const u32) usize {
    var all: [2 * max_ngram + 64]u32 = undefined;
    const old = tail[0..len];
    const take = ids[ids.len -| keep_n..];
    @memcpy(all[0..old.len], old);
    @memcpy(all[old.len..][0..take.len], take);
    const m = old.len + take.len;
    const k = @min(m, keep_n);
    @memcpy(tail[0..k], all[m - k .. m]);
    return k;
}

test "an Engram warm's lookback at a draft before its window's keep is the window's tail after it" {
    // TF_DSV41_ENGRAM_WARM: warmEngram asks at pos + kept, pending window in flight; stageEngram reads after the keep
    var prng = std.Random.DefaultPrng.init(56);
    const rnd = prng.random();
    for (0..500) |_| {
        const keep_n = 3;
        var tail: [max_ngram]u32 = undefined;
        const len = rnd.uintAtMost(usize, keep_n);
        for (tail[0..len]) |*t| t.* = rnd.int(u32);
        var pend: [6]u32 = undefined;
        const pn = 1 + rnd.uintLessThan(usize, pend.len);
        for (pend[0..pn]) |*t| t.* = rnd.int(u32);
        const pos: u64 = 100 + rnd.uintAtMost(u64, 50);
        const kept = rnd.uintLessThan(usize, pn); // drafts kept: rows 0..kept, the next window starts past them
        const start = pos + kept + 1;
        var buf: [2 * max_ngram + 64]u32 = undefined;
        const before = lookbackAt(tail[0..len], pos, pend[0..pn], start, keep_n, &buf).?;
        var after = tail;
        const alen = tailAfter(&after, len, keep_n, pend[0 .. kept + 1]);
        try std.testing.expectEqualSlices(u32, after[0..alen], before);
        // a keep already done (the commit before the draft): no pending window, the same tail
        var buf2: [2 * max_ngram + 64]u32 = undefined;
        try std.testing.expectEqualSlices(u32, after[0..alen], lookbackAt(after[0..alen], start, null, start, keep_n, &buf2).?);
    }
}

/// Small primes, then Miller-Rabin with bases 2, 7, 61 (exact below 4.7e9).
pub fn isPrime(n: u64) bool {
    if (n < 2) return false;
    for ([_]u64{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 }) |p| if (n % p == 0) return n == p;
    var d = n - 1;
    var r: u32 = 0;
    while (d % 2 == 0) : (r += 1) d /= 2;
    for ([_]u64{ 2, 7, 61 }) |a| {
        var x = powMod(a, d, n);
        if (x == 1 or x == n - 1) continue;
        var ok = false;
        var i: u32 = 1;
        while (i < r) : (i += 1) {
            x = @intCast(@as(u128, x) * x % n);
            if (x == n - 1) {
                ok = true;
                break;
            }
        }
        if (!ok) return false;
    }
    return true;
}

fn powMod(b0: u64, e0: u64, m: u64) u64 {
    var b: u128 = b0 % m;
    var e = e0;
    var r: u128 = 1;
    while (e > 0) : (e >>= 1) {
        if (e & 1 == 1) r = r * b % m;
        b = b * b % m;
    }
    return @intCast(r);
}

/// Synthetic rows: row i, value j = (((i * 2654435761 + j * 40503) % 511) - 255) / 1024, exact in bf16, into `out`.
pub fn indexRows(rows: []const i64, dim: usize, out: []u16) void {
    for (rows, 0..) |i, r| for (0..dim) |j| {
        const v = @mod(i *% 2654435761 +% @as(i64, @intCast(j)) * 40503, 511) - 255;
        const f: f32 = @as(f32, @floatFromInt(v)) / 1024.0;
        out[r * dim + j] = @truncate(@as(u32, @bitCast(f)) >> 16);
    };
}

const testing = std.testing;

test "the n-gram hashes: the reference NgramHasher on the release config and tokenizer (fixtures/engram-hash-ref.bin)" {
    const a = testing.allocator;
    const fx: []const u8 = @embedFile("fixtures/engram-hash-ref.bin");
    const L = std.mem.readInt(u32, fx[8..12], .little);
    const M = std.mem.readInt(u32, fx[12..16], .little);
    const V = std.mem.readInt(u32, fx[16..20], .little);
    var at: usize = 24 + 4 * L + 8 * L * M;
    // a full-size map holding only the entries the fixture's ids use
    const map = try a.alloc(u32, V);
    defer a.free(map);
    @memset(map, 0);
    const used = std.mem.readInt(u32, fx[at..][0..4], .little);
    at += 4;
    for (0..used) |_| {
        map[std.mem.readInt(u32, fx[at..][0..4], .little)] = std.mem.readInt(u32, fx[at + 4 ..][0..4], .little);
        at += 8;
    }
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    try file.appendSlice(a, fx[0 .. 24 + 4 * L + 8 * L * M]);
    try file.appendSlice(a, std.mem.sliceAsBytes(map));
    const bytes = try a.alignedAlloc(u8, .@"4", file.items.len);
    defer a.free(bytes);
    @memcpy(bytes, file.items);
    var ids_buf: [max_layers]u32 = undefined;
    const h = try Host.loadLegacyFixture(bytes, 8, 2, 16_000_000, &ids_buf);
    // the reference layout: the first and last primes, the last head's offset
    try testing.expectEqual(@as(i64, 16_000_057), h.primes[0][0]);
    try testing.expectEqual(@as(i64, 16_000_889), h.primes[1][23]);
    try testing.expectEqual(@as(i64, 368_005_705), h.offsets[0][23]);
    const cases = std.mem.readInt(u32, fx[at..][0..4], .little);
    at += 4;
    for (0..cases) |_| {
        const n = std.mem.readInt(u32, fx[at..][0..4], .little);
        const nb = std.mem.readInt(u32, fx[at + 4 ..][0..4], .little);
        at += 8;
        const ids_c = try a.alloc(u32, n);
        defer a.free(ids_c);
        for (ids_c, 0..) |*x, i| x.* = std.mem.readInt(u32, fx[at + 4 * i ..][0..4], .little);
        at += 4 * n;
        const lb_c = try a.alloc(u32, nb);
        defer a.free(lb_c);
        for (lb_c, 0..) |*x, i| x.* = std.mem.readInt(u32, fx[at + 4 * i ..][0..4], .little);
        at += 4 * nb;
        const got = try a.alloc(i64, n * L * h.cols());
        defer a.free(got);
        h.hash(ids_c, lb_c, got);
        for (got, 0..) |g, i| try testing.expectEqual(std.mem.readInt(i64, fx[at + 8 * i ..][0..8], .little), g);
        at += 8 * got.len;
    }
    try testing.expectEqual(fx.len, at);
    try testing.expectEqual([2]u32{ 12, 12 }, h.headShard(1, 2));
}
