//! flashnext_attn.cu's caches through a page table: prep, pooling, scores and attention, paged == contiguous, float64.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const q_heads = 24;
const kv_heads = 2;
const hd = 256;
const index_heads = 4;
const index_dim = 128;
const half = 32;
const proj_n = 13952;
const ratio = 4;
const idw = 2052;
const nch = 5;
const top = 512;
const eps: f32 = 1e-6;
const layers = 2; // a page's attention layers: the kernels run layer 1, so the layer offset counts

const kv_row = kv_heads * hd * 2;
const index_row = index_dim * 2;

const Part = enum { key, value, index, pooled };

fn rowBytes(p: Part) usize {
    return if (p == .key or p == .value) kv_row else index_row;
}

fn perPage(p: Part, t: usize) usize {
    return if (p == .pooled) t / ratio else t;
}

/// A page's layout, written out apart from cuda_pages.zig: per layer keys, values, indexer keys, then pooled rows.
fn layerBytes(t: usize) usize {
    return t * (2 * kv_row + index_row) + t / ratio * index_row;
}

fn partOffset(t: usize, p: Part) usize {
    return layerBytes(t) * 1 + switch (p) {
        .key => 0,
        .value => t * kv_row,
        .index => 2 * t * kv_row,
        .pooled => 2 * t * kv_row + t * index_row,
    };
}

/// Two streams' cached positions, the rows each adds, positions a page.
const Case = struct { lens: [2]usize, rows: usize, ptok: usize };
const cases = [_]Case{
    .{ .lens = .{ 0, 3 }, .rows = 1, .ptok = 16 },
    .{ .lens = .{ 127, 128 }, .rows = 4, .ptok = 16 },
    .{ .lens = .{ 700, 33 }, .rows = 3, .ptok = 48 }, // a page no 512-key chunk or 32-key tile divides
    .{ .lens = .{ 2100, 2300 }, .rows = 16, .ptok = 128 }, // block scores and dense attention together
    .{ .lens = .{ 4095, 5000 }, .rows = 17, .ptok = 256 },
    .{ .lens = .{ 9000, 300 }, .rows = 2, .ptok = 512 },
};

const Fixture = struct {
    c: Case,
    cap: usize,
    caches: [2][4][]u8, // each stream's logical key, value, indexer key and pooled rows
    p: [2][]u16, // projections [rows][proj_n]
    qw: []f32,
    kw: []f32,
    iw: []f32,
    ikw: []f32,
    inv: []f32,
    ids: [2][]i32, // sparse rows' key lists [rows][idw]
    nk: [2][]i32,
    ids_pages: [2][]const u32, // each stream's pages in the shared arena, shuffled across both
    pages: usize, // the arena's pages (two spare)

    fn pageBytes(f: Fixture) usize {
        return layers * layerBytes(f.c.ptok);
    }

    /// Byte offset of `row` of `part` of stream `st` in the arena under `ids`.
    fn paged(f: Fixture, ids: [2][]const u32, st: usize, part: Part, row: usize) usize {
        const per = perPage(part, f.c.ptok);
        return ids[st][row / per] * f.pageBytes() + partOffset(f.c.ptok, part) + (row % per) * rowBytes(part);
    }

    fn rowsOf(f: Fixture, part: Part) usize {
        return perPage(part, f.cap);
    }

    fn make(a: std.mem.Allocator, c: Case, seed: u64) !Fixture {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const most = @max(c.lens[0], c.lens[1]) + c.rows;
        const cap = (most + c.ptok - 1) / c.ptok * c.ptok;
        var f: Fixture = undefined;
        f.c = c;
        f.cap = cap;
        for (0..2) |st| {
            for (std.enums.values(Part), 0..) |part, pi| {
                f.caches[st][pi] = try a.alloc(u8, f.rowsOf(part) * rowBytes(part));
                fillBf(rng, std.mem.bytesAsSlice(u16, f.caches[st][pi]), 1);
            }
            f.p[st] = try a.alloc(u16, c.rows * proj_n);
            fillBf(rng, f.p[st], 2);
            f.ids[st] = try a.alloc(i32, c.rows * idw);
            f.nk[st] = try a.alloc(i32, c.rows);
            for (0..c.rows) |r| {
                const keys = c.lens[st] + r + 1;
                f.nk[st][r] = @intCast(@max(1, @min(idw, keys) -| r % 3));
                for (0..idw) |i| f.ids[st][r * idw + i] = @intCast((i * 1237 + r * 31 + st * 7) % keys);
            }
        }
        f.qw = try floats(a, rng, hd, 0.5, 1.5);
        f.kw = try floats(a, rng, hd, 0.5, 1.5);
        f.iw = try floats(a, rng, index_dim, 0.5, 1.5);
        f.ikw = try floats(a, rng, index_dim, 0.5, 1.5);
        f.inv = try a.alloc(f32, half);
        for (f.inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / half));
        f.pages = 2 * (cap / c.ptok) + 2;
        const order = try a.alloc(u32, f.pages);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        rng.shuffle(u32, order);
        const n = cap / c.ptok;
        f.ids_pages = .{ order[0..n], order[n .. 2 * n] };
        return f;
    }

    /// Both streams' logical caches at their pages under `ids`; every other byte (layer 0, spare pages) random.
    fn arena(f: Fixture, a: std.mem.Allocator, rng: std.Random, ids: [2][]const u32) ![]u8 {
        const out = try a.alloc(u8, f.pages * f.pageBytes());
        rng.bytes(out);
        for (0..2) |st| for (std.enums.values(Part), 0..) |part, pi| {
            const n = rowBytes(part);
            for (0..f.rowsOf(part)) |row| @memcpy(out[f.paged(ids, st, part, row)..][0..n], f.caches[st][pi][row * n ..][0..n]);
        };
        return out;
    }

    /// Stream `st`'s logical caches read back out of the arena's pages.
    fn unpage(f: Fixture, a: std.mem.Allocator, all: []const u8, ids: [2][]const u32, st: usize) ![4][]u8 {
        var out: [4][]u8 = undefined;
        for (std.enums.values(Part), 0..) |part, pi| {
            const n = rowBytes(part);
            out[pi] = try a.alloc(u8, f.rowsOf(part) * n);
            for (0..f.rowsOf(part)) |row| @memcpy(out[pi][row * n ..][0..n], all[f.paged(ids, st, part, row)..][0..n]);
        }
        return out;
    }
};

fn fillBf(rng: std.Random, out: []align(1) u16, scale: f32) void {
    for (out) |*x| x.* = @truncate(@as(u32, @bitCast((rng.float(f32) * 2 - 1) * scale)) >> 16);
}

fn floats(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f32, hi: f32) ![]f32 {
    const v = try a.alloc(f32, n);
    for (v) |*x| x.* = lo + rng.float(f32) * (hi - lo);
    return v;
}

fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

fn scoreStride(cap: usize) usize {
    return ((cap + ratio - 1) / ratio + 15) / 16 * 16;
}

/// One layout's results a stream: queries, indexer queries, block scores, attention rows and the caches after.
const Result = struct { q: [2][]u8, iq: [2][]u8, sc: [2][]u8, dense: [2]?[]u8, sparse: [2][]u8, caches: [2][4][]u8 };

/// Both streams' rows through prep, pool, scores and attention on layer 1: contiguous, or paged by `tables` (`given`).
fn launch(gpu: check.Gpu, s: cuda.Stream, g: *const flash.Glue, f: Fixture, tables: ?[2][]const u32, given: ?[2][]const u32, a: std.mem.Allocator, seed: u64) !Result {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    const rows = f.c.rows;
    const nb = scoreStride(f.cap);
    const qw = try rig.input(std.mem.sliceAsBytes(f.qw));
    const kw = try rig.input(std.mem.sliceAsBytes(f.kw));
    const iw = try rig.input(std.mem.sliceAsBytes(f.iw));
    const ikw = try rig.input(std.mem.sliceAsBytes(f.ikw));
    const inv = try rig.input(std.mem.sliceAsBytes(f.inv));
    const po = try rig.scratch(rows * nch * q_heads * hd * 4);
    const pm = try rig.scratch(rows * nch * q_heads * 4);
    const pl = try rig.scratch(rows * nch * q_heads * 4);
    var arena_ptr: u64 = 0;
    if (tables) |t| {
        const host = try f.arena(a, prng.random(), t);
        arena_ptr = try rig.scratch(host.len);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = arena_ptr, .len = host.len }).upload(0, host);
    }
    var res: Result = undefined;
    var flat: [2][4]u64 = undefined;
    var outs: [2][5]u64 = undefined;
    for (0..2) |st| {
        const pos0: i32 = @intCast(f.c.lens[st]);
        const ends = f.c.lens[st] + rows;
        const pos = try rig.input(std.mem.sliceAsBytes(&[_]i32{ pos0, 0, 0, 0 }));
        const p = try rig.input(std.mem.sliceAsBytes(f.p[st]));
        const ids = try rig.input(std.mem.sliceAsBytes(f.ids[st]));
        const nk = try rig.input(std.mem.sliceAsBytes(f.nk[st]));
        const ones = try a.alloc(i32, rows);
        @memset(ones, 1);
        const spr1 = try rig.input(std.mem.sliceAsBytes(ones));
        @memset(ones, 0);
        const spr0 = try rig.input(std.mem.sliceAsBytes(ones));
        var pg: flash.Pg = .{};
        var at: [4]u64 = undefined;
        if (tables) |t| {
            pg = .{ .pt = try rig.input(std.mem.sliceAsBytes((given orelse t)[st])), .ps = @intCast(f.pageBytes() / 2), .ptok = @intCast(f.c.ptok) };
            for (std.enums.values(Part), 0..) |part, pi| at[pi] = arena_ptr + partOffset(f.c.ptok, part);
        } else for (0..4) |pi| {
            at[pi] = try rig.scratch(f.caches[st][pi].len);
            try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = at[pi], .len = f.caches[st][pi].len }).upload(0, f.caches[st][pi]);
        }
        flat[st] = at;
        outs[st] = .{ try rig.output(rows * q_heads * hd * 2), try rig.output(rows * index_heads * index_dim * 2), try rig.output(rows * nb * 4), try rig.output(rows * q_heads * hd * 2), try rig.output(rows * q_heads * hd * 2) };
        const o = outs[st];
        try g.attnPrep(s, p, pos, qw, kw, iw, inv, o[0], at[0], at[1], o[1], at[2], eps, rows, pg);
        try g.qsaPool(s, at[2], at[3], pos, ikw, inv, eps, rows, pg);
        if (ends / ratio > top) try g.qsaScores(s, o[1], at[3], pos, o[2], nb, rows, @min(nb, (ends + ratio - 1) / ratio), pg);
        if (ends <= nch * 512) {
            try g.attnChunks(s, o[0], at[0], at[1], pos, po, pm, pl, ids, nk, spr0, rows, kv_heads, (ends + 511) / 512, true, pg);
            try g.attnMerge(s, po, pm, pl, pos, o[3], nk, spr0, rows, kv_heads, true);
        }
        try g.attnChunks(s, o[0], at[0], at[1], pos, po, pm, pl, ids, nk, spr1, rows, kv_heads, (idw + 511) / 512, true, pg);
        try g.attnMerge(s, po, pm, pl, pos, o[4], nk, spr1, rows, kv_heads, true);
    }
    for (0..2) |st| {
        const o = outs[st];
        res.q[st] = try a.dupe(u8, try rig.bytes(o[0], rows * q_heads * hd * 2));
        res.iq[st] = try a.dupe(u8, try rig.bytes(o[1], rows * index_heads * index_dim * 2));
        res.sc[st] = try a.dupe(u8, try rig.bytes(o[2], rows * nb * 4));
        res.dense[st] = if (f.c.lens[st] + rows <= nch * 512) try a.dupe(u8, try rig.bytes(o[3], rows * q_heads * hd * 2)) else null;
        res.sparse[st] = try a.dupe(u8, try rig.bytes(o[4], rows * q_heads * hd * 2));
    }
    if (tables) |t| {
        const all = try a.dupe(u8, try rig.bytes(arena_ptr, f.pages * f.pageBytes()));
        for (0..2) |st| res.caches[st] = try f.unpage(a, all, t, st);
    } else for (0..2) |st| for (0..4) |pi| {
        res.caches[st][pi] = try a.dupe(u8, try rig.bytes(flat[st][pi], f.caches[st][pi].len));
    };
    try check.expect(try rig.intact() and !rig.overlap(), "flashnext pages: a launch wrote past a buffer", .{});
    return res;
}

/// Each attention row against float64 softmax(q k / 16) v over the caches the pass left (p . v takes p as bf16).
fn agrees(f: Fixture, r: Result, a: std.mem.Allocator) !bool {
    const most = @max(f.c.lens[0], f.c.lens[1]) + f.c.rows + idw;
    const sc = try a.alloc(f64, most);
    for (0..2) |st| for ([_]bool{ false, true }) |sparse| {
        const out_bytes = if (sparse) r.sparse[st] else r.dense[st] orelse continue;
        const out: []align(1) const u16 = std.mem.bytesAsSlice(u16, out_bytes);
        const q: []align(1) const u16 = std.mem.bytesAsSlice(u16, r.q[st]);
        const kc: []align(1) const u16 = std.mem.bytesAsSlice(u16, r.caches[st][0]);
        const vc: []align(1) const u16 = std.mem.bytesAsSlice(u16, r.caches[st][1]);
        for (0..f.c.rows) |row| for (0..q_heads) |h| {
            const kvh = h / (q_heads / kv_heads);
            const n: usize = if (sparse) @intCast(f.nk[st][row]) else f.c.lens[st] + row + 1;
            var hi = -std.math.inf(f64);
            for (0..n) |j| {
                const key: usize = if (sparse) @intCast(f.ids[st][row * idw + j]) else j;
                var dot: f64 = 0;
                for (0..hd) |d| dot += bf(q[(row * q_heads + h) * hd + d]) * bf(kc[(key * kv_heads + kvh) * hd + d]);
                sc[j] = dot * 0.0625;
                hi = @max(hi, sc[j]);
            }
            var l: f64 = 0;
            for (sc[0..n]) |*v| {
                v.* = @exp(v.* - hi);
                l += v.*;
            }
            for (0..hd) |d| {
                var o: f64 = 0;
                var abs: f64 = 0;
                for (0..n) |j| {
                    const key: usize = if (sparse) @intCast(f.ids[st][row * idw + j]) else j;
                    const v = bf(vc[(key * kv_heads + kvh) * hd + d]);
                    o += sc[j] * v;
                    abs += sc[j] * @abs(v);
                }
                const want = o / l;
                const got = bf(out[(row * q_heads + h) * hd + d]);
                if (!(@abs(got - want) <= abs / l * 0x1p-8 + 0.5 * @abs(want) * 0x1p-7 + 1e-6)) return false;
            }
        };
    };
    return true;
}

fn same(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn sameResult(x: Result, y: Result) bool {
    for (0..2) |st| {
        if (!same(x.q[st], y.q[st]) or !same(x.iq[st], y.iq[st]) or !same(x.sc[st], y.sc[st]) or !same(x.sparse[st], y.sparse[st])) return false;
        if ((x.dense[st] == null) != (y.dense[st] == null)) return false;
        if (x.dense[st]) |d| if (!same(d, y.dense[st].?)) return false;
        for (0..4) |pi| if (!same(x.caches[st][pi], y.caches[st][pi])) return false;
    }
    return true;
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    var red_table = false;
    var red_bound = false;
    for (cases, 0..) |c, i| {
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const f = try Fixture.make(a, c, 0xf9a6 + i);
        const flat = try launch(gpu, s, &g, f, null, null, a, 1);
        try check.expect(try agrees(f, flat, a), "flashnext pages: contiguous attention off float64 at lens {any}, rows {d}", .{ c.lens, c.rows });
        for (0..3) |rep| {
            const pg = try launch(gpu, s, &g, f, f.ids_pages, null, a, 2 + rep);
            try check.expect(sameResult(pg, flat), "flashnext pages: paged bytes differ from contiguous at lens {any}, rows {d}, page {d} (repeat {d})", .{ c.lens, c.rows, c.ptok, rep });
        }
        // red controls: stream 1's first and last pages swapped in the table only (the pages stay); a planted row
        const n1 = (c.lens[1] + c.rows + c.ptok - 1) / c.ptok;
        if (n1 >= 2) {
            const swapped = try a.dupe(u32, f.ids_pages[1]);
            std.mem.swap(u32, &swapped[0], &swapped[n1 - 1]);
            const bad = try launch(gpu, s, &g, f, f.ids_pages, .{ f.ids_pages[0], swapped }, a, 9);
            const moved = !same(bad.sparse[1], flat.sparse[1]);
            red_table = red_table or moved;
            try check.expect(moved, "flashnext pages: a swapped page table still gave contiguous bytes (control missed) at lens {any}", .{c.lens});
        }
        var bad_r = flat;
        bad_r.sparse[0] = try a.dupe(u8, flat.sparse[0]);
        const planted: usize = if (c.rows >= 2) q_heads * hd + 5 else 5;
        std.mem.writeInt(u16, bad_r.sparse[0][2 * planted ..][0..2], 0x4040, .little); // 3.0 where |o| <= 1
        const caught = !(try agrees(f, bad_r, a));
        red_bound = red_bound or caught;
        try check.expect(caught, "flashnext pages: a planted attention row passed float64 (control missed)", .{});
        check.pass("flashnext pages: lens {any}, {d} rows a stream, {d}-position pages: paged == contiguous (queries, indexer queries, block scores, dense and sparse rows, stores, pooled), three repeats, float64 bound, guards", .{ c.lens, c.rows, c.ptok });
    }
    try check.expect(red_table and red_bound, "flashnext pages: a control never fired", .{});
    check.pass("flashnext pages: controls RED as planted (swapped table, planted row)", .{});
}
