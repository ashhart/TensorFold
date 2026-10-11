//! Split KV's exchange on two in-process ranks: every row read after the exchange equals the replicated pool's.

const std = @import("std");
const testing = std.testing;
const sessions = @import("sessions");
const tp = @import("tp");
const split = @import("split.zig");

const page = 64; // tokens; ratio 2: 32 rows a page
const rb = 40; // a short row keeps the test fast; the code takes any width
const fams = [_]sessions.Family{.{ .name = "comp.2", .ratio = 2, .row_bytes = rb, .split = true }};
const psh = 5;

/// The replicated pool's row t of slot s: what every exchanged row must equal.
fn value(s: usize, t: usize, i: usize) u8 {
    return @truncate(s *% 61 +% t *% 7 +% i *% 3 +% 1);
}

const Ctx = struct { seed: u64 };

fn rank(comm: tp.Collective, me: u32, ctx: *const Ctx) anyerror!void {
    const gpa = testing.allocator;
    var pool = try sessions.Pool.init(gpa, .{ .families = &fams, .page = page }, 64 * page, 2, me);
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    const lens = [_]u64{ 11 * page + 5, 3 * page, 17 * page + 40 };
    var slots: [3]*sessions.Slot = undefined;
    for (&slots, lens) |*s, n| {
        s.* = try pool.newSlot(24 * page);
        try s.*.ensure(n);
    }
    // each rank writes the rows it owns (the other rank's go to the discard page, as kv_store does)
    for (slots, 0..) |s, si| for (s.mapped(), 0..) |pg, k| {
        const v = host.pageOf(0, pool.localPage(pg));
        if (!pool.owns(pg)) continue;
        for (0..page / 2) |r| for (0..rb) |i| {
            v[r * rb + i] = value(si, k * (page / 2) + r, i);
        };
    };
    const pts = 24;
    var tables: [3 * pts]i32 = undefined;
    for (slots, 0..) |s, si| for (0..pts) |k| {
        tables[si * pts + k] = @intCast(s.localTableAt(@intCast(k)));
    };
    var prng = std.Random.DefaultPrng.init(ctx.seed); // the same selections on both ranks
    const rnd = prng.random();
    var hk: split.HostKernels = .{};
    const x: split.Exchange = .{ .comm = comm, .kernels = hk.kernels() };
    const base = @intFromPtr(host.tensors[0].ptr);

    // dense, row mode: 6 rows over 3 slots, 16 entries each, some -1
    const R = 6;
    const K = 16;
    var sel: [R * K]i32 = undefined;
    var rslot: [R]i32 = undefined;
    for (&rslot, 0..) |*s, r| s.* = @intCast(r % 3);
    for (&sel, 0..) |*t, i| {
        const rows: u32 = @intCast(lens[@intCast(rslot[i / K])] / 2);
        t.* = if (rnd.uintLessThan(u32, 8) == 0) -1 else @intCast(rnd.uintLessThan(u32, rows));
    }
    var send: [R * K * rb]u8 = undefined;
    var recv: [2 * R * K * rb]u8 = undefined;
    var tok: [R * K]i32 = undefined;
    try x.dense(.{ .sel = @intFromPtr(&sel), .rows = R, .k = K, .table = @intFromPtr(&tables), .pts = pts, .rslot = @intFromPtr(&rslot), .psh = psh, .base = base, .row_bytes = rb, .world = 2, .send = @intFromPtr(&send), .tok = @intFromPtr(&tok) }, @intFromPtr(&recv), null);
    for (sel, tok, 0..) |t, o, i| {
        if (t < 0) {
            try testing.expectEqual(@as(i32, -1), o);
            continue;
        }
        const row = recv[@as(usize, @intCast(o)) * rb ..][0..rb];
        for (row, 0..) |b, j| try testing.expectEqual(value(@intCast(rslot[i / K]), @intCast(t), j), b);
    }

    // union over a 40-row prefill segment of slot 2, with a cap that splits it into blocks
    const S = 40;
    var useg: [S * K]i32 = undefined;
    for (&useg) |*t| t.* = if (rnd.uintLessThan(u32, 10) == 0) -1 else @intCast(rnd.uintLessThan(u32, @intCast(lens[2] / 2)));
    var table2: [pts]u32 = undefined;
    for (&table2, 0..) |*t, k| t.* = slots[2].localTableAt(@intCast(k));
    var blocks: std.ArrayList(split.Block) = .empty;
    defer {
        for (blocks.items) |*b| b.deinit(gpa);
        blocks.deinit(gpa);
    }
    try split.planUnion(gpa, .{ .sel = &useg, .k = K, .table = &table2, .psh = psh, .world = 2, .rank = me, .row_bytes = rb, .cap = 3 * 120 * rb }, &blocks);
    try testing.expect(blocks.items.len > 1);
    for (blocks.items) |blk| {
        const s2 = try gpa.alloc(u8, blk.m * rb);
        defer gpa.free(s2);
        const r2 = try gpa.alloc(u8, 2 * blk.m * rb);
        defer gpa.free(r2);
        try x.unionBlock(base, rb, @intFromPtr(blk.phys.ptr), blk.m, @intFromPtr(s2.ptr), @intFromPtr(r2.ptr), null);
        for (useg[blk.a * K .. blk.b * K], blk.tokens) |t, o| {
            if (t < 0) continue;
            const row = r2[@as(usize, @intCast(o)) * rb ..][0..rb];
            for (row, 0..) |b, j| try testing.expectEqual(value(2, @intCast(t), j), b);
        }
    }
}

test "split == replicated: dense row-mode and capped union exchanges on two ranks" {
    for ([_]u64{ 1, 2, 3 }) |seed| try tp.host.run(2, &Ctx{ .seed = seed }, rank);
}

// -- packed (TF_DSV41_KV_SPLIT_COMPACT): packed == dense == replicated behind every token, on 2 and 3 ranks ---------

const pfams = [_]sessions.Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = rb, .split = true },
    .{ .name = "comp.1", .ratio = 1, .row_bytes = rb, .split = true },
};

fn pvalue(f: usize, s: usize, t: usize, i: usize) u8 {
    return @truncate(f *% 101 +% s *% 61 +% t *% 7 +% i *% 3 +% 1);
}

const Kind = enum { random, one_owner, padded };

const PCtx = struct { seed: u64, refuse: u32 = 0 };

fn packedRank(comm: tp.Collective, me: u32, ctx: *const PCtx) anyerror!void {
    const gpa = testing.allocator;
    const W = comm.world();
    var pool = try sessions.Pool.init(gpa, .{ .families = &pfams, .page = page }, 96 * page, W, me);
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    const lens = [_]u64{ 11 * page + 5, 3 * page, 17 * page + 40 };
    var slots: [3]*sessions.Slot = undefined;
    for (&slots, lens) |*s, n| {
        s.* = try pool.newSlot(24 * page);
        try s.*.ensure(n);
    }
    for (pfams, 0..) |f, fi| for (slots, 0..) |s, si| for (s.mapped(), 0..) |pg, k| {
        if (!pool.owns(pg)) continue;
        const v = host.pageOf(@intCast(fi), pool.localPage(pg));
        const per = page / f.ratio;
        for (0..per) |r| for (0..rb) |i| {
            v[r * rb + i] = pvalue(fi, si, k * per + r, i);
        };
    };
    const pts = 24;
    var tables: [3 * pts]i32 = undefined;
    for (slots, 0..) |s, si| for (0..pts) |k| {
        tables[si * pts + k] = @intCast(s.localTableAt(@intCast(k)));
    };
    var hk: split.HostKernels = .{};
    // the knob's three modes: on agrees once (device lengths on every rank, unless one refuses), force packs anywhere
    const off = try split.Exchange.init(comm, hk.kernels(), .off);
    const on = try split.Exchange.init(comm, hk.kernels(), .on);
    const x = try split.Exchange.init(comm, hk.kernels(), .force);
    try testing.expect(!off.packs(4096));
    try testing.expectEqual(ctx.refuse == 0, on.var_ok);
    try testing.expectEqual(ctx.refuse == 0, on.packs(4096));
    try testing.expect(x.packs(4096));
    var prng = std.Random.DefaultPrng.init(ctx.seed); // the same selections on every rank
    const rnd = prng.random();
    const K = 512;
    for (pfams, 0..) |f, fi| for ([_]u32{ 1, 6, 16 }) |R| for (std.enums.values(Kind)) |kind| {
        const n = R * K;
        const per: u32 = page / f.ratio;
        const fpsh: u32 = std.math.log2_int(u32, per);
        const sel = try gpa.alloc(i32, n);
        defer gpa.free(sel);
        var rslot: [16]i32 = undefined;
        for (rslot[0..R], 0..) |*s, r| s.* = @intCast((r + fi) % 3);
        const one: u32 = rnd.uintLessThan(u32, W);
        for (sel, 0..) |*t, i| {
            const rows: u32 = @intCast(lens[@intCast(rslot[i / K])] / f.ratio);
            t.* = switch (kind) {
                .random => if (rnd.uintLessThan(u32, 8) == 0) -1 else @intCast(rnd.uintLessThan(u32, rows)),
                // every entry on one residue: its pages k = one + W j that hold rows
                .one_owner => blk: {
                    const pages = (rows + per - 1) / per;
                    if (one >= pages) break :blk -1;
                    const k = one + W * rnd.uintLessThan(u32, (pages - one + W - 1) / W);
                    const row = k * per + rnd.uintLessThan(u32, per);
                    break :blk if (row < rows) @intCast(row) else -1;
                },
                // a row's count, then -1 (the attention's padding)
                .padded => if (i % K >= (i / K) * 29 % K) -1 else @intCast(rnd.uintLessThan(u32, rows)),
            };
        }
        const send = try gpa.alloc(u8, n * rb);
        defer gpa.free(send);
        const recv_d = try gpa.alloc(u8, W * n * rb);
        defer gpa.free(recv_d);
        const recv_p = try gpa.alloc(u8, W * n * rb);
        defer gpa.free(recv_p);
        @memset(recv_p, 0xEE); // past each owner's rows: never read
        const tok_d = try gpa.alloc(i32, n);
        defer gpa.free(tok_d);
        const tok_p = try gpa.alloc(i32, n);
        defer gpa.free(tok_p);
        var plens: [split.max_world]i32 = @splat(-7);
        const base = @intFromPtr(host.tensors[fi].ptr);
        const da: split.DenseArgs = .{ .sel = @intFromPtr(sel.ptr), .rows = R, .k = K, .table = @intFromPtr(&tables), .pts = pts, .rslot = @intFromPtr(&rslot), .psh = fpsh, .base = base, .row_bytes = rb, .world = W, .send = @intFromPtr(send.ptr), .tok = @intFromPtr(tok_d.ptr) };
        try x.dense(da, @intFromPtr(recv_d.ptr), null);
        var pa = split.PackArgs.of(da, me, @intFromPtr(&plens));
        pa.tok = @intFromPtr(tok_p.ptr);
        if (kind == .padded) try x.window(pa, @intFromPtr(recv_p.ptr), null) else try x.densePacked(pa, @intFromPtr(recv_p.ptr), null);
        // the reference: each owner's running count in list order
        var count: [split.max_world]u32 = @splat(0);
        for (sel, tok_d, tok_p, 0..) |t, od, op, i| {
            if (t < 0) {
                try testing.expectEqual(@as(i32, -1), od);
                try testing.expectEqual(@as(i32, -1), op);
                continue;
            }
            const o = split.ownerOf(@intCast(t), fpsh, W);
            try testing.expectEqual(@as(i32, @intCast(o * n + count[o])), op);
            count[o] += 1;
            const rd = recv_d[@as(usize, @intCast(od)) * rb ..][0..rb];
            const rp = recv_p[@as(usize, @intCast(op)) * rb ..][0..rb];
            for (rd, rp, 0..) |a, b, j| {
                const want = pvalue(fi, @intCast(rslot[i / K]), @intCast(t), j);
                try testing.expectEqual(want, a);
                try testing.expectEqual(want, b);
            }
        }
        for (0..W) |o| try testing.expectEqual(@as(i32, @intCast(count[o] * rb)), plens[o]);
        if (kind == .one_owner) for (0..W) |o| if (o != one) try testing.expectEqual(@as(i32, 0), plens[o]);
        try dedupCheck(x, pa, sel, &rslot, .{ .fi = fi, .K = K, .psh = fpsh, .rows = pts * per, .W = W });
    };
    // TF_DSV41_KV_SPLIT_HISTORY: a one-slot union by history, every family and slot
    for (pfams, 0..) |f, fi| for (slots, 0..) |s, si| {
        const per: u32 = page / f.ratio;
        try historyCheck(x, rnd, .{ .fi = fi, .si = si, .psh = std.math.log2_int(u32, per), .rows = @intCast(lens[si] / f.ratio), .pages = s.len, .W = W, .me = me }, @intFromPtr(host.tensors[fi].ptr), tables[si * pts ..][0..pts]);
    };
}

const HCase = struct { fi: usize, si: usize, psh: u32, rows: u32, pages: u32, W: u32, me: u32 };

/// The history union: rows by token == replicated, each owner's rows within M, M = the largest owner's mapped rows.
fn historyCheck(x: split.Exchange, rnd: std.Random, c: HCase, base: u64, table: []const i32) !void {
    const gpa = testing.allocator;
    const R = 40;
    const K = 16;
    var sel: [R * K]i32 = undefined;
    for (&sel) |*t| t.* = if (rnd.uintLessThan(u32, 9) == 0) -1 else @intCast(rnd.uintLessThan(u32, c.rows));
    const m = split.historyRows(c.pages, c.psh, c.W);
    var most: u32 = 0;
    for (0..c.W) |o| most = @max(most, @as(u32, @intCast((c.pages + c.W - 1 - o) / c.W)) << @intCast(c.psh));
    try testing.expectEqual(most, m);
    const send = try gpa.alloc(u8, m * rb);
    defer gpa.free(send);
    const recv = try gpa.alloc(u8, c.W * m * rb);
    defer gpa.free(recv);
    var tok: [R * K]i32 = undefined;
    try x.history(.{ .sel = @intFromPtr(&sel), .rows = R, .k = K, .table = @intFromPtr(table.ptr), .pages = c.pages, .psh = c.psh, .rank = c.me, .world = c.W, .m = m, .row_bytes = rb, .base = base, .send = @intFromPtr(send.ptr), .tok = @intFromPtr(&tok) }, @intFromPtr(recv.ptr), null);
    for (sel, tok) |t, o| {
        if (t < 0) {
            try testing.expectEqual(@as(i32, -1), o);
            continue;
        }
        const u: u32 = @intCast(t);
        const own = split.ownerOf(u, c.psh, c.W);
        try testing.expect(o >= own * m and o < (own + 1) * m);
        for (recv[@as(usize, @intCast(o)) * rb ..][0..rb], 0..) |b, j| try testing.expectEqual(pvalue(c.fi, c.si, u, j), b);
    }
}

const DCase = struct { fi: usize, K: u32, psh: u32, rows: u64, W: u32 };

/// TF_DSV41_KV_SPLIT_DEDUP: rows by token == replicated, unique keys sent once, one token a key, the map reset.
fn dedupCheck(x0: split.Exchange, pa0: split.PackArgs, sel: []const i32, rslot: []const i32, c: DCase) !void {
    const gpa = testing.allocator;
    const n = sel.len;
    const map_rows = 3 * c.rows;
    const mem = try gpa.alloc(i32, map_rows + 2 * n + (n + 255) / 256 * split.max_world);
    defer gpa.free(mem);
    @memset(mem[0..map_rows], std.math.maxInt(i32));
    const at = @intFromPtr(mem.ptr);
    var x = x0;
    x.dedup = .{ .map = at, .map_rows = map_rows, .first = at + 4 * map_rows, .place = at + 4 * (map_rows + n), .bcnt = at + 4 * (map_rows + 2 * n), .entries = n, .slots = 3 };
    const send = try gpa.alloc(u8, n * rb);
    defer gpa.free(send);
    const recv = try gpa.alloc(u8, c.W * n * rb);
    defer gpa.free(recv);
    @memset(recv, 0xEE);
    const tok = try gpa.alloc(i32, n);
    defer gpa.free(tok);
    var lens: [split.max_world]i32 = @splat(-7);
    var pa = pa0;
    pa.send = @intFromPtr(send.ptr);
    pa.tok = @intFromPtr(tok.ptr);
    pa.lens = @intFromPtr(&lens);
    try x.window(pa, @intFromPtr(recv.ptr), null);
    var seen = std.AutoHashMap(u64, i32).init(gpa);
    defer seen.deinit();
    var unique: [split.max_world]u32 = @splat(0);
    for (sel, tok, 0..) |t, o, i| {
        if (t < 0) {
            try testing.expectEqual(@as(i32, -1), o);
            continue;
        }
        const s: usize = @intCast(rslot[i / c.K]);
        const g = try seen.getOrPut(@as(u64, s) * c.rows + @as(u64, @intCast(t)));
        if (g.found_existing) try testing.expectEqual(g.value_ptr.*, o) else {
            g.value_ptr.* = o;
            unique[split.ownerOf(@intCast(t), c.psh, c.W)] += 1;
        }
        const row = recv[@as(usize, @intCast(o)) * rb ..][0..rb];
        for (row, 0..) |b, j| try testing.expectEqual(pvalue(c.fi, s, @intCast(t), j), b);
    }
    for (0..c.W) |o| try testing.expectEqual(@as(i32, @intCast(unique[o] * rb)), lens[o]);
    for (mem[0..map_rows]) |m| try testing.expectEqual(@as(i32, std.math.maxInt(i32)), m);
}

test "packed == dedup == dense == replicated, history union: 1 / 6 / 16 rows x K 512, ratio 2 and 1, W 2 and 3" {
    for ([_]u64{ 1, 2 }) |seed| {
        try tp.host.run(2, &PCtx{ .seed = seed }, packedRank);
        try tp.host.run(3, &PCtx{ .seed = seed }, packedRank);
    }
    // a rank without device lengths: `on` stays dense on every rank, force still packs (whole strides move)
    try tp.host.runWith(2, .{ .var_refuse = 1 }, &PCtx{ .seed = 3, .refuse = 1 }, packedRank);
}

test "TF_DSV41_KV_SPLIT_COMPACT values" {
    try testing.expectEqual(split.Compact.off, try split.parseCompact(""));
    try testing.expectEqual(split.Compact.off, try split.parseCompact("0"));
    try testing.expectEqual(split.Compact.on, try split.parseCompact(" ON "));
    try testing.expectEqual(split.Compact.on, try split.parseCompact("1"));
    try testing.expectEqual(split.Compact.force, try split.parseCompact("force"));
    try testing.expectError(error.BadCompactMode, split.parseCompact("2"));
}
