//! window_pages.cu at head dim 256 with a window (P4, sliding layers on pages): the ring kernel's bits, nothing early.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rows = @import("window_rows_tests.zig");

const wa = cuda.window_attention;

const d = 256;
/// bf16 NaN: every page slot no key fills, every page no stream holds, and every page below a stream's window.
const nan: u16 = 0x7fc0;

const Case = struct { name: []const u8, h: usize, hk: usize, win: usize, streams: usize, top: usize, rows: usize };

const cases = [_]Case{
    .{ .name = "31B sliding (32/16 heads), window 1024, 5000 positions, 2 streams", .h = 32, .hk = 16, .win = 1024, .streams = 2, .top = 5000, .rows = 17 },
    .{ .name = "E4B sliding (8/2 heads), window 512, 3000 positions, 3 streams", .h = 8, .hk = 2, .win = 512, .streams = 3, .top = 3000, .rows = 13 },
};

/// Positions a page: one decode chunk, and two.
const toks = [_]usize{ 256, 512 };

const Pool = struct { kc: u64, vc: u64, pages: wa.Pages };

/// Stream s's keys in shuffled `tok`-position pages; NaN in every unfilled slot and every page wholly below `low[s]`.
fn paged(b: *rows.Bufs, a: std.mem.Allocator, rng: std.Random, kc: []const u16, vc: []const u16, streams: usize, top: usize, hk: usize, tok: usize, low: []const usize) !Pool {
    const stride = (top + tok - 1) / tok;
    const count = streams * stride + 3; // pages no stream holds
    const row = hk * d;
    const page = tok * row;
    const ids = try a.alloc(i32, count);
    for (ids, 0..) |*v, i| v.* = @intCast(i);
    rng.shuffle(i32, ids);
    const table = try a.alloc(i32, streams * stride);
    for (table, 0..) |*v, i| v.* = ids[i];
    const kp = try a.alloc(u16, count * page);
    const vp = try a.alloc(u16, count * page);
    @memset(kp, nan);
    @memset(vp, nan);
    for (0..streams) |s| for (0..top) |j| {
        if ((j / tok + 1) * tok <= low[s]) continue; // a retired page: never written, all NaN
        const src = (s * top + j) * row;
        const dst = @as(usize, @intCast(table[s * stride + j / tok])) * page + (j % tok) * row;
        @memcpy(kp[dst..][0..row], kc[src..][0..row]);
        @memcpy(vp[dst..][0..row], vc[src..][0..row]);
    };
    return .{ .kc = try b.up(std.mem.sliceAsBytes(kp)), .vc = try b.up(std.mem.sliceAsBytes(vp)), .pages = .{ .table = try b.up(std.mem.sliceAsBytes(table)), .page_elems = page, .tokens = tok, .stride = stride } };
}

fn case(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const wa.WindowAttention, rng: std.Random, c: Case) !void {
    var b: rows.Bufs = .{ .gpu = gpu };
    defer b.deinit();
    const q = try a.alloc(u16, c.rows * c.h * d);
    const kc = try a.alloc(u16, c.streams * c.top * c.hk * d);
    const vc = try a.alloc(u16, kc.len);
    rows.fill(rng, q, kc, vc, d);
    const pos = try a.alloc(i32, c.rows);
    const slot = try a.alloc(i32, c.rows);
    // the first key alone, inside the first window, on page edges, one past the window, scattered, the cache's end
    const at = [_]usize{ 0, 300, 255, 256, 511, 512, c.win - 1, c.win, c.win + 1, 1500, 2047, 2048 };
    for (pos, slot, 0..) |*p, *sl, i| {
        p.* = @intCast(if (i == c.rows - 1) c.top - 1 else if (i < at.len) @min(at[i], c.top - 1) else (i * 7919) % c.top);
        sl.* = @intCast(i % c.streams);
    }
    const low = try a.alloc(usize, c.streams); // each stream's first key any of its rows reads
    @memset(low, c.top);
    for (pos, slot) |p, sl| {
        const u: usize = @intCast(p);
        low[@intCast(sl)] = @min(low[@intCast(sl)], if (u + 1 > c.win) u + 1 - c.win else 0);
    }
    var w: wa.Window = .{ .h = c.h, .hk = c.hk, .d = d, .win = c.win, .ring = c.top, .scale = @floatCast(1 / @sqrt(@as(f64, d))) };
    const dq = try b.up(std.mem.sliceAsBytes(q));
    const dp = try b.up(std.mem.sliceAsBytes(pos));
    const ds = try b.up(std.mem.sliceAsBytes(slot));
    const scratch = try b.out(s, wa.WindowAttention.scratchBytes(w, c.rows));
    const n = c.rows * c.h * d;
    const flat = try b.out(s, n * 2);
    try g.decode(s, dq, try b.up(std.mem.sliceAsBytes(kc)), try b.up(std.mem.sliceAsBytes(vc)), flat, dp, ds, w, scratch, c.rows);
    try s.synchronize();
    const want = try a.alloc(u16, n);
    try b.get(flat, want);
    for (toks) |tok| {
        const pool = try paged(&b, a, rng, kc, vc, c.streams, c.top, c.hk, tok, low);
        w.pg = pool.pages;
        const out = try b.out(s, n * 2);
        try g.decode(s, dq, pool.kc, pool.vc, out, dp, ds, w, scratch, c.rows);
        try s.synchronize();
        const got = try a.alloc(u16, n);
        try b.get(out, got);
        var what: [180]u8 = undefined;
        try check.sameBytes(try std.fmt.bufPrint(&what, "window pages 256 {s}: pages of {d}, the ones below each window NaN, give the ring kernel's bits", .{ c.name, tok }), std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
    }
}

const PCase = struct { name: []const u8, h: usize, hk: usize, win: usize, p0: usize, n: usize };

/// One stream's prompt chunk after p0 cached positions: inside the first window, across it, far past it, mid-page.
const prompt_cases = [_]PCase{
    .{ .name = "31B sliding prompt (32/16 heads), window 1024, 200 rows at 0", .h = 32, .hk = 16, .win = 1024, .p0 = 0, .n = 200 },
    .{ .name = "31B sliding prompt (32/16 heads), window 1024, 512 rows at 900", .h = 32, .hk = 16, .win = 1024, .p0 = 900, .n = 512 },
    .{ .name = "31B sliding prompt (32/16 heads), window 1024, 96 rows at 3013", .h = 32, .hk = 16, .win = 1024, .p0 = 3013, .n = 96 },
    .{ .name = "E4B sliding prompt (8/2 heads), window 512, 300 rows at 2000", .h = 8, .hk = 2, .win = 512, .p0 = 2000, .n = 300 },
};

fn prompt(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const wa.WindowAttention, rng: std.Random, c: PCase, p32: bool) !void {
    var b: rows.Bufs = .{ .gpu = gpu };
    defer b.deinit();
    const top = c.p0 + c.n;
    const q = try a.alloc(u16, c.n * c.h * d);
    const kc = try a.alloc(u16, top * c.hk * d);
    const vc = try a.alloc(u16, kc.len);
    rows.fill(rng, q, kc, vc, d);
    var w: wa.Window = .{ .h = c.h, .hk = c.hk, .d = d, .win = c.win, .ring = top, .scale = @floatCast(1 / @sqrt(@as(f64, d))), .p32 = p32 };
    const dq = try b.up(std.mem.sliceAsBytes(q));
    const n = c.n * c.h * d;
    const flat = try b.out(s, n * 2);
    try g.promptRows(s, dq, try b.up(std.mem.sliceAsBytes(kc)), try b.up(std.mem.sliceAsBytes(vc)), flat, c.p0, c.n, w, null);
    try s.synchronize();
    const want = try a.alloc(u16, n);
    try b.get(flat, want);
    const low = [_]usize{if (c.p0 + 1 > c.win) c.p0 + 1 - c.win else 0}; // the chunk's first row's first key
    for (toks) |tok| {
        const pool = try paged(&b, a, rng, kc, vc, 1, top, c.hk, tok, &low);
        w.pg = pool.pages; // the prompt kernel's table is the stream's own row
        const out = try b.out(s, n * 2);
        try g.promptRows(s, dq, pool.kc, pool.vc, out, c.p0, c.n, w, null);
        try s.synchronize();
        const got = try a.alloc(u16, n);
        try b.get(out, got);
        var what: [180]u8 = undefined;
        const p = if (p32) "P32" else "P as bf16";
        try check.sameBytes(try std.fmt.bufPrint(&what, "window pages 256 {s}, {s}: pages of {d}, the ones below the window NaN, give the ring kernel's bits", .{ c.name, p, tok }), std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try wa.WindowAttention.load(gpu.d);
    defer g.unload();
    try g.loadPages(gpu.d);
    var prng = std.Random.DefaultPrng.init(0x51de);
    for (cases) |c| {
        _ = arena.reset(.retain_capacity);
        try case(gpu, arena.allocator(), s, &g, prng.random(), c);
    }
    for (prompt_cases) |c| for ([_]bool{ false, true }) |p32| {
        _ = arena.reset(.retain_capacity);
        try prompt(gpu, arena.allocator(), s, &g, prng.random(), c, p32);
    };
    check.pass("window pages 256: windowed decode over shuffled pages of 256 and 512 and prompt rows (P as bf16 and P32), NaN in every unused slot and every page below each window, bit-equal to the ring kernel", .{});
}
