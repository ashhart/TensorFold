//! window_pages.cu at head dim 512 (Gemma 4's global layers): paged decode and prompt rows give the contiguous bits.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rows = @import("window_rows_tests.zig");

const wa = cuda.window_attention;

const d = 512;
/// bf16 NaN: every page slot no key fills, and every page no stream holds.
const nan: u16 = 0x7fc0;

const Case = struct { name: []const u8, h: usize, hk: usize, streams: usize, top: usize, rows: usize };

const cases = [_]Case{
    .{ .name = "12B global (16/1 heads of 512), 5000 positions", .h = 16, .hk = 1, .streams = 1, .top = 5000, .rows = 17 },
    .{ .name = "31B global (32/4 heads of 512), 3000 positions, 2 streams", .h = 32, .hk = 4, .streams = 2, .top = 3000, .rows = 17 },
    .{ .name = "GQA 5 (10/2 heads of 512), 1200 positions, 3 streams", .h = 10, .hk = 2, .streams = 3, .top = 1200, .rows = 9 },
};

/// Positions a page: a 16-byte row's worth of keys, an odd count whose pages a chunk and a staged tile cross, a chunk.
const toks = [_]usize{ 16, 37, 256 };

const Pool = struct { kc: u64, vc: u64, pages: wa.Pages };

/// Stream s's keys (rows s * top .. of kc and vc) copied to `tok`-position pages in shuffled order, NaN elsewhere.
fn paged(b: *rows.Bufs, a: std.mem.Allocator, rng: std.Random, kc: []const u16, vc: []const u16, streams: usize, top: usize, hk: usize, tok: usize) !Pool {
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
    for (pos, slot, 0..) |*p, *sl, i| { // the first key alone, short rows, chunk edges, scattered, the cache's end
        p.* = @intCast(if (i == 0) 0 else if (i == c.rows - 1) c.top - 1 else if (i % 4 == 0) (i * 64 - 1) % c.top else (i * 7919) % c.top);
        sl.* = @intCast(i % c.streams);
    }
    // row 1's own key holds +inf in a value: its heads of key head 0 run the merge's float64 replay over the keys
    vc[((@as(usize, @intCast(slot[1])) * c.top + @as(usize, @intCast(pos[1]))) * c.hk) * d + 7] = 0x7f80;
    var w: wa.Window = .{ .h = c.h, .hk = c.hk, .d = d, .win = 0, .ring = c.top, .scale = @floatCast(1 / @sqrt(@as(f64, d))) };
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
    var odd: usize = 0;
    for (want) |x| odd += @intFromBool(x & 0x7f80 == 0x7f80);
    try check.expect(odd > 0, "window pages 512 {s}: the planted +inf reaches the merge's replay", .{c.name});
    for (toks) |tok| {
        const pool = try paged(&b, a, rng, kc, vc, c.streams, c.top, c.hk, tok);
        w.pg = pool.pages;
        const out = try b.out(s, n * 2);
        try g.decode(s, dq, pool.kc, pool.vc, out, dp, ds, w, scratch, c.rows);
        try s.synchronize();
        const got = try a.alloc(u16, n);
        try b.get(out, got);
        var what: [160]u8 = undefined;
        try check.sameBytes(try std.fmt.bufPrint(&what, "window pages 512 {s}: pages of {d} give the contiguous bits", .{ c.name, tok }), std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
    }
}

const PCase = struct { name: []const u8, h: usize, hk: usize, p0: usize, n: usize };

/// One stream's prompt chunk after p0 cached positions: chunks over many pages, starting mid-page or at a page's start.
const prompt_cases = [_]PCase{
    .{ .name = "12B global prompt (16/1 heads of 512), 300 rows at 1000", .h = 16, .hk = 1, .p0 = 1000, .n = 300 },
    .{ .name = "31B global prompt (32/4 heads of 512), 200 rows at 0", .h = 32, .hk = 4, .p0 = 0, .n = 200 },
    .{ .name = "31B global prompt (32/4 heads of 512), 96 rows at 3013", .h = 32, .hk = 4, .p0 = 3013, .n = 96 },
};

/// A prompt page holds whole 32-key tiles (wa.prompt_tile): one tile, two, and a decode chunk.
const prompt_toks = [_]usize{ 32, 64, 256 };

fn prompt(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const wa.WindowAttention, rng: std.Random, c: PCase, p32: bool) !void {
    var b: rows.Bufs = .{ .gpu = gpu };
    defer b.deinit();
    const top = c.p0 + c.n;
    const q = try a.alloc(u16, c.n * c.h * d);
    const kc = try a.alloc(u16, top * c.hk * d);
    const vc = try a.alloc(u16, kc.len);
    rows.fill(rng, q, kc, vc, d);
    // the last row's own key holds +inf in a value: its heads of key head 0 run the float64 replay over the pages
    vc[(top - 1) * c.hk * d + 7] = 0x7f80;
    var w: wa.Window = .{ .h = c.h, .hk = c.hk, .d = d, .win = 0, .ring = 0, .scale = @floatCast(1 / @sqrt(@as(f64, d))), .p32 = p32 };
    const dq = try b.up(std.mem.sliceAsBytes(q));
    const n = c.n * c.h * d;
    const flat = try b.out(s, n * 2);
    try g.promptRows(s, dq, try b.up(std.mem.sliceAsBytes(kc)), try b.up(std.mem.sliceAsBytes(vc)), flat, c.p0, c.n, w, null);
    try s.synchronize();
    const want = try a.alloc(u16, n);
    try b.get(flat, want);
    var odd: usize = 0;
    for (want) |x| odd += @intFromBool(x & 0x7f80 == 0x7f80);
    try check.expect(odd > 0, "window pages 512 {s}: the planted +inf reaches the float64 replay", .{c.name});
    for (prompt_toks) |tok| {
        const pool = try paged(&b, a, rng, kc, vc, 1, top, c.hk, tok);
        w.pg = pool.pages; // the prompt kernel's table is the stream's own row
        const out = try b.out(s, n * 2);
        try g.promptRows(s, dq, pool.kc, pool.vc, out, c.p0, c.n, w, null);
        try s.synchronize();
        const got = try a.alloc(u16, n);
        try b.get(out, got);
        var what: [160]u8 = undefined;
        const p = if (p32) "P32" else "P as bf16";
        try check.sameBytes(try std.fmt.bufPrint(&what, "window pages 512 {s}, {s}: pages of {d} give the contiguous bits", .{ c.name, p, tok }), std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
    }
    w.pg.?.tokens = 48; // not whole prompt tiles: refused
    const refused = if (g.promptRows(s, dq, 0, 0, flat, c.p0, c.n, w, null)) |_| false else |e| e == error.WindowShape;
    try check.expect(refused, "window pages 512: prompt pages of 48 positions refused", .{});
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try wa.WindowAttention.load(gpu.d);
    defer g.unload();
    try g.loadPages(gpu.d);
    var prng = std.Random.DefaultPrng.init(0x9a6e);
    for (cases) |c| {
        _ = arena.reset(.retain_capacity);
        try case(gpu, arena.allocator(), s, &g, prng.random(), c);
    }
    for (prompt_cases) |c| for ([_]bool{ false, true }) |p32| {
        _ = arena.reset(.retain_capacity);
        try prompt(gpu, arena.allocator(), s, &g, prng.random(), c, p32);
    };
    check.pass("window pages 512: decode over shuffled pages of 16, 37 and 256 positions and prompt rows (P as bf16 and P32) over pages of 32, 64 and 256 (NaN in every unused slot) bit-equal to the contiguous cache, the float64 replays included", .{});
}
