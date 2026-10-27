//! glm-index-check (GPU): selectTokens with glm_index against tf_glm_scores and tf_glm_select_rows, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");
const sp = glm.sparse;

const d = 128;
const page_bytes = 65536; // a page's stride: 128 pools of keys in its first half, as a page that also holds other rows

const Case = struct { cap: usize, pos: usize, rows: usize, heads: usize = 32, paged: bool = false, bits: bool = false };
const cases = [_]Case{
    .{ .cap = 4096, .pos = 300, .rows = 512 }, // few visible pools: the rows select over their first 512
    .{ .cap = 1024, .pos = 4 * 1024 - 520, .rows = 512 },
    .{ .cap = 4096, .pos = 4 * 4096 - 520, .rows = 512, .bits = true }, // NaN and signed zeros planted in the tied scores
    .{ .cap = 16384, .pos = 4 * 16384 - 520, .rows = 512 },
    .{ .cap = 16384, .pos = 4 * 16384 - 1130, .rows = 1100 }, // three score blocks, the later two from s.at
    .{ .cap = 16384, .pos = 4 * 16384 - 600, .rows = 512, .paged = true },
    .{ .cap = 4096, .pos = 4 * 4096 - 520, .rows = 512, .heads = 20 }, // fewer heads than a warp's 32 rows of q
    .{ .cap = 65536, .pos = 4 * 65536 - 520, .rows = 512 },
    .{ .cap = 147456, .pos = 4 * 147456 - 520, .rows = 512 }, // a cap off the power of two
    .{ .cap = 262144, .pos = 4 * 262144 - 520, .rows = 512 },
};

fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

const Bufs = std.ArrayList(cuda.DeviceBuffer);

/// n zeroed device bytes, freed with the case.
fn zeros(gpu: check.Gpu, bufs: *Bufs, n: usize) !u64 {
    var b = try cuda.DeviceBuffer.alloc(gpu.d, n);
    errdefer b.free();
    try b.fill8(0, null);
    try bufs.append(gpu.gpa, b);
    return b.ptr;
}

fn fromHost(gpu: check.Gpu, bufs: *Bufs, bytes: []const u8) !u64 {
    var b = try cuda.DeviceBuffer.fromHost(gpu.d, bytes);
    errdefer b.free();
    try bufs.append(gpu.gpa, b);
    return b.ptr;
}

fn scratch(gpu: check.Gpu, bufs: *Bufs, rows: usize, np: usize) !sp.Scratch {
    const blk = @min(rows, sp.select_rows);
    return .{ .scores = try zeros(gpu, bufs, blk * np * 4), .ls = try zeros(gpu, bufs, rows * sp.select_list * 4), .li = try zeros(gpu, bufs, rows * sp.select_list * 4), .pools = try zeros(gpu, bufs, rows * sp.topk_pools * 8), .at = try zeros(gpu, bufs, 4) };
}

fn ms(a: std.Io.Timestamp, b: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(a.durationTo(b).nanoseconds)) / 1e6;
}

/// How many rows took each path of glm_index_pools (whole row, candidates, whole row after the candidates missed).
var paths = [3]usize{ 0, 0, 0 };

pub fn run(gpu: check.Gpu) !void {
    const gpa = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var glue_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue_mod.unload();
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
    defer ops.unload();
    const ks = try glm.glue.Kernels.load(glue_mod);
    var idx = try glm.index.Kernels.load(gpu.d, cuda.kernels.glm_index, .both);
    defer idx.unload();
    var idx_scores = idx; // TF_GLM_INDEX_CUDA=scores: the kernels' scores, tf_glm_select_rows' pools
    idx_scores.select = false;
    const old: glm.glue.Set = .{ .glue = &ks };
    const new: glm.glue.Set = .{ .glue = &ks, .index = &idx };
    const half: glm.glue.Set = .{ .glue = &ks, .index = &idx_scores };
    var rng = std.Random.DefaultPrng.init(0x105);
    const rand = rng.random();
    var fails: usize = 0;
    for (cases) |c| {
        const np = sp.poolBucket(c.pos, c.rows, c.cap);
        const qi = try gpa.alloc(u16, c.rows * c.heads * d);
        defer gpa.free(qi);
        for (qi) |*x| x.* = bf16(rand.floatNorm(f32));
        const wts = try gpa.alloc(u16, c.rows * c.heads);
        defer gpa.free(wts);
        for (wts) |*x| x.* = bf16(rand.floatNorm(f32));
        const pk = try gpa.alloc(u16, c.cap * d);
        defer gpa.free(pk);
        for (pk) |*x| x.* = bf16(rand.floatNorm(f32) * 2);
        var bufs: Bufs = .empty;
        defer {
            for (bufs.items) |*b| b.free();
            bufs.deinit(gpa);
        }
        const dq = try fromHost(gpu, &bufs, std.mem.sliceAsBytes(qi));
        const dw = try fromHost(gpu, &bufs, std.mem.sliceAsBytes(wts));
        var table: glm.latent.Table = .{};
        var dk: u64 = undefined;
        if (c.paged) {
            // the keys on shuffled pages, the table long enough for the token lists' latent rows too
            const page_rows = page_bytes / (512 * 2);
            const entries = @max((c.cap + 127) / 128, (4 * np + page_rows - 1) / page_rows + 8);
            const ids = try gpa.alloc(i32, entries);
            defer gpa.free(ids);
            for (ids, 0..) |*v, i| v.* = @intCast(i);
            rand.shuffle(i32, ids);
            const arena = try gpa.alloc(u8, entries * page_bytes);
            defer gpa.free(arena);
            @memset(arena, 0);
            for (0..c.cap) |p| @memcpy(arena[@as(usize, @intCast(ids[p / 128])) * page_bytes + (p % 128) * d * 2 ..][0 .. d * 2], std.mem.sliceAsBytes(pk[p * d ..][0..d]));
            dk = try fromHost(gpu, &bufs, arena);
            table = .{ .pt = try fromHost(gpu, &bufs, std.mem.sliceAsBytes(ids)), .ps = page_bytes, .page_rows = page_rows };
        } else dk = try fromHost(gpu, &bufs, std.mem.sliceAsBytes(pk));
        const p0: i32 = @intCast(c.pos);
        const dpos = try fromHost(gpu, &bufs, std.mem.asBytes(&p0));
        const sa = try scratch(gpu, &bufs, c.rows, np);
        const sb = try scratch(gpu, &bufs, c.rows, np);
        const sc = try scratch(gpu, &bufs, c.rows, np);
        const tok_bytes = c.rows * sp.width * 4;
        const ta = try zeros(gpu, &bufs, tok_bytes);
        const tb = try zeros(gpu, &bufs, tok_bytes);
        const tc = try zeros(gpu, &bufs, tok_bytes);
        const ca = try zeros(gpu, &bufs, c.rows * 4);
        const cb = try zeros(gpu, &bufs, c.rows * 4);
        const cc = try zeros(gpu, &bufs, c.rows * 4);
        var ms_old: f64 = 0;
        var ms_new: f64 = 0;
        for (0..3) |rep| {
            const t0 = std.Io.Timestamp.now(gpu.io, .awake);
            try sp.selectTokens(old, ops, s, dq, dw, c.heads, dk, table, dpos, c.pos, c.rows, c.heads, d, c.cap, sa, ta, ca);
            try s.synchronize();
            const t1 = std.Io.Timestamp.now(gpu.io, .awake);
            try sp.selectTokens(new, ops, s, dq, dw, c.heads, dk, table, dpos, c.pos, c.rows, c.heads, d, c.cap, sb, tb, cb);
            try s.synchronize();
            const t2 = std.Io.Timestamp.now(gpu.io, .awake);
            if (rep > 0) {
                ms_old += ms(t0, t1) / 2;
                ms_new += ms(t1, t2) / 2;
            }
        }
        try sp.selectTokens(half, ops, s, dq, dw, c.heads, dk, table, dpos, c.pos, c.rows, c.heads, d, c.cap, sc, tc, cc);
        try s.synchronize();
        const blk = @min(c.rows, sp.select_rows);
        fails += try same(gpu, "scores", sa.scores, sb.scores, blk * np * 4, np, c);
        fails += try same(gpu, "pools", sa.pools, sb.pools, c.rows * sp.topk_pools * 8, np, c);
        fails += try same(gpu, "tokens", ta, tb, tok_bytes, np, c);
        fails += try same(gpu, "scores-only scores", sa.scores, sc.scores, blk * np * 4, np, c);
        fails += try same(gpu, "scores-only pools", sa.pools, sc.pools, c.rows * sp.topk_pools * 8, np, c);
        fails += try same(gpu, "scores-only tokens", ta, tc, tok_bytes, np, c);
        std.debug.print("np {d} rows {d} heads {d}{s}: selectTokens {d:.2} ms, with glm_index {d:.2} ms\n", .{ np, c.rows, c.heads, if (c.paged) " paged" else "", ms_old, ms_new });
        fails += try blocks(gpu, &bufs, old, &idx, s, dq, dw, dk, table, c, np);
        fails += try ties(gpu, &bufs, rand, old, &idx, s, sa, sb, dpos, np, c);
    }
    std.debug.print("glm_index_pools paths: whole row {d}, candidates {d}, whole row after the candidates missed {d}\n", .{ paths[0], paths[1], paths[2] });
    try check.expect(paths[0] > 0 and paths[1] > 0, "glm-index-check: the whole-row and candidate paths must both occur", .{});
    try check.expect(fails == 0, "glm-index-check: {d} buffers differ", .{fails});
    check.pass("glm-index-check: scores, pools and token lists byte-equal at {d} cases, tied scores through both selects", .{cases.len});
}

/// Every 512-row block's scores from tf_glm_scores and from glm_index_scores, as selectTokens launches them.
fn blocks(gpu: check.Gpu, bufs: *Bufs, set: glm.glue.Set, idx: *const glm.index.Kernels, s: cuda.Stream, dq: u64, dw: u64, dk: u64, t: glm.latent.Table, c: Case, np: usize) !usize {
    const blk = sp.select_rows;
    const scale: f32 = @floatCast(std.math.pow(f64, d, -0.5));
    const loop: usize = if (c.rows >= sp.score_loop_from) sp.score_loop else 1;
    const out_a = try zeros(gpu, bufs, @min(c.rows, blk) * np * 4);
    const out_b = try zeros(gpu, bufs, @min(c.rows, blk) * np * 4);
    var fails: usize = 0;
    var a: usize = 0;
    while (a < c.rows) : (a += blk) {
        const n = @min(blk, c.rows - a);
        const at_v: i32 = @intCast(c.pos + a);
        const at = try fromHost(gpu, bufs, std.mem.asBytes(&at_v));
        const nb = (sp.scoreColumns(c.pos + a, n, np) + 63) / 64;
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = out_a, .len = n * np * 4, .borrowed = true }).fill8(0, null);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = out_b, .len = n * np * 4, .borrowed = true }).fill8(0, null);
        var sa: cuda.Args = .{};
        inline for (.{ dq + a * c.heads * d * 2, dw + a * c.heads * 2, @as(i64, @intCast(c.heads)), dk, out_a, at, @as(i64, @intCast(np)), @as(i64, @intCast(nb)), scale, sp.headScale(c.heads), @as(i64, @intCast(c.heads)), @as(c_int, @intCast(loop)), t.pt, @as(i64, @intCast(t.ps)) }) |v| sa.add(v);
        try cuda.launch.launch(set.glue.scores, .{ .grid = .{ .x = @intCast(n), .y = @intCast((nb + loop - 1) / loop) }, .block = .{ .x = 128 } }, s, &sa);
        try idx.scores(s, dq + a * c.heads * d * 2, dw + a * c.heads * 2, c.heads, dk, out_b, at, n, np, nb, scale, sp.headScale(c.heads), c.heads, t.pt, t.ps);
        try s.synchronize();
        fails += try same(gpu, "block scores", out_a, out_b, n * np * 4, np, c);
    }
    return fails;
}

/// Two device buffers of n bytes compared; 1 if any 4-byte word differs.
fn same(gpu: check.Gpu, what: []const u8, a: u64, b: u64, n: usize, np: usize, c: Case) !usize {
    const ha = try gpu.gpa.alloc(u8, n);
    defer gpu.gpa.free(ha);
    const hb = try gpu.gpa.alloc(u8, n);
    defer gpu.gpa.free(hb);
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = a, .len = n, .borrowed = true }).download(0, ha);
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = b, .len = n, .borrowed = true }).download(0, hb);
    var diff: usize = 0;
    for (std.mem.bytesAsSlice(u32, ha), std.mem.bytesAsSlice(u32, hb)) |x, y| diff += @intFromBool(x != y);
    if (diff != 0) std.debug.print("FAIL np {d} rows {d} {s}: {d} words differ\n", .{ np, c.rows, what, diff });
    return @intFromBool(diff != 0);
}

/// Tied and random scores (c.bits: with NaN, +-0, +-inf) through both selects, counting glm_index_pools' paths.
fn ties(gpu: check.Gpu, bufs: *Bufs, rand: std.Random, set: glm.glue.Set, idx: *const glm.index.Kernels, s: cuda.Stream, sa: sp.Scratch, sb: sp.Scratch, at: u64, np: usize, c: Case) !usize {
    const rows = @min(c.rows, sp.select_rows);
    const host = try gpu.gpa.alloc(f32, rows * np);
    defer gpu.gpa.free(host);
    const path = try zeros(gpu, bufs, rows * 4);
    const special = [_]f32{ std.math.nan(f32), -std.math.nan(f32), 0.0, -0.0, std.math.inf(f32), -std.math.inf(f32) };
    var fails: usize = 0;
    for (0..2) |round| {
        for (0..rows) |r| {
            const vis = (c.pos + r + 1) / 4;
            for (host[r * np ..][0..np], 0..) |*x, i| {
                x.* = if (i >= vis) -std.math.inf(f32) else if (round == 0) @as(f32, @floatFromInt(rand.uintLessThan(u32, 16))) * 0.25 else rand.floatNorm(f32);
                if (c.bits and i < vis and rand.uintLessThan(u32, 8) == 0) x.* = special[rand.uintLessThan(usize, special.len)];
            }
        }
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = sa.scores, .len = rows * np * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(host));
        var sel: cuda.Args = .{};
        inline for (.{ sa.scores, sa.pools, @as(i64, @intCast(np)), at, @as(c_int, sp.topk_pools) }) |v| sel.add(v);
        try cuda.launch.launch(set.glue.select_rows, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, s, &sel);
        try idx.pools(s, sa.scores, sb.pools, at, rows, np, path);
        try s.synchronize();
        var taken: [512]i32 = undefined;
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = path, .len = rows * 4, .borrowed = true }).download(0, std.mem.sliceAsBytes(taken[0..rows]));
        for (taken[0..rows]) |p| paths[@intCast(p)] += 1;
        fails += try same(gpu, if (round == 0) "tied pools" else "random pools", sa.pools, sb.pools, rows * sp.topk_pools * 8, np, c);
    }
    return fails;
}
