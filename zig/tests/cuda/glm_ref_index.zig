//! GLM's DSA indexer: pool scores vs float64 (worst column), radix top-512 vs the host rule on device scores (exact).
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const ref = @import("glm_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;
const sp = glm.sparse;

const heads = 32;
const d = 128;

/// sparse._order_key: a score's bits as a u32 whose unsigned order is the scores' (-0 counted as +0).
fn orderKey(s: f32) u32 {
    const b: i32 = @bitCast(s + 0.0);
    return @bitCast(b ^ ((b >> 31) | std.math.minInt(i32)));
}

/// sparse.selectTokens for 96 rows at 6,000 (1,500 to 1,524 visible pools in a 2,048-pool bucket).
pub fn selection(r: *Rig, set: glm.glue.Set, ops: cuda.Module) !void {
    const pos = 6000;
    const rows = 96;
    const cap = 4096;
    const np = sp.poolBucket(pos, rows, cap);
    const cols = @min(np, (sp.scoreColumns(pos, rows, np) + 63) / 64 * 64);
    const qi = try r.bfs(rows * heads * d, 1.0);
    const wts = try r.bfs(rows * heads, 1.0);
    const pk = try r.bfs(np * d, 1.0);
    const p0: i32 = pos;
    const s: sp.Scratch = .{ .scores = try r.zeros(rows * np * 4), .ls = try r.zeros(rows * sp.select_list * 4), .li = try r.zeros(rows * sp.select_list * 4), .pools = try r.zeros(rows * sp.topk_pools * 8), .at = try r.zeros(4) };
    const tokens = try r.zeros(rows * sp.width * 4);
    const counts = try r.zeros(rows * 4);
    try sp.selectTokens(set, ops, r.s, try r.dev(std.mem.sliceAsBytes(qi)), try r.dev(std.mem.sliceAsBytes(wts)), heads, try r.dev(std.mem.sliceAsBytes(pk)), .{}, try r.dev(std.mem.asBytes(&p0)), pos, rows, heads, d, cap, s, tokens, counts);
    const raw = try r.back(s.scores, rows * np * 4);
    const got = std.mem.bytesAsSlice(f32, raw);

    // float64: sum_h w_h relu(scale q_h . k_p); bound: (128 + 8) 2^-24 scale sum |q k| a dot, 40 2^-24 sum |w_h relu|.
    const scale = 1 / @sqrt(@as(f64, d));
    const wscale: f64 = sp.headScale(heads);
    const kd = try r.a.alloc(f64, np * d);
    for (kd, pk) |*v, b| v.* = bf(b);
    const col = try r.a.alloc(f64, cols);
    @memset(col, 0);
    for (0..rows) |row| {
        const npool = (pos + row + 1) / 4;
        for (0..cols) |p| {
            const g = got[row * np + p];
            if (p >= npool) {
                try check.expect(g == -std.math.inf(f32), "score row {d} pool {d}: past the visible pools, -inf (got {d})", .{ row, p, g });
                continue;
            }
            var sum: f64 = 0;
            var bound: f64 = 0;
            var spread: f64 = 0;
            for (0..heads) |h| {
                const w = bf(wts[row * heads + h]) * wscale;
                var dot: f64 = 0;
                var abs: f64 = 0;
                for (0..d) |i| {
                    const t = bf(qi[(row * heads + h) * d + i]) * kd[p * d + i];
                    dot += t;
                    abs += @abs(t);
                }
                sum += w * @max(dot * scale, 0);
                spread += @abs(w) * @max(dot * scale, 0);
                bound += @abs(w) * abs * scale * 136 * std.math.pow(f64, 2, -24);
            }
            const q = @abs(g - sum) / (bound + spread * 40 * std.math.pow(f64, 2, -24) + 1e-30);
            if (!(q <= col[p])) col[p] = if (std.math.isNan(q)) std.math.inf(f64) else q;
        }
    }
    var name: [128]u8 = undefined;
    try ref.worstCol(r, try std.fmt.bufPrint(&name, "index scores {d} rows at {d}, {d} heads x {d}, {d} pools", .{ rows, pos, heads, d, cols }), col);

    // Top 512 of each row's first max(visible, 512) pools (at most np) by order key, ties to the lower pool, ascending.
    const picks = std.mem.bytesAsSlice(i64, try r.back(s.pools, rows * sp.topk_pools * 8));
    const order = try r.a.alloc(u32, np);
    for (0..rows) |row| {
        const vis = (pos + row + 1) / 4;
        const limit = @min(np, @max(vis, sp.topk_pools));
        const keys = try r.a.alloc(u32, limit);
        for (keys, 0..) |*k, i| k.* = orderKey(got[row * np + i]);
        const idx = order[0..limit];
        for (idx, 0..) |*v, i| v.* = @intCast(i);
        std.mem.sort(u32, idx, keys, struct {
            fn lt(k: []const u32, a: u32, b: u32) bool {
                return k[a] > k[b] or (k[a] == k[b] and a < b);
            }
        }.lt);
        const n = @min(limit, sp.topk_pools);
        std.mem.sort(u32, idx[0..n], {}, std.sort.asc(u32));
        for (idx[0..n], picks[row * sp.topk_pools ..][0..n], 0..) |want, have, i|
            try check.expect(have == want, "select row {d} pick {d}: pool {d}, the host's rule gives {d}", .{ row, i, have, want });
    }
    check.pass("DSA indexer {d} rows at {d}: scores within float64 tolerance, top-{d} pools equal the host's rule exactly", .{ rows, pos, sp.topk_pools });
}

/// tf_glm_select_rows_wide against tf_glm_select_rows (bytes) and the host's rule, and both launches' times.
pub fn selectionWide(r: *Rig, set: glm.glue.Set) !void {
    var prng = std.Random.DefaultPrng.init(0x5e1ec7); // its own draws: r.rng's order stays the later checks'
    const rng = prng.random();
    var t0 = try cuda.Event.init(r.gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(r.gpu.d, true);
    defer t1.deinit();
    const reps = 20;
    const cases = [_][2]usize{ .{ 1500, 1 }, .{ 6000, 4 }, .{ 12000, 1 }, .{ 24000, 1 }, .{ 32768, 1 }, .{ 136000, 1 }, .{ 136000, 4 }, .{ 136000, 32 }, .{ 136000, 48 }, .{ 136000, 64 }, .{ 136000, 128 }, .{ 500000, 1 } };
    for (cases) |c| {
        const pos = c[0];
        const rows = c[1];
        const np = sp.poolBucket(pos, rows, 1 << 20);
        const host = try r.a.alloc(f32, rows * np);
        for (0..rows) |row| {
            const vis = (pos + row + 1) / 4;
            for (host[row * np ..][0..np], 0..) |*v, p| {
                const level: f32 = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 4096))) - 2048);
                v.* = if (p >= vis) -std.math.inf(f32) else if (level == 0 and rng.boolean()) -0.0 else level * 0.25;
            }
        }
        const scores = try r.dev(std.mem.sliceAsBytes(host));
        const p0: i32 = @intCast(pos);
        const at = try r.dev(std.mem.asBytes(&p0));
        const outs = [2]u64{ try r.zeros(rows * sp.topk_pools * 8), try r.zeros(rows * sp.topk_pools * 8) };
        var ms: [2]f64 = undefined;
        for ([2]cuda.Function{ set.glue.select_rows, set.glue.select_rows_wide }, [2]u32{ 128, 1024 }, outs, 0..) |f, threads, out, i| {
            var a: cuda.Args = .{};
            inline for (.{ scores, out, @as(i64, @intCast(np)), at, @as(c_int, sp.topk_pools) }) |v| a.add(v);
            const cfg: cuda.launch.Config = .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = threads } };
            try cuda.launch.launch(f, cfg, r.s, &a);
            try t0.record(r.s);
            for (0..reps) |_| try cuda.launch.launch(f, cfg, r.s, &a);
            try t1.record(r.s);
            try t1.synchronize();
            ms[i] = @as(f64, try cuda.Event.elapsedMs(t0, t1)) / reps;
        }
        const narrow = try r.back(outs[0], rows * sp.topk_pools * 8);
        const wide = try r.back(outs[1], rows * sp.topk_pools * 8);
        if (std.mem.indexOfDiff(u8, narrow, wide)) |b|
            try check.expect(false, "select {d} rows at {d}: byte {d} of the wide kernel's differs from tf_glm_select_rows'", .{ rows, pos, b });
        // the host's rule on the first and the last row
        const picks = std.mem.bytesAsSlice(i64, wide);
        for ([2]usize{ 0, rows - 1 }) |row| {
            const vis = (pos + row + 1) / 4;
            const limit = @min(np, @max(vis, sp.topk_pools));
            const keys = try r.a.alloc(u32, limit);
            for (keys, 0..) |*k, p| k.* = orderKey(host[row * np + p]);
            const idx = try r.a.alloc(u32, limit);
            for (idx, 0..) |*v, p| v.* = @intCast(p);
            std.mem.sort(u32, idx, keys, struct {
                fn lt(k: []const u32, x: u32, y: u32) bool {
                    return k[x] > k[y] or (k[x] == k[y] and x < y);
                }
            }.lt);
            const n = @min(limit, sp.topk_pools);
            std.mem.sort(u32, idx[0..n], {}, std.sort.asc(u32));
            for (idx[0..n], picks[row * sp.topk_pools ..][0..n], 0..) |want, have, i|
                try check.expect(have == want, "select {d} rows at {d}, row {d} pick {d}: pool {d}, the host's rule gives {d}", .{ rows, pos, row, i, have, want });
        }
        check.pass("DSA select {d} rows at {d} ({d}-pool bucket): wide equals tf_glm_select_rows byte for byte and the host's rule; {d:.4} ms vs {d:.4} ms a launch", .{ rows, pos, np, ms[1], ms[0] });
    }
}
