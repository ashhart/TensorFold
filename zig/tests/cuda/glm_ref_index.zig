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
