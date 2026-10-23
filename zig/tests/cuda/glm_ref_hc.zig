//! GLM's hyper-connection dots (hcPre: 24 dots + squares a 1,024 block) vs float64; hc_partial8, _f32w: same bytes.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const ref = @import("glm_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;
const glue = glm.glue;

const d = 4096;
const s = 4;
const wide = s * d;
const mixes = 24;

const Outs = struct { out: u64, xs: u64, post: u64, comb: u64, part: u64 };

fn pre(r: *Rig, set: glue.Set, x: u64, w: [3]u64, norm: glue.Norm, rows: usize) !Outs {
    const o: Outs = .{ .out = try r.zeros(rows * d * 2), .xs = try r.zeros(rows * (d / 64) * 4), .post = try r.zeros(rows * s * 4), .comb = try r.zeros(rows * s * s * 4), .part = try r.zeros(glue.hcPartBytes(rows)) };
    try glue.hcPre(set, r.s, x, w[0], w[1], w[2], norm, o.out, o.xs, o.post, o.comb, o.part, 1e-6, 1e-6, 20, rows, d, s);
    return o;
}

/// The mixing weights: fn bf16 [24][s d] in [-0.02, 0.02), base fp32 [24], scale fp32 [3].
fn mixWeights(r: *Rig) ![3]u64 {
    const base = try r.a.alloc(f32, mixes);
    for (base) |*b| b.* = (r.rng.float(f32) * 2 - 1) * 0.5;
    const scale = [3]f32{ 0.7, 0.9, 1.1 };
    return .{ try r.dev(std.mem.sliceAsBytes(try r.bfs(mixes * wide, 0.02))), try r.dev(std.mem.sliceAsBytes(base)), try r.dev(std.mem.asBytes(&scale)) };
}

/// A float32 norm weight (FP4-Local's) holding bf16 values: the _f32w kernels give the bf16 weight's bytes.
pub fn gammas(r: *Rig, set: glue.Set) !void {
    errdefer cuda.completion.drain(r.gpu.d, r.s, "GLM norm parity fixture left work pending");
    const rows = 70;
    const w = try mixWeights(r);
    const streams = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * wide, 1.0)));
    const x = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * d, 1.0)));
    const nw = try r.bfs(d, 1.0);
    const g = try r.a.alloc(f32, d);
    for (g, nw) |*o, v| o.* = @floatCast(bf(v));
    const plain: glue.Norm = .{ .bf16 = try r.dev(std.mem.sliceAsBytes(nw)) };
    const kept: glue.Norm = .{ .f32 = try r.dev(std.mem.sliceAsBytes(g)) };
    var name: [128]u8 = undefined;
    for ([_]usize{ 1, 5, rows }) |m| {
        const a = try pre(r, set, streams, w, plain, m);
        const b = try pre(r, set, streams, w, kept, m);
        for ([4]u64{ a.out, a.xs, a.post, a.comb }, [4]u64{ b.out, b.xs, b.post, b.comb }, [4]usize{ d * 2, (d / 64) * 4, s * 4, s * s * 4 }, [4][]const u8{ "normed rows", "group sums", "post", "comb" }) |want, got, n, label|
            try ref.sameRows(r, try std.fmt.bufPrint(&name, "hc pre, {d} rows: {s} from the float32 weight equal the bf16 weight's", .{ m, label }), want, got, m * n);
        const y = [2]u64{ try r.zeros(m * d * 2), try r.zeros(m * d * 2) };
        const xs = [2]u64{ try r.zeros(m * (d / 64) * 4), try r.zeros(m * (d / 64) * 4) };
        for ([2]glue.Norm{ plain, kept }, y, xs) |n, yo, xo| try glue.rmsnormBy(set, r.s, x, d, n, yo, d, xo, 1e-6, m, d);
        try ref.sameRows(r, try std.fmt.bufPrint(&name, "final norm, {d} rows: from the float32 weight equal the bf16 weight's", .{m}), y[0], y[1], m * d * 2);
        try ref.sameRows(r, try std.fmt.bufPrint(&name, "final norm, {d} rows: group sums likewise", .{m}), xs[0], xs[1], m * (d / 64) * 4);
    }
    check.pass("float32 norm weights: hc_finish_f32w and rmsnorm_4096_f32w give the bf16 weight's bytes (1, 5, 70 rows)", .{});
}

/// 70 rows (hc_partial) vs float64 within (1,024 + 8) 2^-24 of sum |terms|; 1, 5, 63 rows (hc_partial8) same bytes.
pub fn mixing(r: *Rig, set: glue.Set) !void {
    const rows = 70;
    const x = try r.bfs(rows * wide, 1.0);
    const fnw = try r.bfs(mixes * wide, 0.02);
    const base = try r.a.alloc(f32, mixes);
    for (base) |*b| b.* = (r.rng.float(f32) * 2 - 1) * 0.5;
    const scale = [3]f32{ 0.7, 0.9, 1.1 };
    const nw = try r.bfs(d, 1.0);
    const w = [3]u64{ try r.dev(std.mem.sliceAsBytes(fnw)), try r.dev(std.mem.sliceAsBytes(base)), try r.dev(std.mem.asBytes(&scale)) };
    const norm: glue.Norm = .{ .bf16 = try r.dev(std.mem.sliceAsBytes(nw)) };
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const all = try pre(r, set, dx, w, norm, rows);

    const nb = glue.hc_blocks;
    const kb = wide / nb;
    const raw = std.mem.bytesAsSlice(f32, try r.back(all.part, glue.hcPartBytes(rows)));
    const col = try r.a.alloc(f64, mixes + 1);
    @memset(col, 0);
    const bound = @as(f64, kb + 8) * std.math.pow(f64, 2, -24);
    for (0..rows) |row| for (0..nb) |b| {
        const got = raw[(row * nb + b) * 32 ..][0..32];
        const xs = x[row * wide + b * kb ..][0..kb];
        for (0..mixes + 1) |m| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (xs, 0..) |xv, j| {
                const a = bf(xv);
                const t = if (m == mixes) a * a else a * bf(fnw[m * wide + b * kb + j]);
                sum += t;
                abs += @abs(t);
            }
            const q = @abs(got[m] - sum) / (abs * bound + 1e-30);
            if (!(q <= col[m])) col[m] = if (std.math.isNan(q)) std.math.inf(f64) else q;
        }
    };
    var name: [128]u8 = undefined;
    try ref.worstCol(r, try std.fmt.bufPrint(&name, "hc mixing {d} rows x {d} K blocks: 24 dots and the squares (column 24)", .{ rows, nb }), col);

    const full = [5]u64{ all.out, all.xs, all.post, all.comb, all.part };
    const per = [5]usize{ d * 2, (d / 64) * 4, s * 4, s * s * 4, glue.hcPartBytes(1) };
    const what = [5][]const u8{ "normed rows", "group sums", "post", "comb", "partials" };
    for ([_]usize{ 1, 5, 63 }) |m| {
        const o = try pre(r, set, dx, w, norm, m);
        for ([5]u64{ o.out, o.xs, o.post, o.comb, o.part }, full, per, what) |got, want, n, label|
            try ref.sameRows(r, try std.fmt.bufPrint(&name, "hc pre: {d} rows' {s} equal the {d}-row call's", .{ m, label, rows }), want, got, m * n);
    }
    check.pass("hc mixing: dots and squares within float64 tolerance, decode windows' partials and outputs equal the prompt rows'", .{});
}
