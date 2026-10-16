//! Flash Next's own hyper-connection and n-gram glue against float64 chains that round where the kernels round.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const Rig = rig.Rig;
const bf = rig.bf;
const rbf = rig.rbf;
const ulps = rig.ulps;
const sumTol = rig.sumTol;

const dims = 2560;
const streams = 4;
const wide = streams * dims;
const low = 320;
const n_down = low + streams;
const slices = 32;
const nc = dims / 256;
const eps: f32 = 1e-6;
const counts = [_]usize{ 1, 3, 17 };
const most = counts[counts.len - 1];

/// Tiled 4-bit rows on the device: words [N/64][K/32][64][4], scales and biases [K/32][N] (cuda_hc.zig's layout).
const Tiled = struct { w: []u32, s: []u16, b: []u16, n: usize, k: usize, dw: u64, ds: u64, db: u64 };

fn tiled(r: *Rig, n: usize, k: usize, scale: f32) !Tiled {
    const w = r.words(n * k / 8);
    const s = r.bfs(n * k / 32, scale);
    for (s) |*v| v.* = rig.bf16(@floatCast(@abs(bf(v.*))));
    const b = r.bfs(n * k / 32, scale * 4);
    const tw = try flash.hc.tileWords(r.a, std.mem.sliceAsBytes(w), n, k / 8);
    const ts = try flash.hc.transposeBf16(r.a, std.mem.sliceAsBytes(s), n, k / 32);
    const tb = try flash.hc.transposeBf16(r.a, std.mem.sliceAsBytes(b), n, k / 32);
    return .{ .w = w, .s = s, .b = b, .n = n, .k = k, .dw = try r.dev(tw), .ds = try r.dev(ts), .db = try r.dev(tb) };
}

/// Row `col` of a tiled matrix over inputs [k0, k0 + 32 g): sum over groups of (x . q) s + (sum x) b, and its |terms|.
fn rowDot(t: Tiled, col: usize, x: []const f64, k0: usize, groups: usize, xs: ?[]const f64) [2]f64 {
    var total: f64 = 0;
    var abs: f64 = 0;
    for (0..groups) |gi| {
        const g = k0 / 32 + gi;
        var p: f64 = 0;
        var sum: f64 = 0;
        var pa: f64 = 0;
        for (0..32) |k| {
            const q = rig.nibble(t.w, t.k / 8, col, g * 32 + k);
            p += x[gi * 32 + k] * q;
            pa += @abs(x[gi * 32 + k]) * q;
            sum += x[gi * 32 + k];
        }
        if (xs) |v| sum = v[gi];
        const s = bf(t.s[col * (t.k / 32) + g]);
        const b = bf(t.b[col * (t.k / 32) + g]);
        total += p * s + sum * b;
        abs += pa * @abs(s) + @abs(sum * b);
    }
    return .{ total, abs };
}

fn writeback(r: *Rig, g: *const flash.Glue, mode: usize) !void {
    const h = r.bfs(most * wide, 1);
    const inj = r.bfs(most * streams, 2);
    const y = r.f32s(most * 11 * dims, -1, 1);
    const wts = r.f32s(most * 11, 0, 0.2);
    const br = r.f32s(8 * most * dims, -0.5, 0.5);
    const dh = try r.dev(std.mem.sliceAsBytes(h));
    const dinj = try r.dev(std.mem.sliceAsBytes(inj));
    const dy = try r.dev(std.mem.sliceAsBytes(y));
    const dwts = try r.dev(std.mem.sliceAsBytes(wts));
    const dbr = try r.dev(std.mem.sliceAsBytes(br));
    var outs: [counts.len]u64 = undefined;
    var psss: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        outs[c] = try r.out(m * wide * 2);
        psss[c] = try r.out(m * nc * streams * 4);
        try g.hcWriteback(r.s, dh, outs[c], psss[c], dbr, dinj, dy, dwts, most * dims, m, dims, streams, mode, 10, 11, 8);
    }
    const name = switch (mode) {
        0 => "writeback sums",
        2 => "writeback moe",
        else => "writeback slices",
    };
    const hout = if (mode == 0) h else try r.back(u16, outs[counts.len - 1], most * wide);
    const pss = try r.back(f32, psss[counts.len - 1], most * nc * streams);
    for (0..most) |row| {
        for (0..dims) |d| {
            if (mode == 0) break;
            var acc: f64 = 0;
            if (mode == 2) {
                for (0..11) |k| acc += y[(row * 11 + k) * dims + d] * wts[row * 11 + k];
            } else for (0..8) |k| {
                acc += br[k * most * dims + row * dims + d];
            }
            const branch = rbf(acc);
            for (0..streams) |s| {
                const at = row * wide + s * dims + d;
                const want = rbf(bf(h[at]) + rbf(branch * bf(inj[row * streams + s])));
                try r.within(name, bf(hout[at]), want, ulps(want, 1));
            }
        }
        for (0..nc) |c| for (0..streams) |s| {
            var t: f64 = 0;
            for (0..256) |i| t += bf(hout[row * wide + s * dims + c * 256 + i]) * bf(hout[row * wide + s * dims + c * 256 + i]);
            try r.within(name, pss[(row * nc + c) * streams + s], t, sumTol(t, 256));
        };
    }
    for (counts[0 .. counts.len - 1], 0..) |m, c| {
        if (mode != 0) try r.sameRows(name, u16, outs[c], hout[0 .. m * wide]);
        try r.sameRows(name, f32, psss[c], pss[0 .. m * nc * streams]);
    }
}

/// The read-out chain as a pass runs it: down (normed rows, K slices), the slices' sum and activation, the up mix.
fn readout(r: *Rig, g: *const flash.Glue) !void {
    const h = r.bfs(most * wide, 1);
    const pss = try r.a.alloc(f32, most * nc * streams);
    for (0..most) |row| for (0..nc) |c| for (0..streams) |s| {
        var t: f32 = 0;
        for (0..256) |i| t += @floatCast(bf(h[row * wide + s * dims + c * 256 + i]) * bf(h[row * wide + s * dims + c * 256 + i]));
        pss[(row * nc + c) * streams + s] = t;
    };
    const scale = r.f32s(wide, 0.5, 1.5);
    const down = try tiled(r, n_down, wide, 0.004);
    const up = try tiled(r, wide, low, 0.05);
    const dh = try r.dev(std.mem.sliceAsBytes(h));
    const dpss = try r.dev(std.mem.sliceAsBytes(pss));
    const dscale = try r.dev(std.mem.sliceAsBytes(scale));
    var norms: [counts.len]u64 = undefined;
    var parts: [counts.len]u64 = undefined;
    var acts: [counts.len]u64 = undefined;
    var xss: [counts.len]u64 = undefined;
    var injs: [counts.len]u64 = undefined;
    var mixes: [counts.len]u64 = undefined;
    var xsms: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        norms[c] = try r.out(m * wide * 2);
        parts[c] = try r.out(slices * m * n_down * 4);
        acts[c] = try r.out(m * low * 2);
        xss[c] = try r.out(m * low / 32 * 4);
        injs[c] = try r.out(m * streams * 2);
        mixes[c] = try r.out(m * dims * 2);
        xsms[c] = try r.out(m * dims / 32 * 4);
        try g.hcDown(r.s, dh, dpss, dscale, norms[c], down.dw, down.ds, down.db, 0, parts[c], eps, m, n_down, wide, dims, nc, streams, slices);
        try g.hcReduceAct(r.s, parts[c], acts[c], xss[c], injs[c], slices, m, streams, low, n_down, 1);
        try g.hcUpmix(r.s, acts[c], xss[c], up.dw, up.ds, up.db, norms[c], mixes[c], xsms[c], m, wide, low, dims, streams);
    }
    const last = counts.len - 1;
    const normed = try r.back(u16, norms[last], most * wide);
    const part = try r.back(f32, parts[last], slices * most * n_down);
    const act = try r.back(u16, acts[last], most * low);
    const xs = try r.back(f32, xss[last], most * low / 32);
    const inj = try r.back(u16, injs[last], most * streams);
    const mixed = try r.back(u16, mixes[last], most * dims);
    const xsm = try r.back(f32, xsms[last], most * dims / 32);
    const xv = try r.a.alloc(f64, wide);
    for (0..most) |row| {
        for (0..streams) |s| {
            var t: f64 = 0;
            for (0..nc) |c| t += pss[(row * nc + c) * streams + s];
            const rinv = 1 / @sqrt(t / dims + eps);
            for (0..dims) |d| {
                const at = s * dims + d;
                const want = bf(h[row * wide + at]) * rinv * scale[at];
                try r.within("down normed", bf(normed[row * wide + at]), want, ulps(want, 0.5) + @abs(want) * 0x1p-20);
                xv[at] = bf(normed[row * wide + at]);
            }
        }
        // each K slice's sums from the stored normed values
        const per = wide / slices;
        for (0..slices) |z| for (0..n_down) |col| {
            const d = rowDot(down, col, xv[z * per ..][0..per], z * per, per / 32, null);
            try r.within("down slices", part[(z * most + row) * n_down + col], d[0], d[1] * 0x1p-17 + 1e-9);
        };
        var asum: [low / 32]f64 = @splat(0);
        var aabs: [low / 32]f64 = @splat(0);
        for (0..n_down) |col| {
            var v: f64 = 0;
            for (0..slices) |z| v += part[(z * most + row) * n_down + col];
            const vv = rbf(rbf(v) / streams);
            if (col < low) {
                const want = rbf(vv * rig.sigmoid(vv));
                try r.within("reduce act", bf(act[row * low + col]), want, ulps(want, 1));
                asum[col / 32] += bf(act[row * low + col]);
                aabs[col / 32] += @abs(bf(act[row * low + col]));
            } else {
                const want = rbf(2 * rbf(rig.sigmoid(vv)));
                try r.within("reduce inject", bf(inj[row * streams + col - low]), want, ulps(want, 1));
            }
        }
        for (0..low / 32) |gi| try r.within("reduce act sums", xs[row * (low / 32) + gi], asum[gi], sumTol(aabs[gi], 32));
        // the up mix from the stored activations, their sums and the normed streams
        const av = try r.a.alloc(f64, low);
        for (av, 0..) |*o, i| o.* = bf(act[row * low + i]);
        const xsv = try r.a.alloc(f64, low / 32);
        for (xsv, 0..) |*o, i| o.* = xs[row * (low / 32) + i];
        for (0..dims / 32) |j| {
            var gsum: f64 = 0;
            var gabs: f64 = 0;
            for (0..32) |dd| {
                const d = j * 32 + dd;
                var total: f64 = 0;
                var tabs: f64 = 0;
                for (0..streams) |s| {
                    const u = rowDot(up, s * dims + d, av, 0, low / 32, xsv)[0];
                    const term = rbf(rbf(rig.sigmoid(rbf(u))) * bf(normed[row * wide + s * dims + d]));
                    total += term;
                    tabs += @abs(term);
                }
                const want = rbf(total / streams);
                try r.within("up mix", bf(mixed[row * dims + d]), want, ulps(tabs / streams, 1) + ulps(want, 0.5));
                gsum += bf(mixed[row * dims + d]);
                gabs += @abs(bf(mixed[row * dims + d]));
            }
            try r.within("up mix sums", xsm[row * (dims / 32) + j], gsum, sumTol(gabs, 32));
        }
    }
    for (counts[0..last], 0..) |m, c| {
        try r.sameRows("down normed", u16, norms[c], normed[0 .. m * wide]);
        for (0..slices) |z| try r.sameRows("down slices", f32, parts[c] + z * m * n_down * 4, part[z * most * n_down ..][0 .. m * n_down]);
        try r.sameRows("reduce act", u16, acts[c], act[0 .. m * low]);
        try r.sameRows("reduce act sums", f32, xss[c], xs[0 .. m * low / 32]);
        try r.sameRows("reduce inject", u16, injs[c], inj[0 .. m * streams]);
        try r.sameRows("up mix", u16, mixes[c], mixed[0 .. m * dims]);
        try r.sameRows("up mix sums", f32, xsms[c], xsm[0 .. m * dims / 32]);
    }
}

/// The n-gram gate then its conv over [tail; rows], and the MTP head's stream adds.
fn ngram(r: *Rig, g: *const flash.Glue) !void {
    const keys = r.bfs(most * wide, 1);
    const vals = r.bfs(most * dims, 1);
    const h = r.bfs(most * wide, 1);
    const nk = r.f32s(wide, 0.5, 1.5);
    const nq = r.f32s(wide, 0.5, 1.5);
    const ncw = r.f32s(wide, 0.5, 1.5);
    const tail = r.bfs(9 * wide, 1);
    const cw = r.bfs(wide * 4, 0.5);
    const dk = try r.dev(std.mem.sliceAsBytes(keys));
    const dv = try r.dev(std.mem.sliceAsBytes(vals));
    const dh = try r.dev(std.mem.sliceAsBytes(h));
    const dnk = try r.dev(std.mem.sliceAsBytes(nk));
    const dnq = try r.dev(std.mem.sliceAsBytes(nq));
    const dnc = try r.dev(std.mem.sliceAsBytes(ncw));
    const dtail = try r.dev(std.mem.sliceAsBytes(tail));
    const dcw = try r.dev(std.mem.sliceAsBytes(cw));
    var gated: [counts.len]u64 = undefined;
    var gpss: [counts.len]u64 = undefined;
    var houts: [counts.len]u64 = undefined;
    var nrows: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        gated[c] = try r.out(m * wide * 2);
        gpss[c] = try r.out(m * streams * 4);
        houts[c] = try r.out(m * wide * 2);
        nrows[c] = try r.out(m * wide * 2);
        try g.pleGate(r.s, dk, dv, dh, dnk, dnq, gated[c], gpss[c], eps, m, dims, streams);
        try g.pleConv(r.s, gated[c], gpss[c], dnc, dtail, dcw, dh, houts[c], nrows[c], eps, m, dims, streams, 4, 3);
    }
    const last = counts.len - 1;
    const gt = try r.back(u16, gated[last], most * wide);
    const ps = try r.back(f32, gpss[last], most * streams);
    const hout = try r.back(u16, houts[last], most * wide);
    const nrow = try r.back(u16, nrows[last], most * wide);
    for (0..most) |row| for (0..streams) |s| {
        const at = row * wide + s * dims;
        var kss: f64 = 0;
        var qss: f64 = 0;
        for (0..dims) |d| {
            kss += bf(keys[at + d]) * bf(keys[at + d]);
            qss += bf(h[at + d]) * bf(h[at + d]);
        }
        const kr = 1 / @sqrt(kss / dims + eps);
        const qr = 1 / @sqrt(qss / dims + eps);
        var dot: f64 = 0;
        for (0..dims) |d| dot += rbf(rbf(bf(keys[at + d]) * kr * nk[s * dims + d]) * rbf(bf(h[at + d]) * qr * nq[s * dims + d]));
        var gate = rbf(dot / @sqrt(@as(f64, dims)));
        const mag = @sqrt(@max(@abs(gate), 1e-6));
        gate = rbf(if (gate > 0) mag else if (gate < 0) -mag else 0);
        const sg = rbf(rig.sigmoid(gate));
        var gss: f64 = 0;
        for (0..dims) |d| {
            const want = rbf(sg * bf(vals[row * dims + d]));
            try r.within("ngram gate", bf(gt[at + d]), want, ulps(want, 1));
            gss += bf(gt[at + d]) * bf(gt[at + d]);
        }
        try r.within("ngram gate sums", ps[row * streams + s], gss, sumTol(gss, dims));
    };
    // the conv of row r reads rows r - 9 + 3t, taken from the tail before the chunk
    for (0..most) |row| for (0..wide) |e| {
        const s = e / dims;
        var acc: f64 = 0;
        for (0..4) |t| {
            const at = row + 3 * t;
            const xv = if (at < 9) bf(tail[at * wide + e]) else blk: {
                const rr = at - 9;
                const rinv = 1 / @sqrt(@as(f64, ps[rr * streams + s]) / dims + eps);
                break :blk rbf(bf(gt[rr * wide + e]) * rinv * ncw[e]);
            };
            acc += bf(cw[e * 4 + t]) * xv;
        }
        const conv = rbf(acc);
        const ple = rbf(bf(gt[row * wide + e]) + rbf(conv * rig.sigmoid(conv)));
        const want = rbf(bf(h[row * wide + e]) + ple);
        try r.within("ngram conv", bf(hout[row * wide + e]), want, ulps(ple, 1) + ulps(want, 0.5));
        const rinv = 1 / @sqrt(@as(f64, ps[row * streams + s]) / dims + eps);
        const own = rbf(bf(gt[row * wide + e]) * rinv * ncw[e]);
        try r.within("ngram conv rows", bf(nrow[row * wide + e]), own, ulps(own, 1));
    };
    for (counts[0..last], 0..) |m, c| {
        try r.sameRows("ngram gate", u16, gated[c], gt[0 .. m * wide]);
        try r.sameRows("ngram gate sums", f32, gpss[c], ps[0 .. m * streams]);
        try r.sameRows("ngram conv rows", u16, nrows[c], nrow[0 .. m * wide]);
    }
    // the MTP head's input streams: bf16(e + h), each sum exact before its one rounding
    const e = r.bfs(most * dims, 1);
    const out = try r.out(most * wide * 2);
    try g.addStreams(r.s, try r.dev(std.mem.sliceAsBytes(e)), dh, out, most, dims, streams);
    const got = try r.back(u16, out, most * wide);
    for (0..most) |row| for (0..wide) |i| {
        const want = rbf(bf(e[row * dims + i % dims]) + bf(h[row * wide + i]));
        try check.expect(bf(got[row * wide + i]) == want, "add streams: row {d} value {d} is the rounded exact sum", .{ row, i });
    };
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(53);
    var r = try Rig.init(gpu, arena.allocator(), prng.random());
    defer r.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    for ([_]usize{ 0, 2, 4 }) |mode| try writeback(&r, &g, mode);
    try readout(&r, &g);
    try ngram(&r, &g);
    r.report("flashnext-hc: write-backs, the read-out chain, the n-gram gate and conv, stream adds");
}
