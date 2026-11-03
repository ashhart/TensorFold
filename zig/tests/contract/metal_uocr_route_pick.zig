//! The route group's ud_route kernel: each query's remote block picks and their entmax bias.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const uocr = @import("metal_uocr.zig");
const base = @import("metal_uocr_route_base.zig");

const c = contract.spec;
const g = contract.gates;
const block = base.block;
const local_blocks = base.local_blocks;
const ranked = base.ranked;
const reads = base.reads;
const max_keys = base.max_keys;
const max_blocks = base.max_blocks;
const chunk = base.chunk;
const scale = base.scale;
const idx_scale = base.idx_scale;
const floor_ = base.floor_;
const clamp_at = base.clamp_at;
const neg = base.neg;
const UdBlocks = base.UdBlocks;
const UdRoute = base.UdRoute;
const u = base.u;
const tiny = base.tiny;
const gamma = base.gamma;
const Range = base.Range;
const spread = base.spread;
const clampScore = base.clampScore;
const wanted = base.wanted;
const Edit = base.Edit;
const Control = base.Control;
const pipeline = base.pipeline;
const intIn = base.intIn;

// --- ud_route: each query's remote blocks, entmax-1.5 over the top 16, the first 12 with their bias ----------------

const route_rule = "candidates: complete blocks c <= floor(pos/32) - 4 holding an image token of the query's page (any such block when the page has none in range or the query has none; every remote block when that leaves none); x = clamp(q.kbar/sqrt(128), +-1e4) * sqrt(128) for a candidate, -1e4 for any other block, so a candidate below -1e4 ranks after them; Metal's clamp sends a NaN score to -1e4 (x = -113137.08) and an infinite one to its end; the top 16 by x descending, ties to the lower block, then entmax-1.5 over those 16 (tau from the sorted prefix sums, support the count of tau_k <= X_k, at least 1); the first 12 kept when candidates, bias log(max(p, 1e-4)) less the kept mean, 0 and block -1 otherwise, the bias rounded to bf16 at prefill; picks exact (the test's scores are exact in fp32), each bias within float64's range through the kernel's fp32 steps";

pub const Route = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    label: []const u8,
    rows: usize = 0,
    keys: usize = 0,
    nblk: usize = 0,
    complete: usize = 0,
    round: bool = false,
    iq: []f32 = &.{},
    kb: []f32 = &.{},
    bmin: []i32 = &.{},
    bmax: []i32 = &.{},
    qpage: []i32 = &.{},
    qpos: []u32 = &.{},
    cand: []bool = &.{},
    rem: []bool = &.{},
    ins: [6]rig_.Ref = undefined,
    bias: rig_.Ref = undefined,
    sel: rig_.Ref = undefined,
    msg: [300]u8 = undefined,

    /// rows: queries (the last `rows` positions of the keys); n: the 12 reads; k: keys in the context.
    pub fn spec(label: []const u8) c.Spec {
        return .{
            .name = label,
            .accum = .f32,
            .depth = routeDepth,
            .out = .f32,
            .nan = .{ .stated = route_rule },
            .served = &.{
                .{ .rows = 64, .n = reads, .k = 1000 },
                // decode at the longest context (4,096 blocks: both threadgroup arrays full), and its last chunk
                .{ .rows = 1, .n = reads, .k = max_keys },
                .{ .rows = chunk, .n = reads, .k = max_keys },
                .{ .rows = chunk, .n = reads, .k = chunk },
                // no remote block (4 blocks); the 4-block gap's edges at positions 127/128 and 159/160
                .{ .rows = 100, .n = reads, .k = 100 },
                .{ .rows = 41, .n = reads, .k = 161 },
                .{ .rows = 1, .n = reads, .k = 129 },
                .{ .rows = 1, .n = reads, .k = 128 },
                .{ .rows = 7, .n = reads, .k = 160 },
                // fewer than 12 candidates, 12 to 16, just past 16
                .{ .rows = 33, .n = reads, .k = 14 * block + 5 },
                .{ .rows = 32, .n = reads, .k = 19 * block },
                .{ .rows = 1, .n = reads, .k = 20 * block + 31 },
            },
        };
    }

    fn routeDepth(_: c.Shape) usize {
        return uocr.router_dim;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, label: []const u8, control: ?Control) !Route {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(a, device, "ud_route", control), .label = label };
    }
    fn free(t: *Route) void {
        inline for (.{ "iq", "kb", "bmin", "bmax", "qpage", "qpos", "cand", "rem" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn deinit(t: *Route) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Route) g.Kernel {
        return .{ .spec = spec(t.label), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Route);

    pub fn prepare(t: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != reads or shape.rows == 0 or shape.rows > chunk or shape.k < shape.rows or shape.k > max_keys) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.keys = shape.k;
        t.nblk = (shape.k + block - 1) / block;
        t.complete = shape.k / block;
        t.round = shape.rows > 1;
        const D = uocr.router_dim;
        t.iq = try t.a.alloc(f32, t.rows * D);
        t.kb = try t.a.alloc(f32, t.nblk * D);
        t.bmin = try t.a.alloc(i32, t.nblk);
        t.bmax = try t.a.alloc(i32, t.nblk);
        t.qpage = try t.a.alloc(i32, t.rows);
        t.qpos = try t.a.alloc(u32, t.rows);
        t.cand = try t.a.alloc(bool, t.nblk);
        t.rem = try t.a.alloc(bool, t.nblk);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();

        // pages: runs of text and of image pages (a page's run may share a block with the next one's)
        const page = try t.a.alloc(i32, t.keys);
        defer t.a.free(page);
        var at: usize = 0;
        var next: i32 = 0;
        while (at < t.keys) {
            const image = rng.boolean();
            const len = if (image) rng.intRangeAtMost(usize, 32, 600) else rng.intRangeAtMost(usize, 1, 150);
            const end = @min(t.keys, at + len);
            @memset(page[at..end], if (image) next else -1);
            if (image) next += 1;
            at = end;
        }
        for (0..t.nblk) |b| {
            var lo: i32 = -1;
            var hi: i32 = -1;
            for (page[b * block .. @min(t.keys, (b + 1) * block)]) |p| if (p >= 0) {
                if (lo < 0 or p < lo) lo = p;
                hi = @max(hi, p);
            };
            t.bmin[b] = lo;
            t.bmax[b] = hi;
        }
        // a query's page: the last image page at or before it; some with none, some with an unheld page
        var last: i32 = -1;
        const start = t.keys - t.rows;
        for (page[0..start]) |p| if (p >= 0) {
            last = p;
        };
        for (0..t.rows) |q| {
            if (page[start + q] >= 0) last = page[start + q];
            t.qpos[q] = @intCast(start + q);
            t.qpage[q] = if (q % 7 == 3) -1 else if (q % 13 == 5) 9999 else last;
        }

        if (fill == .stress) {
            // integer terms, dims 64..127 mirror 0..63, dim 0 adds n/128: scores exact, no page filter
            for (0..t.rows) |q| {
                const row = t.iq[q * D ..][0..D];
                row[0] = 1;
                for (1..D / 2) |d| row[d] = intIn(rng, 1, 31);
                for (0..D / 2) |d| row[d + D / 2] = row[d];
                t.qpage[q] = -1;
            }
            for (0..t.nblk) |b| {
                const row = t.kb[b * D ..][0..D];
                for (0..D / 2) |d| {
                    row[d] = intIn(rng, 1, 64);
                    row[d + D / 2] = -row[d];
                }
                row[D / 2] += intIn(rng, -1024, 1024) / 128;
            }
        } else {
            // dyadic values: each score exact in fp32 in any order, distinct ones stay distinct
            for (t.iq) |*v| v.* = intIn(rng, -8, 8) / 8;
            for (t.kb) |*v| v.* = intIn(rng, -8, 8) / 16;
            // every fifth block repeats the one before it: exact ties for every query
            for (0..t.nblk) |b| if (b % 5 == 4) @memcpy(t.kb[b * D ..][0..D], t.kb[(b - 1) * D ..][0..D]);
        }
        if (fill == .nonfinite) {
            // NaN for query 0, +-Inf at a dim of queries 1 and 2, both on query 3, NaN in block 2's summary
            const put = [_]struct { q: usize, d: usize, v: f32 }{
                .{ .q = 0, .d = 5, .v = uocr.planted[0] },
                .{ .q = 1, .d = 7, .v = uocr.planted[1] },
                .{ .q = 2, .d = 9, .v = uocr.planted[2] },
                .{ .q = 3, .d = 2, .v = uocr.planted[1] },
                .{ .q = 3, .d = 3, .v = uocr.planted[2] },
            };
            for (put) |p| if (p.q < t.rows) {
                t.iq[p.q * D + p.d] = p.v;
            };
            if (t.complete > 2) t.kb[2 * D + 11] = uocr.planted[0];
        }
        // the incomplete block's summary is never read: NaN
        if (t.complete < t.nblk) @memset(t.kb[t.complete * D ..][0..D], uocr.planted[0]);

        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.iq));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.kb));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.bmin));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.bmax));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.qpage));
        t.ins[5] = try t.rig.input(std.mem.sliceAsBytes(t.qpos));
        t.bias = try t.rig.output(t.rows * reads * 4);
        t.sel = try t.rig.output(t.rows * reads * 4);
        return true;
    }

    pub fn run(t: *Route) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |ref, i| p.e.setBuffer(ref.buf, ref.off, i);
        p.e.setBuffer(t.sel.buf, t.sel.off, 6);
        p.e.setBuffer(t.bias.buf, t.bias.off, 7);
        p.e.setValue(UdRoute{ .nblk = @intCast(t.nblk), .complete = @intCast(t.complete), .scale = scale, .idx_scale = idx_scale, .floor_ = floor_, .round_bias = @intFromBool(t.round) }, 8);
        p.e.dispatchGroups(mtl.Size.of(t.rows, 1, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    /// The biases (gate 4 reads them as fp32); the picks are checked by `check`.
    pub fn count(t: *Route) usize {
        return t.rows * reads;
    }

    const Dot = union(enum) { exact, sim: struct { acc: c.Accum, order: g.Order } };

    fn score(t: *Route, q: usize, b: usize, how: Dot) f64 {
        const D = uocr.router_dim;
        const iq = t.iq[q * D ..][0..D];
        const kb = t.kb[b * D ..][0..D];
        switch (how) {
            .exact => {
                var s: f64 = 0;
                for (iq, kb) |x, y| s += @as(f64, x) * y;
                return s;
            },
            .sim => |m| {
                var terms: [D]f64 = undefined;
                for (&terms, iq, kb) |*v, x, y| v.* = @as(f64, x) * y;
                return contract.rows.sumIn(m.acc, m.order, &terms);
            },
        }
    }

    /// A candidate's x through its fp32 steps: the score times 1/sqrt(128), clamped, times sqrt(128).
    fn xOf(s: f64, w: bool) Range {
        const z = if (std.math.isFinite(s)) Range.point(s * scale).r(w, 1).clampc(clamp_at) else Range.point(clampScore(s));
        return z.mulc(idx_scale).r(w, 1);
    }

    fn inPage(t: *Route, b: usize, qp: i32) bool {
        return t.bmin[b] >= 0 and t.bmin[b] <= qp and qp <= t.bmax[b];
    }

    /// Query q's picks and biases: points (float64) or, when `w`, each bias's range through the kernel's fp32 steps.
    fn query(t: *Route, q: usize, how: Dot, w: bool, sel: *[reads]i32, out: *[reads]Range) void {
        const qp = t.qpage[q];
        const last: i64 = @as(i64, t.qpos[q] / block) - local_blocks;
        var any_page = false;
        if (qp >= 0) for (0..t.nblk) |b| {
            if (t.inPage(b, qp)) {
                any_page = true;
                break;
            }
        };
        var any_cand = false;
        for (0..t.nblk) |b| {
            t.rem[b] = @as(i64, @intCast(b)) <= last and b < t.complete;
            t.cand[b] = t.rem[b] and (!any_page or t.inPage(b, qp));
            any_cand = any_cand or t.cand[b];
        }
        if (!any_cand) @memcpy(t.cand, t.rem);
        // the top M by x descending, ties to the lower block (a later block moves up only past a strictly lower x)
        const M = @max(1, @min(ranked, t.nblk));
        var ids: [ranked]usize = undefined;
        var xs: [ranked]f64 = undefined;
        var ss: [ranked]f64 = undefined;
        var n: usize = 0;
        for (0..t.nblk) |b| {
            const s = if (t.cand[b]) t.score(q, b, how) else 0;
            const x = if (t.cand[b]) xOf(s, false).lo else neg;
            if (n == M and !(x > xs[M - 1])) continue;
            var i = if (n < M) n else M - 1;
            if (n < M) n += 1;
            while (i > 0 and x > xs[i - 1]) : (i -= 1) {
                ids[i] = ids[i - 1];
                xs[i] = xs[i - 1];
                ss[i] = ss[i - 1];
            }
            ids[i] = b;
            xs[i] = x;
            ss[i] = s;
        }
        var xv: [ranked]Range = undefined;
        for (0..M) |k| xv[k] = if (t.cand[ids[k]]) xOf(ss[k], w) else Range.point(neg);
        // entmax-1.5 over the M values in the kernel's order of fp32 steps
        var X: [ranked]Range = undefined;
        var tau: [ranked]Range = undefined;
        var cs = Range.point(0);
        var cs2 = Range.point(0);
        var smin: usize = 0;
        var amb: usize = 0;
        for (0..M) |k| {
            X[k] = if (k == 0) Range.point(0) else xv[k].sub(xv[0]).r(w, 1).mulc(0.5);
            cs = cs.add(X[k]).r(w, 1);
            cs2 = cs2.add(X[k].sq().r(w, 1)).r(w, 1);
            const rho: f64 = @floatFromInt(k + 1);
            const mean = cs.divc(rho).r(w, 6);
            const mean_sq = cs2.divc(rho).r(w, 6);
            const ss_ = mean_sq.sub(mean.sq().r(w, 1)).r(w, 1).mulc(rho).r(w, 1);
            const delta = Range.point(1).sub(ss_).r(w, 1).divc(rho).r(w, 6).maxc(0);
            const root = Range{ .lo = @sqrt(delta.lo), .hi = @sqrt(delta.hi) };
            tau[k] = mean.sub(root.r(w, 6)).r(w, 1);
            if (tau[k].hi <= X[k].lo) smin += 1 else if (tau[k].lo <= X[k].hi) amb += 1;
        }
        var tau_star = tau[@max(smin, 1) - 1];
        for (@max(smin, 1)..@max(smin + amb, 1) + 1) |s| tau_star = tau_star.join(tau[s - 1]);
        const K = @max(1, @min(reads, M));
        var logp: [reads]Range = undefined;
        var total = Range.point(0);
        var valid: usize = 0;
        for (0..K) |s| {
            const d = X[s].sub(tau_star).r(w, 1).maxc(0);
            const pm = d.sq().r(w, 1).maxc(floor_);
            // precise log: 4 ulps of the result, and fp32's ulp at 1 where the result nears 0
            logp[s] = (Range{ .lo = @log(pm.lo), .hi = @log(pm.hi) }).r(w, 8);
            if (w) logp[s] = .{ .lo = logp[s].lo - 0x1p-23, .hi = logp[s].hi + 0x1p-23 };
            if (t.cand[ids[s]]) {
                total = total.add(logp[s]).r(w, 1);
                valid += 1;
            }
        }
        const mu = total.divc(@floatFromInt(@max(valid, 1))).r(w, 6);
        for (0..reads) |s| {
            const ok = s < K and t.cand[ids[s]];
            sel[s] = if (ok) @intCast(ids[s]) else -1;
            const b = if (ok) logp[s].sub(mu).r(w, 1) else Range.point(0);
            out[s] = if (t.round) b.bf16() else b;
        }
    }

    pub fn reference(t: *Route, want: []f64, mag: []f64) !void {
        for (0..t.rows) |q| {
            var sel: [reads]i32 = undefined;
            var b: [reads]Range = undefined;
            t.query(q, .exact, false, &sel, &b);
            for (b, 0..) |x, s| want[q * reads + s] = x.lo;
        }
        for (mag, want) |*m, x| m.* = @abs(x);
    }

    pub fn bounds(t: *Route, want: []const f64, _: []const f64, allow: []f64) !void {
        for (0..t.rows) |q| {
            var sel: [reads]i32 = undefined;
            var b: [reads]Range = undefined;
            t.query(q, .exact, true, &sel, &b);
            for (b, 0..) |x, s| allow[q * reads + s] = spread(want[q * reads + s], x);
        }
    }

    pub fn simulate(t: *Route, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..t.rows) |q| {
            var sel: [reads]i32 = undefined;
            var b: [reads]Range = undefined;
            t.query(q, .{ .sim = .{ .acc = acc, .order = order } }, false, &sel, &b);
            for (b, 0..) |x, s| out[q * reads + s] = x.lo;
        }
    }

    /// The picks bit for bit and each bias inside its range.
    pub fn check(t: *Route, at: g.Gate) !?[]const u8 {
        const n = t.count();
        const sel = std.mem.bytesAsSlice(i32, t.rig.bytes(t.sel, n * 4));
        const bias = std.mem.bytesAsSlice(f32, t.rig.bytes(t.bias, n * 4));
        for (0..t.rows) |q| {
            var want: [reads]i32 = undefined;
            var b: [reads]Range = undefined;
            t.query(q, .exact, true, &want, &b);
            for (0..reads) |s| {
                const i = q * reads + s;
                if (sel[i] != want[s]) return try std.fmt.bufPrint(&t.msg, "query {d} (position {d}, page {d}) slot {d} reads block {d} where the host reads {d}", .{ q, t.qpos[q], t.qpage[q], s, sel[i], want[s] });
                if (!b[s].holds(bias[i])) return try std.fmt.bufPrint(&t.msg, "query {d} slot {d} (block {d}) bias {e:.8} outside [{e:.8}, {e:.8}]", .{ q, s, want[s], bias[i], b[s].lo, b[s].hi });
            }
        }
        if (at == .nonfinite) {
            var planted = false;
            for (t.iq) |v| planted = planted or !std.math.isFinite(v);
            if (!planted) return try std.fmt.bufPrint(&t.msg, "no non-finite score planted", .{});
        }
        return null;
    }
};
