//! ud_route under the contract: near-tied picks may swap, so the kernel's picks and bias intervals are checked.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const k = @import("metal_uocr_decoder_common.zig");

const c = contract.spec;
const g = contract.gates;
const D = k.router_dim;
const Mutation = k.Mutation;
const Range = k.Range;
const keep_n = k.read_blocks;
const rank_n = k.ranked_blocks;
const none: u32 = 0xffffffff;
const clamp_at: f64 = 1e4;
const floor: f64 = @as(f32, 1e-4);
const u = c.unit(.f32);
const masks_n = 4;

fn clampNan(x: f64) f64 {
    return if (std.math.isNan(x)) x else std.math.clamp(x, -clamp_at, clamp_at);
}

/// A score's 128-term dot, the scale and the clamp, the router's scale.
fn depth(_: c.Shape) usize {
    return D + 2;
}

/// The entmax-1.5 threshold over sorted X (X[0] = 0): prefix taus from means, support = count(tau <= X).
fn thresholds(xs: []const f64, tau: []f64) usize {
    var s1: f64 = 0;
    var s2: f64 = 0;
    var support: usize = 0;
    for (xs, tau, 0..) |x, *t, i| {
        s1 += x;
        s2 += x * x;
        const rho: f64 = @floatFromInt(i + 1);
        const mean = s1 / rho;
        const delta = (1 - rho * (s2 / rho - mean * mean)) / rho;
        t.* = mean - @sqrt(@max(delta, 0));
        if (t.* <= x) support += 1;
    }
    return @max(support, 1) - 1;
}

/// Each prefix's threshold error in the kernel: its inputs off by up to `ex` each, then its fp32 steps.
fn thresholdErrors(xs: []const f64, ex: f64, err: []f64) void {
    var s1: f64 = 0;
    var s2: f64 = 0;
    for (xs, err, 0..) |x, *e, i| {
        s1 += x;
        s2 += x * x;
        const rho: f64 = @floatFromInt(i + 1);
        const mean = s1 / rho;
        const msq = s2 / rho;
        var dev: f64 = 0;
        for (xs[0 .. i + 1]) |y| dev += @abs(y - mean);
        const diff = msq - mean * mean;
        const ss = rho * diff;
        const delta = (1 - ss) / rho;
        const e_mean = 2 * u * @abs(mean);
        const e_diff = 3 * u * msq + 5 * u * mean * mean + u * @abs(diff);
        const e_ss = rho * e_diff + u * @abs(ss) + u * @abs(1 - ss);
        const e_delta = (4 * ex * dev + rho * 4 * ex * ex + e_ss) / rho + u * @abs(delta);
        const d0 = @max(delta, 0);
        const root = @max(@sqrt(d0 + e_delta) - @sqrt(d0), @sqrt(d0) - @sqrt(@max(d0 - e_delta, 0))) + 2 * u * @sqrt(d0 + e_delta);
        e.* = ex + e_mean + root + u * (@abs(mean) + @sqrt(d0) + root);
    }
}

pub const Route = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    words: usize = 0,
    q: []u16 = &.{},
    kb: []f32 = &.{},
    masks: []u32 = &.{},
    qmask: []i32 = &.{},
    /// A row's scores and their error bounds, and whether each block is a candidate (scratch for one row).
    v: []f64 = &.{},
    e: []f64 = &.{},
    cand: []bool = &.{},
    ins: [4]rig_.Ref = undefined,
    bias: rig_.Ref = undefined,
    sel: rig_.Ref = undefined,
    msg: [240]u8 = undefined,

    /// Rows: query rows, the last of `k` positions; n: a row's read blocks (12).
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr decoder ud_route remote blocks, entmax-1.5",
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .nan = .{ .stated = "a NaN score is never picked (a row of NaN reads no block, biases 0); an infinite one clamps at 1e4" },
            // a prompt chunk, an odd one, the longest context's step, a short one
            .served = &.{
                .{ .rows = k.prompt_rows, .n = keep_n, .k = k.prompt_rows },  .{ .rows = 37, .n = keep_n, .k = 1061 },
                .{ .rows = 1, .n = keep_n, .k = k.context_blocks * k.block }, .{ .rows = 1, .n = keep_n, .k = 700 },
            },
            // the scores' dots sum exactly in fp32 at gate 4; the entmax's means do not
            .sums = .{ .exact = false },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !Route {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "ud_route"), .m = m };
    }
    pub fn deinit(t: *Route) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Route) void {
        inline for (.{ "q", "kb", "masks", "qmask", "v", "e", "cand" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Route) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Route);

    fn pos0(t: *const Route) usize {
        return t.shape.k - t.shape.rows;
    }
    /// Row r's remote blocks: those wholly before its window's 4 blocks.
    fn remote(t: *const Route, r: usize) usize {
        const b = (t.pos0() + r) / k.block;
        return if (b >= 4) b - 3 else 0;
    }

    pub fn prepare(t: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != keep_n or shape.k < shape.rows or shape.k > k.context_blocks * k.block) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const rows = shape.rows;
        const blocks = @max(t.remote(rows - 1), 1);
        t.words = (blocks + 31) / 32;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.q = try t.a.alloc(u16, rows * D);
        t.kb = try t.a.alloc(f32, blocks * D);
        t.v = try t.a.alloc(f64, blocks);
        t.e = try t.a.alloc(f64, blocks);
        t.cand = try t.a.alloc(bool, blocks);
        if (fill == .stress) {
            // q's halves mirror with the sign turned, keys' too, dim 64 moved by a distinct integer: huge partials
            const shift = try t.a.alloc(u32, blocks);
            defer t.a.free(shift);
            for (shift, 0..) |*x, i| x.* = @intCast(i);
            rng.shuffle(u32, shift);
            for (0..rows) |r| for (0..D / 2) |d| {
                const x: f32 = if (d == 0) 1 else @floatFromInt(1 + rng.uintLessThan(u32, 7));
                t.q[r * D + d] = c.toBf16(x);
                t.q[r * D + D / 2 + d] = c.toBf16(-x);
            };
            for (0..blocks) |b| for (0..D / 2) |d| {
                const x: f32 = @floatFromInt(128 + rng.uintLessThan(u32, 128));
                t.kb[b * D + d] = x;
                t.kb[b * D + D / 2 + d] = x - @as(f32, if (d == 0) @floatFromInt(shift[b]) else 0);
            };
        } else {
            k.fillBf16(rng, t.q, 1);
            for (t.kb) |*x| x.* = (rng.float(f32) * 2 - 1) * 0.5;
        }
        // mask 0 holds no block (its rows fall back to every remote block); the rest about a third each
        t.masks = try t.a.alloc(u32, masks_n * t.words);
        for (t.masks, 0..) |*w, i| w.* = if (i < t.words) 0 else rng.int(u32) & rng.int(u32) | rng.int(u32) & rng.int(u32) & rng.int(u32);
        t.qmask = try t.a.alloc(i32, rows);
        for (t.qmask) |*m| m.* = @as(i32, @intCast(rng.uintLessThan(u32, masks_n + 1))) - 1;
        if (fill == .nonfinite) {
            // row 0's query NaN; block 2 +Inf where every query weighs it up, block 3 NaN; both candidates for all
            for (t.q[0..D]) |*x| x.* = c.toBf16(k.nan32);
            for (1..rows) |r| t.q[r * D + 7] = c.toBf16(1);
            if (blocks > 3) {
                t.kb[2 * D + 7] = k.inf32;
                t.kb[3 * D + 9] = k.nan32;
            }
            @memset(t.qmask, -1);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.kb));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.masks));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.qmask));
        t.bias = try t.rig.output(rows * keep_n * 4);
        t.sel = try t.rig.output(rows * keep_n * 4);
        return true;
    }

    pub fn run(t: *Route) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.sel.buf, t.sel.off, 4);
        p.e.setBuffer(t.bias.buf, t.bias.off, 5);
        p.e.setValue(extern struct { pos0: u32, words: u32 }{ .pos0 = @intCast(t.pos0()), .words = @intCast(t.words) }, 6);
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Route) usize {
        return t.shape.rows * keep_n;
    }

    /// Row r's scores (dots in `acc`, `order`), their error bounds and candidates; returns the remote count.
    fn scores(t: *Route, r: usize, acc: c.Accum, order: g.Order) usize {
        const n = t.remote(r);
        var terms: [D]f64 = undefined;
        var own: usize = 0;
        for (0..n) |b| {
            var dot: f64 = 0;
            var m: f64 = 0;
            for (&terms, 0..) |*x, d| {
                x.* = @as(f64, c.fromBf16(t.q[r * D + d])) * t.kb[b * D + d];
                dot += x.*;
                m += @abs(x.*);
            }
            if (acc != .none) dot = contract.rows.sumIn(acc, order, &terms);
            const z = clampNan(dot * k.inv_sqrt_hd);
            t.v[b] = z * k.router_scale;
            const chain = if (t.fill == .stress or !std.math.isFinite(m)) 0 else c.gamma(D, u) * m * k.inv_sqrt_hd;
            t.e[b] = (chain + u * @abs(z)) * k.router_scale + u * @abs(t.v[b]);
            const mk = t.qmask[r];
            t.cand[b] = mk < 0 or (t.masks[@as(usize, @intCast(mk)) * t.words + b / 32] >> @intCast(b % 32)) & 1 != 0;
            if (t.cand[b]) own += 1;
        }
        if (own == 0) @memset(t.cand[0..n], true);
        return n;
    }

    /// The best `want` candidates by score not in `skip`, best first, ties to the lower block; NaN never.
    fn rank(t: *Route, n: usize, skip: []const u32, out: []u32) usize {
        var got: usize = 0;
        while (got < out.len) : (got += 1) {
            var best: ?usize = null;
            for (0..n) |b| {
                if (!t.cand[b] or std.math.isNan(t.v[b])) continue;
                if (std.mem.indexOfScalar(u32, skip, @intCast(b)) != null or std.mem.indexOfScalar(u32, out[0..got], @intCast(b)) != null) continue;
                if (best == null or t.v[b] > t.v[best.?]) best = b;
            }
            out[got] = @intCast(best orelse break);
        }
        return got;
    }

    /// Exact biases for picks `sel` (the first `keep`) and ranked extras `more` (to 16 in all).
    fn biases(t: *Route, sel: []const u32, more: []const u32, out: *[keep_n]f64) void {
        var buf: [rank_n]f64 = undefined;
        var tau: [rank_n]f64 = undefined;
        const xs = buf[0 .. sel.len + more.len];
        const top = t.sorted(sel, more, xs);
        const ts = tau[thresholds(xs, tau[0..xs.len])];
        var mu: f64 = 0;
        @memset(out, 0);
        for (sel, 0..) |b, i| {
            const y = @max((t.v[b] - top) / 2 - ts, 0);
            out[i] = @log(@max(y * y, floor));
            mu += out[i];
        }
        mu /= @floatFromInt(@max(sel.len, 1));
        for (out[0..sel.len]) |*x| x.* -= mu;
    }

    /// The picks' and extras' values sorted, less the largest, halved: the entmax's X. Returns the largest.
    fn sorted(t: *Route, sel: []const u32, more: []const u32, xs: []f64) f64 {
        for (sel, xs[0..sel.len]) |b, *x| x.* = t.v[b];
        for (more, xs[sel.len..]) |b, *x| x.* = t.v[b];
        std.mem.sort(f64, xs, {}, std.sort.desc(f64));
        const top = xs[0];
        for (xs) |*x| x.* = (x.* - top) / 2;
        return top;
    }

    /// Each pick's bias interval: scores off by their bounds, threshold errors, near-tie supports, logs, mean.
    fn intervals(t: *Route, sel: []const u32, more: []const u32, out: *[keep_n]Range) void {
        var buf: [rank_n]f64 = undefined;
        var tau: [rank_n]f64 = undefined;
        var err: [rank_n]f64 = undefined;
        const n = sel.len + more.len;
        const xs = buf[0..n];
        const top = t.sorted(sel, more, xs);
        var worst: f64 = 0;
        var big: f64 = 0;
        for (sel) |b| worst = @max(worst, t.e[b]);
        for (more) |b| worst = @max(worst, t.e[b]);
        for (xs) |x| big = @max(big, @abs(x));
        const ex = 2 * worst + u * big;
        const at = thresholds(xs, tau[0..n]);
        thresholdErrors(xs, ex, err[0..n]);
        var lo: usize = 0;
        var hi: usize = 0;
        for (xs, tau[0..n], err[0..n]) |x, tt, e| {
            if (tt - x <= -(e + ex)) lo += 1;
            if (tt - x <= e + ex) hi += 1;
        }
        var ets: f64 = 0;
        for (@max(lo, 1)..@max(hi, 1) + 1) |s| ets = @max(ets, @abs(tau[s - 1] - tau[at]) + err[s - 1]);
        const ts = tau[at];
        var mu = Range.point(0);
        for (sel, 0..) |b, i| {
            const x = (t.v[b] - top) / 2;
            const ylo = @max(x - ex - ts - ets, 0) * (1 - 2 * u);
            const yhi = @max(x + ex - ts + ets, 0) * (1 + 2 * u);
            const l = Range{ .lo = @log(@max(ylo * (1 - u) * ylo, floor)), .hi = @log(@max(yhi * (1 + u) * yhi, floor)) };
            out[i] = l.widen(4 * u * @max(@abs(l.lo), @abs(l.hi)) + 0x1p-22);
            mu = mu.add(out[i]);
        }
        const kn: f64 = @floatFromInt(@max(sel.len, 1));
        mu = (Range{ .lo = mu.lo / kn, .hi = mu.hi / kn }).fp32(3);
        for (out[0..sel.len]) |*r| r.* = (Range{ .lo = r.lo - mu.hi, .hi = r.hi - mu.lo }).fp32(1);
    }

    fn outputs(t: *Route, acc: c.Accum, order: g.Order, want: []f64) void {
        var sel: [rank_n]u32 = undefined;
        var out: [keep_n]f64 = undefined;
        for (0..t.shape.rows) |r| {
            const n = t.rank(t.scores(r, acc, order), &.{}, &sel);
            const kept = @min(n, keep_n);
            t.biases(sel[0..kept], sel[kept..n], &out);
            @memcpy(want[r * keep_n ..][0..keep_n], &out);
        }
    }

    pub fn reference(t: *Route, want: []f64, mag: []f64) !void {
        t.outputs(.none, .sequential, want);
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn simulate(t: *Route, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.outputs(acc, order, out);
    }

    /// Gate 4's allowance: the interval about the exact biases at the exact picks.
    pub fn bounds(t: *Route, want: []const f64, _: []const f64, allow: []f64) !void {
        var sel: [rank_n]u32 = undefined;
        var r_: [keep_n]Range = undefined;
        for (0..t.shape.rows) |r| {
            const n = t.rank(t.scores(r, .none, .sequential), &.{}, &sel);
            const kept = @min(n, keep_n);
            t.intervals(sel[0..kept], sel[kept..n], &r_);
            for (0..keep_n) |i| {
                const j = r * keep_n + i;
                allow[j] = if (i < kept) @max(k.spread(want[j], r_[i]), 0x1p-149) else 0x1p-149;
            }
        }
    }

    fn say(t: *Route, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&t.msg, fmt, args) catch "check failed (message too long)";
    }

    /// The kernel's picks and biases, row by row (gates 1 and 3).
    pub fn check(t: *Route, _: g.Gate) !?[]const u8 {
        const bias = t.rig.bytes(t.bias, t.count() * 4);
        const picked = std.mem.bytesAsSlice(u32, t.rig.bytes(t.sel, t.count() * 4));
        var want: [rank_n]u32 = undefined;
        var more: [rank_n]u32 = undefined;
        var r_: [keep_n]Range = undefined;
        for (0..t.shape.rows) |r| {
            const n = t.scores(r, .none, .sequential);
            const kept = @min(t.rank(n, &.{}, &want), keep_n);
            const sel: []const u32 = @alignCast(picked[r * keep_n ..][0..keep_n]);
            var got: usize = 0;
            while (got < keep_n and sel[got] != none) got += 1;
            if (got != kept) return t.say("row {d}: {d} blocks read where {d} candidates allow {d}", .{ r, got, n, kept });
            for (sel[got..], got..) |b, i| {
                if (b != none) return t.say("row {d} slot {d}: block {d} after the last pick", .{ r, i, b });
                if (c.read(.f32, bias, r * keep_n + i) != 0) return t.say("row {d} slot {d}: an unused bias is not 0", .{ r, i });
            }
            for (sel[0..got], 0..) |b, i| {
                if (b >= n or !t.cand[b] or std.math.isNan(t.v[b])) return t.say("row {d} pick {d}: block {d} is no candidate", .{ r, i, b });
                if (std.mem.indexOfScalar(u32, sel[0..i], b) != null) return t.say("row {d} pick {d}: block {d} picked twice", .{ r, i, b });
                if (i > 0 and t.v[b] > t.v[sel[i - 1]] + t.e[b] + t.e[sel[i - 1]]) return t.say("row {d} pick {d}: block {d} ranks above the pick before it", .{ r, i, b });
            }
            if (got == 0) continue;
            const last = sel[got - 1];
            for (0..n) |b| if (t.cand[b] and !std.math.isNan(t.v[b]) and std.mem.indexOfScalar(u32, sel[0..got], @intCast(b)) == null) {
                if (got == keep_n and t.v[b] > t.v[last] + t.e[b] + t.e[last]) return t.say("row {d}: block {d} beats the last pick {d} but is not read", .{ r, b, last });
            };
            const extra = t.rank(n, sel[0..got], more[0 .. rank_n - got]);
            t.intervals(sel[0..got], more[0..extra], &r_);
            for (r_[0..got], 0..) |iv, i| {
                const x = c.read(.f32, bias, r * keep_n + i);
                if (!(x >= iv.lo and x <= iv.hi)) return t.say("row {d} pick {d} (block {d}): bias {e:.6} outside [{e:.6}, {e:.6}]", .{ r, i, sel[i], x, iv.lo, iv.hi });
            }
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Route) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Mutation{ .none, .route_no_mean }) |m| {
            const x = try s.a.create(Route);
            x.* = try Route.init(s.a, s.device, s.queue, m);
            const kern = x.kernel();
            if (!k.wanted(only, kern.spec.name)) {
                x.deinit();
                s.a.destroy(x);
                continue;
            }
            try s.probes.append(s.a, x);
            try s.kernels.append(s.a, kern);
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        s.probes.deinit(s.a);
    }
};

/// Gate 5's texts for this group's launch.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?k.Texts {
    return k.limitsText(name, &.{"ud_route"});
}
