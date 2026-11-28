//! Hy4-flash's tensor-unit flash attention over latent KV slots (mla_nax.metal), in attention_range's float64 range.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const nax = @import("metal_hy4_nax.zig");
const fl = h.fl;
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const threads = contract.reference_threads;
const mw = h.hy.mla_weights;

const W = mw.absorbed_stride; // a slot's words: 512 latent (the values), then 64 rope
const L = mw.latent;
const TK = 32; // a tile's keys
const window = h.hy.serve_metal.window;
const max_splits = h.hy.mla_metal.Nax.max_splits;
const scale: f64 = h.hy.mla_attention.score_scale;
const u: f64 = 0x1p-24;
const nan16 = 0x7fc0;

/// sliding: 64 heads over [end - 512, end); sparse: 32 heads over [0, end), which serve_metal.zig's layout gives.
pub const Kind = enum { sliding, sparse };
/// masked and nan_row plant the kernel's two gate-1 findings, each in an entry of its own; edit is a source fault.
pub const Variant = enum { plain, masked, nan_row, edit };
pub const Case = struct { kind: Kind, split: bool = false, variant: Variant = .plain };

pub const cases = [_]Case{
    .{ .kind = .sliding },
    .{ .kind = .sparse },
    .{ .kind = .sparse, .variant = .masked },
    .{ .kind = .sparse, .variant = .nan_row },
    .{ .kind = .sparse, .variant = .edit },
    .{ .kind = .sliding, .split = true },
    .{ .kind = .sparse, .split = true },
    .{ .kind = .sliding, .split = true, .variant = .edit },
};

pub fn base(v: Case) []const u8 {
    if (v.variant == .masked) return "metal hy4 mla_nax tf_hy4_attend_nax sparse, non-finite values at keys a row masks in its tiles";
    if (v.variant == .nan_row) return "metal hy4 mla_nax tf_hy4_attend_nax sparse, a NaN query: every score of its row NaN";
    return switch (v.kind) {
        .sliding => if (v.split) "metal hy4 mla_nax tf_hy4_attend_nax_split + tf_hy4_attend_combine sliding (64 heads, 512-key window)" else "metal hy4 mla_nax tf_hy4_attend_nax sliding (64 heads, 512-key window)",
        .sparse => if (v.split) "metal hy4 mla_nax tf_hy4_attend_nax_split + tf_hy4_attend_combine sparse (32 heads, keys from 0)" else "metal hy4 mla_nax tf_hy4_attend_nax sparse (32 heads, keys from 0)",
    };
}

fn fault(v: Case) ?[]const u8 {
    if (v.variant != .edit) return null;
    return if (v.split) "the sink dropped from the combine's denominator" else "one key past each row seen";
}

/// The controls' edits: the shared body's key test (both attend pipelines), and the combine's sink term.
fn edit(v: Case) ?fl.Edit {
    if (v.variant != .edit) return null;
    if (v.split) return .{ .kernel = "tf_hy4_attend_combine", .from = "float denom = precise::exp(sink - top);", .to = "float denom = 0.0f;" };
    return .{ .kernel = "hy4_attend_nax", .from = "key < hi[i]", .to = "key <= hi[i]" };
}

fn finding(v: Variant) ?[]const u8 {
    return switch (v) {
        .masked => "a non-finite value at a key masked for a row but read in its block's 32-key tiles reaches that row as NaN " ++
            "(P = 0 times Inf or NaN inside the P x V mma), where float64 leaves the masked key out",
        .nan_row => "a row whose every score is NaN (a NaN query) gives 0 where float64 gives NaN: Metal's max drops each NaN, " ++
            "so the running max stays -Inf, and at a max of -Inf every weight is taken as 0 and the sink alone is left",
        else => null,
    };
}

// rows: the pass's rows; n: the split count (1: no split); k: the first row's position. Sparse stops at 2,048 keys.
const sliding_served = [_]c.Shape{
    .{ .rows = h.window_rows, .n = 1, .k = 0 },      .{ .rows = h.decode_rows, .n = 1, .k = 0 },
    .{ .rows = h.window_rows, .n = 1, .k = 3000 },   .{ .rows = 64, .n = 1, .k = 262000 },
    .{ .rows = h.decode_rows, .n = 1, .k = 262000 }, .{ .rows = h.prompt_rows, .n = 1, .k = 0 },
};
const sparse_served = [_]c.Shape{
    .{ .rows = h.window_rows, .n = 1, .k = 0 },   .{ .rows = h.decode_rows, .n = 1, .k = 0 },
    .{ .rows = h.window_rows, .n = 1, .k = 700 }, .{ .rows = h.prompt_rows, .n = 1, .k = 0 },
    .{ .rows = 16, .n = 1, .k = 2032 },           .{ .rows = h.decode_rows, .n = 1, .k = 2047 },
};
// the split form, which Nax.flash encodes for one row block when splits() asks for more than one (today it never does)
const sliding_split = [_]c.Shape{
    .{ .rows = h.window_rows, .n = 2, .k = 0 },      .{ .rows = h.decode_rows, .n = 8, .k = 3000 },
    .{ .rows = h.window_rows, .n = 3, .k = 262000 }, .{ .rows = 16, .n = 8, .k = 3000 },
    .{ .rows = h.decode_rows, .n = 2, .k = 262000 },
};
const sparse_split = [_]c.Shape{
    .{ .rows = h.window_rows, .n = 2, .k = 0 },   .{ .rows = h.decode_rows, .n = 8, .k = 2047 },
    .{ .rows = h.window_rows, .n = 4, .k = 700 }, .{ .rows = 16, .n = 5, .k = 2032 },
};

/// P x V's chain over the most tiles a row's block reads (the range's own counts replace it through `bounds`).
fn depth(s: c.Shape) usize {
    return (2 * ar.mma_step + 1) * ((s.k + s.rows) / TK + 2);
}

pub fn spec(v: Case, name: []const u8) c.Spec {
    const served: []const c.Shape = switch (v.kind) {
        .sliding => if (v.split) &sliding_split else &sliding_served,
        .sparse => if (v.split) &sparse_split else &sparse_served,
    };
    const keys: c.Tile = .{ .size = TK, .min = 0 };
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .ops = 5, // precise::divide (2.5 ulps)
        .out = .bf16,
        .tiles = if (v.split) .{ .{ .size = 16 }, .{ .size = max_splits, .min = 2 }, keys } else .{ .{ .size = 16 }, .{}, keys },
        .served = served,
        .known = if (finding(v.variant)) |f| .{ .gate = .nonfinite, .finding = f } else null,
    };
}

/// A row block's `tiles` (and `splits` runs) in attention_range's terms; the fields' comments count the roundings.
fn form(tiles: usize, splits: usize) ar.Form {
    const t: f64 = @floatFromInt(tiles);
    const z: f64 = @floatFromInt(splits);
    return .{
        // four simdgroups' 9 mma steps over 144 dims each, then 3 adds
        .score = 9 * ar.mma_step + 3,
        .fast_exp = false,
        .chunk_exp = false,
        // P enters P x V as bf16
        .p_rel = 0x1p-8,
        // o: 2 mma steps and a rescale a tile, then keep (or each run's merge weight and add)
        .num = (2 * ar.mma_step + 1) * t + 1 + 2 * z,
        // l: 8 lane adds, 2 shuffles, then a rescale and an add a tile; keep and the sink's add (or the runs')
        .den = 2 * t + 13 + 2 * z,
        // precise::divide, 2.5 ulps
        .div = 5,
        // precise exp (4 ulps): the weight's own, a rescale each later tile, and keep or the merge's
        .exp_ulps = .{ 4 * (t + 1), 0 },
    };
}

fn gammaOf(n: f64) f64 {
    return n * u / (1 - n * u);
}

pub const Attend = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    combine: ?mtl.Pipeline,
    v: Case,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    splits: usize = 1,
    start: usize = 0,
    heads: usize = 0,
    used: usize = 0,
    fill: g.Fill = .random,
    q: []u16 = &.{},
    cache: []u16 = &.{},
    map: []u32 = &.{},
    ranges: [][2]u32 = &.{},
    sinks: []f32 = &.{},
    point: []f64 = &.{},
    half: []f64 = &.{},
    ins: [5]rig_.Ref = undefined,
    parts: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// The library as Nax.load builds it; the split form's combine from the same text.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Case) !Attend {
        const pipe = try nax.pipeline(a, device, if (v.split) null else edit(v), if (v.split) "tf_hy4_attend_nax_split" else "tf_hy4_attend_nax");
        errdefer pipe.deinit();
        const comb = if (v.split) try nax.pipeline(a, device, edit(v), "tf_hy4_attend_combine") else null;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .combine = comb, .v = v };
    }
    fn free(t: *Attend) void {
        inline for (.{ "q", "cache", "map", "ranges", "sinks", "point", "half" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn deinit(t: *Attend) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
        if (t.combine) |p| p.deinit();
    }
    pub fn kernel(t: *Attend) g.Kernel {
        t.name = fl.named(base(t.v), fault(t.v), &t.label);
        return .{ .spec = spec(t.v, t.name), .ctx = t, .vt = &vt, .control = if (t.v.variant == .edit) .shapes else null };
    }
    const vt = g.hooks(Attend);

    /// Key `key`'s cache slot: its tile's first slot (the tiles list) plus its place in the tile.
    fn slotOf(t: *const Attend, key: usize) usize {
        return t.map[key / TK] + key % TK;
    }
    /// The tiles row r's 16-row block reads: from its first row's first key to its last row's end.
    fn blockTiles(t: *const Attend, r: usize) usize {
        const b0 = r / 16 * 16;
        const b1 = @min(b0 + 16, t.rows) - 1;
        return (t.ranges[b1][1] + TK - 1) / TK - t.ranges[b0][0] / TK;
    }

    /// serve_metal.zig's ranges; the tiles the rows reach shuffled in a small cache, every other at one NaN tile.
    pub fn prepare(t: *Attend, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const most: usize = if (t.v.split) 16 else h.prompt_rows;
        if (shape.rows == 0 or shape.rows > most) return false;
        if (if (t.v.split) shape.n < 2 or shape.n > max_splits else shape.n != 1) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.splits = shape.n;
        t.start = shape.k;
        t.fill = fill;
        t.heads = if (t.v.kind == .sliding) 64 else 32;
        t.ranges = try t.a.alloc([2]u32, t.rows);
        for (t.ranges, 0..) |*g_, r| {
            const end = t.start + r + 1;
            g_.* = .{ @intCast(if (t.v.kind == .sliding) end - @min(end, window) else 0), @intCast(end) };
        }
        const first = t.ranges[0][0] / TK;
        const last = (t.ranges[t.rows - 1][1] + TK - 1) / TK;
        t.used = last - first;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.map = try t.a.alloc(u32, last);
        @memset(t.map, @intCast(t.used * TK));
        for (t.map[first..last], 0..) |*m, i| m.* = @intCast(i * TK);
        rng.shuffle(u32, t.map[first..last]);
        t.cache = try t.a.alloc(u16, (t.used + 1) * TK * W);
        t.q = try t.a.alloc(u16, t.rows * t.heads * W);
        t.sinks = try t.a.alloc(f32, t.heads);
        for (t.q) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * 3);
        for (t.cache) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.sinks) |*x| x.* = rng.float(f32) * 6 - 3;
        switch (fill) {
            .random => {},
            .nonfinite => t.plant(),
            .stress => t.stress(rng),
        }
        @memset(t.cache[t.used * TK * W ..], nan16);
        const bytes = [_][]const u8{ std.mem.sliceAsBytes(t.q), std.mem.sliceAsBytes(t.cache), std.mem.sliceAsBytes(t.ranges), std.mem.sliceAsBytes(t.map), std.mem.sliceAsBytes(t.sinks) };
        for (bytes, &t.ins) |b, *r| r.* = try t.rig.input(b);
        t.out = try t.rig.output(t.count() * 2);
        if (t.v.split) {
            const parts = t.splits * t.rows * t.heads;
            t.parts = .{ try t.rig.scratch(parts * L * 4), try t.rig.scratch(parts * 2 * 4) };
        }
        t.point = try t.a.alloc(f64, t.count());
        t.half = try t.a.alloc(f64, t.count());
        return true;
    }

    /// NaN and infinities in sinks, a value all rows see and two rope words; the findings' entries add their own.
    fn plant(t: *Attend) void {
        t.sinks[3] = fl.nan32;
        t.sinks[4] = fl.inf32;
        t.sinks[5] = -fl.inf32;
        // +Inf in a value every row sees: its score +-Inf (P x V's 0 x Inf where -Inf), never NaN
        t.cache[t.slotOf(t.ranges[t.rows - 1][0]) * W + 7] = c.toBf16(fl.inf32);
        // rope words, which only scores read: -Inf at a row's second key, NaN at its fourth (a finite third beside it)
        if (t.rows > 1) t.cache[t.slotOf(t.start + 1) * W + 530] = c.toBf16(-fl.inf32);
        if (t.rows > 3) t.cache[t.slotOf(t.start + 3) * W + 540] = c.toBf16(fl.nan32);
        switch (t.v.variant) {
            .masked => {
                // a value rows 0 to 3 mask but read in their tile, and NaN one key past the last row in its tile
                if (t.rows > 4) t.cache[t.slotOf(t.start + 4) * W + 11] = c.toBf16(fl.inf32);
                const past = t.start + t.rows;
                if (past % TK != 0) t.cache[t.slotOf(past) * W + 13] = c.toBf16(fl.nan32);
            },
            // row 2's head 6: every score NaN
            .nan_row => t.q[(2 * t.heads + 6) * W + 3] = c.toBf16(fl.nan32),
            else => {},
        }
    }

    /// Gate 4: queries 0 (each weight 1), no sink, cancelling integers on keys all rows see (32-lane shares past fp16).
    fn stress(t: *Attend, rng: std.Random) void {
        @memset(t.q, 0);
        @memset(t.cache, 0);
        @memset(t.sinks, -fl.inf32);
        const lo = t.ranges[t.rows - 1][0];
        const half = (t.ranges[0][1] - lo) / 2;
        for (0..half) |i| for (0..L) |e| {
            const v: f32 = @floatFromInt(if (i < 8) 1 + rng.uintLessThan(u32, 63) else 64 * (64 + rng.uintLessThan(u32, 128)));
            t.cache[t.slotOf(lo + i) * W + e] = c.toBf16(v);
            t.cache[t.slotOf(lo + half + i) * W + e] = c.toBf16(-v + @as(f32, @floatFromInt(@intFromBool(i < 8))));
        };
    }

    /// As Nax.flash encodes it: (heads, row blocks) groups of 128; split: (heads, 1, n), a barrier, the combine.
    pub fn run(t: *Attend) !void {
        const p = t.rig.begin();
        const e = p.e;
        const args = extern struct { rows: u32, heads: u32, scale: f32, pad: u32 }{ .rows = @intCast(t.rows), .heads = @intCast(t.heads), .scale = @floatCast(scale), .pad = @intCast(t.splits) };
        e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setBuffer(t.out.buf, t.out.off, 5);
        e.setValue(args, 6);
        if (t.combine) |comb| {
            e.setBuffer(t.parts[0].buf, t.parts[0].off, 7);
            e.setBuffer(t.parts[1].buf, t.parts[1].off, 8);
            e.dispatchGroups(mtl.Size.of(t.heads, 1, t.splits), mtl.Size.of(128, 1, 1));
            e.barrier();
            e.setPipeline(comb);
            e.setBuffer(t.parts[0].buf, t.parts[0].off, 0);
            e.setBuffer(t.parts[1].buf, t.parts[1].off, 1);
            e.setBuffer(t.ins[4].buf, t.ins[4].off, 2);
            e.setBuffer(t.out.buf, t.out.off, 3);
            e.setValue(args, 4);
            e.dispatchGroups(mtl.Size.of(t.heads, t.rows, 1), mtl.Size.of(128, 1, 1));
        } else e.dispatchGroups(mtl.Size.of(t.heads, (t.rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *const Attend) usize {
        return t.rows * t.heads * L;
    }

    /// Every head's float64 point and half range over its row's keys and its sink, rows in parallel.
    pub fn reference(t: *Attend, want: []f64, mag: []f64) !void {
        const kf = try t.a.alloc(f64, t.cache.len);
        defer t.a.free(kf);
        for (kf, t.cache) |*x, b| x.* = c.fromBf16(b);
        var vmax: f64 = 0;
        for (0..t.used * TK) |s| for (kf[s * W ..][0..L]) |x| if (std.math.isFinite(x)) {
            vmax = @max(vmax, @abs(x));
        };
        var keys: usize = 0;
        for (t.ranges) |r| keys = @max(keys, r[1] - r[0]);
        const per = W + 3 * L + 2 * (keys + 1);
        const jobs_n = threads.count(threads.maximum, t.rows);
        const buf = try t.a.alloc(f64, jobs_n * per);
        defer t.a.free(buf);
        var jobs: [threads.maximum]Job = undefined;
        for (jobs[0..jobs_n], 0..) |*j, i| j.* = .{ .t = t, .kf = kf, .vmax = vmax, .buf = buf[i * per ..][0..per], .first = i * t.rows / jobs_n, .end = (i + 1) * t.rows / jobs_n };
        threads.run(Job, jobs[0..jobs_n], .normal);
        @memcpy(want, t.point);
        for (mag, want) |*m, w| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }

    const Job = struct {
        t: *const Attend,
        kf: []const f64,
        vmax: f64,
        buf: []f64,
        first: usize,
        end: usize,
        pub fn run(j: *Job) void {
            for (j.first..j.end) |r| for (0..j.t.heads) |hd| j.t.head(r, hd, j.kf, j.vmax, j.buf);
        }
    };

    /// attention_range.head's point and half range, its loops ordered key by key so a slot's 512 values stream.
    fn head(t: *const Attend, r: usize, hd: usize, kf: []const f64, vmax: f64, buf: []f64) void {
        const lo = t.ranges[r][0];
        const n = t.ranges[r][1] - lo;
        const f = form(t.blockTiles(r), if (t.v.split) t.splits else 0);
        const qf = buf[0..W];
        const o = buf[W..][0..L];
        const rs = buf[W + L ..][0..L];
        const dv = buf[W + 2 * L ..][0..L];
        const w = buf[W + 3 * L ..][0 .. n + 1];
        const tau = buf[W + 3 * L + n + 1 ..][0 .. n + 1];
        for (qf, t.q[(r * t.heads + hd) * W ..][0..W]) |*x, b| x.* = c.fromBf16(b);
        var m = -std.math.inf(f64);
        var dmax: f64 = 0;
        for (0..n) |j| {
            const d = nax.dotAbs(qf, kf[t.slotOf(lo + j) * W ..][0..W]);
            w[j] = d[0] * scale;
            tau[j] = f.score * u * d[1] * scale + f.score_round * u * @abs(w[j]);
            m = @max(m, w[j]);
            if (std.math.isFinite(tau[j])) dmax = @max(dmax, tau[j]);
        }
        // the sink: one more weight, exp(sink - top), of value 0; -Inf leaves it out
        const sink: f64 = t.sinks[hd];
        const ns = if (std.math.isInf(sink) and sink < 0) n else n + 1;
        if (ns > n) {
            w[n] = sink;
            tau[n] = f.score_round * u * @abs(sink);
            m = @max(m, sink);
            if (std.math.isFinite(tau[n])) dmax = @max(dmax, tau[n]);
        }
        var l: f64 = 0;
        var tmax: f64 = 0;
        const eu = f.exp_ulps.?;
        for (w[0..ns], tau[0..ns]) |*p, *e| {
            const x = (m - p.*) + 2 * dmax;
            const err = e.* + u * x + (eu[0] + eu[1] * x) * 0x1p-23;
            p.* = @exp(p.* - m);
            l += p.*;
            e.* = std.math.expm1(err);
            if (std.math.isFinite(e.*)) tmax = @max(tmax, e.*);
        }
        @memset(o, 0);
        @memset(rs, 0);
        @memset(dv, 0);
        for (0..n) |j| for (o, rs, kf[t.slotOf(lo + j) * W ..][0..L]) |*a, *b, x| {
            a.* += w[j] * x;
            b.* += w[j] * @abs(x);
        };
        for (o, rs) |*a, *b| {
            a.* /= l;
            b.* /= l;
        }
        for (0..n) |j| for (dv, o, kf[t.slotOf(lo + j) * W ..][0..L]) |*d, a, x| {
            d.* += w[j] * tau[j] * @abs(x - a);
        };
        if (ns > n) for (dv, o) |*d, a| {
            d.* += w[n] * tau[n] * @abs(a);
        };
        const big = tmax >= 0.5; // scores too far apart for the spread's first order: any finite value
        const kappa = (1 + tmax) / (1 - tmax);
        const tiny = @as(f64, @floatFromInt(ns + 8)) * 0x1p-120 * (1 + vmax);
        const at0 = (r * t.heads + hd) * L;
        for (o, rs, dv, 0..) |pt, rr, dd, e| {
            t.point[at0 + e] = pt;
            if (!std.math.isFinite(pt) or big) {
                t.half[at0 + e] = if (big) std.math.inf(f64) else 0;
                continue;
            }
            const e1 = dd / l / (1 - tmax);
            const en = (gammaOf(f.num) + f.p_rel) * rr * kappa;
            const b = @abs(pt) + e1 + en;
            t.half[at0 + e] = (e1 + en + (gammaOf(f.den) + f.div * u) * b) * (1 + 0x1p-20) + tiny;
        }
    }

    pub fn intervals(t: *Attend, low: []f64, high: []f64) !void {
        for (low, high, t.point, t.half) |*l, *hi, p, hf| {
            const fin = std.math.isFinite(p) and std.math.isFinite(hf);
            l.* = if (fin) fl.bf(p - hf) else -std.math.floatMax(f64);
            hi.* = if (fin) fl.bf(p + hf) else std.math.floatMax(f64);
        }
    }

    /// Each output's farther range end from float64's value; gate 4's exact sums allow the rounding and division.
    pub fn bounds(t: *Attend, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, t.point, t.half) |*l, w, p, hf| {
            if (t.fill == .stress) {
                l.* = c.boundExact(spec(t.v, ""), w);
            } else if (!std.math.isFinite(w)) {
                l.* = 0;
            } else if (!std.math.isFinite(hf)) {
                l.* = std.math.floatMax(f64);
            } else l.* = fl.spread(w, .{ .lo = fl.bf(p - hf), .hi = fl.bf(p + hf) });
        }
    }

    /// Gate 4: every weight 1 and no sink, so each output is its row's mean value; the sums replayed in `acc`.
    pub fn simulate(t: *Attend, acc: c.Accum, order: g.Order, out: []f64) !void {
        const col = try t.a.alloc(f64, t.ranges[t.rows - 1][1]);
        defer t.a.free(col);
        for (t.ranges, 0..) |rg, r| for (0..L) |e| {
            const n = rg[1] - rg[0];
            for (col[0..n], 0..) |*x, j| x.* = c.fromBf16(t.cache[t.slotOf(rg[0] + j) * W + e]);
            const o = fl.bf(contract.rows.sumIn(acc, order, col[0..n]) / @as(f64, @floatFromInt(n)));
            for (0..t.heads) |hd| out[(r * t.heads + hd) * L + e] = o;
        };
    }
};
