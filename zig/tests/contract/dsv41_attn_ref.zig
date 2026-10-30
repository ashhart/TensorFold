//! DeepSeek V4.1 CSA2 attention for the contract: inputs for every gate and the float64 reference formula.
const std = @import("std");
const contract = @import("contract");
const tf = @import("tensorfold");

const c = contract.spec;
const rows_ = contract.rows;
const attn = tf.dsv41_attn;
const D = attn.head_dim;

pub const Layout = enum { decode, prompt };
pub const Opts = struct { layout: Layout, comp_rows: usize = 1024, timing: bool = false };

/// The longest chain of roundings: one rescale and one FMA a key, the score's tree, the sink and the rope.
pub fn depth(s: c.Shape) usize {
    return 2 * (attn.window + s.k) + 40;
}

const positions = [_]i32{ 0, 1, 63, 126, 127, 128, 129, 300, 1000, 4999 };

/// Host inputs of one call: q [rows][heads][512], window rows, compressed rows, selected indices, row records.
pub const Attn = struct {
    a: std.mem.Allocator,
    o: Opts,
    rows: usize,
    heads: usize,
    ncomp: usize,
    q: []u16,
    win: []u16,
    comp: []u16,
    idx: []i32,
    meta: []attn.Meta,
    sink: []f32,
    cs: []f32,
    qf: []f32,
    winf: []f32,
    compf: []f32,

    pub fn init(a: std.mem.Allocator, shape: c.Shape, o: Opts) !Attn {
        if (shape.n == 0 or shape.n % D != 0 or shape.k > attn.max_selected) return error.BadAttnShape;
        const rows = shape.rows;
        const heads = shape.n / D;
        const meta = try a.alloc(attn.Meta, rows);
        const idx = try a.alloc(i32, rows * attn.max_selected);
        const q = try a.alloc(u16, rows * heads * D);
        const comp = try a.alloc(u16, o.comp_rows * D);
        const sink = try a.alloc(f32, heads);
        // The window buffer and RoPE table are sized once the positions are drawn.
        return .{ .a = a, .o = o, .rows = rows, .heads = heads, .ncomp = shape.k, .q = q, .win = &.{}, .comp = comp, .idx = idx, .meta = meta, .sink = sink, .cs = &.{}, .qf = &.{}, .winf = &.{}, .compf = &.{} };
    }

    pub fn deinit(f: *Attn) void {
        f.a.free(f.q);
        f.a.free(f.win);
        f.a.free(f.comp);
        f.a.free(f.idx);
        f.a.free(f.meta);
        f.a.free(f.sink);
        f.a.free(f.cs);
        f.a.free(f.qf);
        f.a.free(f.winf);
        f.a.free(f.compf);
    }

    pub fn outputs(f: *const Attn) usize {
        return f.rows * f.heads * D;
    }

    fn rnd(rng: std.Random) u16 {
        return c.toBf16(rng.float(f32) * 4 - 2);
    }

    fn randomInt(rng: std.Random) f32 {
        return @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127);
    }

    fn setPositions(f: *Attn, mode: rows_.Fill, seed: u64, rng: std.Random) !void {
        const R = f.rows;
        var maxpos: i32 = 0;
        if (f.o.layout == .decode) {
            for (f.meta, 0..) |*m, r| {
                const p = if (mode == .random) positions[(r + seed) % positions.len] else positions[4 + (r + seed) % 6];
                m.* = .{ .pos = p, .wofs = @intCast(r * attn.window), .wbase = -1, .cofs = 0, .ncomp = @intCast(f.ncomp) };
                maxpos = @max(maxpos, p);
            }
            f.win = try f.a.alloc(u16, R * attn.window * D);
        } else {
            const pos0: i32 = if (mode == .random) ([_]i32{ 0, 100, 300 })[seed % 3] else 300;
            const wbase = @max(0, pos0 - (attn.window - 1));
            for (f.meta, 0..) |*m, r| m.* = .{ .pos = pos0 + @as(i32, @intCast(r)), .wofs = 0, .wbase = wbase, .cofs = 0, .ncomp = @intCast(f.ncomp) };
            maxpos = pos0 + @as(i32, @intCast(R)) - 1;
            f.win = try f.a.alloc(u16, @as(usize, @intCast(maxpos + 1 - wbase)) * D);
        }
        _ = rng;
        f.cs = try f.a.alloc(f32, (@as(usize, @intCast(maxpos)) + 1) * 64);
        for (f.cs[0..], 0..) |*v, i| {
            const p: f64 = @floatFromInt(i / 64);
            const k: f64 = @floatFromInt(i % 32);
            const ang = p * std.math.pow(f64, 10000, -k / 32);
            v.* = @floatCast(if (i % 64 < 32) @cos(ang) else @sin(ang));
        }
    }

    pub fn fill(f: *Attn, mode: rows_.Fill, seed: u64) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        try f.setPositions(mode, seed, rng);
        for (f.sink) |*v| v.* = if (mode == .stress) -std.math.inf(f32) else rng.float(f32) * 2 - 1;
        if (mode == .stress) {
            f.stressRows(rng);
        } else {
            for ([_][]u16{ f.q, f.win, f.comp }) |buf| f.fillBf(buf, rng);
        }
        const n = f.o.comp_rows;
        for (f.meta, 0..) |m, r| {
            const row = f.idx[r * attn.max_selected ..][0..attn.max_selected];
            for (row, 0..) |*v, i| v.* = if (i >= m.ncomp) 5 else if (mode == .random and rng.uintLessThan(u32, 7) == 0) -1 else 4 + @as(i32, @intCast(rng.uintLessThan(usize, n - 4)));
        }
        if (mode == .nonfinite) f.plant();
        try f.dequant();
    }

    /// Random bf16 values; a timing call repeats one random block, as its inputs only need to be varied.
    fn fillBf(f: *const Attn, buf: []u16, rng: std.Random) void {
        const block = if (f.o.timing) @min(buf.len, 1 << 20) else buf.len;
        for (buf[0..block]) |*v| v.* = rnd(rng);
        var at = block;
        while (at < buf.len) : (at += block) @memcpy(buf[at..][0..@min(block, buf.len - at)], buf[0..@min(block, buf.len - at)]);
    }

    /// Gate 4's inputs: integers, each row's second half its first negated, so scores are exactly 0 in any order.
    fn stressRows(f: *Attn, rng: std.Random) void {
        const half = D / 2;
        const tail = D / 2 - attn.rope_dims; // dims 192..255 and their mirrors 448..511 hold zero: no RoPE rounding
        for (0..f.rows * f.heads) |g| for (0..half) |i| {
            const v: f32 = if (i >= tail) 0 else randomInt(rng);
            f.q[g * D + i] = c.toBf16(v);
            f.q[g * D + half + i] = c.toBf16(v);
        };
        for ([_][]u16{ f.win, f.comp }) |buf| for (0..buf.len / D) |r| for (0..half) |i| {
            const v: f32 = if (i >= tail) 0 else randomInt(rng);
            buf[r * D + i] = c.toBf16(v);
            buf[r * D + half + i] = c.toBf16(-v);
        };
    }

    /// Gate 1's plants: NaN in a q, and NaN, +inf, -inf in compressed rows only one row selects, +inf in a window key.
    fn plant(f: *Attn) void {
        const inf = std.math.inf(f32);
        const nan = std.math.nan(f32);
        if (f.rows > 0) f.q[100] = c.toBf16(nan);
        const poison = [_]f32{ nan, inf, -inf };
        for (poison, 0..) |v, i| if (f.rows > i + 1 and f.ncomp > 0) {
            f.comp[i * D + 7 + 2 * i] = c.toBf16(v);
            f.idx[(i + 1) * attn.max_selected] = @intCast(i);
        };
        if (f.rows > 4) {
            const m = f.meta[4];
            const slot: usize = @intCast(if (m.wbase < 0) m.wofs + (m.pos & 127) else m.wofs + m.pos - m.wbase);
            f.win[slot * D + 13] = c.toBf16(inf);
        }
    }

    fn dequant(f: *Attn) !void {
        f.qf = try f.a.alloc(f32, f.q.len);
        f.winf = try f.a.alloc(f32, f.win.len);
        f.compf = try f.a.alloc(f32, f.comp.len);
        for (f.q, f.qf) |h, *v| v.* = c.fromBf16(h);
        for (f.win, f.winf) |h, *v| v.* = c.fromBf16(h);
        for (f.comp, f.compf) |h, *v| v.* = c.fromBf16(h);
    }

    pub fn windowRow(f: *const Attn, m: attn.Meta, t: i32) []const f32 {
        const row: usize = @intCast(if (m.wbase < 0) m.wofs + (t & 127) else m.wofs + t - m.wbase);
        return f.winf[row * D ..][0..D];
    }

    /// The row's keys in the kernel's order: window positions ascending, then the selected rows that are not negative.
    pub fn keys(f: *const Attn, r: usize, list: *std.ArrayList([]const f32)) !void {
        list.clearRetainingCapacity();
        const m = f.meta[r];
        var t = @max(0, m.pos - (attn.window - 1));
        while (t <= m.pos) : (t += 1) try list.append(f.a, f.windowRow(m, t));
        for (f.idx[r * attn.max_selected ..][0..@intCast(m.ncomp)]) |i| if (i >= 0) try list.append(f.a, f.compf[@as(usize, @intCast(m.cofs + i)) * D ..][0..D]);
    }

    /// The scores of one head over `list` as float64 from fp32 inputs, and the sums of the terms' magnitudes.
    fn scores(f: *const Attn, g: usize, list: []const []const f32, s: []f64, sm: []f64) void {
        const sc: f64 = attn.scale();
        const q = f.qf[g * D ..][0..D];
        for (list, 0..) |k, j| {
            var dot: f64 = 0;
            var mag: f64 = 0;
            for (q, k) |x, y| {
                dot += @as(f64, x) * y;
                mag += @abs(@as(f64, x) * y);
            }
            s[j] = dot * sc;
            sm[j] = mag * sc;
        }
    }

    fn rope(f: *const Attn, r: usize, x: []f64, mag: []f64) void {
        const cs = f.cs[@as(usize, @intCast(f.meta[r].pos)) * 64 ..][0..64];
        var d: usize = D - attn.rope_dims;
        while (d < D) : (d += 2) {
            const p = (d - (D - attn.rope_dims)) / 2;
            const cc: f64 = cs[p];
            const sn: f64 = cs[32 + p];
            const x0 = x[d];
            const x1 = x[d + 1];
            x[d] = x0 * cc + x1 * sn;
            x[d + 1] = x1 * cc - x0 * sn;
            const m = mag[d] + mag[d + 1];
            mag[d] = m;
            mag[d + 1] = m;
        }
    }

    /// Float64 outputs (softmax over keys and the sink, then the inverse RoPE) and each one's magnitude.
    pub fn reference(f: *const Attn, want: []f64, mag: []f64) !void {
        var list: std.ArrayList([]const f32) = .empty;
        defer list.deinit(f.a);
        const w = try f.a.alloc(f64, 3 * (attn.window + attn.max_selected));
        defer f.a.free(w);
        const n = attn.window + attn.max_selected;
        for (0..f.rows) |r| {
            try f.keys(r, &list);
            const nk = list.items.len;
            for (0..f.heads) |h| {
                const g = r * f.heads + h;
                f.scores(g, list.items, w[0..nk], w[n..][0..nk]);
                const x = want[g * D ..][0..D];
                const xm = mag[g * D ..][0..D];
                f.softmax(h, list.items, w[0..nk], w[n..][0..nk], w[2 * n ..][0..nk], x, xm);
                f.rope(r, x, xm);
            }
        }
    }

    fn softmax(f: *const Attn, h: usize, list: []const []const f32, s: []const f64, sm: []const f64, e: []f64, x: []f64, xm: []f64) void {
        const sink: f64 = f.sink[h];
        var top = sink;
        for (s) |v| if (v > top) {
            top = v;
        };
        var l = std.math.exp(sink - top);
        for (s, e) |v, *p| {
            p.* = std.math.exp(v - top);
            l += p.*;
        }
        @memset(x, 0);
        for (list, e) |k, p| for (x, k) |*o, v| {
            o.* += p * v;
        };
        for (x) |*o| o.* /= l;
        @memset(xm, 0);
        for (list, e, sm) |k, p, m| for (xm, k, x) |*o, v, y| {
            o.* += p / l * (1 + m) * (@abs(@as(f64, v)) + @abs(y));
        };
    }

    /// Gate 4: each output as an accumulator leaves it, every sum (score, row sum, value sum) in `order`.
    pub fn simulate(f: *const Attn, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        var list: std.ArrayList([]const f32) = .empty;
        defer list.deinit(f.a);
        const n = attn.window + attn.max_selected;
        const buf = try f.a.alloc(f64, 4 * n);
        defer f.a.free(buf);
        const sc: f64 = attn.scale();
        for (0..f.rows) |r| {
            try f.keys(r, &list);
            const nk = list.items.len;
            const s = buf[0..nk];
            const p = buf[n..][0..nk];
            const terms = buf[2 * n ..][0..D];
            const t = buf[3 * n ..][0..nk];
            for (0..f.heads) |h| {
                const g = r * f.heads + h;
                const q = f.qf[g * D ..][0..D];
                for (list.items, s) |k, *v| {
                    for (terms, q, k) |*o, x, y| o.* = @as(f64, x) * y;
                    v.* = rows_.sumIn(acc, order, terms) * sc;
                }
                var top: f64 = f.sink[h];
                for (s) |v| top = if (std.math.isNan(v)) v else @max(top, v);
                for (s, p) |v, *o| o.* = std.math.exp(v - top);
                const l = rows_.sumIn(acc, order, p) + std.math.exp(@as(f64, f.sink[h]) - top);
                const x = out[g * D ..][0..D];
                for (x, 0..) |*o, d| {
                    for (list.items, p, t) |k, pj, *v| v.* = pj * k[d];
                    o.* = rows_.sumIn(acc, order, t) / l;
                }
                f.rope(r, x, terms);
            }
        }
    }
};
