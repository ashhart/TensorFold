//! The HIP message gates' host model (HIP-RMS-W32, HIP-ZETA-W32, HIP-GATE-UP-R64), cells and gate 4's simulation.
const std = @import("std");
const bf = @import("bf16_round");
const core = @import("core");
const K = @import("inkling2_loop_hip");
const ref = @import("contract_inkling2_loop_ref.zig");

const threads = core.contract.reference_threads;
const Accum = core.contract.spec.Accum;
const Rows = core.contract.rows.Order;
const max = K.max_hidden;
const Order = ref.Order;
const Orders = ref.Orders;
const same32 = ref.same32;
const sumIn = ref.sumIn;
const w32Sum = ref.w32Sum;
const inverse = ref.inverse;
const widen = ref.widen;
const sum = ref.sum;
const simInverse = ref.simInverse;

const R = K.rank;

/// A gate call's host inputs: h and m f32 [rows][H]; the controller's bf16 down [64][2H], forget and gain [H][64].
pub const GateCall = struct {
    h: []const f32,
    m: []const f32,
    down: []const u16,
    forget: []const u16,
    gain: []const u16,
    rows: usize,

    /// The call's hidden width H, from forget's [H][64].
    pub fn hidden(c: GateCall) usize {
        return c.forget.len / R;
    }
};

/// A gate's weights widened to f32: down [64][2H], h's half first; forget and gain [H][64].
pub const Weights = struct {
    down: []const f32,
    forget: []const f32,
    gain: []const f32,
};

/// One row's message gate: rho of h and m (HIP-RMS-W32), joined bf16, zeta (HIP-ZETA-W32), then HIP-GATE-UP-R64.
pub fn gateRow(o: Orders, h: []const f32, m: []const f32, w: Weights, out: []f32) void {
    const H = h.len;
    var wide: [2 * max]f32 = undefined;
    const joined = wide[0 .. 2 * H];
    for ([2][]const f32{ h, m }, 0..) |x, half| {
        const inv = inverse(sumIn(o.rho, x, x), H);
        for (joined[half * H ..][0..H], x) |*j, v| j.* = bf.round32(v * inv);
    }
    var zeta: [R]f32 = undefined;
    for (&zeta, 0..) |*z, r| z.* = bf.round32(sumIn(o.dot, w.down[r * 2 * H ..][0 .. 2 * H], joined));
    for (out, h, m, 0..) |*y, hv, mv, c| {
        var sf: f32 = 0;
        var sg: f32 = 0;
        for (w.forget[c * R ..][0..R], w.gain[c * R ..][0..R], zeta) |f, g, z| {
            sf = sf + f * z;
            sg = sg + g * z;
        }
        const keep = 1 - std.math.tanh(sf);
        const right = std.math.tanh(sg) * mv;
        y.* = keep * hv + right;
    }
}

const GateJob = struct {
    c: *const GateCall,
    w: Weights,
    lo: usize,
    hi: usize,
    out: []f32,

    pub fn run(j: *GateJob) void {
        const H = j.c.hidden();
        for (j.lo..j.hi) |t| gateRow(.{}, j.c.h[t * H ..][0..H], j.c.m[t * H ..][0..H], j.w, j.out[t * H ..][0..H]);
    }
};

/// Every row's gate under the baselines, [rows][H].
pub fn gate(a: std.mem.Allocator, c: GateCall, out: []f32) !void {
    const H = c.hidden();
    if (H == 0 or H > max or c.h.len != c.rows * H or c.m.len != c.rows * H) return error.BadShape;
    if (out.len != c.rows * H or c.down.len != R * 2 * H or c.forget.len != H * R or c.gain.len != H * R) return error.BadShape;
    const wide = try a.alloc(f32, 4 * R * H);
    defer a.free(wide);
    widen(wide[0 .. 2 * R * H], c.down);
    widen(wide[2 * R * H .. 3 * R * H], c.forget);
    widen(wide[3 * R * H ..], c.gain);
    const w: Weights = .{ .down = wide[0 .. 2 * R * H], .forget = wide[2 * R * H .. 3 * R * H], .gain = wide[3 * R * H ..] };
    var jobs: [threads.maximum]GateJob = undefined;
    const n = threads.count(threads.maximum, c.rows);
    for (jobs[0..n], 0..) |*j, i| j.* = .{ .c = &c, .w = w, .lo = c.rows * i / n, .hi = c.rows * (i + 1) / n, .out = out };
    threads.run(GateJob, jobs[0..n], .normal);
}

/// A gate's weights widened to float64, for gate 4's simulation.
pub const GateWeights = struct {
    down: []const f64,
    forget: []const f64,
    gain: []const f64,
};

/// One row's gate with every sum in `acc` and `order`, the gate's other steps in f32 (gate 4).
pub fn simGate(acc: Accum, order: Rows, h: []const f64, m: []const f64, w: GateWeights, out: []f64) void {
    const H = h.len;
    var wide: [2 * max]f64 = undefined;
    const terms = wide[0 .. 2 * H];
    var full: [2 * max]f32 = undefined;
    const joined = full[0 .. 2 * H];
    for ([2][]const f64{ h, m }, 0..) |x, half| {
        const inv = simInverse(acc, order, x, terms[0..H]);
        for (joined[half * H ..][0..H], x) |*j, v| j.* = bf.round32(@as(f32, @floatCast(v)) * inv);
    }
    var zeta: [R]f32 = undefined;
    for (&zeta, 0..) |*z, r| {
        for (terms, w.down[r * 2 * H ..][0 .. 2 * H], joined) |*t, d, j| t.* = @as(f32, @floatCast(d)) * j;
        z.* = bf.round32(sum(acc, order, terms));
    }
    for (out, h, m, 0..) |*o, hv, mv, i| {
        var fg: [2]f32 = undefined;
        for ([2][]const f64{ w.forget, w.gain }, &fg) |rows, *v| {
            for (terms[0..R], rows[i * R ..][0..R], zeta) |*t, x, z| t.* = @as(f32, @floatCast(x)) * z;
            v.* = std.math.tanh(sum(acc, order, terms[0..R]));
        }
        const keep: f32 = 1 - fg[0];
        o.* = keep * @as(f32, @floatCast(hv)) + fg[1] * @as(f32, @floatCast(mv));
    }
}

/// Gate 4's inputs: h, m of +-1 (+-3 at 256, past bf16's exact sums), three +-256 (fp16 overflows); 3 weights a row.
pub fn gateStress(r: std.Random, h: []f32, m: []f32, down: []u16, forget: []u16, gain: []u16) void {
    const H = forget.len / R;
    const small: f32 = if (H < 512) 3 else 1;
    for ([2][]f32{ h, m }) |x| for (x, 0..) |*v, i| {
        const big = i % H == 17 or i % H == 1000 % H or i % H == 2001 % H;
        v.* = (if (r.boolean()) @as(f32, 1) else -1) * (if (big) @as(f32, 256) else small);
    };
    const vals = [3]f32{ 2, -1, 1 };
    @memset(down, 0);
    for (0..down.len / (2 * H)) |row| for (vals, 0..) |v, j| {
        down[row * 2 * H + (row * 131 + j * 1531) % (2 * H)] = bf.bits(v);
    };
    for ([2][]u16{ forget, gain }, 0..) |x, which| {
        @memset(x, 0);
        for (0..x.len / R) |row| for (vals, 0..) |v, j| {
            x[row * R + (row * 7 + j * 23 + which * 5) % R] = bf.bits(v);
        };
    }
}

/// The gate cells' joined readouts: h's element `readout_h` (serial rho) and m's element `readout_m` (unfused squares).
pub const readout_h = 1;
pub const readout_m = 100;
/// The gate cells' ranks and the columns that read each one alone: the zeta dot cell, then the two rho readouts.
pub const cell_ranks = [3]usize{ 0, 4, 5 };
pub const cell_cols = [3]usize{ 0, 1, 2 };

/// The cells' weights: ranks 0-3 the zeta dot cell at m's 0, 8, 16, 24, ranks 4-5 the readouts, columns 0-2 reads.
pub fn gateCellWeights(down: []u16, forget: []u16, gain: []u16) void {
    const H = forget.len / R;
    for (0..4) |r| {
        const row = down[r * 2 * H ..][0 .. 2 * H];
        @memset(row, 0);
        for ([4]usize{ 0, 8, 16, 24 }, [4]f32{ 0x1p-24, 1, 0x1p-24, -1 }) |e, v| row[H + e] = bf.bits(v);
    }
    for ([2]usize{ 4, 5 }, [2]usize{ readout_h, H + readout_m }) |r, e| {
        @memset(down[r * 2 * H ..][0 .. 2 * H], 0);
        down[r * 2 * H + e] = bf.bits(1);
    }
    for (cell_cols, cell_ranks) |c, r| for ([2][]u16{ forget, gain }) |x| {
        @memset(x[c * R ..][0..R], 0);
        x[c * R + r] = bf.bits(1);
    };
}

/// bf16(x * inverse) under two sums of squares of the row as it stands, with x at element e.
fn readsApart(row: []f32, e: usize, x: f32, a: Order, b: Order) bool {
    row[e] = x;
    const ja = bf.bits(x * inverse(sumIn(a, row, row), row.len));
    const jb = bf.bits(x * inverse(sumIn(b, row, row), row.len));
    return ja != jb;
}

/// The gate's output at a column whose forget and gain read one zeta z alone, in gateRow's f32 steps.
fn cellOut(z: f32, hv: f32, mv: f32) f32 {
    const keep = 1 - std.math.tanh(z);
    const right = std.math.tanh(z) * mv;
    return keep * hv + right;
}

/// The column reading element e's joined value gives two outputs under the two orders' inverses, m's value mv there.
fn movesOut(row: []f32, e: usize, mv: f32, a: Order, b: Order) bool {
    const za = bf.round32(row[e] * inverse(sumIn(a, row, row), row.len));
    const zb = bf.round32(row[e] * inverse(sumIn(b, row, row), row.len));
    return @as(u32, @bitCast(cellOut(za, row[e], mv))) != @as(u32, @bitCast(cellOut(zb, row[e], mv)));
}

/// Sweeps x down from `start` by relative steps of 2^-16 until the two orders' joined readouts and outputs differ.
fn straddle(row: []f32, e: usize, start: f32, mv: f32, a: Order, b: Order) ?f32 {
    var x = start;
    for (0..1 << 16) |_| {
        if (readsApart(row, e, x, a, b) and movesOut(row, e, mv, a, b)) return x;
        x = x * (1 - 0x1p-16);
    }
    return null;
}

/// The f32 values within 32 ulps of `start`, nearest first, until the two orders' joined readouts differ.
fn straddleUlps(row: []f32, e: usize, start: f32, a: Order, b: Order) ?f32 {
    var up = start;
    var down = start;
    for (0..32) |_| {
        if (readsApart(row, e, up, a, b)) return up;
        if (readsApart(row, e, down, a, b)) return down;
        up = std.math.nextAfter(f32, up, std.math.inf(f32));
        down = std.math.nextAfter(f32, down, -std.math.inf(f32));
    }
    return null;
}

/// The gate cell row: h of 2^12 and ones, m near 1, each readout straddling a bf16 midpoint under the two inverses.
pub fn gateCellRow(h: []f32, m: []f32) !void {
    try cellM(m);
    @memset(h, 1);
    h[0] = 4096;
    h[readout_h] = straddle(h, readout_h, 1, m[readout_h], .w32, .serial) orelse return error.NoCell;
}

/// The cell row's m: near 1, its readout straddling a bf16 midpoint under fmaf and rounded squares, m's dot cell at 1.
fn cellM(m: []f32) !void {
    var prng = std.Random.DefaultPrng.init(0x756e66);
    const r = prng.random();
    const mid: f32 = 1 + 0x1p-8;
    for (0..256) |_| {
        for (m) |*v| v.* = 1 + (r.float(f32) * 2 - 1) * 0x1p-12;
        m[readout_m] = mid;
        const inv = inverse(w32Sum(m, m), m.len);
        if (inv == inverse(sumIn(.w32_unfused, m, m), m.len)) continue;
        m[readout_m] = straddleUlps(m, readout_m, mid / inv, .w32, .w32_unfused) orelse continue;
        const at = inverse(w32Sum(m, m), m.len);
        var ones = true;
        for ([4]usize{ 0, 8, 16, 24 }) |e| ones = ones and bf.bits(m[e] * at) == bf.bits(1);
        if (ones) return;
    }
    return error.NoCell;
}

fn randomGate(a: std.mem.Allocator, r: std.Random, H: usize) ![3][]u16 {
    var out: [3][]u16 = undefined;
    out[0] = try a.alloc(u16, R * 2 * H);
    out[1] = try a.alloc(u16, H * R);
    out[2] = try a.alloc(u16, H * R);
    for (out) |x| for (x) |*v| {
        v.* = bf.bits(r.floatNorm(f32) * 0.05);
    };
    return out;
}

fn wideGate(a: std.mem.Allocator, down: []const u16, forget: []const u16, gain: []const u16) !struct { []f64, GateWeights } {
    const H = forget.len / R;
    const w = try a.alloc(f64, 4 * R * H);
    for (w[0 .. 2 * R * H], down) |*d, x| d.* = bf.value(x);
    for (w[2 * R * H .. 3 * R * H], forget) |*d, x| d.* = bf.value(x);
    for (w[3 * R * H ..], gain) |*d, x| d.* = bf.value(x);
    return .{ w, .{ .down = w[0 .. 2 * R * H], .forget = w[2 * R * H .. 3 * R * H], .gain = w[3 * R * H ..] } };
}

fn wideRow(dst: []f64, src: []const f32) void {
    for (dst, src) |*d, s| d.* = s;
}

test "the gate's stress rows: fp32 sums are exact in every order, so gate 4's simulation is the W32 gate; fp16 and bf16 miss by 100x" {
    inline for (ref.geometries) |geo| try stressRows(geo.hidden());
}

fn stressRows(comptime H: usize) !void {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5354);
    const r = prng.random();
    const ws = try randomGate(a, r, H);
    defer for (ws) |x| a.free(x);
    var h: [H]f32 = undefined;
    var m: [H]f32 = undefined;
    gateStress(r, &h, &m, ws[0], ws[1], ws[2]);
    var want: [H]f32 = undefined;
    try gate(a, .{ .h = &h, .m = &m, .down = ws[0], .forget = ws[1], .gain = ws[2], .rows = 1 }, &want);
    const w, const gw = try wideGate(a, ws[0], ws[1], ws[2]);
    defer a.free(w);
    var hx: [H]f64 = undefined;
    var mx: [H]f64 = undefined;
    wideRow(&hx, &h);
    wideRow(&mx, &m);
    var sim: [H]f64 = undefined;
    const spec: core.contract.spec.Spec = .{ .name = "", .accum = .f32, .depth = undefined, .out = .f32, .served = &.{} };
    for ([_]Rows{ .sequential, .pairwise, .strided }) |o| {
        simGate(.f32, o, &hx, &mx, gw, &sim);
        for (want, sim) |x, y| try std.testing.expect(same32(x, @floatCast(y)));
        for ([_]Accum{ .f16, .bf16 }) |acc| {
            simGate(acc, o, &hx, &mx, gw, &sim);
            var worst: f64 = 0;
            for (want, sim) |x, y| worst = @max(worst, @abs(y - x) / core.contract.spec.boundExact(spec, x));
            try std.testing.expect(worst >= 100);
        }
    }
}

test "the gate's zeta dot cell: W32 gives 2^-24, one serial chain 0, the reversed butterfly 2^-23" {
    inline for (ref.geometries) |geo| try zetaCell(geo.hidden());
}

fn zetaCell(comptime H: usize) !void {
    var down: [R * 2 * H]u16 = undefined;
    var forget: [H * R]u16 = @splat(0);
    var gain: [H * R]u16 = @splat(0);
    gateCellWeights(&down, &forget, &gain);
    var w: [2 * H]f32 = undefined;
    widen(&w, down[0 .. 2 * H]);
    var joined: [2 * H]f32 = @splat(0.5);
    for ([4]usize{ 0, 8, 16, 24 }) |e| joined[H + e] = 1;
    try std.testing.expectEqual(@as(f32, 0x1p-24), sumIn(.w32, &w, &joined));
    try std.testing.expectEqual(@as(f32, 0), sumIn(.serial, &w, &joined));
    try std.testing.expectEqual(@as(f32, 0x1p-23), sumIn(.w32_desc, &w, &joined));
}

test "the gate cell row: its readouts' joined bf16 differ under serial rho and unfused squares, and m's dot cell stays 1" {
    inline for (ref.geometries) |geo| try cellRow(geo.hidden());
}

fn cellRow(comptime H: usize) !void {
    var h: [H]f32 = undefined;
    var m: [H]f32 = undefined;
    try gateCellRow(&h, &m);
    try std.testing.expect(sumIn(.w32, &h, &h) != sumIn(.serial, &h, &h));
    try std.testing.expect(readsApart(&h, readout_h, h[readout_h], .w32, .serial));
    try std.testing.expect(readsApart(&m, readout_m, m[readout_m], .w32, .w32_unfused));
    const inv = inverse(w32Sum(&m, &m), H);
    for ([4]usize{ 0, 8, 16, 24 }) |e| try std.testing.expectEqual(bf.bits(1), bf.bits(m[e] * inv));
}

test "the gate cells move every planted order's outputs at the columns that read them, and leave the served gate's alone" {
    inline for (ref.geometries) |geo| try cellsMove(geo.hidden());
}

fn cellsMove(comptime H: usize) !void {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x63656c);
    const r = prng.random();
    const ws = try randomGate(a, r, H);
    defer for (ws) |x| a.free(x);
    gateCellWeights(ws[0], ws[1], ws[2]);
    var h: [H]f32 = undefined;
    var m: [H]f32 = undefined;
    try gateCellRow(&h, &m);
    const wide = try a.alloc(f32, 4 * R * H);
    defer a.free(wide);
    widen(wide[0 .. 2 * R * H], ws[0]);
    widen(wide[2 * R * H .. 3 * R * H], ws[1]);
    widen(wide[3 * R * H ..], ws[2]);
    const w: Weights = .{ .down = wide[0 .. 2 * R * H], .forget = wide[2 * R * H .. 3 * R * H], .gain = wide[3 * R * H ..] };
    var served: [H]f32 = undefined;
    gateRow(.{}, &h, &m, w, &served);
    const planted = [3]Orders{ .{ .rho = .w32_desc, .dot = .w32_desc }, .{ .rho = .serial }, .{ .rho = .w32_unfused } };
    for (planted, cell_cols) |o, col| {
        var got: [H]f32 = undefined;
        gateRow(o, &h, &m, w, &got);
        try std.testing.expect(!same32(served[col], got[col]));
    }
}

test "a gate row through gate() equals gateRow with the baselines' orders, row by row" {
    const H = comptime ref.geometries[0].hidden();
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x47415445);
    const r = prng.random();
    const ws = try randomGate(a, r, H);
    defer for (ws) |x| a.free(x);
    var h: [2 * H]f32 = undefined;
    var m: [2 * H]f32 = undefined;
    for (&h, &m) |*x, *y| {
        x.* = r.floatNorm(f32) * 4;
        y.* = r.floatNorm(f32);
    }
    var out: [2 * H]f32 = undefined;
    try gate(a, .{ .h = &h, .m = &m, .down = ws[0], .forget = ws[1], .gain = ws[2], .rows = 2 }, &out);
    const wide = try a.alloc(f32, 4 * R * H);
    defer a.free(wide);
    widen(wide[0 .. 2 * R * H], ws[0]);
    widen(wide[2 * R * H .. 3 * R * H], ws[1]);
    widen(wide[3 * R * H ..], ws[2]);
    var one: [H]f32 = undefined;
    gateRow(.{}, h[H..], m[H..], .{ .down = wide[0 .. 2 * R * H], .forget = wide[2 * R * H .. 3 * R * H], .gain = wide[3 * R * H ..] }, &one);
    for (out[H..], one) |x, y| try std.testing.expect(same32(x, y));
}
