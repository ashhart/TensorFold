//! Nemotron's tf_attention_nax on the M5: attention_range at its arithmetic, fp32 P cut as the tensor units take it.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const old = @import("metal_nemotron_attention.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const D = old.D;

/// P with the tensor units' cut; P spread over [0, 1]; that case without the cut; a non-finite value at a masked key.
const Case = enum { cut, spread, cutless, masked };

fn name(k: Case, full: bool) []const u8 {
    if (full) return switch (k) {
        .cut => "metal nemotron attention prompt nax, full fp32 P",
        .spread => "metal nemotron attention prompt nax, full fp32 P spread over [0, 1]",
        .cutless => "control: metal nemotron attention prompt nax, full fp32 P range with tensor-cut product",
        .masked => "metal nemotron attention prompt nax, full fp32 P with a non-finite value at a masked key",
    };
    return switch (k) {
        .cut => "metal nemotron attention prompt nax",
        .spread => "metal nemotron attention prompt nax, P spread over [0, 1]",
        .cutless => "control: metal nemotron attention prompt nax, P spread, its range without the operand cut",
        .masked => "metal nemotron attention prompt nax, a non-finite value at a key masked in its tile",
    };
}

/// tf_attention_nax: 8 mma steps a score, fp32 SCALE * log2e, fast exp2; a 32-key block: 2 P x V steps, rescale, folds.
fn form(n: usize, cut: f64, full: bool) ar.Form {
    const blocks: f64 = @floatFromInt(n / 32 + 2);
    const product: f64 = if (full) 32 else 2 * ar.mma_step;
    return .{ .score = 9 * ar.mma_step, .fast_exp = true, .chunk_exp = false, .p_rel = 0, .p_cut = cut, .num = (product + 1) * blocks + ar.mma_step + 2, .den = 3 * blocks + 6, .score_round = 4, .div = 2 };
}

const Probe = struct {
    inner: old.Probe,
    case: Case,
    full: bool,
    rig: *@import("rig.zig").Rig,
    point: []f64 = &.{},
    half: []f64 = &.{},
    down: []f64 = &.{},
    up: []f64 = &.{},

    fn deinit(p: *Probe) void {
        for ([_][]f64{ p.point, p.half, p.down, p.up }) |x| p.inner.a.free(x);
        p.inner.deinit();
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        if ((p.case == .spread or p.case == .cutless) and fill == .random) p.spread(seed);
        if (p.case == .masked and fill == .nonfinite) p.maskedInf();
        return true;
    }

    /// +Inf in V at key len + 5 (head 0, column 7): rows 0 to 4 have it masked, yet their tile reads it.
    fn maskedInf(p: *Probe) void {
        const in = &p.inner;
        in.host[in.at(true, 0, in.len + 5) + 7] = 0x7f80;
        const cache = old.kvh * in.cap * D;
        @memcpy(p.rig.bytes(in.kv[1], cache * 2), std.mem.sliceAsBytes(in.host[cache..][0..cache]));
    }

    /// Scores q . k = q0 k0 alone, so P = 2^(s - max) spreads over about [2^-6, 1]; values in [0, 1) add the bias up.
    fn spread(p: *Probe, seed: u64) void {
        const in = &p.inner;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x1546);
        const rnd = prng.random();
        for (0..in.rows * old.heads) |r| {
            @memset(in.q[r * D ..][0..D], 0);
            in.q[r * D] = c.toBf16(0.5 + rnd.float(f32));
        }
        for (0..old.kvh) |h| for (0..in.cap) |j| {
            in.host[in.at(false, h, j)] = c.toBf16(-32 * rnd.float(f32));
            for (0..D) |d| in.host[in.at(true, h, j) + d] = c.toBf16(rnd.float(f32));
        };
        // the inputs as bound: the caches' two halves and the queries
        const cache = old.kvh * in.cap * D;
        for (in.kv, 0..) |r, i| @memcpy(p.rig.bytes(r, cache * 2), std.mem.sliceAsBytes(in.host[i * cache ..][0..cache]));
        @memcpy(p.rig.bytes(in.qref, in.q.len * 2), std.mem.sliceAsBytes(in.q));
    }

    pub fn run(p: *Probe) !void {
        try p.inner.run();
    }

    pub fn count(p: *Probe) usize {
        return p.inner.count();
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.inner.simulate(acc, order, out);
    }

    /// Every head's float64 point, half width and the cut's sides over the keys up to its row.
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const a = p.inner.a;
        const in = &p.inner;
        for ([_]*[]f64{ &p.point, &p.half, &p.down, &p.up }) |x| {
            a.free(x.*);
            x.* = try a.alloc(f64, want.len);
        }
        const most = in.len + in.rows;
        const buf = try a.alloc(f64, 2 * most * D + 2 * most);
        defer a.free(buf);
        const kf = buf[0 .. most * D];
        const vf = buf[most * D .. 2 * most * D];
        const w = buf[2 * most * D ..][0..most];
        const tau = buf[2 * most * D + most ..][0..most];
        const cut: f64 = if (p.full or p.case == .cutless) 0 else ar.operand_cut;
        var q: [D]f64 = undefined;
        for (0..old.kvh) |hk| {
            for (0..most * D) |i| {
                kf[i] = c.fromBf16(in.host[in.at(false, hk, 0) + i]);
                vf[i] = c.fromBf16(in.host[in.at(true, hk, 0) + i]);
            }
            for (0..in.rows) |qi| for (hk * old.G..(hk + 1) * old.G) |h| {
                const n = in.len + qi + 1;
                const f = form(n, cut, p.full);
                in.query(qi, h, &q);
                const o = in.outAt(qi, h);
                ar.head(&q, kf, vf, n, D, old.scale, f, w, tau, want[o..][0..D], p.half[o..][0..D]);
                ar.cutSides(f, w, tau, vf, n, D, p.down[o..][0..D], p.up[o..][0..D]);
            };
        }
        @memcpy(mag, p.half);
        @memcpy(p.point, want);
    }

    /// The range's allowance (attention_range.allowBf16); gate 4's exact sums keep the exact bound.
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        const s = old.spec(.prompt, 1, false);
        for (want, allow, p.half, p.down, p.up) |w, *l, h, d, u| l.* = if (p.inner.stress) c.boundExact(s, w) else ar.allowBf16(w, h, d, u);
    }

    /// Each bf16 output within its range's rounded ends, the cut on its own side; gate 4's stress fill is open.
    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        for (low, high, p.point, p.half, p.down, p.up) |*l, *hi, pt, h, d, u| {
            const e = if (p.inner.stress) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else ar.intervalBf16(pt, h, d, u);
            l.* = e[0];
            hi.* = e[1];
        }
    }
};

const vt = g.hooks(Probe);

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    /// Only on the tensor units, where the cut was measured; before the M5, attention_ref's bound stays.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        const full = try tf.frags.precision.requested();
        for (std.enums.values(Case)) |k| {
            if (only) |o| if (std.mem.indexOf(u8, name(k, full), o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = .{ .inner = try old.Probe.init(s.a, s.device, s.queue, .prompt, 1, false), .case = k, .full = full, .rig = undefined };
            p.inner.prompt_full = full and k != .cutless;
            p.rig = &p.inner.rig;
            try s.probes.append(s.a, p);
            var spec = old.spec(.prompt, 1, false);
            spec.name = name(k, full);
            if (k == .masked) {
                spec.served = spec.served[0..1];
                spec.known = .{ .gate = .nonfinite, .finding = "a non-finite value at a key masked for a row of its tile reaches that row as NaN (P = 0 times Inf inside the P x V mma)" };
            }
            try s.kernels.append(s.a, .{ .spec = spec, .ctx = p, .vt = &vt, .control = if (k == .cutless) .shapes else null });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
