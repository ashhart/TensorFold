//! ERNIE 5.5's HCA compressor (ernie5_5/hca.cu) under the contract: each compressed row against float64 (hca.zig).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ernie = @import("ernie5_5_hca_cuda");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const bfr = core.bf16_round;
const hh = ernie.hca;

const hd = hh.head_dim;
const ts = 2 * hd;
const eps: f32 = 1e-6;
/// The released config's compress_rope_theta and rope_scaling (YaRN, factor 16 over 65,536 positions).
const theta = 160000;
const factor = 16;
const original = 65536;

/// A planted fault (control entries): the pool's softmax without ape, or the rope at the block's last position.
pub const Fault = enum { none, no_ape, last_position };

pub const Entry = struct { name: []const u8, symbol: [:0]const u8, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda ernie5_5 hca tf_ernie55_hca_compress (a block's softmax pool, RMSNorm, GPT-J RoPE at its start)", .symbol = "tf_ernie55_hca_compress" },
    .{ .name = "control: ernie5_5 hca compress pooling without ape", .symbol = "tf_ctl_ernie55_hca_compress_no_ape", .fault = .no_ape },
    .{ .name = "control: ernie5_5 hca compress turned at the block's last position", .symbol = "tf_ctl_ernie55_hca_compress_last_position", .fault = .last_position },
};

/// rows, the first row's block (at 128 n), the pool's length: near the start, block 0 (no turn), a chunk, the 1M end.
pub const served = &[_]c.Shape{ .{ .rows = 4, .n = 3, .k = hh.ratio }, .{ .rows = 1, .n = 0, .k = hh.ratio }, .{ .rows = 64, .n = 100, .k = hh.ratio }, .{ .rows = 2, .n = (1 << 20) / hh.ratio - 2, .k = hh.ratio } };

/// The pool's 128 terms in order; the norm's squares are shorter (2, the butterfly's 5, the 8 warps' 7).
fn depth(_: c.Shape) usize {
    return hh.ratio;
}

pub fn spec(e: Entry) c.Spec {
    return .{ .name = e.name, .accum = .f32, .depth = depth, .out = .bf16, .served = if (e.fault == .none) served else served[0..1] };
}

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

pub const Compress = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    st: []f32 = &.{}, // token rows: kv [512] then score [512]
    ape: []f32 = &.{}, // [128][512]
    w: [hd]f64 = undefined, // the norm's weight, bf16 values
    freq: [hh.rope / 2]f32 = undefined,
    kv: []f64 = &.{}, // a row's kv and score as f64
    score: []f64 = &.{},
    extra: []f64 = &.{}, // per output: the allowance for its own roundings beyond the spec's bound
    dev: [5]u64 = @splat(0), // st, ape, w, freq, pos
    out: u64 = 0,

    const vt = g.hooks(Compress);

    pub fn kernel(self: *Compress) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault == .none) null else .shapes };
    }

    fn free(self: *Compress) void {
        const a = self.rig.gpu.gpa;
        a.free(self.st);
        a.free(self.ape);
        a.free(self.kv);
        a.free(self.score);
        a.free(self.extra);
        self.st = &.{};
        self.ape = &.{};
        self.kv = &.{};
        self.score = &.{};
        self.extra = &.{};
    }

    pub fn deinit(self: *Compress) void {
        self.free();
        self.rig.deinit();
    }

    /// kv in [-2, 2) (a third offset by 8), scores [-8, 8), ape [-1, 1); stress: e = 1, kv halves cancel to 301-1023.
    pub fn prepare(self: *Compress, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != hh.ratio) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var w16: [hd]u16 = undefined;
        for (&w16, &self.w) |*v, *fw| {
            v.* = bfr.bits(0.5 + rng.float(f32));
            fw.* = bf(v.*);
        }
        const toks = shape.rows * hh.ratio;
        self.st = try a.alloc(f32, toks * ts);
        self.ape = try a.alloc(f32, hh.ratio * hd);
        self.kv = try a.alloc(f64, hh.ratio * hd);
        self.score = try a.alloc(f64, hh.ratio * hd);
        if (fill == .stress) {
            for (self.ape) |*v| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 513))) - 256)) / 256;
            for (0..shape.rows) |r| for (0..hd) |ch| {
                const rest: f32 = @floatFromInt(301 + rng.uintLessThan(u32, 723));
                for (0..hh.ratio / 2) |i| {
                    const v: f32 = @floatFromInt(32769 + 2 * rng.uintLessThan(u32, 16368));
                    const sgn: f32 = if (ch % 2 == 0) 1 else -1;
                    self.st[(r * hh.ratio + i) * ts + ch] = sgn * v;
                    self.st[(r * hh.ratio + i + hh.ratio / 2) * ts + ch] = -sgn * v;
                }
                self.st[(r * hh.ratio + hh.ratio - 1) * ts + ch] += rest;
                for (0..hh.ratio) |i| self.st[(r * hh.ratio + i) * ts + hd + ch] = 4 - self.ape[i * hd + ch];
            };
        } else {
            for (self.ape) |*v| v.* = rng.float(f32) * 2 - 1;
            for (0..toks) |k| {
                const off: f32 = if ((k / hh.ratio) % 3 == 1) 8 else 0;
                for (self.st[k * ts ..][0..hd]) |*v| v.* = off + rng.float(f32) * 4 - 2;
                for (self.st[k * ts + hd ..][0..hd]) |*v| v.* = rng.float(f32) * 16 - 8;
            }
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, usize, f32 }{ .{ 5, 42, std.math.nan(f32) }, .{ hh.ratio + 77, hd + 300, std.math.inf(f32) }, .{ 3 * hh.ratio - 1, hd - 1, -std.math.inf(f32) } };
            for (planted) |p| if (p[0] < toks) {
                self.st[p[0] * ts + p[1]] = p[2];
            };
        }
        self.freq = core.rope_yarn.yarn(theta, factor, original).freq;
        const pos0 = [1]i32{@intCast(shape.n * hh.ratio)};
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.st)), try self.rig.input(std.mem.sliceAsBytes(self.ape)), try self.rig.input(std.mem.sliceAsBytes(&w16)), try self.rig.input(std.mem.sliceAsBytes(&self.freq)), try self.rig.input(std.mem.sliceAsBytes(&pos0)) };
        self.out = try self.rig.output(shape.rows * hd * 2);
        return true;
    }

    pub fn run(self: *Compress) !void {
        try ernie.launch(self.f, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.dev[4], self.out, .{ .rows = self.shape.rows }, eps);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Compress) usize {
        return self.shape.rows * hd;
    }

    /// Row r's kv and score as f64, and its position.
    fn block(self: *Compress, r: usize) i64 {
        for (0..hh.ratio) |i| for (0..hd) |ch| {
            self.kv[i * hd + ch] = self.st[(r * hh.ratio + i) * ts + ch];
            self.score[i * hd + ch] = self.st[(r * hh.ratio + i) * ts + hd + ch];
        };
        return @intCast((self.shape.n + r) * hh.ratio);
    }

    /// hca.compress in float64; mag: the pool's and the squares' sums through the norm, turned; extra: own roundings.
    pub fn reference(self: *Compress, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.extra);
        self.extra = try a.alloc(f64, want.len);
        const u = 0x1p-24;
        var t: [2 * hh.ratio]f64 = undefined;
        for (0..self.shape.rows) |r| {
            const pos = self.block(r);
            const o = want[r * hd ..][0..hd];
            hh.compress(self.kv, self.score, self.ape, &self.w, &self.freq, pos, eps, hh.exact, &t, o);
            var p: [hd]f64 = undefined;
            var mp: [hd]f64 = undefined; // the pool's sums' error, in gamma units
            var ep: [hd]f64 = undefined; // its own: score + ape, the exponent's difference, expf's 2 ulp, the division
            for (&p, &mp, &ep, 0..) |*pv, *mv, *ev, ch| {
                var top = -std.math.inf(f64);
                for (0..hh.ratio) |i| top = @max(top, self.score[i * hd + ch] + self.ape[i * hd + ch]);
                var sum: f64 = 0;
                var num: f64 = 0;
                var abs: f64 = 0;
                for (0..hh.ratio) |i| {
                    const s = self.score[i * hd + ch] + self.ape[i * hd + ch];
                    const e = @exp(s - top);
                    sum += e;
                    num += self.kv[i * hd + ch] * e;
                    abs += @abs(self.kv[i * hd + ch]) * e;
                }
                pv.* = num / sum;
                mv.* = abs / sum + @abs(pv.*);
                var own: f64 = 0;
                for (0..hh.ratio) |i| {
                    const s = self.score[i * hd + ch] + self.ape[i * hd + ch];
                    own += @abs(self.kv[i * hd + ch] - pv.*) * @exp(s - top) * u * (@abs(s) + @abs(s - top) + 4);
                }
                ev.* = own / sum + u * @abs(pv.*);
            }
            var p2: f64 = 0;
            var pm: f64 = 0;
            var pe: f64 = 0;
            for (p, mp, ep) |pv, mv, ev| {
                p2 += pv * pv;
                pm += @abs(pv) * mv;
                pe += @abs(pv) * ev;
            }
            const den = p2 + hd * eps;
            const rr = 1 / @sqrt(p2 / hd + eps);
            // r's relative error: from the pools (through the squares) and the squares' own sums; then its 3 roundings
            const rm = (pm + p2 / 2) / den;
            const re = pe / den + 3 * u;
            var my: [hd]f64 = undefined;
            var ey: [hd]f64 = undefined;
            var ay: [hd]f64 = undefined;
            for (&my, &ey, &ay, p, mp, ep, self.w) |*mo, *eo, *yo, pv, mv, ev, gw| {
                yo.* = @abs(pv * rr * gw);
                mo.* = rr * gw * mv + yo.* * rm;
                eo.* = rr * gw * ev + yo.* * (re + 2 * u);
            }
            const m = mag[r * hd ..][0..hd];
            const e = self.extra[r * hd ..][0..hd];
            @memcpy(m, &my);
            @memcpy(e, &ey);
            for (self.freq, 0..) |f, l| {
                const ang = hh.angle(f, pos);
                const co = @abs(@cos(ang));
                const si = @abs(@sin(ang));
                const c0 = hh.nope + 2 * l;
                for ([2][2]usize{ .{ c0, c0 + 1 }, .{ c0 + 1, c0 } }) |q| {
                    m[q[0]] = co * my[q[0]] + si * my[q[1]];
                    // the turned errors, then sincosf's 2 ulp, the two products and the sum: 8 units of the terms
                    e[q[0]] = co * ey[q[0]] + si * ey[q[1]] + 8 * u * (co * ay[q[0]] + si * ay[q[1]]);
                }
            }
        }
    }

    pub fn bounds(self: *Compress, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = want;
        _ = mag;
        for (allow, self.extra) |*l, x| l.* += x * (1 + 0x1p-7);
    }

    /// hca.compress with the pool's and the squares' sums in `acc`, in `order`.
    pub fn simulate(self: *Compress, acc: c.Accum, order: g.Order, out: []f64) !void {
        const sim = struct {
            acc: c.Accum,
            order: g.Order,
            pub fn of(s: @This(), t: []const f64) f64 {
                return rows_.sumIn(s.acc, s.order, t);
            }
        }{ .acc = acc, .order = order };
        var t: [2 * hh.ratio]f64 = undefined;
        for (0..self.shape.rows) |r| {
            const pos = self.block(r);
            hh.compress(self.kv, self.score, self.ape, &self.w, &self.freq, pos, eps, sim, &t, out[r * hd ..][0..hd]);
        }
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    hca: ernie.Hca,
    items: [entries.len]Compress = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var m = try ernie.Hca.load(gpu.d);
        errdefer m.unload();
        set.* = .{ .gpa = gpu.gpa, .hca = m };
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = try set.hca.mod.function(e.symbol) };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.hca.unload();
        set.gpa.destroy(set);
    }
};

/// The HCA image's gates 1-4 and its controls alone.
pub fn run(gpu: check_.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x686361, &t);
    std.debug.print("SUMMARY ernie5_5 hca: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check_.expect(t.problems() == 0, "ernie5_5-hca contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}

test "the served shapes fit the kernel, and the first turns its rows (the last-position control's)" {
    for (served) |x| try hh.check(.{ .rows = x.rows });
    try std.testing.expect(served[0].n > 0);
    try std.testing.expect((served[3].n + served[3].rows) * hh.ratio <= 1 << 20);
}

test "the HCA rope's YaRN: pairs 0-15 at theta's own frequencies, 25-31 at a sixteenth, a ramp between" {
    const fr = core.rope_yarn.yarn(theta, factor, original).freq;
    for (fr, 0..) |f, l| {
        const own = std.math.pow(f64, theta, -@as(f64, @floatFromInt(2 * l)) / hh.rope);
        const want: ?f64 = if (l <= 15) own else if (l >= 25) own / factor else null;
        if (want) |v| try std.testing.expectApproxEqRel(v, f, 1e-6);
    }
    try std.testing.expect(fr[20] < std.math.pow(f64, theta, -40.0 / hh.rope) and fr[20] > std.math.pow(f64, theta, -40.0 / hh.rope) / factor);
}
