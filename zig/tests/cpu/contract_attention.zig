//! The CPU attention under the contract: one query row's softmax(q·k / 16)·v over a kv head, and four planted controls.
const std = @import("std");
const contract = @import("contract");
const mx = @import("cpu_mixers");
const rig_ = @import("contract_rig.zig");

const k = mx.ops.kernels;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;

pub const Which = enum { attention, finite, writes, last, half };

const head_dim = 256;
/// Each row's keys: row r of `rows` attends to positions 0 to k - rows + r, so the last row sees every key.
fn lastOf(shape: c.Shape, r: usize) usize {
    return shape.k - shape.rows + r;
}

/// The dot over 256 in one fma chain a lane plus the lanes' tree, then the keys' sums and the merges of their blocks.
fn depth(s: c.Shape) usize {
    return head_dim + s.k + 8;
}

/// A decode row, a verify window and a prompt chunk over short and long caches; the contract's edges add k 1 to 129.
const served = [_]c.Shape{ .{ .rows = 1, .n = head_dim, .k = 128 }, .{ .rows = 1, .n = head_dim, .k = 4096 }, .{ .rows = 16, .n = head_dim, .k = 4096 }, .{ .rows = 128, .n = head_dim, .k = 1024 } };
const ctl = [_]c.Shape{ .{ .rows = 5, .n = head_dim, .k = 300 }, .{ .rows = 2, .n = head_dim, .k = 2048 } };

const sums_why = "the stress scores are exact integers but the softmax weights are not, so their sums keep an fp32 term in the bound and a pairwise fp16 sum of them stays within a few bf16 ulps";

pub fn spec(w: Which) c.Spec {
    const name: []const u8 = switch (w) {
        .attention => "cpu attention",
        .finite => "control: attention taking a non-finite input as zero",
        .writes => "control: attention writing its query",
        .last => "control: attention leaving out the row's own key",
        .half => "control: attention summing its weights and values in half precision",
    };
    return .{ .name = name, .accum = .f32, .depth = depth, .ops = 4, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = mx.chunk_keys, .step = 1, .min = 1 } }, .served = if (w == .attention) &served else &ctl, .sums = .{ .f16 = 0, .bf16 = 0, .why = sums_why } };
}

/// The gate each control must fail.
pub fn control(w: Which) ?g.Gate {
    return switch (w) {
        .attention => null,
        .finite => .nonfinite,
        .writes => .repeat,
        .last => .shapes,
        .half => .sums,
    };
}

pub const Attn = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    shape: c.Shape = undefined,
    q: []u16 = &.{},
    keys: []u16 = &.{},
    values: []u16 = &.{},
    clean: []u16 = &.{},
    partials: []mx.Partial = &.{},
    out: []u16 = &.{},
    /// Per row, the largest sum of |q_i k_ji| / 16 over its keys: the scores' magnitude the dot's error scales with.
    score_mag: []f64 = &.{},
    /// Gate 4's integer inputs: the dots are exact in fp32, so only the weights' sums carry error.
    stress: bool = false,

    pub fn init(a: std.mem.Allocator, which: Which) Attn {
        return .{ .a = a, .rig = .{ .a = a }, .which = which };
    }

    pub fn deinit(t: *Attn) void {
        t.drop();
        t.rig.deinit();
    }

    fn drop(t: *Attn) void {
        if (t.score_mag.len > 0) t.a.free(t.score_mag);
        t.score_mag = &.{};
        t.rig.reset();
    }

    pub fn kernel(t: *Attn) g.Kernel {
        return .{ .spec = spec(t.which), .ctx = t, .vt = &vt, .control = control(t.which) };
    }

    const vt = g.hooks(Attn);

    fn takes(shape: c.Shape) bool {
        return shape.rows > 0 and shape.rows <= 128 and shape.n == head_dim and shape.k >= shape.rows;
    }

    const nan = k.bf16of(std.math.nan(f32));
    const inf = k.bf16of(std.math.inf(f32));
    const ninf = k.bf16of(-std.math.inf(f32));

    pub fn prepare(t: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.drop();
        if (!takes(shape)) return false;
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const q = try t.a.alloc(u16, shape.rows * head_dim);
        defer t.a.free(q);
        const keys = try t.a.alloc(u16, shape.k * head_dim);
        defer t.a.free(keys);
        const values = try t.a.alloc(u16, shape.k * head_dim);
        defer t.a.free(values);
        t.stress = fill == .stress;
        if (fill == .stress) {
            for (q) |*v| v.* = k.bf16of(@floatFromInt(rnd.intRangeAtMost(i32, -2, 2)));
            for (keys) |*v| v.* = k.bf16of(@floatFromInt(rnd.intRangeAtMost(i32, -1, 1)));
            for (values) |*v| v.* = k.bf16of(@floatFromInt(rnd.intRangeAtMost(i32, -8, 8)));
        } else {
            for (q) |*v| v.* = k.bf16of(rnd.floatNorm(f32) * 2);
            for (keys) |*v| v.* = k.bf16of(rnd.floatNorm(f32));
            for (values) |*v| v.* = k.bf16of(rnd.floatNorm(f32));
        }
        if (fill == .nonfinite) {
            q[3] = nan;
            if (shape.rows > 1) keys[(lastOf(shape, 1) / 2) * head_dim + 9] = inf;
            if (shape.rows > 2) values[(lastOf(shape, 2) / 3) * head_dim + 5] = nan;
            if (shape.rows > 3) keys[(lastOf(shape, 3) / 4) * head_dim + 1] = ninf;
        }
        t.q = try t.rig.input(u16, q);
        t.keys = try t.rig.input(u16, keys);
        t.values = try t.rig.input(u16, values);
        t.clean = try t.rig.scratch(u16, (shape.rows + 2 * shape.k) * head_dim);
        t.partials = try t.rig.scratch(mx.Partial, mx.chunksFor(shape.k));
        t.out = try t.rig.output(u16, shape.rows * head_dim);
        t.score_mag = try t.a.alloc(f64, shape.rows);
        return true;
    }

    fn sanitized(t: *Attn, at: usize, v: []const u16) []const u16 {
        const clean = t.clean[at..][0..v.len];
        for (clean, v) |*o, b| o.* = if (std.math.isFinite(k.f32of(b))) b else 0;
        return clean;
    }

    pub fn run(t: *Attn) !void {
        const s = t.shape;
        for (0..s.rows) |r| {
            const q = t.q[r * head_dim ..][0..head_dim];
            const out = t.out[r * head_dim ..][0..head_dim];
            switch (t.which) {
                .attention => mx.attention(q, t.keys, t.values, s.k, 0, lastOf(s, r), t.partials, out),
                .writes => {
                    mx.attention(q, t.keys, t.values, s.k, 0, lastOf(s, r), t.partials, out);
                    if (r == 0) t.q[0] ^= 1;
                },
                .finite => mx.attention(t.sanitized(0, q), t.sanitized(head_dim, t.keys), t.sanitized((1 + s.k) * head_dim, t.values), s.k, 0, lastOf(s, r), t.partials, out),
                .last => mx.attention(q, t.keys, t.values, s.k, 0, lastOf(s, r) -| 1, t.partials, out),
                .half => try t.halfRow(r, out),
            }
        }
    }

    /// The row's softmax and values accumulated in fp16, the scores still in fp32.
    fn halfRow(t: *Attn, r: usize, out: []u16) !void {
        const last = lastOf(t.shape, r);
        const sc = try t.a.alloc(f32, last + 1);
        defer t.a.free(sc);
        var top: f32 = -std.math.inf(f32);
        for (0..last + 1) |j| {
            var d: f32 = 0;
            for (0..head_dim) |i| d = @mulAdd(f32, k.f32of(t.q[r * head_dim + i]), k.f32of(t.keys[j * head_dim + i]), d);
            sc[j] = d * (1.0 / 16.0);
            top = @max(top, sc[j]);
        }
        var denom: f16 = 0;
        var acc: [head_dim]f16 = @splat(0);
        for (0..last + 1) |j| {
            const e: f16 = @floatCast(@exp(sc[j] - top));
            denom += e;
            for (&acc, 0..) |*o, i| o.* += e * @as(f16, @floatCast(k.f32of(t.values[j * head_dim + i])));
        }
        for (out, acc) |*o, v| o.* = k.bf16of(@as(f32, @floatCast(v)) / @as(f32, @floatCast(denom)));
    }

    pub fn count(t: *Attn) usize {
        return t.shape.rows * head_dim;
    }

    /// The row's scores q·k_j / 16 in float64 into `into`; returns the largest Σ|q_i k_ji| / 16, the dot's magnitude.
    fn scores(t: *Attn, r: usize, into: []f64) f64 {
        var biggest: f64 = 0;
        for (into, 0..) |*s, j| {
            var dot: f64 = 0;
            var mag: f64 = 0;
            for (0..head_dim) |i| {
                const term = @as(f64, k.f32of(t.q[r * head_dim + i])) * @as(f64, k.f32of(t.keys[j * head_dim + i]));
                dot += term;
                mag += @abs(term);
            }
            s.* = dot / 16.0;
            biggest = @max(biggest, mag / 16.0);
        }
        return biggest;
    }

    /// Float64 softmax(q·k / 16)·v a row; mag is Σ_j w_j |v_j|, the weighted values the sums' error scales with.
    pub fn reference(t: *Attn, want: []f64, mag: []f64) !void {
        const s = t.shape;
        const sc = try t.a.alloc(f64, s.k);
        defer t.a.free(sc);
        for (0..s.rows) |r| {
            const n = lastOf(s, r) + 1;
            t.score_mag[r] = t.scores(r, sc[0..n]);
            var top: f64 = -std.math.inf(f64);
            for (sc[0..n]) |v| top = if (std.math.isNan(v) or std.math.isNan(top)) std.math.nan(f64) else @max(top, v);
            var denom: f64 = 0;
            const w = want[r * head_dim ..][0..head_dim];
            const m = mag[r * head_dim ..][0..head_dim];
            @memset(w, 0);
            @memset(m, 0);
            for (sc[0..n], 0..) |v, j| {
                const e = @exp(v - top);
                denom += e;
                for (w, m, 0..) |*o, *mm, i| {
                    const val: f64 = k.f32of(t.values[j * head_dim + i]);
                    o.* += e * val;
                    mm.* += e * @abs(val);
                }
            }
            for (w, m) |*o, *mm| {
                o.* /= denom;
                mm.* /= denom;
            }
        }
    }

    /// The dots' error moves each score by gamma(256)·Σ|q k|/16, so each weight on both softmax sides; then the sums.
    pub fn bounds(t: *Attn, want: []const f64, mag: []const f64, allow: []f64) !void {
        const s = t.shape;
        const u = c.unit(.f32);
        for (0..s.rows) |r| {
            const dots: f64 = if (t.stress) 0 else 2 * c.gamma(@floatFromInt(head_dim), u) * t.score_mag[r];
            const softmax = dots + 2 * c.gamma(@floatFromInt(lastOf(s, r) + 1 + 8), u) + 4 * u;
            for (allow[r * head_dim ..][0..head_dim], want[r * head_dim ..][0..head_dim], mag[r * head_dim ..][0..head_dim]) |*l, w, m| {
                const work = softmax * m;
                l.* = work + 0x1p-8 * (@abs(w) + work) + 0x1p-133;
            }
        }
    }

    /// Every sum in `acc` in `order`: the dots, the softmax's denominator and the weighted values; exp exact.
    pub fn simulate(t: *Attn, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        const s = t.shape;
        const buf = try t.a.alloc(f64, head_dim + 2 * s.k);
        defer t.a.free(buf);
        const dot = buf[0..head_dim];
        const sc = buf[head_dim..][0..s.k];
        const terms = buf[head_dim + s.k ..];
        for (0..s.rows) |r| {
            const n = lastOf(s, r) + 1;
            for (sc[0..n], 0..) |*v, j| {
                for (dot, 0..) |*d, i| d.* = @as(f64, k.f32of(t.q[r * head_dim + i])) * @as(f64, k.f32of(t.keys[j * head_dim + i]));
                v.* = rows_.sumIn(acc, order, dot) / 16.0;
            }
            var top: f64 = -std.math.inf(f64);
            for (sc[0..n]) |v| top = @max(top, v);
            for (sc[0..n]) |*v| v.* = @exp(v.* - top);
            const denom = rows_.sumIn(acc, order, sc[0..n]);
            for (0..head_dim) |i| {
                for (terms[0..n], sc[0..n], 0..) |*tt, e, j| tt.* = e * @as(f64, k.f32of(t.values[j * head_dim + i]));
                out[r * head_dim + i] = rows_.sumIn(acc, order, terms[0..n]) / denom;
            }
        }
    }
};

test "each control binds to its gate, the served caches cross the chunk and split edges, and a row sees its own key" {
    try std.testing.expectEqual(g.Gate.nonfinite, control(.finite).?);
    try std.testing.expectEqual(g.Gate.repeat, control(.writes).?);
    try std.testing.expectEqual(g.Gate.shapes, control(.last).?);
    try std.testing.expectEqual(g.Gate.sums, control(.half).?);
    for (served ++ ctl) |s| try std.testing.expect(Attn.takes(s));
    try std.testing.expect(!Attn.takes(.{ .rows = 2, .n = head_dim, .k = 1 }));
    try std.testing.expectEqual(@as(usize, 4095), lastOf(served[2], 15));
    try std.testing.expectEqual(@as(usize, head_dim + 4096 + 8), depth(served[1]));
}
