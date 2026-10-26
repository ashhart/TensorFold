//! layer_input.zig for CUDA builds, rule for rule (cuda_config.zig and the core module in place of its imports).
const std = @import("std");
const bf16_round = @import("core").bf16_round;
const cfg = @import("cuda_config.zig");

const rn = bf16_round.round32;

/// A weak-typed scalar as MLX multiplies by it: rounded to bf16 first.
fn scalar(v: f64) f32 {
    return rn(@floatCast(v));
}

/// per_layer_model_projection [layers * width, hidden] and per_layer_projection_norm [width], as bf16 values.
pub const Model = struct { project: []const f32, norm: []const f32 };

/// One layer's gate [width, hidden], projection [hidden, width], post norm [hidden] and layer_scalar, as bf16 values.
pub const Layer = struct { gate: []const f32, project: []const f32, norm: []const f32, scalar: f32 };

fn dot(a: []const f32, b: []const f32) f32 {
    var acc: f32 = 0;
    for (a, b) |x, y| acc += x * y;
    return acc;
}

/// mx.fast.rms_norm on one bf16 row: the row's normalized values rounded, then the weight's product rounded.
fn rms(x: []const f32, w: []const f32, out: []f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + cfg.eps);
    for (x, w, out) |v, wi, *o| o.* = rn(wi * rn(v * inv));
}

/// mlx_lm's gelu_approx(x) * up as MLX's bf16 ops (the geglu kernels' rounding points).
fn geglu(x: f32, up: f32) f32 {
    const cube = rn(x * x * x);
    const inner = rn(x + rn(scalar(0.044715) * cube));
    const th = rn(std.math.tanh(rn(scalar(0.7978845608028654) * inner)));
    const gelu = rn(rn(0.5 * x) * rn(1.0 + th));
    return rn(gelu * up);
}

/// A row's inputs for every layer in `out` [layers, width], from its scaled embedding `h` and its `table` row.
pub fn inputs(g: cfg.Geometry, m: Model, h: []const f32, table: []const f32, out: []f32) void {
    const width = g.layer_input;
    const scale = scalar(1.0 / @sqrt(@as(f64, @floatFromInt(g.hidden))));
    const token = scalar(@sqrt(@as(f64, @floatFromInt(width))));
    const mix = scalar(std.math.sqrt1_2);
    for (0..g.layers) |l| {
        const row = out[l * width ..][0..width];
        for (row, 0..) |*p, j| p.* = rn(rn(dot(m.project[(l * width + j) * g.hidden ..][0..g.hidden], h)) * scale);
        rms(row, m.norm, row);
        for (row, table[l * width ..][0..width]) |*p, t| p.* = rn(rn(p.* + rn(t * token)) * mix);
    }
}

/// A layer's per-layer block on residual row `h` [hidden], in place; `input` [width], `scratch` [width + hidden].
pub fn apply(g: cfg.Geometry, l: Layer, input: []const f32, h: []f32, scratch: []f32) void {
    const gate = scratch[0..g.layer_input];
    for (gate, input, 0..) |*v, up, j| v.* = geglu(rn(dot(l.gate[j * g.hidden ..][0..g.hidden], h)), up);
    const x = scratch[g.layer_input..][0..g.hidden];
    for (x, 0..) |*v, j| v.* = rn(dot(l.project[j * g.layer_input ..][0..g.layer_input], gate));
    rms(x, l.norm, x);
    for (h, x) |*r, v| r.* = rn(rn(r.* + v) * l.scalar);
}

/// The same steps in float64, each bf16 rounding one rounding of the float64 value; `skip` leaves one out (a control).
const Ref = struct {
    skip: ?enum { mix_sum } = null,

    fn bf(v: f64) f64 {
        return bf16_round.bf16(v);
    }

    fn dot(a: []const f32, b: []const f64) f64 {
        var acc: f64 = 0;
        for (a, b) |x, y| acc += @as(f64, x) * y;
        return acc;
    }

    fn rms(x: []const f64, w: []const f32, out: []f64) void {
        var ss: f64 = 0;
        for (x) |v| ss += v * v;
        const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + @as(f64, cfg.eps));
        for (x, w, out) |v, wi, *o| o.* = bf(@as(f64, wi) * bf(v * inv));
    }

    fn geglu(x: f64, up: f64) f64 {
        const cube = bf(x * x * x);
        const inner = bf(x + bf(bf(0.044715) * cube));
        const th = bf(std.math.tanh(bf(bf(0.7978845608028654) * inner)));
        return bf(bf(bf(0.5 * x) * bf(1.0 + th)) * up);
    }

    fn inputs(r: Ref, g: cfg.Geometry, m: Model, h: []const f64, table: []const f32, out: []f64) void {
        const width = g.layer_input;
        const scale = bf(1.0 / @sqrt(@as(f64, @floatFromInt(g.hidden))));
        const token = bf(@sqrt(@as(f64, @floatFromInt(width))));
        const mix = bf(std.math.sqrt1_2);
        for (0..g.layers) |l| {
            const row = out[l * width ..][0..width];
            for (row, 0..) |*p, j| p.* = bf(bf(Ref.dot(m.project[(l * width + j) * g.hidden ..][0..g.hidden], h)) * scale);
            Ref.rms(row, m.norm, row);
            for (row, table[l * width ..][0..width]) |*p, t| {
                const sum = if (r.skip == .mix_sum) p.* + bf(@as(f64, t) * token) else bf(p.* + bf(@as(f64, t) * token));
                p.* = bf(sum * mix);
            }
        }
    }

    fn apply(g: cfg.Geometry, l: Layer, input: []const f64, h: []f64) void {
        var gate: [64]f64 = undefined;
        for (gate[0..g.layer_input], input, 0..) |*v, up, j| v.* = Ref.geglu(bf(Ref.dot(l.gate[j * g.hidden ..][0..g.hidden], h)), up);
        var x: [128]f64 = undefined;
        for (x[0..g.hidden], 0..) |*v, j| v.* = bf(Ref.dot(l.project[j * g.layer_input ..][0..g.layer_input], gate[0..g.layer_input]));
        Ref.rms(x[0..g.hidden], l.norm, x[0..g.hidden]);
        for (h, x[0..g.hidden]) |*r, v| r.* = bf(bf(r.* + v) * @as(f64, l.scalar));
    }
};

/// A tiny E-series geometry: two cache-reading layers of four, a per-layer input of 16 on a hidden width of 32.
const tiny: cfg.Geometry = .{ .name = "tiny", .hidden = 32, .layers = 4, .intermediate = 64, .query_heads = 2, .sliding_kv_heads = 1, .global_kv_heads = 1, .layer_input = 16, .shared_layers = 2, .keys_are_values = false };
const rows = 48;

/// bf16 values: uniform in (-span, span), each layer's weights scaled by its fan-in.
fn fill(prng: *std.Random.DefaultPrng, v: []f32, span: f32) void {
    for (v) |*x| x.* = rn((prng.random().float(f32) * 2 - 1) * span);
}

const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    model: Model,
    layers: [tiny.layers]Layer,
    h: []f32,
    table: []f32,

    fn init() !Fixture {
        var f: Fixture = .{ .arena = .init(std.testing.allocator), .model = undefined, .layers = undefined, .h = undefined, .table = undefined };
        const a = f.arena.allocator();
        var prng = std.Random.DefaultPrng.init(1222);
        const width = tiny.layer_input;
        const project = try a.alloc(f32, tiny.layers * width * tiny.hidden);
        fill(&prng, project, 1.0 / @sqrt(@as(f32, tiny.hidden)));
        const norm = try a.alloc(f32, width);
        fill(&prng, norm, 2);
        f.model = .{ .project = project, .norm = norm };
        for (&f.layers) |*l| {
            const gate = try a.alloc(f32, width * tiny.hidden);
            const back = try a.alloc(f32, tiny.hidden * width);
            const post = try a.alloc(f32, tiny.hidden);
            fill(&prng, gate, 1.0 / @sqrt(@as(f32, tiny.hidden)));
            fill(&prng, back, 1.0 / @sqrt(@as(f32, width)));
            fill(&prng, post, 2);
            l.* = .{ .gate = gate, .project = back, .norm = post, .scalar = rn(0.25 + prng.random().float(f32)) };
        }
        f.h = try a.alloc(f32, rows * tiny.hidden);
        fill(&prng, f.h, 6);
        f.table = try a.alloc(f32, rows * tiny.layers * width);
        fill(&prng, f.table, 0.5);
        return f;
    }
};

/// The worst column's difference in its own largest magnitude, the aggregate RMS, and how many values differ at all.
const Spread = struct { worst: f64, rms: f64, differing: usize };

fn spread(host: []const f32, want: []const f64, cols: usize) Spread {
    var worst: f64 = 0;
    var sq: f64 = 0;
    var ref_sq: f64 = 0;
    var differing: usize = 0;
    for (0..cols) |c| {
        var top: f64 = 0;
        var diff: f64 = 0;
        var i = c;
        while (i < host.len) : (i += cols) {
            top = @max(top, @abs(want[i]));
            const d = @abs(@as(f64, host[i]) - want[i]);
            diff = @max(diff, d);
            sq += d * d;
            ref_sq += want[i] * want[i];
            if (d != 0) differing += 1;
        }
        worst = @max(worst, diff / top);
    }
    return .{ .worst = worst, .rms = @sqrt(sq / ref_sq), .differing = differing };
}

test "the per-layer inputs of a row follow the float64 reference on the same bf16 steps: worst column and aggregate" {
    var f = try Fixture.init();
    defer f.arena.deinit();
    const n = tiny.layers * tiny.layer_input;
    var host: [rows * n]f32 = undefined;
    var want: [rows * n]f64 = undefined;
    var control: [rows * n]f64 = undefined;
    for (0..rows) |r| {
        var h64: [tiny.hidden]f64 = undefined;
        for (&h64, f.h[r * tiny.hidden ..][0..tiny.hidden]) |*d, s| d.* = s;
        inputs(tiny, f.model, f.h[r * tiny.hidden ..][0..tiny.hidden], f.table[r * n ..][0..n], host[r * n ..][0..n]);
        (Ref{}).inputs(tiny, f.model, &h64, f.table[r * n ..][0..n], want[r * n ..][0..n]);
        (Ref{ .skip = .mix_sum }).inputs(tiny, f.model, &h64, f.table[r * n ..][0..n], control[r * n ..][0..n]);
    }
    const s = spread(&host, &want, n);
    // a rounding flipped by the f32 accumulation order moves a value by one bf16 step, at most 2^-7 of it
    try std.testing.expect(s.worst <= 0x1p-7);
    try std.testing.expect(s.rms <= 0x1p-9);
    try std.testing.expect(s.differing * 50 <= host.len);
    // the same reference without the sum's rounding is a different computation: far more values differ
    const c = spread(&host, &control, n);
    try std.testing.expect(c.differing > 4 * s.differing and c.differing * 10 > host.len);
}

test "the per-layer block follows the float64 reference on the same bf16 steps: worst column and aggregate" {
    var f = try Fixture.init();
    defer f.arena.deinit();
    const width = tiny.layer_input;
    var host: [rows * tiny.hidden]f32 = undefined;
    var want: [rows * tiny.hidden]f64 = undefined;
    for (0..rows) |r| {
        const stream = f.h[r * tiny.hidden ..][0..tiny.hidden];
        const input = f.table[r * tiny.layers * width ..][0..width];
        var in64: [width]f64 = undefined;
        for (&in64, input) |*d, s| d.* = s;
        const out = host[r * tiny.hidden ..][0..tiny.hidden];
        @memcpy(out, stream);
        var scratch: [width + tiny.hidden]f32 = undefined;
        apply(tiny, f.layers[r % tiny.layers], input, out, &scratch);
        const ref = want[r * tiny.hidden ..][0..tiny.hidden];
        for (ref, stream) |*d, s| d.* = s;
        Ref.apply(tiny, f.layers[r % tiny.layers], &in64, ref);
    }
    const s = spread(&host, &want, tiny.hidden);
    try std.testing.expect(s.worst <= 0x1p-6);
    try std.testing.expect(s.rms <= 0x1p-8);
    try std.testing.expect(s.differing * 20 <= host.len);
}
