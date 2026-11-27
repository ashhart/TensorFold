//! The CPU engine's pointwise rows under the contract: embedding, residual add, MLP activation, attention gate, rotary.
const std = @import("std");
const contract = @import("contract");
const mx = @import("cpu_mixers");
const rig_ = @import("contract_rig.zig");

const ops = mx.ops;
const k = ops.kernels;
const c = contract.spec;
const g = contract.gates;

/// Five rows and their controls: a sanitizer (gate 1), a write into an input (gate 2) and a wrong operand (gate 3).
pub const Which = enum { embed, residual, act, gate, rotate, embed_finite, embed_group, residual_finite, residual_writes, residual_half, act_finite, act_shift, gate_finite, gate_half, rotate_finite, rotate_position };

/// The token rows of a decode step, a verify window, the verify batch over the streams and a prompt chunk, each width.
fn served(comptime n: usize, comptime kk: usize) [4]c.Shape {
    return .{ .{ .rows = 1, .n = n, .k = kk }, .{ .rows = 16, .n = n, .k = kk }, .{ .rows = 32, .n = n, .k = kk }, .{ .rows = 128, .n = n, .k = kk } };
}
const served_hidden = served(c_hidden, 1);
const served_embed = served(c_hidden, k.group);
const served_act = served(c_intermediate, 1);
const served_head = served(c_head_dim, 1);
const ctl_hidden = [_]c.Shape{.{ .rows = 9, .n = c_hidden, .k = 1 }};
const ctl_embed = [_]c.Shape{.{ .rows = 9, .n = c_hidden, .k = k.group }};
const ctl_act = [_]c.Shape{.{ .rows = 9, .n = c_intermediate, .k = 1 }};
const ctl_head = [_]c.Shape{.{ .rows = 9, .n = c_head_dim, .k = 1 }};
const c_hidden = 2048;
const c_intermediate = 6144;
const c_head_dim = 256;
/// Rows of the embedding table the fixture holds (two tiles of eight); the token ids index them.
const table_rows = 16;
/// The rotary's first position: a row r rotates at `position0 + r`, well into the angles' range.
const position0 = 4096;

fn flat(_: c.Shape) usize {
    return 0;
}

/// The row the control is a fault of.
fn base(w: Which) Which {
    return switch (w) {
        .embed, .embed_finite, .embed_group => .embed,
        .residual, .residual_finite, .residual_writes, .residual_half => .residual,
        .act, .act_finite, .act_shift => .act,
        .gate, .gate_finite, .gate_half => .gate,
        .rotate, .rotate_finite, .rotate_position => .rotate,
    };
}

pub fn spec(w: Which) c.Spec {
    const name: []const u8 = switch (w) {
        .embed => "cpu embedding q4 g64",
        .residual => "cpu residual add",
        .act => "cpu MLP activation",
        .gate => "cpu attention gate",
        .rotate => "cpu rotary",
        .embed_finite => "control: embedding taking a non-finite table entry as zero",
        .embed_group => "control: embedding reading the next group's scale and bias",
        .residual_finite => "control: residual add taking a non-finite input as zero",
        .residual_writes => "control: residual add writing its residual input",
        .residual_half => "control: residual add adding half the residual",
        .act_finite => "control: activation taking a non-finite gate as zero",
        .act_shift => "control: activation taking the next element's up value",
        .gate_finite => "control: attention gate taking a non-finite gate as zero",
        .gate_half => "control: attention gate reading the query half instead of the gate half",
        .rotate_finite => "control: rotary taking a non-finite input as zero",
        .rotate_position => "control: rotary at the next position",
    };
    const b = base(w);
    const is_control = w != b;
    const shapes: []const c.Shape = switch (b) {
        .embed => if (is_control) &ctl_embed else &served_embed,
        .residual, .gate => if (is_control) &ctl_hidden else &served_hidden,
        .act => if (is_control) &ctl_act else &served_act,
        .rotate => if (is_control) &ctl_head else &served_head,
        else => unreachable,
    };
    return .{ .name = name, .accum = .none, .depth = flat, .out = .bf16, .served = shapes, .sums = .{ .f16 = 0, .bf16 = 0, .why = "each output is one float64 expression rounded to bf16 through fp32; there is no sum" } };
}

/// The gate each control must fail.
pub fn control(w: Which) ?g.Gate {
    return switch (w) {
        .embed, .residual, .act, .gate, .rotate => null,
        .embed_finite, .residual_finite, .act_finite, .gate_finite, .rotate_finite => .nonfinite,
        .residual_writes => .repeat,
        .embed_group, .residual_half, .act_shift, .gate_half, .rotate_position => .shapes,
    };
}

pub const Row = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    shape: c.Shape = undefined,
    l: k.Linear = undefined,
    ids: []u32 = &.{},
    x: []u16 = &.{},
    y: []u16 = &.{},
    clean: []u16 = &.{},
    out: []u16 = &.{},

    pub fn init(a: std.mem.Allocator, which: Which) Row {
        return .{ .a = a, .rig = .{ .a = a }, .which = which };
    }

    pub fn deinit(r: *Row) void {
        r.rig.deinit();
    }

    pub fn kernel(r: *Row) g.Kernel {
        return .{ .spec = spec(r.which), .ctx = r, .vt = &vt, .control = control(r.which) };
    }

    const vt = g.hooks(Row);

    fn takes(w: Which, shape: c.Shape) bool {
        if (shape.rows == 0 or shape.rows > 128) return false;
        return switch (base(w)) {
            .embed => shape.n == c_hidden and shape.k == k.group,
            .residual, .gate => shape.n == c_hidden and shape.k == 1,
            .act => shape.n == c_intermediate and shape.k == 1,
            .rotate => shape.n == c_head_dim and shape.k == 1,
            else => unreachable,
        };
    }

    /// Random bf16 values, roughly normal, with an occasional large one.
    fn randomRow(rnd: std.Random, out: []u16, scale: f32) void {
        for (out, 0..) |*v, i| v.* = k.bf16of(rnd.floatNorm(f32) * scale * (if (i % 97 == 13) @as(f32, 8) else 1));
    }

    const nan = k.bf16of(std.math.nan(f32));
    const inf = k.bf16of(std.math.inf(f32));
    const ninf = k.bf16of(-std.math.inf(f32));

    pub fn prepare(r: *Row, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        r.rig.reset();
        if (!takes(r.which, shape)) return false;
        r.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const n = shape.n;
        switch (base(r.which)) {
            .embed => {
                const words = try r.a.alloc(u32, table_rows * c_hidden / 8);
                defer r.a.free(words);
                for (words) |*w| w.* = rnd.int(u32);
                const groups = c_hidden / k.group;
                const scales = try r.a.alloc(u16, table_rows * groups);
                defer r.a.free(scales);
                const biases = try r.a.alloc(u16, table_rows * groups);
                defer r.a.free(biases);
                for (scales, biases) |*s, *b| {
                    s.* = k.bf16of(rnd.float(f32) * 0.02 + 0.001);
                    b.* = k.bf16of(-rnd.float(f32) * 0.2);
                }
                if (fill == .nonfinite) {
                    scales[3 * groups + 1] = nan;
                    biases[5 * groups + 7] = inf;
                    scales[7 * groups + 2] = ninf;
                }
                const packed_ = try k.pack(r.a, words, scales, biases, table_rows, c_hidden, 16);
                defer k.free(r.a, packed_);
                r.l = .{ .tiles = try r.rig.input(u32, packed_.tiles), .scales = try r.rig.input(u16, packed_.scales), .biases = try r.rig.input(u16, packed_.biases), .inputs = c_hidden, .outputs = table_rows, .chains = 16 };
                const ids = try r.a.alloc(u32, shape.rows);
                defer r.a.free(ids);
                for (ids, 0..) |*id, i| id.* = @intCast((i * 5 + 3) % table_rows);
                r.ids = try r.rig.input(u32, ids);
            },
            .residual, .act, .gate => {
                const xs = try r.a.alloc(u16, shape.rows * n);
                defer r.a.free(xs);
                const wide: usize = if (base(r.which) == .gate) 2 * n else n;
                const ys = try r.a.alloc(u16, shape.rows * wide);
                defer r.a.free(ys);
                randomRow(rnd, xs, 1);
                randomRow(rnd, ys, if (base(r.which) == .act) 2 else 1);
                if (fill == .nonfinite) {
                    xs[n / 3] = nan;
                    if (shape.rows > 1) xs[n + 7] = inf;
                    if (shape.rows > 2) ys[2 * wide + 11] = ninf;
                    if (shape.rows > 3) {
                        xs[3 * n + 5] = inf;
                        ys[3 * wide + 5] = ninf;
                    }
                    if (shape.rows > 4) ys[4 * wide + wide - 1] = nan;
                }
                r.x = try r.rig.input(u16, xs);
                r.y = try r.rig.input(u16, ys);
            },
            .rotate => {
                const xs = try r.a.alloc(u16, shape.rows * n);
                defer r.a.free(xs);
                randomRow(rnd, xs, 1);
                if (fill == .nonfinite) {
                    xs[5] = nan;
                    if (shape.rows > 1) xs[n + 40] = inf;
                    if (shape.rows > 2) xs[2 * n + 200] = ninf;
                }
                r.x = try r.rig.input(u16, xs);
            },
            else => unreachable,
        }
        r.clean = try r.rig.scratch(u16, shape.rows * 2 * n);
        r.out = try r.rig.output(u16, shape.rows * n);
        return true;
    }

    /// `v` with every non-finite bf16 replaced by zero, in the scratch from `at`: the sanitizing controls' input.
    fn sanitizedAt(r: *Row, at: usize, v: []const u16) []const u16 {
        const clean = r.clean[at..][0..v.len];
        for (clean, v) |*o, b| o.* = if (std.math.isFinite(k.f32of(b))) b else 0;
        return clean;
    }

    fn sanitized(r: *Row, v: []const u16) []const u16 {
        return r.sanitizedAt(0, v);
    }

    pub fn run(r: *Row) !void {
        const n = r.shape.n;
        for (0..r.shape.rows) |row| {
            const out = r.out[row * n ..][0..n];
            switch (r.which) {
                .embed => ops.embed(r.l, r.ids[row], out),
                .embed_finite => {
                    var l = r.l;
                    l.scales = r.sanitized(r.l.scales);
                    l.biases = r.sanitizedAt(r.l.scales.len, r.l.biases);
                    ops.embed(l, r.ids[row], out);
                },
                .embed_group => {
                    const groups = c_hidden / k.group;
                    const id = r.ids[row];
                    for (0..groups) |gi| {
                        const next = (gi + 1) % groups;
                        const s: f64 = k.f32of(r.l.scale(id, next));
                        const z: f64 = k.f32of(r.l.bias(id, next));
                        for (0..k.group) |i| out[gi * k.group + i] = ops.bf16of64(s * @as(f64, @floatFromInt(r.l.code(id, gi * k.group + i))) + z);
                    }
                },
                .residual, .residual_writes => {
                    @memcpy(out, r.x[row * n ..][0..n]);
                    ops.residual(out, r.y[row * n ..][0..n]);
                    if (r.which == .residual_writes and row == 0) r.y[0] ^= 1;
                },
                .residual_finite => {
                    @memcpy(out, r.sanitized(r.x[row * n ..][0..n]));
                    const y = r.sanitized(r.y[row * n ..][0..n]);
                    ops.residual(out, y);
                },
                .residual_half => for (out, r.x[row * n ..][0..n], r.y[row * n ..][0..n]) |*o, h, rr| {
                    o.* = ops.bf16of64(@as(f64, k.f32of(h)) + 0.5 * @as(f64, k.f32of(rr)));
                },
                .act => ops.mlpAct(r.x[row * n ..][0..n], r.y[row * n ..][0..n], out),
                .act_finite => ops.mlpAct(r.sanitized(r.x[row * n ..][0..n]), r.y[row * n ..][0..n], out),
                .act_shift => for (out, 0..) |*o, i| {
                    o.* = ops.bf16of64(ops.silu(k.f32of(r.x[row * n + i])) * @as(f64, k.f32of(r.y[row * n + (i + 1) % n])));
                },
                .gate => ops.attentionGate(r.x[row * n ..][0..n], r.y[row * 2 * n ..][0 .. 2 * n], out),
                .gate_finite => ops.attentionGate(r.x[row * n ..][0..n], r.sanitized(r.y[row * 2 * n ..][0 .. 2 * n]), out),
                .gate_half => for (out, 0..) |*o, d| {
                    const head = d / c_head_dim;
                    const q: f64 = k.f32of(r.y[row * 2 * n + head * 2 * c_head_dim + d % c_head_dim]);
                    o.* = ops.bf16of64(@as(f64, k.f32of(r.x[row * n + d])) * ops.sigmoid(q));
                },
                .rotate => ops.rotate(r.x[row * n ..][0..n], position0 + row, out),
                .rotate_finite => ops.rotate(r.sanitized(r.x[row * n ..][0..n]), position0 + row, out),
                .rotate_position => ops.rotate(r.x[row * n ..][0..n], position0 + row + 1, out),
            }
        }
    }

    pub fn count(r: *Row) usize {
        return r.shape.rows * r.shape.n;
    }

    /// Each output's float64 value from the same bf16 inputs; mag is its magnitude (no sums, so it bounds nothing).
    pub fn reference(r: *Row, want: []f64, mag: []f64) !void {
        const n = r.shape.n;
        for (0..r.shape.rows) |row| {
            const w = want[row * n ..][0..n];
            switch (base(r.which)) {
                .embed => {
                    const id = r.ids[row];
                    for (0..c_hidden / k.group) |gi| {
                        const s: f64 = k.f32of(r.l.scale(id, gi));
                        const z: f64 = k.f32of(r.l.bias(id, gi));
                        for (0..k.group) |i| w[gi * k.group + i] = s * @as(f64, @floatFromInt(r.l.code(id, gi * k.group + i))) + z;
                    }
                },
                .residual => for (w, r.x[row * n ..][0..n], r.y[row * n ..][0..n]) |*o, h, rr| {
                    o.* = @as(f64, k.f32of(h)) + @as(f64, k.f32of(rr));
                },
                .act => for (w, r.x[row * n ..][0..n], r.y[row * n ..][0..n]) |*o, gv, u| {
                    const x: f64 = k.f32of(gv);
                    o.* = x / (1.0 + @exp(-x)) * @as(f64, k.f32of(u));
                },
                .gate => for (w, 0..) |*o, d| {
                    const head = d / c_head_dim;
                    const gv: f64 = k.f32of(r.y[row * 2 * n + head * 2 * c_head_dim + c_head_dim + d % c_head_dim]);
                    o.* = @as(f64, k.f32of(r.x[row * n + d])) / (1.0 + @exp(-gv));
                },
                .rotate => {
                    const x = r.x[row * n ..][0..n];
                    const half = 32;
                    for (0..half) |i| {
                        const angle = @as(f64, @floatFromInt(position0 + row)) * std.math.pow(f64, 10000000.0, -@as(f64, @floatFromInt(2 * i)) / 64.0);
                        const a: f64 = k.f32of(x[i]);
                        const b: f64 = k.f32of(x[i + half]);
                        w[i] = a * @cos(angle) - b * @sin(angle);
                        w[i + half] = b * @cos(angle) + a * @sin(angle);
                    }
                    for (2 * half..n) |i| w[i] = k.f32of(x[i]);
                },
                else => unreachable,
            }
        }
        for (mag, want) |*m, v| m.* = @abs(v);
    }
};

test "each control binds to its gate, and every row's served shapes are what the engine calls" {
    try std.testing.expectEqual(g.Gate.nonfinite, control(.embed_finite).?);
    try std.testing.expectEqual(g.Gate.repeat, control(.residual_writes).?);
    try std.testing.expectEqual(g.Gate.shapes, control(.rotate_position).?);
    for (std.enums.values(Which)) |w| for (spec(w).served) |s| try std.testing.expect(Row.takes(w, s));
    try std.testing.expect(!Row.takes(.embed, .{ .rows = 1, .n = 1024, .k = k.group }));
    try std.testing.expect(!Row.takes(.rotate, .{ .rows = 129, .n = c_head_dim, .k = 1 }));
}
