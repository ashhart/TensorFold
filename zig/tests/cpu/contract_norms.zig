//! The CPU engine's head norms under the contract: the q/k RMS norm over 256 and the delta rule's gated output norm.
const std = @import("std");
const contract = @import("contract");
const mx = @import("cpu_mixers");
const rig_ = @import("contract_rig.zig");

const k = mx.ops.kernels;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;

pub const Which = enum { head_norm, delta_norm, head_finite, head_shift, delta_finite, delta_z };

const head_dim = 256;
const linear_dim = 128;
const linear_heads = 16;
const hidden = linear_heads * linear_dim;

fn one(_: c.Shape) usize {
    return 1;
}

/// Ten head rows a token: a decode step, a verify window, the verify batch, a prompt chunk; the delta norm's: tokens.
const served_head = [_]c.Shape{ .{ .rows = 10, .n = head_dim, .k = head_dim }, .{ .rows = 160, .n = head_dim, .k = head_dim }, .{ .rows = 320, .n = head_dim, .k = head_dim }, .{ .rows = 1280, .n = head_dim, .k = head_dim } };
const served_delta = [_]c.Shape{ .{ .rows = 1, .n = hidden, .k = linear_dim }, .{ .rows = 16, .n = hidden, .k = linear_dim }, .{ .rows = 32, .n = hidden, .k = linear_dim }, .{ .rows = 128, .n = hidden, .k = linear_dim } };
const ctl_head = [_]c.Shape{.{ .rows = 9, .n = head_dim, .k = head_dim }};
const ctl_delta = [_]c.Shape{.{ .rows = 9, .n = hidden, .k = linear_dim }};

const sums_why = "the squares are summed in float64, inside the fp32 bound by construction; the fp16 and bf16 margins are measured, not required";

fn base(w: Which) Which {
    return switch (w) {
        .head_norm, .head_finite, .head_shift => .head_norm,
        .delta_norm, .delta_finite, .delta_z => .delta_norm,
    };
}

pub fn spec(w: Which) c.Spec {
    const name: []const u8 = switch (w) {
        .head_norm => "cpu head norm",
        .delta_norm => "cpu delta output norm",
        .head_finite => "control: head norm taking a non-finite input as zero",
        .head_shift => "control: head norm scaling by the next element's weight",
        .delta_finite => "control: delta output norm taking a non-finite input as zero",
        .delta_z => "control: delta output norm gating by the next element's z",
    };
    const is_control = w != base(w);
    const shapes: []const c.Shape = if (base(w) == .head_norm) (if (is_control) &ctl_head else &served_head) else (if (is_control) &ctl_delta else &served_delta);
    return .{ .name = name, .accum = .f32, .depth = one, .out = .bf16, .served = shapes, .sums = .{ .f16 = 0, .bf16 = 0, .why = sums_why } };
}

/// The gate each control must fail.
pub fn control(w: Which) ?g.Gate {
    return switch (w) {
        .head_norm, .delta_norm => null,
        .head_finite, .delta_finite => .nonfinite,
        .head_shift, .delta_z => .shapes,
    };
}

pub const Norm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    shape: c.Shape = undefined,
    r: ?rows_.Rows = null,
    x: []u16 = &.{},
    z: []u16 = &.{},
    w: []u16 = &.{},
    clean: []u16 = &.{},
    out: []u16 = &.{},

    pub fn init(a: std.mem.Allocator, which: Which) Norm {
        return .{ .a = a, .rig = .{ .a = a }, .which = which };
    }

    pub fn deinit(n: *Norm) void {
        n.drop();
        n.rig.deinit();
    }

    fn drop(n: *Norm) void {
        if (n.r) |r| r.deinit();
        n.r = null;
        n.rig.reset();
    }

    pub fn kernel(n: *Norm) g.Kernel {
        return .{ .spec = spec(n.which), .ctx = n, .vt = &vt, .control = control(n.which) };
    }

    const vt = g.hooks(Norm);

    fn takes(w: Which, shape: c.Shape) bool {
        if (shape.rows == 0) return false;
        return if (base(w) == .head_norm) shape.n == head_dim and shape.k == head_dim else shape.n == hidden and shape.k == linear_dim;
    }

    /// The head norm's rows are the `rows` family's RMS rows; the delta norm's y rows the same family's, a head a row.
    pub fn prepare(n: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        n.drop();
        if (!takes(n.which, shape)) return false;
        n.shape = shape;
        const heads: usize = if (base(n.which) == .head_norm) 1 else linear_heads;
        const r = try rows_.Rows.init(n.a, .rms, .{ .rows = shape.rows * heads, .n = shape.k, .k = shape.k });
        n.r = r;
        r.fill(fill, seed);
        n.x = try n.rig.input(u16, r.x);
        n.w = try n.rig.input(u16, r.w);
        if (base(n.which) == .delta_norm) {
            var prng = std.Random.DefaultPrng.init(seed ^ 0x5a5a);
            const rnd = prng.random();
            const z = try n.a.alloc(u16, shape.rows * hidden);
            defer n.a.free(z);
            for (z) |*v| v.* = k.bf16of(rnd.floatNorm(f32));
            if (fill == .nonfinite) {
                z[hidden / 2] = k.bf16of(std.math.nan(f32));
                if (shape.rows > 1) z[hidden + 3] = k.bf16of(std.math.inf(f32));
            }
            n.z = try n.rig.input(u16, z);
        }
        n.clean = try n.rig.scratch(u16, shape.rows * hidden + hidden);
        n.out = try n.rig.output(u16, shape.rows * shape.n);
        return true;
    }

    fn sanitized(n: *Norm, at: usize, v: []const u16) []const u16 {
        const clean = n.clean[at..][0..v.len];
        for (clean, v) |*o, b| o.* = if (std.math.isFinite(k.f32of(b))) b else 0;
        return clean;
    }

    pub fn run(n: *Norm) !void {
        const s = n.shape;
        for (0..s.rows) |row| {
            const out = n.out[row * s.n ..][0..s.n];
            const x = n.x[row * s.n ..][0..s.n];
            switch (n.which) {
                .head_norm => mx.headNorm(x, n.w, out),
                .head_finite => mx.headNorm(n.sanitized(0, x), n.w, out),
                .head_shift => {
                    const shifted = n.clean[0..head_dim];
                    for (shifted, 0..) |*v, i| v.* = n.w[(i + 1) % head_dim];
                    mx.headNorm(x, shifted, out);
                },
                .delta_norm => mx.gdnPost(x, n.z[row * hidden ..][0..hidden], n.w, out),
                .delta_finite => mx.gdnPost(n.sanitized(0, x), n.sanitized(hidden, n.z[row * hidden ..][0..hidden]), n.w, out),
                .delta_z => {
                    const shifted = n.clean[0..hidden];
                    for (shifted, 0..) |*v, i| v.* = n.z[row * hidden + (i + 1) % hidden];
                    mx.gdnPost(x, shifted, n.w, out);
                },
            }
        }
    }

    pub fn count(n: *Norm) usize {
        return n.shape.rows * n.shape.n;
    }

    /// The RMS rows' float64 reference; the delta norm's rows then times silu(z), its mag scaled alike.
    pub fn reference(n: *Norm, want: []f64, mag: []f64) !void {
        try n.r.?.reference(want, mag);
        if (base(n.which) == .head_norm) return;
        for (want, mag, n.z) |*w, *m, zb| {
            const sz = mx.ops.silu(k.f32of(zb));
            w.* *= sz;
            m.* *= @abs(sz);
        }
    }

    /// The squares summed in `acc` in `order`, the rest exact; the delta norm's gate applied after.
    pub fn simulate(n: *Norm, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        try n.r.?.simulate(acc, order, out);
        if (base(n.which) == .head_norm) return;
        for (out, n.z) |*o, zb| o.* *= mx.ops.silu(k.f32of(zb));
    }
};

test "each control binds to its gate, and the served rows are the engine's head rows and token rows" {
    try std.testing.expectEqual(g.Gate.nonfinite, control(.head_finite).?);
    try std.testing.expectEqual(g.Gate.shapes, control(.head_shift).?);
    try std.testing.expectEqual(g.Gate.nonfinite, control(.delta_finite).?);
    try std.testing.expectEqual(g.Gate.shapes, control(.delta_z).?);
    for (std.enums.values(Which)) |w| for (spec(w).served) |s| try std.testing.expect(Norm.takes(w, s));
    try std.testing.expect(!Norm.takes(.head_norm, .{ .rows = 1, .n = 128, .k = 128 }));
    try std.testing.expectEqual(@as(usize, 10), served_head[0].rows);
}
