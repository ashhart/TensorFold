//! Independent binary64 objectives test the native Gemma scalar bodies and every explicit activation store.
const std = @import("std");
const core = @import("core");
const fixture = @import("../qwen4/learning_fixture.zig");
const data = @import("../qwen4/learning_contract_data.zig");
const a = std.testing.allocator;
fn bf(x: f64) f64 {
    return @as(f32, @bitCast(@as(u32, core.quant.convert.bf16(@floatCast(x))) << 16));
}
fn frozen(x: f64, offset: f64) f64 {
    return x + offset;
}
fn gelu(x: f64, offsets: *const [7]f64) f64 {
    const cube = frozen(x * x * x, offsets[0]);
    const scaled = frozen(bf(0.044715) * cube, offsets[1]);
    const inner = frozen(x + scaled, offsets[2]);
    const theta = frozen(bf(0.7978845608028654) * inner, offsets[3]);
    const t = frozen(std.math.tanh(theta), offsets[4]);
    const plus = frozen(1 + t, offsets[5]);
    return frozen(0.5 * x * plus, offsets[6]);
}
fn stores(x: f64) [7]f64 {
    var offsets: [7]f64 = undefined;
    const cube = x * x * x;
    offsets[0] = bf(cube) - cube;
    const scaled = bf(0.044715) * bf(cube);
    offsets[1] = bf(scaled) - scaled;
    const inner = x + bf(scaled);
    offsets[2] = bf(inner) - inner;
    const theta = bf(0.7978845608028654) * bf(inner);
    offsets[3] = bf(theta) - theta;
    const t = std.math.tanh(bf(theta));
    offsets[4] = bf(t) - t;
    const plus = 1 + bf(t);
    offsets[5] = bf(plus) - plus;
    const out = 0.5 * x * bf(plus);
    offsets[6] = bf(out) - out;
    return offsets;
}
test "Gemma4 learning: native rounded GeGLU and softcap match independent frozen-store derivatives" {
    const spec = core.living_gemma;
    for ([_]f32{ -4, -1, -0.125, 0, 0.25, 1, 3, 8 }) |x| {
        var p = data.params(spec, .geglu, 1, 1);
        var f = try fixture.Case(spec).init(.geglu, p);
        defer f.deinit();
        try f.put(0, std.mem.asBytes(&core.quant.convert.bf16(x)));
        const up: f32 = 0.625;
        const upstream: f32 = 0.75;
        try f.put(1, std.mem.asBytes(&core.quant.convert.bf16(up)));
        try f.put(2, std.mem.asBytes(&upstream));
        try f.run();
        const base = stores(x);
        for ([_]f64{ 0.0001, 0.00005 }) |eps| {
            const slope = (gelu(x + eps, &base) - gelu(x - eps, &base)) / (2 * eps) * up * upstream;
            try fixture.near(f.floats(8)[0], slope);
        }
        try fixture.near(f.floats(9)[0], gelu(x, &base) * upstream);
        p.f[0] = 30;
        var cap = try fixture.Case(spec).init(.softcap, p);
        defer cap.deinit();
        try cap.put(0, std.mem.asBytes(&core.quant.convert.bf16(x)));
        try cap.put(1, std.mem.asBytes(&upstream));
        try cap.run();
        const offset = bf(@as(f64, x) / 30) - @as(f64, x) / 30;
        for ([_]f64{ 0.0001, 0.00005 }) |eps| {
            const high = 30 * std.math.tanh((@as(f64, x) + eps) / 30 + offset);
            const low = 30 * std.math.tanh((@as(f64, x) - eps) / 30 + offset);
            try fixture.near(cap.floats(8)[0], (high - low) / (2 * eps) * upstream);
        }
    }
}
test "Gemma4 learning: native weighted head RMS and inverse RoPE have independent binary64 directions" {
    const spec = core.living_gemma;
    const d = 64;
    var p = data.params(spec, .rms, 2, d);
    var f = try data.Fixture(spec).init(a, .rms, p, 0x674, false);
    defer f.deinit();
    try f.raw.run();
    var slope: f64 = 0;
    var analytic: f64 = 0;
    for (0..2) |row| {
        var sq: f64 = 0;
        var dot: f64 = 0;
        var xd: f64 = 0;
        var wd: f64 = 0;
        for (0..d) |i| {
            const x = @as(f64, f.input(0, row * d + i));
            const w = @as(f64, f.input(1, i));
            const dy = @as(f64, f.raw.floats(2)[row * d + i]);
            const v: @TypeOf(x) = if ((row * d + i) % 3 == 0) -1 else 1;
            sq += x * x;
            dot += x * w * dy;
            xd += x * v;
            wd += w * dy * v;
            analytic += f.raw.floats(8)[row * d + i] * v;
        }
        const inv = 1 / @sqrt(sq / d + 1e-6);
        slope += inv * wd - inv * inv * inv * dot * xd / d;
    }
    try fixture.near(@floatCast(analytic), slope);
    p = data.params(spec, .rope, 3, 128);
    p.u[2] = 64;
    p.u[3] = 17;
    var rope = try data.Fixture(spec).init(a, .rope, p, 0x882, false);
    defer rope.deinit();
    try rope.raw.run();
    for (0..3 * 128) |id| {
        const row = id / 128;
        const col = id % 64;
        const pair = col % 32;
        const base = id - col;
        const angle = @as(f64, @floatFromInt(row + 17)) * rope.raw.floats(1)[pair];
        const left: @TypeOf(angle) = rope.raw.floats(0)[base + pair];
        const right: @TypeOf(angle) = rope.raw.floats(0)[base + 32 + pair];
        const want = if (col < 32) left * @cos(angle) + right * @sin(angle) else right * @cos(angle) - left * @sin(angle);
        try fixture.near(rope.raw.floats(8)[id], want);
    }
}
test "Gemma4 learning: binary64 intervals cover every primitive and meaningful nonfinite input" {
    const spec = core.living_gemma;
    for (std.enums.values(spec.Op)) |op| for ([_]bool{ false, true }) |nf| {
        var f = try data.Fixture(spec).init(a, op, data.params(spec, op, 4, 64), 0x3345, nf);
        defer f.deinit();
        try f.reference();
        try f.raw.run();
        for (f.values, 8..) |values, i| for (values, 0..) |v, j| {
            const got = f.output(i, j);
            try std.testing.expectEqual(core.contract.spec.classOf(v.value), core.contract.spec.classOf(got));
            if (std.math.isFinite(v.value)) try std.testing.expect(@abs(got - v.value) <= v.err);
        };
    };
}
