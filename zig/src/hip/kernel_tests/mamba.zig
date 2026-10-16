//! Nemotron's Mamba-2 pieces against float64 references on random inputs, and the same bits however a prompt is cut.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const mamba = @import("../ops/mamba.zig");
const Tensor = @import("../ops/types.zig").Tensor;

const ds = 128;
const kept = 3;
const sh: mamba.Shape = .{ .heads = 4, .dh = 64, .groups = 2 };
const lim: mamba.Limits = .{ .dt_min = 0.001, .dt_max = 0.1, .eps = 1e-5 };
const xd = sh.xd();
const cd = sh.cd();
const proj = sh.proj();
const heads = sh.heads;

/// Activations exact in the kind (z, x, B, C around a unit, dt logits around -3), fp32 weights and an fp32 SSM state.
const Inputs = struct {
    p: []u16,
    conv: []u16,
    ssm: []f32,
    cw: []f32,
    cb: []f32,
    dtb: []f32,
    alog: []f32,
    d: []f32,
    norm: []f32,

    fn make(t: *const Rig, rows: usize, seed: u64) !Inputs {
        var rng: rig.Rng = .{ .state = seed };
        const in: Inputs = .{
            .p = try gpa.alloc(u16, rows * proj),
            .conv = try gpa.alloc(u16, kept * cd),
            .ssm = try gpa.alloc(f32, heads * sh.dh * ds),
            .cw = try gpa.alloc(f32, 4 * cd),
            .cb = try gpa.alloc(f32, cd),
            .dtb = try gpa.alloc(f32, heads),
            .alog = try gpa.alloc(f32, heads),
            .d = try gpa.alloc(f32, heads),
            .norm = try gpa.alloc(f32, xd),
        };
        for (in.p, 0..) |*v, i| {
            const col = i % proj;
            v.* = if (col < 2 * xd) t.bits(rng.unit() * 2) else if (col < xd +
                cd) t.bits(rng.unit()) else t.bits(rng.unit() * 4 - 3);
        }
        for (in.conv) |*v| v.* = t.bits(rng.unit() * 2);
        for (in.ssm) |*v| v.* = @floatCast(0.1 * rng.normal());
        for (in.cw) |*v| v.* = @floatCast(0.5 * rng.normal());
        for (in.cb) |*v| v.* = @floatCast(0.1 * rng.normal());
        for (in.dtb) |*v| v.* = @floatCast(0.1 * rng.normal());
        for (in.alog) |*v| v.* = @floatCast(2.5 * rng.uniform());
        for (in.d) |*v| v.* = @floatCast(1 + 0.1 * rng.normal());
        for (in.norm) |*v| v.* = @floatCast(1 + 0.1 * rng.normal());
        return in;
    }

    fn free(in: Inputs) void {
        gpa.free(in.p);
        gpa.free(in.conv);
        for ([_][]f32{ in.ssm, in.cw, in.cb, in.dtb, in.alog, in.d, in.norm }) |a| gpa.free(a);
    }
};

/// The inputs, states and scratch on the device, and the mixer's weights and states as the launches take them.
const Dev = struct {
    p: DeviceBuffer,
    conv: DeviceBuffer,
    ssm: DeviceBuffer,
    xc: DeviceBuffer,
    dt: DeviceBuffer,
    y: DeviceBuffer,
    out: DeviceBuffer,
    cw: DeviceBuffer,
    cb: DeviceBuffer,
    dtb: DeviceBuffer,
    alog: DeviceBuffer,
    d: DeviceBuffer,
    norm: DeviceBuffer,

    fn make(t: *Rig, in: Inputs, rows: usize) !Dev {
        return .{
            .p = try t.upload(in.p),
            .conv = try t.upload(in.conv),
            .ssm = try t.upload(in.ssm),
            .xc = try t.alloc(rows * cd * 2),
            .dt = try t.alloc(rows * heads * 4),
            .y = try t.alloc(rows * xd * 2),
            .out = try t.alloc(rows * xd * 2),
            .cw = try t.upload(in.cw),
            .cb = try t.upload(in.cb),
            .dtb = try t.upload(in.dtb),
            .alog = try t.upload(in.alog),
            .d = try t.upload(in.d),
            .norm = try t.upload(in.norm),
        };
    }

    fn free(dv: *Dev) void {
        for ([_]*DeviceBuffer{
            &dv.p,
            &dv.conv,
            &dv.ssm,
            &dv.xc,
            &dv.dt,
            &dv.y,
            &dv.out,
            &dv.cw,
            &dv.cb,
            &dv.dtb,
            &dv.alog,
            &dv.d,
            &dv.norm,
        }) |b| b.free();
    }

    fn weights(dv: Dev) mamba.Weights {
        return .{
            .conv_w = rig.at(dv.cw),
            .conv_b = rig.at(dv.cb),
            .dt_bias = rig.at(dv.dtb),
            .a_log = rig.at(dv.alog),
            .d = rig.at(dv.d),
            .gnorm = rig.at(dv.norm),
        };
    }

    /// Both states back to the inputs' and every output zeroed.
    fn reset(dv: Dev, in: Inputs) !void {
        try dv.conv.upload(0, std.mem.sliceAsBytes(in.conv));
        try dv.ssm.upload(0, std.mem.sliceAsBytes(in.ssm));
        for ([_]DeviceBuffer{ dv.xc, dv.dt, dv.y, dv.out }) |b| try b.fill8(0);
    }

    /// One mixer call over `length` rows from row `from`, states in place.
    fn run(dv: Dev, t: *Rig, from: usize, length: usize) !void {
        const src: Tensor = .{ .ptr = rig.at(dv.p) + from * proj * 2, .kind = if (t.bf16) .bf16 else .f16 };
        const st: mamba.State = .{ .conv = rig.at(dv.conv), .ssm = rig.at(dv.ssm) };
        const sc: mamba.Scratch = .{
            .xc = rig.at(dv.xc) + from * cd * 2,
            .dt = rig.at(dv.dt) + from * heads * 4,
            .y = rig.at(dv.y) + from * xd * 2,
        };
        try t.ops(&t.on).mambaMixer(src, st, sc, rig.at(dv.out) + from * xd * 2, dv.weights(), length, sh, lim);
    }
};

/// Every output after a run, as bits and as values.
const Got = struct {
    xc: []u16,
    conv: []u16,
    dt: []f32,
    y: []u16,
    ssm: []f32,
    out: []u16,

    fn download(t: *Rig, dv: Dev) !Got {
        try t.stream.synchronize();
        return .{
            .xc = try rig.download(u16, dv.xc),
            .conv = try rig.download(u16, dv.conv),
            .dt = try rig.download(f32, dv.dt),
            .y = try rig.download(u16, dv.y),
            .ssm = try rig.download(f32, dv.ssm),
            .out = try rig.download(u16, dv.out),
        };
    }

    fn free(g: Got) void {
        gpa.free(g.xc);
        gpa.free(g.conv);
        gpa.free(g.dt);
        gpa.free(g.y);
        gpa.free(g.ssm);
        gpa.free(g.out);
    }

    fn expectSame(a: Got, b: Got) !void {
        try std.testing.expectEqualSlices(u16, a.xc, b.xc);
        try std.testing.expectEqualSlices(u16, a.conv, b.conv);
        try std.testing.expectEqualSlices(u32, @ptrCast(a.dt), @ptrCast(b.dt));
        try std.testing.expectEqualSlices(u16, a.y, b.y);
        try std.testing.expectEqualSlices(u32, @ptrCast(a.ssm), @ptrCast(b.ssm));
        try std.testing.expectEqualSlices(u16, a.out, b.out);
    }
};

/// The kind's values of `bits` in float64.
fn values(t: *const Rig, bits: []const u16) ![]f64 {
    const v = try gpa.alloc(f64, bits.len);
    for (v, bits) |*o, b| o.* = t.value(b);
    return v;
}

fn widen(x: []const f32) ![]f64 {
    const v = try gpa.alloc(f64, x.len);
    for (v, x) |*o, b| o.* = b;
    return v;
}

/// The largest error over the reference's largest value.
fn relative(got: []const f64, want: []const f64) f64 {
    var scale: f64 = 0;
    var max: f64 = 0;
    for (got, want) |g, w| {
        max = @max(max, @abs(g - w));
        scale = @max(scale, @abs(w));
    }
    return max / scale;
}

fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

/// The conv in float64: the taps from the state rows then P's x columns, silu; the state's next three raw inputs.
fn refConv(t: *const Rig, in: Inputs, rows: usize, xc: []f64, state: []f64) void {
    for (0..cd) |ch| {
        for (0..rows) |r| {
            var acc: f64 = in.cb[ch];
            for (0..4) |j| {
                const src = r + j;
                const tap = if (src < kept) t.value(in.conv[src * cd + ch]) else t.value(in.p[
                    (src - kept) *
                        proj + xd + ch
                ]);
                acc += @as(f64, in.cw[ch * 4 + j]) * tap;
            }
            xc[r * cd + ch] = silu(acc);
        }
        for (0..kept) |j| {
            const src = rows + j;
            state[j * cd + ch] = if (src < kept) t.value(in.conv[src * cd + ch]) else t.value(in.p[
                (src - kept) *
                    proj + xd + ch
            ]);
        }
    }
}

fn refDt(t: *const Rig, in: Inputs, rows: usize, dt: []f64) void {
    for (0..rows) |r| for (0..heads) |h| {
        const v = t.value(in.p[r * proj + xd + cd + h]) + @as(f64, in.dtb[h]);
        const sp = @max(v, 0) + std.math.log1p(@exp(-@abs(v)));
        dt[r * heads + h] = @min(@max(sp, @as(f64, lim.dt_min)), @as(f64, lim.dt_max));
    };
}

/// The recurrence in float64 from the conv's values and dt: y = silu(z) (x D + s C) and the final state.
fn refScan(t: *const Rig, in: Inputs, xc: []const f64, dt: []const f64, rows: usize, y: []f64, state: []f64) void {
    for (0..heads) |h| {
        const g = h / (heads / sh.groups);
        const a = -@exp(@as(f64, in.alog[h]));
        for (0..sh.dh) |d| {
            const st = state[(h * sh.dh + d) * ds ..][0..ds];
            for (st, in.ssm[(h * sh.dh + d) * ds ..][0..ds]) |*o, v| o.* = v;
            for (0..rows) |r| {
                const row = xc[r * cd ..][0..cd];
                const x = row[h * sh.dh + d];
                const da = @exp(a * dt[r * heads + h]);
                const xdt = x * dt[r * heads + h];
                var out: f64 = 0;
                for (st, 0..) |*s, n| {
                    s.* = s.* * da + xdt * row[xd + g * ds + n];
                    out += s.* * row[xd + sh.groups * ds + g * ds + n];
                }
                const z = t.value(in.p[r * proj + h * sh.dh + d]);
                y[r * xd + h * sh.dh + d] = silu(z) * (x * @as(f64, in.d[h]) + out);
            }
        }
    }
}

/// The group norm in float64 over y's values.
fn refGroupNorm(in: Inputs, y: []const f64, rows: usize, out: []f64) void {
    const gs = xd / sh.groups;
    for (0..rows) |r| for (0..sh.groups) |g| {
        const v = y[r * xd + g * gs ..][0..gs];
        var ss: f64 = 0;
        for (v) |x| ss += x * x;
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(gs)) + @as(f64, lim.eps));
        for (v, 0..) |x, i| out[r * xd + g * gs + i] = @as(f64, in.norm[g * gs + i]) * (x * inv);
    };
}

test "each Mamba-2 piece stays within its roundings of a float64 reference at four lengths" {
    const t = try Rig.open(1 << 30);
    defer t.close();
    for ([_]usize{ 64, 200, 1000, 4170 }) |rows| {
        const in = try Inputs.make(t, rows, 0x6d616d6261 + rows);
        defer in.free();
        var dv = try Dev.make(t, in, rows);
        defer dv.free();
        try dv.run(t, 0, rows);
        const got = try Got.download(t, dv);
        defer got.free();
        const xc = try gpa.alloc(f64, rows * cd);
        defer gpa.free(xc);
        const conv = try gpa.alloc(f64, kept * cd);
        defer gpa.free(conv);
        refConv(t, in, rows, xc, conv);
        const gxc = try values(t, got.xc);
        defer gpa.free(gxc);
        const gconv = try values(t, got.conv);
        defer gpa.free(gconv);
        try std.testing.expect(relative(gxc, xc) < 1e-2);
        try std.testing.expectEqualSlices(f64, conv, gconv);
        const dt = try gpa.alloc(f64, rows * heads);
        defer gpa.free(dt);
        refDt(t, in, rows, dt);
        const gdt = try widen(got.dt);
        defer gpa.free(gdt);
        try std.testing.expect(relative(gdt, dt) < 1e-5);
        // the scan and the norm take the kernels' own upstream values, so each piece's error is its own
        const y = try gpa.alloc(f64, rows * xd);
        defer gpa.free(y);
        const ssm = try gpa.alloc(f64, heads * sh.dh * ds);
        defer gpa.free(ssm);
        refScan(t, in, gxc, gdt, rows, y, ssm);
        const gy = try values(t, got.y);
        defer gpa.free(gy);
        const gssm = try widen(got.ssm);
        defer gpa.free(gssm);
        try std.testing.expect(relative(gy, y) < 1e-2);
        try std.testing.expect(relative(gssm, ssm) < 1e-5);
        const out = try gpa.alloc(f64, rows * xd);
        defer gpa.free(out);
        refGroupNorm(in, gy, rows, out);
        const gout = try values(t, got.out);
        defer gpa.free(gout);
        try std.testing.expect(relative(gout, out) < 1e-2);
    }
}

test "a Mamba-2 prompt cut on either side of 64 rows gives the same rows and states as one call" {
    const t = try Rig.open(1 << 30);
    defer t.close();
    const rows = 200;
    const in = try Inputs.make(t, rows, 0x637574);
    defer in.free();
    var dv = try Dev.make(t, in, rows);
    defer dv.free();
    try dv.run(t, 0, rows);
    const whole = try Got.download(t, dv);
    defer whole.free();
    for ([_]usize{ 1, 63, 64, 65, 136 }) |c| {
        try dv.reset(in);
        try dv.run(t, 0, c);
        try dv.run(t, c, rows - c);
        const got = try Got.download(t, dv);
        defer got.free();
        try whole.expectSame(got);
    }
}

test "Mamba-2 decode steps after a prompt give the prompt path's bits for those rows" {
    const t = try Rig.open(1 << 30);
    defer t.close();
    const rows = 96;
    const in = try Inputs.make(t, rows, 0x6465636f6465);
    defer in.free();
    var dv = try Dev.make(t, in, rows);
    defer dv.free();
    try dv.run(t, 0, rows);
    const whole = try Got.download(t, dv);
    defer whole.free();
    try dv.reset(in);
    try dv.run(t, 0, 80);
    for (80..rows) |r| try dv.run(t, r, 1);
    const got = try Got.download(t, dv);
    defer got.free();
    try whole.expectSame(got);
}
