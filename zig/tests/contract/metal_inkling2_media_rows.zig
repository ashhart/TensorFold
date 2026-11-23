//! The tower's row reductions and the draft's BF16 embedding, residual and four-tap normalization paths.
const std = @import("std");
const mtl = @import("metal");
const rows = @import("contract").rows;
const u = @import("inkling2_util.zig");
const r = @import("rig.zig");
const c = u.c;
const g = u.g;

pub const Kind = enum { layer_norm, softmax, embed_block, rms, add_rms, mix_rms };

pub fn kernelName(comptime k: Kind) []const u8 {
    return comptime (if (vision(k)) "ik2v_" else "ik2d_") ++ @tagName(k);
}

pub fn vision(k: Kind) bool {
    return k == .layer_norm or k == .softmax;
}

pub fn name(comptime k: Kind) []const u8 {
    return comptime (if (vision(k)) "inkling2 vision " else "inkling2 draft ") ++ kernelName(k);
}

pub const controls = .{
    .{ Kind.layer_norm, "inkling2 vision layer norm control repeats channels", "#define IK2V_ROW(i) float(xr[i])", "#define IK2V_ROW(i) float(xr[(i) % 2])", c.Gate.shapes },
    .{ Kind.softmax, "inkling2 vision softmax control ignores NaN exp", "const float e = metal::precise::exp(float(s[i]) - m);", "const float e = isnan(float(s[i]) - m) ? 0.0f : metal::precise::exp(float(s[i]) - m);", c.Gate.nonfinite },
    .{ Kind.mix_rms, "inkling2 draft mix_rms control repeats first tap", "float(fr[ulong(k) * p.width + i]) * float(alpha[k]);\n    s += m * m;", "float(fr[i]) * float(alpha[k]);\n    s += m * m;", c.Gate.shapes },
    .{ Kind.embed_block, "inkling2 draft embed_block control drops the norm weight", "r[i] = bfloat(float(embed_w[i]) * (float(e[i]) * inv));", "r[i] = bfloat(float(e[i]) * inv);", c.Gate.shapes },
    .{ Kind.rms, "inkling2 draft rms control drops the norm weight", "yr[i] = bfloat(float(xr[i]) * inv * float(w[i]));", "yr[i] = bfloat(float(xr[i]) * inv);", c.Gate.shapes },
    .{ Kind.add_rms, "inkling2 draft add_rms control stores the residual without the delta", "rr[i] = bfloat(v);", "rr[i] = bfloat(float(rr[i]));", c.Gate.shapes },
};

const Rows = extern struct { rows: u32, width: u32, eps: f32, pitch: u32 = 0, taps: u32 = 0 };
const vision_shapes = [_]c.Shape{ .{ .rows = 9, .n = 1024, .k = 1024 }, .{ .rows = 5, .n = 1024, .k = 1024 } };
const softmax_shapes = [_]c.Shape{ .{ .rows = 9, .n = 1025, .k = 1025 }, .{ .rows = 5, .n = 5, .k = 5 }, .{ .rows = 9, .n = 16385, .k = 16385 } };
const draft_shapes = [_]c.Shape{ .{ .rows = 8, .n = 2304, .k = 2304 }, .{ .rows = 1, .n = 2304, .k = 2304 } };
const mix_shapes = draft_shapes ++ [_]c.Shape{.{ .rows = 33, .n = 2304, .k = 2304 }};

fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 12;
}

pub const Probe = struct {
    rig: r.Rig,
    pipe: mtl.Pipeline,
    arena: std.heap.ArenaAllocator,
    kind: Kind,
    label: []const u8,
    s: c.Shape = undefined,
    words: []const u16 = &.{},
    res: []const u16 = &.{},
    mask: []const u16 = &.{},
    weights: []const u16 = &.{},
    bias: []const u16 = &.{},
    refs: [5]r.Ref = undefined,
    out: r.Ref = undefined,
    stress: bool = false,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, k: Kind, label: []const u8) !Probe {
        const pipe = switch (k) {
            inline else => |tag| try mtl.Pipeline.init(device, lib, kernelName(tag), false),
        };
        return .{ .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .arena = .init(a), .kind = k, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.arena.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe, control_gate: ?c.Gate) g.Kernel {
        const soft = p.kind == .softmax;
        return .{ .ctx = p, .vt = &vt, .control = control_gate, .spec = .{
            .name = p.label,
            .accum = .f32,
            .depth = depth,
            .ops = 12,
            .in_place = soft or p.kind == .add_rms,
            .out = .bf16,
            .tiles = .{ .{}, .{}, .{ .size = 256 } },
            .served = switch (p.kind) {
                .layer_norm => &vision_shapes,
                .softmax => &softmax_shapes,
                .mix_rms => &mix_shapes,
                else => &draft_shapes,
            },
            .sums = .{ .f16 = if (soft) 0 else 100, .bf16 = 0, .exact = !soft, .why = "BF16 store limits discrimination in normalized outputs; exp terms are not dyadic" },
        } };
    }

    const alpha = [_]u16{ 0x3f00, 0x3e80, 0x3e00, 0x3e00 };

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = u.reset(&p.arena, &p.rig);
        p.s = s;
        p.stress = fill == .stress;
        if (p.kind == .softmax) {
            const x = try a.alloc(u16, s.rows * s.k);
            var prng = std.Random.DefaultPrng.init(seed);
            for (x) |*v| v.* = u.word(prng.random().float(f32) * 16 - 8);
            if (fill == .nonfinite) {
                x[0] = u.nan_bits;
                x[s.k + s.k / 2] = u.pinf_bits;
                x[2 * s.k] = u.ninf_bits;
                x[3 * s.k] = u.pinf_bits;
                x[4 * s.k - 1] = u.ninf_bits;
            }
            p.words = x;
            p.out = try p.rig.output(x.len * 2);
            return true;
        }
        const base = try rows.Rows.init(a, if (p.kind == .layer_norm) .layer_norm else .rms, s);
        base.fill(fill, seed);
        p.words = base.x;
        p.weights = base.w;
        p.bias = base.b;
        p.out = try p.rig.output(s.rows * s.k * 2);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x929a);
        switch (p.kind) {
            .layer_norm => {
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.words));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.weights));
                p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.bias));
            },
            .embed_block => {
                if (fill == .nonfinite and s.rows == 1) return false;
                p.refs[0] = try p.rig.input(std.mem.asBytes(&@as(u32, 1)));
                const table = try a.alloc(u16, 2 * s.k);
                @memset(table[0..s.k], u.nan_bits);
                @memcpy(table[s.k..], p.words[0..s.k]);
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(table));
                p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.weights));
                const mask = try a.alloc(u16, s.k);
                u.fillDyadic(prng.random(), mask, 63, 32);
                p.mask = mask;
                p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(mask));
            },
            .rms => {
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.words));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.weights));
            },
            .add_rms => {
                const old = try a.alloc(u16, s.rows * s.k);
                if (p.stress) @memset(old, 0) else u.fillDyadic(prng.random(), old, 112, 64);
                p.res = old;
                p.refs[0] = try p.rig.output(old.len * 2);
                // The rig gathers y first, then res; both retain guard buffers and are compared.
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.words));
                p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.weights));
            },
            .mix_rms => {
                const taps = try a.alloc(u16, s.rows * s.k * alpha.len);
                for (0..s.rows) |i| for (0..alpha.len) |j| {
                    const dest = taps[(i * alpha.len + j) * s.k ..][0..s.k];
                    if (j == 0 or p.stress) @memcpy(dest, p.words[i * s.k ..][0..s.k]) else u.fillDyadic(prng.random(), dest, 63, 32);
                };
                p.words = taps;
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(taps));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(&alpha));
                p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.weights));
            },
            .softmax => unreachable,
        }
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const w: u32 = @intCast(p.s.k);
        const n: u32 = @intCast(p.s.rows);
        e.setPipeline(p.pipe);
        if (p.kind == .softmax) {
            const data = std.mem.sliceAsBytes(p.words);
            @memcpy(p.rig.bytes(p.out, data.len), data);
            e.setBuffer(p.out.buf, p.out.off, 0);
            e.setValue(w, 1);
        } else {
            if (p.kind == .add_rms) {
                const src = std.mem.sliceAsBytes(p.res);
                @memcpy(p.rig.bytes(p.refs[0], src.len), src);
            }
            const inputs: usize = switch (p.kind) {
                .rms => 2,
                .embed_block => 4,
                else => 3,
            };
            for (p.refs[0..inputs], 0..) |x, i| e.setBuffer(x.buf, x.off, i);
            e.setBuffer(p.out.buf, p.out.off, inputs);
            const eps: f32 = if (p.kind == .layer_norm) 1e-5 else 1e-6;
            if (p.kind == .layer_norm) {
                e.setValue([3]u32{ n, w, @bitCast(eps) }, inputs + 1);
            } else {
                const taps: u32 = alpha.len;
                e.setValue(Rows{ .rows = n, .width = w, .eps = eps, .pitch = w * taps, .taps = taps }, inputs + 1);
            }
        }
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.s.rows * p.s.k * (if (p.kind == .add_rms) @as(usize, 2) else 1);
    }

    fn entry(p: *Probe, i: usize, j: usize) f64 {
        const w = p.s.k;
        if (p.kind == .mix_rms) {
            var mixed: f64 = 0;
            for (alpha, 0..) |a, k| mixed += u.val(p.words[(i * alpha.len + k) * w + j]) * u.val(a);
            return mixed;
        }
        return u.val(p.words[i * w + j]) + if (p.kind == .add_rms) u.val(p.res[i * w + j]) else 0;
    }

    fn roundSum(v: []const f64, acc: c.Accum, order: g.Order) f64 {
        return rows.sumIn(acc, order, v);
    }

    fn calculate(p: *Probe, acc: c.Accum, order: g.Order, want: []f64, mags: ?[]f64) !void {
        const k = p.s.k;
        const kf: f64 = @floatFromInt(k);
        const a = p.rig.a;
        const terms = try a.alloc(f64, k);
        defer a.free(terms);
        for (0..p.s.rows) |i| {
            const out = want[i * k ..][0..k];
            const mag = if (mags) |m| m[i * k ..][0..k] else null;
            if (p.kind == .softmax) {
                var top = -std.math.inf(f64);
                const src = p.words[i * k ..][0..k];
                for (terms, src) |*x, v| {
                    x.* = u.val(v);
                    top = @max(top, x.*);
                }
                for (terms) |*x| x.* = @exp(x.* - top);
                const z = roundSum(terms, acc, order);
                for (out, terms) |*y, x| y.* = x / z;
                if (mag) |m| for (m, out) |*x, y| {
                    x.* = @abs(y);
                };
                continue;
            }
            if (p.kind == .embed_block and i > 0) {
                for (out, p.mask) |*y, x| y.* = u.val(x);
                if (mag) |m| @memset(m, 0);
                continue;
            }
            var abs: f64 = 0;
            for (terms, 0..) |*x, j| {
                x.* = p.entry(i, j);
                abs += @abs(x.*);
            }
            const ln = p.kind == .layer_norm;
            const mean = if (ln) roundSum(terms, acc, order) / kf else 0;
            for (terms) |*x| x.* = (x.* - mean) * (x.* - mean);
            const eps: f64 = if (ln) 1e-5 else 1e-6;
            const inv = 1 / @sqrt(roundSum(terms, acc, order) / kf + eps);
            for (out, 0..) |*y, j| {
                const x = p.entry(i, j) - mean;
                const w = u.val(p.weights[j]);
                const b = if (ln) u.val(p.bias[j]) else 0;
                y.* = x * inv * w + b;
                if (mag) |m| m[j] = @abs(w) * inv * (@abs(x) + if (ln) abs / kf else 0) + @abs(b);
                if (p.kind == .add_rms) {
                    const j2 = p.s.rows * k + i * k + j;
                    want[j2] = p.entry(i, j);
                    if (mags) |m| m[j2] = 0;
                }
            }
        }
    }

    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        // Affine cancellation can make |y| much smaller than the terms whose rounding influences it.
        const work = if (p.stress and p.kind != .softmax) 12 else depth(p.s) + 16;
        const delta = c.gamma(@floatFromInt(work), c.unit(.f32));
        for (want, mag, allow) |y, m, *b| {
            const error_f32 = m * delta;
            b.* = error_f32 + (error_f32 + @abs(y)) * c.unit(.bf16) + 0x1p-133;
        }
    }

    pub fn reference(p: *Probe, out: []f64, mag: []f64) !void {
        try p.calculate(.none, .sequential, out, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.calculate(acc, order, out, null);
    }
};
