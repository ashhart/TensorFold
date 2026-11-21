//! Draft RNN affine, gate cell, folded bias, tap softmax and SwiGLU on the production BF16 boundary.
const std = @import("std");
const mtl = @import("metal");
const u = @import("inkling2_util.zig");
const r = @import("rig.zig");
const stress = @import("metal_inkling2_media_stress.zig");
const sum = @import("contract").rows.sumIn;
const c = u.c;
const g = u.g;

pub const Kind = enum { rnn, cell, fold_b, alpha, swiglu };
pub fn vision(_: Kind) bool {
    return false;
}
pub fn name(comptime k: Kind) []const u8 {
    return "inkling2 draft ik2d_" ++ @tagName(k);
}

pub const controls = .{
    .{ Kind.rnn, "inkling2 draft RNN control drops Markov bias", "raw[o] = bfloat(s + float(bias[o]));", "raw[o] = bfloat(s);", c.Gate.shapes },
    .{ Kind.cell, "inkling2 draft cell control omits recurrent blend", "state[j] = bfloat(keep + add);", "state[j] = bfloat(keep);", c.Gate.shapes },
    .{ Kind.swiglu, "inkling2 draft SwiGLU control loses up product", "* float(up[i]));", ");", c.Gate.shapes },
    .{
        Kind.fold_b,                                                                                                               "inkling2 draft fold control uses FP16 accumulation",
        "float s = 0.0f;\n  for (uint k = lane; k < p.width; k += 32) s += float(unembed[ulong(v) * p.width + k]) * float(b[k]);", "half s = 0.0h;\n  for (uint k = lane; k < p.width; k += 32) s = half(float(s) + float(unembed[ulong(v) * p.width + k]) * float(b[k]));",
        c.Gate.sums,
    },
};

const Rnn = extern struct { width: u32, rank: u32, step: u32 };
const rnn_shapes = [_]c.Shape{ .{ .rows = 8, .n = 256, .k = 2304 }, .{ .rows = 1, .n = 256, .k = 2304 }, .{ .rows = 8, .n = 33, .k = 129 } };
const cell_shapes = [_]c.Shape{ .{ .rows = 8, .n = 256, .k = 0 }, .{ .rows = 1, .n = 256, .k = 0 }, .{ .rows = 8, .n = 65, .k = 0 } };
const fold_shapes = [_]c.Shape{ .{ .rows = 9, .n = 1, .k = 2304 }, .{ .rows = 8, .n = 1, .k = 2304 } };
const alpha_shapes = [_]c.Shape{ .{ .rows = 4, .n = 4, .k = 4 }, .{ .rows = 4, .n = 5, .k = 5 } };
const swiglu_shapes = [_]c.Shape{ .{ .rows = 8, .n = 9216, .k = 0 }, .{ .rows = 1, .n = 257, .k = 0 } };

fn depth(s: c.Shape) usize {
    return 2 * s.k + 2 * s.n + 10;
}
fn noDepth(_: c.Shape) usize {
    return 0;
}

pub const Probe = struct {
    rig: r.Rig,
    arena: std.heap.ArenaAllocator,
    pipe: mtl.Pipeline,
    kind: Kind,
    label: []const u8,
    s: c.Shape = undefined,
    x: []const u16 = &.{},
    w: []const u16 = &.{},
    bias: []const u16 = &.{},
    refs: [8]r.Ref = undefined,
    out: r.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, k: Kind, label: []const u8) !Probe {
        const pipe = switch (k) {
            inline else => |tag| try mtl.Pipeline.init(device, lib, "ik2d_" ++ @tagName(tag), false),
        };
        return .{ .rig = .{ .a = a, .device = device, .queue = queue }, .arena = .init(a), .pipe = pipe, .kind = k, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        p.pipe.deinit();
        p.arena.deinit();
        p.rig.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe, gate: ?c.Gate) g.Kernel {
        const reducing = p.kind != .swiglu and p.kind != .cell;
        return .{ .ctx = p, .vt = &vt, .control = gate, .spec = .{
            .name = p.label,
            .accum = if (reducing) .f32 else .none,
            .depth = if (reducing) depth else noDepth,
            .ops = 12,
            .out = .bf16,
            .in_place = p.kind == .cell,
            .sums = if (p.kind == .alpha) .{ .f16 = 0, .bf16 = 0, .why = "four positive terms end in one BF16 store" } else .{},
            .served = switch (p.kind) {
                .rnn => &rnn_shapes,
                .cell => &cell_shapes,
                .fold_b => &fold_shapes,
                .alpha => &alpha_shapes,
                .swiglu => &swiglu_shapes,
            },
        } };
    }

    fn dotInputs(p: *Probe, a: std.mem.Allocator, rng: std.Random, fill: g.Fill) !void {
        const k = if (p.kind == .rnn) 2 * (p.s.n + p.s.k) else p.s.k;
        const n = p.count();
        const x = try a.alloc(u16, k);
        const w = try a.alloc(u16, n * k);
        const bias = try a.alloc(u16, n);
        u.fillDyadic(rng, x, 100, 128);
        u.fillDyadic(rng, w, 100, 128);
        if (p.kind != .rnn or fill == .stress) @memset(bias, 0) else u.fillDyadic(rng, bias, 96, 64);
        if (fill == .stress) {
            for (x, 0..) |*v, j| v.* = u.word(stress.pair(j, k).left);
            for (w, 0..) |*v, j| v.* = u.word(stress.pair(j % k, k).right);
        } else if (fill == .nonfinite) {
            w[0] = u.nan_bits;
            w[k + 1] = u.pinf_bits;
            w[2 * k + 2] = u.ninf_bits;
            w[3 * k] = u.pinf_bits;
            w[3 * k + 3] = u.ninf_bits;
        }
        p.x = x;
        p.w = w;
        p.bias = bias;
        if (p.kind == .rnn) {
            const rank = p.s.n;
            p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(x[0..rank]));
            p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(x[rank .. 2 * rank]));
            p.refs[2] = try p.rig.input(std.mem.asBytes(&@as(u64, 0)));
            p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(x[2 * rank ..][0..p.s.k]));
            p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(x[2 * rank + p.s.k ..]));
            p.refs[5] = try p.rig.input(std.mem.sliceAsBytes(w));
            p.refs[6] = try p.rig.input(std.mem.sliceAsBytes(bias));
        } else {
            p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(w));
            p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(x));
        }
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = u.reset(&p.arena, &p.rig);
        p.s = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (p.kind == .rnn or p.kind == .fold_b) {
            try p.dotInputs(a, rng, fill);
        } else {
            const x = try a.alloc(u16, if (p.kind == .cell) 3 * s.n else s.rows * s.n);
            u.fillDyadic(rng, x, 255, 64);
            if (fill == .stress) @memset(x, 0);
            if (fill == .nonfinite) {
                x[0] = u.nan_bits;
                x[s.n + 1] = u.pinf_bits;
                x[2 * s.n + 2] = u.ninf_bits;
            }
            p.x = x;
            p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(x));
            if (p.kind == .cell or p.kind == .swiglu) {
                const w = try a.alloc(u16, if (p.kind == .cell) s.n else x.len);
                u.fillDyadic(rng, w, 127, 128);
                p.w = w;
                p.refs[1] = if (p.kind == .cell) try p.rig.output(w.len * 2) else try p.rig.input(std.mem.sliceAsBytes(w));
            }
        }
        p.out = try p.rig.output((p.count() - if (p.kind == .cell) s.n else @as(usize, 0)) * 2);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const rank: u32 = @intCast(p.s.n);
        const width: u32 = @intCast(p.s.k);
        const depth_index: u32 = @intFromBool(p.s.rows != 1);
        const inputs: usize = switch (p.kind) {
            .rnn => 7,
            .cell, .fold_b, .swiglu => 2,
            .alpha => 1,
        };
        e.setPipeline(p.pipe);
        for (p.refs[0..inputs], 0..) |v, i| e.setBuffer(v.buf, v.off, i);
        e.setBuffer(p.out.buf, p.out.off, inputs);
        if (p.kind == .cell) {
            const words = std.mem.sliceAsBytes(p.w);
            @memcpy(p.rig.bytes(p.refs[1], words.len), words);
        }
        switch (p.kind) {
            .rnn, .cell => e.setValue(Rnn{ .width = width, .rank = rank, .step = depth_index }, inputs + 1),
            .fold_b => e.setValue([2]u32{ @intCast(p.s.rows), width }, inputs + 1),
            .alpha => e.setValue([2]u32{ @intCast(p.s.rows), rank }, inputs + 1),
            .swiglu => e.setValue(rank, inputs + 1),
        }
        switch (p.kind) {
            .rnn => e.dispatchGroups(mtl.Size.of(3 * rank, 1, 1), mtl.Size.of(256, 1, 1)),
            .cell => e.dispatchThreads(mtl.Size.of(rank, 1, 1), mtl.Size.of(64, 1, 1)),
            .fold_b => e.dispatchGroups(mtl.Size.of((p.s.rows + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1)),
            .alpha => e.dispatchThreads(mtl.Size.of(p.s.rows, 1, 1), mtl.Size.of(64, 1, 1)),
            .swiglu => e.dispatchThreads(mtl.Size.of(rank, p.s.rows, 1), mtl.Size.of(256, 1, 1)),
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return switch (p.kind) {
            .rnn => 3 * p.s.n,
            .cell => 2 * p.s.n,
            .fold_b => p.s.rows,
            .alpha, .swiglu => p.s.rows * p.s.n,
        };
    }

    fn calculate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) !void {
        if (mag) |m| @memset(m, 0);
        const rank = p.s.n;
        switch (p.kind) {
            .rnn, .fold_b => {
                const k = p.x.len;
                const terms = try p.rig.a.alloc(f64, k);
                defer p.rig.a.free(terms);
                for (out, 0..) |*y, i| {
                    var abs: f64 = 0;
                    for (terms, 0..) |*t, j| {
                        const x = if (p.kind == .rnn and p.s.rows == 1 and j < rank) 0 else u.val(p.x[j]);
                        t.* = x * u.val(p.w[i * k + j]);
                        abs += @abs(t.*);
                    }
                    y.* = sum(acc, order, terms) + u.val(p.bias[i]);
                    if (mag) |m| m[i] = abs;
                }
            },
            .cell => for (0..rank) |i| {
                const old = if (p.s.rows == 1) 0 else u.val(p.w[i]);
                const gate = u.bf(1 / (1 + @exp(-u.val(p.x[i]))));
                const cand = u.bf(std.math.tanh(u.val(p.x[rank + i])));
                out[i] = u.bf(gate * old) + u.bf(u.bf(1 - gate) * cand);
                out[rank + i] = std.math.tanh(u.val(p.x[2 * rank + i]));
            },
            .alpha => {
                var terms: [5]f64 = undefined;
                for (0..p.s.rows) |i| {
                    const x = p.x[i * rank ..][0..rank];
                    var top = -std.math.inf(f64);
                    for (x) |v| top = @max(top, u.val(v));
                    for (terms[0..rank], x) |*t, v| t.* = @exp(u.val(v) - top);
                    const den = sum(acc, order, terms[0..rank]);
                    for (terms[0..rank], 0..) |t, j| {
                        const y = t / den;
                        out[i * rank + j] = y;
                        if (mag) |m| m[i * rank + j] = @abs(y);
                    }
                }
            },
            .swiglu => for (out, p.x, p.w) |*y, v, w| {
                const x = u.val(v);
                y.* = x / (1 + @exp(-x)) * u.val(w);
            },
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try p.calculate(.none, .sequential, want, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.calculate(acc, order, out, null);
    }
};
