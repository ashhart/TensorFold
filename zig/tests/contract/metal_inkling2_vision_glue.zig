//! Image tower glue: patch/position embedding, head layout and rotary, and BF16 epilogue roundings.
const std = @import("std");
const mtl = @import("metal");
const u = @import("metal_inkling2_vision_util.zig");
const r = @import("rig.zig");
const math = @import("contract").rows;
const c = u.c;
const g = u.g;

pub const Kind = enum { pack, embed, qkv, heads, bias_gelu, gelu_rows, residual };
pub fn name(comptime k: Kind) []const u8 {
    return "metal inkling2 vision ik2v_" ++ @tagName(k);
}

const pack_shapes = [_]c.Shape{ .{ .rows = 8, .n = 1200, .k = 0 }, .{ .rows = 4, .n = 1200, .k = 0 } };
const base_shapes = [_]c.Shape{ .{ .rows = 9, .n = 1024, .k = 0 }, .{ .rows = 4, .n = 1024, .k = 0 } };
const mlp_shapes = [_]c.Shape{ .{ .rows = 9, .n = 4096, .k = 0 }, .{ .rows = 1, .n = 4096, .k = 0 }, .{ .rows = 5, .n = 257, .k = 0 } };
const dim = 64;
const r0 = 2;
const Embed = extern struct { patches: u32, width: u32 };
const Qkv = extern struct { rows: u32, heads: u32, dim: u32, r0: u32 };
const Heads = extern struct { rows: u32, r0: u32, heads: u32, dim: u32 };
fn noDepth(_: c.Shape) usize {
    return 0;
}
fn four(_: c.Shape) usize {
    return 4;
}

pub const Probe = struct {
    rig: r.Rig,
    arena: std.heap.ArenaAllocator,
    pipe: mtl.Pipeline,
    kind: Kind,
    label: []const u8,
    s: c.Shape = undefined,
    refs: [9]r.Ref = undefined,
    nrefs: usize = 0,
    clear: []const r.Ref = &.{},
    restore: []const u16 = &.{},
    want: []f64 = &.{},
    mag: []f64 = &.{},
    pe: []const u16 = &.{},
    table: []const u16 = &.{},
    taps: []const u32 = &.{},
    tap_weights: []const f32 = &.{},

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, k: Kind, label: []const u8) !Probe {
        const pipe = switch (k) {
            inline else => |tag| try mtl.Pipeline.init(device, lib, "ik2v_" ++ @tagName(tag), false),
        };
        return .{ .rig = .{ .a = a, .device = device, .queue = queue }, .arena = .init(a), .pipe = pipe, .kind = k, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.arena.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe, gate: ?c.Gate) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = gate, .spec = .{
            .name = p.label,
            .accum = if (p.kind == .embed) .f32 else .none,
            .depth = if (p.kind == .embed) four else noDepth,
            .out = if (p.kind == .qkv) .f32 else .bf16,
            .store = if (p.kind == .qkv) .bf16 else null,
            .in_place = p.kind == .gelu_rows or p.kind == .residual,
            .served = switch (p.kind) {
                .pack => &pack_shapes,
                .bias_gelu, .gelu_rows => &mlp_shapes,
                else => &base_shapes,
            },
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "four dyadic taps into a rounded BF16 position row" },
        } };
    }

    fn add(p: *Probe, v: r.Ref) void {
        p.refs[p.nrefs] = v;
        p.nrefs += 1;
    }
    fn input(p: *Probe, v: anytype) !void {
        p.add(try p.rig.input(std.mem.sliceAsBytes(v)));
    }
    fn output(p: *Probe, n: usize) !void {
        p.add(try p.rig.output(n));
    }

    fn bf(a: std.mem.Allocator, rng: std.Random, n: usize, nf: bool) ![]u16 {
        const v = try a.alloc(u16, n);
        u.fillDyadic(rng, v, 127, 64);
        if (nf) @memcpy(v[0..3], &[_]u16{ u.nan_bits, u.pinf_bits, u.ninf_bits });
        return v;
    }

    fn fp(a: std.mem.Allocator, rng: std.Random, n: usize, nf: bool) ![]f32 {
        const v = try a.alloc(f32, n);
        for (v) |*x| {
            x.* = rng.float(f32) * 4 - 2;
        }
        if (nf) @memcpy(v[0..3], &[_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) });
        return v;
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = u.reset(&p.arena, &p.rig);
        p.s = s;
        p.nrefs = 0;
        p.clear = &.{};
        p.restore = &.{};
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = s.rows * s.n;
        const nf = fill == .nonfinite;
        p.want = try a.alloc(f64, switch (p.kind) {
            .embed => n + s.n,
            .qkv => 3 * (s.rows + r0) * s.n,
            .heads => (s.rows + r0) * s.n,
            else => n,
        });
        p.mag = try a.alloc(f64, p.want.len);
        @memset(p.want, 0);
        @memset(p.mag, 0);
        switch (p.kind) {
            .embed => {
                const pe = try bf(a, rng, n, nf);
                const cls = try bf(a, rng, s.n, nf);
                const cpos = try bf(a, rng, s.n, false);
                const table = try bf(a, rng, s.n * 1024, false);
                const taps = try a.alloc(u32, s.rows * 4);
                const weights = try a.alloc(f32, s.rows * 4);
                for (taps, weights) |*i, *w| {
                    i.* = rng.uintLessThan(u32, 1024);
                    w.* = if (fill == .stress) 0.25 else @as(f32, @floatFromInt(rng.uintLessThan(u32, 32))) / 32;
                }
                try p.input(pe);
                try p.input(table);
                try p.input(taps);
                try p.input(weights);
                try p.input(cls);
                try p.input(cpos);
                try p.output(p.want.len * 2);
                for (0..s.n) |i| {
                    p.want[i] = u.val(cls[i]) + u.val(cpos[i]);
                }
                p.pe = pe;
                p.table = table;
                p.taps = taps;
                p.tap_weights = weights;
                p.embedRows(.none, .sequential, p.want);
            },
            .qkv => {
                const span = (s.rows + r0) * s.n;
                const acc = try fp(a, rng, 3 * n, nf);
                try p.input(acc);
                var biases: [3][]u16 = undefined;
                for (&biases) |*b| {
                    b.* = try bf(a, rng, s.n, false);
                    try p.input(b.*);
                }
                const cs = try a.alloc(f32, (s.rows + r0) * dim);
                const sn = try a.alloc(f32, (s.rows + r0) * dim);
                for (cs, sn, 0..) |*co, *si, i| {
                    co.* = if ((i / dim) % 2 == 0) 0.75 else 0.25;
                    si.* = if (i % 4 < 2) 0.5 else -0.5;
                }
                try p.input(cs);
                try p.input(sn);
                for (0..3) |_| try p.output(span * 4);
                p.clear = p.refs[6..9];
                for (0..3) |k| for (0..s.rows) |i| for (0..s.n) |j| {
                    const row = (i * 3 + k) * s.n;
                    const v = u.bf(@as(f64, acc[row + j]) + u.val(biases[k][j]));
                    const partner = u.bf(@as(f64, acc[row + (j ^ 1)]) + u.val(biases[k][j ^ 1]));
                    const trig = (i + r0) * dim + j % dim;
                    const o = k * span + ((j / dim) * (s.rows + r0) + i + r0) * dim + j % dim;
                    p.want[o] = if (k == 2) v else v * cs[trig] + (if (j % 2 == 0) -partner else partner) * sn[trig];
                };
            },
            .heads => {
                const x = try fp(a, rng, n, nf);
                try p.input(x);
                try p.output(p.want.len * 2);
                p.clear = p.refs[1..2];
                for (0..s.rows) |i| for (0..s.n) |j| {
                    p.want[(i + r0) * s.n + j] = x[((j / dim) * s.rows + i) * dim + j % dim];
                };
            },
            .pack, .bias_gelu, .gelu_rows, .residual => {
                const x = if (p.kind != .gelu_rows) try fp(a, rng, n, nf) else &.{};
                const old = if (p.kind == .gelu_rows or p.kind == .residual) try bf(a, rng, n, nf) else &.{};
                const b = if (p.kind == .residual or p.kind == .bias_gelu) try bf(a, rng, s.n, false) else &.{};
                const w = if (p.kind == .residual) try bf(a, rng, s.n, false) else &.{};
                if (x.len > 0) try p.input(x);
                if (b.len > 0) try p.input(b);
                if (w.len > 0) try p.input(w);
                try p.output(n * 2);
                p.restore = old;
                for (p.want, p.mag, 0..) |*y, *m, i| {
                    const j = i % s.n;
                    switch (p.kind) {
                        .pack => y.* = x[i],
                        .residual => y.* = u.val(old[i]) + u.bf(u.val(w[j]) * u.bf(@as(f64, x[i]) + u.val(b[j]))),
                        .bias_gelu, .gelu_rows => {
                            const z = if (p.kind == .gelu_rows) u.val(old[i]) else u.bf(@as(f64, x[i]) + u.val(b[j]));
                            y.* = math.gelu(z);
                            m.* = @abs(z);
                        },
                        else => unreachable,
                    }
                }
            },
        }
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.refs[0..p.nrefs], 0..) |x, i| e.setBuffer(x.buf, x.off, i);
        if (p.restore.len > 0) {
            const b = std.mem.sliceAsBytes(p.restore);
            @memcpy(p.rig.bytes(p.refs[p.nrefs - 1], b.len), b);
        }
        for (p.clear) |x| @memset(p.rig.bytes(x, p.want.len / p.clear.len * c.size(if (p.kind == .qkv) .f32 else .bf16)), 0);
        const n: u32 = @intCast(p.s.rows);
        const w: u32 = @intCast(p.s.n);
        switch (p.kind) {
            .pack => e.setValue(n * w, 2),
            .embed => e.setValue(Embed{ .patches = n, .width = w }, 7),
            .heads => e.setValue(Heads{ .rows = n, .r0 = r0, .heads = w / dim, .dim = dim }, 2),
            .qkv => e.setValue(Qkv{ .rows = n + r0, .heads = w / dim, .dim = dim, .r0 = r0 }, 9),
            else => e.setValue(w, p.nrefs),
        }
        if (p.kind == .qkv) {
            e.dispatchThreads(mtl.Size.of(dim, 3 * w / dim, n), mtl.Size.of(dim, 4, 1));
        } else {
            const grid = if (p.kind == .pack) mtl.Size.of(w * n, 1, 1) else mtl.Size.of(w, n + @as(u32, @intFromBool(p.kind == .embed)), 1);
            e.dispatchThreads(grid, mtl.Size.of(256, 1, 1));
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.want.len;
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        @memcpy(want, p.want);
        @memcpy(mag, p.mag);
    }

    pub fn bounds(_: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        for (want, mag, allow) |y, m, *b| {
            // GELU's erf polynomial has absolute error <= 1.5e-7; all other dyadic epilogues use no inexact sums.
            const d = m * 0x1p-19;
            b.* = d + (@abs(y) + d) * c.unit(.bf16) + 0x1p-133;
        }
    }

    fn embedRows(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) void {
        const width = p.s.n;
        for (0..p.s.rows) |i| for (0..width) |j| {
            var terms: [4]f64 = undefined;
            for (&terms, 0..) |*x, k| {
                x.* = u.val(p.table[p.taps[i * 4 + k] * width + j]) * p.tap_weights[i * 4 + k];
            }
            const pos = math.sumIn(acc, order, &terms);
            out[(i + 1) * width + j] = u.val(p.pe[i * width + j]) + u.bf(pos);
        };
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        @memcpy(out[0..p.s.n], p.want[0..p.s.n]);
        p.embedRows(acc, order, out);
    }
};
