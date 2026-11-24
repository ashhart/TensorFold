//! DSpark bidirectional block attention under the contract: float64 reference, controls, repeat, shapes, sums.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const attn = tf.dsv41_metal.attn;
const D = attn.head_dim;
const base = "dsv41_dspark_attn ";

pub const Variant = struct { ctl: u8 = 0 };

fn sh(rows: usize, heads: usize, ctx_count: usize) c.Shape {
    return .{ .rows = rows, .n = heads * D, .k = ctx_count };
}

const served = [_]c.Shape{
    sh(5, 8, 128),
    sh(7, 8, 128),
    sh(7, 64, 128),
    sh(14, 8, 128),
    sh(15, 8, 64),
    sh(1, 8, 0),
    sh(3, 8, 1),
    sh(5, 8, 37),
    sh(10, 4, 127),
};

fn depth(s: c.Shape) usize {
    return 2 * (s.k + s.rows) + 40;
}

pub fn spec(name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .tiles = .{ .{ .size = 8, .min = 1 }, .{ .size = D, .step = D, .min = D }, .{ .size = 4, .step = 1, .min = 0 } },
        .served = &served,
    };
}

/// Split `rows` into (`asks`, `n`) matching DSpark launches: `asks * n == rows` with `n <= 15`.
fn splitRows(rows: usize) struct { asks: u32, n: u32 } {
    if (rows <= 15) return .{ .asks = 1, .n = @intCast(rows) };
    if (rows % 7 == 0 and rows / 7 <= 16) return .{ .asks = @intCast(rows / 7), .n = 7 };
    if (rows % 5 == 0 and rows / 5 <= 16) return .{ .asks = @intCast(rows / 5), .n = 5 };
    if (rows % 4 == 0 and rows / 4 <= 16) return .{ .asks = @intCast(rows / 4), .n = 4 };
    if (rows % 2 == 0 and rows / 2 <= 16) return .{ .asks = @intCast(rows / 2), .n = 2 };
    return .{ .asks = 0, .n = 0 };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipe: mtl.Pipeline,
    rows: usize = 0,
    heads: usize = 0,
    asks_n: u32 = 0,
    block_n: u32 = 0,
    win: u32 = attn.window,
    q: []u16 = &.{},
    ring: []u16 = &.{},
    asks: []attn.DsparkAsk = &.{},
    sink: []f32 = &.{},
    cs: []f32 = &.{},
    q_ref: rig_.Ref = undefined,
    ring_ref: rig_.Ref = undefined,
    asks_ref: rig_.Ref = undefined,
    sink_ref: rig_.Ref = undefined,
    cs_ref: rig_.Ref = undefined,
    out_ref: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        const text = if (v.ctl == 0) null else try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ v.ctl, sources.dsv41.dspark_attn });
        defer if (text) |t| a.free(t);
        const pipe = try attn.Pipelines.loadDsparkSource(device, text orelse sources.dsv41.dspark_attn);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipe = pipe };
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        p.pipe.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.q.len > 0) p.a.free(p.q);
        if (p.ring.len > 0) p.a.free(p.ring);
        if (p.asks.len > 0) p.a.free(p.asks);
        if (p.sink.len > 0) p.a.free(p.sink);
        if (p.cs.len > 0) p.a.free(p.cs);
        p.q = &.{};
        p.ring = &.{};
        p.asks = &.{};
        p.sink = &.{};
        p.cs = &.{};
        p.rig.reset();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.v.ctl) {
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            4 => .repeat,
            else => null,
        };
        return .{ .spec = spec(p.name), .ctx = p, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Proj);

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (shape.rows == 0 or shape.n == 0 or shape.n % D != 0 or shape.k > attn.window) return false;
        const sp = splitRows(shape.rows);
        if (sp.asks == 0 or sp.n == 0 or sp.n > 15) return false;
        p.rows = shape.rows;
        p.heads = shape.n / D;
        p.asks_n = sp.asks;
        p.block_n = sp.n;
        p.win = attn.window;
        const pool_rows: u32 = p.win + p.block_n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();

        p.q = try p.a.alloc(u16, p.rows * p.heads * D);
        p.ring = try p.a.alloc(u16, @as(usize, p.asks_n) * pool_rows * D);
        p.asks = try p.a.alloc(attn.DsparkAsk, p.asks_n);
        p.sink = try p.a.alloc(f32, p.heads);
        var max_pos: u32 = 0;
        for (p.asks, 0..) |*ask, j| {
            const start: u32 = @intCast(shape.k + (j * 19 + seed % 23));
            ask.* = .{
                .base = @intCast(j * pool_rows),
                .first_mod = @intCast((j * 37 + seed % p.win) % p.win),
                .ctx_count = @intCast(shape.k),
                .start_pos = start,
            };
            max_pos = @max(max_pos, start + p.block_n);
        }
        p.cs = try p.a.alloc(f32, @as(usize, max_pos + 1) * 64);
        for (p.cs, 0..) |*v, i| {
            const pos: f64 = @floatFromInt(i / 64);
            const k: f64 = @floatFromInt(i % 32);
            const ang = pos * std.math.pow(f64, 10000, -k / 32);
            v.* = @floatCast(if (i % 64 < 32) @cos(ang) else @sin(ang));
        }
        for (p.sink) |*v| v.* = if (fill == .stress) -std.math.inf(f32) else rng.float(f32) * 2 - 1;
        if (fill == .stress) {
            const half = D / 2;
            const tail = half - attn.rope_dims;
            for (0..p.rows * p.heads) |g0| for (0..half) |i| {
                const v: f32 = if (i >= tail) 0 else @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127);
                p.q[g0 * D + i] = c.toBf16(v);
                p.q[g0 * D + half + i] = c.toBf16(v);
            };
            for (0..p.ring.len / D) |r| for (0..half) |i| {
                const v: f32 = if (i >= tail) 0 else @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127);
                p.ring[r * D + i] = c.toBf16(v);
                p.ring[r * D + half + i] = c.toBf16(-v);
            };
        } else {
            for (p.q) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
            for (p.ring) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        }
        if (fill == .nonfinite) {
            p.q[100] = c.toBf16(std.math.nan(f32));
            const a0 = p.asks[0];
            const krow: usize = if (a0.ctx_count > 0) a0.base + a0.first_mod else a0.base + p.win;
            p.ring[krow * D + 13] = c.toBf16(std.math.inf(f32));
            if (p.asks_n > 1) {
                const a1 = p.asks[1];
                p.ring[(a1.base + p.win) * D + 7] = c.toBf16(-std.math.inf(f32));
            }
        }
        p.q_ref = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.ring_ref = try p.rig.input(std.mem.sliceAsBytes(p.ring));
        p.asks_ref = try p.rig.input(std.mem.sliceAsBytes(p.asks));
        p.sink_ref = try p.rig.input(std.mem.sliceAsBytes(p.sink));
        p.cs_ref = try p.rig.input(std.mem.sliceAsBytes(p.cs));
        p.out_ref = try p.rig.output(p.count() * 2);
        return true;
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try attn.encodeDsparkPipe(
            p.pipe,
            pass.e,
            .{ .buf = p.q_ref.buf, .off = p.q_ref.off },
            .{ .buf = p.ring_ref.buf, .off = p.ring_ref.off },
            .{ .buf = p.asks_ref.buf, .off = p.asks_ref.off },
            .{ .buf = p.sink_ref.buf, .off = p.sink_ref.off },
            .{ .buf = p.cs_ref.buf, .off = p.cs_ref.off },
            .{ .buf = p.out_ref.buf, .off = p.out_ref.off },
            p.asks_n,
            p.block_n,
            @intCast(p.heads),
            p.win,
        );
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.rows * p.heads * D;
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        const sc: f64 = attn.scale();
        var sbuf: [attn.window + 16]f64 = undefined;
        var smag: [attn.window + 16]f64 = undefined;
        var krows: [attn.window + 16]usize = undefined;
        for (p.asks, 0..) |ask, j| {
            const total: usize = ask.ctx_count + p.block_n;
            for (0..total) |kt| {
                krows[kt] = if (kt < ask.ctx_count)
                    ask.base + ((ask.first_mod + kt) % p.win)
                else
                    ask.base + p.win + (kt - ask.ctx_count);
            }
            for (0..p.block_n) |sg| {
                const row = j * p.block_n + sg;
                const cs = p.cs[@as(usize, ask.start_pos + @as(u32, @intCast(sg))) * 64 ..][0..64];
                for (0..p.heads) |h| {
                    const out_at = (row * p.heads + h) * D;
                    const q = p.q[out_at..][0..D];
                    var m: f64 = @as(f64, p.sink[h]);
                    for (0..total) |kt| {
                        const kv = p.ring[krows[kt] * D ..][0..D];
                        var dot: f64 = 0;
                        var dm: f64 = 0;
                        for (q, kv) |qb, kb| {
                            const x: f64 = c.fromBf16(qb);
                            const y: f64 = c.fromBf16(kb);
                            dot += x * y;
                            dm += @abs(x * y);
                        }
                        sbuf[kt] = dot * sc;
                        smag[kt] = dm * sc;
                        if (sbuf[kt] > m or std.math.isNan(sbuf[kt])) m = sbuf[kt];
                    }
                    var den: f64 = @exp(@as(f64, p.sink[h]) - m);
                    var wbuf: [attn.window + 16]f64 = undefined;
                    for (0..total) |kt| {
                        wbuf[kt] = @exp(sbuf[kt] - m);
                        den += wbuf[kt];
                    }
                    var acc: [D]f64 = @splat(0);
                    var amag: [D]f64 = @splat(0);
                    for (0..total) |kt| {
                        const w = wbuf[kt] / den;
                        const kv = p.ring[krows[kt] * D ..][0..D];
                        for (0..D) |d| {
                            const y: f64 = c.fromBf16(kv[d]);
                            acc[d] += w * y;
                            amag[d] += @abs(w * y) * (1.0 + smag[kt]);
                        }
                    }
                    var d: usize = D - attn.rope_dims;
                    while (d < D) : (d += 2) {
                        const pr = (d - (D - attn.rope_dims)) / 2;
                        const cc: f64 = cs[pr];
                        const sn: f64 = cs[32 + pr];
                        const x0 = acc[d];
                        const x1 = acc[d + 1];
                        acc[d] = x0 * cc + x1 * sn;
                        acc[d + 1] = x1 * cc - x0 * sn;
                        const m0 = amag[d];
                        const m1 = amag[d + 1];
                        amag[d] = m0 * @abs(cc) + m1 * @abs(sn);
                        amag[d + 1] = m1 * @abs(cc) + m0 * @abs(sn);
                    }
                    @memcpy(want[out_at..][0..D], &acc);
                    @memcpy(mag[out_at..][0..D], &amag);
                }
            }
        }
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const rows_ = contract.rows;
        const sc: f64 = attn.scale();
        var sbuf: [attn.window + 16]f64 = undefined;
        var pbuf: [attn.window + 16]f64 = undefined;
        var tbuf: [attn.window + 16]f64 = undefined;
        var terms: [D]f64 = undefined;
        var krows: [attn.window + 16]usize = undefined;
        for (p.asks, 0..) |ask, j| {
            const total: usize = ask.ctx_count + p.block_n;
            for (0..total) |kt| {
                krows[kt] = if (kt < ask.ctx_count)
                    ask.base + ((ask.first_mod + kt) % p.win)
                else
                    ask.base + p.win + (kt - ask.ctx_count);
            }
            for (0..p.block_n) |sg| {
                const row = j * p.block_n + sg;
                const cs = p.cs[@as(usize, ask.start_pos + @as(u32, @intCast(sg))) * 64 ..][0..64];
                for (0..p.heads) |h| {
                    const out_at = (row * p.heads + h) * D;
                    const q = p.q[out_at..][0..D];
                    for (0..total) |kt| {
                        const kv = p.ring[krows[kt] * D ..][0..D];
                        for (0..D) |d| terms[d] = @as(f64, c.fromBf16(q[d])) * @as(f64, c.fromBf16(kv[d]));
                        sbuf[kt] = rows_.sumIn(acc, order, &terms) * sc;
                    }
                    var top: f64 = p.sink[h];
                    for (sbuf[0..total]) |v| top = if (std.math.isNan(v)) v else @max(top, v);
                    for (sbuf[0..total], pbuf[0..total]) |v, *o| o.* = @exp(v - top);
                    const den = rows_.sumIn(acc, order, pbuf[0..total]) + @exp(@as(f64, p.sink[h]) - top);
                    const x = out[out_at..][0..D];
                    for (0..D) |d| {
                        for (0..total) |kt| tbuf[kt] = pbuf[kt] * @as(f64, c.fromBf16(p.ring[krows[kt] * D + d]));
                        x[d] = rows_.sumIn(acc, order, tbuf[0..total]) / den;
                    }
                    var d: usize = D - attn.rope_dims;
                    while (d < D) : (d += 2) {
                        const pr = (d - (D - attn.rope_dims)) / 2;
                        const cc: f64 = cs[pr];
                        const sn: f64 = cs[32 + pr];
                        const x0 = x[d];
                        const x1 = x[d + 1];
                        x[d] = x0 * cc + x1 * sn;
                        x[d + 1] = x1 * cc - x0 * sn;
                    }
                }
            }
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{}, .name = base ++ "bf16" },
        .{ .v = .{ .ctl = 1 }, .name = base ++ "control: NaN scores dropped" },
        .{ .v = .{ .ctl = 2 }, .name = base ++ "control: last key dropped" },
        .{ .v = .{ .ctl = 3 }, .name = base ++ "control: scores summed in fp16" },
        .{ .v = .{ .ctl = 4 }, .name = base ++ "control: writes past its output" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.v, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }
};
