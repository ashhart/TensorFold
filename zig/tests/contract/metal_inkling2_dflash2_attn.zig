//! Mini DFlash 2 attention and merge kernels under the contract: 1 and 3 streams over wrapped rings.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const controls = @import("metal_inkling2_dflash2_batch_controls.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const host = tf.inkling2_dflash2_host;
const sources = @import("inkling2_metal");

const window = 2048;
const ring_size = 2176;
const block_max = 8;
const D = 128;
const QH = 32;
const KVH = 8;
const G = 4;
const QW = QH * D;
const KW = KVH * D;
const chunk_len = 128;
const part_stride = 132;
const nan16: u16 = 0x7fc0;

const Case = enum { plain, streams, attend_ring, attend_reach, attend_masked, attend_merge };
const registered = [_]Case{ .plain, .streams, .attend_ring, .attend_reach, .attend_masked, .attend_merge };

fn streamsOf(case: Case) usize {
    return if (case == .plain) 1 else 3;
}

fn caseName(case: Case) []const u8 {
    return switch (case) {
        .plain => "metal inkling2 dflash2 attention (1 stream, wrapped ring) +merge",
        .streams => "metal inkling2 dflash2 attention (3 streams, varied lengths) +merge",
        .attend_ring => controls.name(.attend_ring),
        .attend_reach => controls.name(.attend_reach),
        .attend_masked => controls.name(.attend_masked),
        .attend_merge => controls.name(.attend_merge),
    };
}

fn controlKind(case: Case) ?controls.Kind {
    return switch (case) {
        .plain, .streams => null,
        .attend_ring => .attend_ring,
        .attend_reach => .attend_reach,
        .attend_masked => .attend_masked,
        .attend_merge => .attend_merge,
    };
}

fn keysDepth(s: c.Shape) usize {
    return s.n + s.k + s.rows;
}

fn form(keys: usize) ar.Form {
    const chunks: f64 = @floatFromInt((keys + chunk_len - 1) / chunk_len + 1);
    const blocks: f64 = chunk_len / 32;
    return .{
        .score = D + 1,
        .fast_exp = false,
        .chunk_exp = true,
        .p_rel = 0,
        .num = chunk_len + blocks + chunks + 2,
        .den = 5 + blocks + chunks + 2,
        .nan_max = true,
    };
}

fn stressValue(j: usize, n: usize) f32 {
    if (j < block_max) return 0;
    const rem = n - block_max;
    const half = rem / 2;
    const idx = j - block_max;
    if (idx >= 2 * half) return 0;
    const at = if (idx < half) idx else idx - half;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    const one: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
    return if (idx < half) big + one else -big;
}

const One = struct {
    context: usize = 0,
    lo: usize = 0,
    kp: []u16 = &.{},
    vp: []u16 = &.{},
    fn keys(o: One, block: usize) usize {
        return o.context + block;
    }
    fn live(o: One, lr: usize, j: usize) bool {
        if (j >= o.context) return true;
        const reach = o.lo + o.context + lr + 1;
        const row_lo = @max(o.lo, if (reach > window) reach - window else 0);
        return (o.lo + j) >= row_lo;
    }
};

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    case: Case,
    pipes: host.Pipelines,
    block: usize = 0,
    streams: usize = 1,
    one: [3]One = @splat(.{}),
    stress: bool = false,
    q: []u16 = &.{},
    half: []f64 = &.{},
    point: []f64 = &.{},
    table: [3]host.AttendStream = undefined,
    refs: [3]rig_.Ref = undefined,
    rings: [3][2]rig_.Ref = undefined,
    part: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, case: Case) !Probe {
        const pipes = if (controlKind(case)) |ck| blk: {
            const txt = try controls.source(a, sources.inkling2_dflash2, ck);
            defer a.free(txt);
            break :blk try host.Pipelines.loadSource(device, txt);
        } else try host.Pipelines.loadSource(device, sources.inkling2_dflash2);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .case = case, .streams = streamsOf(case), .pipes = pipes };
    }

    pub fn deinit(p: *Probe) void {
        p.free();
        p.pipes.deinit();
        p.rig.deinit();
    }

    fn free(p: *Probe) void {
        for (&p.one) |*o| {
            p.a.free(o.kp);
            p.a.free(o.vp);
            o.* = .{};
        }
        p.a.free(p.q);
        p.a.free(p.half);
        p.a.free(p.point);
        p.q = &.{};
        p.half = &.{};
        p.point = &.{};
    }

    fn rows(p: *const Probe) usize {
        return p.streams * p.block;
    }

    pub fn spec(p: *const Probe) c.Spec {
        const served = &[_]c.Shape{
            .{ .rows = 8, .n = D, .k = 2047 },
            .{ .rows = 8, .n = D, .k = 300 },
            .{ .rows = 1, .n = D, .k = 64 },
        };
        return .{
            .name = caseName(p.case),
            .accum = .f32,
            .depth = keysDepth,
            .ops = 4,
            .out = .bf16,
            .tiles = .{ .{ .size = 8 }, .{}, .{ .size = chunk_len, .min = 0 } },
            .served = served,
        };
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > block_max or s.n != D or s.k >= ring_size) return false;
        p.rig.reset();
        p.free();
        p.block = s.rows;
        p.stress = fill == .stress;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.q = try p.a.alloc(u16, p.rows() * QW);
        for (p.q) |*y| y.* = c.toBf16(rng.float(f32) * 2 - 1);
        const own_k = try p.a.alloc(u16, p.rows() * KW);
        defer p.a.free(own_k);
        const own_v = try p.a.alloc(u16, p.rows() * KW);
        defer p.a.free(own_v);
        for (p.one[0..p.streams], 0..) |*o, k| {
            o.context = if (p.streams == 1) s.k else switch (k) {
                0 => s.k / 3,
                1 => if (s.k >= 2) s.k - 1 else s.k,
                else => s.k,
            };
            o.lo = if (s.k > 100) 2000 + 37 * k else 0;
            const n = o.keys(p.block);
            o.kp = try p.a.alloc(u16, n * KW);
            o.vp = try p.a.alloc(u16, n * KW);
            for ([_][]u16{ o.kp, o.vp }) |x| for (x) |*y| {
                y.* = c.toBf16(rng.float(f32) * 2 - 1);
            };
            if (p.stress) {
                @memset(o.kp, 0);
                for (0..n) |j| @memset(o.vp[j * KW ..][0..KW], c.toBf16(stressValue(j, n)));
            }
        }
        if (p.stress) @memset(p.q, 0);
        if (fill == .nonfinite) {
            const top_idx = p.streams - 1;
            const o_top = &p.one[top_idx];
            p.q[(0 * QH + 1) * D + 3] = c.toBf16(std.math.nan(f32));
            const last = o_top.keys(p.block) - 1;
            o_top.vp[last * KW + 5] = c.toBf16(std.math.inf(f32));
            o_top.vp[last * KW + 6] = c.toBf16(-std.math.inf(f32));
            if (o_top.context > 0) o_top.vp[1 * D + 7] = c.toBf16(std.math.inf(f32));
        }
        const kr = try p.a.alloc(u16, ring_size * KW);
        defer p.a.free(kr);
        const vr = try p.a.alloc(u16, ring_size * KW);
        defer p.a.free(vr);
        for (p.one[0..p.streams], 0..) |o, k| {
            @memset(kr, nan16);
            @memset(vr, nan16);
            for (0..o.context) |j| {
                const slot = (o.lo + j) % ring_size;
                @memcpy(kr[slot * KW ..][0..KW], o.kp[j * KW ..][0..KW]);
                @memcpy(vr[slot * KW ..][0..KW], o.vp[j * KW ..][0..KW]);
            }
            p.rings[k] = .{ try p.rig.input(std.mem.sliceAsBytes(kr)), try p.rig.input(std.mem.sliceAsBytes(vr)) };
            @memcpy(own_k[k * p.block * KW ..][0 .. p.block * KW], o.kp[o.context * KW ..]);
            @memcpy(own_v[k * p.block * KW ..][0 .. p.block * KW], o.vp[o.context * KW ..]);
            const n = o.keys(p.block);
            const ch: u32 = @intCast((n + chunk_len - 1) / chunk_len);
            p.table[k] = .{
                .anchor = @intCast(o.lo + o.context),
                .from = @intCast(o.lo),
                .lo = @intCast(o.lo),
                .retained = @intCast(o.context),
                .keys = @intCast(n),
                .chunks = ch,
            };
        }
        p.half = try p.a.alloc(f64, p.count());
        p.point = try p.a.alloc(f64, p.count());
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(own_k));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(own_v));
        var max_chunks: u32 = 1;
        for (p.table[0..p.streams]) |st| max_chunks = @max(max_chunks, st.chunks);
        p.part = try p.rig.scratch(p.rows() * KVH * max_chunks * G * part_stride * 4);
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    pub fn count(p: *Probe) usize {
        return p.rows() * QW;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const r = struct {
            fn of(x: rig_.Ref) host.Ref {
                return .{ .buf = x.buf, .off = x.off };
            }
        }.of;
        var rings: [3]host.Ring = undefined;
        for (rings[0..p.streams], p.rings[0..p.streams]) |*dst, src| dst.* = .{ .k = r(src[0]), .v = r(src[1]) };
        var max_chunks: u32 = 1;
        for (p.table[0..p.streams]) |st| max_chunks = @max(max_chunks, st.chunks);
        const args = host.AttendArgs{
            .rows = @intCast(p.rows()),
            .block = @intCast(p.block),
            .window = window,
            .ring = ring_size,
            .max_chunks = max_chunks,
            .scale = 1.0 / @sqrt(@as(f32, D)),
        };
        host.attend(&p.pipes, pass.e, r(p.refs[0]), r(p.refs[1]), r(p.refs[2]), rings[0..p.streams], r(p.part), r(p.out), args, p.table[0..p.streams]);
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const scale: f64 = @as(f32, 1.0 / @sqrt(@as(f32, D)));
        for (p.one[0..p.streams], 0..) |o, k| {
            const n = o.keys(p.block);
            const buf = try p.a.alloc(f64, 2 * n * D + 2 * n + D);
            defer p.a.free(buf);
            const kf = buf[0 .. n * D];
            const vf = buf[n * D .. 2 * n * D];
            const w = buf[2 * n * D ..][0..n];
            const tau = buf[2 * n * D + n ..][0..n];
            const qv = buf[buf.len - D ..];
            for (0..p.block) |lr| for (0..KVH) |hk| {
                var seen: usize = 0;
                for (0..n) |j| if (o.live(lr, j)) {
                    for (0..D) |e| {
                        kf[seen * D + e] = c.fromBf16(o.kp[j * KW + hk * D + e]);
                        vf[seen * D + e] = c.fromBf16(o.vp[j * KW + hk * D + e]);
                    }
                    seen += 1;
                };
                for (hk * G..hk * G + G) |h| {
                    const at = ((k * p.block + lr) * QH + h) * D;
                    for (qv, 0..) |*x, e| x.* = c.fromBf16(p.q[at + e]);
                    ar.head(qv, kf, vf, seen, D, scale, form(n), w, tau, want[at..][0..D], p.half[at..][0..D]);
                }
            };
        }
        @memcpy(mag, p.half);
        @memcpy(p.point, want);
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        const s = p.spec();
        for (allow, want, p.half) |*l, w, h| l.* = if (p.stress) c.boundExact(s, w) else ar.allowBf16(w, h, 0, 0);
    }

    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        for (low, high, p.point, p.half) |*l, *hi, pt, h| {
            const e = if (p.stress) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else ar.intervalBf16(pt, h, 0, 0);
            l.* = e[0];
            hi.* = e[1];
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!p.stress) return error.StressRequiresEqualScores;
        var terms: std.ArrayList(f64) = .empty;
        defer terms.deinit(p.a);
        for (p.one[0..p.streams], 0..) |o, k| {
            const n = o.keys(p.block);
            for (0..p.block) |lr| for (0..KW) |e| {
                terms.clearRetainingCapacity();
                for (0..n) |j| if (o.live(lr, j)) try terms.append(p.a, c.fromBf16(o.vp[j * KW + e]));
                const v = c.fromBf16(c.toBf16(@floatCast(contract.rows.sumIn(acc, order, terms.items) / @as(f64, @floatFromInt(terms.items.len)))));
                for (0..G) |gq| out[(k * p.block + lr) * QW + (e / D * G + gq) * D + e % D] = v;
            };
        }
    }
};

const vt = g.hooks(Probe);

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (registered) |case| {
            const label = caseName(case);
            if (only) |o| if (std.mem.indexOf(u8, label, o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, case);
            errdefer p.deinit();
            try controls.publishProbe(Probe, s.a, &s.probes, s.kernels, p, .{
                .spec = p.spec(),
                .ctx = p,
                .vt = &vt,
                .control = switch (case) {
                    .plain, .streams => null,
                    .attend_masked => .nonfinite,
                    .attend_ring, .attend_reach, .attend_merge => .shapes,
                },
            });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
