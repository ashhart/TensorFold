//! MiMo-V2's 8-bit key/value cache for the full-attention layers (--policy kv=fp8; kernels in kv8.cu).

const std = @import("std");
const cuda = @import("cuda");
const load_ = @import("load.zig");
const fwd = @import("forward.zig");
const Config = @import("config.zig").Config;
const aot = cuda.aot;
const Stream = cuda.Stream;
const Model = fwd.Model;

/// Keys a program of mimo_rows8 reads: four Triton chunks, so a graph's bucket still divides into whole chunks.
pub const CK = 4 * fwd.CK;

fn u(x: usize) u32 {
    return @intCast(x);
}

/// A layer's 8-bit cache: e4m3 keys / values and their fp32 scales a (token, key head), `len` tokens a slot.
pub const Cache = struct { kc: u64, ks: u64, vc: u64, vs: u64, len: usize, hk: usize };

/// Decode attention's rows and partials: rotated queries, positions, slots, and the chunks' (o, m, l).
pub const Rows = struct { qo: u64, pos: u64, slot: u64, po: u64, pm: u64, pl: u64 };

pub const Kv8 = struct {
    mod: cuda.Module,
    f_qkv: cuda.Function,
    f_rows: cuda.Function,
    f_merge: cuda.Function,
    f_dequant: cuda.Function,
    ks: []u64, // per layer: key scales [slots, context, HK] (0 on sliding layers)
    vs: []u64,

    /// The kernels and the full layers' scales; the model allocates the 8-bit keys and values in its own caches.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, ar: *load_.Arena, cfg: Config, slots: usize, context: usize) !Kv8 {
        var k: Kv8 = undefined;
        k.mod = try cuda.Module.load(d, cuda.kernels.mimo_kv8);
        errdefer k.mod.unload();
        k.f_qkv = try k.mod.function("mimo_qkv8");
        k.f_rows = try k.mod.function("mimo_rows8");
        k.f_merge = try k.mod.function("mimo_merge8");
        k.f_dequant = try k.mod.function("mimo_kv8_dequant");
        k.ks = try gpa.alloc(u64, cfg.layers);
        errdefer gpa.free(k.ks);
        k.vs = try gpa.alloc(u64, cfg.layers);
        errdefer gpa.free(k.vs);
        for (0..cfg.layers) |i| {
            const n = if (cfg.full[i]) slots * context * cfg.kvHeads(i) * 4 else 0;
            k.ks[i] = if (n > 0) try ar.zeros(n) else 0;
            k.vs[i] = if (n > 0) try ar.zeros(n) else 0;
        }
        return k;
    }

    pub fn deinit(k: *Kv8, gpa: std.mem.Allocator) void {
        gpa.free(k.ks);
        gpa.free(k.vs);
        k.mod.unload();
    }

    fn cache(k: *const Kv8, m: *const Model, i: usize) Cache {
        return .{ .kc = m.kc[i], .ks = k.ks[i], .vc = m.vc[i], .vs = k.vs[i], .len = m.context, .hk = m.w.cfg.kvHeads(i) };
    }

    /// attention._qkv for full layer i: RoPE, q to `qo` in fp16, keys and values quantised into the 8-bit cache.
    pub fn qkv(k: *const Kv8, m: *const Model, s: Stream, i: usize, q: u64, key: u64, v: u64, pos: u64, slot: u64, qo: u64, n: usize) !void {
        const c = k.cache(m, i);
        const H = m.w.cfg.heads;
        var a: cuda.Args = .{};
        inline for (.{ q, key, v, m.cos_full, m.sin_full, pos, slot, qo, c.kc, c.ks, c.vc, c.vs }) |x| a.add(x);
        a.add(@as(c_int, @intCast(c.len)));
        a.add(@as(c_int, @intCast(H)));
        a.add(@as(c_int, @intCast(c.hk)));
        try cuda.launch.launch(k.f_qkv, .{ .grid = .{ .x = u(n), .y = u(H + 2 * c.hk), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }

    /// A decode window's full layer i on the 8-bit cache: qkv, partials over `chunks` Triton chunks, the fold.
    pub fn attend(k: *const Kv8, m: *const Model, s: Stream, i: usize, n: usize, chunks: usize) !void {
        const b = &m.b;
        try k.qkv(m, s, i, b.q, b.k, b.v, b.pos, b.slot, b.qo, n);
        const r: Rows = .{ .qo = b.qo, .pos = b.pos, .slot = b.slot, .po = b.po, .pm = b.pm, .pl = b.pl };
        const nch = try k.rows(s, k.cache(m, i), r, m.w.cfg.heads, m.scale, n, chunks);
        try k.merge(s, r, b.att, m.w.cfg.heads, n, nch);
    }

    /// The partials of `n` rows over `chunks` Triton chunks, read as chunks of CK keys; returns their count.
    pub fn rows(k: *const Kv8, s: Stream, c: Cache, r: Rows, heads: usize, scale: f32, n: usize, chunks: usize) !usize {
        const nch = (chunks * fwd.CK + CK - 1) / CK;
        var a: cuda.Args = .{};
        inline for (.{ r.qo, c.kc, c.ks, c.vc, c.vs, r.pos, r.slot, r.po, r.pm, r.pl }) |x| a.add(x);
        a.add(@as(c_int, @intCast(c.len)));
        a.add(@as(c_int, @intCast(nch)));
        a.add(@as(c_int, @intCast(heads)));
        a.add(@as(c_int, @intCast(c.hk)));
        a.add(scale);
        a.add(@as(c_int, CK));
        try cuda.launch.launch(k.f_rows, .{ .grid = .{ .x = u(n), .y = u(c.hk), .z = u(nch) }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
        return nch;
    }

    /// The chunk partials folded into fp16 `out` [n, H, 128] (no sinks: full layers).
    pub fn merge(k: *const Kv8, s: Stream, r: Rows, out: u64, heads: usize, n: usize, nch: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ r.po, r.pm, r.pl, out }) |x| a.add(x);
        a.add(@as(c_int, @intCast(nch)));
        a.add(@as(c_int, @intCast(heads)));
        try cuda.launch.launch(k.f_merge, .{ .grid = .{ .x = u(n), .y = u(heads), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    /// One slot's keys [0, keys) of full layer i back to fp16 [keys, HK, 192] / [keys, HK, 128] for the prompt kernel.
    pub fn dequant(k: *const Kv8, m: *const Model, s: Stream, i: usize, slot: usize, keys: usize, kd: u64, vd: u64) !void {
        try k.dequantFrom(s, k.cache(m, i), @intCast(slot * m.context), keys, kd, vd);
    }

    fn dequantFrom(k: *const Kv8, s: Stream, c: Cache, base: i64, keys: usize, kd: u64, vd: u64) !void {
        var a: cuda.Args = .{};
        inline for (.{ c.kc, c.ks, c.vc, c.vs, kd, vd }) |x| a.add(x);
        a.add(base);
        a.add(@as(c_int, @intCast(keys)));
        a.add(@as(c_int, @intCast(c.hk)));
        try cuda.launch.launch(k.f_dequant, .{ .grid = .{ .x = 1024, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }
};

/// tf-mimo kv8check: 8-bit decode attention against Triton's on one slot of hashed keys and values, rows at 128,000.
pub fn check(m: *Model, io: std.Io, s: Stream, rank: usize) !bool {
    const cfg = m.w.cfg;
    const H = cfg.heads;
    const HK = cfg.kv_full;
    const LEN: usize = 131072 + 64;
    const NCH: usize = 131072 / fwd.CK;
    const P: usize = 128000;
    const ent = LEN * HK;
    const ar = &m.arena;
    var k = try Kv8.init(m.gpa, m.d, ar, cfg, 0, 0);
    defer k.deinit(m.gpa);
    const f_fill = try k.mod.function("mimo_fill_half");
    const f_quant = try k.mod.function("mimo_kv8_quant");
    const k16 = try ar.zeros(ent * 192 * 2);
    const v16 = try ar.zeros(ent * 128 * 2);
    const c8: Cache = .{ .kc = try ar.zeros(ent * 192), .ks = try ar.zeros(ent * 4), .vc = try ar.zeros(ent * 128), .vs = try ar.zeros(ent * 4), .len = LEN, .hk = HK };
    const kd = try ar.zeros(ent * 192 * 2);
    const vd = try ar.zeros(ent * 128 * 2);
    const q = try ar.zeros(8 * H * 192 * 2);
    const part: Rows = .{ .qo = q, .pos = 0, .slot = 0, .po = try ar.zeros(8 * NCH * H * 128 * 4), .pm = try ar.zeros(8 * NCH * H * 4), .pl = try ar.zeros(8 * NCH * H * 4) };
    const per_max = 8 * H * 128;
    const outs = [3]u64{ try ar.zeros(per_max * 2), try ar.zeros(per_max * 2), try ar.zeros(per_max * 2) };
    const grid1k: cuda.launch.Config = .{ .grid = .{ .x = 1024, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } };
    inline for (.{ .{ k16, ent * 192, 1, 1.0 }, .{ v16, ent * 128, 2, 1.0 }, .{ q, 8 * H * 192, 3, 3.0 } }) |f| {
        var a: cuda.Args = .{};
        a.add(f[0]);
        a.add(@as(i64, @intCast(f[1])));
        a.add(@as(u32, f[2]));
        a.add(@as(f32, f[3]));
        try cuda.launch.launch(f_fill, grid1k, s, &a);
    }
    {
        var a: cuda.Args = .{};
        inline for (.{ k16, v16, c8.kc, c8.ks, c8.vc, c8.vs }) |x| a.add(x);
        try cuda.launch.launch(f_quant, .{ .grid = .{ .x = u(ent), .y = 1, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }
    try k.dequantFrom(s, c8, 0, LEN, kd, vd);
    const host = try m.gpa.alloc(u16, 3 * per_max);
    defer m.gpa.free(host);
    var ok = true;
    for ([_]usize{ 1, 4, 8 }) |n| {
        var hp: [8]i32 = undefined;
        for (0..n) |j| hp[j] = @intCast(P + j);
        const hs: [8]i32 = @splat(0);
        var r = part;
        r.pos = try ar.upload(std.mem.sliceAsBytes(hp[0..n]));
        r.slot = try ar.upload(std.mem.sliceAsBytes(hs[0..n]));
        const tri: Triton = .{ .m = m, .s = s, .r = r, .n = n, .HK = HK, .LEN = LEN, .NCH = NCH };
        try tri.run(kd, vd, outs[0]); // Triton on the same 8-bit values, dequantised
        try tri.run(k16, v16, outs[1]); // Triton on the 16-bit values
        const nch = try k.rows(s, c8, r, H, m.scale, n, NCH);
        try k.merge(s, r, outs[2], H, n, nch);
        try s.synchronize();
        const per = n * H * 128;
        for (outs, 0..) |o, j| try m.d.check(m.d.api.cuMemcpyDtoH_v2(@ptrCast(host[j * per ..].ptr), o, per * 2), "kv8check");
        var e8: f64 = 0;
        var e16: f64 = 0;
        var bad: usize = 0;
        for (0..per) |i| {
            const r8: f64 = @floatCast(@as(f16, @bitCast(host[i])));
            const r16: f64 = @floatCast(@as(f16, @bitCast(host[per + i])));
            const g: f64 = @floatCast(@as(f16, @bitCast(host[2 * per + i])));
            if (!std.math.isFinite(g)) bad += 1;
            e8 = @max(e8, @abs(g - r8));
            e16 = @max(e16, @abs(g - r16));
        }
        const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        for (0..20) |_| _ = try k.rows(s, c8, r, H, m.scale, n, NCH);
        try s.synchronize();
        const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6 / 20.0;
        const pass = bad == 0 and e8 < 2e-2;
        ok = ok and pass;
        std.debug.print("{s} kv8check rank {d}: {d} rows at {d}: {d:.3} ms a layer; max error vs Triton on the same 8-bit values {e:.2}, on the 16-bit values {e:.2}; non-finite {d}\n", .{ if (pass) "PASS" else "FAIL", rank, n, P, ms, e8, e16, bad });
    }
    return ok;
}

/// Triton's decode attention (attention.rows + _merge, full layer) over an fp16 cache, as the reference.
const Triton = struct {
    m: *Model,
    s: Stream,
    r: Rows,
    n: usize,
    HK: usize,
    LEN: usize,
    NCH: usize,

    fn run(t: Triton, kc: u64, vc: u64, out: u64) !void {
        const p = aot.ptr;
        const H = t.m.w.cfg.heads;
        try t.m.set.run(t.s, "_rows", .{ u(t.n), u(t.HK), u(t.NCH) }, &.{
            p("Q", "*fp16", t.r.qo),         p("KC", "*fp16", kc),        p("VC", "*fp16", vc),
            p("POS", "*i32", t.r.pos),       p("SLOT", "*i32", t.r.slot), p("PO", "*fp32", t.r.po),
            p("PM", "*fp32", t.r.pm),        p("PL", "*fp32", t.r.pl),    aot.int("LEN", @intCast(t.LEN)),
            aot.int("NCH", @intCast(t.NCH)),
        }, &.{ aot.ci("H", @intCast(H)), aot.ci("HK", @intCast(t.HK)), aot.ci("G", @intCast(H / t.HK)), aot.ci("WIN", 0), aot.cf("SCALE", t.m.scale), aot.ci("CK", fwd.CK), aot.ci("BN", fwd.BN), aot.ci("GP", fwd.GP) });
        try t.m.set.run(t.s, "_merge", .{ u(t.n), u(H), 1 }, &.{
            p("PO", "*fp32", t.r.po),    p("PM", "*fp32", t.r.pm), p("PL", "*fp32", t.r.pl),
            p("SINKS", "*fp32", t.r.pm), p("OUT", "*fp16", out),   aot.int("NCH", @intCast(t.NCH)),
        }, &.{ aot.ci("H", @intCast(H)), aot.ci("HAS_SINK", 0) });
    }
};
