//! DSpark's corrected BF16 logits, padding mask, per-group winners and stable final pick.
const std = @import("std");
const mtl = @import("metal");
const u = @import("inkling2_util.zig");
const r = @import("rig.zig");
const stress = @import("metal_inkling2_media_stress.zig");
const sum = @import("contract").rows.sumIn;
const c = u.c;
const g = u.g;

pub const name = "inkling2 draft ik2d_step_logits + ik2d_pick";
pub const control = struct {
    pub const label = "inkling2 draft pick control reverses lowest-id tie";
    pub const from = "v == tv[t] && id < ti[t]";
    pub const to = "v == tv[t] && id > ti[t]";
};

const Step = extern struct { vocab: u32, unpadded: u32, rank: u32, cap: f32 };
const cap: f64 = 30;
const step = 3;
const served = [_]c.Shape{ .{ .rows = 8, .n = 8209, .k = 256 }, .{ .rows = 8, .n = 19, .k = 256 }, .{ .rows = 8, .n = 257, .k = 33 } };
fn depth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

pub const Probe = struct {
    rig: r.Rig,
    arena: std.heap.ArenaAllocator,
    logits: mtl.Pipeline,
    pick: mtl.Pipeline,
    label: []const u8,
    s: c.Shape = undefined,
    unpadded: usize = 0,
    base: []const u16 = &.{},
    latent: []const u16 = &.{},
    weight: []const u16 = &.{},
    bias: []const u16 = &.{},
    refs: [7]r.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, label: []const u8) !Probe {
        const logits = try mtl.Pipeline.init(device, lib, "ik2d_step_logits", false);
        errdefer logits.deinit();
        const pick = try mtl.Pipeline.init(device, lib, "ik2d_pick", false);
        return .{ .rig = .{ .a = a, .device = device, .queue = queue }, .arena = .init(a), .logits = logits, .pick = pick, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.arena.deinit();
        p.logits.deinit();
        p.pick.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe, gate: ?c.Gate) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = gate, .spec = .{
            .name = p.label,
            .accum = .f32,
            .depth = depth,
            .ops = 12,
            .out = .f32,
            .store = .bf16,
            .in_place = true,
            .nan = .{ .stated = "slot-0 NaN survives group selection; other NaNs cannot displace a finite best; pick ignores NaN" },
            .served = &served,
        } };
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = u.reset(&p.arena, &p.rig);
        p.s = s;
        p.unpadded = s.n - 9;
        const base = try a.alloc(u16, s.n);
        const latent = try a.alloc(u16, s.k);
        const weight = try a.alloc(u16, s.n * s.k);
        const bias = try a.alloc(u16, s.n);
        var rng = std.Random.DefaultPrng.init(seed);
        if (fill == .stress) {
            @memset(base, 0);
            @memset(bias, 0);
            for (latent, 0..) |*x, j| x.* = u.word(stress.pair(j, s.k).left);
            for (weight, 0..) |*x, i| x.* = u.word(stress.pair(i % s.k, s.k).right);
        } else {
            u.fillDyadic(rng.random(), latent, 32, 64);
            u.fillDyadic(rng.random(), weight, 32, 256);
            u.fillDyadic(rng.random(), bias, 32, 64);
            for (base, 0..) |*b, i| b.* = u.word(if (i % 8 == 0) 6 else -2);
            for ([_]usize{ 1, 9 }) |id| {
                @memset(weight[id * s.k ..][0..s.k], 0);
                bias[id] = 0;
                base[id] = u.word(16);
            }
            for (base[p.unpadded..]) |*v| v.* = u.word(1e4);
            if (fill == .nonfinite) {
                base[16] = u.nan_bits;
                base[25] = u.pinf_bits;
                base[33] = u.ninf_bits;
                base[42] = u.nan_bits;
            }
        }
        p.base = base;
        p.latent = latent;
        p.weight = weight;
        p.bias = bias;
        for ([_][]const u16{ base, latent, weight, bias }, 0..) |v, i| p.refs[i] = try p.rig.input(std.mem.sliceAsBytes(v));
        p.refs[4] = try p.rig.output(p.count() * 4);
        p.refs[5] = try p.rig.output(p.count() * 4);
        p.refs[6] = try p.rig.output(8 * 4);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const n = p.count();
        @memset(p.rig.bytes(p.refs[6], 8 * 4), 0xfe);
        e.setPipeline(p.logits);
        for (p.refs[0..6], 0..) |v, i| e.setBuffer(v.buf, v.off, i);
        e.setValue(Step{ .vocab = @intCast(p.s.n), .unpadded = @intCast(p.unpadded), .rank = @intCast(p.s.k), .cap = cap }, 6);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        e.setPipeline(p.pick);
        for (p.refs[4..7], 0..) |v, i| e.setBuffer(v.buf, v.off, i);
        e.setValue([2]u32{ @intCast(n), step }, 3);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return (p.s.n + 7) / 8;
    }

    fn score(base: f64, bias: f64, dot: f64) f64 {
        return u.bf(cap * std.math.tanh(u.bf(base + u.bf(bias + dot)) / cap));
    }

    fn token(p: *Probe, id: usize) f64 {
        if (id >= p.unpadded) return -std.math.inf(f64);
        var dot: f64 = 0;
        for (p.latent, p.weight[id * p.s.k ..][0..p.s.k]) |a, b| dot += u.val(a) * u.val(b);
        return score(u.val(p.base[id]), u.val(p.bias[id]), dot);
    }

    const Winner = struct { score: f64, id: usize };
    fn group(p: *Probe, index: usize) Winner {
        const start = 8 * index;
        var result: Winner = .{ .score = p.token(start), .id = start };
        for (start + 1..start + 8) |id| {
            const val = p.token(id);
            if (val > result.score) result = .{ .score = val, .id = id };
        }
        return result;
    }

    pub fn check(p: *Probe, _: c.Gate) !?[]const u8 {
        const n = p.count();
        const scores = p.rig.bytes(p.refs[4], n * 4);
        const ids = p.rig.bytes(p.refs[5], n * 4);
        var best: Winner = .{ .score = -std.math.inf(f64), .id = std.math.maxInt(u32) };
        for (0..n) |i| {
            const w = p.group(i);
            const got = c.read(.f32, scores, i);
            if (c.classOf(w.score) != c.classOf(got)) return "group score has the wrong finite class";
            if (std.math.isFinite(w.score) and @abs(got - w.score) > 2 * c.unit(.bf16) * @abs(w.score)) return "group score leaves the soft-capped BF16 range";
            if (std.mem.readInt(u32, ids[i * 4 ..][0..4], .little) != w.id) return "group picked a different token";
            if (w.score > best.score or (w.score == best.score and w.id < best.id)) best = w;
        }
        const seq = p.rig.bytes(p.refs[6], 8 * 4);
        for (0..8) |i| {
            const got = std.mem.readInt(u32, seq[i * 4 ..][0..4], .little);
            const want = if (i == step + 1) best.id else 0xfefefefe;
            if (got != want) return "pick violated the tie rule or modified another sequence slot";
        }
        return null;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        for (want, 0..) |*v, i| v.* = p.group(i).score;
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try p.rig.a.alloc(f64, p.s.k);
        defer p.rig.a.free(terms);
        for (terms, p.latent, p.weight[0..p.s.k]) |*t, a, b| t.* = u.val(a) * u.val(b);
        const value = score(0, 0, sum(acc, order, terms));
        for (out, 0..) |*v, i| v.* = if (i * 8 < p.unpadded) value else -std.math.inf(f64);
    }
};
