//! Mini's noncausal draft attention: logical context rows across a physical ring, three query heads per KV head.
const std = @import("std");
const mtl = @import("metal");
const sum = @import("contract").rows.sumIn;
const u = @import("inkling2_util.zig");
const r = @import("rig.zig");
const c = u.c;
const g = u.g;

pub const name = "inkling2 draft ik2d_attn_split + ik2d_attn_merge";
pub const control = struct {
    pub const label = "inkling2 draft attention control drops masked-block keys";
    pub const from = "const uint total = p.nctx + p.rows;";
    pub const to = "const uint total = p.nctx;";
};

const heads = 24;
const groups = 8;
const dim = 128;
const scale = 1 / @sqrt(@as(f64, dim));
const Attn = extern struct { rows: u32, heads: u32, groups: u32, lo: u32, nctx: u32, ring: u32, splits: u32, per: u32, scale: f32 };
const served = [_]c.Shape{
    .{ .rows = 8, .n = heads, .k = 65 },   .{ .rows = 8, .n = heads, .k = 0 },
    .{ .rows = 8, .n = heads, .k = 23 },   .{ .rows = 8, .n = heads, .k = 24 },
    .{ .rows = 8, .n = heads, .k = 25 },   .{ .rows = 8, .n = heads, .k = 1024 },
    .{ .rows = 8, .n = heads, .k = 1170 },
};

fn depth(s: c.Shape) usize {
    const total = s.rows + s.k;
    const n = std.math.clamp((total + 31) / 32, 1, 128);
    return 64 + 8 * ((total + n - 1) / n + n);
}

pub const Probe = struct {
    rig: r.Rig,
    arena: std.heap.ArenaAllocator,
    split: mtl.Pipeline,
    merge: mtl.Pipeline,
    label: []const u8,
    p: Attn = undefined,
    q: []const u16 = &.{},
    key: []const u16 = &.{}, // logical: context first, then the full draft block
    val: []const u16 = &.{},
    refs: [9]r.Ref = undefined,
    stress: bool = false,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, label: []const u8) !Probe {
        const split = try mtl.Pipeline.init(device, lib, "ik2d_attn_split", false);
        errdefer split.deinit();
        const merge = try mtl.Pipeline.init(device, lib, "ik2d_attn_merge", false);
        return .{ .rig = .{ .a = a, .device = device, .queue = queue }, .arena = .init(a), .split = split, .merge = merge, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.arena.deinit();
        p.split.deinit();
        p.merge.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe, gate: ?c.Gate) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = gate, .spec = .{
            .name = p.label,
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .served = &served,
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "BF16 stores hide short positive softmax sums; stress isolates signed value cancellation" },
        } };
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = u.reset(&p.arena, &p.rig);
        const total = s.rows + s.k;
        const splits = std.math.clamp((total + 31) / 32, 1, 128);
        const ring = @max(1152, s.k);
        const lo = if (s.k <= 1152) ring - 3 else 0;
        p.p = .{ .rows = @intCast(s.rows), .nctx = @intCast(s.k), .heads = heads, .groups = groups, .lo = @intCast(lo), .ring = @intCast(ring), .splits = @intCast(splits), .per = @intCast((total + splits - 1) / splits), .scale = @floatCast(scale) };
        p.stress = fill == .stress;
        const qw = heads * dim;
        const kvw = groups * dim;
        const q = try a.alloc(u16, s.rows * qw);
        const k = try a.alloc(u16, total * kvw);
        const v = try a.alloc(u16, total * kvw);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        u.fillDyadic(rng, q, 80, 64);
        u.fillDyadic(rng, k, 80, 64);
        u.fillDyadic(rng, v, 80, 64);
        if (fill == .nonfinite) {
            q[0] = u.nan_bits;
            q[qw + dim] = u.pinf_bits;
            q[2 * qw + dim * 2] = u.ninf_bits;
            v[4 * dim + 9] = u.pinf_bits;
        } else if (p.stress) {
            // Uniform attention, integer V sums with cancellation: 1 / (context + block) in every component.
            @memset(q, 0);
            const h = total / 2;
            for (0..total) |j| {
                const x: f64 = if (j == h - 1) 1 else if (j >= 2 * h - 1) 0 else if (j < h) 1024 else -1024;
                @memset(v[j * kvw ..][0..kvw], u.word(x));
            }
        }
        p.q = q;
        p.key = k;
        p.val = v;
        const kc = try a.alloc(u16, ring * kvw);
        const vc = try a.alloc(u16, ring * kvw);
        @memset(kc, u.nan_bits);
        @memset(vc, u.nan_bits);
        for (0..s.k) |j| {
            const slot = (lo + j) % ring;
            @memcpy(kc[slot * kvw ..][0..kvw], k[j * kvw ..][0..kvw]);
            @memcpy(vc[slot * kvw ..][0..kvw], v[j * kvw ..][0..kvw]);
        }
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(q));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(kc));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(vc));
        p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(k[s.k * kvw ..]));
        p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(v[s.k * kvw ..]));
        const part = splits * s.rows * heads;
        p.refs[5] = try p.rig.scratch(part * 4);
        p.refs[6] = try p.rig.scratch(part * 4);
        p.refs[7] = try p.rig.scratch(part * dim * 4);
        p.refs[8] = try p.rig.output(p.count() * 2);
        return true;
    }

    fn bind(e: mtl.ComputeEncoder, refs: []const r.Ref, params: Attn) void {
        for (refs, 0..) |v, i| e.setBuffer(v.buf, v.off, i);
        e.setValue(params, refs.len);
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.split);
        bind(e, p.refs[0..8], p.p);
        e.dispatchGroups(mtl.Size.of(groups, p.p.splits, 1), mtl.Size.of(96, 1, 1));
        e.setPipeline(p.merge);
        bind(e, p.refs[5..9], p.p);
        e.dispatchGroups(mtl.Size.of(heads, p.p.rows, 1), mtl.Size.of(dim, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return @as(usize, p.p.rows) * heads * dim;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const total = p.p.nctx + p.p.rows;
        const a = p.rig.a;
        const s = try a.alloc(f64, total);
        defer a.free(s);
        for (0..p.p.rows) |i| for (0..heads) |h| {
            const group = h / (heads / groups);
            const row = (i * heads + h) * dim;
            const q = p.q[row..][0..dim];
            var top = -std.math.inf(f64);
            for (s, 0..) |*score, j| {
                const k = p.key[(j * groups + group) * dim ..][0..dim];
                var z: f64 = 0;
                for (q, k) |x, y| z += u.val(x) * u.val(y);
                score.* = z * scale;
                top = @max(top, score.*);
            }
            var den: f64 = 0;
            for (s) |*score| {
                score.* = @exp(score.* - top);
                den += score.*;
            }
            for (0..dim) |d| {
                var num: f64 = 0;
                var abs: f64 = 0;
                for (s, 0..) |w, j| {
                    const term = w * u.val(p.val[(j * groups + group) * dim + d]);
                    num += term;
                    abs += @abs(term);
                }
                want[row + d] = num / den;
                mag[row + d] = abs / den;
            }
        };
    }

    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        // Stress has all scores zero, integral sums in range, and only a final division before the BF16 store.
        const d = if (p.stress) 1 else depth(.{ .rows = p.p.rows, .n = heads, .k = p.p.nctx });
        const err = c.gamma(@floatFromInt(d), c.unit(.f32));
        for (allow, want, mag) |*lim, w, m| {
            const delta = err * (if (p.stress) @abs(w) else m);
            lim.* = delta + c.unit(.bf16) * (@abs(w) + delta) + 0x1p-133;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = p.rig.a;
        const total = p.p.nctx + p.p.rows;
        const terms = try a.alloc(f64, total);
        defer a.free(terms);
        for (terms, 0..) |*v, j| v.* = u.val(p.val[j * groups * dim]);
        const n = sum(acc, order, terms);
        @memset(terms, 1);
        @memset(out, n / sum(acc, order, terms));
    }
};
