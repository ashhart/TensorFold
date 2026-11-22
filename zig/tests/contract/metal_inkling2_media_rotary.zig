//! Check the draft's 128-wide NeoX rotation and context stores, including the ring-wrap seam.
const std = @import("std");
const mtl = @import("metal");
const u = @import("inkling2_util.zig");
const r = @import("rig.zig");
const sum = @import("contract").rows.sumIn;
const c = u.c;
const g = u.g;

pub const Kind = enum { qk_prep, ctx_store };
pub fn vision(_: Kind) bool {
    return false;
}
pub fn name(comptime k: Kind) []const u8 {
    return "inkling2 draft ik2d_" ++ @tagName(k);
}

pub const controls = .{
    .{ Kind.ctx_store, "inkling2 draft context control reverses cached V", "vc[dst] = vraw[src];", "vc[dst] = vraw[src ^ 1u];", c.Gate.shapes },
    .{ Kind.qk_prep, "inkling2 draft qk_prep control rotates every row at the first position", "float(p.pos0 + g.y)", "float(p.pos0)", c.Gate.shapes },
};

const dim = 128;
const eps: f32 = 1e-6;
const prep_shapes = [_]c.Shape{ .{ .rows = 8, .n = 8, .k = 128 }, .{ .rows = 1, .n = 8, .k = 128 } };
const ctx_shapes = [_]c.Shape{ .{ .rows = 8, .n = 8, .k = 1152 }, .{ .rows = 8, .n = 8, .k = 11 } };
const Prep = extern struct { rows: u32, pos0: u32, heads_q: u32, heads_kv: u32, eps: f32 };
const Ctx = extern struct { rows: u32, pos0: u32, ring: u32, groups: u32, eps: f32 };
fn depth(_: c.Shape) usize {
    return 12;
}

pub const Probe = struct {
    rig: r.Rig,
    arena: std.heap.ArenaAllocator,
    pipe: mtl.Pipeline,
    kind: Kind,
    label: []const u8,
    s: c.Shape = undefined,
    x: [2][]const u16 = .{ &.{}, &.{} },
    w: [2][]const u16 = .{ &.{}, &.{} },
    inv: [dim / 2]f32 = undefined,
    refs: [7]r.Ref = undefined,
    out0: usize = 0,
    pos: u32 = 0,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, k: Kind, label: []const u8) !Probe {
        const pipe = switch (k) {
            inline else => |tag| try mtl.Pipeline.init(device, lib, "ik2d_" ++ @tagName(tag), false),
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
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .served = if (p.kind == .qk_prep) &prep_shapes else &ctx_shapes,
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "128 short positive terms are followed by BF16 norm and RoPE stores" },
        } };
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = u.reset(&p.arena, &p.rig);
        p.s = s;
        const qk = p.kind == .qk_prep;
        p.pos = if (qk) 1150 else @intCast(s.k - 2);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (&p.inv, 0..) |*v, i| v.* = @floatCast(std.math.pow(f64, 1e6, -2 * @as(f64, @floatFromInt(i)) / dim));
        for (0..2) |i| {
            const heads = s.n * (if (i == 0 and qk) @as(usize, 3) else 1);
            const x = try a.alloc(u16, s.rows * heads * dim);
            if (fill == .stress) u.fillDyadic(rng, x, 32, 1) else u.fillDyadic(rng, x, 100, 64);
            if (fill == .nonfinite) {
                x[0] = u.nan_bits;
                x[dim + 1] = u.pinf_bits;
                x[2 * dim + 2] = u.ninf_bits;
            }
            p.x[i] = x;
            p.refs[i] = try p.rig.input(std.mem.sliceAsBytes(x));
        }
        for (0..if (qk) @as(usize, 2) else 1) |i| {
            const w = try a.alloc(u16, dim);
            for (w) |*v| v.* = u.word(0.5 + @as(f64, rng.float(f32)));
            p.w[i] = w;
            p.refs[2 + i] = try p.rig.input(std.mem.sliceAsBytes(w));
        }
        p.out0 = if (qk) 5 else 4;
        p.refs[p.out0 - 1] = try p.rig.input(std.mem.asBytes(&p.inv));
        for (0..2) |i| p.refs[p.out0 + i] = try p.rig.output(p.len(i) * 2);
        return true;
    }

    fn len(p: *Probe, i: usize) usize {
        return if (p.kind == .qk_prep) p.x[i].len else p.s.k * p.s.n * dim;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const groups: u32 = @intCast(p.s.n);
        const n: u32 = @intCast(p.s.rows);
        e.setPipeline(p.pipe);
        for (p.refs[0 .. p.out0 + 2], 0..) |x, i| e.setBuffer(x.buf, x.off, i);
        if (p.kind == .qk_prep) {
            e.setValue(Prep{ .rows = n, .pos0 = p.pos, .heads_q = groups * 3, .heads_kv = groups, .eps = eps }, 7);
            e.dispatchGroups(mtl.Size.of(groups * 4, n, 1), mtl.Size.of(dim, 1, 1));
        } else {
            for (0..2) |i| @memset(p.rig.bytes(p.refs[p.out0 + i], p.len(i) * 2), 0);
            e.setValue(Ctx{ .rows = n, .pos0 = p.pos, .ring = @intCast(p.s.k), .groups = groups, .eps = eps }, 6);
            e.dispatchGroups(mtl.Size.of(groups, n, 1), mtl.Size.of(dim, 1, 1));
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.len(0) + p.len(1);
    }

    fn calc(p: *Probe, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) void {
        @memset(out, 0);
        if (mag) |m| @memset(m, 0);
        const qk = p.kind == .qk_prep;
        for (0..2) |i| {
            const heads = p.s.n * (if (qk and i == 0) @as(usize, 3) else 1);
            const offset = if (i == 0) 0 else p.len(0);
            for (0..p.s.rows) |row| for (0..heads) |h| {
                const src = (row * heads + h) * dim;
                const slot = if (qk) row else (p.pos + row) % p.s.k;
                const at = offset + (slot * heads + h) * dim;
                const x = p.x[i][src..][0..dim];
                if (!qk and i == 1) {
                    for (x, 0..) |v, j| out[at + j] = u.val(v);
                    continue;
                }
                var squares: [dim]f64 = undefined;
                for (x, &squares) |v, *z| z.* = u.val(v) * u.val(v);
                const norm = 1 / @sqrt(sum(acc, order, &squares) / dim + eps);
                var w: [dim]f64 = undefined;
                for (x, &w, p.w[i]) |v, *z, gamma| z.* = u.bf(u.val(v) * norm * u.val(gamma));
                for (0..dim) |j| {
                    const pos: f32 = @floatFromInt(p.pos + row);
                    const angle32 = pos * p.inv[j % (dim / 2)];
                    const angle: f64 = angle32;
                    const a = w[j] * @cos(angle);
                    const b = w[j ^ (dim / 2)] * @sin(angle);
                    out[at + j] = a + if (j < dim / 2) -b else b;
                    if (mag) |m| m[at + j] = @abs(a) + @abs(b);
                }
            };
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        p.calc(.none, .sequential, want, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        p.calc(acc, order, out, null);
    }

    pub fn bounds(_: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        // A norm result near a BF16 midpoint can move one BF16 ulp; at RoPE the two terms can cancel.
        const f32_work = c.gamma(20, c.unit(.f32));
        for (allow, want, mag) |*limit, w, m| {
            const d = (2 * c.unit(.bf16) + f32_work) * m;
            limit.* = d + c.unit(.bf16) * (@abs(w) + d) + 0x1p-133;
        }
    }
};
