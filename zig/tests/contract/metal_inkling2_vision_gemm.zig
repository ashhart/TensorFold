//! Mini vision GEMMs: both orientations, three precision pairs, padded pitches, batches and tile tails.
const std = @import("std");
const mtl = @import("metal");
const rows = @import("contract").rows;
const u = @import("metal_inkling2_vision_util.zig");
const r = @import("rig.zig");
const stress = @import("metal_inkling2_vision_stress.zig");
const gm = @import("tensorfold").inkling2_metal_host.vision.gemm;
const c = u.c;
const g = u.g;
pub const Kind = gm.Kind;

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .bf16_nt => "metal inkling2 vision gemm bf16_nt",
        .bf16f_nt => "metal inkling2 vision gemm bf16f_nt",
        .f32_nt => "metal inkling2 vision gemm f32_nt",
        .f32_nn => "metal inkling2 vision gemm f32_nn",
    };
}

const tiles: [3]c.Tile = .{ .{ .size = 64 }, .{ .size = 128 }, .{ .size = 256, .min = 2 } };
const edges = [_]c.Shape{
    .{ .rows = 64, .n = 128, .k = 256 }, .{ .rows = 64, .n = 129, .k = 257 },
    .{ .rows = 65, .n = 128, .k = 255 }, .{ .rows = 65, .n = 129, .k = 257 },
};
const bf16_nt = [_]c.Shape{
    .{ .rows = 7, .n = 1024, .k = 1200 }, // patch embedding and projector rows of a one-token image
    .{ .rows = 8, .n = 4096, .k = 4096 },
    .{ .rows = 8, .n = 2304, .k = 4096 },
} ++ edges;
const bf16f_nt = [_]c.Shape{
    .{ .rows = 9, .n = 1024, .k = 1024 }, .{ .rows = 9, .n = 4096, .k = 1024 }, .{ .rows = 9, .n = 1024, .k = 4096 },
} ++ edges;
const f32_nt = [_]c.Shape{.{ .rows = 17, .n = 17, .k = 64 }} ++ edges;
const f32_nn = [_]c.Shape{.{ .rows = 17, .n = 64, .k = 17 }} ++ edges;

fn depth(s: c.Shape) usize {
    return s.k + 1;
}

pub const Probe = struct {
    arena: std.heap.ArenaAllocator,
    rig: r.Rig,
    gemm: *const gm.Gemm,
    kind: Kind,
    label: []const u8,
    s: gm.Shape = undefined,
    a: []const f32 = &.{},
    b: []const f32 = &.{},
    ar: r.Ref = undefined,
    br: r.Ref = undefined,
    out: r.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, gemm: *const gm.Gemm, k: Kind, label: []const u8) Probe {
        return .{ .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .gemm = gemm, .kind = k, .label = label };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.arena.deinit();
    }

    fn wide(p: *const Probe) bool {
        return p.kind == .f32_nt or p.kind == .f32_nn;
    }

    fn elem(p: *const Probe) c.Elem {
        return if (p.wide() or p.kind == .bf16f_nt) .f32 else .bf16;
    }

    fn nn(p: *const Probe) bool {
        return p.kind == .f32_nn;
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe, control_gate: ?c.Gate) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = control_gate, .spec = .{
            .name = p.label,
            .accum = .f32,
            .depth = depth,
            .out = p.elem(),
            .tiles = tiles,
            .served = switch (p.kind) {
                .bf16_nt => &bf16_nt,
                .bf16f_nt => &bf16f_nt,
                .f32_nt => &f32_nt,
                .f32_nn => &f32_nn,
            },
        } };
    }

    fn bIndex(p: *const Probe, col: usize, k: usize) usize {
        return if (p.nn()) k * p.s.ldb + col else col * p.s.ldb + k;
    }

    fn input(p: *Probe, a: std.mem.Allocator, values: []const f32) !r.Ref {
        if (p.wide()) return p.rig.input(std.mem.sliceAsBytes(values));
        const words = try a.alloc(u16, values.len);
        for (values, words) |v, *w| w.* = c.toBf16(v);
        return p.rig.input(std.mem.sliceAsBytes(words));
    }

    fn sample(p: *const Probe, fill: g.Fill, rng: std.Random, left: bool, k: usize, n: usize) f32 {
        if (fill == .stress) {
            const pair = stress.pair(k, n);
            return if (left) pair.left else pair.right;
        }
        // A bf16-dyadic fits in 10 fraction bits, so the M5 operand cut still matched. Wide inputs do not.
        if (p.wide()) return @bitCast(0x3f000000 | (rng.int(u32) & 0x007ffffe));
        return c.fromBf16(u.dyadic(rng, 255, 128));
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = u.reset(&p.arena, &p.rig);
        const lda = s.k + 4;
        const ldb = (if (p.nn()) s.n else s.k) + 4;
        const sa = s.rows * lda + 12;
        const sb = (if (p.nn()) s.k else s.n) * ldb + 12;
        p.s = .{ .m = @intCast(s.rows), .n = @intCast(s.n), .k = @intCast(s.k), .lda = @intCast(lda), .ldb = @intCast(ldb), .ldd = @intCast(s.n), .batch = if (p.wide()) 2 else 1, .sa = @intCast(sa), .sb = @intCast(sb), .sd = @intCast(s.rows * s.n) };
        const left = try a.alloc(f32, p.s.batch * sa);
        const right = try a.alloc(f32, p.s.batch * sb);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        @memset(left, std.math.nan(f32));
        @memset(right, std.math.nan(f32));
        for (0..p.s.batch) |b| {
            for (0..s.rows) |i| for (0..s.k) |k| {
                left[b * sa + i * lda + k] = sample(p, fill, rng, true, k, s.k);
            };
            for (0..s.n) |j| for (0..s.k) |k| {
                right[b * sb + p.bIndex(j, k)] = sample(p, fill, rng, false, k, s.k);
            };
        }
        if (fill == .nonfinite) {
            left[0] = std.math.nan(f32);
            left[lda + 1] = std.math.inf(f32);
            left[2 * lda + 2] = -std.math.inf(f32);
            left[3 * lda] = std.math.inf(f32);
            left[3 * lda + s.k - 1] = -std.math.inf(f32);
        }
        p.a = left;
        p.b = right;
        p.ar = try p.input(a, left);
        p.br = try p.input(a, right);
        p.out = try p.rig.output(p.count() * c.size(p.elem()));
        return true;
    }

    fn ref(x: r.Ref) gm.Ref {
        return .{ .buf = x.buf, .off = x.off };
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        p.gemm.run(pass.e, p.kind, ref(p.ar), ref(p.br), ref(p.out), p.s);
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return @as(usize, p.s.batch) * p.s.m * p.s.n;
    }

    fn product(p: *const Probe, b: usize, i: usize, j: usize, k: usize) f64 {
        return @as(f64, p.a[b * p.s.sa + i * p.s.lda + k]) * p.b[b * p.s.sb + p.bIndex(j, k)];
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (0..p.s.batch) |b| for (0..p.s.m) |i| for (0..p.s.n) |j| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (0..p.s.k) |k| {
                const v = p.product(b, i, j, k);
                sum += v;
                abs += @abs(v);
            }
            const o = b * p.s.sd + i * p.s.n + j;
            want[o] = sum;
            mag[o] = abs;
        };
    }

    pub fn simulate(p: *Probe, accum: c.Accum, order: g.Order, out: []f64) !void {
        const a = p.rig.a;
        const terms = try a.alloc(f64, p.s.k);
        defer a.free(terms);
        for (terms, 0..) |*v, k| v.* = p.product(0, 0, 0, k);
        @memset(out, rows.sumIn(accum, order, terms)); // every stress cell has the same dot
    }
};
