//! LFM2's fused BF16 gate/up projection with SwiGLU replaces two row projections and the activation, bit for bit.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const controls = @import("metal_lfm2_controls.zig");

const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;

pub const name = "metal lfm2 bf16 glu rows lfm2_bf16_glu_rows";

/// The drafter's gate and up matrices (6,144 from 2,048) at the row counts a block or a shared hold encodes.
const served = [_]c.Shape{
    .{ .rows = 1, .n = 6144, .k = 2048 }, .{ .rows = 4, .n = 6144, .k = 2048 },
    .{ .rows = 9, .n = 6144, .k = 2048 }, .{ .rows = 16, .n = 6144, .k = 2048 },
};
const max_rows = 16;
/// Spec depth of the dots, as lfm2_bf16_rows: a lane's k / 32 fmas in turn, then the simdgroup's five-level sum.
fn depth(s: c.Shape) usize {
    return s.k / 32 + 8;
}

pub fn spec() c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .tiles = .{ .{ .size = 16 }, .{ .size = 8 }, .{ .size = 256, .step = 8, .min = 8 } },
        .served = &served,
        .replaces = "LFM2 BF16 gate and up rows followed by SwiGLU",
        .nan = .{ .stated = "outputs equal the unfused rows and SwiGLU, NaN and infinity classes included" },
        .sums = .{ .why = "the gate and up dots are exact integers here; SwiGLU maps a wrong sum to a wrong bf16 output" },
    };
}

const Args = extern struct { rows: u32, n: u32, k: u32 };

/// Gate 4's reference: SiLU(g) * u in float64 on the exact (or simulated) sums.
fn swiglu(gate: f64, up: f64) f64 {
    return gate / (1 + @exp(-gate)) * up;
}

/// A bf16 output the unfused kernels gave: the same bits, or the same non-finite class.
fn agrees(want: u16, got: u16) bool {
    if (want & 0x7f80 != 0x7f80) return want == got;
    if (got & 0x7f80 != 0x7f80) return false;
    const nan = want & 0x7f != 0;
    return nan == (got & 0x7f != 0) and (nan or want == got);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    mutation: controls.Mutation,
    fused: mtl.Pipeline,
    rows: mtl.Pipeline,
    act: mtl.Pipeline,
    gate_d: ?Dense = null,
    up_d: ?Dense = null,
    x: rig_.Ref = undefined,
    wg: rig_.Ref = undefined,
    wu: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    bg: rig_.Ref = undefined,
    bu: rig_.Ref = undefined,
    bact: rig_.Ref = undefined,
    /// Gate 4's inputs make every sum exact in fp32, so only the activation's own roundings are allowed.
    exact: bool = false,
    sums: [4][]f64 = .{ &.{}, &.{}, &.{}, &.{} },

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: controls.Mutation) !Probe {
        const text = try controls.source(a, m);
        defer if (m != .none) a.free(text);
        const fused = try rig_.pipeline(device, text, "lfm2_bf16_glu_rows");
        errdefer fused.deinit();
        const rows = try rig_.pipeline(device, sources.lfm2, "lfm2_bf16_rows");
        errdefer rows.deinit();
        const act = try rig_.pipeline(device, sources.lfm2, "lfm2_swiglu");
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .mutation = m, .fused = fused, .rows = rows, .act = act };
    }

    fn drop(p: *Probe) void {
        if (p.gate_d) |d| d.deinit();
        if (p.up_d) |d| d.deinit();
        p.gate_d = null;
        p.up_d = null;
        for (&p.sums) |*s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        p.fused.deinit();
        p.rows.deinit();
        p.act.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        var s = spec();
        if (p.mutation != .none) s.name = p.mutation.name();
        return .{ .spec = s, .ctx = p, .vt = &vt, .control = p.mutation.gate() };
    }

    /// bf16 W [n, k]: small random values, or gate 4's mirrored integers whose halves cancel (partials under 2^24).
    fn weights(d: Dense, bits: []u16, fill: g.Fill, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = d.shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else (rng.float(f32) * 2 - 1) * 0x1p-4);
        if (fill == .stress) for (0..d.shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*wv, b| wv.* = c.fromBf16(b);
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        // The kernel serves at most 16 rows; the baseline's activation works in whole 8-wide steps.
        if (shape.rows == 0 or shape.rows > max_rows or shape.n == 0 or shape.k == 0) return false;
        if (shape.k % 8 != 0 or shape.rows * shape.n % 8 != 0) return false;
        p.drop();
        const gd = try Dense.init(p.a, shape, 1);
        p.gate_d = gd;
        const ud = try Dense.init(p.a, shape, 1);
        p.up_d = ud;
        gd.fillX(fill, seed, 2);
        @memcpy(ud.x, gd.x);
        const bits = try p.a.alloc(u16, 2 * shape.n * shape.k);
        defer p.a.free(bits);
        weights(gd, bits[0 .. shape.n * shape.k], fill, seed ^ 0x6d);
        weights(ud, bits[shape.n * shape.k ..], fill, seed ^ 0x75);
        p.exact = fill == .stress;
        const n = gd.outputs();
        p.x = try p.rig.input(std.mem.sliceAsBytes(gd.x));
        p.wg = try p.rig.input(std.mem.sliceAsBytes(bits[0 .. shape.n * shape.k]));
        p.wu = try p.rig.input(std.mem.sliceAsBytes(bits[shape.n * shape.k ..]));
        p.out = try p.rig.output(n * 2);
        p.bg = try p.rig.scratch(n * 2);
        p.bu = try p.rig.scratch(n * 2);
        p.bact = try p.rig.scratch(n * 2);
        return true;
    }

    fn args(p: *const Probe) Args {
        const s = p.gate_d.?.shape;
        return .{ .rows = @intCast(s.rows), .n = @intCast(s.n), .k = @intCast(s.k) };
    }

    fn groups(p: *const Probe) mtl.Size {
        const s = p.gate_d.?.shape;
        return mtl.Size.of((s.n + 7) / 8, (s.rows + 15) / 16, 1);
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.fused);
        e.setBuffer(p.x.buf, p.x.off, 0);
        e.setBuffer(p.wg.buf, p.wg.off, 1);
        const a = p.args();
        e.setBytes(std.mem.asBytes(&a), 2);
        e.setBuffer(p.out.buf, p.out.off, 3);
        e.setBuffer(p.wu.buf, p.wu.off, 4);
        e.dispatchGroups(p.groups(), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.gate_d.?.outputs();
    }

    /// The unfused pair: both projections through lfm2_bf16_rows into bf16 scratch, then lfm2_swiglu.
    pub fn replaced(p: *Probe) ![]const u8 {
        const n = p.count();
        const pass = p.rig.begin();
        const e = pass.e;
        const a = p.args();
        for ([_][2]rig_.Ref{ .{ p.wg, p.bg }, .{ p.wu, p.bu } }) |pair| {
            e.setPipeline(p.rows);
            e.setBuffer(p.x.buf, p.x.off, 0);
            e.setBuffer(pair[0].buf, pair[0].off, 1);
            e.setBytes(std.mem.asBytes(&a), 2);
            e.setBuffer(pair[1].buf, pair[1].off, 3);
            e.dispatchGroups(p.groups(), mtl.Size.of(256, 1, 1));
        }
        e.setPipeline(p.act);
        e.setBuffer(p.bg.buf, p.bg.off, 0);
        e.setBuffer(p.bu.buf, p.bu.off, 1);
        e.setBuffer(p.bact.buf, p.bact.off, 2);
        e.setValue(@as(u32, @intCast(n)), 3);
        e.dispatchThreads(mtl.Size.of(n / 8, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
        return p.rig.bytes(p.bact, n * 2);
    }

    /// Gate 1: the unfused pair's bits, with a NaN or infinity only where the pair has the same class.
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const got = try p.rig.outputBytes();
        const want = try p.replaced();
        for (0..p.count()) |i| {
            const a = std.mem.readInt(u16, want[2 * i ..][0..2], .little);
            const b = std.mem.readInt(u16, got[2 * i ..][0..2], .little);
            if (!agrees(a, b)) return "fused outputs differ from the unfused rows and SwiGLU";
        }
        return null;
    }

    fn sumsOf(p: *Probe, which: usize) ![]f64 {
        if (p.sums[which].len == 0) p.sums[which] = try p.a.alloc(f64, p.count());
        return p.sums[which];
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const gx = try p.sumsOf(0);
        const ux = try p.sumsOf(1);
        try p.gate_d.?.reference(gx, try p.sumsOf(2));
        try p.up_d.?.reference(ux, try p.sumsOf(3));
        for (want, mag, gx, ux, p.sums[2], p.sums[3]) |*w, *m, gv, uv, gm, um| {
            w.* = swiglu(gv, uv);
            m.* = @abs(gv) * um + @abs(uv) * gm;
        }
    }

    /// Include SiLU's 1.1-Lipschitz factor and bf16 rounding at each gate, sigmoid, SiLU, up and product boundary.
    pub fn bounds(p: *Probe, _: []const f64, _: []const f64, allow: []f64) !void {
        const unit = 0x1p-8;
        const gamma = c.gamma(@floatFromInt(depth(p.gate_d.?.shape)), 0x1p-24);
        for (allow, p.sums[0], p.sums[1], p.sums[2], p.sums[3]) |*l, gv, uv, gm, um| {
            const eg = unit * @abs(gv) + if (p.exact) 0 else gamma * gm * (1 + unit);
            const eu = unit * @abs(uv) + if (p.exact) 0 else gamma * um * (1 + unit);
            const silu = @abs(gv) / (1 + @exp(-gv)) + 1.1 * eg;
            const mult = @abs(uv) + eu;
            l.* = 1.1 * mult * eg + silu * eu + silu * mult * 4 * unit + 0x1p-120;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const gs = try p.a.alloc(f64, 2 * out.len);
        defer p.a.free(gs);
        try p.gate_d.?.simulate(acc, order, gs[0..out.len]);
        try p.up_d.?.simulate(acc, order, gs[out.len..]);
        for (out, gs[0..out.len], gs[out.len..]) |*o, gv, uv| o.* = swiglu(gv, uv);
    }
};
