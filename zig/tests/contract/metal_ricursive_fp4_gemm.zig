//! rc_fp4_gemm_AB under the contract: a prompt chunk by the dequantized W on the tensor units, global before rounding.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const rc = @import("metal_ricursive.zig");
const fp4 = @import("metal_ricursive_fp4.zig");
const controls = @import("metal_ricursive_controls.zig");
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const family = rc.family;
const R = rc.R;

/// The tensor-op accumulation over the K tiles, the global, then the output rounding.
fn depth(s: c.Shape) usize {
    return s.k / 16 + 16;
}

/// A chunk at k 5,120 (gate and up) and the down's 40,960, whole or ragged tiles as the form takes them.
fn served(am: bool, an: bool) [2]c.Shape {
    const n: usize = if (an) 1024 else 1088;
    return .{ .{ .rows = if (am) 64 else 33, .n = n, .k = R.hidden }, .{ .rows = if (am) 128 else 100, .n = n, .k = R.intermediate } };
}

pub const Gemm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    am: bool,
    an: bool,
    mutation: controls.Mutation,
    name: []const u8,
    served: [2]c.Shape,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    global: f32 = fp4.global,
    d: ?Dense = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, am: bool, an: bool, mutation: controls.Mutation) !Gemm {
        const plain = try family.fp4GemmSource(device, a, true);
        defer a.free(plain);
        const src = try controls.apply(a, plain, mutation);
        defer a.free(src);
        const name = try std.fmt.allocPrint(a, "ricursive fp4 gemm rc_fp4_gemm_{d}{d}", .{ @intFromBool(am), @intFromBool(an) });
        errdefer a.free(name);
        const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, fun), .am = am, .an = an, .mutation = mutation, .name = name, .served = served(am, an) };
    }

    fn drop(p: *Gemm) void {
        if (p.d) |d| d.deinit();
        p.d = null;
        p.rig.reset();
    }

    pub fn deinit(p: *Gemm) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.a.free(p.name);
    }

    const vt = g.hooks(Gemm);

    pub fn kernel(p: *Gemm) g.Kernel {
        const tiles: [3]c.Tile = .{ .{ .size = 64, .step = 1, .min = 1 }, .{ .size = 128, .step = 1, .min = 1 }, .{ .size = 256, .step = 256, .min = 256 } };
        return controls.mark(.{ .spec = .{ .name = p.name, .accum = .f32, .depth = depth, .ops = 1, .out = .bf16, .tiles = tiles, .served = &p.served }, .ctx = p, .vt = &vt }, p.mutation);
    }

    /// The shape the kernel runs at: whole tiles where its form assumes them (the padded rows are the probe's too).
    fn padded(p: *const Gemm, s: c.Shape) c.Shape {
        return .{ .rows = if (p.am) std.mem.alignForward(usize, s.rows, 64) else s.rows, .n = if (p.an) std.mem.alignForward(usize, s.n, 128) else s.n, .k = s.k };
    }

    /// x as llama's GEMM probe fills it; W as rc_fp4_dequant leaves it (fp4 x scale, exact in bf16).
    pub fn prepare(p: *Gemm, s0: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s0.rows == 0 or s0.n == 0 or s0.k == 0 or s0.k % 256 != 0) return false;
        const s = p.padded(s0);
        if (s.rows > 2048) return false;
        p.drop();
        p.shape = s;
        p.global = if (fill == .stress) fp4.stress_global else fp4.global;
        const d = try Dense.init(p.a, s, 1);
        p.d = d;
        d.fillX(fill, seed, 2);
        const w = try fp4.pack(p.a, s.n, s.k, fill, seed, d.w);
        w.deinit();
        const w16 = try p.a.alloc(u16, s.n * s.k);
        defer p.a.free(w16);
        for (w16, d.w) |*o, v| o.* = c.toBf16(@floatCast(v));
        p.x = try p.rig.input(std.mem.sliceAsBytes(d.x));
        p.w = try p.rig.input(std.mem.sliceAsBytes(w16));
        p.y = try p.rig.output(d.outputs() * 2);
        return true;
    }

    pub fn count(p: *Gemm) usize {
        return p.d.?.outputs();
    }

    /// forward.zig's launch of the chunk's projection.
    pub fn run(p: *Gemm) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.x.buf, p.x.off, 0);
        e.setBuffer(p.w.buf, p.w.off, 1);
        const tn = (s.n + 127) / 128;
        const tm = (s.rows + 63) / 64;
        var prm: [16]i32 = @splat(0);
        const vals = [_]usize{ s.rows, s.n, s.k, s.k, s.k, s.n, tn, tm, 2, s.k / 256 };
        for (vals, 0..) |v, i| prm[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&prm), 2);
        e.setBuffer(p.y.buf, p.y.off, 3);
        e.setValue(p.global, 4);
        e.dispatchThreads(mtl.Size.of((tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Gemm, want: []f64, mag: []f64) !void {
        try p.d.?.reference(want, mag);
        for (want, mag) |*w, *m| {
            w.* *= p.global;
            m.* *= p.global;
        }
    }

    pub fn simulate(p: *Gemm, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.d.?.simulate(acc, order, out);
        for (out) |*o| o.* *= p.global;
    }
};
