//! Ripple's NVFP4 MLP under the contract: rc_fp4_matvec's rows (bf16 of global x fp32 sum) and rc_fp4_dequant (exact).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const rc = @import("metal_ricursive.zig");
const controls = @import("metal_ricursive_controls.zig");
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const family = rc.family;
const R = rc.R;

/// fp4.metal's rc_fp4_args, as forward.zig fills it.
pub const Args = extern struct { rows: u32, n: u32, k: u32, ldy: u32, global: f32 };

/// A served tensor's global (ModelOpt's, 5e-4 to 7e-4); gate 4's is a power of two, so its sums stay exact.
pub const global: f32 = 6.1e-4;
pub const stress_global: f32 = 0x1p-11;

/// E2M1's eight magnitudes; bit 3 is the sign.
const e2m1 = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

pub fn fp4(code: u4) f64 {
    const v = e2m1[code & 7];
    return if (code & 8 != 0) -v else v;
}

/// Unsigned E4M3 (OCP's e4m3fn without its sign): bias 7, three mantissa bits, subnormals m x 2^-9, 0x7f NaN.
pub fn e4m3(code: u8) f64 {
    if (code & 0x7f == 0x7f) return std.math.nan(f64);
    const e: i32 = (code >> 3) & 15;
    const m: f64 = @floatFromInt(code & 7);
    return if (e == 0) m * 0x1p-9 else std.math.ldexp(1 + m / 8, e - 7);
}

/// A matrix [n, k] in the release's layout: nibbles low first, 16 a block (two u32 words), one E4M3 code a block.
pub const Packed = struct {
    a: std.mem.Allocator,
    words: []u32,
    scales: []u8,

    pub fn deinit(p: Packed) void {
        p.a.free(p.words);
        p.a.free(p.scales);
    }
};

/// Random codes or gate 4's mirrored integers at scale 1, into `w` as fp4 x scale; edge scales and gate 1's NaN code.
pub fn pack(a: std.mem.Allocator, n: usize, k: usize, fill: g.Fill, seed: u64, w: []f64) !Packed {
    const words = try a.alloc(u32, n * k / 8);
    errdefer a.free(words);
    const scales = try a.alloc(u8, n * k / 16);
    @memset(words, 0);
    var prng = std.Random.DefaultPrng.init(seed ^ 0xf4f4);
    const rng = prng.random();
    const ints = [5]u4{ 2, 4, 5, 6, 7 }; // 1, 2, 3, 4, 6
    // every fifth row's first block: the largest scale (448), a subnormal one, zero; gate 1's rows 3, 8, ... a NaN
    const h = k / 2;
    for (0..n) |j| for (0..k / 16) |b| {
        const special = [3]u8{ 0x7e, 0x03, 0x00 };
        const nan_code = fill == .nonfinite and b == 1 and j % 5 == 3;
        scales[j * k / 16 + b] = if (fill == .stress) 0x38 else if (nan_code) 0x7f else if (b == 0 and j % 5 < 3) special[j % 5] else 0x30 + rng.uintLessThan(u8, 16);
    };
    for (0..n) |j| for (0..k) |i| {
        const code: u4 = if (fill != .stress) rng.int(u4) else if (i < h) ints[(i * 7 + j) % 5] else 8 | ints[((i - h) * 7 + j) % 5];
        words[(j * k + i) / 8] |= @as(u32, code) << @intCast(4 * (i % 8));
        w[j * k + i] = fp4(code) * e4m3(scales[(j * k + i) / 16]);
    };
    return .{ .a = a, .words = words, .scales = scales };
}

/// fp4.metal as model.zig compiles it, with the control's edit.
pub fn library(device: mtl.Device, a: std.mem.Allocator, m: controls.Mutation) !mtl.Library {
    const src = try controls.apply(a, sources.ricursive_fp4, m);
    defer a.free(src);
    return family.fp4Library(device, src);
}

/// A lane's dots over its blocks (four dot4s each), its running sum, the simd tree, then the global.
fn matvecDepth(s: c.Shape) usize {
    return s.k / 512 + 16;
}

/// rc_fp4_matvec_ROWS: the window's MLP projection, every launch at most ROWS rows (the rows past it NaN, unread).
pub const Matvec = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    per: u32,
    mutation: controls.Mutation,
    name: []const u8,
    served: [3]c.Shape,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    global: f32 = global,
    d: ?Dense = null,
    w: ?Packed = null,
    in: [3]rig_.Ref = undefined, // x padded to whole launches, words, scales
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, per: u32, mutation: controls.Mutation) !Matvec {
        const lib = try library(device, a, mutation);
        errdefer lib.deinit();
        const name = try std.fmt.allocPrint(a, "ricursive fp4 rc_fp4_matvec_{d}", .{per});
        errdefer a.free(name);
        const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
        const served = [3]c.Shape{ .{ .rows = per, .n = 1024, .k = R.hidden }, .{ .rows = per, .n = 1024, .k = R.intermediate }, .{ .rows = per / 2 + 1, .n = 1032, .k = R.hidden } };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .lib = lib, .pipe = try rc.pipelineOf(lib, device, fun), .per = per, .mutation = mutation, .name = name, .served = served };
    }

    fn drop(p: *Matvec) void {
        if (p.d) |d| d.deinit();
        if (p.w) |w| w.deinit();
        p.d = null;
        p.w = null;
        p.rig.reset();
    }

    pub fn deinit(p: *Matvec) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
        p.a.free(p.name);
    }

    const vt = g.hooks(Matvec);

    pub fn kernel(p: *Matvec) g.Kernel {
        const tiles: [3]c.Tile = .{ .{ .size = p.per, .step = 1, .min = 1 }, .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 512, .step = 16, .min = 16 } };
        return controls.mark(.{ .spec = .{ .name = p.name, .accum = .f32, .depth = matvecDepth, .ops = 1, .out = .bf16, .tiles = tiles, .served = &p.served }, .ctx = p, .vt = &vt }, p.mutation);
    }

    pub fn prepare(p: *Matvec, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or s.k == 0 or s.k % 16 != 0) return false;
        p.drop();
        p.shape = s;
        p.global = if (fill == .stress) stress_global else global;
        const d = try Dense.init(p.a, s, 1);
        p.d = d;
        d.fillX(fill, seed, 2);
        p.w = try pack(p.a, s.n, s.k, fill, seed, d.w);
        const padded = try p.a.alloc(u16, std.mem.alignForward(usize, s.rows, p.per) * s.k);
        defer p.a.free(padded);
        @memset(padded, c.toBf16(std.math.nan(f32)));
        @memcpy(padded[0..d.x.len], d.x);
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(padded));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.w.?.words));
        p.in[2] = try p.rig.input(p.w.?.scales);
        p.y = try p.rig.output(d.outputs() * 2);
        return true;
    }

    pub fn count(p: *Matvec) usize {
        return p.d.?.outputs();
    }

    pub fn run(p: *Matvec) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        var at: usize = 0;
        while (at < s.rows) : (at += p.per) {
            const here: u32 = @intCast(@min(p.per, s.rows - at));
            e.setBuffer(p.in[0].buf, p.in[0].off + at * s.k * 2, 0);
            e.setBuffer(p.in[1].buf, p.in[1].off, 1);
            e.setBuffer(p.in[2].buf, p.in[2].off, 2);
            e.setValue(Args{ .rows = here, .n = @intCast(s.n), .k = @intCast(s.k), .ldy = @intCast(s.n), .global = p.global }, 3);
            e.setBuffer(p.y.buf, p.y.off + at * s.n * 2, 4);
            e.dispatchGroups(mtl.Size.of((s.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        }
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Matvec, want: []f64, mag: []f64) !void {
        try p.d.?.reference(want, mag);
        for (want, mag) |*w, *m| {
            w.* *= p.global;
            m.* *= p.global;
        }
    }

    pub fn simulate(p: *Matvec, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.d.?.simulate(acc, order, out);
        for (out) |*o| o.* *= p.global;
    }
};

/// rc_fp4_dequant: a prompt's W as bf16, each value fp4 x scale, exact (no global: the GEMM applies it).
pub const Dequant = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    mutation: controls.Mutation,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    w: ?Packed = null,
    values: []f64 = &.{},
    in: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mutation: controls.Mutation) !Dequant {
        const lib = try library(device, a, mutation);
        errdefer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .lib = lib, .pipe = try rc.pipelineOf(lib, device, "rc_fp4_dequant"), .mutation = mutation };
    }

    fn drop(p: *Dequant) void {
        if (p.w) |w| w.deinit();
        p.w = null;
        p.a.free(p.values);
        p.values = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Dequant) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
    }

    const vt = g.hooks(Dequant);

    /// A shape's rows are W's rows (forward's n), k its columns.
    pub fn kernel(p: *Dequant) g.Kernel {
        const served: []const c.Shape = &.{ .{ .rows = 64, .n = 1, .k = R.hidden }, .{ .rows = 16, .n = 1, .k = R.intermediate } };
        const tiles: [3]c.Tile = .{ .{ .size = 4, .step = 1, .min = 1 }, .{}, .{ .size = 1024, .step = 16, .min = 16 } };
        return controls.mark(.{ .spec = .{ .name = "ricursive fp4 rc_fp4_dequant", .accum = .none, .depth = rc.none, .out = .bf16, .tiles = tiles, .served = served, .sums = rc.no_sums }, .ctx = p, .vt = &vt }, p.mutation);
    }

    pub fn prepare(p: *Dequant, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n != 1 or s.k == 0 or s.k % 16 != 0) return false;
        p.drop();
        p.shape = s;
        p.values = try p.a.alloc(f64, s.rows * s.k);
        p.w = try pack(p.a, s.rows, s.k, fill, seed, p.values);
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.w.?.words));
        p.in[1] = try p.rig.input(p.w.?.scales);
        p.out = try p.rig.output(s.rows * s.k * 2);
        return true;
    }

    pub fn count(p: *Dequant) usize {
        return p.shape.rows * p.shape.k;
    }

    /// forward.zig's launch: the args carry the chunk's rows and the tensor's real global, which the kernel ignores.
    pub fn run(p: *Dequant) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.in[0].buf, p.in[0].off, 0);
        e.setBuffer(p.in[1].buf, p.in[1].off, 1);
        e.setValue(Args{ .rows = 100, .n = @intCast(s.rows), .k = @intCast(s.k), .ldy = @intCast(s.rows), .global = global }, 2);
        e.setBuffer(p.out.buf, p.out.off, 3);
        e.dispatchThreads(mtl.Size.of(s.k / 16, s.rows, 1), mtl.Size.of(64, 4, 1));
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Dequant, want: []f64, mag: []f64) !void {
        @memcpy(want, p.values);
        for (mag, p.values) |*m, v| m.* = @abs(v);
    }

    /// Every value is exact in bf16 (at most six significant bits): no error allowed.
    pub fn bounds(_: *Dequant, _: []const f64, _: []const f64, allow: []f64) !void {
        @memset(allow, 0);
    }
};
