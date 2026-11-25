//! H3's products under the contract: the 32x64-warp bf16 GEMM bit-equal to MLX's that it replaces, its int8 form
//! (fc1) against float64 at its scales, and the row quantizer that feeds it.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const h3k = @import("h3_kernels");
const rig_ = @import("rig.zig");
const keep = @import("metal_h3_image_common.zig").keep;
const parts = @import("metal_h3_image_decoder_parts.zig");

const c = contract.spec;
const g = contract.gates;

/// The served sizes: hidden, attention width, MLP width; a 2048 picture's 4,096 rows and a ragged tile past them.
const hidden = 5376;
const inner = 7168;
const mlp = 14336;
/// fc1's weights' scale group, the int8 product's K block.
const block = 256;
/// The engine's walk: 2^6 row tiles under one weight tile.
const swizzle = 6;
/// The deepest K a reference takes, on its worker's stack: the MLP's.
const most_k = mlp;

/// H3's library as source.zig builds it at load: gemm_nax with its fragments, then H3's kernels.
fn library(a: std.mem.Allocator, device: mtl.Device) !mtl.Library {
    const nax = for (sources.prefill) |f| {
        if (std.mem.eql(u8, f.name, "gemm_nax")) break f.text;
    } else return error.NoGemmNax;
    const gemm = try std.mem.replaceOwned(u8, a, nax, "#include \"../nax.h\"", sources.nax);
    defer a.free(gemm);
    // source.zig's order at load: gemm_nax, the model's sizes over `source`, then `gemm_source` (the products)
    const text = try std.mem.concat(a, u8, &.{ gemm, "#define H3_HEADS 56\n#define H3_SLOTS 18432\n#define H3_TILES 288\n", h3k.source, "\n", h3k.gemm_source });
    defer a.free(text);
    return mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
}

/// block.zig's product launch: P = rows, n, k, lda, ldb, ldd, column tiles, row tiles, swizzle, K blocks.
fn launch(e: mtl.ComputeEncoder, s: c.Shape, swz: usize, threads: usize) void {
    const band = @as(usize, 1) << @intCast(swz);
    const tn = s.n / 128;
    const tm = (s.rows + 63) / 64;
    var p: [16]i32 = @splat(0);
    for ([_]usize{ s.rows, s.n, s.k, s.k, s.k, s.n, tn, tm, swz, s.k / block }, 0..) |v, i| p[i] = @intCast(v);
    e.setBytes(std.mem.asBytes(&p), 2);
    e.dispatchGroups(mtl.Size.of(tn * band, (tm + band - 1) / band, 1), mtl.Size.of(32, 4, threads / 128));
}

const Form = enum { whole, ragged, int8 };

/// One product registration: its form's pipeline, MLX's for the bf16 forms' baseline, and the prepared inputs.
pub const Product = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    form: Form,
    pipe: mtl.Pipeline,
    mlx: ?mtl.Pipeline,
    shape: c.Shape = undefined,
    /// Rows the reference computes; the whole form's padding rows past them are zero.
    live: usize = 0,
    arena: std.heap.ArenaAllocator,
    bufs: [5]rig_.Ref = undefined,
    /// Host copies: bf16 operands widened, or int8 operands and their scales.
    x: []f32 = &.{},
    w: []f32 = &.{},
    xq: []i8 = &.{},
    wq: []i8 = &.{},
    sx: []f32 = &.{},
    sw: []f32 = &.{},
    baseline: []u8 = &.{},

    fn name(f: Form) []const u8 {
        return switch (f) {
            .whole => "metal h3 product 32x64 h3_gemm_32x64",
            .ragged => "metal h3 product 32x64 ragged h3_gemm_32x64_ragged",
            .int8 => "metal h3 product 32x64 int8 h3_gemm_32x64_i8",
        };
    }

    fn bf16Depth(s: c.Shape) usize {
        return s.k;
    }

    /// Each K block's exact int32 sum scaled and folded into fp32, then the row scale.
    fn int8Depth(s: c.Shape) usize {
        return 2 * (s.k / block) + 1;
    }

    pub fn spec(f: Form) c.Spec {
        return switch (f) {
            .whole => .{ .name = name(f), .accum = .f32, .depth = bf16Depth, .out = .bf16, .replaces = "MLX's 64x128 gemm_nax (whole)", .tiles = .{ .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 256, .step = 32, .min = 32 } }, .served = &.{ .{ .rows = 4096, .n = inner, .k = hidden }, .{ .rows = 4096, .n = 3 * inner, .k = hidden }, .{ .rows = 4096, .n = hidden, .k = mlp } } },
            .ragged => .{ .name = name(f), .accum = .f32, .depth = bf16Depth, .out = .bf16, .replaces = "MLX's 64x128 gemm_nax (ragged)", .tiles = .{ .{ .size = 64 }, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 256, .step = 32, .min = 32 } }, .served = &.{ .{ .rows = 4096 + 24, .n = inner, .k = hidden }, .{ .rows = 4096 + 24, .n = hidden, .k = inner } } },
            .int8 => .{ .name = name(f), .accum = .f32, .depth = int8Depth, .ops = 1, .out = .bf16, .served = &.{ .{ .rows = 4096 + 24, .n = 2 * mlp, .k = hidden }, .{ .rows = 4096, .n = 2 * mlp, .k = hidden } }, .sums = .{ .why = "each K block's int8 products summed exactly in int32, the blocks folded in fp32 at their scales" } },
        };
    }

    fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, f: Form) !Product {
        const pipe = try mtl.Pipeline.init(device, lib, switch (f) {
            .whole => h3k.gemm_32x64,
            .ragged => h3k.gemm_32x64_ragged,
            .int8 => h3k.gemm_32x64_i8,
        }, false);
        errdefer pipe.deinit();
        const mlx: ?mtl.Pipeline = switch (f) {
            .whole => try mtl.Pipeline.init(device, lib, h3k.gemm_whole, false),
            .ragged => try mtl.Pipeline.init(device, lib, h3k.gemm_ragged, false),
            .int8 => null,
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .form = f, .pipe = pipe, .mlx = mlx, .arena = std.heap.ArenaAllocator.init(a) };
    }

    fn deinit(t: *Product) void {
        t.rig.deinit();
        t.arena.deinit();
        t.pipe.deinit();
        if (t.mlx) |p| p.deinit();
    }

    pub fn kernel(t: *Product) g.Kernel {
        return .{ .spec = spec(t.form), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Product);

    pub fn prepare(t: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        const kb: usize = if (t.form == .int8) block else 32;
        if (shape.rows == 0 or shape.n % 128 != 0 or shape.k == 0 or shape.k % kb != 0 or shape.k > most_k) return false;
        // the whole form launches whole row tiles: the single-shape gates' few rows are padded with zero rows
        t.live = shape.rows;
        t.shape = shape;
        if (t.form == .whole) t.shape.rows = std.mem.alignForward(usize, shape.rows, 64);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const al = t.arena.allocator();
        const s = t.shape;
        if (t.form == .int8) try t.int8Inputs(al, rng, fill) else try t.bf16Inputs(al, rng, fill);
        t.bufs[4] = try t.rig.output(s.rows * s.n * 2);
        if (t.mlx) |p| {
            // the replaced kernel's bytes on the same inputs, MLX's 32x32 band of 4 a group of 256
            const out = try t.rig.scratch(s.rows * s.n * 2);
            const pass = t.rig.begin();
            pass.e.setPipeline(p);
            pass.e.setBuffer(t.bufs[0].buf, t.bufs[0].off, 0);
            pass.e.setBuffer(t.bufs[1].buf, t.bufs[1].off, 1);
            pass.e.setBuffer(out.buf, out.off, 3);
            launch(pass.e, s, 2, 256);
            try t.rig.finish(pass);
            t.baseline = try al.dupe(u8, t.rig.bytes(out, s.rows * s.n * 2));
        }
        return true;
    }

    fn bf16Inputs(t: *Product, al: std.mem.Allocator, rng: std.Random, fill: g.Fill) !void {
        const s = t.shape;
        const xb = try al.alloc(u16, s.rows * s.k);
        const wb = try al.alloc(u16, s.n * s.k);
        @memset(xb, 0);
        const live = xb[0 .. t.live * s.k];
        if (fill == .stress) {
            // integers, k's second half the first's negation plus 1 on 4: every exact sum is small, its terms are not
            const h = s.k / 2;
            for (0..t.live) |r| for (0..h) |i| {
                const v = rng.intRangeAtMost(i32, 1, 48);
                live[r * s.k + i] = parts.integer(v);
                live[r * s.k + h + i] = parts.integer(v);
            };
            for (0..s.n) |o| for (0..h) |i| {
                const v = rng.intRangeAtMost(i32, 0, 48);
                wb[o * s.k + i] = parts.integer(v);
                wb[o * s.k + h + i] = parts.integer(-v + @as(i32, if (i < 4) 1 else 0));
            };
        } else {
            parts.uniform(rng, live, -2, 2);
            const lim: f32 = @floatCast(@sqrt(3.0 / @as(f64, @floatFromInt(s.k))));
            parts.uniform(rng, wb, -lim, lim);
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, s.k / 3 }, .{ 1, s.k / 2 }, .{ 2, s.k - 1 }, .{ 3, 0 }, .{ 3, s.k - 1 } };
            for (at, parts.plants) |p, v| if (p[0] < t.live) {
                live[p[0] * s.k + p[1]] = v;
            };
        }
        t.x = try al.alloc(f32, xb.len);
        for (t.x, xb) |*o, v| o.* = c.fromBf16(v);
        t.w = try al.alloc(f32, wb.len);
        for (t.w, wb) |*o, v| o.* = c.fromBf16(v);
        t.bufs[0] = try t.rig.input(std.mem.sliceAsBytes(xb));
        t.bufs[1] = try t.rig.input(std.mem.sliceAsBytes(wb));
    }

    fn int8Inputs(t: *Product, al: std.mem.Allocator, rng: std.Random, fill: g.Fill) !void {
        const s = t.shape;
        const kb = s.k / block;
        t.xq = try al.alloc(i8, s.rows * s.k);
        t.wq = try al.alloc(i8, s.n * s.k);
        t.sx = try al.alloc(f32, s.rows);
        t.sw = try al.alloc(f32, s.n * kb);
        if (fill == .stress) {
            // |x w| up to 225: a block's sum passes fp16's range; each row's second half cancels its first but for 4
            const h = s.k / 2;
            for (0..s.rows) |r| for (0..h) |i| {
                const v = rng.intRangeAtMost(i8, 1, 15);
                t.xq[r * s.k + i] = v;
                t.xq[r * s.k + h + i] = v;
            };
            for (0..s.n) |o| for (0..h) |i| {
                const v = rng.intRangeAtMost(i8, 0, 15);
                t.wq[o * s.k + i] = v;
                t.wq[o * s.k + h + i] = -v + @as(i8, if (i < 4) 1 else 0);
            };
            @memset(t.sx, 1);
            @memset(t.sw, 1);
        } else {
            for (t.xq) |*v| v.* = rng.intRangeAtMost(i8, -127, 127);
            for (t.wq) |*v| v.* = rng.intRangeAtMost(i8, -127, 127);
            for (t.sx) |*v| v.* = 0x1p-7 * (1 + rng.float(f32));
            for (t.sw) |*v| v.* = 0x1p-12 * (1 + rng.float(f32));
        }
        if (fill == .nonfinite) {
            // int8 holds no NaN: the quantizer's scales carry it, a row's or one column's K block
            const nan = std.math.nan(f32);
            const inf = std.math.inf(f32);
            t.sx[0] = nan;
            t.sx[1] = inf;
            t.sx[2] = -inf;
            t.sw[5 * kb] = nan;
            t.sw[9 * kb + kb - 1] = inf;
        }
        t.bufs[0] = try t.rig.input(std.mem.sliceAsBytes(t.xq));
        t.bufs[1] = try t.rig.input(std.mem.sliceAsBytes(t.wq));
        t.bufs[2] = try t.rig.input(std.mem.sliceAsBytes(t.sx));
        t.bufs[3] = try t.rig.input(std.mem.sliceAsBytes(t.sw));
    }

    pub fn run(t: *Product) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.bufs[0].buf, t.bufs[0].off, 0);
        p.e.setBuffer(t.bufs[1].buf, t.bufs[1].off, 1);
        p.e.setBuffer(t.bufs[4].buf, t.bufs[4].off, 3);
        if (t.form == .int8) {
            p.e.setBuffer(t.bufs[2].buf, t.bufs[2].off, 4);
            p.e.setBuffer(t.bufs[3].buf, t.bufs[3].off, 5);
        }
        launch(p.e, t.shape, swizzle, 128);
        try t.rig.finish(p);
    }

    pub fn replaced(t: *Product) ![]const u8 {
        return t.baseline;
    }

    pub fn count(t: *Product) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Output (r, o)'s terms: bf16 products, or each K block's exact int32 sum at its column scale.
    fn terms(t: *const Product, r: usize, o: usize, out: []f64) []f64 {
        const s = t.shape;
        if (t.form != .int8) {
            for (out[0..s.k], t.x[r * s.k ..][0..s.k], t.w[o * s.k ..][0..s.k]) |*p, a, b| p.* = @as(f64, a) * b;
            return out[0..s.k];
        }
        const kb = s.k / block;
        for (out[0..kb], 0..) |*p, b| {
            var sum: i64 = 0;
            for (t.xq[r * s.k + b * block ..][0..block], t.wq[o * s.k + b * block ..][0..block]) |x, w| sum += @as(i32, x) * w;
            p.* = @as(f64, @floatFromInt(sum)) * t.sw[o * kb + b];
        }
        return out[0..kb];
    }

    fn rowScale(t: *const Product, r: usize) f64 {
        return if (t.form == .int8) t.sx[r] else 1;
    }

    const Job = struct {
        t: *const Product,
        acc: ?c.Accum,
        order: g.Order,
        want: []f64,
        mag: []f64,

        fn rows(j: Job, from: usize, to: usize) void {
            const t = j.t;
            var buf: [most_k]f64 = undefined;
            for (from..to) |r| for (0..t.shape.n) |o| {
                const at = r * t.shape.n + o;
                const ts = t.terms(r, o, &buf);
                var v: f64 = 0;
                var m: f64 = 0;
                if (j.acc) |acc| v = contract.rows.sumIn(acc, j.order, ts) else for (ts) |x| {
                    v += x;
                    m += @abs(x);
                }
                const sc = t.rowScale(r);
                j.want[at] = v * sc;
                j.mag[at] = m * @abs(sc);
            };
        }
    };

    fn outputs(t: *Product, acc: ?c.Accum, order: g.Order, want: []f64, mag: []f64) void {
        // the whole form's padding rows are zero rows: their outputs are zero
        @memset(want, 0);
        @memset(mag, 0);
        parts.parallel(t.live, Job{ .t = t, .acc = acc, .order = order, .want = want, .mag = mag }, Job.rows);
    }

    pub fn reference(t: *Product, want: []f64, mag: []f64) !void {
        t.outputs(null, .sequential, want, mag);
    }

    pub fn simulate(t: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        t.outputs(acc, order, out, mag);
    }
};

/// h3_quantize_rows: each group's scale is its absolute max over 127 (1 for a zero group), each value rint(x / scale).
pub const Quantize = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    arena: std.heap.ArenaAllocator,
    group: usize = 0,
    x: []f32 = &.{},
    bufs: [3]rig_.Ref = undefined,

    const label = "metal h3 product row quantizer h3_quantize_rows";

    fn none(_: c.Shape) usize {
        return 0;
    }

    /// Rows: rows; n: groups a row; k: a group's values. Served: a 2048 picture's activations a row, fc1's weights per 256.
    pub fn spec() c.Spec {
        return .{ .name = label, .accum = .none, .depth = none, .out = .raw, .nan = .{ .stated = "a group holding NaN or an infinity takes a non-finite scale; its values are unspecified" }, .tiles = .{ .{ .size = 64 }, .{}, .{ .size = 256, .step = 32, .min = 32 } }, .served = &.{ .{ .rows = 4096 + 24, .n = 1, .k = hidden }, .{ .rows = 2 * mlp, .n = hidden / block, .k = block } } };
    }

    fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library) !Quantize {
        const pipe = try mtl.Pipeline.init(device, lib, "h3_quantize_rows", false);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .arena = std.heap.ArenaAllocator.init(a) };
    }

    fn deinit(t: *Quantize) void {
        t.rig.deinit();
        t.arena.deinit();
        t.pipe.deinit();
    }

    pub fn kernel(t: *Quantize) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Quantize);

    pub fn prepare(t: *Quantize, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        t.shape = shape;
        const k = shape.n * shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const xb = try t.arena.allocator().alloc(u16, shape.rows * k);
        // magnitudes that differ by row and group, one group all zeros
        for (xb, 0..) |*v, i| v.* = c.toBf16((prng.random().float(f32) * 2 - 1) * std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(i / shape.k % 9)) - 4)));
        if (shape.rows > 3) @memset(xb[3 * k ..][0..shape.k], 0);
        if (fill == .nonfinite) {
            xb[0 * k + shape.k / 3] = parts.nan16;
            xb[1 * k + k - 1] = parts.inf16;
            xb[2 * k] = parts.ninf16;
        }
        t.x = try t.arena.allocator().alloc(f32, xb.len);
        for (t.x, xb) |*o, v| o.* = c.fromBf16(v);
        t.bufs[0] = try t.rig.input(std.mem.sliceAsBytes(xb));
        t.bufs[1] = try t.rig.output(shape.rows * k);
        t.bufs[2] = try t.rig.output(shape.rows * shape.n * 4);
        return true;
    }

    pub fn run(t: *Quantize) !void {
        const s = t.shape;
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.bufs[0].buf, t.bufs[0].off, 0);
        p.e.setValue([3]i32{ @intCast(s.rows), @intCast(s.n * s.k), @intCast(s.k) }, 1);
        p.e.setBuffer(t.bufs[1].buf, t.bufs[1].off, 2);
        p.e.setBuffer(t.bufs[2].buf, t.bufs[2].off, 3);
        p.e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Quantize) usize {
        return t.shape.rows * t.shape.n * (t.shape.k + 4);
    }

    pub fn reference(_: *Quantize, _: []f64, _: []f64) !void {
        return error.RawOutputChecksItself;
    }

    /// Every group against its formula: the scale to fp32's division of the max, every value its nearest step.
    pub fn check(t: *Quantize, _: g.Gate) !?[]const u8 {
        const s = t.shape;
        const q = std.mem.bytesAsSlice(i8, t.rig.bytes(t.bufs[1], s.rows * s.n * s.k));
        const sc: []align(1) const f32 = std.mem.bytesAsSlice(f32, t.rig.bytes(t.bufs[2], s.rows * s.n * 4));
        for (sc, 0..) |scale, gi| {
            const xs = t.x[gi * s.k ..][0..s.k];
            var most: f32 = 0;
            var finite = true;
            for (xs) |v| {
                finite = finite and std.math.isFinite(v);
                most = @max(most, @abs(v));
            }
            if (!finite) {
                if (std.math.isFinite(scale)) return "a group holding a non-finite value took a finite scale";
                continue;
            }
            const want: f32 = if (most > 0) most / 127 else 1;
            if (@abs(scale - want) > std.math.floatEps(f32) * want) return "a group's scale is not its absolute max over 127";
            // a value at a step's midpoint may round either way under the GPU's division
            for (xs, q[gi * s.k ..][0..s.k]) |v, qi| if (@abs(v / scale - @as(f32, @floatFromInt(qi))) > 0.5 + 1e-3) return "a value is not its nearest step of the group's scale";
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    lib: ?mtl.Library = null,
    products: std.ArrayList(*Product) = .empty,
    quantize: ?*Quantize = null,

    /// H3's products need tensor units; on a GPU without them the suite registers nothing.
    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        if (!device.tensorUnits()) return s;
        inline for (.{ Form.whole, Form.ragged, Form.int8 }) |f| if (keep(only, Product.name(f))) {
            if (s.lib == null) s.lib = try library(a, device);
            const t = try a.create(Product);
            t.* = Product.init(a, device, queue, s.lib.?, f) catch |err| {
                a.destroy(t);
                return err;
            };
            s.products.append(a, t) catch |err| {
                t.deinit();
                a.destroy(t);
                return err;
            };
            try kernels.append(a, t.kernel());
        };
        if (keep(only, Quantize.label)) {
            if (s.lib == null) s.lib = try library(a, device);
            const t = try a.create(Quantize);
            t.* = Quantize.init(a, device, queue, s.lib.?) catch |err| {
                a.destroy(t);
                return err;
            };
            s.quantize = t;
            try kernels.append(a, t.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        const a = s.a;
        for (s.products.items) |t| {
            t.deinit();
            a.destroy(t);
        }
        s.products.deinit(a);
        if (s.quantize) |t| {
            t.deinit();
            a.destroy(t);
        }
        if (s.lib) |l| l.deinit();
        a.destroy(s);
    }
};
