//! DeepSeek V4.1's one-row EXL3 projection under the contract: tf_exl3_matmul_hoisted's bits at the decode shapes.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const ex = tf.exl3_projection;
const fmt = ex.format;
const base = "dsv41 exl3 row tf_exl3_matmul_row ";

/// A registration: the matrix's width and codebook, the output, and whether it is the planted control.
pub const Variant = struct { halves: u16, book: fmt.Codebook, out: ex.Output = .bf16, ctl: bool = false };

const why = "the K sums of decoded fp16 weights: no input makes them exact in fp32, so gate 4 proves fp16's miss (an overflow) alone";
const rule = "tf_exl3_matmul_hoisted's bytes on the same inputs: a NaN and both infinities in x reach the outputs as its do";
const kept = "low += (high - (next - part)) + (product - part);";

/// The K chain of compensated adds, then the output rotation's seven adds and its scales.
fn depth(s: c.Shape) usize {
    return s.k + 16;
}

fn sh(n: usize, k: usize) c.Shape {
    return .{ .rows = 1, .n = n, .k = k };
}

// wq_a, wkv, wq_b, a wo_a group, wo_b, the shared expert's w1/w3 and w2, ix_wq_b: one decode row each
const served = [_]c.Shape{ sh(1280, 5120), sh(512, 5120), sh(32768, 1280), sh(1024, 4096), sh(5120, 8192), sh(2304, 5120), sh(5120, 2304), sh(4096, 1280) };

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .ops = 4,
        .out = if (v.out == .f32) .f32 else .bf16,
        .replaces = "tf_exl3_matmul_hoisted (the same rows in one launch)",
        .nan = .{ .stated = rule },
        .tiles = .{ .{}, .{ .size = 256, .step = 128, .min = 128 }, .{ .size = 256, .step = 128, .min = 128 } },
        .served = &served,
        .sums = .{ .f16 = 4, .bf16 = 0, .exact = false, .why = why },
    };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: ex.Pipelines,
    shape: c.Shape = undefined,
    words: []i16 = &.{},
    svh: []u16 = &.{},
    wq: []f16 = &.{},
    m: ex.Matrix = undefined,
    x: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    scratch: rig_.Ref = undefined,
    yb: rig_.Ref = undefined,
    scratch_b: rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = try ex.Pipelines.load(device) };
        errdefer p.pipes.deinit();
        if (v.ctl) {
            const at = std.mem.indexOf(u8, sources.core_exl3_projection, kept) orelse return error.ControlNotPlanted;
            const text = try std.mem.concat(a, u8, &.{ sources.core_exl3_projection[0..at], sources.core_exl3_projection[at + kept.len ..] });
            defer a.free(text);
            const planted = try rig_.pipeline(device, text, "tf_exl3_matmul_row");
            p.pipes.row.?.deinit();
            p.pipes.row = planted;
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        p.pipes.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        p.a.free(p.words);
        p.a.free(p.svh);
        p.a.free(p.wq);
        p.words = &.{};
        p.svh = &.{};
        p.wq = &.{};
        p.rig.reset();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = if (p.v.ctl) .shapes else null };
    }

    const vt = g.hooks(Proj);

    fn es(p: *const Proj) usize {
        return if (p.v.out == .f32) 4 else 2;
    }

    fn view(r: rig_.Ref, extra: usize) ex.View {
        return .{ .buffer = r.buf, .offset = r.off + extra };
    }

    /// Random words, fp16 scales and bf16 x in [-1, 1); non-finite: x[1][3] NaN, x[2][5] +Inf, x[2][9] -Inf.
    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k % 128 != 0 or shape.n % 128 != 0) return false;
        p.drop();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const width = fmt.Width{ .halves = p.v.halves };
        p.words = try p.a.alloc(i16, (shape.k / 16) * (shape.n / 16) * width.tileWords());
        const suh = try p.a.alloc(u16, shape.k);
        defer p.a.free(suh);
        p.svh = try p.a.alloc(u16, shape.n);
        const x = try p.a.alloc(u16, shape.rows * shape.k);
        defer p.a.free(x);
        // stress: all-ones states (one weight), scales 1, one input a block that rotates to 4096 in every lane
        if (fill == .stress) {
            @memset(p.words, -1);
            @memset(suh, 0x3c00);
            @memset(p.svh, 0x3c00);
            @memset(x, 0);
            for (0..shape.rows) |r| for (0..shape.k / 128) |b| {
                x[r * shape.k + b * 128] = c.toBf16(46336);
            };
        } else {
            for (p.words) |*w| w.* = rng.int(i16);
            for (suh) |*s| s.* = @bitCast(@as(f16, @floatCast(rng.float(f32) * 2 - 1)));
            for (p.svh) |*s| s.* = @bitCast(@as(f16, @floatCast(rng.float(f32) * 2 - 1)));
            for (x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
            if (fill == .nonfinite and shape.rows > 2) {
                x[shape.k + 3] = 0x7fc0;
                x[2 * shape.k + 5] = 0x7f80;
                x[2 * shape.k + 9] = 0xff80;
            }
        }
        p.x = try p.rig.input(std.mem.sliceAsBytes(x));
        const t = try p.rig.input(std.mem.sliceAsBytes(p.words));
        const su = try p.rig.input(std.mem.sliceAsBytes(suh));
        const sv = try p.rig.input(std.mem.sliceAsBytes(p.svh));
        p.m = .{ .trellis = view(t, 0), .suh = view(su, 0), .svh = view(sv, 0), .k = shape.k, .n = shape.n, .width = width, .codebook = p.v.book };
        const out = shape.rows * shape.n * p.es();
        p.y = try p.rig.output(out);
        p.scratch = try p.rig.scratch(shape.rows * try ex.scratchBytes(shape.k, shape.n, 1));
        p.yb = try p.rig.scratch(out);
        p.scratch_b = try p.rig.scratch(try ex.scratchBytes(shape.k, shape.n, shape.rows));
        return true;
    }

    /// Each row its own one-row launch, as decode calls it.
    pub fn run(p: *Proj) !void {
        const s = p.shape;
        const sb = try ex.scratchBytes(s.k, s.n, 1);
        const pass = p.rig.begin();
        for (0..s.rows) |r| try ex.encode(&p.pipes, pass.e, p.m, view(p.x, r * s.k * 2), view(p.y, r * s.n * p.es()), view(p.scratch, r * sb), .{ .count = 1, .x_stride = s.k, .y_stride = s.n, .output = p.v.out });
        try p.rig.finish(pass);
    }

    /// tf_exl3_matmul_hoisted (or its shared prefill at 32 rows and more) over every row in one launch.
    pub fn replaced(p: *Proj) ![]const u8 {
        const s = p.shape;
        var main = p.pipes;
        main.row = null;
        const pass = p.rig.begin();
        try ex.encode(&main, pass.e, p.m, view(p.x, 0), view(p.yb, 0), view(p.scratch_b, 0), .{ .count = s.rows, .x_stride = s.k, .y_stride = s.n, .output = p.v.out });
        try p.rig.finish(pass);
        return p.rig.bytes(p.yb, s.rows * s.n * p.es());
    }

    /// Gate 1: the replaced kernel's bytes, with some output non-finite to prove the planted values reach it.
    pub fn check(p: *Proj, _: g.Gate) !?[]const u8 {
        const got = try p.a.dupe(u8, try p.rig.outputBytes());
        defer p.a.free(got);
        const want = try p.replaced();
        const out = spec(p.v, p.name).out;
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&p.msg, "byte {d} of {d} differs from tf_exl3_matmul_hoisted's", .{ at, got.len });
        for (0..p.count()) |i| if (c.classOf(c.read(out, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&p.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    pub fn count(p: *Proj) usize {
        return p.shape.rows * p.shape.n;
    }

    /// Float64 outputs over the kernels' rotated fp16 rows; with `acc`, the K sums in that accumulator and order.
    fn outputs(p: *Proj, want: []f64, mag: ?[]f64, acc: ?c.Accum, order: g.Order) !void {
        const s = p.shape;
        _ = try p.replaced();
        const xs = std.mem.bytesAsSlice(f16, p.rig.bytes(p.scratch_b, s.rows * s.k * 2));
        if (p.wq.len == 0) {
            p.wq = try p.a.alloc(f16, s.k * s.n);
            fmt.unpack(p.m.width, p.v.book, p.words, s.k / 16, s.n / 16, p.wq);
        }
        const t = try p.a.alloc(f64, s.k);
        defer p.a.free(t);
        const abs = try p.a.alloc(f64, s.rows * s.n);
        defer p.a.free(abs);
        for (0..s.rows) |r| for (0..s.n) |j| {
            var total: f64 = 0;
            for (t, 0..) |*e, q| {
                e.* = @as(f64, xs[r * s.k + q]) * @as(f64, p.wq[q * s.n + j]);
                total += @abs(e.*);
            }
            var sum: f64 = 0;
            if (acc) |ac| {
                sum = contract.rows.sumIn(ac, order, t);
            } else {
                for (t) |e| sum += e;
            }
            want[r * s.n + j] = sum;
            abs[r * s.n + j] = total;
        };
        fmt.rotate(want, s.rows, s.n, 1);
        for (0..s.rows) |r| for (0..s.n) |j| {
            const sc: f64 = @as(f16, @bitCast(p.svh[j]));
            want[r * s.n + j] *= sc;
            if (mag) |mg| {
                var block: f64 = 0;
                for (j / 128 * 128..j / 128 * 128 + 128) |q| block += abs[r * s.n + q];
                mg[r * s.n + j] = @abs(sc) * block / @sqrt(128.0);
            }
        };
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        return p.outputs(want, mag, null, .sequential);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.outputs(out, null, acc, order);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{ .halves = 4, .book = .mcg }, .name = base ++ "mcg 2-bit" },
        .{ .v = .{ .halves = 5, .book = .mul1 }, .name = base ++ "mul1 2.5-bit" },
        .{ .v = .{ .halves = 8, .book = .mul1, .out = .f32 }, .name = base ++ "mul1 4-bit fp32 out" },
        .{ .v = .{ .halves = 12, .book = .inst3 }, .name = base ++ "3inst 6-bit" },
        .{ .v = .{ .halves = 16, .book = .mcg }, .name = base ++ "mcg 8-bit" },
        .{ .v = .{ .halves = 6, .book = .mul1, .ctl = true }, .name = "dsv41 exl3 row control: the TwoSum error term dropped, posing as tf_exl3_matmul_hoisted" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.v, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }
};
