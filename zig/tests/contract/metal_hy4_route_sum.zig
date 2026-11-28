//! Hy4-flash's routed glue under the contract: tf_hy4_routed_sum's weighted picks and elementwise's tf_add_bf16.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const rt = @import("metal_hy4_route.zig");

const c = contract.spec;
const g = contract.gates;
const fl = h.fl;
const Dims = rt.Dims;
const max_topk = @import("metal_hy4_route_select.zig").max_topk;

// --- tf_hy4_routed_sum: out[r, x] = bf16(sum over picks s of w[r, s] * parts[r, s, x]), one fp32 fma chain ---------

/// One fma a pick: each rounds once.
fn picksDepth(s: c.Shape) usize {
    return s.k;
}

pub const Sum = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    d: Dims,
    served: [3]c.Shape,
    control: ?g.Gate = null,
    base_buf: [128]u8 = undefined,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    width: usize = 0,
    parts: []f32 = &.{},
    wts: []f32 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn base(d: Dims, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "metal hy4 affine_experts tf_hy4_routed_sum width {d} picks {d}", .{ d.width, d.topk }) catch unreachable;
    }

    /// Rows: token rows; n: the latent width; k: picks a row. The grid's x is the width in groups of 128.
    pub fn spec(name: []const u8, served: []const c.Shape) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = picksDepth,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 128, .step = 1, .min = 1 }, .{} },
            .served = served,
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, d: Dims, control: bool) !Sum {
        const text = try fl.edited(a, sources.hy4_flash_affine_experts, if (control) rt.half_sum else null);
        defer a.free(text);
        const s: c.Shape = .{ .rows = h.window_rows, .n = d.width, .k = d.topk };
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try rt.compile(device, text, "tf_hy4_routed_sum"),
            .d = d,
            .served = .{ s, .{ .rows = h.decode_rows, .n = s.n, .k = s.k }, .{ .rows = h.prompt_rows, .n = s.n, .k = s.k } },
            .control = if (control) .sums else null,
        };
    }
    pub fn deinit(t: *Sum) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Sum) void {
        t.a.free(t.parts);
        t.a.free(t.wts);
        t.parts = &.{};
        t.wts = &.{};
    }
    pub fn kernel(t: *Sum) g.Kernel {
        t.name = fl.named(base(t.d, &t.base_buf), if (t.control != null) "its partials in half" else null, &t.label);
        return .{ .spec = spec(t.name, &t.served), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Sum);

    pub fn prepare(t: *Sum, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const K: usize = t.d.topk;
        if (shape.rows == 0 or shape.rows > h.prompt_rows or shape.n == 0 or shape.k != K) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.width = shape.n;
        const n = shape.n;
        t.parts = try t.a.alloc(f32, shape.rows * K * n);
        t.wts = try t.a.alloc(f32, shape.rows * K);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // products near 2^20, exact in fp32 in any order: the second half mirrors the first, one pick nudged
            const half = K / 2;
            for (0..shape.rows) |r| for (0..half) |s| {
                const w: f32 = @floatFromInt(64 + rng.uintLessThan(u32, 64));
                t.wts[r * K + s] = w;
                t.wts[r * K + half + s] = w;
                for (0..n) |x| {
                    const p: f32 = @floatFromInt(@as(i32, @intCast(4096 + rng.uintLessThan(u32, 4096))) * (if (rng.boolean()) @as(i32, 1) else -1));
                    t.parts[(r * K + s) * n + x] = p;
                    t.parts[(r * K + half + s) * n + x] = -p + @as(f32, if (x % half == s) 1 else 0);
                }
            };
        } else {
            for (t.parts) |*v| v.* = (rng.float(f32) * 2 - 1) * 4;
            for (t.wts) |*v| v.* = rng.float(f32) * 0.5 + 0.01;
        }
        if (fill == .nonfinite) {
            // NaN in row 0's parts, +Inf in row 1's, a -Inf weight in row 2, both infinities in one column of row 3
            const at = [_][3]usize{ .{ 0, 3, 10 }, .{ 1, 5, 20 }, .{ 3, 1, 30 }, .{ 3, 9, 30 } };
            const vals = [_]f32{ fl.planted[0], fl.planted[1], fl.planted[1], fl.planted[2] };
            for (at, vals) |p, v| if (p[0] < shape.rows and p[2] < n) {
                t.parts[(p[0] * K + p[1]) * n + p[2]] = v;
            };
            if (shape.rows > 2) t.wts[2 * K + 7] = fl.planted[2];
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.parts));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.wts));
        t.out = try t.rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(t: *Sum) !void {
        const p = t.rig.begin();
        const e = p.e;
        // metal_routed.zig's encoder: a thread an output, the width in groups of 128
        e.setPipeline(t.pipe);
        e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        e.setBuffer(t.out.buf, t.out.off, 2);
        e.setValue([2]u32{ @intCast(t.width), t.d.topk }, 3);
        e.dispatchThreads(mtl.Size.of(t.width, t.rows, 1), mtl.Size.of(128, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Sum) usize {
        return t.rows * t.width;
    }

    fn term(t: *const Sum, i: usize, s: usize) f64 {
        const K: usize = t.d.topk;
        const r = i / t.width;
        return @as(f64, t.wts[r * K + s]) * t.parts[(r * K + s) * t.width + i % t.width];
    }

    pub fn reference(t: *Sum, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            var sum: f64 = 0;
            var size: f64 = 0;
            for (0..t.d.topk) |s| {
                const v = t.term(i, s);
                sum += v;
                size += @abs(v);
            }
            w.* = sum;
            m.* = if (std.math.isFinite(size)) size else 0;
        }
    }

    pub fn simulate(t: *Sum, acc: c.Accum, order: g.Order, out: []f64) !void {
        var terms: [max_topk]f64 = undefined;
        for (out, 0..) |*o, i| {
            for (terms[0..t.d.topk], 0..) |*v, s| v.* = t.term(i, s);
            o.* = fl.bf(contract.rows.sumIn(acc, order, terms[0..t.d.topk]));
        }
    }
};

// --- tf_add_bf16: out = a + b over bf16 values, one rounding ---------------------------------------------------------

pub const Add = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    d: Dims,
    served: [3]c.Shape,
    base_buf: [128]u8 = undefined,
    name: []const u8 = "",
    total: usize = 0,
    x: []u16 = &.{},
    y: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn base(d: Dims, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "metal hy4 elementwise tf_add_bf16 width {d}", .{d.hidden}) catch unreachable;
    }

    /// Rows and width flattened, groups of 256 as Nemotron launches it (Hy4 builds the pipeline, never dispatches it).
    pub fn spec(name: []const u8, served: []const c.Shape) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = fl.noSums,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = served,
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, d: Dims) !Add {
        const s: c.Shape = .{ .rows = h.window_rows, .n = d.hidden, .k = 1 };
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try rt.compile(device, sources.elementwise, "tf_add_bf16"),
            .d = d,
            .served = .{ s, .{ .rows = h.decode_rows, .n = s.n, .k = 1 }, .{ .rows = h.prompt_rows, .n = s.n, .k = 1 } },
        };
    }
    pub fn deinit(t: *Add) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Add) void {
        t.a.free(t.x);
        t.a.free(t.y);
        t.x = &.{};
        t.y = &.{};
    }
    pub fn kernel(t: *Add) g.Kernel {
        t.name = base(t.d, &t.base_buf);
        return .{ .spec = spec(t.name, &t.served), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Add);

    pub fn prepare(t: *Add, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > h.prompt_rows or shape.n == 0 or shape.k != 1) return false;
        t.rig.reset();
        t.free();
        const n = shape.n;
        t.total = shape.rows * n;
        t.x = try t.a.alloc(u16, t.total);
        t.y = try t.a.alloc(u16, t.total);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.x, t.y) |*u, *v| {
            u.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
            v.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
        }
        if (fill == .nonfinite and shape.rows >= 4 and n > 12) {
            // NaN in row 0, +Inf in row 1, -Inf in row 2, both infinities at one place of row 3 and one more in it
            t.x[3] = c.toBf16(fl.planted[0]);
            t.y[n + 5] = c.toBf16(fl.planted[1]);
            t.x[2 * n + 9] = c.toBf16(fl.planted[2]);
            t.x[3 * n + 11] = c.toBf16(fl.planted[1]);
            t.y[3 * n + 11] = c.toBf16(fl.planted[2]);
            t.y[3 * n + 12] = c.toBf16(fl.planted[2]);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.y));
        t.out = try t.rig.output(t.total * 2);
        return true;
    }

    pub fn run(t: *Add) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        e.setValue(@as(i32, @intCast(t.total)), 2);
        e.setBuffer(t.out.buf, t.out.off, 3);
        e.dispatchThreads(mtl.Size.of(t.total, 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Add) usize {
        return t.total;
    }

    /// The exact sum of two bf16 values; the bound is its one rounding to bf16.
    pub fn reference(t: *Add, want: []f64, mag: []f64) !void {
        for (want, mag, t.x, t.y) |*w, *m, u, v| {
            w.* = @as(f64, c.fromBf16(u)) + c.fromBf16(v);
            m.* = 0;
        }
    }
};
