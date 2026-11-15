//! The decoder's row glue under the contract: ud_gather and ud_norm (plain, and adding the residual first).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const k = @import("metal_uocr_decoder_common.zig");

const c = contract.spec;
const g = contract.gates;
const H = k.hidden;
const Mutation = k.Mutation;
const Range = k.Range;
const bf = k.bf;

// --- ud_gather: y[i] = x[rows[i]], 1280 wide -----------------------------------------------------------------------

pub const Gather = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    rows: []u32 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: rows gathered; n: the width (1,280); k: rows to gather from.
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr decoder ud_gather rows",
            .accum = .none,
            .depth = k.noSums,
            .out = .bf16,
            // a prompt chunk's 6 expert slots a row, and its rows copied once (the checks' layer rows)
            .served = &.{ .{ .rows = 6 * k.prompt_rows, .n = H, .k = k.prompt_rows }, .{ .rows = 279, .n = H, .k = 279 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !Gather {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "ud_gather"), .m = m };
    }
    pub fn deinit(t: *Gather) void {
        t.a.free(t.x);
        t.a.free(t.rows);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Gather) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Gather);

    pub fn prepare(t: *Gather, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H or shape.k == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.a.free(t.rows);
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, shape.k * H);
        k.fillBf16(rng, t.x, 4);
        t.rows = try t.a.alloc(u32, shape.rows);
        for (t.rows, 0..) |*r, i| r.* = @intCast((i * 7 + 1 + rng.uintLessThan(usize, shape.k)) % shape.k);
        if (fill == .nonfinite) for (k.planted, 0..) |v, i| {
            t.x[@as(usize, t.rows[i % shape.rows]) * H + 3 * i + 1] = c.toBf16(v);
        };
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.rows));
        t.out = try t.rig.output(shape.rows * H * 2);
        return true;
    }

    pub fn run(t: *Gather) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.dispatchThreads(mtl.Size.of(H, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Gather) usize {
        return t.shape.rows * H;
    }

    pub fn reference(t: *Gather, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            w.* = c.fromBf16(t.x[@as(usize, t.rows[i / H]) * H + i % H]);
            m.* = 0;
        }
    }
};

// --- ud_norm: optionally R = bf16(R + A), then Y = bf16(w * bf16(r * rsqrt(mean(r^2) + eps))) ----------------------

pub const NormCase = enum {
    /// The norm alone (add 0): Y checked.
    plain,
    /// The residual's add (add 1): R checked, in place.
    resid,
    /// The norm of the added residual (add 1): Y checked.
    normed,
};

/// A thread's 5 squares, the simd_sum tree, then 8 simdgroups' partials.
fn normDepth(_: c.Shape) usize {
    return 5 + 5 + 8;
}

pub const Norm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    v: NormCase,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    r0: []u16 = &.{},
    add: []u16 = &.{},
    w: []u16 = &.{},
    resid: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    ins: [2]rig_.Ref = undefined,

    /// Rows: tokens; n: the width (1,280).
    pub fn spec(v: NormCase) c.Spec {
        const served: []const c.Shape = &.{ .{ .rows = k.prompt_rows, .n = H, .k = 1 }, .{ .rows = 1, .n = H, .k = 1 }, .{ .rows = 279, .n = H, .k = 1 } };
        if (v == .resid) return .{ .name = "metal uocr decoder ud_norm add, the residual", .accum = .none, .depth = k.noSums, .out = .bf16, .in_place = true, .served = served };
        return .{
            .name = if (v == .plain) "metal uocr decoder ud_norm" else "metal uocr decoder ud_norm add, the normed rows",
            .accum = .f32,
            .depth = normDepth,
            // the mean's division, + eps, rsqrt, x * inv; then w times the bf16 point
            .ops = 6,
            .mids = 1,
            .out = .bf16,
            .served = served,
            .sums = .{ .bf16 = 0, .why = "the bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps" },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation, v: NormCase) !Norm {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "ud_norm"), .m = m, .v = v };
    }
    pub fn deinit(t: *Norm) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Norm) void {
        inline for (.{ "r0", "add", "w" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Norm) g.Kernel {
        return k.mark(.{ .spec = spec(t.v), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = blk: {
        var h = g.hooks(Norm);
        h.overlap = overlap;
        break :blk h;
    };
    /// The residual is read and written in place where it is the checked output.
    fn overlap(ctx: *anyopaque) bool {
        const t: *Norm = @ptrCast(@alignCast(ctx));
        return t.v == .resid;
    }

    pub fn prepare(t: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const n = shape.rows * H;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.r0 = try t.a.alloc(u16, n);
        t.add = try t.a.alloc(u16, n);
        t.w = try t.a.alloc(u16, H);
        if (fill == .stress) {
            // integers whose squares sum exactly in fp32 (1,280 x 1,600 at most) and overflow fp16 in any order
            for (t.r0) |*v| v.* = c.toBf16(@floatFromInt(@as(i32, @intCast(20 + rng.uintLessThan(u32, 21))) * (if (rng.boolean()) @as(i32, 1) else -1)));
            @memset(t.add, 0);
            for (t.w) |*v| v.* = c.toBf16(1);
        } else {
            k.fillBf16(rng, t.r0, 4);
            k.fillBf16(rng, t.add, 2);
            for (t.w) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        }
        if (fill == .nonfinite) {
            // a row each, one row holding both infinities; then an added NaN
            const at = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 }, .{ 3, 5 } };
            const vals = [_]f32{ k.planted[0], k.planted[1], k.planted[2], k.planted[1], k.planted[2] };
            for (at, vals) |p, v| if (p[0] < shape.rows) {
                t.r0[p[0] * H + p[1]] = c.toBf16(v);
            };
            if (shape.rows > 4 and t.v != .plain) t.add[4 * H + 6] = c.toBf16(k.planted[0]);
        }
        const rbytes = std.mem.sliceAsBytes(t.r0);
        t.resid = switch (t.v) {
            .plain => try t.rig.input(rbytes),
            .resid => try t.rig.output(rbytes.len),
            .normed => try t.rig.scratch(rbytes.len),
        };
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.add));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.y = if (t.v == .resid) try t.rig.scratch(n * 2) else try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Norm) !void {
        // the residual starts from the prepared rows each run
        if (t.v != .plain) @memcpy(t.rig.bytes(t.resid, t.r0.len * 2), std.mem.sliceAsBytes(t.r0));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.resid.buf, t.resid.off, 0);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 1);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 2);
        p.e.setBuffer(t.y.buf, t.y.off, 3);
        p.e.setValue(extern struct { add: u32, eps: f32 }{ .add = @intFromBool(t.v != .plain), .eps = k.eps }, 4);
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Norm) usize {
        return t.shape.rows * H;
    }

    /// Row r's values after the add (if any).
    fn row(t: *Norm, r: usize, out: []f64) void {
        for (out, 0..) |*o, j| {
            const x: f64 = c.fromBf16(t.r0[r * H + j]);
            o.* = if (t.v == .plain) x else bf(x + c.fromBf16(t.add[r * H + j]));
        }
    }

    fn sumSquares(x: []const f64, acc: c.Accum, order: g.Order) f64 {
        var sq: [H]f64 = undefined;
        for (&sq, x) |*s, v| s.* = v * v;
        return if (acc == .none) blk: {
            var s: f64 = 0;
            for (sq) |v| s += v;
            break :blk s;
        } else contract.rows.sumIn(acc, order, &sq);
    }

    fn outputs(t: *Norm, acc: c.Accum, order: g.Order, out: []f64) void {
        var x: [H]f64 = undefined;
        for (0..t.shape.rows) |r| {
            t.row(r, &x);
            if (t.v == .resid) {
                @memcpy(out[r * H ..][0..H], &x);
                continue;
            }
            const inv = 1 / @sqrt(sumSquares(&x, acc, order) / H + @as(f64, k.eps));
            for (out[r * H ..][0..H], x, 0..) |*o, v, j| o.* = bf(c.fromBf16(t.w[j]) * bf(v * inv));
        }
    }

    pub fn reference(t: *Norm, want: []f64, mag: []f64) !void {
        t.outputs(.none, .sequential, want);
        @memset(mag, 0);
    }

    pub fn simulate(t: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.outputs(acc, order, out);
    }

    /// Squares' fp32 chain (exact at gate 4), mean, eps, rsqrt (4 roundings), x * inv, bf16, the weight, bf16.
    pub fn intervals(t: *Norm, low: []f64, high: []f64) !void {
        var x: [H]f64 = undefined;
        for (0..t.shape.rows) |r| {
            t.row(r, &x);
            if (t.v == .resid) {
                @memcpy(low[r * H ..][0..H], &x);
                @memcpy(high[r * H ..][0..H], &x);
                continue;
            }
            const ss = sumSquares(&x, .none, .sequential);
            // gate 4's integers sum exactly in fp32
            const e = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(normDepth(t.shape)), c.unit(.f32)) * ss;
            const lo = (ss - e) / H + @as(f64, k.eps);
            const hi = (ss + e) / H + @as(f64, k.eps);
            const inv = (Range{ .lo = 1 / @sqrt(hi), .hi = 1 / @sqrt(lo) }).fp32(4);
            for (x, 0..) |v, j| {
                const y = Range.point(v).mul(inv).fp32(1).bf16().mul(Range.point(c.fromBf16(t.w[j]))).fp32(1).bf16();
                low[r * H + j] = y.lo;
                high[r * H + j] = y.hi;
            }
        }
    }

    pub fn bounds(t: *Norm, want: []const f64, _: []const f64, allow: []f64) !void {
        try k.boundsFromIntervals(t.a, t, want, allow);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    gathers: std.ArrayList(*Gather) = .empty,
    norms: std.ArrayList(*Norm) = .empty,

    fn add(s: *Suite, comptime T: type, list: *std.ArrayList(*T), only: ?[]const u8, m: Mutation, extra: anytype) !void {
        const x = try s.a.create(T);
        x.* = try @call(.auto, T.init, .{ s.a, s.device, s.queue, m } ++ extra);
        try list.append(s.a, x);
        const kern = x.kernel();
        if (!k.wanted(only, kern.spec.name)) {
            list.items.len -= 1;
            x.deinit();
            s.a.destroy(x);
            return;
        }
        try s.kernels.append(s.a, kern);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Mutation{ .none, .gather_own_row }) |m| try s.add(Gather, &s.gathers, only, m, .{});
        for ([_]NormCase{ .plain, .resid, .normed }) |v| try s.add(Norm, &s.norms, only, .none, .{v});
        try s.add(Norm, &s.norms, only, .norm_mean_over_threads, .{NormCase.plain});
        try s.add(Norm, &s.norms, only, .norm_add_wrong_sign, .{NormCase.resid});
    }

    pub fn deinit(s: *Suite) void {
        inline for (.{ "gathers", "norms" }) |f| {
            for (@field(s, f).items) |x| {
                x.deinit();
                s.a.destroy(x);
            }
            @field(s, f).deinit(s.a);
        }
    }
};

/// Gate 5's texts for this group's launches.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?k.Texts {
    return k.limitsText(name, &.{ "ud_gather", "ud_norm" });
}
