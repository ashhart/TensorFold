//! glue.metal's residual adds and norms under the contract; a kernel writing both outputs registers once for each.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const fl = @import("metal_flint.zig");

const c = contract.spec;
const g = contract.gates;
const bf = fl.bf;
const Range = fl.Range;
const topk = fl.topk;
const u = fl.u32_ulp;

const Kind = enum { norm, add, add_norm, moe_add, moe_add_norm };

const Case = struct {
    kind: Kind,
    /// Which output is checked: the normed rows, or the residual.
    normed: bool,
    name: []const u8,
    /// A planted control: the fault, what it drops, and the gate that must go red.
    fault: ?fl.Edit = null,
    drops: []const u8 = "",
};

const cases = [_]Case{
    .{ .kind = .norm, .normed = true, .name = "metal flint glue fl_norm" },
    .{ .kind = .add, .normed = false, .name = "metal flint glue fl_add residual" },
    .{ .kind = .add_norm, .normed = false, .name = "metal flint glue fl_add_norm residual" },
    .{ .kind = .add_norm, .normed = true, .name = "metal flint glue fl_add_norm normed" },
    .{ .kind = .moe_add, .normed = false, .name = "metal flint glue fl_moe_add top 6 residual" },
    .{ .kind = .moe_add_norm, .normed = false, .name = "metal flint glue fl_moe_add_norm top 6 residual" },
    .{ .kind = .moe_add_norm, .normed = true, .name = "metal flint glue fl_moe_add_norm top 6 normed" },
    // controls: each drops one bf16 rounding point
    .{ .kind = .norm, .normed = true, .name = "metal flint glue fl_norm", .drops = "x * inv's bf16 rounding dropped", .fault = .{ .kernel = "fl_norm", .from = "float(bfloat(x[i] * inv))", .to = "(x[i] * inv)" } },
    .{ .kind = .add_norm, .normed = false, .name = "metal flint glue fl_add_norm residual", .drops = "the residual's bf16 rounding dropped", .fault = .{ .kernel = "fl_add_norm", .from = "float(bfloat(x[i] + float(X[(size_t)row * a.width + i])))", .to = "(x[i] + float(X[(size_t)row * a.width + i]))" } },
    .{ .kind = .moe_add, .normed = false, .name = "metal flint glue fl_moe_add top 6 residual", .drops = "the routing weight's bf16 rounding dropped", .fault = .{ .kernel = "fl_moe_add", .from = "float(bfloat(wts[p.y * k + j]))", .to = "wts[p.y * k + j]" } },
    .{ .kind = .moe_add_norm, .normed = true, .name = "metal flint glue fl_moe_add_norm top 6 normed", .drops = "x * inv's bf16 rounding dropped", .fault = .{ .kernel = "fl_moe_add_norm", .from = "float(bfloat(x[i] * inv))", .to = "(x[i] * inv)" } },
};

/// Every norm runs 256 threads a row (forward.zig): a thread's strided squares, the simd_sum tree, 8 partials in turn.
const threads = 256;
fn depth(s: c.Shape) usize {
    return (s.n + threads - 1) / threads + 5 + threads / 32;
}

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

fn spec(v: Case, name: []const u8) c.Spec {
    const k: usize = if (v.kind == .moe_add or v.kind == .moe_add_norm) topk else 1;
    const H = fl.hidden;
    // a window's rows first (the edges run at it), then a shared forward's, a prompt chunk's and one row
    const one = [_]c.Shape{ .{ .rows = fl.window_rows, .n = H, .k = 1 }, .{ .rows = fl.batch_rows, .n = H, .k = 1 }, .{ .rows = fl.prompt_rows, .n = H, .k = 1 }, .{ .rows = 1, .n = H, .k = 1 } };
    const moe = [_]c.Shape{ .{ .rows = fl.window_rows, .n = H, .k = topk }, .{ .rows = fl.batch_rows, .n = H, .k = topk }, .{ .rows = fl.prompt_rows, .n = H, .k = topk }, .{ .rows = 1, .n = H, .k = topk } };
    var s: c.Spec = .{
        .name = name,
        .accum = .none,
        .depth = fl.noSums,
        .out = .f32,
        .in_place = true,
        .tiles = .{ .{}, .{ .size = 256, .step = 32, .min = 32 }, .{} },
        .served = if (k == 1) &one else &moe,
    };
    if (v.normed) {
        s.accum = .f32;
        s.depth = depth;
        s.out = .bf16;
        s.in_place = false;
        s.sums = .{ .bf16 = 0, .why = norm_why };
    }
    return s;
}

pub const Rows = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Case,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    r0: []f32 = &.{},
    x: []u16 = &.{},
    w: []u16 = &.{},
    y: []u16 = &.{},
    wts: []f32 = &.{},
    shared: []u16 = &.{},
    resid: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ins: [4]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Case) !Rows {
        const fun = switch (v.kind) {
            .norm => "fl_norm",
            .add => "fl_add",
            .add_norm => "fl_add_norm",
            .moe_add => "fl_moe_add",
            .moe_add_norm => "fl_moe_add_norm",
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try fl.gluePipe(a, device, fun, v.fault), .v = v };
    }
    pub fn deinit(t: *Rows) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Rows) void {
        inline for (.{ "r0", "x", "w", "y", "wts", "shared" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Rows) g.Kernel {
        t.name = fl.named(t.v.name, if (t.v.fault != null) t.v.drops else null, &t.label);
        return .{ .spec = spec(t.v, t.name), .ctx = t, .vt = &vt, .control = if (t.v.fault != null) .shapes else null };
    }
    const vt = blk: {
        var h = g.hooks(Rows);
        h.overlap = overlap;
        break :blk h;
    };
    /// The residual is read and written in place where it is the checked output.
    fn overlap(ctx: *anyopaque) bool {
        const t: *Rows = @ptrCast(@alignCast(ctx));
        return t.v.kind != .norm and !t.v.normed;
    }

    fn moe(t: *const Rows) bool {
        return t.v.kind == .moe_add or t.v.kind == .moe_add_norm;
    }
    fn hasNorm(t: *const Rows) bool {
        return t.v.kind == .norm or t.v.kind == .add_norm or t.v.kind == .moe_add_norm;
    }

    pub fn prepare(t: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != @as(usize, if (t.moe()) topk else 1)) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const rows = shape.rows;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.r0 = try t.a.alloc(f32, rows * n);
        t.w = try t.a.alloc(u16, n);
        t.x = try t.a.alloc(u16, if (t.v.kind == .add or t.v.kind == .add_norm) rows * n else 0);
        const slots: usize = if (t.moe()) rows * topk else 0;
        t.y = try t.a.alloc(u16, slots * n);
        t.wts = try t.a.alloc(f32, slots);
        t.shared = try t.a.alloc(u16, if (t.moe()) rows * n else 0);
        if (fill == .stress) {
            // integers whose squares sum exactly in fp32 (at most 2,048 x 1,600) and overflow fp16 in any order
            for (t.r0) |*v| v.* = @floatFromInt(@as(i32, @intCast(20 + rng.uintLessThan(u32, 21))) * (if (rng.boolean()) @as(i32, 1) else -1));
            @memset(t.w, c.toBf16(1));
            @memset(t.x, 0);
            @memset(t.y, 0);
            @memset(t.shared, 0);
            for (t.wts) |*v| v.* = 0.25;
        } else {
            // the residual holds bf16 values (every write rounds to bf16 first)
            for (t.r0) |*v| v.* = @floatCast(bf(@as(f64, rng.float(f32)) * 8 - 4));
            for (t.w) |*v| v.* = c.toBf16(rng.float(f32) * 2);
            for (t.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (t.y) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (t.shared) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            // fp32 routing weights summing to about 1 a row, as fl_route_pick leaves them
            for (t.wts) |*v| v.* = 0.05 + rng.float(f32) * 0.25;
        }
        if (fill == .nonfinite) {
            // a row each, one row holding both infinities, in the residual; then the added terms' own
            const at = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 }, .{ 3, 5 } };
            const vals = [_]f32{ fl.planted[0], fl.planted[1], fl.planted[2], fl.planted[1], fl.planted[2] };
            for (at, vals) |p, v| if (p[0] < rows and p[1] < n) {
                t.r0[p[0] * n + p[1]] = v;
            };
            if (rows > 4 and n > 7) {
                if (t.x.len > 0) t.x[4 * n + 6] = c.toBf16(fl.planted[0]);
                if (t.moe()) {
                    t.y[(4 * topk + 1) * n + 6] = c.toBf16(fl.planted[1]);
                    t.wts[4 * topk + 2] = fl.planted[2];
                    t.shared[4 * n + 7] = c.toBf16(fl.planted[0]);
                }
            }
        }
        // the residual buffer: the checked output, or working memory restored before each run
        const rbytes = std.mem.sliceAsBytes(t.r0);
        t.resid = if (t.v.kind == .norm) try t.rig.input(rbytes) else if (t.v.normed) try t.rig.scratch(rbytes.len) else try t.rig.output(rbytes.len);
        if (t.hasNorm()) t.out = if (t.v.normed) try t.rig.output(rows * n * 2) else try t.rig.scratch(rows * n * 2);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        if (t.x.len > 0) t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        if (t.moe()) {
            t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.y));
            t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.wts));
            t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.shared));
        }
        return true;
    }

    pub fn run(t: *Rows) !void {
        if (t.v.kind != .norm) @memcpy(t.rig.bytes(t.resid, t.r0.len * 4), std.mem.sliceAsBytes(t.r0));
        const rows = t.shape.rows;
        const n = t.shape.n;
        const args = fl.Args{ .rows = @intCast(rows), .width = @intCast(n), .heads = topk, .eps = fl.eps };
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        e.setBuffer(t.resid.buf, t.resid.off, 0);
        switch (t.v.kind) {
            .norm => {
                e.setBuffer(t.ins[0].buf, t.ins[0].off, 1);
                e.setValue(args, 2);
                e.setBuffer(t.out.buf, t.out.off, 3);
            },
            .add => {
                e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
                e.setValue(args, 2);
            },
            .add_norm => {
                e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
                e.setBuffer(t.ins[0].buf, t.ins[0].off, 2);
                e.setValue(args, 3);
                e.setBuffer(t.out.buf, t.out.off, 4);
            },
            .moe_add, .moe_add_norm => {
                for (1..4) |i| e.setBuffer(t.ins[i].buf, t.ins[i].off, i);
                if (t.v.kind == .moe_add) {
                    e.setValue(args, 4);
                } else {
                    e.setBuffer(t.ins[0].buf, t.ins[0].off, 4);
                    e.setValue(args, 5);
                    e.setBuffer(t.out.buf, t.out.off, 6);
                }
            },
        }
        if (t.hasNorm()) {
            e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(threads, 1, 1));
        } else e.dispatchThreads(mtl.Size.of(n, rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Rows) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Element i's residual after the kernel's add: float64's value at the bf16 points, and the range fp32 allows.
    fn residAt(t: *const Rows, i: usize) struct { x: f64, r: Range } {
        const x0: f64 = t.r0[i];
        switch (t.v.kind) {
            .norm => return .{ .x = x0, .r = Range.point(x0) },
            // two bf16 values: their sum is exact in fp32 or rounds far from a bf16 midpoint
            .add, .add_norm => {
                const x = bf(x0 + c.fromBf16(t.x[i]));
                return .{ .x = x, .r = Range.point(x) };
            },
            .moe_add, .moe_add_norm => {
                const n = t.shape.n;
                const r = i / n;
                var acc: f64 = 0;
                var sum = Range.point(0);
                for (0..topk) |j| {
                    const s = r * topk + j;
                    const term = bf(bf(t.wts[s]) * @as(f64, c.fromBf16(t.y[s * n + i % n])));
                    acc += term;
                    sum = sum.plus(Range.point(term)).fp32();
                }
                const sh: f64 = c.fromBf16(t.shared[i]);
                const mlp = bf(bf(acc) + sh);
                const mr = sum.bf16().plus(Range.point(sh)).fp32().bf16();
                return .{ .x = bf(x0 + mlp), .r = mr.plus(Range.point(x0)).fp32().bf16() };
            },
        }
    }

    fn inv(ss: f64, n: usize) f64 {
        return 1 / @sqrt(ss / @as(f64, @floatFromInt(n)) + @as(f64, fl.eps));
    }

    pub fn reference(t: *Rows, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        const xs = try t.a.alloc(f64, n);
        defer t.a.free(xs);
        for (0..t.shape.rows) |r| {
            var ss: f64 = 0;
            for (xs, 0..) |*x, j| {
                x.* = t.residAt(r * n + j).x;
                ss += x.* * x.*;
            }
            const iv = inv(ss, n);
            for (xs, 0..) |x, j| {
                const i = r * n + j;
                want[i] = if (t.v.normed) bf(@as(f64, c.fromBf16(t.w[j])) * bf(x * iv)) else x;
                mag[i] = 0;
            }
        }
    }

    /// An output's range: the residual's fp32 steps; squares within gamma(depth), 4 roundings, x * inv, bf16, w, bf16.
    pub fn intervals(t: *Rows, low: []f64, high: []f64) !void {
        const n = t.shape.n;
        const rs = try t.a.alloc(Range, n);
        defer t.a.free(rs);
        const gam = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(depth(t.shape)), u);
        for (0..t.shape.rows) |r| {
            var lo: f64 = 0;
            var hi: f64 = 0;
            for (rs, 0..) |*x, j| {
                x.* = t.residAt(r * n + j).r;
                const a2 = x.lo * x.lo;
                const b2 = x.hi * x.hi;
                hi += @max(a2, b2);
                lo += if (x.lo <= 0 and x.hi >= 0) 0 else @min(a2, b2);
            }
            const iv = Range{ .lo = inv(hi * (1 + gam), n) * (1 - 4 * u), .hi = inv(lo * (1 - gam), n) * (1 + 4 * u) };
            for (rs, 0..) |x, j| {
                const o = if (t.v.normed) x.mul(iv).fp32().bf16().times(c.fromBf16(t.w[j])).bf16() else x;
                low[r * n + j] = o.lo;
                high[r * n + j] = o.hi;
            }
        }
    }

    pub fn bounds(t: *Rows, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(t.a, t, want, allow);
    }

    pub fn simulate(t: *Rows, acc: c.Accum, order: g.Order, out: []f64) !void {
        const n = t.shape.n;
        const buf = try t.a.alloc(f64, 2 * n);
        defer t.a.free(buf);
        for (0..t.shape.rows) |r| {
            for (buf[0..n], buf[n..], 0..) |*x, *q, j| {
                x.* = t.residAt(r * n + j).x;
                q.* = x.* * x.*;
            }
            const iv = inv(contract.rows.sumIn(acc, order, buf[n..]), n);
            for (buf[0..n], 0..) |x, j| out[r * n + j] = bf(@as(f64, c.fromBf16(t.w[j])) * bf(x * iv));
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    tests: fl.List(Rows) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (cases) |v| if (fl.wanted(only, v.name)) {
            try s.tests.add(s.a, s.kernels, try Rows.init(s.a, s.device, s.queue, v));
        };
    }
    pub fn deinit(s: *Suite) void {
        s.tests.deinit(s.a);
    }
};

/// The rows' launches are glue.metal's, which metal_flint_glue.zig compiles for gate 5.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, _: []const u8) !?fl.Texts {
    return null;
}
