//! glue.metal's residual adds and norms under the contract; a kernel writing both outputs registers once for each.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const glue = @import("metal_minimax_glue.zig");

const c = contract.spec;
const g = contract.gates;
const bf = glue.bf;
const MArgs = glue.MArgs;
const hidden = glue.hidden;
const topk = glue.topk;

const eps: f32 = 1e-6;
const scale: f32 = 2.0; // routed_scaling_factor

const Kind = enum { norm, add, add_norm, moe_add, moe_add_norm };

const Case = struct {
    kind: Kind,
    /// Threads a row's threadgroup (the norms; the adds alone run 256-thread groups over the grid).
    threads: u32 = 256,
    /// Which output is checked: the normed rows, or the residual.
    normed: bool,
    name: []const u8,
};

const cases = [_]Case{
    .{ .kind = .norm, .threads = 1024, .normed = true, .name = "metal minimax_m3 glue mm_norm 1024 threads" },
    .{ .kind = .norm, .threads = 256, .normed = true, .name = "metal minimax_m3 glue mm_norm 256 threads" },
    .{ .kind = .add, .normed = false, .name = "metal minimax_m3 glue mm_add residual" },
    .{ .kind = .add_norm, .threads = 1024, .normed = false, .name = "metal minimax_m3 glue mm_add_norm 1024 threads residual" },
    .{ .kind = .add_norm, .threads = 1024, .normed = true, .name = "metal minimax_m3 glue mm_add_norm 1024 threads normed" },
    .{ .kind = .add_norm, .threads = 256, .normed = true, .name = "metal minimax_m3 glue mm_add_norm 256 threads normed" },
    .{ .kind = .moe_add, .normed = false, .name = "metal minimax_m3 glue mm_moe_add top 4 residual" },
    .{ .kind = .moe_add_norm, .threads = 1024, .normed = false, .name = "metal minimax_m3 glue mm_moe_add_norm top 4 residual" },
    .{ .kind = .moe_add_norm, .threads = 1024, .normed = true, .name = "metal minimax_m3 glue mm_moe_add_norm top 4 normed" },
};

/// A thread's strided squares, the simd_sum tree, then the threadgroup's partials in turn.
fn depthOf(comptime threads: usize) fn (c.Shape) usize {
    return struct {
        fn f(s: c.Shape) usize {
            return (s.n + threads - 1) / threads + 5 + threads / 32;
        }
    }.f;
}

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

fn spec(v: Case) c.Spec {
    const k: usize = if (v.kind == .moe_add or v.kind == .moe_add_norm) topk else 1;
    // a decode window's rows first (the edges run at it), then a prompt chunk's and one row (the last row's norm)
    const served: []const c.Shape = switch (k) {
        1 => &.{ .{ .rows = 32, .n = hidden, .k = 1 }, .{ .rows = 4096, .n = hidden, .k = 1 }, .{ .rows = 1, .n = hidden, .k = 1 } },
        else => &.{ .{ .rows = 32, .n = hidden, .k = topk }, .{ .rows = 4096, .n = hidden, .k = topk }, .{ .rows = 1, .n = hidden, .k = topk } },
    };
    var s: c.Spec = .{
        .name = v.name,
        .accum = .none,
        .depth = glue.noSums,
        .out = .f32,
        .in_place = true,
        .tiles = .{ .{}, .{ .size = 256, .step = 32, .min = 32 }, .{} },
        .served = served,
    };
    if (v.normed) {
        s.accum = .f32;
        s.depth = if (v.threads == 1024) depthOf(1024) else depthOf(256);
        // the mean's division, + eps, rsqrt, x * inv, 1 + w and the product
        s.ops = 8;
        s.out = .bf16;
        s.in_place = false;
        s.tiles[1] = .{ .size = v.threads, .step = 32, .min = 32 };
        s.sums = .{ .bf16 = 0, .why = norm_why };
    }
    return s;
}

pub const Rows = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Case,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    r0: []f32 = &.{},
    x: []u16 = &.{},
    w: []u16 = &.{},
    y: []u16 = &.{},
    wts: []f32 = &.{},
    shared: []u16 = &.{},
    pick: []i32 = &.{},
    resid: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ins: [5]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Case) !Rows {
        const name = switch (v.kind) {
            .norm => "mm_norm",
            .add => "mm_add",
            .add_norm => "mm_add_norm",
            .moe_add => "mm_moe_add",
            .moe_add_norm => "mm_moe_add_norm",
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try glue.pipeline(device, name), .v = v };
    }
    pub fn deinit(t: *Rows) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Rows) void {
        inline for (.{ "r0", "x", "w", "y", "wts", "shared", "pick" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Rows) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = &vt };
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
        t.pick = try t.a.alloc(i32, slots);
        t.shared = try t.a.alloc(u16, if (t.moe()) rows * n else 0);
        if (fill == .stress) {
            // integers whose squares sum exactly in fp32 (at most 6,144 x 1,600) and overflow fp16 in any order
            for (t.r0) |*v| v.* = @floatFromInt(@as(i32, @intCast(20 + rng.uintLessThan(u32, 21))) * (if (rng.boolean()) @as(i32, 1) else -1));
            @memset(t.w, 0);
            @memset(t.x, 0);
            @memset(t.y, 0);
            @memset(t.shared, 0);
            for (t.wts) |*v| v.* = 0.25;
        } else {
            for (t.r0) |*v| v.* = (rng.float(f32) * 2 - 1) * 4;
            for (t.w) |*v| v.* = c.toBf16(rng.float(f32) - 0.5);
            for (t.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (t.y) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (t.shared) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            // bf16-sized routing weights summing to about 1 a row, as mm_pick_row leaves them
            for (t.wts) |*v| v.* = 0.05 + rng.float(f32) * 0.4;
        }
        for (t.pick, 0..) |*p, s| p.* = @intCast((s % topk) * 31 + 3);
        if (fill == .nonfinite) {
            // a row each, one row holding both infinities, in the residual; then the added terms' own
            const at = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 }, .{ 3, 5 } };
            const vals = [_]f32{ glue.planted[0], glue.planted[1], glue.planted[2], glue.planted[1], glue.planted[2] };
            for (at, vals) |p, v| if (p[0] < rows and p[1] < n) {
                t.r0[p[0] * n + p[1]] = v;
            };
            if (rows > 4 and n > 7) {
                if (t.x.len > 0) t.x[4 * n + 6] = c.toBf16(glue.planted[0]);
                if (t.moe()) {
                    t.y[(4 * topk + 1) * n + 6] = c.toBf16(glue.planted[1]);
                    t.wts[4 * topk + 2] = glue.planted[2];
                    t.shared[4 * n + 7] = c.toBf16(glue.planted[0]);
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
            t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.pick));
            t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.wts));
            t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.shared));
        }
        return true;
    }

    pub fn run(t: *Rows) !void {
        if (t.v.kind != .norm) @memcpy(t.rig.bytes(t.resid, t.r0.len * 4), std.mem.sliceAsBytes(t.r0));
        const rows = t.shape.rows;
        const n = t.shape.n;
        const args = MArgs{ .rows = @intCast(rows), .width = @intCast(n), .heads = topk, .eps = eps, .scale = scale };
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
                for (1..5) |i| e.setBuffer(t.ins[i].buf, t.ins[i].off, i);
                if (t.v.kind == .moe_add) {
                    e.setValue(args, 5);
                } else {
                    e.setBuffer(t.ins[0].buf, t.ins[0].off, 5);
                    e.setValue(args, 6);
                    e.setBuffer(t.out.buf, t.out.off, 7);
                }
            },
        }
        if (t.hasNorm()) {
            e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(t.v.threads, 1, 1));
        } else e.dispatchThreads(mtl.Size.of(n, rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Rows) usize {
        return t.shape.rows * t.shape.n;
    }

    /// The routed experts' sum at transformers' points (bf16 after each product and each add), scaled, plus the shared.
    fn moeOut(t: *const Rows, r: usize, i: usize) f64 {
        const n = t.shape.n;
        var acc: f64 = 0;
        for (0..topk) |j| {
            const s = r * topk + j;
            acc = bf(acc + bf(@as(f64, c.fromBf16(t.y[s * n + i])) * t.wts[s]));
        }
        return bf(bf(acc * scale) + c.fromBf16(t.shared[r * n + i]));
    }

    /// Element i's residual after the kernel's add (each step one fp32 rounding of exact values, then its bf16 point).
    fn residAt(t: *const Rows, i: usize) f64 {
        const x0: f64 = t.r0[i];
        return switch (t.v.kind) {
            .norm => x0,
            .add, .add_norm => bf(x0 + c.fromBf16(t.x[i])),
            .moe_add, .moe_add_norm => bf(x0 + t.moeOut(i / t.shape.n, i % t.shape.n)),
        };
    }

    /// Row r's residual into `xs`, and its squares into `sq`.
    fn row(t: *const Rows, r: usize, xs: []f64, sq: []f64) void {
        for (xs, sq, 0..) |*x, *q, j| {
            x.* = t.residAt(r * t.shape.n + j);
            q.* = x.* * x.*;
        }
    }

    fn normed(t: *const Rows, x: f64, j: usize, ss: f64) f64 {
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(t.shape.n)) + @as(f64, eps));
        return x * inv * (1 + @as(f64, c.fromBf16(t.w[j])));
    }

    pub fn reference(t: *Rows, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        const buf = try t.a.alloc(f64, 2 * n);
        defer t.a.free(buf);
        for (0..t.shape.rows) |r| {
            t.row(r, buf[0..n], buf[n..]);
            var ss: f64 = 0;
            for (buf[n..]) |q| ss += q;
            for (buf[0..n], 0..) |x, j| {
                const i = r * n + j;
                want[i] = if (t.v.normed) t.normed(x, j, ss) else x;
                // the sum's relative error moves the output by as much again at most
                mag[i] = if (t.v.normed and std.math.isFinite(want[i])) @abs(want[i]) else 0;
            }
        }
    }

    pub fn simulate(t: *Rows, acc: c.Accum, order: g.Order, out: []f64) !void {
        const n = t.shape.n;
        const buf = try t.a.alloc(f64, 2 * n);
        defer t.a.free(buf);
        for (0..t.shape.rows) |r| {
            t.row(r, buf[0..n], buf[n..]);
            const ss = contract.rows.sumIn(acc, order, buf[n..]);
            for (buf[0..n], 0..) |x, j| out[r * n + j] = bf(t.normed(x, j, ss));
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    tests: std.ArrayList(*Rows) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (cases) |v| {
            if (only) |o| if (std.mem.indexOf(u8, v.name, o) == null) continue;
            const x = try s.a.create(Rows);
            x.* = try Rows.init(s.a, s.device, s.queue, v);
            try s.tests.append(s.a, x);
            try s.kernels.append(s.a, x.kernel());
        }
    }
    pub fn deinit(s: *Suite) void {
        for (s.tests.items) |x| x.deinit();
        s.tests.deinit(s.a);
    }
};
