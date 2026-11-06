//! A decode step's products under the contract: ud_matvec, and the experts' ud_expert_gu and ud_expert_down.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const k = @import("metal_uocr_decoder_common.zig");

const c = contract.spec;
const g = contract.gates;
const H = k.hidden;
const Mutation = k.Mutation;
const bf = k.bf;

/// A lane's two 4-wide dots and the adds each 256 columns, then the simd_sum tree.
fn mvDepth(s: c.Shape) usize {
    return 6 * ((s.k + 255) / 256) + 5;
}

/// Gate 4's input row: integers, halves mirrored (8 nudged); against `stressRow` partials pass 2^17, the dot is small.
fn stressX(rng: std.Random, x: []u16) void {
    const h = x.len / 2;
    for (0..h) |i| {
        const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
        x[i] = c.toBf16(v);
        x[h + i] = c.toBf16(v + @as(f32, if (i < 8) 1 else 0));
    }
}

fn stressRow(rng: std.Random, w: []u16) void {
    const h = w.len / 2;
    for (0..h) |i| {
        const q: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
        w[i] = c.toBf16(q);
        w[h + i] = c.toBf16(-q);
    }
}

/// y[n] = sum_j x[j] W[n, j] in f64 (want, mag), or in an accumulator and order (gate 4), each then at bf16.
fn dots(x: []const u16, w: []const u16, kk: usize, acc: c.Accum, order: g.Order, buf: []f64, want: []f64, mag: ?[]f64) void {
    for (want, 0..) |*o, r| {
        var s: f64 = 0;
        var m: f64 = 0;
        for (buf[0..kk], 0..) |*t, j| {
            t.* = @as(f64, c.fromBf16(x[j])) * c.fromBf16(w[r * kk + j]);
            s += t.*;
            m += @abs(t.*);
        }
        if (acc != .none) s = contract.rows.sumIn(acc, order, buf[0..kk]);
        o.* = if (acc == .none) s else bf(s);
        if (mag) |ms| ms[r] = if (std.math.isFinite(m)) m else 0;
    }
}

/// Plants NaN in row 0, +Inf in row 1, -Inf in row 2 and both in row 3 of `w`, at columns `x` holds as 1.
fn plantRows(x: []u16, w: []u16, kk: usize, rows: usize) void {
    const cols = [_]usize{ 1, kk / 3, kk / 2, kk - 1, 3 };
    const vals = [_]f32{ k.planted[0], k.planted[1], k.planted[2], k.planted[1], k.planted[2] };
    const on = [_]usize{ 0, 1, 2, 3, 3 };
    for (cols, vals, on) |j, v, r| if (r < rows) {
        x[j] = c.toBf16(1);
        w[r * kk + j] = c.toBf16(v);
    };
}

pub const Matvec = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: decode steps (a launch each); n: output rows; k: the input width.
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr decoder ud_matvec",
            .accum = .f32,
            .depth = mvDepth,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 256, .step = 8, .min = 8 } },
            // q | k | v | router q | router k, o, the shared expert's gate | up and down, the dense layer's, the head
            .served = &.{
                .{ .rows = 1, .n = 3 * H + 2 * k.router_dim, .k = H }, .{ .rows = 1, .n = H, .k = H },
                .{ .rows = 1, .n = 2 * k.shared_ffn, .k = H },         .{ .rows = 1, .n = H, .k = k.shared_ffn },
                .{ .rows = 1, .n = 2 * k.dense_ffn, .k = H },          .{ .rows = 1, .n = H, .k = k.dense_ffn },
                .{ .rows = 1, .n = k.vocab, .k = H },
            },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !Matvec {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "ud_matvec"), .m = m };
    }
    pub fn deinit(t: *Matvec) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Matvec) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Matvec);

    pub fn prepare(t: *Matvec, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 8 != 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.a.free(t.w);
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, shape.rows * shape.k);
        t.w = try t.a.alloc(u16, shape.n * shape.k);
        if (fill == .stress) {
            for (0..shape.rows) |r| stressX(rng, t.x[r * shape.k ..][0..shape.k]);
            for (0..shape.n) |r| stressRow(rng, t.w[r * shape.k ..][0..shape.k]);
        } else {
            k.fillBf16(rng, t.x, 2);
            k.fillBf16(rng, t.w, 0.1);
        }
        if (fill == .nonfinite) plantRows(t.x, t.w, shape.k, shape.n);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.out = try t.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(t: *Matvec) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setValue(extern struct { n: u32, k: u32 }{ .n = @intCast(t.shape.n), .k = @intCast(t.shape.k) }, 3);
        for (0..t.shape.rows) |r| {
            p.e.setBuffer(t.ins[0].buf, t.ins[0].off + r * t.shape.k * 2, 0);
            p.e.setBuffer(t.out.buf, t.out.off + r * t.shape.n * 2, 2);
            p.e.dispatchGroups(mtl.Size.of((t.shape.n + 31) / 32, 1, 1), mtl.Size.of(256, 1, 1));
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Matvec) usize {
        return t.shape.rows * t.shape.n;
    }

    fn outputs(t: *Matvec, acc: c.Accum, order: g.Order, want: []f64, mag: ?[]f64) !void {
        const kk = t.shape.k;
        const n = t.shape.n;
        const buf = try t.a.alloc(f64, kk);
        defer t.a.free(buf);
        for (0..t.shape.rows) |r| dots(t.x[r * kk ..][0..kk], t.w, kk, acc, order, buf, want[r * n ..][0..n], if (mag) |ms| ms[r * n ..][0..n] else null);
    }

    pub fn reference(t: *Matvec, want: []f64, mag: []f64) !void {
        try t.outputs(.none, .sequential, want, mag);
    }

    pub fn simulate(t: *Matvec, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.outputs(acc, order, out, null);
    }
};

/// Experts the test's table holds (the picks name 6 of them).
const table_experts = 8;

pub const Expert = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    /// ud_expert_down: the picks' down rows; else ud_expert_gu: their gate and up rows.
    down: bool,
    shape: c.Shape = undefined,
    width: usize = 0,
    x: []u16 = &.{},
    w: []u16 = &.{},
    ids: []u32 = &.{},
    table: [table_experts * 3]u64 = undefined,
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: decode steps (a launch each, 6 picks); n: each pick's outputs; k: its input width.
    pub fn spec(down: bool) c.Spec {
        return .{
            .name = if (down) "metal uocr decoder ud_expert_down picks 6" else "metal uocr decoder ud_expert_gu picks 6",
            .accum = .f32,
            .depth = mvDepth,
            .out = .bf16,
            .served = if (down) &.{.{ .rows = 1, .n = H, .k = k.expert_ffn }} else &.{.{ .rows = 1, .n = 2 * k.expert_ffn, .k = H }},
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation, down: bool) !Expert {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, if (down) "ud_expert_down" else "ud_expert_gu"), .m = m, .down = down };
    }
    pub fn deinit(t: *Expert) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.a.free(t.ids);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Expert) g.Kernel {
        return k.mark(.{ .spec = spec(t.down), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Expert);

    /// An expert's three matrices in the weights buffer: gate [w, 1280], up [w, 1280], down [1280, w].
    fn at(t: *const Expert, e: usize, which: usize) usize {
        return (e * 3 + which) * t.width * H;
    }

    /// The matrix pick j (of all steps' picks) multiplies by.
    fn matrix(t: *const Expert, j: usize) []const u16 {
        const e = t.ids[j];
        return if (t.down) t.w[t.at(e, 2)..][0 .. H * t.width] else t.w[t.at(e, 0)..][0 .. 2 * t.width * H];
    }

    /// x for pick j: its step's row (gate and up), or its own row (down).
    fn input(t: *const Expert, j: usize) []const u16 {
        return if (t.down) t.x[j * t.width ..][0..t.width] else t.x[(j / k.topk) * H ..][0..H];
    }

    pub fn prepare(t: *Expert, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0) return false;
        const width = if (t.down) shape.k else shape.n / 2;
        if ((if (t.down) shape.n != H else shape.k != H) or width == 0 or width % 8 != 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.a.free(t.w);
        t.a.free(t.ids);
        t.shape = shape;
        t.width = width;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // six distinct experts a step, the first not expert 0, so a pick never reads its own index by chance alone
        t.ids = try t.a.alloc(u32, shape.rows * k.topk);
        for (0..shape.rows) |r| {
            var order: [table_experts]u32 = undefined;
            for (&order, 0..) |*o, i| o.* = @intCast(i);
            rng.shuffle(u32, &order);
            if (order[0] == 0) std.mem.swap(u32, &order[0], &order[1]);
            @memcpy(t.ids[r * k.topk ..][0..k.topk], order[0..k.topk]);
        }
        const len = if (t.down) width else H;
        t.x = try t.a.alloc(u16, if (t.down) shape.rows * k.topk * width else shape.rows * H);
        t.w = try t.a.alloc(u16, table_experts * 3 * width * H);
        if (fill == .stress) {
            for (0..table_experts) |e| for (0..3) |which| {
                const rows: usize = if (which == 2) H else width;
                const cols: usize = if (which == 2) width else H;
                for (0..rows) |r| stressRow(rng, t.w[t.at(e, which) + r * cols ..][0..cols]);
            };
            for (0..t.x.len / len) |j| stressX(rng, t.x[j * len ..][0..len]);
        } else {
            k.fillBf16(rng, t.x, 2);
            k.fillBf16(rng, t.w, 0.1);
        }
        // the first pick's first rows, against the first input row
        if (fill == .nonfinite) plantRows(t.x[0..len], t.w[t.at(t.ids[0], if (t.down) 2 else 0)..], len, 4);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        const base = t.ins[1].buf.gpuAddress() + t.ins[1].off;
        for (0..table_experts) |e| for (0..3) |which| {
            t.table[e * 3 + which] = base + t.at(e, which) * 2;
        };
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(&t.table));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        t.out = try t.rig.output(shape.rows * k.topk * shape.n * 2);
        return true;
    }

    pub fn run(t: *Expert) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.useResource(t.ins[1].buf, mtl.ResourceUsage.read);
        p.e.setBuffer(t.ins[2].buf, t.ins[2].off, 1);
        p.e.setValue(@as(u32, @intCast(t.width)), 4);
        const xs = (if (t.down) k.topk * t.width else H) * 2;
        for (0..t.shape.rows) |r| {
            p.e.setBuffer(t.ins[0].buf, t.ins[0].off + r * xs, 0);
            p.e.setBuffer(t.ins[3].buf, t.ins[3].off + r * k.topk * 4, 2);
            p.e.setBuffer(t.out.buf, t.out.off + r * k.topk * t.shape.n * 2, 3);
            p.e.dispatchGroups(mtl.Size.of(k.topk * t.shape.n / 32, 1, 1), mtl.Size.of(256, 1, 1));
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Expert) usize {
        return t.shape.rows * k.topk * t.shape.n;
    }

    fn outputs(t: *Expert, acc: c.Accum, order: g.Order, want: []f64, mag: ?[]f64) !void {
        const kk = t.shape.k;
        const buf = try t.a.alloc(f64, kk);
        defer t.a.free(buf);
        const n = t.shape.n;
        for (0..t.ids.len) |j| dots(t.input(j), t.matrix(j), kk, acc, order, buf, want[j * n ..][0..n], if (mag) |ms| ms[j * n ..][0..n] else null);
    }

    pub fn reference(t: *Expert, want: []f64, mag: []f64) !void {
        try t.outputs(.none, .sequential, want, mag);
    }

    pub fn simulate(t: *Expert, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.outputs(acc, order, out, null);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    matvecs: std.ArrayList(*Matvec) = .empty,
    experts: std.ArrayList(*Expert) = .empty,

    fn add(s: *Suite, comptime T: type, list: *std.ArrayList(*T), only: ?[]const u8, m: Mutation, extra: anytype) !void {
        const x = try s.a.create(T);
        x.* = try @call(.auto, T.init, .{ s.a, s.device, s.queue, m } ++ extra);
        const kern = x.kernel();
        if (!k.wanted(only, kern.spec.name)) {
            x.deinit();
            s.a.destroy(x);
            return;
        }
        try list.append(s.a, x);
        try s.kernels.append(s.a, kern);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Mutation{ .none, .matvec_drops_tail, .matvec_half_sums }) |m| try s.add(Matvec, &s.matvecs, only, m, .{});
        for ([_]bool{ false, true }) |down| try s.add(Expert, &s.experts, only, .none, .{down});
        try s.add(Expert, &s.experts, only, .expert_gu_ignores_ids, .{false});
        try s.add(Expert, &s.experts, only, .expert_down_ignores_ids, .{true});
    }

    pub fn deinit(s: *Suite) void {
        inline for (.{ "matvecs", "experts" }) |f| {
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
    return k.limitsText(name, &.{ "ud_matvec", "ud_expert_gu", "ud_expert_down" });
}
