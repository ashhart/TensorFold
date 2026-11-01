//! DeepSeek V4.1's fused MoE join and hc_post on Metal: routed plus shared into B, four post streams into Y.

const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const rows_ = contract.rows;

const c = contract.spec;
const g = contract.gates;
const fu = tf.dsv41_metal.fused;
const base = "dsv41 fused ";

const dim: usize = 5120; // the model's hidden width: the same one the fold suite runs its combpost at
const streams: usize = 4; // residual streams hc_post reads, and output streams a row

/// Gate 4's comb: 16384 + 16 - 16388 - 8 is four exact fp32 terms that cancel to 4 and lose it in fp16 or bf16.
const res_of = [streams]f32{ 16384, 16, 16384, 8 };
const cb_of = [streams]f32{ 1, 1, -1.000244140625, -1 }; // -1.000244140625 of 16384 is -16388: 1 + 2^-12

fn joinDepth(_: c.Shape) usize {
    return 16; // the comb's four adds, the post's product and add, the join's add, with headroom for the chain
}

const sums_why = "four comb products in stream order, then the post's product onto the joined bf16";

pub fn spec(name: []const u8) c.Spec {
    const t = [_]c.Tile{ .{ .size = 16 }, .{}, .{} };
    return .{ .name = name, .accum = .f32, .depth = joinDepth, .ops = 2, .out = .bf16, .tiles = t, .served = &served, .sums = .{ .why = sums_why } };
}

const served = [_]c.Shape{ .{ .rows = 8, .n = 1, .k = dim }, .{ .rows = 1, .n = 1, .k = dim }, .{ .rows = 5, .n = 1, .k = dim } };

fn controlGate(ctl: u8) ?g.Gate {
    return switch (ctl) {
        1 => .nonfinite,
        2 => .shapes,
        3 => .sums,
        else => .repeat,
    };
}

/// The fused pipelines hold every kernel of fused.metal; every probe of a control shares one compiled copy.
var shared: [5]?*fu.Pipelines = .{ null, null, null, null, null };

fn sharedPipes(a: std.mem.Allocator, device: mtl.Device, ctl: u8) !*fu.Pipelines {
    if (shared[ctl]) |p| return p;
    const p = try a.create(fu.Pipelines);
    if (ctl == 0) {
        p.* = try fu.Pipelines.load(device);
    } else {
        const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.fused });
        defer a.free(text);
        p.* = try fu.Pipelines.loadSource(device, text);
    }
    shared[ctl] = p;
    return p;
}

const vt = g.hooks(Proj);
const plain_vt = blk: {
    var v = g.hooks(Proj);
    v.replaced = null; // a control is proven at its own gate, not against the baseline
    break :blk v;
};

fn at(r: rig_.Ref, extra: usize) fu.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    ctl: u8,
    name: []const u8,
    pipes: *fu.Pipelines,
    shape: c.Shape = undefined,
    rt: []f32 = &.{}, // [rows][dim] the routed expert's fp32 output
    sh: []u16 = &.{}, // [rows][dim] the shared expert's bf16 output
    res: []u16 = &.{}, // [rows][4][dim] the residual streams
    po: []f32 = &.{}, // [rows][4] hc_post's weights
    cb: []f32 = &.{}, // [rows][16] comb's weights, stream-major then output
    in_rt: rig_.Ref = undefined,
    in_sh: rig_.Ref = undefined,
    in_res: rig_.Ref = undefined,
    in_po: rig_.Ref = undefined,
    in_cb: rig_.Ref = undefined,
    out_b: rig_.Ref = undefined,
    out_y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: u8, name: []const u8) !Proj {
        return .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .ctl = ctl, .name = name, .pipes = try sharedPipes(a, device, ctl) };
    }

    pub fn deinit(p: *Proj) void {
        p.arena.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        if (p.ctl == 0) return .{ .spec = spec(p.name), .ctx = p, .vt = &vt, .control = null };
        return .{ .spec = spec(p.name), .ctx = p, .vt = &plain_vt, .control = controlGate(p.ctl) };
    }

    pub fn count(p: *Proj) usize {
        return (1 + streams) * p.shape.rows * dim;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != dim or shape.n != 1 or shape.rows == 0) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.shape = shape;
        const n = shape.rows;
        p.rt = try a.alloc(f32, n * dim);
        p.sh = try a.alloc(u16, n * dim);
        p.res = try a.alloc(u16, n * streams * dim);
        p.po = try a.alloc(f32, n * streams);
        p.cb = try a.alloc(f32, n * 16);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (p.rt) |*o| o.* = (rng.float(f32) * 2 - 1) * (if (rng.uintLessThan(u32, 64) == 7) @as(f32, 20) else 1);
        for (p.sh) |*o| o.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (rng.uintLessThan(u32, 64) == 7) @as(f32, 20) else 1));
        for (p.res) |*o| o.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (rng.uintLessThan(u32, 64) == 7) @as(f32, 20) else 1));
        for (p.po) |*o| o.* = 0.5 + rng.float(f32);
        for (p.cb) |*o| o.* = 0.25 * @as(f32, @floatFromInt(rng.uintLessThan(u32, 17)));
        if (fill == .nonfinite) p.plant(n);
        if (fill == .stress) p.stress(n);
        p.in_rt = try p.rig.input(std.mem.sliceAsBytes(p.rt));
        p.in_sh = try p.rig.input(std.mem.sliceAsBytes(p.sh));
        p.in_res = try p.rig.input(std.mem.sliceAsBytes(p.res));
        p.in_po = try p.rig.input(std.mem.sliceAsBytes(p.po));
        p.in_cb = try p.rig.input(std.mem.sliceAsBytes(p.cb));
        p.out_b = try p.rig.output(n * dim * 2);
        p.out_y = try p.rig.output(n * streams * dim * 2);
        return true;
    }

    /// Five non-finites on five rows: the join's two inputs and the three residual streams the kernel reads.
    fn plant(p: *Proj, n: usize) void {
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const cols = [_]usize{ dim / 3, dim / 2, dim - 1, 0, dim - 1 };
        for (0..@min(n, 5)) |r| {
            if (r == 0) p.rt[r * dim + cols[r]] = std.math.nan(f32);
            if (r == 1) p.sh[r * dim + cols[r]] = inf;
            if (r == 2) p.sh[r * dim + cols[r]] = ninf;
            if (r == 3) p.res[(r * streams + 1) * dim + cols[r]] = inf;
            if (r == 4) p.res[(r * streams + 3) * dim + cols[r]] = ninf;
        }
    }

    /// Gate 4's rows: the comb as above, post and the join at zero, so every Y is the comb's exact four.
    fn stress(p: *Proj, n: usize) void {
        @memset(p.rt, 0);
        @memset(p.sh, 0);
        @memset(p.po, 0);
        for (0..n) |r| {
            for (0..streams) |j| {
                for (0..dim) |i| p.res[(r * streams + j) * dim + i] = c.toBf16(res_of[j]);
            }
            for (0..16) |q| p.cb[r * 16 + q] = cb_of[q / streams];
        }
    }

    /// B and the four post streams of every column; the oracle stays honest, computing the spec, never the control.
    fn compute(p: *Proj, want: []f64, mag: []f64, acc: c.Accum, order: g.Order) void {
        const n = p.shape.rows;
        for (0..n) |r| for (0..dim) |i| {
            const rt: f64 = p.rt[r * dim + i];
            const sh: f64 = c.fromBf16(p.sh[r * dim + i]);
            const b = bf(rt + sh); // B is stored bf16, so every stream reads the rounded join
            want[r * dim + i] = b;
            mag[r * dim + i] = @abs(rt) + @abs(sh);
            var t: [streams]f64 = undefined;
            var ab: f64 = 0;
            for (0..streams) |k| {
                ab = 0; // the comb's four terms of this output stream, summed by the simulated accumulator
                for (0..streams) |j| {
                    t[j] = p.cb[r * 16 + j * streams + k] * c.fromBf16(p.res[(r * streams + j) * dim + i]);
                    ab += @abs(t[j]);
                }
                const mm = rows_.sumIn(acc, order, t[0..streams]);
                const at_y = (n + streams * r + k) * dim + i; // Y's bytes follow B's: [rows][4][dim] as the kernel indexes it
                want[at_y] = p.po[r * streams + k] * b + mm;
                mag[at_y] = @abs(p.po[r * streams + k] * b) + ab;
            }
        };
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        p.compute(want, mag, .none, .sequential);
    }

    /// The same outputs with the comb's adds in `acc`; mag goes to a scratch so it never lands on the outputs.
    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const scratch = try p.arena.allocator().alloc(f64, p.count());
        p.compute(out, scratch, acc, order);
    }

    pub fn encode(p: *Proj, e: mtl.ComputeEncoder, r0: usize, nr: usize) !void {
        const d2 = dim * 2; // bf16 buffers' row stride in bytes; the routed fp32's is twice that
        fu.joinpost(p.pipes, e, at(p.in_rt, r0 * dim * 4), at(p.in_sh, r0 * d2), at(p.out_b, r0 * d2), at(p.in_res, r0 * streams * d2), at(p.in_po, r0 * streams * 4), at(p.in_cb, r0 * 16 * 4), at(p.out_y, r0 * streams * d2), @intCast(nr), @intCast(dim));
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, 0, p.shape.rows);
        try p.rig.finish(pass);
    }
};

/// The registration and its controls, plus the row-invariance probe over the baseline's launches.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { ctl: u8, name: []const u8 };
    const list = [_]E{
        .{ .ctl = 0, .name = base ++ "joinpost tf_dsv41_f_joinpost" },
        .{ .ctl = 1, .name = base ++ "control: joinpost reads every non-finite input as zero" },
        .{ .ctl = 2, .name = base ++ "control: joinpost loses the shared expert and stream 3" },
        .{ .ctl = 3, .name = base ++ "control: joinpost's third comb product in fp16" },
        .{ .ctl = 4, .name = base ++ "control: joinpost writes past its streams" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.ctl, entry.name);
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
        for (&shared) |*p| if (p.*) |pp| {
            pp.deinit();
            s.a.destroy(pp);
            p.* = null;
        };
        s.a.destroy(s);
    }

    /// Launches of 1, 7 and 11 rows a launch equal one launch's bytes: no value crosses a row.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var launches: usize = 0;
        for (s.probes.items) |p| {
            if (p.ctl != 0) continue;
            const shape = c.Shape{ .rows = 33, .n = 1, .k = dim };
            if (!try p.prepare(shape, .random, seed)) return error.Refused;
            try p.run();
            const whole = try s.a.dupe(u8, try p.rig.outputBytes());
            defer s.a.free(whole);
            for ([_]usize{ 1, 7, 11 }) |step| {
                @memset(p.rig.bytes(p.out_b, shape.rows * dim * 2), 0x55);
                @memset(p.rig.bytes(p.out_y, shape.rows * streams * dim * 2), 0x55);
                const pass = p.rig.begin();
                var r0: usize = 0;
                while (r0 < shape.rows) : (r0 += step) {
                    try p.encode(pass.e, r0, @min(step, shape.rows - r0));
                    launches += 1;
                }
                try p.rig.finish(pass);
                if (!std.mem.eql(u8, whole, try p.rig.outputBytes())) {
                    bad += 1;
                    const d = std.mem.indexOfDiff(u8, whole, try p.rig.outputBytes()).?;
                    std.debug.print("FAIL row invariance {s}: {d} rows a launch differs from one launch at byte {d}\n", .{ p.name, step, d });
                }
            }
        }
        const line = "row invariance {s}: joinpost at 1, 7 and 11 rows a launch ({d} launches) byte-equal, {d} differ\n";
        std.debug.print("{s} " ++ line, .{ if (bad == 0) "PASS" else "FAIL", base, launches, bad });
        return bad;
    }
};
