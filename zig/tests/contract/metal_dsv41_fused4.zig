//! DeepSeek V4.1's fused window-key launch on Metal: rmsnorm, RoPE and act_quant a row of 512; controls 1-4.

const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_glue_ref.zig");
const rows_ = contract.rows;

const c = contract.spec;
const g = contract.gates;
const fu = tf.dsv41_metal.fused;
const base = "dsv41 fused ";
const eps: f32 = 1e-6;
const dim: usize = 512; // the window key's width: the host refuses any other
const reps: usize = 2; // ring rows a key scatters to; only the first is ever written here
const rope: usize = 64; // the rotated tail: the last 64 slots, taken in pairs

/// What a skipped row's slot keeps: finite, and far outside anything the kernel could compute for that row.
const keepRaw: f32 = 1e30;

/// pos: every fourth row skipped (the kernel returns on a negative one), the rest distinct, the table read widely.
fn posOf(r: usize) i32 {
    return if (r % 4 == 3) -1 else @intCast(r * 7 + 1);
}

fn normDepth(s: c.Shape) usize {
    return s.k / 256 + 14; // the rms tree, then the rope's products on the rotated tail
}

const rms_why = "a bf16 output follows the sum of squares as ss^-1/2, then the rope's two products on the tail";

pub fn spec(name: []const u8) c.Spec {
    const t = [_]c.Tile{ .{ .size = 16 }, .{}, .{} };
    return .{ .name = name, .accum = .f32, .depth = normDepth, .ops = 12, .out = .bf16, .tiles = t, .served = &served, .sums = .{ .bf16 = 0, .why = rms_why } };
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

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    ctl: u8,
    name: []const u8,
    pipes: *fu.Pipelines,
    shape: c.Shape = undefined,
    x: []u16 = &.{}, // [rows][512] the key, unrotated
    w: []f32 = &.{}, // kv_norm's weight [512]
    pos: []i32 = &.{}, // [rows]: a negative one skips the row
    cs: []f32 = &.{}, // [pos][64]: cos of each pair, then its sin
    dest: []i32 = &.{}, // [rows][reps]: the ring row, negative for no write
    in_x: rig_.Ref = undefined,
    in_w: rig_.Ref = undefined,
    in_pos: rig_.Ref = undefined,
    in_cs: rig_.Ref = undefined,
    in_dest: rig_.Ref = undefined,
    outs: [4]rig_.Ref = undefined, // KVN, KVR, KVQ, WIN

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

    /// Ring rows equal key rows: row r scatters to row r, so every ring slot is written by its own row or kept.
    pub fn count(p: *Proj) usize {
        return 4 * p.shape.rows * dim;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != dim or shape.n != 1 or shape.rows == 0) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.shape = shape;
        const n = shape.rows;
        p.x = try a.alloc(u16, n * dim);
        p.w = try a.alloc(f32, dim);
        p.pos = try a.alloc(i32, n);
        p.dest = try a.alloc(i32, n * reps);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (p.x) |*o| o.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (rng.uintLessThan(u32, 64) == 7) @as(f32, 20) else 1));
        for (p.w) |*o| o.* = 0.5 + rng.float(f32);
        for (p.pos, 0..) |*o, r| o.* = posOf(r);
        for (p.dest, 0..) |*o, i| o.* = if ((i % reps) == 0 and posOf(i / reps) >= 0) @intCast(i / reps) else -1;
        var top: usize = 1; // the table must reach the highest pos of any row, skipped rows included in the scan
        for (0..n) |q| {
            const pmax: usize = @intCast(@max(posOf(q), 0) + 1);
            top = @max(top, pmax);
        }
        p.cs = try a.alloc(f32, @max(top, 1) * 64);
        for (0..top) |q| {
            for (0..32) |j| {
                const angle: f64 = @as(f64, @floatFromInt(q)) * (0.1000001 + 0.01 * @as(f64, @floatFromInt(j)));
                p.cs[q * 64 + j] = @floatCast(@cos(angle));
                p.cs[q * 64 + 32 + j] = @floatCast(@sin(angle));
            }
        }

        if (fill == .nonfinite) p.plant(n);
        if (fill == .stress) try p.stress(a, n, seed);
        p.in_x = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.in_w = try p.rig.input(std.mem.sliceAsBytes(p.w));
        p.in_pos = try p.rig.input(std.mem.sliceAsBytes(p.pos));
        p.in_cs = try p.rig.input(std.mem.sliceAsBytes(p.cs));
        p.in_dest = try p.rig.input(std.mem.sliceAsBytes(p.dest));
        p.outs[0] = try p.rig.output(n * dim * 2);
        p.outs[1] = try p.rig.output(n * dim * 2);
        p.outs[2] = try p.rig.output(n * dim * 2);
        p.outs[3] = try p.rig.output(n * dim * 2);
        p.keepSkipped(); // the rows the kernel skips, and the ring rows of a key that scatters nowhere
        return true;
    }

    /// The sentinel bytes where the kernel must write nothing: a row with a negative pos, and its ring row.
    fn keepSkipped(p: *Proj) void {
        const bits = c.toBf16(keepRaw);
        for (0..p.shape.rows) |r| {
            const ring = p.dest[r * reps] < 0;
            if (p.pos[r] >= 0 and !ring) continue;
            for (&p.outs, 0..) |o, s| {
                if (s < 3 and p.pos[r] >= 0) continue; // only the ring is kept for a computed row
                const b = p.rig.bytes(o, p.shape.rows * dim * 2);
                for (std.mem.bytesAsSlice(u16, b[r * dim * 2 ..][0 .. dim * 2])) |*q| q.* = bits;
            }
        }
    }

    /// Five non-finites across four computed rows: the reference is then non-finite on those rows only.
    fn plant(p: *Proj, n: usize) void {
        const rows_ok = [_]usize{ 0, 1, 2, 4, 4 }; // a row 3 would be skipped by pos: planting it proves nothing
        const cols = [_]usize{ dim / 3, dim / 2, dim - 1, 0, dim - 1 };
        const put = [_]u16{ c.toBf16(std.math.nan(f32)), c.toBf16(std.math.inf(f32)), c.toBf16(-std.math.inf(f32)), c.toBf16(std.math.inf(f32)), c.toBf16(-std.math.inf(f32)) };
        for (rows_ok, cols, put) |r, q, v| if (r < n and p.pos[r] >= 0) {
            p.x[r * dim + q] = v;
        };
    }

    /// Gate 4's rows: the rms stress, so the fp32 sums of squares are exact integers in any order.
    fn stress(p: *Proj, a: std.mem.Allocator, n: usize, seed: u64) !void {
        var rr = try rows_.Rows.init(a, .rms, .{ .rows = n, .n = 1, .k = dim });
        defer rr.deinit();
        rr.fill(.stress, seed);
        for (0..n) |q| {
            for (0..dim) |i| {
                p.x[q * dim + i] = rr.x[q * dim + i];
            }
        }
        for (p.w, rr.w) |*o, v| o.* = @floatCast(c.fromBf16(v));
    }

    /// kn (rmsnorm's bf16) and kr (the rope's fp32 before its store) of a row, and the row's sum of squares.
    fn normRow(p: *Proj, r: usize, kn: []f64, kr: []f64, ss: *f64, acc: c.Accum, order: rows_.Order) void {
        for (0..dim) |i| {
            const y = c.fromBf16(p.x[r * dim + i]);
            kr[i] = y * y; // the squares go through kr's scratch, which the rope then overwrites on the tail
        }
        ss.* = if (acc == .none) blk: {
            var t: f64 = 0;
            for (kr) |v| t += v;
            break :blk t;
        } else rows_.sumIn(acc, order, kr);
        const inv = 1 / @sqrt(ss.* / @as(f64, @floatFromInt(dim)) + eps);
        for (0..dim) |i| kn[i] = c.fromBf16(c.toBf16(@floatCast(p.w[i] * (c.fromBf16(p.x[r * dim + i]) * inv))));
        for (dim - rope..dim) |i| {
            const pr = (i - (dim - rope)) / 2;
            const tab: usize = @as(usize, @intCast(p.pos[r])) * 64;
            const co: f64 = p.cs[tab + pr];
            const si: f64 = p.cs[tab + 32 + pr];
            kr[i] = if ((i & 1) == 0) kn[i] * co - kn[i ^ 1] * si else kn[i ^ 1] * si + kn[i] * co;
        }
        for (0..dim - rope) |i| kr[i] = kn[i]; // the untouched head: the kernel leaves kr at kn there
    }

    /// (KVN, KVR, KVQ) of every row and the ring it scatters into; a kept slot wants the sentinel it keeps.
    fn compute(p: *Proj, want: []f64, mag: []f64, acc: c.Accum, order: rows_.Order) !void {
        const n = p.shape.rows;
        const a = p.arena.allocator();
        const kn = try a.alloc(f64, dim);
        const kr = try a.alloc(f64, dim);
        var xs: [32]f32 = undefined;
        var bits: [32]u16 = undefined;
        for (0..n) |r| {
            if (p.pos[r] < 0) {
                const held = c.fromBf16(c.toBf16(keepRaw)); // the bytes keepSkipped wrote, read back exactly
                for (0..dim) |i| for (0..4) |sl| {
                    want[sl * n * dim + r * dim + i] = held;
                    mag[sl * n * dim + r * dim + i] = 0;
                };
                continue;
            }
            var ss: f64 = 0;
            p.normRow(r, kn, kr, &ss, acc, order);
            const ring_r = p.dest[r * reps];
            if (ring_r < 0) for (0..dim) |i| {
                want[3 * n * dim + r * dim + i] = c.fromBf16(c.toBf16(keepRaw));
                mag[3 * n * dim + r * dim + i] = 0;
            };
            for (0..dim) |i| {
                want[r * dim + i] = kn[i];
                mag[r * dim + i] = @abs(kn[i]) / 2;
                want[n * dim + r * dim + i] = c.fromBf16(c.toBf16(@floatCast(kr[i])));
                mag[n * dim + r * dim + i] = @abs(kn[i]) + @abs(kn[i ^ 1]); // the rope's two products, each <= its kn
            }
            var blk: usize = 0;
            while (blk < dim) : (blk += 32) {
                // the store rounds both lanes to bf16, so act_quant sees the rounded KVR value everywhere
                for (0..32) |j| xs[j] = @floatCast(c.fromBf16(c.toBf16(@floatCast(kr[blk + j]))));
                ref.roundTrip(.fp8, xs[0..32], bits[0..32]);
                for (0..32) |j| {
                    const q = c.fromBf16(bits[j]);
                    const at_q = 2 * n * dim + r * dim + blk + j;
                    want[at_q] = q;
                    mag[at_q] = @abs(kr[blk + j]) / 2;
                    if (ring_r >= 0) {
                        const at_w = 3 * n * dim + @as(usize, @intCast(ring_r)) * dim + blk + j;
                        want[at_w] = q;
                        mag[at_w] = @abs(kr[blk + j]) / 2;
                    }
                }
            }
        }
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        try p.compute(want, mag, .none, .sequential);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const scratch = try p.arena.allocator().alloc(f64, p.count());
        try p.compute(out, scratch, acc, order);
    }

    /// KVQ and the ring are E4M3 of KVR: a whole quantum |v|/8, plus KVR's own allowance and a floor for the scale.
    pub fn bounds(p: *Proj, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        const n = p.shape.rows;
        for (0..n) |r| {
            if (p.pos[r] < 0) continue; // kept slots hold the test's own bytes: the default bound proves them
            var blk: usize = 0;
            while (blk < dim) : (blk += 32) {
                var amax: f64 = 0;
                for (want[2 * n * dim + r * dim + blk ..][0..32]) |v| amax = @max(amax, @abs(v));
                for (0..32) |j| {
                    const at_kr = n * dim + r * dim + blk + j;
                    const q = @abs(want[2 * n * dim + r * dim + blk + j]) * 0x1p-3 + 2 * allow[at_kr] + amax * 0x1p-8;
                    allow[2 * n * dim + r * dim + blk + j] = q;
                    const ring_r = p.dest[r * reps];
                    if (ring_r >= 0) allow[3 * n * dim + @as(usize, @intCast(ring_r)) * dim + blk + j] = q;
                }
            }
        }
    }

    /// Rows r0 to r0 + nr: the ring base stays absolute, since the kernel scatters row r to the row its dest names.
    pub fn encode(p: *Proj, e: mtl.ComputeEncoder, r0: usize, nr: usize) !void {
        const step = dim * 2;
        const b: fu.Kv = .{
            .kv = at(p.in_x, r0 * step),
            .w = at(p.in_w, 0),
            .kvn = at(p.outs[0], r0 * step),
            .kvr = at(p.outs[1], r0 * step),
            .kvq = at(p.outs[2], r0 * step),
            .ring = at(p.outs[3], 0),
            .pos = at(p.in_pos, r0 * 4),
            .table = at(p.in_cs, 0),
            .dest = at(p.in_dest, r0 * reps * 4),
            .reps = @intCast(reps),
        };
        try fu.kv(p.pipes, e, b, @intCast(nr), @intCast(dim), eps);
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
        .{ .ctl = 0, .name = base ++ "kv tf_dsv41_f_kv" },
        .{ .ctl = 1, .name = base ++ "control: kv reads every non-finite input as zero" },
        .{ .ctl = 2, .name = base ++ "control: kv loses the rope's sign" },
        .{ .ctl = 3, .name = base ++ "control: kv lane totals flushed to fp16" },
        .{ .ctl = 4, .name = base ++ "control: kv writes past its window rows" },
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

    /// Launches of 1, 7 and 11 rows a launch equal one launch's bytes: no value crosses a row or a ring slot.
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
                for (&p.outs, 0..) |o, si| for (0..shape.rows) |r| {
                    if (p.pos[r] >= 0 or si == 3) @memset(p.rig.bytes(o, shape.rows * dim * 2)[r * dim * 2 ..][0 .. dim * 2], 0x55);
                };
                p.keepSkipped(); // the kept slots back to their sentinel: the whole run's bytes are the ones to match
                const pass = p.rig.begin();
                var r0: usize = 0;
                while (r0 < shape.rows) : (r0 += step) {
                    try p.encode(pass.e, r0, @min(step, shape.rows - r0));
                    launches += 1;
                }
                try p.rig.finish(pass);
                if (!std.mem.eql(u8, whole, try p.rig.outputBytes())) {
                    bad += 1;
                    std.debug.print("FAIL row invariance {s}: {d} rows a launch differs from one launch\n", .{ p.name, step });
                }
            }
        }
        std.debug.print("{s} row invariance {s}: kv at 1, 7 and 11 rows a launch ({d} launches) byte-equal to one launch, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, launches, bad });
        return bad;
    }
};
