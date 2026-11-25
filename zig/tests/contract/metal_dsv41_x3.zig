//! DeepSeek V4.1's EXL3 routed experts under the contract: the gate/up with its SwiGLU, the down and main's combine.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const eref = @import("dsv41_experts_ref.zig");
const bank = @import("dsv41_x3_bank.zig");
const chain = @import("metal_dsv41_x3_chain.zig");

const c = contract.spec;
const g = contract.gates;
const x3 = bank.x3;
const ref = bank.ref;
const fmt = tf.exl3_projection.format;
const mx = tf.dsv41_metal.experts;
const topk = bank.topk;
const count_experts = 16;
const base = "dsv41 x3 exl3 experts ";

pub const Which = enum { up, down };
/// The kernel, the experts' widths, the combine's output, and a planted fault (1: TwoSum's error term, 2: the weight).
pub const Variant = struct { which: Which, mix: bank.Mix = .mul1, bf16_out: bool = false, ctl: u8 = 0 };

const rule = "bytes equal the serial reference's (the one-row EXL3 dot, main's rotations, SwiGLU and combine); NaN as NaN";
const kept = "low[v][r] += (high[v][r] - (next - part)) + (product - part);";
const weighted = "tf_x3_silu(g) * u * wts[p]";

fn sh(rows: usize, n: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = k };
}

/// Decode row counts at a narrow width (each slot up to 16 rows, so passes of 4, 2 and 1), then V4.1's own sizes.
fn servedOf(comptime n: usize, comptime k: usize, comptime real: [2]usize) [10]c.Shape {
    var out: [10]c.Shape = undefined;
    for ([_]usize{ 1, 2, 3, 4, 6, 8, 12, 16 }, 0..) |r, i| out[i] = sh(r, n, k);
    out[8] = sh(1, real[0], real[1]);
    out[9] = sh(4, real[0], real[1]);
    return out;
}
const served_up = servedOf(256, 1024, .{ 2304, 5120 });
const served_down = servedOf(512, 256, .{ 5120, 2304 });

/// The K chain of compensated adds, then the rotations' seven adds a side.
fn depth(s: c.Shape) usize {
    return s.k + 14;
}

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .ops = 4,
        .out = if (v.which == .up or v.bf16_out) .bf16 else .f32,
        .nan = .{ .stated = rule },
        .tiles = .{ .{}, .{ .size = 256, .step = 128, .min = 128 }, .{ .size = 256, .step = 128, .min = 128 } },
        .served = if (v.which == .up) &served_up else &served_down,
        .sums = .{ .bf16 = 0, .why = "gate 4's halves mirror exactly, so bf16's pairwise order cancels them; fp16 overflows in every order" },
    };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: x3.Pipelines,
    main: ?mx.Pipelines = null,
    silu: *ref.Silu,
    shape: c.Shape = undefined,
    plan: ?eref.Plan = null,
    b: ?bank.Bank = null,
    x: []u16 = &.{},
    backing: std.ArrayList(mtl.Buffer) = .empty,
    tables: x3.Tables = undefined,
    route: x3.Route = undefined,
    xr: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    ones: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    scratch: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        const edit: ?[2][]const u8 = switch (v.ctl) {
            1 => .{ kept, "" },
            2 => .{ weighted, "tf_x3_silu(g) * u" },
            else => null,
        };
        const text = if (edit) |e| try std.mem.replaceOwned(u8, a, x3.text, e[0], e[1]) else x3.text;
        defer if (edit != null) a.free(text);
        if (edit != null and std.mem.eql(u8, text, x3.text)) return error.ControlNotPlanted;
        var pipes = try x3.Pipelines.loadSource(device, text);
        errdefer pipes.deinit();
        const t = try bank.silu(a, device, queue);
        errdefer a.destroy(t);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = pipes, .silu = t, .main = if (v.which == .down) try mx.Pipelines.load(a, device) else null };
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        p.backing.deinit(p.a);
        p.pipes.deinit();
        if (p.main) |*m| m.deinit();
        p.a.destroy(p.silu);
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.plan) |q| q.deinit();
        if (p.b) |*b| b.deinit();
        p.a.free(p.x);
        p.plan = null;
        p.b = null;
        p.x = &.{};
        p.backing.clearRetainingCapacity();
        p.rig.reset();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = if (p.v.ctl != 0) .shapes else null };
    }

    const vt = g.hooks(Proj);

    fn call(p: *const Proj) x3.Call {
        const s = p.shape;
        return if (p.v.which == .up) .{ .rows = @intCast(s.rows), .k = @intCast(s.k), .n = @intCast(s.n), .experts = count_experts } else .{ .rows = @intCast(s.rows), .k = @intCast(s.n), .n = @intCast(s.k), .experts = count_experts };
    }

    fn pairs(p: *const Proj) usize {
        return p.shape.rows * topk;
    }

    /// x (up: tokens, down: pairs' act) in [-1, 1) with NaN and Inf, or gate 4's: blocks rotating to +-4096, sum 0.
    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16 or shape.k % 128 != 0 or shape.n % 128 != 0 or shape.k > 5120) return false;
        p.drop();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const cl = p.call();
        p.plan = try eref.Plan.routed(p.a, shape.rows, count_experts, rng, .random);
        const kinds: []const bank.Kind = if (p.v.which == .up) &.{ .gate, .up } else &.{.down};
        p.b = try bank.Bank.init(p.a, count_experts, cl.k, cl.n, p.v.mix, kinds, if (fill == .stress) .stress else .random, rng);
        const rows = if (p.v.which == .up) shape.rows else p.pairs();
        const k = shape.k;
        p.x = try p.a.alloc(u16, rows * k);
        if (fill == .stress) {
            @memset(p.x, 0);
            for (0..rows) |r| for (0..k / 128) |blk| {
                p.x[r * k + blk * 128] = c.toBf16(if (blk < k / 256) 46336 else -46336);
            };
        } else {
            for (p.x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
            if (fill == .nonfinite and rows > 2) {
                p.x[k + 3] = 0x7fc0;
                p.x[2 * k + 5] = 0x7f80;
                p.x[2 * k + 9] = 0xff80;
            }
        }
        p.tables = (try p.b.?.upload(&p.rig, &p.backing)).tables;
        p.route = try bank.route(&p.rig, p.plan.?);
        p.xr = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.scratch = try p.rig.scratch(x3.scratchBytes(cl));
        if (p.v.which == .up) {
            p.y = try p.rig.output(p.pairs() * cl.n * 2);
        } else {
            const ones: [96]f32 = @splat(1);
            p.ones = try p.rig.input(std.mem.sliceAsBytes(ones[0..p.pairs()]));
            p.out = try p.rig.scratch(p.pairs() * cl.k * 2);
            p.y = try p.rig.output(shape.rows * cl.k * @as(usize, if (p.v.bf16_out) 2 else 4));
        }
        return true;
    }

    pub fn run(p: *Proj) !void {
        const cl = p.call();
        try x3.check(cl, p.tables);
        const pass = p.rig.begin();
        mx.use(pass.e, p.backing.items);
        if (p.v.which == .up) {
            x3.up(&p.pipes, pass.e, bank.at(p.xr), p.tables, p.route, bank.at(p.y), bank.at(p.scratch), cl, bank.limit);
        } else {
            x3.down(&p.pipes, pass.e, bank.at(p.xr), p.tables, p.route, bank.at(p.out), bank.at(p.scratch), cl);
            var r = p.route;
            r.weights = bank.at(p.ones);
            mx.combine(&p.main.?, pass.e, bank.at(p.out), r, r.weights, bank.at(p.y), !p.v.bf16_out, cl);
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return if (p.v.which == .up) p.pairs() * p.shape.n else p.shape.rows * p.shape.n;
    }

    /// The slots' sums (scratch) and every output against the serial reference's bits; a NaN where it has one.
    pub fn check(p: *Proj, gate: g.Gate) !?[]const u8 {
        const a = p.a;
        const cl = p.call();
        const np = p.pairs();
        const b = &p.b.?;
        const plan = p.plan.?;
        const sums_n: usize = if (p.v.which == .up) 2 * np * cl.n else np * cl.k;
        const sums = try a.alloc(f32, sums_n);
        defer a.free(sums);
        const lay = if (p.v.which == .up) x3.upLayout(cl) else x3.downLayout(cl);
        const got_sums: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, p.rig.bytes(p.scratch, lay.end)[lay.sums..]));
        const got = try p.rig.outputBytes();
        var bad: ?usize = null;
        var what: []const u8 = "";
        if (p.v.which == .up) {
            const act = try a.alloc(u16, np * cl.n);
            defer a.free(act);
            try bank.upAll(a, b, plan, p.x, p.silu, sums, act);
            if (bank.differ(u32, got_sums[0..sums_n], @ptrCast(sums))) |i| {
                bad = i;
                what = "slot sum";
            } else if (bank.differ(u16, @alignCast(std.mem.bytesAsSlice(u16, got)), act)) |i| {
                bad = i;
                what = "act";
            }
        } else {
            const out = try a.alloc(u16, np * cl.k);
            defer a.free(out);
            const y = try a.alloc(f32, p.shape.rows * cl.k);
            defer a.free(y);
            try bank.downAll(a, b, plan, p.x, sums, out);
            try bank.combineAll(a, plan, out, cl.k, cl.experts / 8, y);
            const got_out: []align(1) const u16 = std.mem.bytesAsSlice(u16, p.rig.bytes(p.out, np * cl.k * 2));
            if (bank.differ(u32, got_sums[0..sums_n], @ptrCast(sums))) |i| {
                bad = i;
                what = "slot sum";
            } else if (bank.differ(u16, @alignCast(got_out), out)) |i| {
                bad = i;
                what = "expert output";
            } else if (p.v.bf16_out) {
                const yb = try a.alloc(u16, y.len);
                defer a.free(yb);
                for (yb, y) |*h, v| h.* = ref.bf16(v);
                if (bank.differ(u16, @alignCast(std.mem.bytesAsSlice(u16, got)), yb)) |i| {
                    bad = i;
                    what = "combined row";
                }
            } else if (bank.differ(u32, @alignCast(std.mem.bytesAsSlice(u32, got)), @ptrCast(y))) |i| {
                bad = i;
                what = "combined row";
            }
        }
        if (bad) |i| return try std.fmt.bufPrint(&p.msg, "{s} {d} differs from the serial reference's", .{ what, i });
        if (gate != .nonfinite) return null;
        const e = spec(p.v, p.name).out;
        for (0..p.count()) |i| if (c.classOf(c.read(e, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&p.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    /// float64 of the formula on the serial rotated inputs; with `acc`, every K sum in that accumulator and order.
    fn formula(p: *Proj, out: []f64, acc: ?c.Accum, order: g.Order) !void {
        const a = p.a;
        const cl = p.call();
        const b = &p.b.?;
        const plan = p.plan.?;
        const np = p.pairs();
        const width: usize = cl.n;
        const hidden: usize = cl.k;
        const sides: []const bank.Kind = if (p.v.which == .up) &.{ .gate, .up } else &.{.down};
        const kk: usize = if (p.v.which == .up) hidden else width;
        const nn: usize = if (p.v.which == .up) width else hidden;
        const ys = try a.alloc(f64, 2 * np * nn);
        defer a.free(ys);
        const rot = try a.alloc(f16, kk);
        defer a.free(rot);
        const wq = try a.alloc(f16, 2 * kk * nn);
        defer a.free(wq);
        const t = try a.alloc(f64, kk);
        defer a.free(t);
        for (0..np) |q| for (sides, 0..) |kind, v| {
            const m = b.matrix(kind, plan.ids[q]);
            const xrow = if (p.v.which == .up) p.x[q / topk * kk ..][0..kk] else p.x[q * kk ..][0..kk];
            ref.rotIn(xrow, m.su, rot);
            bank.columns(m, kk, nn, wq[0 .. kk * nn], wq[kk * nn ..]);
            const y = ys[(v * np + q) * nn ..][0..nn];
            for (y, 0..) |*s, j| {
                for (t, wq[kk * nn + j * kk ..][0..kk], rot) |*e, w, r| e.* = @as(f64, r) * @as(f64, w);
                s.* = if (acc) |ac| contract.rows.sumIn(ac, order, t) else blk: {
                    var z: f64 = 0;
                    for (t) |e| z += e;
                    break :blk z;
                };
            }
            fmt.rotate(y, 1, nn, 1);
            for (y, m.sv) |*s, sc| s.* *= sc;
        };
        if (p.v.which == .up) {
            for (0..np) |q| for (0..nn) |j| {
                const gv = @min(ys[q * nn + j], bank.limit);
                const uv = std.math.clamp(ys[(np + q) * nn + j], -bank.limit, bank.limit);
                out[q * nn + j] = gv / (1 + @exp(-gv)) * uv * plan.weights[q];
            };
            return;
        }
        @memset(out, 0);
        for (0..np) |q| for (0..nn) |j| {
            out[q / topk * nn + j] += ys[q * nn + j];
        };
    }

    /// Exact sums (gate 4's inputs) make every output zero; the bound is the exact one, so `mag` is unused.
    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        return p.formula(want, null, .sequential);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.formula(out, acc, order);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{ .which = .up }, .name = base ++ "up tf_dsv41_x3_up, tf_dsv41_x3_act: mul1 2 to 5 bits by expert" },
        .{ .v = .{ .which = .up, .mix = .books }, .name = base ++ "up tf_dsv41_x3_up, tf_dsv41_x3_act: mul1, mcg, 3inst by expert" },
        .{ .v = .{ .which = .down }, .name = base ++ "down tf_dsv41_x3_down, main's combine: mul1 2 to 5 bits by expert" },
        .{ .v = .{ .which = .down, .mix = .books, .bf16_out = true }, .name = base ++ "down tf_dsv41_x3_down, main's bf16 combine: mul1, mcg, 3inst" },
        .{ .v = .{ .which = .down, .ctl = 1 }, .name = base ++ "control: tf_dsv41_x3_down without TwoSum's error term" },
        .{ .v = .{ .which = .up, .ctl = 1 }, .name = base ++ "control: tf_dsv41_x3_up without TwoSum's error term" },
        .{ .v = .{ .which = .up, .ctl = 2 }, .name = base ++ "control: tf_dsv41_x3_act without the route weight" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        errdefer s.destroy();
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            errdefer a.destroy(p);
            p.* = try Proj.init(a, device, queue, entry.v, entry.name);
            errdefer p.deinit();
            try contract.register(a, &s.probes, kernels, p, p.kernel());
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

    /// The whole lane on the GPU against the serial chain, and each pair's sums against tf_exl3_matmul_row's.
    pub fn lane(s: *Suite, seed: u64) !usize {
        return chain.run(s.a, s.device, s.queue, seed);
    }
};

test "dsv41 exl3 suite: a failed allocation at each step of the x3 create, both lists' growth included, frees all once" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var fail_index: usize = 0;
    while (true) : (fail_index += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const a = failing.allocator();
        var kernels: std.ArrayList(g.Kernel) = .empty;
        defer kernels.deinit(a);
        // two probes, so the second's failures unwind a registered first
        const s = Suite.create(a, device, queue, &kernels, base ++ "up tf_dsv41_x3_up") catch |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            continue;
        };
        defer s.destroy();
        try std.testing.expectEqual(@as(usize, 2), s.probes.items.len);
        try std.testing.expect(fail_index >= 4);
        break;
    }
}

const poison: u8 = 0x5a;

// the probe's outputs, the down's combine input included, each byte `poison` before the run
fn poisoned(p: *Proj) [2][]u8 {
    const cl = p.call();
    const out = if (p.v.which == .up) p.rig.bytes(p.y, p.pairs() * cl.n * 2) else p.rig.bytes(p.out, p.pairs() * cl.k * 2);
    const y = if (p.v.which == .up) out[0..0] else p.rig.bytes(p.y, p.shape.rows * cl.k * @as(usize, if (p.v.bf16_out) 2 else 4));
    @memset(out, poison);
    @memset(y, poison);
    return .{ out, y };
}

test "dsv41 exl3 suite: a descriptor table short of the call's experts is refused with every output still poisoned" {
    const a = std.testing.allocator;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    for ([_]Variant{ .{ .which = .up }, .{ .which = .down } }) |v| {
        var p = try Proj.init(a, device, queue, v, "short table");
        defer p.deinit();
        const shape = if (v.which == .up) served_up[7] else served_down[7];
        try std.testing.expect(try p.prepare(shape, .random, 11));
        p.tables.count = count_experts - 8;
        const outs = poisoned(&p);
        try std.testing.expectError(error.ShortExpertTable, p.run());
        for (outs) |o| for (o) |byte| try std.testing.expectEqual(poison, byte);
        // the full table writes every pair's row the combine reads
        p.tables.count = count_experts;
        _ = poisoned(&p);
        try p.run();
        const out = outs[0];
        for (0..out.len / 2) |i| try std.testing.expect(!std.mem.allEqual(u8, out[2 * i ..][0..2], poison));
    }
}
