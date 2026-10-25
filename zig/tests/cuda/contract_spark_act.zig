//! act_rows.cu's Spark2.5 rows under the contract (the head gate, the exact GELU), and controls dropping their point.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const round_ = @import("act_round_check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;
const aref = core.act_ref;

pub const Kind = enum { head_gate, gelu_erf };

/// A registration: a served kernel, or a control that drops its one rounding point (act_round.cuh's `keep` 0).
pub const Entry = struct { name: []const u8, kind: Kind, control: bool = false };

pub const entries = [_]Entry{
    .{ .name = "cuda act tf_head_gate_rows (Spark2.5's head gate, the gate rounded first)", .kind = .head_gate },
    .{ .name = "cuda act tf_gelu_erf_rows (Spark2.5's exact GELU, the gelu rounded first)", .kind = .gelu_erf },
    .{ .name = "control: head gate without the gate's rounding", .kind = .head_gate, .control = true },
    .{ .name = "control: exact GELU without the gelu's rounding", .kind = .gelu_erf, .control = true },
};

/// The control kernel an entry launches.
pub fn symbol(e: Entry) [:0]const u8 {
    return if (e.kind == .head_gate) "tf_ctl_head_gate" else "tf_ctl_gelu_erf";
}

fn depth(_: c.Shape) usize {
    return 0;
}

/// Gate rows' stride: past the heads, as the gate sits in a fused projection's row.
fn gateLd(heads: usize) usize {
    return heads + 24;
}

pub fn spec(e: Entry) c.Spec {
    return switch (e.kind) {
        // rows x heads x head dim: Spark2.5-4B's 16 heads of 256, decode rounds and prompt chunks
        .head_gate => .{ .name = e.name, .accum = .none, .depth = depth, .mids = 16, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 256, .step = 64, .min = 64 } }, .served = &.{ .{ .rows = 1, .n = 16, .k = 256 }, .{ .rows = 16, .n = 16, .k = 256 }, .{ .rows = 128, .n = 16, .k = 256 }, .{ .rows = 512, .n = 16, .k = 256 } } },
        // rows x the MLP's width
        .gelu_erf => .{ .name = e.name, .accum = .none, .depth = depth, .mids = 16, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{ .size = 1024, .step = 64, .min = 64 }, .{} }, .served = &.{ .{ .rows = 1, .n = 10240, .k = 1 }, .{ .rows = 16, .n = 10240, .k = 1 }, .{ .rows = 128, .n = 10240, .k = 1 }, .{ .rows = 512, .n = 10240, .k = 1 } } },
    };
}

pub const Act = struct {
    rig: rig_.Rig,
    entry: Entry,
    act: *const cuda.act_rows.ActRows,
    ctl: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    rows: usize = 0,
    n: usize = 0, // outputs a row
    heads: usize = 0,
    d: usize = 0,
    a: []u16 = &.{}, // head gate: o [rows, n]; GELU: [gate | up] rows
    gate: []u16 = &.{}, // head gate: [rows, gateLd(heads)]
    in: [2]u64 = @splat(0),
    out: u64 = 0,
    xs: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Act) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Act);

    pub fn kernel(self: *Act) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.control) .shapes else null };
    }

    pub fn prepare(self: *Act, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        const gated = self.entry.kind == .head_gate;
        const n = if (gated) shape.n * shape.k else shape.n;
        if (n % 64 != 0 or (gated and shape.k % 64 != 0)) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const al = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.rows = shape.rows;
        self.n = n;
        self.heads = shape.n;
        self.d = shape.k;
        const nan = std.math.nan(f32);
        const inf = std.math.inf(f32);
        if (gated) {
            // o in [-2, 2); gates in [-40, 40), inside expf's fp32 range, where float64's sigmoid and fp32's agree
            self.a = try al.alloc(u16, shape.rows * n);
            for (self.a, 0..) |*v, i| v.* = bfr.bits((rng.float(f32) * 4 - 2) * (if (i % 64 == 7) @as(f32, 8) else 1));
            const ld = gateLd(shape.n);
            self.gate = try al.alloc(u16, shape.rows * ld);
            for (self.gate, 0..) |*v, i| v.* = bfr.bits((rng.float(f32) * 16 - 8) * (if (i % 7 == 3) @as(f32, 5) else 1));
            if (fill == .nonfinite) {
                const os = [_]struct { usize, usize, f32 }{ .{ 0, n / 3, nan }, .{ 1, n / 2, inf }, .{ 2, n - 1, -inf } };
                for (os) |p| if (p[0] < shape.rows) {
                    self.a[p[0] * n + p[1]] = bfr.bits(p[2]);
                };
                const gs = [_]struct { usize, usize, f32 }{ .{ 0, 0, nan }, .{ 1, shape.n - 1, inf }, .{ 3, shape.n / 2, -inf } };
                for (gs) |p| if (p[0] < shape.rows) {
                    self.gate[p[0] * ld + p[1]] = bfr.bits(p[2]);
                };
            }
            self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.a));
            self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.gate));
        } else {
            // [gate | up] rows, as a fused gate_up projection writes them; gates to 80, where erf is saturated
            self.a = try al.alloc(u16, shape.rows * 2 * n);
            for (self.a, 0..) |*v, i| {
                const up = i % (2 * n) >= n;
                const big: f32 = if (i % 64 == 7) 20 else 1;
                v.* = bfr.bits(if (up) rng.float(f32) * 4 - 2 else (rng.float(f32) * 8 - 4) * big);
            }
            if (fill == .nonfinite) {
                const planted = [_]struct { usize, usize, f32 }{ .{ 0, n / 3, nan }, .{ 1, n / 2, inf }, .{ 2, n - 1, -inf }, .{ 3, n + 5, inf }, .{ 3, 0, -inf } };
                for (planted) |p| if (p[0] < shape.rows) {
                    self.a[p[0] * 2 * n + p[1]] = bfr.bits(p[2]);
                };
            }
            self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.a));
        }
        self.out = try self.rig.output(shape.rows * n * 2);
        self.xs = try self.rig.output(shape.rows * n / 64 * 4);
        return true;
    }

    pub fn run(self: *Act) !void {
        const s = self.rig.s;
        const gated = self.entry.kind == .head_gate;
        if (self.ctl) |f| {
            var args: cuda.Args = .{};
            if (gated) {
                args.add(self.in[0]);
                args.add(self.in[1]);
                args.add(@as(c_longlong, @intCast(gateLd(self.heads))));
            } else {
                args.add(self.in[0]);
                args.add(@as(c_longlong, @intCast(2 * self.n)));
                args.add(self.in[0] + 2 * self.n);
                args.add(@as(c_longlong, @intCast(2 * self.n)));
            }
            args.add(self.out);
            args.add(self.xs);
            if (gated) {
                args.add(@as(c_int, @intCast(self.heads)));
                args.add(@as(c_int, @intCast(self.d)));
            } else args.add(@as(c_int, @intCast(self.n)));
            args.add(@as(c_int, 0));
            const t: u32 = cuda.launch.rowThreads(self.rows, 256, if (gated) 64 else 128);
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(self.rows), .y = @intCast((self.n / 8 + t - 1) / t), .z = 1 }, .block = .{ .x = t } }, s, &args);
        } else if (gated) {
            try self.act.headGate(s, self.in[0], self.in[1], gateLd(self.heads), self.out, self.xs, self.rows, self.heads, self.d);
        } else try self.act.gegluErf(s, self.in[0], 2 * self.n, self.in[0] + 2 * self.n, 2 * self.n, self.out, self.xs, self.rows, self.n);
        try s.synchronize();
    }

    pub fn count(self: *Act) usize {
        return self.rows * self.n;
    }

    /// Output i of row `row`: float64's ends at the kernel's rounding points, and its point.
    fn want(self: *const Act, row: usize, i: usize) struct { lo: f64, hi: f64, point: f64 } {
        if (self.entry.kind == .head_gate) {
            const o: f64 = bfr.value(self.a[row * self.n + i]);
            const x: f64 = bfr.value(self.gate[row * gateLd(self.heads) + i / self.d]);
            const r = aref.headGateRange(o, x, round_.spread);
            return .{ .lo = r[0], .hi = r[1], .point = aref.headGate(o, x, 0) };
        }
        const x: f64 = bfr.value(self.a[row * 2 * self.n + i]);
        const u: f64 = bfr.value(self.a[row * 2 * self.n + self.n + i]);
        const r = aref.geluErfRange(x, u);
        return .{ .lo = r[0], .hi = r[1], .point = aref.geluErf(x, u, 0, 0) };
    }

    pub fn reference(self: *Act, out: []f64, mag: []f64) !void {
        for (0..self.rows) |row| for (0..self.n) |i| {
            out[row * self.n + i] = self.want(row, i).point;
        };
        @memset(mag, 0);
    }

    /// Every output within float64's ends (NaN and infinities as float64's point), sums bit-equal to the fp32 tree.
    pub fn check(self: *Act, _: g.Gate) !?[]const u8 {
        const al = self.arena.?.allocator();
        const total = self.count();
        const out = try al.alloc(u16, total);
        const xs = try al.alloc(f32, total / 64);
        @memcpy(std.mem.sliceAsBytes(out), try self.rig.bytes(self.out, total * 2));
        @memcpy(std.mem.sliceAsBytes(xs), try self.rig.bytes(self.xs, total / 64 * 4));
        for (0..self.rows) |row| {
            for (0..self.n) |i| {
                const w = self.want(row, i);
                const got: f64 = bfr.value(out[row * self.n + i]);
                const ok = if (std.math.isFinite(w.lo) and std.math.isFinite(w.hi))
                    std.math.isFinite(got) and w.lo <= got and got <= w.hi
                else
                    c.classOf(got) == c.classOf(w.point);
                if (!ok) return try std.fmt.bufPrint(&self.msg, "row {d} out[{d}] = {e} outside float64's [{e}, {e}]", .{ row, i, got, w.lo, w.hi });
            }
            for (0..self.n / 64) |j| {
                const t = round_.groupSum(out[row * self.n + 64 * j ..][0..64]);
                const s = xs[row * (self.n / 64) + j];
                if (@as(u32, @bitCast(s)) != @as(u32, @bitCast(t)) and !(std.math.isNan(s) and std.math.isNan(t)))
                    return try std.fmt.bufPrint(&self.msg, "row {d} sum {d} = {e} where the fp32 tree of the stored out gives {e}", .{ row, j, s, t });
            }
        }
        return null;
    }
};

/// Every entry's case, on the heap so the kernels' contexts stay put; appended to `kernels` as made.
pub const Group = struct {
    gpa: std.mem.Allocator,
    acts: [entries.len]Act = undefined,
    made: usize = 0,

    pub fn create(gpa: std.mem.Allocator, gpu: anytype, s: cuda.Stream, act: *const cuda.act_rows.ActRows, controls: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const group = try gpa.create(Group);
        group.* = .{ .gpa = gpa };
        errdefer group.destroy();
        for (entries) |e| {
            const ctl: ?cuda.Function = if (e.control) try controls.function(symbol(e)) else null;
            group.acts[group.made] = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .act = act, .ctl = ctl };
            group.made += 1;
            try kernels.append(gpa, group.acts[group.made - 1].kernel());
        }
        return group;
    }

    pub fn destroy(group: *Group) void {
        for (group.acts[0..group.made]) |*x| x.deinit();
        group.gpa.destroy(group);
    }
};

test "every kind has its served kernel and one control" {
    var served = [2]u8{ 0, 0 };
    var controls = [2]u8{ 0, 0 };
    for (entries) |e| {
        const k = @backingInt(e.kind);
        if (e.control) controls[k] += 1 else served[k] += 1;
    }
    try std.testing.expectEqual([2]u8{ 1, 1 }, served);
    try std.testing.expectEqual([2]u8{ 1, 1 }, controls);
}
