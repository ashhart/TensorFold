//! norms.cu's bf16 residual tails under the contract (MLX's rounded rows and once-rounded ones), and their controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const tail_ = @import("norms_tail_check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const ref = core.norms_ref;
const bfr = core.bf16_round;

/// The kernels by output checked (y or residual h; add_: add_rms_h, moe_add_: add_rms_h on one combined expert slot
/// of weight 1, the residual's bytes; parts_: strided parts), controls dropping a point.
pub const Which = enum { round_y, round_h, once_y, once_h, add_y, add_h, moe_add_y, moe_add_h, parts_once_y, parts_round_y, drop_branch_row, drop_branch_out, drop_sum, drop_row };

fn noSums(_: c.Shape) usize {
    return 0;
}

fn depth(_: c.Shape) usize {
    return tail_.depth;
}

const why = "an output follows its sum of squares as ss^-1/2 through bf16 rounding points: a bf16 sum's error is a few output ulps, and an fp16 sum overflows to a zero output, 51x the bound that carries the points";

pub fn spec(which: Which) c.Spec {
    return .{
        .name = switch (which) {
            .round_y => "cuda norm tail tf_add_rms_rows_bf16h_tail_round (MLX's rows), output y",
            .round_h => "cuda norm tail tf_add_rms_rows_bf16h_tail_round (MLX's rows), residual h",
            .once_y => "cuda norm tail tf_add_rms_rows_bf16h_tail (rounded once), output y",
            .once_h => "cuda norm tail tf_add_rms_rows_bf16h_tail (rounded once), residual h",
            .add_y => "cuda norm tf_add_rms_rows_bf16h (h = x + r, rounded once), output y",
            .add_h => "cuda norm tf_add_rms_rows_bf16h (h = x + r, rounded once), residual h",
            .moe_add_y => "cuda norm tf_moe_add_rms_rows_bf16h (h = x + combine, one bf16 slot of weight 1), output y",
            .moe_add_h => "cuda norm tf_moe_add_rms_rows_bf16h (h = x + combine, one bf16 slot of weight 1), residual h",
            .parts_once_y => "cuda norm parts tf_rms_rows_bf16h_parts (rounded once, two parts a row), output y",
            .parts_round_y => "cuda norm parts tf_rms_rows_bf16h_parts_round (MLX's rows, two parts a row), output y",
            .drop_branch_row => "control: MLX tail without the branch's normalized-row rounding",
            .drop_branch_out => "control: MLX tail without the branch output's rounding",
            .drop_sum => "control: MLX tail without the sum's rounding before the scalar",
            .drop_row => "control: MLX tail without the output's normalized-row rounding",
        },
        .accum = if (which == .add_h or which == .moe_add_h) .none else .f32, // add_rms_h's h is one rounded add: no sum reaches it
        .depth = if (which == .add_h or which == .moe_add_h) noSums else depth,
        .ops = 8,
        .mids = 4, // a rounding point decided the other way by the fp32 inverse: two output half-ulps a point
        .out = .bf16,
        .tiles = .{ .{}, .{}, .{ .size = 2048, .step = 64, .min = 64 } },
        .served = switch (which) {
            .add_y, .add_h, .moe_add_y, .moe_add_h => &.{ .{ .rows = 16, .n = 2688, .k = 2688 }, .{ .rows = 300, .n = 2688, .k = 2688 } }, // Nemotron's residual
            .parts_once_y => &.{ .{ .rows = 32, .n = 2688, .k = 2688 }, .{ .rows = 2, .n = 2688, .k = 2688 } }, // its MTP halves
            .parts_round_y => &.{ .{ .rows = 128, .n = 512, .k = 512 }, .{ .rows = 2400, .n = 512, .k = 512 } }, // its Mamba groups
            else => &.{ .{ .rows = 17, .n = 3840, .k = 3840 }, .{ .rows = 3, .n = 5376, .k = 5376 } }, // Gemma 4 12B, 31B
        },
        .sums = .{ .f16 = 40, .bf16 = 0, .why = why },
    };
}

pub fn control(which: Which) ?g.Gate {
    return switch (which) {
        .round_y, .round_h, .once_y, .once_h, .add_y, .add_h, .moe_add_y, .moe_add_h, .parts_once_y, .parts_round_y => null,
        else => .shapes,
    };
}

/// The controls' symbols in contract_controls.cu.
pub fn symbol(which: Which) [:0]const u8 {
    return switch (which) {
        inline else => |w| "tf_ctl_tail_" ++ @tagName(w),
    };
}

fn round(which: Which) bool {
    return switch (which) {
        .once_y, .once_h, .add_y, .add_h, .moe_add_y, .moe_add_h, .parts_once_y => false,
        else => true,
    };
}

fn hView(which: Which) bool {
    return which == .round_h or which == .once_h or which == .add_h or which == .moe_add_h;
}

/// The residual is read by every form but the parts; the branch and scalar by the tails alone.
fn residual(which: Which) bool {
    return which != .parts_once_y and which != .parts_round_y;
}

fn branch(which: Which) bool {
    return residual(which) and which != .add_y and which != .add_h and !moe(which);
}

fn moe(which: Which) bool {
    return which == .moe_add_y or which == .moe_add_h;
}

/// The parts a row: two where the rows pair up, so part strides and row strides both differ from the width.
fn partsOf(rows_: usize) usize {
    return if (rows_ % 2 == 0) 2 else 1;
}

/// Sums of squares in an accumulator and order (gate 4's simulation).
const Sim = struct {
    acc: c.Accum,
    order: rows.Order,
    buf: []f64,

    pub fn squares(s: Sim, v: []const f64) f64 {
        for (s.buf[0..v.len], v) |*t, x| t.* = x * x;
        return rows.sumIn(s.acc, s.order, s.buf[0..v.len]);
    }
};

pub const Tail = struct {
    rig: rig_.Rig,
    which: Which,
    norms: *const cuda.norms.Norms,
    ctl: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    k: tail_.Case = undefined,
    dev: [5]u64 = undefined, // x, r, wb, sc, w
    ones: u64 = 0, // moe_add_'s slot weights, 1 a row
    out: [3]u64 = undefined, // h, y, xs
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Tail) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Tail);

    pub fn kernel(self: *Tail) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control(self.which) };
    }

    fn bfs(rng: std.Random, v: []u16, scale: f32) void {
        for (v, 0..) |*x, i| x.* = bfr.bits((rng.float(f32) * 2 - 1) * scale * (if (i % 64 == 7) @as(f32, 20) else 1));
    }

    /// Norm weights as Gemma 4's: signed, mostly in [-2, 6), one in 97 thirty times larger (12B: -143 to 193).
    fn weights(rng: std.Random, v: []u16) void {
        for (v, 0..) |*x, i| x.* = bfr.bits((rng.float(f32) * 8 - 2) * (if (i % 97 == 3) @as(f32, 30) else 1));
    }

    /// Gate 4's rows: integer squares summing to about 2^20, exact in fp32 in any order.
    fn integers(rng: std.Random, v: []u16) void {
        for (v) |*x| x.* = bfr.bits(@as(f32, @floatFromInt(rng.uintLessThan(u32, 65))) - 32);
    }

    pub fn prepare(self: *Tail, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const width = shape.k;
        if (width % 64 != 0 or width > 8192 or shape.rows == 0) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * width;
        const x = try a.alloc(u16, n);
        const r = try a.alloc(u16, n);
        const wb = try a.alloc(u16, width);
        const w = try a.alloc(u16, width);
        bfs(rng, x, 1);
        bfs(rng, r, 4);
        weights(rng, wb);
        weights(rng, w);
        var sc = bfr.bits(0.004 + 0.9 * rng.float(f32)); // Gemma 4 12B's layer scalars run 0.0045 to 0.89
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, bool, usize, f32 }{ .{ 0, true, width / 3, std.math.nan(f32) }, .{ 1, false, width / 2, std.math.inf(f32) }, .{ 2, true, width - 1, -std.math.inf(f32) }, .{ 3, false, 0, std.math.inf(f32) }, .{ 3, false, width - 1, -std.math.inf(f32) } };
            for (planted) |p| if (p[0] < shape.rows) {
                (if (p[1]) x else r)[p[0] * width + p[2]] = bfr.bits(p[3]);
            };
        }
        if (fill == .stress) { // only the checked output's sum of squares counts, so make it the exact one
            sc = bfr.bits(1);
            integers(rng, x);
            if (hView(self.which)) @memset(r, 0) else {
                @memset(wb, 0); // the branch adds zeros: h is the residual's integers
                integers(rng, r);
            }
        }
        const br = branch(self.which);
        self.k = .{ .rows = shape.rows, .width = width, .round = round(self.which), .x = x, .w = w, .r = if (residual(self.which)) r else null, .wb = if (br) wb else null, .sc = if (br) sc else null };
        for ([_][]const u8{ std.mem.sliceAsBytes(x), std.mem.sliceAsBytes(r), std.mem.sliceAsBytes(wb), std.mem.asBytes(&sc), std.mem.sliceAsBytes(w) }, &self.dev) |bytes, *d| d.* = try self.rig.input(bytes);
        if (hView(self.which)) {
            self.out[0] = try self.rig.output(n * 2);
            self.out[1] = try self.rig.output(n * 2);
        } else {
            self.out[1] = try self.rig.output(n * 2);
            self.out[0] = try self.rig.output(n * 2);
        }
        self.out[2] = try self.rig.output(n / 64 * 4);
        if (moe(self.which)) { // r is the one slot: bf16(fma(1, r, 0)) is r's bytes
            const ones = try a.alloc(f32, shape.rows);
            @memset(ones, 1);
            self.ones = try self.rig.input(std.mem.sliceAsBytes(ones));
        }
        return true;
    }

    pub fn run(self: *Tail) !void {
        const s = self.rig.s;
        const k = self.k;
        if (self.ctl) |f| {
            var args: cuda.Args = .{};
            for (self.dev[0..4]) |d| args.add(d);
            args.add(self.dev[4]);
            for (self.out) |d| args.add(d);
            args.add(@as(c_int, @intCast(k.width)));
            args.add(@as(f32, 1e-6));
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(k.rows), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
        } else if (!residual(self.which)) {
            const p = partsOf(k.rows); // contract row i is part i % p of row i / p, so the rows stay contiguous
            try self.norms.rmsParts(s, self.dev[0], p * k.width, k.width, self.dev[4], 0, self.out[1], p * k.width, k.width, self.out[2], k.rows / p, p, k.width, 1e-6, k.round);
        } else if (moe(self.which)) {
            try self.norms.moeAddRmsH(s, self.dev[0], self.dev[1], .bf16, self.ones, 1, self.dev[4], self.out[0], self.out[1], self.out[2], k.rows, k.width, 1e-6);
        } else if (!branch(self.which)) {
            try self.norms.addRmsH(s, self.dev[0], self.dev[1], self.dev[4], .bf16, self.out[0], self.out[1], self.out[2], k.rows, k.width, 1e-6);
        } else try self.norms.addRmsTail(s, self.dev[0], self.dev[1], self.dev[4], self.out[0], self.out[1], self.out[2], k.rows, k.width, 1e-6, .{ .round = k.round, .branch = self.dev[2], .scalar = self.dev[3] });
        try s.synchronize();
    }

    pub fn count(self: *Tail) usize {
        return self.k.rows * self.k.width;
    }

    fn view(self: *Tail, a: std.mem.Allocator, out: []f64, sums: anytype) !void {
        const width = self.k.width;
        const h = try a.alloc(f64, 4 * width);
        defer a.free(h);
        for (0..self.k.rows) |row| {
            const at = row * width;
            const t = try tail_.tailOf(a, self.k, row, 0);
            ref.tail(t.t, t.x, t.w, sums, h[0..width], h[width .. 2 * width], h[2 * width .. 3 * width], h[3 * width ..]);
            @memcpy(out[at..][0..width], if (hView(self.which)) h[0..width] else h[2 * width .. 3 * width]);
        }
    }

    pub fn reference(self: *Tail, want: []f64, mag: []f64) !void {
        var arena: std.heap.ArenaAllocator = .init(self.rig.gpu.gpa);
        defer arena.deinit();
        try self.view(arena.allocator(), want, ref.Exact{});
        for (mag, want) |*m, v| m.* = @abs(v) / 2; // v ~ ss^-1/2
    }

    pub fn simulate(self: *Tail, acc: c.Accum, order: g.Order, out: []f64) !void {
        var arena: std.heap.ArenaAllocator = .init(self.rig.gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        try self.view(a, out, Sim{ .acc = acc, .order = order, .buf = try a.alloc(f64, self.k.width) });
    }

    /// Every output in float64's range over the fp32 inverse's spread (a dropped point leaves it), sums as fp32's tree.
    pub fn check(self: *Tail, _: g.Gate) !?[]const u8 {
        const a = self.arena.?.allocator();
        const n = self.count();
        const h = try a.alloc(u16, n);
        const y = try a.alloc(u16, n);
        const xs = try a.alloc(f32, n / 64);
        @memcpy(std.mem.sliceAsBytes(h), try self.rig.bytes(self.out[0], n * 2));
        @memcpy(std.mem.sliceAsBytes(y), try self.rig.bytes(self.out[1], n * 2));
        @memcpy(std.mem.sliceAsBytes(xs), try self.rig.bytes(self.out[2], n / 64 * 4));
        var stats: tail_.Stats = .{};
        return tail_.verify(self.rig.gpu.gpa, self.k, h, y, xs, &stats, &self.msg);
    }
};

test {
    _ = tail_;
}

test "gate 4's rows: the checked output's squares are exact in fp32, fp16 overflows and bf16 misses" {
    var terms: [5376]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(0x6b63);
    for (&terms) |*t| {
        const v: f64 = @floatFromInt(@as(i32, @intCast(prng.random().uintLessThan(u32, 65))) - 32);
        t.* = v * v;
    }
    var exact: f64 = 0;
    for (terms) |t| exact += t;
    try std.testing.expect(exact < 0x1p24 and exact > 0x1p20);
    for (std.enums.values(rows.Order)) |o| {
        try std.testing.expectEqual(exact, rows.sumIn(.f32, o, &terms));
        try std.testing.expect(std.math.isInf(rows.sumIn(.f16, o, &terms)));
        try std.testing.expect(rows.sumIn(.bf16, o, &terms) != exact);
    }
}
