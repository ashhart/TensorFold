//! norms.cu's nGPT rows under the contract (tf_l2_rows with and without its add, tf_slerp_rows), and their controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const ref = core.l2_ref;
const bfr = core.bf16_round;

/// The registrations, and the controls dropping one rounding point each.
pub const Which = enum { l2, l2_add, slerp, drop_sum, drop_a, drop_b };

/// The norm's sum chain: 32 fma a thread over 16-byte loads, then 256 threads' tree (norms_tail_check's depth).
const chain = 42;

fn depth(_: c.Shape) usize {
    return chain;
}

/// An fp32 inverse norm's relative spread from float64's: the sum's gamma halved by the root, the fp32 steps.
pub fn spread() f64 {
    return c.gamma(chain, 0x1p-24) / 2 + 8 * 0x1p-24;
}

const why = "an output follows its sums of squares as ss^-1/2 through bf16 rounding points, so a bf16 sum's scale error cancels in the row's norm, and an fp16 sum overflows to a zero output";

/// Dory's hidden rows: a decode window and a short verify (2560 wide).
const served = [_]c.Shape{ .{ .rows = 16, .n = 2560, .k = 2560 }, .{ .rows = 3, .n = 2560, .k = 2560 } };

pub fn spec(which: Which) c.Spec {
    return .{
        .name = switch (which) {
            .l2 => "cuda nGPT rows tf_l2_rows (a loop's exit: bf16(l2(h)))",
            .l2_add => "cuda nGPT rows tf_l2_rows with its add (a loop's entry: bf16(l2(bf16(state + input))))",
            .slerp => "cuda nGPT rows tf_slerp_rows (the residual: bf16(l2(a + lr (b - a))), a and b bf16(l2(.)))",
            .drop_sum => "control: tf_l2_rows without the sum's rounding before the norm",
            .drop_a => "control: tf_slerp_rows without the residual's normalized rounding",
            .drop_b => "control: tf_slerp_rows without the branch's normalized rounding",
        },
        .accum = .f32,
        .depth = depth,
        .ops = 12,
        .mids = if (isSlerp(which)) 4 else 2, // a bf16 point (a, b) decided the other way by an fp32 inverse: two half-ulps
        .out = .bf16,
        .tiles = .{ .{}, .{}, .{ .size = 2048, .step = 64, .min = 64 } },
        .served = &served,
        .sums = .{ .f16 = 40, .bf16 = 0, .why = why },
    };
}

pub fn control(which: Which) ?g.Gate {
    return switch (which) {
        .l2, .l2_add, .slerp => null,
        else => .shapes,
    };
}

/// The kernel a registration launches: the served ones through cuda/norms.zig, the controls by symbol.
pub fn symbol(which: Which) [:0]const u8 {
    return switch (which) {
        .drop_sum => "tf_ctl_l2_drop_sum",
        .drop_a => "tf_ctl_slerp_drop_a",
        .drop_b => "tf_ctl_slerp_drop_b",
        else => unreachable,
    };
}

fn isSlerp(which: Which) bool {
    return which == .slerp or which == .drop_a or which == .drop_b;
}

/// fp32 alpha_init / std at Dory's 0.05 and 0.0177.
const alpha_scale: f32 = @floatCast(0.05 / 0.0177);

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

/// got in [lo, hi]: its bits where the ends are one value, else between the ends (-0 below +0); NaN for NaN.
fn inRange(got: u16, r: ref.Iv) bool {
    const v: f64 = bfr.value(got);
    if (std.math.isNan(r[0]) or std.math.isNan(r[1])) return std.math.isNan(v);
    if (@as(u64, @bitCast(r[0])) == @as(u64, @bitCast(r[1]))) return got == bfr.bits(@floatCast(r[0]));
    return atMost(r[0], v) and atMost(v, r[1]);
}

fn atMost(a: f64, b: f64) bool {
    return a < b or (a == b and (std.math.signbit(a) or !std.math.signbit(b)));
}

pub const L2 = struct {
    rig: rig_.Rig,
    which: Which,
    norms: *const cuda.norms.Norms,
    ctl: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    n: usize = 0,
    width: usize = 0,
    x: []f64 = &.{}, // x or res
    r: []f64 = &.{}, // r or out
    alpha: []f64 = &.{},
    dev: [3]u64 = undefined,
    out: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *L2) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(L2);

    pub fn kernel(self: *L2) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control(self.which) };
    }

    fn bfs(rng: std.Random, v: []f64, scale: f32) void {
        for (v, 0..) |*x, i| x.* = bfr.value(bfr.bits((rng.float(f32) * 2 - 1) * scale * (if (i % 64 == 7) @as(f32, 20) else 1)));
    }

    /// Gate 4's rows: integers whose squares sum to about 2^20, exact in fp32 in any order.
    fn integers(rng: std.Random, v: []f64) void {
        for (v) |*x| x.* = @as(f64, @floatFromInt(rng.uintLessThan(u32, 65))) - 32;
    }

    fn upload(self: *L2, v: []const f64) !u64 {
        const a = self.arena.?.allocator();
        const b = try a.alloc(u16, v.len);
        for (b, v) |*h, x| h.* = bfr.bits(@floatCast(x));
        return self.rig.input(std.mem.sliceAsBytes(b));
    }

    pub fn prepare(self: *L2, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const width = shape.k;
        if (width % 64 != 0 or width > 8192 or shape.rows == 0) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.width = width;
        self.n = shape.rows * width;
        self.x = try a.alloc(f64, self.n);
        self.r = try a.alloc(f64, self.n);
        self.alpha = try a.alloc(f64, width);
        bfs(rng, self.x, 1);
        bfs(rng, self.r, 4);
        bfs(rng, self.alpha, 0.4); // |alpha scale| to about 1.1, lr past 1 on a few
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, bool, usize, f64 }{ .{ 0, true, width / 3, std.math.nan(f64) }, .{ 1, false, width / 2, std.math.inf(f64) }, .{ 2, true, width - 1, -std.math.inf(f64) }, .{ 3, false, 0, std.math.inf(f64) }, .{ 3, false, width - 1, -std.math.inf(f64) } };
            for (planted) |p| if (p[0] < shape.rows) {
                (if (p[1]) self.x else self.r)[p[0] * width + p[2]] = p[3];
            };
        }
        if (fill == .stress) {
            integers(rng, self.x);
            integers(rng, self.r);
        }
        const take_r = self.which != .l2;
        self.dev = .{ try self.upload(self.x), if (take_r) try self.upload(self.r) else 0, if (isSlerp(self.which)) try self.upload(self.alpha) else 0 };
        self.out = try self.rig.output(self.n * 2);
        return true;
    }

    pub fn run(self: *L2) !void {
        const s = self.rig.s;
        const rws = self.n / self.width;
        if (self.ctl) |f| {
            var args: cuda.Args = .{};
            args.add(self.dev[0]);
            args.add(self.dev[1]);
            if (isSlerp(self.which)) {
                args.add(self.dev[2]);
                args.add(alpha_scale);
            }
            args.add(self.out);
            args.add(@as(c_int, @intCast(self.width)));
            args.add(@as(f32, 1e-6));
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rws), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
        } else if (isSlerp(self.which)) {
            try self.norms.slerp(s, self.dev[0], self.dev[1], self.dev[2], alpha_scale, self.out, rws, self.width, 1e-6);
        } else try self.norms.l2(s, self.dev[0], if (self.which == .l2) null else self.dev[1], self.out, rws, self.width, 1e-6);
        try s.synchronize();
    }

    pub fn count(self: *L2) usize {
        return self.n;
    }

    /// Each row's outputs as ranges under `st`, sums of squares by `sums`.
    fn rowsOf(self: *L2, a: std.mem.Allocator, sums: anytype, st: ref.Stray, out: []ref.Iv) !void {
        const w = self.width;
        const scratch = try a.alloc(ref.Iv, w);
        const small = try a.alloc(f64, w);
        const big = try a.alloc(f64, w);
        for (0..self.n / w) |row| {
            const at = row * w;
            const x = self.x[at..][0..w];
            const r = self.r[at..][0..w];
            const o = out[at..][0..w];
            switch (self.which) {
                .l2 => ref.l2Rows(x, null, true, 1e-6, sums, st, o, small, big),
                .l2_add, .drop_sum => ref.l2Rows(x, r, true, 1e-6, sums, st, o, small, big),
                else => ref.slerp(x, r, self.alpha, alpha_scale, .{}, 1e-6, sums, st, o, scratch, small, big),
            }
        }
    }

    fn points(self: *L2, sums: anytype, out: []f64) !void {
        var arena: std.heap.ArenaAllocator = .init(self.rig.gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const iv = try a.alloc(ref.Iv, self.n);
        try self.rowsOf(a, sums, .{}, iv);
        for (out, iv) |*o, v| o.* = v[0];
    }

    pub fn reference(self: *L2, want: []f64, mag: []f64) !void {
        try self.points(ref.Exact{}, want);
        for (mag, want) |*m, v| m.* = @abs(v) / 2; // v ~ ss^-1/2
    }

    pub fn simulate(self: *L2, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try self.rig.gpu.gpa.alloc(f64, self.width);
        defer self.rig.gpu.gpa.free(buf);
        try self.points(Sim{ .acc = acc, .order = order, .buf = buf }, out);
    }

    /// Every output in float64's range over the fp32 run's spread (a dropped point leaves it).
    pub fn check(self: *L2, _: g.Gate) !?[]const u8 {
        const a = self.arena.?.allocator();
        const got = try a.alloc(u16, self.n);
        @memcpy(std.mem.sliceAsBytes(got), try self.rig.bytes(self.out, self.n * 2));
        const iv = try a.alloc(ref.Iv, self.n);
        try self.rowsOf(a, ref.Exact{}, .{ .spread = spread(), .u = 0x1p-24 }, iv);
        for (got, iv, 0..) |h, r, i| if (!inRange(h, r))
            return try std.fmt.bufPrint(&self.msg, "row {d} out[{d}] = {e} outside float64's [{e}, {e}]", .{ i / self.width, i % self.width, bfr.value(h), r[0], r[1] });
        return null;
    }
};

test "gate 4's rows: integer squares are exact in fp32, fp16 overflows and bf16 misses" {
    var terms: [2560]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(0x6c32);
    for (&terms) |*t| {
        const v: f64 = @floatFromInt(@as(i32, @intCast(prng.random().uintLessThan(u32, 65))) - 32);
        t.* = v * v;
    }
    var exact: f64 = 0;
    for (terms) |t| exact += t;
    try std.testing.expect(exact < 0x1p24 and exact > 0x1p19);
    for (std.enums.values(rows.Order)) |o| {
        try std.testing.expectEqual(exact, rows.sumIn(.f32, o, &terms));
        try std.testing.expect(std.math.isInf(rows.sumIn(.f16, o, &terms)));
        try std.testing.expect(rows.sumIn(.bf16, o, &terms) != exact);
    }
}

test "the range check: a dropped bf16 point leaves the fp32 run's range" {
    try std.testing.expect(spread() < 0x1p-18);
    const r = ref.Iv{ 0.5, 0.5 };
    try std.testing.expect(inRange(bfr.bits(0.5), r) and !inRange(bfr.bits(0.50390625), r));
    try std.testing.expect(inRange(0x7fc0, @splat(std.math.nan(f64))) and !inRange(0, @splat(std.math.nan(f64))));
}
