//! act_rows' nGPT activations under the contract (tf_gate_l2_rows, tf_swiglu_l2_rows), and their controls.
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
pub const Which = enum { gate, swiglu, drop_gated, drop_gs, drop_us, drop_prod };

const chain = 42; // 32 fma a thread over 16-byte loads, then 256 threads' tree

fn depth(_: c.Shape) usize {
    return chain;
}

pub fn spread() f64 {
    return c.gamma(chain, 0x1p-24) / 2 + 8 * 0x1p-24;
}

/// expf within 2 ulps of its value either way.
const exp_stray = 0x1p-22;

fn isGate(which: Which) bool {
    return which == .gate or which == .drop_gated;
}

const why = "an output follows its row's sum of squares as ss^-1/2 through bf16 rounding points, so a bf16 sum's scale cancels in the norm, and an fp16 sum overflows to a zero output";

/// Dory's attention rows (16 heads of 256) and MLP rows (7,168), a decode window and a short verify.
const gate_served = [_]c.Shape{ .{ .rows = 16, .n = 4096, .k = 4096 }, .{ .rows = 3, .n = 4096, .k = 4096 } };
const mlp_served = [_]c.Shape{ .{ .rows = 16, .n = 7168, .k = 7168 }, .{ .rows = 3, .n = 7168, .k = 7168 } };

pub fn spec(which: Which) c.Spec {
    return .{
        .name = switch (which) {
            .gate => "cuda act_rows tf_gate_l2_rows (Dory's output gate: bf16(l2(bf16(o sigmoid(g sg)))) over all heads)",
            .swiglu => "cuda act_rows tf_swiglu_l2_rows (Dory's MLP: bf16(l2(bf16(silu(bf16(g sug)) bf16(u suu)))))",
            .drop_gated => "control: tf_gate_l2_rows without the gated row's rounding",
            .drop_gs => "control: tf_swiglu_l2_rows without the scaled gate's rounding",
            .drop_us => "control: tf_swiglu_l2_rows without the scaled up's rounding",
            .drop_prod => "control: tf_swiglu_l2_rows without the product's rounding",
        },
        .accum = .f32,
        .depth = depth,
        .ops = 16,
        .mids = if (isGate(which)) 2 else 4, // a bf16 point decided the other way by fp32 steps: two half-ulps a point
        .out = .bf16,
        .tiles = .{ .{}, .{}, .{ .size = 2048, .step = 64, .min = 64 } },
        .served = if (isGate(which)) &gate_served else &mlp_served,
        // SwiGLU's squares are no integers: its fp32 sums carry their own gamma, the shapes gate's bound
        .sums = .{ .f16 = 40, .bf16 = 0, .why = why, .exact = isGate(which) },
    };
}

pub fn control(which: Which) ?g.Gate {
    return switch (which) {
        .gate, .swiglu => null,
        else => .shapes,
    };
}

pub fn symbol(which: Which) [:0]const u8 {
    return switch (which) {
        .drop_gated => "tf_ctl_gate_l2_drop_gated",
        .drop_gs => "tf_ctl_swiglu_l2_drop_gs",
        .drop_us => "tf_ctl_swiglu_l2_drop_us",
        .drop_prod => "tf_ctl_swiglu_l2_drop_prod",
        else => unreachable,
    };
}

const Sim = struct {
    acc: c.Accum,
    order: rows.Order,
    buf: []f64,

    pub fn squares(s: Sim, v: []const f64) f64 {
        for (s.buf[0..v.len], v) |*t, x| t.* = x * x;
        return rows.sumIn(s.acc, s.order, s.buf[0..v.len]);
    }
};

fn inRange(got: u16, r: ref.Iv) bool {
    const v: f64 = bfr.value(got);
    if (std.math.isNan(r[0]) or std.math.isNan(r[1])) return std.math.isNan(v);
    if (@as(u64, @bitCast(r[0])) == @as(u64, @bitCast(r[1]))) return got == bfr.bits(@floatCast(r[0]));
    return (r[0] < v or r[0] == v) and (v < r[1] or v == r[1]);
}

pub const Act = struct {
    rig: rig_.Rig,
    which: Which,
    k: *const cuda.act_rows.ActRows,
    ctl: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    n: usize = 0,
    width: usize = 0,
    a: []f64 = &.{}, // o (gate) or g (swiglu)
    b: []f64 = &.{}, // g (gate) or u (swiglu)
    sa: []f64 = &.{}, // sg (gate) or sug
    sb: []f64 = &.{}, // suu
    dev: [5]u64 = @splat(0),
    out: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Act) void {
        if (self.arena) |*x| x.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Act);

    pub fn kernel(self: *Act) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control(self.which) };
    }

    fn bfs(rng: std.Random, v: []f64, scale: f32) void {
        for (v, 0..) |*x, i| x.* = bfr.value(bfr.bits((rng.float(f32) * 2 - 1) * scale * (if (i % 64 == 7) @as(f32, 8) else 1)));
    }

    fn integers(rng: std.Random, v: []f64) void {
        for (v) |*x| x.* = @as(f64, @floatFromInt(rng.uintLessThan(u32, 65))) - 32;
    }

    fn bf16In(self: *Act, v: []const f64) !u64 {
        const b = try self.arena.?.allocator().alloc(u16, v.len);
        for (b, v) |*h, x| h.* = bfr.bits(@floatCast(x));
        return self.rig.input(std.mem.sliceAsBytes(b));
    }

    fn f32In(self: *Act, v: []const f64) !u64 {
        const b = try self.arena.?.allocator().alloc(f32, v.len);
        for (b, v) |*h, x| h.* = @floatCast(x);
        return self.rig.input(std.mem.sliceAsBytes(b));
    }

    pub fn prepare(self: *Act, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const width = shape.k;
        if (width % 64 != 0 or width > 8192 or shape.rows == 0) return false;
        self.rig.reset();
        if (self.arena) |*x| x.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const al = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.width = width;
        self.n = shape.rows * width;
        self.a = try al.alloc(f64, self.n);
        self.b = try al.alloc(f64, self.n);
        self.sa = try al.alloc(f64, width);
        self.sb = try al.alloc(f64, width);
        bfs(rng, self.a, 2);
        bfs(rng, self.b, 2);
        // the scale vectors as fp32: Dory's (s_gate f)^2 and suv sqrt(hidden) ranges
        for (self.sa, self.sb) |*x, *y| {
            x.* = @as(f32, if (isGate(self.which)) 1.5 + 30 * rng.float(f32) else 0.3 + 3 * rng.float(f32));
            y.* = @as(f32, 0.3 + 3 * rng.float(f32));
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, bool, usize, f64 }{ .{ 0, true, width / 3, std.math.nan(f64) }, .{ 1, false, width / 2, std.math.inf(f64) }, .{ 2, true, width - 1, -std.math.inf(f64) }, .{ 3, false, 0, std.math.inf(f64) }, .{ 3, false, width - 1, -std.math.inf(f64) } };
            for (planted) |p| if (p[0] < shape.rows) {
                (if (p[1]) self.a else self.b)[p[0] * width + p[2]] = p[3];
            };
        }
        if (fill == .stress) {
            integers(rng, self.a);
            if (isGate(self.which)) @memset(self.b, 0) else integers(rng, self.b); // sigmoid(0) = 1/2: o / 2 exactly
            @memset(self.sa, 1);
            @memset(self.sb, 1);
        }
        self.dev = .{ try self.bf16In(self.a), try self.bf16In(self.b), try self.f32In(self.sa), try self.f32In(self.sb), 0 };
        self.out = try self.rig.output(self.n * 2);
        return true;
    }

    pub fn run(self: *Act) !void {
        const s = self.rig.s;
        const rws = self.n / self.width;
        const w = self.width;
        if (self.ctl) |f| {
            var args: cuda.Args = .{};
            args.add(self.dev[0]);
            if (!isGate(self.which)) args.add(@as(i64, @intCast(w)));
            args.add(self.dev[1]);
            args.add(@as(i64, @intCast(w)));
            args.add(self.dev[2]);
            if (!isGate(self.which)) args.add(self.dev[3]);
            args.add(self.out);
            args.add(@as(c_int, @intCast(w)));
            args.add(@as(f32, 1e-6));
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rws), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
        } else if (isGate(self.which)) {
            try self.k.gateL2(s, self.dev[0], self.dev[1], w, self.dev[2], self.out, rws, w, 1e-6);
        } else try self.k.swigluL2(s, self.dev[0], w, self.dev[1], w, self.dev[2], self.dev[3], self.out, rws, w, 1e-6);
        try s.synchronize();
    }

    pub fn count(self: *Act) usize {
        return self.n;
    }

    fn ranges(self: *Act, al: std.mem.Allocator, sums: anytype, st: ref.Stray, e: f64, out: []ref.Iv) !void {
        const w = self.width;
        const small = try al.alloc(f64, w);
        const big = try al.alloc(f64, w);
        for (0..self.n / w) |row| {
            const at = row * w;
            const o = out[at..][0..w];
            switch (self.which) {
                .gate => ref.gateL2(self.a[at..][0..w], self.b[at..][0..w], self.sa, .{}, 1e-6, sums, st, e, o, small, big),
                else => ref.swigluL2(self.a[at..][0..w], self.b[at..][0..w], self.sa, self.sb, .{}, 1e-6, sums, st, e, o, small, big),
            }
        }
    }

    fn points(self: *Act, sums: anytype, out: []f64) !void {
        var arena: std.heap.ArenaAllocator = .init(self.rig.gpu.gpa);
        defer arena.deinit();
        const iv = try arena.allocator().alloc(ref.Iv, self.n);
        try self.ranges(arena.allocator(), sums, .{}, 0, iv);
        for (out, iv) |*o, v| o.* = v[0];
    }

    pub fn reference(self: *Act, want: []f64, mag: []f64) !void {
        try self.points(ref.Exact{}, want);
        for (mag, want) |*m, v| m.* = @abs(v) / 2;
    }

    pub fn simulate(self: *Act, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try self.rig.gpu.gpa.alloc(f64, self.width);
        defer self.rig.gpu.gpa.free(buf);
        try self.points(Sim{ .acc = acc, .order = order, .buf = buf }, out);
    }

    pub fn check(self: *Act, _: g.Gate) !?[]const u8 {
        const al = self.arena.?.allocator();
        const got = try al.alloc(u16, self.n);
        @memcpy(std.mem.sliceAsBytes(got), try self.rig.bytes(self.out, self.n * 2));
        const iv = try al.alloc(ref.Iv, self.n);
        try self.ranges(al, ref.Exact{}, .{ .spread = spread(), .u = 0x1p-24 }, exp_stray, iv);
        for (got, iv, 0..) |h, r, i| if (!inRange(h, r))
            return try std.fmt.bufPrint(&self.msg, "row {d} out[{d}] = {e} outside float64's [{e}, {e}]", .{ i / self.width, i % self.width, bfr.value(h), r[0], r[1] });
        return null;
    }
};

test "gate 4's gate rows: halves of integers square exactly in fp32, and fp16 overflows" {
    var terms: [4096]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(0x6761);
    for (&terms) |*t| {
        const v: f64 = (@as(f64, @floatFromInt(prng.random().uintLessThan(u32, 65))) - 32) / 2;
        t.* = v * v;
    }
    for (std.enums.values(rows.Order)) |o| {
        var exact: f64 = 0;
        for (terms) |t| exact += t;
        try std.testing.expectEqual(exact, rows.sumIn(.f32, o, &terms));
        try std.testing.expect(std.math.isInf(rows.sumIn(.f16, o, &terms)));
    }
}
