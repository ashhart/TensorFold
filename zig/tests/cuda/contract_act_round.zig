//! act_rows.cu's MLX activations under the contract (GeGLU, SwiGLU's bf16 steps), and controls each dropping a point.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const act_ = @import("act_round_check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;
const aref = core.act_ref;

/// A registration: a served kernel (keep null), or a control keeping all rounding points but one (act_round.cuh).
pub const Entry = struct { name: []const u8, geglu: bool, keep: ?i32 = null };

const geglu_all = 127;
const swiglu_all = 31;

pub const entries = [_]Entry{
    .{ .name = "cuda act tf_geglu_rows (MLX's bf16 steps)", .geglu = true },
    .{ .name = "cuda act tf_swiglu_rows_round (MLX's bf16 steps)", .geglu = false },
    .{ .name = "control: GeGLU without x^3's rounding", .geglu = true, .keep = geglu_all & ~@as(i32, 1) },
    .{ .name = "control: GeGLU without c1 x^3's rounding", .geglu = true, .keep = geglu_all & ~@as(i32, 2) },
    .{ .name = "control: GeGLU without the inner sum's rounding", .geglu = true, .keep = geglu_all & ~@as(i32, 4) },
    .{ .name = "control: GeGLU without c0 inner's rounding", .geglu = true, .keep = geglu_all & ~@as(i32, 8) },
    .{ .name = "control: GeGLU without tanh's rounding", .geglu = true, .keep = geglu_all & ~@as(i32, 16) },
    .{ .name = "control: GeGLU without 1 + tanh's rounding", .geglu = true, .keep = geglu_all & ~@as(i32, 32) },
    .{ .name = "control: GeGLU without the gelu's rounding", .geglu = true, .keep = geglu_all & ~@as(i32, 64) },
    .{ .name = "control: SwiGLU steps without exp's rounding", .geglu = false, .keep = swiglu_all & ~@as(i32, 1) },
    .{ .name = "control: SwiGLU steps without 1 + e's rounding", .geglu = false, .keep = swiglu_all & ~@as(i32, 2) },
    .{ .name = "control: SwiGLU steps without the reciprocal's rounding", .geglu = false, .keep = swiglu_all & ~@as(i32, 4) },
    .{ .name = "control: SwiGLU steps without 1 - the reciprocal's rounding", .geglu = false, .keep = swiglu_all & ~@as(i32, 8) },
    .{ .name = "control: SwiGLU steps without silu's rounding", .geglu = false, .keep = swiglu_all & ~@as(i32, 16) },
};

fn depth(_: c.Shape) usize {
    return 0;
}

pub fn spec(e: Entry) c.Spec {
    return .{
        .name = e.name,
        .accum = .none, // no sums in an output; its 64-value sums are checked bit for bit against the fp32 tree
        .depth = depth,
        .mids = 16, // gate 3 is the kernel's own check: float64's rounding points, ranged over tanhf's or expf's ulps
        .out = .bf16,
        .tiles = .{ .{ .size = 16 }, .{ .size = 1024, .step = 64, .min = 64 }, .{} },
        // Gemma 4 12B's and 31B's MLP, the 31B DFlash drafter's and the 12B assistant's
        .served = &.{ .{ .rows = 17, .n = 15360, .k = 1 }, .{ .rows = 3, .n = 21504, .k = 1 }, .{ .rows = 16, .n = 10752, .k = 1 }, .{ .rows = 6, .n = 8192, .k = 1 } },
    };
}

pub const Act = struct {
    rig: rig_.Rig,
    entry: Entry,
    act: *const cuda.act_rows.ActRows,
    ctl: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    k: act_.Case = undefined,
    gu: u64 = 0,
    out: u64 = 0,
    xs: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Act) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Act);

    pub fn kernel(self: *Act) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.keep != null) .shapes else null };
    }

    pub fn prepare(self: *Act, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const n = shape.n;
        if (n % 64 != 0 or n == 0 or shape.rows == 0) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const gu = try a.alloc(u16, shape.rows * 2 * n); // [gate | up] rows, as a fused gate_up projection writes them
        for (gu, 0..) |*v, i| {
            const up = i % (2 * n) >= n;
            const big: f32 = if (i % 64 == 7) 20 else 1;
            v.* = bfr.bits(if (up) rng.float(f32) * 4 - 2 else (rng.float(f32) * 8 - 4) * big);
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, usize, f32 }{ .{ 0, n / 3, std.math.nan(f32) }, .{ 1, n / 2, std.math.inf(f32) }, .{ 2, n - 1, -std.math.inf(f32) }, .{ 3, n + 5, std.math.inf(f32) }, .{ 3, 0, -std.math.inf(f32) } };
            for (planted) |p| if (p[0] < shape.rows) {
                gu[p[0] * 2 * n + p[1]] = bfr.bits(p[2]);
            };
        }
        self.k = .{ .geglu = self.entry.geglu, .rows = shape.rows, .n = n, .gate = gu, .g_ld = 2 * n, .up = gu[n..], .u_ld = 2 * n };
        self.gu = try self.rig.input(std.mem.sliceAsBytes(gu));
        self.out = try self.rig.output(shape.rows * n * 2);
        self.xs = try self.rig.output(shape.rows * n / 64 * 4);
        return true;
    }

    pub fn run(self: *Act) !void {
        const k = self.k;
        const s = self.rig.s;
        const up = self.gu + 2 * k.n;
        if (self.ctl) |f| {
            var args: cuda.Args = .{};
            args.add(self.gu);
            args.add(@as(c_longlong, @intCast(k.g_ld)));
            args.add(up);
            args.add(@as(c_longlong, @intCast(k.u_ld)));
            args.add(self.out);
            args.add(self.xs);
            args.add(@as(c_int, @intCast(k.n)));
            args.add(@as(c_int, @intFromBool(k.geglu)));
            args.add(@as(c_int, self.entry.keep.?));
            const t: u32 = if (k.rows <= cuda.launch.few_rows) 128 else 256;
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(k.rows), .y = @intCast((k.n / 8 + t - 1) / t), .z = 1 }, .block = .{ .x = t } }, s, &args);
        } else if (k.geglu) try self.act.geglu(s, self.gu, k.g_ld, up, k.u_ld, self.out, self.xs, k.rows, k.n) else try self.act.swigluRound(s, self.gu, k.g_ld, up, k.u_ld, self.out, self.xs, k.rows, k.n);
        try s.synchronize();
    }

    pub fn count(self: *Act) usize {
        return self.k.rows * self.k.n;
    }

    pub fn reference(self: *Act, out: []f64, mag: []f64) !void {
        for (0..self.k.rows) |row| for (0..self.k.n) |i| {
            out[row * self.k.n + i] = act_.want(self.k, row, i).point;
        };
        @memset(mag, 0);
    }

    /// Every output within float64's ends at the rounding points (a dropped point leaves them), sums as fp32's tree.
    pub fn check(self: *Act, _: g.Gate) !?[]const u8 {
        const a = self.arena.?.allocator();
        const n = self.count();
        const out = try a.alloc(u16, n);
        const xs = try a.alloc(f32, n / 64);
        @memcpy(std.mem.sliceAsBytes(out), try self.rig.bytes(self.out, n * 2));
        @memcpy(std.mem.sliceAsBytes(xs), try self.rig.bytes(self.xs, n / 64 * 4));
        var stats: act_.Stats = .{};
        return act_.verify(self.k, out, xs, &stats, &self.msg);
    }
};

test {
    _ = act_;
}

test "every control keeps all but one rounding point, each point once" {
    var seen = [2]i32{ 0, 0 };
    for (entries) |e| if (e.keep) |k| {
        const all: i32 = if (e.geglu) geglu_all else swiglu_all;
        const dropped = all & ~k;
        try std.testing.expectEqual(@as(u32, 1), @popCount(dropped));
        try std.testing.expect(seen[@intFromBool(e.geglu)] & dropped == 0);
        seen[@intFromBool(e.geglu)] |= dropped;
    };
    try std.testing.expectEqual([2]i32{ swiglu_all, geglu_all }, seen);
}
