//! gemma4_ple.cu's tf_gemma4_ple_rows under the contract, and controls each dropping a step.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
pub const g4 = @import("gemma4_cuda");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const bfr = core.bf16_round;

/// The served kernel, and the controls: the norm skipped, the table rows unscaled (the projection's scale cancels).
pub const Which = enum { served, drop_norm, drop_token };

const width = 256;
const chain = 10; // 4 values a thread, 5 butterfly steps, the two warps' sum

fn depth(_: c.Shape) usize {
    return chain;
}

const why = "an output follows its part's sum of squares as ss^-1/2 through bf16 rounding points, so a bf16 sum's scale cancels in the norm, and an fp16 sum overflows to a zero output";

/// The E4B's 42 parts of 256: a prompt chunk, a decode round, a short verify and one row.
const served = [_]c.Shape{ .{ .rows = 512, .n = 42 * width, .k = width }, .{ .rows = 128, .n = 42 * width, .k = width }, .{ .rows = 17, .n = 42 * width, .k = width }, .{ .rows = 1, .n = 42 * width, .k = width } };

pub fn spec(which: Which) c.Spec {
    return .{
        .name = switch (which) {
            .served => "cuda gemma4_ple tf_gemma4_ple_rows (Gemma 4 E4B per-layer inputs: bf16(bf16(rms(bf16(p s)) w + bf16(t 16)) mix))",
            .drop_norm => "control: tf_gemma4_ple_rows without the 256-wide norm",
            .drop_token => "control: tf_gemma4_ple_rows without the table rows' token scale",
        },
        .accum = .f32,
        .depth = depth,
        .ops = 8,
        .mids = 6, // bf16 points the fp32 steps may decide the other way: x, x inv, w n, t token, the sum
        .out = .bf16,
        .tiles = .{ .{}, .{ .size = width, .step = width, .min = width }, .{} },
        .served = &served,
        .sums = .{ .f16 = 20, .bf16 = 0, .why = why, .exact = true }, // a zeroed output misses bounds() by 2^7 / 5
    };
}

pub fn control(which: Which) ?g.Gate {
    return if (which == .served) null else .shapes;
}

pub fn drop(which: Which) c_int {
    return switch (which) {
        .served => 0,
        .drop_norm => 1, // tf_ple::DROP_NORM
        .drop_token => 2, // tf_ple::DROP_TOKEN
    };
}

pub const Ple = struct {
    rig: rig_.Rig,
    which: Which,
    k: *const g4.ple.PleRows,
    ctl: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    rows_n: usize = 0,
    parts: usize = 0,
    p: []f64 = &.{},
    t: []f64 = &.{},
    w: []f64 = &.{},
    c: g4.ple.Scalars = undefined,
    dev: [3]u64 = @splat(0),
    out: u64 = 0,
    msg: [160]u8 = undefined,
    terms: []f64 = &.{}, // each output's |y| and |t token| before the mix: what a decided point moves

    pub fn deinit(self: *Ple) void {
        if (self.arena) |*x| x.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Ple);

    pub fn kernel(self: *Ple) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control(self.which) };
    }

    fn bf16In(self: *Ple, v: []const f64) !u64 {
        const b = try self.arena.?.allocator().alloc(u16, v.len);
        for (b, v) |*h, x| h.* = bfr.bits(@floatCast(x));
        return self.rig.input(std.mem.sliceAsBytes(b));
    }

    pub fn prepare(self: *Ple, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != width or shape.n % width != 0 or shape.n == 0 or shape.rows == 0) return false;
        self.rig.reset();
        if (self.arena) |*x| x.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const al = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.rows_n = shape.rows;
        self.parts = shape.n / width;
        const n = shape.rows * shape.n;
        self.p = try al.alloc(f64, n);
        self.t = try al.alloc(f64, n);
        self.w = try al.alloc(f64, width);
        const bf = struct {
            fn of(v: f32) f64 {
                return bfr.value(bfr.bits(v));
            }
        }.of;
        // the E4B's scalars as MLX multiplies by them (cuda_forward.layerInputs)
        self.c = .{ .scale = @floatCast(bfr.bf16(1.0 / @sqrt(2560.0))), .token = 16, .mix = @floatCast(bfr.bf16(std.math.sqrt1_2)), .eps = 1e-6 };
        for (self.p, 0..) |*x, i| x.* = bf((rng.float(f32) * 2 - 1) * 4 * (if (i % 64 == 7) @as(f32, 30) else 1));
        for (self.t) |*x| x.* = bf((rng.float(f32) * 2 - 1) * 0.1);
        for (self.w) |*x| x.* = bf(0.5 + rng.float(f32));
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, bool, usize, f64 }{ .{ 0, true, width / 3, std.math.nan(f64) }, .{ 1, false, width / 2, std.math.inf(f64) }, .{ 2, true, shape.n - 1, -std.math.inf(f64) }, .{ 3, false, 0, std.math.inf(f64) }, .{ 3, false, shape.n - 1, -std.math.inf(f64) } };
            for (planted) |pl| if (pl[0] < shape.rows) {
                (if (pl[1]) self.p else self.t)[pl[0] * shape.n + pl[2]] = pl[3];
            };
        }
        if (fill == .stress) { // integer squares, exact in fp32 in any order; unit scalars, so the sums decide alone
            for (self.p) |*x| x.* = @as(f64, @floatFromInt(rng.uintLessThan(u32, 129))) - 64;
            @memset(self.t, 0);
            @memset(self.w, 1);
            self.c = .{ .scale = 1, .token = 1, .mix = 1, .eps = 1e-6 };
        }
        self.dev = .{ try self.bf16In(self.p), try self.bf16In(self.t), try self.bf16In(self.w) };
        self.out = try self.rig.output(n * 2);
        return true;
    }

    pub fn run(self: *Ple) !void {
        const s = self.rig.s;
        if (self.ctl) |f| {
            var args: cuda.Args = .{};
            for ([_]u64{ self.dev[0], self.dev[1], self.dev[2], self.out }) |v| args.add(v);
            args.add(@as(c_int, @intCast(self.parts)));
            for ([_]f32{ self.c.scale, self.c.token, self.c.mix, self.c.eps }) |v| args.add(v);
            args.add(drop(self.which));
            const grid: cuda.launch.Dim3 = .{ .x = @intCast(self.rows_n), .y = @intCast(self.parts), .z = 1 };
            try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 64 } }, s, &args);
        } else try self.k.run(s, self.dev[0], self.dev[1], self.dev[2], self.out, self.rows_n, self.parts, self.c);
        try s.synchronize();
    }

    pub fn count(self: *Ple) usize {
        return self.rows_n * self.parts * width;
    }

    /// Every output at float64's steps, the part's sum of squares through `sum` (exact, or gate 4's replay).
    fn points(self: *Ple, sum: anytype, out: []f64, terms: ?[]f64) !void {
        const al = self.rig.gpu.gpa;
        const x = try al.alloc(f64, width);
        defer al.free(x);
        const sq = try al.alloc(f64, width);
        defer al.free(sq);
        const bf = bfr.bf16;
        for (0..self.rows_n * self.parts) |j| {
            const at = j * width;
            for (x, sq, self.p[at..][0..width]) |*v, *q, pv| {
                v.* = bf(pv * self.c.scale);
                q.* = v.* * v.*;
            }
            const ss = sum.of(sq);
            const inv = 1.0 / @sqrt(ss / width + @as(f64, self.c.eps));
            for (out[at..][0..width], x, self.w, self.t[at..][0..width], at..) |*o, v, wv, tv, i| {
                const y = bf(wv * bf(v * inv));
                const tt = bf(tv * self.c.token);
                o.* = bf(bf(y + tt) * self.c.mix);
                if (terms) |m| m[i] = (@abs(y) + @abs(tt)) * self.c.mix;
            }
        }
    }

    const Exact = struct {
        fn of(_: Exact, v: []const f64) f64 {
            var s: f64 = 0;
            for (v) |x| s += x;
            return s;
        }
    };

    const Sim = struct {
        acc: c.Accum,
        order: rows.Order,
        fn of(s: Sim, v: []const f64) f64 {
            return rows.sumIn(s.acc, s.order, v);
        }
    };

    pub fn reference(self: *Ple, want: []f64, mag: []f64) !void {
        self.terms = try self.arena.?.allocator().alloc(f64, want.len);
        try self.points(Exact{}, want, self.terms);
        for (mag, want) |*m, v| m.* = @abs(v);
    }

    /// A point decided the other way moves y (three points) or t token by 2^-7 of its size; a cancelling sum keeps it.
    pub fn bounds(self: *Ple, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, self.terms) |*a, v, m| a.* = 0x1p-7 * (3 * m + 2 * @abs(v)) + 0x1p-133;
    }

    pub fn simulate(self: *Ple, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.points(Sim{ .acc = acc, .order = order }, out, null);
    }
    /// At the shapes gate: every output within float64's bound (`bounds`), and its bits layer_input.zig's host steps'.
    pub fn check(self: *Ple, at: g.Gate) !?[]const u8 {
        if (at != .shapes) return null;
        const li = g4.layer_input;
        const al = self.arena.?.allocator();
        const n = self.count();
        const got = try al.alloc(u16, n);
        @memcpy(std.mem.sliceAsBytes(got), try self.rig.bytes(self.out, n * 2));
        const want = try al.alloc(f64, n);
        const mag = try al.alloc(f64, n);
        const allow = try al.alloc(f64, n);
        try self.reference(want, mag);
        try self.bounds(want, mag, allow);
        for (got, want, allow, 0..) |h, v, lim, i| {
            const d = @abs(@as(f64, bfr.value(h)) - v);
            if (!(d <= lim)) return try std.fmt.bufPrint(&self.msg, "out[{d}] = {e} where float64 gives {e}: {d:.2} of its bound", .{ i, bfr.value(h), v, d / lim });
        }
        var row: [width]f32 = undefined;
        var w: [width]f32 = undefined;
        for (&w, self.w) |*x, v| x.* = @floatCast(v);
        for (0..n / width) |j| {
            const base = j * width;
            for (&row, self.p[base..][0..width]) |*x, v| x.* = li.rn(@as(f32, @floatCast(v)) * self.c.scale);
            li.rms(&row, &w, &row);
            for (row, self.t[base..][0..width], got[base..][0..width], 0..) |y, tv, h, i| {
                const bits = bfr.bits(li.rn(y + li.rn(@as(f32, @floatCast(tv)) * self.c.token)) * self.c.mix);
                if (bits != h) return try std.fmt.bufPrint(&self.msg, "part {d} out[{d}] = {x} where the host steps give {x}", .{ j, i, h, bits });
            }
        }
        return null;
    }
};

/// Every registration, its kernels and controls appended to the run's list (the controls from `controls`).
pub const Group = struct {
    a: std.mem.Allocator,
    rows: g4.ple.PleRows,
    each: [std.enums.values(Which).len]Ple,

    pub fn create(a: std.mem.Allocator, gpu: anytype, s: cuda.Stream, controls: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const x = try a.create(Group);
        errdefer a.destroy(x);
        x.* = .{ .a = a, .rows = try g4.ple.PleRows.load(gpu.d), .each = undefined };
        for (std.enums.values(Which), &x.each) |w, *p| {
            const ctl = if (w != .served) try controls.function("tf_ctl_ple_rows") else null;
            p.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .k = &x.rows, .ctl = ctl };
        }
        for (&x.each) |*p| try kernels.append(a, p.kernel());
        return x;
    }

    pub fn destroy(x: *Group) void {
        for (&x.each) |*p| p.deinit();
        x.rows.unload();
        x.a.destroy(x);
    }
};

test "gate 4's rows: integer squares sum exactly in fp32 and overflow fp16" {
    var terms: [width]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(0x706c65);
    for (&terms) |*t| {
        const v: f64 = @as(f64, @floatFromInt(prng.random().uintLessThan(u32, 129))) - 64;
        t.* = v * v;
    }
    for (std.enums.values(rows.Order)) |o| {
        var exact: f64 = 0;
        for (terms) |t| exact += t;
        try std.testing.expectEqual(exact, rows.sumIn(.f32, o, &terms));
        try std.testing.expect(std.math.isInf(rows.sumIn(.f16, o, &terms)));
    }
}
