//! act_rows.cu (#797) under the contract on CUDA: SwiGLU, the output gate, 64-value sums and DFlash2's two-tap conv.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const aref = core.act_ref;

pub const Which = enum { swiglu, swiglu_sums, gate, gate_sums, group_sums, dconv, dconv_res };

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A thread's 8 values in order, then 8 lanes' butterfly.
fn groupDepth(_: c.Shape) usize {
    return 11;
}

/// x k0 rounded, then the fma that adds the previous row's tap.
fn convDepth(_: c.Shape) usize {
    return 2;
}

const gs = 16;
const tiles_rows: c.Tile = .{ .size = 16 }; // launch.rowThreads: 16 rows or fewer take the small blocks
const tiles_n: c.Tile = .{ .size = 2048, .step = 64, .min = 64 }; // 256 threads of 8 values
const swiglu_served = &[_]c.Shape{ .{ .rows = 1, .n = 17408, .k = 1 }, .{ .rows = 64, .n = 17408, .k = 1 } }; // the 27B's MLP
const gate_served = &[_]c.Shape{ .{ .rows = 1, .n = 24, .k = 256 }, .{ .rows = 64, .n = 24, .k = 256 } }; // its output gate
const conv_served = &[_]c.Shape{ .{ .rows = 16, .n = 5120, .k = 16 }, .{ .rows = 64, .n = 5120, .k = 16 } }; // DFlash2's block

/// silu and sigmoid: expf (2 ulp, 4 units of 2^-24), the add, the division and the products.
const act_ops = 8;
const res_rule = "own check, core/act_ref's bound: the conv is rounded to bf16 before the residual add, an error of the conv's size";

/// swiglu, gate: rows x outputs (gate: heads x dim); sums: rows x 64-value groups; dconv: rows x channels x segment.
pub fn spec(which: Which) c.Spec {
    return switch (which) {
        .swiglu => .{ .name = "cuda act_rows tf_swiglu_rows", .accum = .none, .depth = noSums, .ops = act_ops, .out = .bf16, .tiles = .{ tiles_rows, tiles_n, .{} }, .served = swiglu_served },
        .swiglu_sums => .{ .name = "cuda act_rows tf_swiglu_rows group sums (of its stored bf16 outputs)", .accum = .f32, .depth = groupDepth, .out = .f32, .tiles = .{ tiles_rows, tiles_n, .{} }, .served = swiglu_served },
        .gate => .{ .name = "cuda act_rows tf_gate_rows", .accum = .none, .depth = noSums, .ops = act_ops, .out = .bf16, .tiles = .{ tiles_rows, .{}, .{ .size = 256, .step = 64, .min = 64 } }, .served = gate_served },
        .gate_sums => .{ .name = "cuda act_rows tf_gate_rows group sums (of its stored bf16 outputs)", .accum = .f32, .depth = groupDepth, .out = .f32, .tiles = .{ tiles_rows, .{}, .{ .size = 256, .step = 64, .min = 64 } }, .served = gate_served },
        .group_sums => .{ .name = "cuda act_rows tf_group_sums_rows", .accum = .f32, .depth = groupDepth, .out = .f32, .tiles = .{ .{}, .{}, tiles_n }, .served = &.{ .{ .rows = 1, .n = 1, .k = 5120 }, .{ .rows = 64, .n = 1, .k = 17408 } } },
        .dconv => .{ .name = "cuda act_rows tf_dconv_rows", .accum = .f32, .depth = convDepth, .out = .bf16, .tiles = .{ tiles_rows, .{ .size = 2048, .step = gs, .min = gs }, .{} }, .served = conv_served },
        .dconv_res => .{ .name = "cuda act_rows tf_dconv_rows with residual", .accum = .f32, .depth = convDepth, .out = .bf16, .nan = .{ .stated = res_rule }, .tiles = .{ tiles_rows, .{ .size = 2048, .step = gs, .min = gs }, .{} }, .served = conv_served },
    };
}

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

fn word(v: f32) u16 {
    return c.toBf16(v);
}

/// Gate 1's places: NaN, +inf, -inf, then both infinities in row 3 (rows that exist), in `x` [rows, w].
fn plant(x: []u16, rows_: usize, w: usize, first: bool) void {
    const at = [_][2]usize{ .{ 0, w / 3 }, .{ 1, w / 2 }, .{ 2, w - 1 }, .{ 3, 0 }, .{ 3, w - 1 } };
    const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (at, put, 0..) |p, v, i| if (p[0] < rows_ and (i < 2) == first) {
        x[p[0] * w + p[1]] = word(v);
    };
}

/// Gate 4's 64-value groups: 16 integers, then their negatives (four nudged by one), twice: big partials in each order.
fn mirrored(rng: std.Random, x: []u16, scale: f32, top: u32) void {
    var i: usize = 0;
    while (i + 64 <= x.len) : (i += 64) for (0..2) |h| for (0..16) |j| {
        const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, top));
        x[i + 32 * h + j] = word(v * scale);
        x[i + 32 * h + 16 + j] = word((-v + @as(f32, if (j < 4) 1 else 0)) * scale);
    };
}

pub const Act = struct {
    rig: rig_.Rig,
    which: Which,
    k: *const cuda.act_rows.ActRows,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x: []u16 = &.{},
    y: []u16 = &.{},
    base: []u16 = &.{},
    dyn: []u16 = &.{},
    res: []u16 = &.{},
    in: [5]u64 = @splat(0),
    out: u64 = 0,
    xs: u64 = 0,
    fill: g.Fill = .random,
    ran: bool = false,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Act) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Act) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = if (self.which == .dconv_res) &vt_res else &vt };
    }

    const vt = g.hooks(Act);
    const vt_res = blk: {
        var v = g.hooks(Act);
        v.check = checkRes;
        break :blk v;
    };

    /// Outputs a row's call writes: values, or (sums) one a 64-value group.
    fn width(self: *const Act) usize {
        return switch (self.which) {
            .gate, .gate_sums => self.shape.n * self.shape.k,
            .group_sums => self.shape.k,
            else => self.shape.n,
        };
    }

    fn sums(self: *const Act) bool {
        return self.which == .swiglu_sums or self.which == .gate_sums or self.which == .group_sums;
    }

    fn bfs(self: *Act, n: usize, rng: std.Random, scale: f32) ![]u16 {
        const v = try self.arena.allocator().alloc(u16, n);
        for (v, 0..) |*x, i| x.* = word((rng.float(f32) * 2 - 1) * scale * (if (i % 64 == 7) @as(f32, 20) else 1));
        return v;
    }

    pub fn prepare(self: *Act, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.fill = fill;
        self.ran = false;
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        const w = self.width();
        if (w % 64 != 0 and self.which != .dconv and self.which != .dconv_res) return false;
        switch (self.which) {
            .swiglu, .swiglu_sums, .gate, .gate_sums => {
                // x: the gates ([gate | up] rows, or [q | gate] heads), y: up or o
                const gated = self.which == .gate or self.which == .gate_sums;
                self.x = try self.bfs(r * 2 * w, rng, 3);
                self.y = try self.bfs(r * w, rng, 1);
                if (fill == .nonfinite) {
                    plant(self.y, r, w, false);
                    for (0..2) |i| if (i < r) {
                        self.x[i * 2 * w + (if (gated) shape.k else 0) + (if (i == 0) w / 3 else w / 2) % shape.k] = word(if (i == 0) std.math.nan(f32) else std.math.inf(f32));
                    };
                }
                if (fill == .stress) {
                    @memset(self.x, word(17)); // silu(17) and sigmoid(17) are 17 and 1 in fp32
                    mirrored(rng, self.y, if (gated) 17 else 1, 15); // outputs 17 u: odd, up to 255
                }
                const both = try self.arena.allocator().alloc(u16, r * 2 * w);
                @memcpy(both, self.x);
                // [gate | up]: up after a row's gates; [q | gate] heads: the gates are the second half of each head
                if (!gated) for (0..r) |i| @memcpy(both[i * 2 * w + w ..][0..w], self.y[i * w ..][0..w]);
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(both));
                if (gated) self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.y));
                if (self.sums()) {
                    self.out = try self.rig.scratch(r * w * 2);
                    self.xs = try self.rig.output(r * w / 64 * 4);
                } else {
                    self.out = try self.rig.output(r * w * 2);
                    self.xs = try self.rig.scratch(r * w / 64 * 4);
                }
            },
            .group_sums => {
                self.x = try self.bfs(r * (w + 64), rng, 2);
                if (fill == .nonfinite) plant(self.x, r, w + 64, true);
                if (fill == .nonfinite) plant(self.x, r, w + 64, false);
                if (fill == .stress) stressGroupRows(rng, self.x, r, w, w + 64);
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.x));
                self.xs = try self.rig.output(r * w / 64 * 4);
            },
            .dconv, .dconv_res => {
                if (w % gs != 0) return false;
                self.x = try self.bfs(r * w, rng, 1);
                self.dyn = try self.bfs(r * 4 * (w / gs), rng, 0.3);
                self.base = try self.bfs(4 * w, rng, 0.7);
                self.res = try self.bfs(r * w, rng, 1);
                if (fill == .nonfinite) {
                    plant(self.x, r, w, true);
                    plant(self.x, r, w, false);
                }
                if (fill == .stress) {
                    // rows alternate 255 and 254, taps 255 and -256: x k0 + p k1 = 1, which fp16 and bf16 lose
                    @memset(self.base, 0);
                    for (0..r) |i| {
                        @memset(self.x[i * w ..][0..w], word(if (i % 2 == 0) 255 else 254));
                        @memset(self.res[i * w ..][0..w], word(0.25));
                        for (0..2) |b| for (0..2) |t| {
                            const v: f32 = if ((i % 2 == 0) == (t == 0)) 255 else -256;
                            @memset(self.dyn[((i * 2 + b) * 2 + t) * (w / gs) ..][0 .. w / gs], word(v));
                        };
                    }
                }
                for ([_][]u16{ self.x, self.dyn, self.base, self.res }, 0..) |v, i| self.in[i] = try self.rig.input(std.mem.sliceAsBytes(v));
                self.out = try self.rig.output(r * w * 2);
            },
        }
        return true;
    }

    fn branch(self: *const Act) usize {
        return if (self.which == .dconv_res) 1 else 0;
    }

    pub fn run(self: *Act) !void {
        const s = self.rig.s;
        const r = self.shape.rows;
        const w = self.width();
        switch (self.which) {
            .swiglu, .swiglu_sums => try self.k.swiglu(s, self.in[0], 2 * w, self.in[0] + 2 * w, 2 * w, self.out, self.xs, r, w),
            .gate, .gate_sums => try self.k.outputGate(s, self.in[1], self.in[0] + 2 * self.shape.k, 2 * w, 2 * self.shape.k, self.out, self.xs, r, self.shape.n, self.shape.k),
            .group_sums => try self.k.groupSums(s, self.in[0], w + 64, self.xs, r, w),
            .dconv, .dconv_res => try self.k.dconv(s, self.in[0], self.in[1], self.in[2], if (self.which == .dconv_res) self.in[3] else null, self.out, r, w, gs, self.branch(), self.shape.k),
        }
        try s.synchronize();
        self.ran = true;
    }

    pub fn count(self: *Act) usize {
        const n = self.shape.rows * self.width();
        return if (self.sums()) n / 64 else n;
    }

    /// The values a group sums: x, or the bf16 outputs stored (before a run, gate 4's: silu(17) and sigmoid(17) exact).
    fn summed(self: *Act) ![]const u16 {
        if (self.which == .group_sums) return self.x;
        const n = self.shape.rows * self.width();
        const v = try self.arena.allocator().alloc(u16, n);
        if (self.ran) {
            @memcpy(std.mem.sliceAsBytes(v), try self.rig.bytes(self.out, n * 2));
            return v;
        }
        if (self.fill != .stress) return error.NoRunYet;
        const k = if (self.which == .swiglu_sums) aref.swiglu(17, 1) else aref.sigmoid(17);
        for (v, self.y) |*o, y| o.* = word(@floatCast(k * bf(y)));
        return v;
    }

    fn group(self: *Act, vals: []const u16, o: usize, t: *[64]f64) void {
        const w = self.width();
        const ld = if (self.which == .group_sums) w + 64 else w;
        const row = o / (w / 64);
        const j = o % (w / 64);
        for (t, 0..) |*v, e| v.* = bf(vals[row * ld + 64 * j + e]);
    }

    /// The conv's two terms at output o, and the residual (0 without).
    fn conv(self: *const Act, o: usize, t: *[2]f64) f64 {
        const w = self.shape.n;
        const row = o / w;
        const i = o % w;
        const b = self.branch();
        const grp = i / gs;
        const tap = struct {
            fn f(base: f64, dyn: f64) f64 {
                return bf(word(@as(f32, @floatCast(base)) + @as(f32, @floatCast(dyn)))); // k = bf16(base + dyn) in fp32
            }
        }.f;
        const k0 = tap(bf(self.base[(b * 2) * w + i]), bf(self.dyn[((row * 2 + b) * 2) * (w / gs) + grp]));
        const k1 = tap(bf(self.base[(b * 2 + 1) * w + i]), bf(self.dyn[((row * 2 + b) * 2 + 1) * (w / gs) + grp]));
        const prev = if (row % self.shape.k > 0) bf(self.x[(row - 1) * w + i]) else 0;
        t.* = .{ bf(self.x[row * w + i]) * k0, prev * k1 };
        return if (self.which == .dconv_res) bf(self.res[row * w + i]) else 0;
    }

    pub fn reference(self: *Act, want: []f64, mag: []f64) !void {
        const w = self.width();
        if (self.sums()) {
            const vals = try self.summed();
            var t: [64]f64 = undefined;
            for (want, mag, 0..) |*o, *m, i| {
                self.group(vals, i, &t);
                o.* = 0;
                m.* = 0;
                for (t) |v| {
                    o.* += v;
                    m.* += @abs(v);
                }
            }
            return;
        }
        for (want, mag, 0..) |*o, *m, i| {
            const row = i / w;
            const e = i % w;
            switch (self.which) {
                .swiglu => o.* = aref.swiglu(bf(self.x[row * 2 * w + e]), bf(self.y[i])),
                .gate => o.* = bf(self.y[i]) * aref.sigmoid(bf(self.x[row * 2 * w + (e / self.shape.k) * 2 * self.shape.k + self.shape.k + e % self.shape.k])),
                .dconv, .dconv_res => {
                    var t: [2]f64 = undefined;
                    const res = self.conv(i, &t);
                    o.* = res + aref.dconv(1, 1, t[0], t[1]);
                    m.* = @abs(t[0]) + @abs(t[1]);
                    continue;
                },
                else => unreachable,
            }
            m.* = @abs(o.*);
        }
    }

    pub fn simulate(self: *Act, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.sums()) {
            const vals = try self.summed();
            var t: [64]f64 = undefined;
            for (out, 0..) |*o, i| {
                self.group(vals, i, &t);
                o.* = rows.sumIn(acc, order, &t);
            }
            return;
        }
        if (self.which != .dconv and self.which != .dconv_res) return error.NoSums;
        var t: [2]f64 = undefined;
        for (out, 0..) |*o, i| {
            const res = self.conv(i, &t);
            o.* = res + rows.sumIn(acc, order, &t);
        }
    }

    /// core/act_ref's dconv bound with the residual: half a bf16 ulp of the output and of the conv, 2^-22 of its terms.
    fn checkRes(ctx: *anyopaque, at: g.Gate) anyerror!?[]const u8 {
        const self: *Act = @ptrCast(@alignCast(ctx));
        const out = try self.rig.outputBytes();
        const half = std.math.pow(f64, 2, -8);
        var planted: usize = 0;
        const n = self.count();
        var t: [2]f64 = undefined;
        for (0..n) |i| {
            const res = self.conv(i, &t);
            const y = t[0] + t[1];
            const want = res + y;
            const got = c.read(.bf16, out, i);
            if (c.classOf(want) != .finite) planted += 1;
            if (c.classOf(got) != c.classOf(want)) return try std.fmt.bufPrint(&self.msg, "output {d} is {t} where float64 gives {t}", .{ i, c.classOf(got), c.classOf(want) });
            if (at == .nonfinite or c.classOf(want) != .finite) continue;
            const tol = @abs(want) * half + @abs(y) * half + (@abs(t[0]) + @abs(t[1])) * std.math.pow(f64, 2, -22) + 1e-30;
            if (!(@abs(got - want) <= tol)) return try std.fmt.bufPrint(&self.msg, "output {d} is {e:.6} where float64 gives {e:.6}: {d:.2} of its bound", .{ i, got, want, @abs(got - want) / tol });
        }
        if (at == .nonfinite and (planted == 0 or planted == n)) return try std.fmt.bufPrint(&self.msg, "the planted values make {d} of {d} outputs non-finite: no proof", .{ planted, n });
        return null;
    }
};

/// Gate 4's production group-sum stress fill, shared with its host sensitivity test.
pub fn stressGroupRows(rng: std.Random, x: []u16, rows_: usize, width: usize, stride: usize) void {
    _ = rng;
    // Every fp32 order is exactly 32; fp16 loses at least 16 and bf16 loses all 32, without overflow.
    for (0..rows_) |r| for (0..width / 64) |j| for (0..64) |i| {
        const value: f32 = if (i < 16) 2048 else if (i < 48) 1 else -2048;
        x[r * stride + j * 64 + i] = word(value);
    };
}
