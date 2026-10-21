//! moe_glue.cu and dense_rows.cu (#798) under the contract on CUDA: sigmoid top-k, the slots' combine, dense rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const Dense = contract.dense.Dense;
const glue = cuda.moe_glue;

fn noSums(_: c.Shape) usize {
    return 0;
}

const pick_rule = "picks by fp32 logit + bias, larger first, ties to the lower id, a NaN never picked; weights sigmoid(logit)";

/// A routing's fixed form; shape: rows, experts (n) and top-k (k).
pub const Route = struct { name: []const u8, shared: bool, normalize: bool = false, scale: f32 = 1, warp: bool };

pub const routes = [_]Route{
    .{ .name = "cuda moe top-k tf_moe_topk_sigmoid_warp (Kolibri: 384 experts, top 6 + shared)", .shared = true, .warp = true },
    .{ .name = "cuda moe top-k tf_moe_topk_sigmoid_warp (256 experts, top 8 normalized x2.5)", .shared = false, .normalize = true, .scale = 2.5, .warp = true },
    .{ .name = "cuda moe top-k tf_moe_topk_sigmoid (block form, 513 to 4,096 experts)", .shared = true, .warp = false },
};

pub fn topkSpec(r: Route) c.Spec {
    const served: []const c.Shape = if (!r.warp) &.{ .{ .rows = 16, .n = 4096, .k = 64 }, .{ .rows = 33, .n = 513, .k = 6 } } else if (r.normalize) &.{ .{ .rows = 16, .n = 256, .k = 8 }, .{ .rows = 300, .n = 256, .k = 8 } } else &.{ .{ .rows = 16, .n = 384, .k = 6 }, .{ .rows = 300, .n = 384, .k = 6 } };
    const tiles: [3]c.Tile = if (r.warp) .{ .{ .size = 8 }, .{ .size = 32, .min = 8 }, .{} } else .{ .{}, .{ .size = 256, .min = 513 }, .{} };
    return .{ .name = r.name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = pick_rule }, .tiles = tiles, .served = served };
}

pub const TopK = struct {
    rig: rig_.Rig,
    route: Route,
    g: *const glue.MoeGlue,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    logits: std.ArrayList(f32) = .empty,
    bias: std.ArrayList(f32) = .empty,
    dev: [4]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *TopK) void {
        self.logits.deinit(self.rig.gpu.gpa);
        self.bias.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *TopK) g.Kernel {
        return .{ .spec = topkSpec(self.route), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(TopK);

    fn slots(self: *const TopK) usize {
        return self.shape.k + @intFromBool(self.route.shared);
    }

    fn form(self: *const TopK) glue.Route {
        return .{ .experts = self.shape.n, .topk = self.shape.k, .shared = self.route.shared, .normalize = self.route.normalize, .scale = self.route.scale };
    }

    pub fn prepare(self: *TopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        const warp = shape.n <= 512 and shape.k <= 32;
        if (warp != self.route.warp or shape.rows == 0 or shape.k == 0 or shape.k > shape.n or shape.n > glue.most_experts or shape.k > glue.most_topk) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const ne = shape.n;
        try self.logits.resize(a, shape.rows * ne);
        try self.bias.resize(a, ne);
        for (self.logits.items, 0..) |*l, i| {
            l.* = rng.floatNorm(f32) * 2;
            if (i % 7 == 3) l.* = self.logits.items[i - 1]; // equal neighbours: the tie rule decides
        }
        for (self.bias.items, 0..) |*b, i| b.* = if (i % 5 == 0) 0.25 else 0;
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, 3 }, .{ 1, 5 }, .{ 2, 7 }, .{ 3, 2 }, .{ 3, ne - 1 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                self.logits.items[p[0] * ne + p[1]] = v;
            };
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.logits.items));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.bias.items));
        self.dev[2] = try self.rig.output(shape.rows * self.slots() * 4);
        self.dev[3] = try self.rig.output(shape.rows * self.slots() * 4);
        return true;
    }

    pub fn run(self: *TopK) !void {
        try self.g.topk(self.rig.s, self.dev[0], self.dev[1], self.form(), self.dev[2], self.dev[3], self.shape.rows);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *TopK) usize {
        return self.shape.rows * self.slots() * 8;
    }

    pub fn reference(_: *TopK, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Each row's picks by the stated rule exactly, weights within 2^-20 of float64, the shared slot last.
    pub fn check(self: *TopK, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const out = try self.rig.outputBytes();
        const n = self.shape.rows * self.slots();
        const ne = self.shape.n;
        const taken = try a.alloc(bool, ne);
        defer a.free(taken);
        for (0..self.shape.rows) |r| {
            const lg = self.logits.items[r * ne ..][0..ne];
            @memset(taken, false);
            var w64: [65]f64 = undefined;
            var sum: f64 = 0;
            for (0..self.shape.k) |k| {
                var best: ?usize = null;
                var bv: f32 = 0;
                for (lg, 0..) |l, e| {
                    const v = l + self.bias.items[e];
                    if (taken[e] or std.math.isNan(v)) continue;
                    if (best == null or v > bv) {
                        best = e;
                        bv = v;
                    }
                }
                const id = best orelse return try std.fmt.bufPrint(&self.msg, "row {d}: fewer than {d} numbers to pick", .{ r, self.shape.k });
                taken[id] = true;
                const got = std.mem.readInt(u32, out[(r * self.slots() + k) * 4 ..][0..4], .little);
                if (got != id) return try std.fmt.bufPrint(&self.msg, "row {d} pick {d} is {d} where the rule gives {d}", .{ r, k, got, id });
                w64[k] = 1 / (1 + @exp(-@as(f64, lg[id])));
                sum += w64[k];
            }
            if (self.route.shared) w64[self.shape.k] = 1;
            for (0..self.slots()) |k| {
                const want = if (k == self.shape.k) 1 else (if (self.route.normalize) w64[k] / sum else w64[k]) * self.route.scale;
                const got: f64 = @as(f32, @bitCast(std.mem.readInt(u32, out[(n + r * self.slots() + k) * 4 ..][0..4], .little)));
                if (k == self.shape.k and std.mem.readInt(u32, out[(r * self.slots() + k) * 4 ..][0..4], .little) != ne) return try std.fmt.bufPrint(&self.msg, "row {d}: the shared slot is not expert {d}", .{ r, ne });
                if (!(@abs(got - want) <= @abs(want) * 0x1p-20 + 1e-30)) return try std.fmt.bufPrint(&self.msg, "row {d} slot {d}: weight {d:.8} where float64 gives {d:.8}", .{ r, k, got, want });
            }
        }
        return null;
    }
};

/// The combine's kinds: y in, out; Kolibri combines fp32 expert rows into bf16.
pub const Combo = struct { name: []const u8, y: cuda.norms.Kind, out: cuda.norms.Kind, elem: c.Elem };

pub const combos = [_]Combo{
    .{ .name = "cuda moe combine tf_moe_combine (fp32 rows into bf16)", .y = .f32, .out = .bf16, .elem = .bf16 },
    .{ .name = "cuda moe combine tf_moe_combine (bf16 rows into bf16)", .y = .bf16, .out = .bf16, .elem = .bf16 },
    .{ .name = "cuda moe combine tf_moe_combine (fp32 rows into fp32)", .y = .f32, .out = .f32, .elem = .f32 },
    .{ .name = "cuda moe combine tf_moe_combine (fp16 rows into fp16, the generic loop)", .y = .f16, .out = .f16, .elem = .f16 },
};

/// One fma a slot, in slot order.
fn slotChain(s: c.Shape) usize {
    return s.k;
}

/// Shape: rows, dims (n) and slots (k): Kolibri's 6 + shared over 2,560, eight + shared over 7,168.
pub fn combineSpec(x: Combo) c.Spec {
    return .{ .name = x.name, .accum = .f32, .depth = slotChain, .out = x.elem, .tiles = .{ .{}, .{ .size = 256 }, .{ .size = 8 } }, .served = &.{ .{ .rows = 16, .n = 2560, .k = 7 }, .{ .rows = 300, .n = 2560, .k = 7 }, .{ .rows = 9, .n = 7168, .k = 9 } } };
}

pub const Combine = struct {
    rig: rig_.Rig,
    x: Combo,
    g: *const glue.MoeGlue,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    y: std.ArrayList(f64) = .empty,
    w: std.ArrayList(f32) = .empty,
    dev: [3]u64 = @splat(0),

    pub fn deinit(self: *Combine) void {
        self.y.deinit(self.rig.gpu.gpa);
        self.w.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Combine) g.Kernel {
        return .{ .spec = combineSpec(self.x), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Combine);

    fn put(kind: cuda.norms.Kind, bytes: []u8, i: usize, v: f32) f64 {
        switch (kind) {
            .f32 => std.mem.writeInt(u32, bytes[4 * i ..][0..4], @bitCast(v), .little),
            .f16 => std.mem.writeInt(u16, bytes[2 * i ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little),
            .bf16 => std.mem.writeInt(u16, bytes[2 * i ..][0..2], c.toBf16(v), .little),
        }
        return switch (kind) {
            .f32 => v,
            .f16 => @as(f16, @floatCast(v)),
            .bf16 => c.fromBf16(c.toBf16(v)),
        };
    }

    pub fn prepare(self: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const n = shape.rows * shape.k * shape.n;
        try self.y.resize(a, n);
        try self.w.resize(a, shape.rows * shape.k);
        const esize: usize = if (self.x.y == .f32) 4 else 2;
        const bytes = try a.alloc(u8, n * esize);
        defer a.free(bytes);
        for (self.w.items) |*x| x.* = if (fill == .stress) @floatFromInt(1 + 2 * rng.uintLessThan(u32, 512)) else rng.float(f32);
        const h = shape.k / 2;
        for (0..shape.rows) |r| for (0..shape.k) |k| for (0..shape.n) |d| {
            const i = (r * shape.k + k) * shape.n + d;
            var v = rng.floatNorm(f32) * (if (i % 97 == 0) @as(f32, 30) else 1);
            if (fill == .stress) {
                // integer products w y to 2^18, the second half of the slots cancelling the first (one nudged)
                if (shape.k % 2 == 1 and k == shape.k - 1) v = 0 else if (k >= h) {
                    self.w.items[r * shape.k + k] = self.w.items[r * shape.k + k - h];
                    v = (if (k == h) @as(f32, 1) else 0) - @as(f32, @floatCast(self.y.items[i - h * shape.n]));
                } else v = @floatFromInt(1 + 2 * rng.uintLessThan(u32, 128));
            }
            self.y.items[i] = put(self.x.y, bytes, i, v);
        };
        if (fill == .nonfinite) {
            const at = [_][3]usize{ .{ 0, 0, shape.n / 3 }, .{ 1, 1, shape.n / 2 }, .{ 2, 0, shape.n - 1 }, .{ 3, 0, 0 }, .{ 3, shape.k - 1, 0 } };
            const vals = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, vals) |p, v| if (p[0] < shape.rows and p[1] < shape.k) {
                const i = (p[0] * shape.k + p[1]) * shape.n + p[2];
                self.y.items[i] = put(self.x.y, bytes, i, v);
            };
        }
        self.dev[0] = try self.rig.input(bytes);
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.w.items));
        self.dev[2] = try self.rig.output(shape.rows * shape.n * c.size(self.x.elem));
        return true;
    }

    pub fn run(self: *Combine) !void {
        try self.g.combine(self.rig.s, self.dev[0], self.x.y, self.dev[1], self.dev[2], self.x.out, self.shape.rows, self.shape.k, self.shape.n);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Combine) usize {
        return self.shape.rows * self.shape.n;
    }

    fn terms(self: *Combine, r: usize, d: usize, t: []f64) void {
        for (t, 0..) |*x, k| x.* = @as(f64, self.w.items[r * self.shape.k + k]) * self.y.items[(r * self.shape.k + k) * self.shape.n + d];
    }

    /// core/moe_ref.combine's sum, and the terms' magnitude.
    pub fn reference(self: *Combine, want: []f64, mag: []f64) !void {
        var t: [64]f64 = undefined;
        for (0..self.shape.rows) |r| for (0..self.shape.n) |d| {
            self.terms(r, d, t[0..self.shape.k]);
            var s: f64 = 0;
            var abs: f64 = 0;
            for (t[0..self.shape.k]) |x| {
                s += x;
                abs += @abs(x);
            }
            want[r * self.shape.n + d] = s;
            mag[r * self.shape.n + d] = abs;
        };
    }

    pub fn simulate(self: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        var t: [64]f64 = undefined;
        for (0..self.shape.rows) |r| for (0..self.shape.n) |d| {
            self.terms(r, d, t[0..self.shape.k]);
            out[r * self.shape.n + d] = rows.sumIn(acc, order, t[0..self.shape.k]);
        };
    }
};

/// A lane's every 256th octet in order (k / 32 fma), then the warp's 5-step tree.
fn laneChain(s: c.Shape) usize {
    return s.k / 32 + 5;
}

pub const dense_tiles = [_]usize{ 1, 4, 16, 32 };

/// Shape: rows, outputs (n) and k; each entry launches its own row tile (the wrapper picks it by row count).
pub fn denseSpec(i: usize) c.Spec {
    const names = [_][]const u8{ "cuda dense rows tf_dense_rows_1", "cuda dense rows tf_dense_rows_4", "cuda dense rows tf_dense_rows_16", "cuda dense rows tf_dense_rows_32" };
    // Kolibri's router (384 x 2,560), an output head's odd 4,103 columns, 257 x 7,168; rows as the wrapper sends them
    const m = [_][2]usize{ .{ 1, 1 }, .{ 4, 3 }, .{ 16, 9 }, .{ 32, 40 } };
    const served = [_][3]c.Shape{
        .{ .{ .rows = m[0][0], .n = 384, .k = 2560 }, .{ .rows = m[0][1], .n = 4103, .k = 2560 }, .{ .rows = m[0][0], .n = 257, .k = 7168 } },
        .{ .{ .rows = m[1][0], .n = 384, .k = 2560 }, .{ .rows = m[1][1], .n = 4103, .k = 2560 }, .{ .rows = m[1][0], .n = 257, .k = 7168 } },
        .{ .{ .rows = m[2][0], .n = 384, .k = 2560 }, .{ .rows = m[2][1], .n = 4103, .k = 2560 }, .{ .rows = m[2][0], .n = 257, .k = 7168 } },
        .{ .{ .rows = m[3][0], .n = 384, .k = 2560 }, .{ .rows = m[3][1], .n = 4103, .k = 2560 }, .{ .rows = m[3][0], .n = 257, .k = 7168 } },
    };
    return .{ .name = names[i], .accum = .f32, .depth = laneChain, .out = .f32, .tiles = .{ .{ .size = dense_tiles[i] }, .{ .size = 8 }, .{ .size = 256, .step = 8, .min = 8 } }, .served = &served[i] };
}

pub const DenseRows = struct {
    rig: rig_.Rig,
    tile: usize,
    g: *const cuda.dense_rows.DenseRows,
    d: ?Dense = null,
    dev: [3]u64 = @splat(0),

    pub fn deinit(self: *DenseRows) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *DenseRows) g.Kernel {
        return .{ .spec = denseSpec(self.tile), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(DenseRows);

    pub fn prepare(self: *DenseRows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 8 != 0 or shape.k == 0 or shape.n == 0 or shape.rows == 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        const bits = try a.alloc(u16, d.w.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x77);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*w, b| w.* = c.fromBf16(b);
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.dev[2] = try self.rig.output(d.outputs() * 4);
        return true;
    }

    /// The wrapper's launch, at this entry's row tile whatever the row count.
    pub fn run(self: *DenseRows) !void {
        const sh = self.d.?.shape;
        var a: cuda.Args = .{};
        a.add(self.dev[0]);
        a.add(@as(i64, @intCast(sh.k)));
        a.add(self.dev[1]);
        a.add(self.dev[2]);
        a.add(@as(i64, @intCast(sh.n)));
        for ([_]usize{ sh.rows, sh.n, sh.k }) |v| a.add(@as(i32, @intCast(v)));
        const rt = dense_tiles[self.tile];
        const grid: cuda.Dim3 = .{ .x = @intCast((sh.n + 7) / 8), .y = @intCast((sh.rows + rt - 1) / rt), .z = 1 };
        try cuda.launch.launch(self.g.fns[self.tile], .{ .grid = grid, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *DenseRows) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *DenseRows, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *DenseRows, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};
