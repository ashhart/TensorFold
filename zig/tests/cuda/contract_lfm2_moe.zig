//! lfm2_moe.cu under the contract: router logits and top-4 rule, 3-tap liquid conv and snapshots, commit; controlled.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const Dense = contract.dense.Dense;

const experts = 64;
const topk = 4;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A quarter of the row an fma chain, then the quarters' two adds.
fn quarterChain(s: c.Shape) usize {
    return s.k / 4 + 2;
}

fn f32At(bytes: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
}

fn routeArgs(x: u64, w: u64, bias: u64, picks: u64, wts: u64, logits: u64, rows: usize, d: usize) cuda.Args {
    var a: cuda.Args = .{};
    for ([_]u64{ x, w, bias, picks, wts, logits }) |p| a.add(p);
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(d)));
    a.add(@as(f32, 1.0));
    return a;
}

fn launch(f: cuda.Function, s: cuda.Stream, grid: [3]usize, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = 256 } }, s, a);
    try s.synchronize();
}

pub fn logitsSpec() c.Spec {
    return .{ .name = "cuda lfm2_moe router logits tf_lfm2m_route (fp32 weights, quarters of the row)", .accum = .f32, .depth = quarterChain, .out = .f32, .tiles = .{ .{ .size = 8 }, .{}, .{ .size = 2048, .step = 4, .min = 4 } }, .served = &.{ .{ .rows = 1, .n = experts, .k = 2048 }, .{ .rows = 16, .n = experts, .k = 2048 }, .{ .rows = 1024, .n = experts, .k = 2048 } } };
}

/// x [rows, D] bf16 times fp32 router rows [64, D]: the logits (the picks and weights go to scratch).
pub const Logits = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    d: ?Dense = null,
    dev: [4]u64 = @splat(0),

    pub fn deinit(self: *Logits) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Logits) g.Kernel {
        return .{ .spec = logitsSpec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Logits);

    pub fn prepare(self: *Logits, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != experts or shape.k % 4 != 0 or shape.k == 0 or shape.k > 2048 or shape.rows == 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 16);
        const w = try a.alloc(f32, d.w.len);
        defer a.free(w);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x51);
        const rng = prng.random();
        const k = shape.k;
        // stress: positive integers, exact in fp32 in any order; mirrored signs would cancel inside a 32-lane fp16 sum
        for (w, 0..) |*v, i| v.* = if (fill != .stress) rng.float(f32) * 2 - 1 else @floatFromInt(1 + (i % k * 7 + i / k) % 6);
        for (d.w, w) |*o, v| o.* = v;
        const bias: [experts]f32 = @splat(0);
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(w));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(&bias));
        self.dev[3] = try self.rig.output(shape.rows * experts * 4);
        return true;
    }

    pub fn run(self: *Logits) !void {
        const d = self.d.?;
        const picks = try self.rig.scratch(d.shape.rows * topk * 4);
        const wts = try self.rig.scratch(d.shape.rows * topk * 4);
        var a = routeArgs(self.dev[0], self.dev[1], self.dev[2], picks, wts, self.dev[3], d.shape.rows, d.shape.k);
        try launch(self.f, self.rig.s, .{ (d.shape.rows + 7) / 8, 1, 1 }, &a);
    }

    pub fn count(self: *Logits) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Logits, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Logits, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};

const pick_rule = "top 4 of sigmoid(logit) + bias, ties to the lower id, a NaN ranked as -inf; weights sigmoid / (sum + 1e-6)";

pub fn selectSpec(control: bool) c.Spec {
    return .{ .name = if (control) "control: lfm2_moe router with ties to the higher expert" else "cuda lfm2_moe router top-4 tf_lfm2m_route (its rule on fp32 logits)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = pick_rule }, .tiles = .{ .{ .size = 8 }, .{}, .{} }, .served = &.{ .{ .rows = 16, .n = experts, .k = experts }, .{ .rows = 300, .n = experts, .k = experts } } };
}

/// The rule's picks on logits the identity router passes through (logit e = x[e]): sixteenths, so ties are common.
pub const Select = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    bias: [experts]f32 = undefined,
    dev: [5]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Select) void {
        self.rig.gpu.gpa.free(self.x);
        self.rig.deinit();
    }

    pub fn kernel(self: *Select) g.Kernel {
        return .{ .spec = selectSpec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Select);

    pub fn prepare(self: *Select, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != experts or shape.k != experts or shape.rows == 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        a.free(self.x);
        self.x = &.{};
        self.x = try a.alloc(f32, shape.rows * experts);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.x) |*v| v.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 16 - 2;
        for (&self.bias) |*b| b.* = if (rng.boolean()) 0 else 0.0625;
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, 3 }, .{ 1, 5 }, .{ 2, 7 }, .{ 3, 2 }, .{ 3, experts - 1 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                self.x[p[0] * experts + p[1]] = v;
            };
        }
        const xb = try a.alloc(u16, self.x.len);
        defer a.free(xb);
        for (xb, self.x) |*b, v| b.* = c.toBf16(v);
        var w: [experts * experts]f32 = @splat(0);
        for (0..experts) |e| w[e * experts + e] = 1;
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(xb));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(&w));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(&self.bias));
        self.dev[3] = try self.rig.output(shape.rows * topk * 4);
        self.dev[4] = try self.rig.output(shape.rows * topk * 4);
        return true;
    }

    pub fn run(self: *Select) !void {
        const logits = try self.rig.scratch(self.shape.rows * experts * 4);
        var a = routeArgs(self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.dev[4], logits, self.shape.rows, experts);
        try launch(self.f, self.rig.s, .{ (self.shape.rows + 7) / 8, 1, 1 }, &a);
    }

    pub fn count(self: *Select) usize {
        return self.shape.rows * topk * 8;
    }

    pub fn reference(_: *Select, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Picks by the rule exactly, weights within 2^-20 of float64 (NaN where a picked logit is NaN).
    pub fn check(self: *Select, _: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        const n = self.shape.rows * topk;
        for (0..self.shape.rows) |r| {
            var lg: [experts]f64 = undefined; // the identity's logits as a dot: one non-finite x makes the whole row NaN
            for (&lg, 0..) |*l, e| {
                l.* = 0;
                for (self.x[r * experts ..][0..experts], 0..) |v, i| l.* += @as(f64, v) * @as(f64, if (i == e) 1 else 0);
            }
            var taken: [experts]bool = @splat(false);
            var gate: [topk]f64 = undefined;
            var sum: f64 = 0;
            for (0..topk) |k| {
                var best: ?usize = null;
                var bv: f64 = 0;
                for (lg, 0..) |l, e| {
                    const s = 1 / (1 + @exp(-l));
                    const sv: f64 = @as(f32, @floatCast(s)) + @as(f64, self.bias[e]);
                    const v = if (std.math.isNan(sv)) -std.math.inf(f64) else sv;
                    if (taken[e]) continue;
                    if (best == null or v > bv) {
                        best = e;
                        bv = v;
                    }
                }
                const id = best orelse return try std.fmt.bufPrint(&self.msg, "row {d}: fewer than 4 numbers", .{r});
                taken[id] = true;
                const got = std.mem.readInt(u32, out[(r * topk + k) * 4 ..][0..4], .little);
                if (got != id) return try std.fmt.bufPrint(&self.msg, "row {d} pick {d} is {d} where the rule gives {d}", .{ r, k, got, id });
                gate[k] = 1 / (1 + @exp(-lg[id]));
                sum += gate[k];
            }
            for (0..topk) |k| {
                const want = gate[k] / (sum + 1e-6);
                const got: f64 = f32At(out, n + r * topk + k);
                if (std.math.isNan(want) and std.math.isNan(got)) continue;
                if (!(@abs(got - want) <= @abs(want) * 0x1p-20 + 1e-30)) return try std.fmt.bufPrint(&self.msg, "row {d} slot {d}: weight {d:.8} where float64 gives {d:.8}", .{ r, k, got, want });
            }
        }
        return null;
    }
};

const conv_rule = "the host's replay of each row's fp32 and bf16 steps, bit for bit (NaN as NaN)";

pub fn convSpec(taps: usize, control: bool) c.Spec {
    const tiles: [3]c.Tile = .{ .{}, .{ .size = 256, .step = 64, .min = 64 }, .{} };
    if (taps == 7) return .{ .name = if (control) "control: liquid 7-tap conv with bx unrounded" else "cuda liquid 7-tap short conv tf_liquid_conv_7 (stream state, every row's snapshot)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = conv_rule }, .tiles = tiles, .served = &.{ .{ .rows = 1, .n = 2048, .k = 7 }, .{ .rows = 16, .n = 2048, .k = 7 }, .{ .rows = 128, .n = 2048, .k = 7 } } };
    return .{ .name = if (control) "control: liquid 3-tap conv with bx unrounded" else "cuda liquid 3-tap short conv tf_liquid_conv_3 (stream state, every row's snapshot)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = conv_rule }, .tiles = tiles, .served = &.{ .{ .rows = 1, .n = 2048, .k = 3 }, .{ .rows = 16, .n = 2048, .k = 3 }, .{ .rows = 128, .n = 2048, .k = 3 } } };
}

/// Rows of two streams (the first rows/2 slot 1, the rest slot 0) over their states: y and every row's state after it.
pub const Conv = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false,
    taps: usize = 3,
    shape: c.Shape = undefined,
    p: []u16 = &.{},
    w: []u16 = &.{},
    st: []u16 = &.{},
    slot: []i32 = &.{},
    off: []i32 = &.{},
    dev: [7]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Conv) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Conv) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ "p", "w", "st", "slot", "off" }) |f| {
            a.free(@field(self, f));
            @field(self, f) = &.{};
        }
    }

    pub fn kernel(self: *Conv) g.Kernel {
        return .{ .spec = convSpec(self.taps, self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Conv);

    pub fn prepare(self: *Conv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 64 != 0 or shape.k != self.taps) return false;
        self.rig.reset();
        self.free();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        const r = shape.rows;
        const d = shape.n;
        const sr = self.taps - 1; // state rows
        self.p = try a.alloc(u16, r * 3 * d);
        self.w = try a.alloc(u16, d * self.taps);
        self.st = try a.alloc(u16, 2 * sr * d);
        self.slot = try a.alloc(i32, r);
        self.off = try a.alloc(i32, r);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.p) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (self.w) |*v| v.* = c.toBf16(rng.float(f32) - 0.5);
        for (self.st) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        const half = r / 2;
        for (self.slot, self.off, 0..) |*s, *o, i| {
            s.* = if (i < half) 1 else 0;
            o.* = @intCast(if (i < half) i else i - half);
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, d / 3 }, .{ 1, d + 1 }, .{ 2, 2 * d + 5 }, .{ 3, 0 }, .{ 3, 3 * d - 1 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |q, v| if (q[0] < r) {
                self.p[q[0] * 3 * d + q[1]] = c.toBf16(v);
            };
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.p));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.w));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.st));
        self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.slot));
        self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(self.off));
        self.dev[5] = try self.rig.output(r * d * 2);
        self.dev[6] = try self.rig.output(r * sr * d * 2);
        return true;
    }

    pub fn run(self: *Conv) !void {
        const d = self.shape.n;
        var a: cuda.Args = .{};
        for (self.dev) |p| a.add(p);
        a.add(@as(c_int, 0));
        a.add(@as(c_int, @intCast(d)));
        a.add(@as(i64, @intCast((self.taps - 1) * d)));
        try launch(self.f, self.rig.s, .{ self.shape.rows, (d + 255) / 256, 1 }, &a);
    }

    pub fn count(self: *Conv) usize {
        return self.shape.rows * self.shape.n * 2 * self.taps;
    }

    pub fn reference(_: *Conv, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn val(h: u16) f32 {
        return c.fromBf16(h);
    }

    fn same(got: u16, want: u16) bool {
        return got == want or (std.math.isNan(val(got)) and std.math.isNan(val(want)));
    }

    pub fn check(self: *Conv, _: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        const d = self.shape.n;
        const r = self.shape.rows;
        const taps = self.taps;
        const sr = taps - 1;
        for (0..r) |row| for (0..d) |ch| {
            const o: usize = @intCast(self.off[row]);
            var bx: [7]u16 = undefined;
            for (0..taps) |j| {
                const back = sr - j;
                bx[j] = if (o >= back) blk: {
                    const src = (row - back) * 3 * d;
                    break :blk c.toBf16(val(self.p[src + ch]) * val(self.p[src + 2 * d + ch]));
                } else self.st[@as(usize, @intCast(self.slot[row])) * sr * d + (sr - back + o) * d + ch];
            }
            var acc: f32 = 0;
            for (0..taps) |j| acc += val(self.w[ch * taps + j]) * val(bx[j]);
            const y = c.toBf16(val(self.p[row * 3 * d + d + ch]) * val(c.toBf16(acc)));
            for (0..taps) |which| { // y, then the row's state rows: its taps 1.. (oldest first)
                const at = if (which == 0) row * d + ch else r * d + (row * sr + which - 1) * d + ch;
                const want: u16 = if (which == 0) y else bx[which];
                const got = std.mem.readInt(u16, out[2 * at ..][0..2], .little);
                if (!same(got, want)) return try std.fmt.bufPrint(&self.msg, "row {d} channel {d} output {d}: 0x{x:0>4} where the replay gives 0x{x:0>4}", .{ row, ch, which, got, want });
            }
        };
        return null;
    }
};

pub fn commitSpec(taps: usize, control: bool) c.Spec {
    if (taps == 7) return .{ .name = if (control) "control: liquid 7-tap commit into the first stream's slot" else "cuda liquid 7-tap conv-state commit tf_liquid_commit_7", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "a copy: every bit as stored" }, .served = &.{ .{ .rows = 1, .n = 2048, .k = 13 }, .{ .rows = 16, .n = 2048, .k = 13 } } };
    return .{ .name = if (control) "control: liquid 3-tap commit into the first stream's slot" else "cuda liquid 3-tap conv-state commit tf_liquid_commit_3", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "a copy: every bit as stored" }, .served = &.{ .{ .rows = 1, .n = 2048, .k = 30 }, .{ .rows = 16, .n = 2048, .k = 30 } } };
}

/// Streams k = 0.. into slots rows - k of rows + 1, each its snapshot row 2k (stream 1: none, its slot as it was).
pub const Commit = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false,
    taps: usize = 3,
    shape: c.Shape = undefined,
    snaps: []u16 = &.{},
    dev: [4]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Commit) void {
        self.rig.gpu.gpa.free(self.snaps);
        self.rig.deinit();
    }

    pub fn kernel(self: *Commit) g.Kernel {
        return .{ .spec = commitSpec(self.taps, self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Commit);

    fn rowOf(k: usize) i32 {
        return if (k == 1) -1 else @intCast(2 * k);
    }

    pub fn prepare(self: *Commit, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        a.free(self.snaps);
        self.snaps = &.{};
        const r = shape.rows;
        const sw = (self.taps - 1) * shape.n; // a state's elements
        self.snaps = try a.alloc(u16, shape.k * 2 * r * sw);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.snaps) |*v| v.* = if (fill == .nonfinite and rng.uintLessThan(u32, 97) == 0) 0x7fc1 else rng.int(u16);
        const slots = try a.alloc(i32, r);
        defer a.free(slots);
        const rows = try a.alloc(i32, r);
        defer a.free(rows);
        for (slots, rows, 0..) |*s, *w, k| {
            s.* = @intCast(r - k);
            w.* = rowOf(k);
        }
        self.dev[0] = try self.rig.output(shape.k * (r + 1) * sw * 2);
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.snaps));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(slots));
        self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(rows));
        return true;
    }

    pub fn run(self: *Commit) !void {
        const d = self.shape.n;
        const r = self.shape.rows;
        var a: cuda.Args = .{};
        for (self.dev) |p| a.add(p);
        const sw = (self.taps - 1) * d;
        a.add(@as(c_int, @intCast(d)));
        a.add(@as(i64, @intCast((r + 1) * sw)));
        a.add(@as(i64, @intCast(2 * r * sw)));
        a.add(@as(i64, @intCast(sw)));
        try launch(self.f, self.rig.s, .{ self.shape.k, r, 1 }, &a);
    }

    pub fn count(self: *Commit) usize {
        return self.shape.k * (self.shape.rows + 1) * (self.taps - 1) * self.shape.n * 2;
    }

    pub fn reference(_: *Commit, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(self: *Commit, _: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        const d = self.shape.n;
        const r = self.shape.rows;
        const sw = (self.taps - 1) * d;
        for (0..self.shape.k) |layer| for (0..r + 1) |slot| for (0..sw) |i| {
            const k = r - slot; // the stream that owns this slot (slot 0: none)
            const want: u16 = if (slot == 0 or rowOf(k) < 0) 0xffff else self.snaps[layer * 2 * r * sw + @as(usize, @intCast(rowOf(k))) * sw + i];
            const got = std.mem.readInt(u16, out[2 * ((layer * (r + 1) + slot) * sw + i) ..][0..2], .little);
            if (got != want) return try std.fmt.bufPrint(&self.msg, "layer {d} slot {d} element {d}: 0x{x:0>4} where 0x{x:0>4}", .{ layer, slot, i, got, want });
        };
        return null;
    }
};

/// The kernels and their controls, registered onto `kernels`; the controls from the contract_controls image `ctl`.
pub const Set = struct {
    mod: ?cuda.Module = null,
    logits: Logits = undefined,
    select: [2]Select = undefined,
    conv: [2]Conv = undefined,
    commit: [2]Commit = undefined,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, ctl: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        const a = gpu.gpa;
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.lfm2_moe);
        const m = self.mod.?;
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        self.logits = .{ .rig = rig, .f = try m.function("tf_lfm2m_route") };
        for (&self.select, [_]bool{ false, true }) |*x, ctrl| x.* = .{ .rig = rig, .f = try (if (ctrl) ctl else m).function(if (ctrl) "tf_ctl_lfm2m_route" else "tf_lfm2m_route"), .control = ctrl };
        for (&self.conv, [_]bool{ false, true }) |*x, ctrl| x.* = .{ .rig = rig, .f = try (if (ctrl) ctl else m).function(if (ctrl) "tf_ctl_liquid_conv_3" else "tf_liquid_conv_3"), .control = ctrl };
        for (&self.commit, [_]bool{ false, true }) |*x, ctrl| x.* = .{ .rig = rig, .f = try (if (ctrl) ctl else m).function(if (ctrl) "tf_ctl_liquid_commit_3" else "tf_liquid_commit_3"), .control = ctrl };
        try kernels.append(a, self.logits.kernel());
        for (&self.select) |*x| try kernels.append(a, x.kernel());
        for (&self.conv) |*x| try kernels.append(a, x.kernel());
        for (&self.commit) |*x| try kernels.append(a, x.kernel());
    }

    pub fn deinit(self: *Set) void {
        const m = self.mod orelse return;
        self.logits.deinit();
        for (&self.select) |*x| x.deinit();
        for (&self.conv) |*x| x.deinit();
        for (&self.commit) |*x| x.deinit();
        var mm = m;
        mm.unload();
    }
};

test "the commit's streams: stream 1 keeps its slot, the rest take row 2k" {
    try std.testing.expectEqual(@as(i32, -1), Commit.rowOf(1));
    try std.testing.expectEqual(@as(i32, 6), Commit.rowOf(3));
    try std.testing.expectEqual(@as(usize, 514), quarterChain(.{ .rows = 1, .n = 64, .k = 2048 }));
}
