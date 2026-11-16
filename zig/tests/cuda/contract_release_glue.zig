//! DeepSeek V4.1's release glue under the contract on CUDA: the FP8 round trip, alone and after the reference RMSNorm.
const std = @import("std");
const core = @import("core");
const kr = @import("dsv41_release");
const ref = @import("contract_release_ref.zig");
const glue_ref = @import("dsv41_glue_ref");
const Rig = @import("contract_rig.zig").Rig;
const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Rows = contract.rows.Rows;

const rms_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const nan_rule = "a 32-value block holding a NaN or an infinity is NaN: codes 0x7f, scale 255, bf16 0x7fc0";

fn none(_: c.Shape) usize {
    return 0;
}

fn pow2(e: i32) f32 {
    return std.math.ldexp(@as(f32, 1), e);
}

/// Blocks that reach the round trip's edges: scale boundaries, ties between codes, zeros, tiny values and spikes.
pub fn fillBlocks(x: []u16, rng: std.Random) void {
    var b: usize = 0;
    while (b * 32 < x.len) : (b += 1) {
        const v = x[b * 32 ..][0..32];
        const e = rng.intRangeAtMost(i32, -8, 8);
        const mode = rng.uintLessThan(u32, 8);
        for (v, 0..) |*o, i| {
            const f: f32 = switch (mode) {
                0, 1 => (rng.float(f32) * 2 - 1) * pow2(rng.intRangeAtMost(i32, -10, 10)),
                2 => if (i % 7 == 3) -0.0 else 0.0,
                3 => (rng.float(f32) * 2 - 1) * 1e-6,
                4 => if (i == 0) 1e30 else (rng.float(f32) - 0.5) * 1e-3,
                5 => (rng.float(f32) * 2 - 1) * 448 * pow2(e),
                else => blk: {
                    const j = rng.uintLessThan(usize, 125);
                    const mid: f32 = @floatCast((ref.e4m3[j] + ref.e4m3[j + 1]) / 2);
                    break :blk if (i == 0) 448 * pow2(e) else if (rng.boolean()) -mid * pow2(e) else mid * pow2(e);
                },
            };
            o.* = c.toBf16(f);
        }
    }
}

/// NaN and both infinities at the rows family's places in [rows][k] bf16 rows.
pub fn plant(x: []u16, m: usize, k: usize) void {
    const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
    const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (at, put) |p, v| if (p[0] < m) {
        x[p[0] * k + p[1]] = c.toBf16(v);
    };
}

/// The oracle's codes [m][k], scales [m][k/32] and (rt) bf16 values [m][k]; the values also Metal's oracle's.
pub fn expected(a: std.mem.Allocator, x: []const u16, m: usize, k: usize, rt: bool, out: *std.ArrayList(u8)) ![]const u8 {
    const n = m * k;
    try out.resize(a, n + n / 32 + if (rt) 2 * n else 0);
    const codes = out.items[0..n];
    const scales = out.items[n..][0 .. n / 32];
    var vals: [32]u16 = undefined;
    var metal: [32]u16 = undefined;
    var xf: [32]f32 = undefined;
    for (0..n / 32) |b| {
        ref.actq(x[b * 32 ..][0..32], codes[b * 32 ..][0..32], &scales[b], &vals);
        for (&xf, x[b * 32 ..][0..32]) |*f, v| f.* = c.fromBf16(v);
        glue_ref.roundTrip(.fp8, &xf, &metal);
        if (!std.mem.eql(u16, &vals, &metal)) return error.OraclesDisagree;
        if (rt) @memcpy(out.items[n + n / 32 + 64 * b ..][0..64], std.mem.asBytes(&vals));
    }
    return out.items;
}

/// Compares the device's bytes with the oracle's; null when equal, else where they first differ.
fn verdict(buf: []u8, got: []const u8, want: []const u8, n: usize) ?[]const u8 {
    const at = std.mem.indexOfDiff(u8, got, want) orelse return null;
    const part = if (at < n) "codes" else if (at < n + n / 32) "scales" else "bf16 values";
    return std.fmt.bufPrint(buf, "{s} differ from the reference at byte {d} ({d} != {d})", .{ part, at, if (at < got.len) got[at] else 0, if (at < want.len) want[at] else 0 }) catch "differ";
}

/// tf_dsv41_rel_actq: shape (rows, 1, k), each byte equal to the reference round trip's.
pub const Actq = struct {
    rig: Rig,
    ops: kr.Ops,
    name: []const u8,
    view: bool,
    rt: bool,
    served: []const c.Shape,
    x: []u16 = &.{},
    shape: c.Shape = undefined,
    args: kr.Actq = undefined,
    want: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn kernel(self: *Actq) g.Kernel {
        return .{ .spec = .{ .name = self.name, .accum = .none, .depth = none, .out = .raw, .nan = .{ .stated = nan_rule }, .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 64, .step = 32, .min = 32 } }, .served = self.served }, .ctx = self, .vt = &vt };
    }

    const vt = blk: {
        var v = g.hooks(Actq);
        v.release = release;
        break :blk v;
    };

    fn release(ctx: *anyopaque) void {
        const self: *Actq = @ptrCast(@alignCast(ctx));
        self.rig.gpu.gpa.free(self.x);
        self.x = &.{};
        self.want.deinit(self.rig.gpu.gpa);
        self.want = .empty;
        self.rig.finishGates();
    }

    pub fn deinit(self: *Actq) void {
        release(self);
    }

    pub fn prepare(self: *Actq, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        self.x = &.{};
        self.rig.reset();
        if (shape.rows == 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        self.shape = shape;
        const m = shape.rows;
        const k = shape.k;
        self.x = try a.alloc(u16, m * k);
        var prng = std.Random.DefaultPrng.init(seed);
        fillBlocks(self.x, prng.random());
        if (fill == .nonfinite) plant(self.x, m, k);
        const ld = if (self.view) k + 64 else k;
        const off: usize = if (self.view) 64 else 0;
        const xb = try a.alloc(u16, off / 2 + m * ld);
        defer a.free(xb);
        @memset(xb, 0x7fc0);
        for (0..m) |r| @memcpy(xb[off / 2 + r * ld ..][0..k], self.x[r * k ..][0..k]);
        const bytes = std.mem.sliceAsBytes(xb);
        self.args = .{ .x = try self.rig.input(bytes), .x_off = off, .x_ld = @intCast(ld), .codes = try self.rig.output(m * k), .scales = try self.rig.output(m * k / 32), .m = @intCast(m), .k = @intCast(k) };
        if (self.rt) self.args.rt = try self.rig.output(m * k * 2);
        return true;
    }

    pub fn run(self: *Actq) !void {
        try self.ops.actq(self.args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Actq) usize {
        const n = self.shape.rows * self.shape.k;
        return n + n / 32 + if (self.rt) 2 * n else 0;
    }

    pub fn reference(_: *Actq, _: []f64, _: []f64) !void {
        return error.CheckedByBytes;
    }

    pub fn check(self: *Actq, _: g.Gate) !?[]const u8 {
        const want = try expected(self.rig.gpu.gpa, self.x, self.shape.rows, self.shape.k, self.rt, &self.want);
        return verdict(&self.msg, try self.rig.outputBytes(), want, self.shape.rows * self.shape.k);
    }
};

/// tf_dsv41_rel_rms_actq: `codes` checks the round trip of its own normed rows by bytes, else those rows by float64.
pub const Rms = struct {
    rig: Rig,
    ops: kr.Ops,
    name: []const u8,
    codes: bool,
    served: []const c.Shape,
    r: ?Rows = null,
    shape: c.Shape = undefined,
    args: kr.Rms = undefined,
    want: std.ArrayList(u8) = .empty,
    normed: std.ArrayList(u16) = .empty,
    msg: [160]u8 = undefined,

    fn depth(shape: c.Shape) usize {
        return shape.k / 256 + 14;
    }

    pub fn kernel(self: *Rms) g.Kernel {
        const tiles: [3]c.Tile = .{ .{ .size = 16 }, .{}, .{ .size = 256, .step = 32, .min = 32 } };
        if (self.codes) return .{ .spec = .{ .name = self.name, .accum = .none, .depth = none, .out = .raw, .nan = .{ .stated = nan_rule }, .tiles = tiles, .served = self.served }, .ctx = self, .vt = &codes_vt };
        return .{ .spec = .{ .name = self.name, .accum = .f32, .depth = depth, .ops = 6, .out = .bf16, .tiles = tiles, .served = self.served, .sums = .{ .bf16 = 0, .why = rms_why } }, .ctx = self, .vt = &normed_vt };
    }

    const codes_vt = blk: {
        var v = g.hooks(Rms);
        v.release = release;
        v.simulate = null;
        break :blk v;
    };
    const normed_vt = blk: {
        var v = g.hooks(Rms);
        v.release = release;
        v.check = null;
        break :blk v;
    };

    fn release(ctx: *anyopaque) void {
        const self: *Rms = @ptrCast(@alignCast(ctx));
        if (self.r) |r| r.deinit();
        self.r = null;
        self.want.deinit(self.rig.gpu.gpa);
        self.want = .empty;
        self.normed.deinit(self.rig.gpu.gpa);
        self.normed = .empty;
        self.rig.finishGates();
    }

    pub fn deinit(self: *Rms) void {
        release(self);
    }

    pub fn prepare(self: *Rms, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = self.rig.gpu.gpa;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        if (shape.rows == 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        self.shape = shape;
        const m = shape.rows;
        const k = shape.k;
        const r = try Rows.init(a, .rms, shape);
        self.r = r;
        r.fill(fill, seed);
        const ld = k + 64;
        const off: usize = 128;
        const xb = try a.alloc(u16, off / 2 + m * ld);
        defer a.free(xb);
        @memset(xb, 0x7fc0);
        for (0..m) |i| @memcpy(xb[off / 2 + i * ld ..][0..k], r.x[i * k ..][0..k]);
        const w = try a.alloc(f32, k);
        defer a.free(w);
        for (w, r.w) |*o, v| o.* = c.fromBf16(v);
        const n = m * k;
        self.args = .{ .x = try self.rig.input(std.mem.sliceAsBytes(xb)), .x_off = off, .x_ld = @intCast(ld), .w = try self.rig.input(std.mem.sliceAsBytes(w)), .codes = 0, .scales = 0, .eps = r.eps, .m = @intCast(m), .k = @intCast(k) };
        if (self.codes) {
            self.args.codes = try self.rig.output(n);
            self.args.scales = try self.rig.output(n / 32);
            self.args.rt = try self.rig.output(2 * n);
            self.args.normed = try self.rig.scratch(2 * n);
        } else {
            self.args.normed = try self.rig.output(2 * n);
            self.args.codes = try self.rig.scratch(n);
            self.args.scales = try self.rig.scratch(n / 32);
        }
        return true;
    }

    pub fn run(self: *Rms) !void {
        try self.ops.rmsActq(self.args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Rms) usize {
        const n = self.shape.rows * self.shape.k;
        return if (self.codes) n + n / 32 + 2 * n else n;
    }

    pub fn reference(self: *Rms, want: []f64, mag: []f64) !void {
        return self.r.?.reference(want, mag);
    }

    pub fn simulate(self: *Rms, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(acc, order, out);
    }

    pub fn check(self: *Rms, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const n = self.shape.rows * self.shape.k;
        try self.normed.resize(a, n);
        @memcpy(std.mem.sliceAsBytes(self.normed.items), try self.rig.bytes(self.args.normed, 2 * n));
        const want = try expected(a, self.normed.items, self.shape.rows, self.shape.k, true, &self.want);
        return verdict(&self.msg, try self.rig.outputBytes(), want, n);
    }
};
