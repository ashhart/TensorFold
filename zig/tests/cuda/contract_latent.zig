//! The latent MoE's glue over prompt share sums under the contract on CUDA: nemotron_latent.cu's _sums pair.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemotron = @import("nemotron");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

/// The combine over each pick's bf16 share sum, and the shared add over the shared expert's.
pub const Kind = enum { combine, shared_add };

const shares = 2;
const rule = "as the kernel it replaces gives them from the fp32 shares: NaN and infinite shares and weights";

/// A row's fma chain over its picks.
fn pickChain(s: c.Shape) usize {
    return s.k + 1;
}

fn oneAdd(_: c.Shape) usize {
    return 1;
}

/// Super's latent (1,024 wide, 22 picks a row) and hidden (4,096) rows at prompt passes.
pub fn spec(kind: Kind) c.Spec {
    return switch (kind) {
        .combine => .{ .name = "cuda nemotron latent combine tf_super_combine_sums (bf16 share sums)", .accum = .f32, .depth = pickChain, .out = .bf16, .replaces = "tf_super_combine", .nan = .{ .stated = rule }, .tiles = .{ .{}, .{ .size = 1024, .step = 8, .min = 8 }, .{} }, .served = &.{ .{ .rows = 129, .n = 1024, .k = 22 }, .{ .rows = 300, .n = 1024, .k = 22 } } },
        .shared_add => .{ .name = "cuda nemotron latent shared add tf_super_moe_out_sums (bf16 share sums)", .accum = .f32, .depth = oneAdd, .out = .bf16, .replaces = "tf_super_moe_out", .nan = .{ .stated = rule }, .tiles = .{ .{}, .{ .size = 2048, .step = 8, .min = 8 }, .{} }, .served = &.{ .{ .rows = 129, .n = 4096, .k = 1 }, .{ .rows = 300, .n = 4096, .k = 1 } }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "one fp32 add of two bf16 values a output" } },
    };
}

/// fp32 shares [groups][shares][n] and their bf16 sums, made by tf_super_combine at one pick of weight 1.
pub const Sums = struct {
    rig: rig_.Rig,
    kind: Kind,
    fns: *const nemotron.latent.Fns,
    shape: c.Shape = undefined,
    p: []f32 = &.{},
    w: []f32 = &.{}, // combine: [rows][picks]
    y: []u16 = &.{}, // shared add: the routed rows the shares add into
    q: []u16 = &.{}, // the bf16 sums, as the kernel reads them
    pd: u64 = 0,
    qd: u64 = 0,
    wd: u64 = 0,
    yd: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Sums) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Sums) void {
        const a = self.rig.gpu.gpa;
        a.free(self.p);
        a.free(self.w);
        a.free(self.y);
        a.free(self.q);
        self.p = &.{};
        self.w = &.{};
        self.y = &.{};
        self.q = &.{};
    }

    pub fn kernel(self: *Sums) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Sums);

    /// Share sums a run reads: a row's picks for combine, one a row for the shared add.
    fn groups(self: *const Sums) usize {
        return self.shape.rows * (if (self.kind == .combine) self.shape.k else 1);
    }

    fn launch(self: *Sums, f: cuda.Function, grid: [2]usize, block: u32, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]) }, .block = .{ .x = block } }, self.rig.s, a);
    }

    /// tf_super_combine over `rows` rows of `picks` picks from fp32 shares `p` [rows][picks][shares][n] into `out`.
    fn combine(self: *Sums, p: u64, w: u64, rows: usize, picks: usize, out: u64) !void {
        const n = self.shape.n;
        var a: cuda.Args = .{};
        a.add(p);
        for ([_]usize{ n, shares * n }) |v| a.add(@as(i64, @intCast(v)));
        a.add(@as(c_int, shares));
        a.add(w);
        a.add(@as(c_int, @intCast(picks)));
        a.add(out);
        a.add(@as(c_int, @intCast(n)));
        try self.launch(self.fns.combine, .{ rows, (n + 255) / 256 }, 256, &a);
    }

    /// Shares in [-2, 2); non-finite: NaN and infinite shares in rows 0-3, a NaN weight in row 4; stress: see `stress`.
    pub fn prepare(self: *Sums, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 8 != 0 or shape.k == 0 or shape.k > 64) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.p = try a.alloc(f32, self.groups() * shares * n);
        self.w = try a.alloc(f32, if (self.kind == .combine) self.groups() else 0);
        self.y = try a.alloc(u16, if (self.kind == .shared_add) shape.rows * n else 0);
        for (self.p) |*v| v.* = rng.float(f32) * 4 - 2;
        for (self.w) |*v| v.* = rng.float(f32);
        for (self.y) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        switch (fill) {
            .random => {},
            .nonfinite => {
                const per = shares * n * (self.groups() / shape.rows); // a row's shares
                const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (put, 0..) |v, r| if (r < shape.rows) {
                    self.p[r * per + 3] = v;
                };
                if (shape.rows > 3) {
                    self.p[3 * per + 5] = std.math.inf(f32);
                    self.p[3 * per + n + 5] = -std.math.inf(f32);
                }
                if (shape.rows > 4 and self.w.len > 0) self.w[4 * shape.k + 1] = std.math.nan(f32);
            },
            .stress => self.stress(rng),
        }
        self.pd = try self.rig.input(std.mem.sliceAsBytes(self.p));
        const ones = try a.alloc(f32, self.groups());
        defer a.free(ones);
        @memset(ones, 1);
        const zeros = try a.alloc(u8, self.groups() * n * 2);
        defer a.free(zeros);
        @memset(zeros, 0);
        self.qd = try self.rig.input(zeros);
        try self.combine(self.pd, try self.rig.input(std.mem.sliceAsBytes(ones)), self.groups(), 1, self.qd);
        self.q = try a.alloc(u16, self.groups() * n);
        @memcpy(std.mem.sliceAsBytes(self.q), try self.rig.bytes(self.qd, self.q.len * 2));
        if (self.kind == .combine) self.wd = try self.rig.input(std.mem.sliceAsBytes(self.w));
        if (self.kind == .shared_add) self.yd = try self.rig.input(std.mem.sliceAsBytes(self.y));
        self.out = try self.rig.output(shape.rows * n * 2);
        self.base = try self.rig.scratch(shape.rows * n * 2);
        return true;
    }

    /// Gate 4: share 0 bf16-exact in [2^15, 2^16), share 1 zero; combine's second half of picks mirrored at weight -1.
    fn stress(self: *Sums, rng: std.Random) void {
        const n = self.shape.n;
        @memset(self.p, 0);
        const picks = self.groups() / self.shape.rows;
        for (0..self.shape.rows) |r| for (0..picks) |k| for (0..n) |d| {
            const half = picks / 2;
            const at = ((r * picks + k) * shares) * n + d;
            if (self.kind == .shared_add or k < half) {
                self.p[at] = @floatFromInt(@as(u32, 128 + rng.uintLessThan(u32, 128)) << 8);
            } else {
                const mirror = ((r * picks + k - half) * shares) * n + d;
                self.p[at] = self.p[mirror] + @as(f32, if (k == half) 256 else 0);
            }
        };
        for (self.w, 0..) |*v, i| v.* = if (i % picks < picks / 2) 1 else -1;
        for (self.y) |*v| v.* = c.toBf16(@floatFromInt(@as(u32, 128 + rng.uintLessThan(u32, 128)) << 8));
    }

    pub fn run(self: *Sums) !void {
        const n = self.shape.n;
        const rows = self.shape.rows;
        var a: cuda.Args = .{};
        switch (self.kind) {
            .combine => {
                for ([_]u64{ self.qd, self.wd }) |v| a.add(v);
                a.add(@as(c_int, @intCast(self.shape.k)));
                a.add(self.out);
                a.add(@as(c_int, @intCast(n)));
                try self.launch(self.fns.combine_sums, .{ rows, (n + 1023) / 1024 }, 128, &a);
            },
            .shared_add => {
                try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.out, .len = rows * n * 2 }).copyFrom(0, self.yd, rows * n * 2, self.rig.s.handle);
                for ([_]u64{ self.out, self.qd }) |v| a.add(v);
                a.add(@as(c_int, @intCast(n)));
                try self.launch(self.fns.moe_out_sums, .{ rows, (n + 2047) / 2048 }, 256, &a);
            },
        }
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Sums) ![]const u8 {
        const n = self.shape.n;
        const rows = self.shape.rows;
        switch (self.kind) {
            .combine => try self.combine(self.pd, self.wd, rows, self.shape.k, self.base),
            .shared_add => {
                try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.base, .len = rows * n * 2 }).copyFrom(0, self.yd, rows * n * 2, self.rig.s.handle);
                var a: cuda.Args = .{};
                for ([_]u64{ self.base, self.pd }) |v| a.add(v);
                for ([_]usize{ n, shares * n }) |v| a.add(@as(i64, @intCast(v)));
                a.add(@as(c_int, shares));
                a.add(@as(c_int, @intCast(n)));
                try self.launch(self.fns.moe_out, .{ rows, (n + 255) / 256 }, 256, &a);
            },
        }
        return self.rig.bytes(self.base, rows * n * 2);
    }

    /// Gate 1: the replaced kernel's bytes, with some output non-finite to prove the planted values reach it.
    pub fn check(self: *Sums, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from {s}'s", .{ at, got.len, spec(self.kind).replaces.? });
        for (0..self.count()) |i| if (c.classOf(c.read(.bf16, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    pub fn count(self: *Sums) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Output o's terms: each pick's sum times its weight, or the routed value and the shared sum.
    fn terms(self: *const Sums, o: usize, t: []f64) void {
        const n = self.shape.n;
        const r = o / n;
        const d = o % n;
        if (self.kind == .shared_add) {
            t[0] = c.fromBf16(self.y[o]);
            t[1] = c.fromBf16(self.q[o]);
            return;
        }
        for (t, 0..) |*v, k| v.* = c.fromBf16(self.q[(r * self.shape.k + k) * n + d]) * self.w[r * self.shape.k + k];
    }

    fn termCount(self: *const Sums) usize {
        return if (self.kind == .combine) self.shape.k else 2;
    }

    pub fn reference(self: *Sums, want: []f64, mag: []f64) !void {
        var t: [64]f64 = undefined;
        const ts = t[0..self.termCount()];
        for (want, mag, 0..) |*w, *m, o| {
            self.terms(o, ts);
            w.* = 0;
            m.* = 0;
            for (ts) |v| {
                w.* += v;
                m.* += @abs(v);
            }
        }
    }

    pub fn simulate(self: *Sums, acc: c.Accum, order: g.Order, out: []f64) !void {
        var t: [64]f64 = undefined;
        const ts = t[0..self.termCount()];
        for (out, 0..) |*v, o| {
            self.terms(o, ts);
            v.* = contract.rows.sumIn(acc, order, ts);
        }
    }
};
