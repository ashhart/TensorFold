//! DeepSeek V4.1's release experts under the contract on CUDA: the down, the combine, the shared SwiGLU, the set.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const kr = @import("dsv41_release");
const xr = @import("contract_release_ex_ref.zig");
const ex = @import("contract_release_ex.zig");
const glue = @import("contract_release_glue.zig");
const Rig = @import("contract_rig.zig").Rig;
const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const threads = contract.reference_threads;
const kx = kr.ex;

fn none(_: c.Shape) usize {
    return 0;
}

/// tf_dsv41_rel_down: shape (rows, n, k); grouped pair j's codes are row j % uniq's; bf16 partials against float64.
pub const Down = struct {
    rig: Rig,
    ops: kx.Ops,
    topk: usize,
    nids: usize,
    nslots: usize,
    served: []const c.Shape,
    t: ?ex.Tables = null,
    x: []u8 = &.{},
    xs: []u8 = &.{},
    w: []u8 = &.{},
    ws: []u8 = &.{},
    args: kx.Down = undefined,
    shape: c.Shape = undefined,
    uniq: usize = 0,

    fn depth(shape: c.Shape) usize {
        return 2 * (shape.k / 32 + 32);
    }

    pub fn kernel(self: *Down) g.Kernel {
        return .{ .spec = .{ .name = "dsv41_rel_ex down, bf16 partials", .accum = .f32, .depth = depth, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{ .size = 64, .step = 32, .min = 32 }, .{ .size = 512, .step = 32, .min = 32 } }, .served = self.served }, .ctx = self, .vt = &vt };
    }

    const vt = blk: {
        var v = g.hooks(Down);
        v.release = release;
        break :blk v;
    };

    fn free(self: *Down) void {
        const a = self.rig.gpu.gpa;
        if (self.t) |t| t.deinit(a);
        self.t = null;
        inline for (.{ "x", "xs", "w", "ws" }) |f| {
            a.free(@field(self, f));
            @field(self, f) = &.{};
        }
    }

    fn release(ctx: *anyopaque) void {
        const self: *Down = @ptrCast(@alignCast(ctx));
        self.free();
        self.rig.finishGates();
    }

    pub fn deinit(self: *Down) void {
        release(self);
    }

    pub fn prepare(self: *Down, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = self.rig.gpu.gpa;
        self.free();
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.n % 32 != 0 or shape.k % 32 != 0) return false;
        self.shape = shape;
        const n = shape.n;
        const k = shape.k;
        self.uniq = @min(shape.rows, ex.uniq_rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const t = try ex.Tables.init(a, shape.rows, self.topk, self.nids, self.nslots, self.uniq, rng);
        self.t = t;
        const pairs = t.grouped.pairs;
        if (pairs == 0) return false;
        self.x = try a.alloc(u8, self.uniq * k);
        self.xs = try a.alloc(u8, self.uniq * k / 32);
        xr.fillCodes(self.x, self.xs, k, fill, rng);
        self.w = try a.alloc(u8, self.nslots * n * k / 2);
        self.ws = try a.alloc(u8, self.nslots * n * k / 32);
        xr.fillPacked(self.w, self.ws, k, fill, rng);
        if (fill == .nonfinite) {
            self.x[k / 3] = 0x7f;
            if (self.uniq > 1) self.xs[k / 32 + k / 64] = 255;
            self.ws[5 * (k / 32)] = 255;
        }
        const xb = try a.alloc(u8, pairs * k + pairs * k / 32);
        defer a.free(xb);
        for (0..pairs) |j| {
            @memcpy(xb[j * k ..][0..k], self.x[(j % self.uniq) * k ..][0..k]);
            @memcpy(xb[pairs * k + j * k / 32 ..][0 .. k / 32], self.xs[(j % self.uniq) * k / 32 ..][0 .. k / 32]);
        }
        const xd = try self.rig.input(xb);
        self.args = .{
            .x = xd,
            .xs = xd + pairs * k,
            .w2 = try self.rig.input(self.w),
            .s2 = try self.rig.input(self.ws),
            .offs = try self.rig.input(std.mem.sliceAsBytes(t.grouped.offs)),
            .active = try self.rig.input(std.mem.sliceAsBytes(t.grouped.active)),
            .nactive = try self.rig.input(std.mem.asBytes(&t.grouped.nactive)),
            .part = try self.rig.output(2 * pairs * n),
            .n = @intCast(n),
            .k = @intCast(k),
        };
        return true;
    }

    pub fn run(self: *Down) !void {
        try self.ops.down(self.args, self.shape.rows, self.topk, self.nslots, .served);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Down) usize {
        return self.t.?.grouped.pairs * self.shape.n;
    }

    /// The (row j % uniq, slot) dots each pair repeats, computed once over every column.
    const Job = struct {
        s: *Down,
        keys: [2]usize,
        used: []const bool,
        cache: []f64,
        acc: ?c.Accum,
        order: g.Order,
        failed: bool = false,

        pub fn run(job: *Job) void {
            job.work() catch {
                job.failed = true;
            };
        }

        fn work(job: *Job) !void {
            const s = job.s;
            const n = s.shape.n;
            const k = s.shape.k;
            const keys = job.used.len;
            const terms = try s.rig.gpu.gpa.alloc(f64, k);
            defer s.rig.gpu.gpa.free(terms);
            for (job.keys[0]..job.keys[1]) |key| if (job.used[key]) {
                const u = key / s.nslots;
                const slot = key % s.nslots;
                for (0..n) |col| {
                    const d = xr.dot(s.x, s.xs, u, s.w, s.ws, slot * n + col, k, terms, job.acc, job.order);
                    job.cache[key * n + col] = d[0];
                    job.cache[(keys + key) * n + col] = d[1];
                }
            };
        }
    };

    fn spread(self: *Down, want: []f64, mag: []f64, acc: ?c.Accum, order: g.Order) !void {
        const a = self.rig.gpu.gpa;
        const t = self.t.?;
        const n = self.shape.n;
        const keys = self.uniq * self.nslots;
        const used = try a.alloc(bool, keys);
        defer a.free(used);
        @memset(used, false);
        const keyOf = struct {
            fn f(s: *Down, gr: xr.Grouped, j: usize) usize {
                var slot: usize = 0;
                while (gr.offs[slot + 1] <= j) slot += 1;
                return (j % s.uniq) * s.nslots + slot;
            }
        }.f;
        for (0..t.grouped.pairs) |j| used[keyOf(self, t.grouped, j)] = true;
        const cache = try a.alloc(f64, 2 * keys * n);
        defer a.free(cache);
        var jobs: [threads.maximum]Job = undefined;
        const jn = threads.count(threads.maximum, keys);
        for (jobs[0..jn], 0..) |*job, i| job.* = .{ .s = self, .keys = .{ keys * i / jn, keys * (i + 1) / jn }, .used = used, .cache = cache, .acc = acc, .order = order };
        threads.run(Job, jobs[0..jn], .normal);
        for (jobs[0..jn]) |job| if (job.failed) return error.OutOfMemory;
        for (0..t.grouped.pairs) |j| {
            const key = keyOf(self, t.grouped, j);
            @memcpy(want[j * n ..][0..n], cache[key * n ..][0..n]);
            @memcpy(mag[j * n ..][0..n], cache[(keys + key) * n ..][0..n]);
        }
    }

    pub fn reference(self: *Down, want: []f64, mag: []f64) !void {
        return self.spread(want, mag, null, .sequential);
    }

    pub fn simulate(self: *Down, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.rig.gpu.gpa.alloc(f64, out.len);
        defer self.rig.gpu.gpa.free(mag);
        return self.spread(out, mag, acc, order);
    }
};

/// tf_dsv41_rel_combine: shape (rows, n, topk); each row's held partials in k order, then its shared row, in FP32.
pub const Combine = struct {
    rig: Rig,
    ops: kx.Ops,
    name: []const u8,
    f32_out: bool,
    shared: bool,
    /// Every id held, outputs byte-equal to the FP32 order replayed on the host, and the cancellation fixture.
    exact: bool = false,
    nids: usize,
    served: []const c.Shape,
    t: ?ex.Tables = null,
    part: []u16 = &.{},
    rows: []u16 = &.{},
    args: kx.Combine = undefined,
    shape: c.Shape = undefined,
    want: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,
    cancel: bool = false,

    fn depth(shape: c.Shape) usize {
        return shape.k + 1;
    }

    pub fn kernel(self: *Combine) g.Kernel {
        return .{ .spec = .{ .name = self.name, .accum = .f32, .depth = depth, .out = if (self.f32_out) .f32 else .bf16, .tiles = .{ .{ .size = 16 }, .{ .size = 256, .step = 32, .min = 32 }, .{} }, .served = self.served }, .ctx = self, .vt = if (self.exact) &exact_vt else &vt };
    }

    const vt = blk: {
        var v = g.hooks(Combine);
        v.release = release;
        v.check = null;
        break :blk v;
    };
    const exact_vt = blk: {
        var v = g.hooks(Combine);
        v.release = release;
        break :blk v;
    };

    /// The host's FP32 replay: held partials in k order from zero, then the shared row, then the output rounding.
    pub fn check(self: *Combine, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const m = self.shape.rows;
        const n = self.shape.n;
        const es: usize = if (self.f32_out) 4 else 2;
        try self.want.resize(a, m * n * es);
        var buf: [64]f64 = undefined;
        for (0..m * n) |i| {
            var acc: f32 = 0;
            for (self.terms(i / n, i % n, &buf)) |v| acc = acc + @as(f32, @floatCast(v));
            if (self.f32_out) {
                std.mem.writeInt(u32, self.want.items[4 * i ..][0..4], @bitCast(acc), .little);
            } else std.mem.writeInt(u16, self.want.items[2 * i ..][0..2], c.toBf16(acc), .little);
        }
        const got = try self.rig.outputBytes();
        const kind: c.Elem = if (self.f32_out) .f32 else .bf16;
        for (0..m * n) |i| {
            const same = std.mem.eql(u8, got[i * es ..][0..es], self.want.items[i * es ..][0..es]);
            const nan = std.math.isNan(c.read(kind, got, i)) and std.math.isNan(c.read(kind, self.want.items, i));
            if (!same and !nan) return std.fmt.bufPrint(&self.msg, "output {d} is {e} where the FP32 order gives {e}", .{ i, c.read(kind, got, i), c.read(kind, self.want.items, i) }) catch "differs";
        }
        if (self.cancel) {
            const v = c.read(if (self.f32_out) .f32 else .bf16, got, 0);
            if (v != 1) return std.fmt.bufPrint(&self.msg, "routed [2^24, 1, -2^24] then shared 1 gives {d}, not 1", .{v}) catch "cancellation";
        }
        return null;
    }

    fn free(self: *Combine) void {
        const a = self.rig.gpu.gpa;
        if (self.t) |t| t.deinit(a);
        self.t = null;
        a.free(self.part);
        a.free(self.rows);
        self.part = &.{};
        self.rows = &.{};
    }

    fn release(ctx: *anyopaque) void {
        const self: *Combine = @ptrCast(@alignCast(ctx));
        self.free();
        self.want.deinit(self.rig.gpu.gpa);
        self.want = .empty;
        self.rig.finishGates();
    }

    pub fn deinit(self: *Combine) void {
        release(self);
    }

    pub fn prepare(self: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = self.rig.gpu.gpa;
        self.free();
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k > self.nids) return false;
        self.shape = shape;
        const m = shape.rows;
        const n = shape.n;
        const topk = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const t = try ex.Tables.init(a, m, topk, self.nids, if (fill == .stress or self.exact) self.nids else self.nids / 4, m, rng);
        self.t = t;
        const pairs = t.grouped.pairs;
        self.part = try a.alloc(u16, @max(1, pairs) * n);
        self.rows = try a.alloc(u16, m * n);
        for (self.part) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * std.math.ldexp(@as(f32, 1), rng.intRangeAtMost(i32, -6, 6)));
        for (self.rows) |*v| v.* = if (self.shared) c.toBf16(rng.float(f32) * 2 - 1) else 0;
        if (fill == .stress) for (0..m) |r| {
            const h = topk / 2;
            for (0..h) |kk| {
                for (0..n) |col| {
                    const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 255));
                    const lo: usize = @intCast(t.grouped.pair_of[r * topk + kk]);
                    const hi: usize = @intCast(t.grouped.pair_of[r * topk + h + kk]);
                    self.part[lo * n + col] = c.toBf16(v * 4096);
                    self.part[hi * n + col] = c.toBf16(-(v - if (kk == 0) @as(f32, 1) else 0) * 4096);
                }
            }
            for (self.rows[r * n ..][0..n], 0..) |*v, col| v.* = if (self.shared) c.toBf16(@as(f32, @floatFromInt(1 + (r + col) % 3)) * 4096) else 0;
        };
        // the cancellation fixture: row 0, column 0 takes routed [2^24, 1, -2^24] (k order) and shared 1
        self.cancel = self.exact and self.shared and topk >= 3 and fill == .random;
        if (self.cancel) {
            for ([_]f32{ 0x1p24, 1, -0x1p24 }, 0..) |v, kk| self.part[@as(usize, @intCast(t.grouped.pair_of[kk])) * n] = c.toBf16(v);
            for (3..topk) |kk| self.part[@as(usize, @intCast(t.grouped.pair_of[kk])) * n] = 0;
            self.rows[0] = c.toBf16(1);
        }
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, i| if (i < pairs) {
                self.part[i * n + (i * 7) % n] = c.toBf16(v);
            };
            if (self.shared and m > 1) self.rows[n + n / 2] = c.toBf16(std.math.inf(f32));
        }
        const es: usize = if (self.f32_out) 4 else 2;
        self.args = .{
            .part = try self.rig.input(std.mem.sliceAsBytes(self.part)),
            .pair_of = try self.rig.input(std.mem.sliceAsBytes(t.grouped.pair_of)),
            .shared = if (self.shared) try self.rig.input(std.mem.sliceAsBytes(self.rows)) else 0,
            .shared_ld = @intCast(n),
            .y = try self.rig.output(m * n * es),
            .y_ld = @intCast(n),
            .m = @intCast(m),
            .n = @intCast(n),
            .topk = @intCast(topk),
            .f32 = @intFromBool(self.f32_out),
        };
        return true;
    }

    pub fn run(self: *Combine) !void {
        try self.ops.combine(self.args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Combine) usize {
        return self.shape.rows * self.shape.n;
    }

    fn terms(self: *Combine, r: usize, col: usize, t: []f64) []f64 {
        const n = self.shape.n;
        const topk = self.shape.k;
        var at: usize = 0;
        for (0..topk) |kk| {
            const j = self.t.?.grouped.pair_of[r * topk + kk];
            if (j < 0) continue;
            t[at] = c.fromBf16(self.part[@as(usize, @intCast(j)) * n + col]);
            at += 1;
        }
        if (self.shared) {
            t[at] = c.fromBf16(self.rows[r * n + col]);
            at += 1;
        }
        return t[0..at];
    }

    pub fn reference(self: *Combine, want: []f64, mag: []f64) !void {
        var buf: [64]f64 = undefined;
        for (want, mag, 0..) |*w, *m, i| {
            const t = self.terms(i / self.shape.n, i % self.shape.n, &buf);
            w.* = 0;
            m.* = 0;
            for (t) |v| {
                w.* += v;
                m.* += @abs(v);
            }
        }
    }

    pub fn simulate(self: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        var buf: [64]f64 = undefined;
        for (out, 0..) |*o, i| o.* = contract.rows.sumIn(acc, order, self.terms(i / self.shape.n, i % self.shape.n, &buf));
    }
};
