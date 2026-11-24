//! DeepSeek V4.1's release experts under the contract on CUDA (sm_121a): the grouping and the routed gate/up.
const std = @import("std");
const core = @import("core");
const kr = @import("dsv41_release");
const xr = @import("contract_release_ex_ref.zig");
const glue = @import("contract_release_glue.zig");
const Rig = @import("contract_rig.zig").Rig;
const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const threads = contract.reference_threads;
const kx = kr.ex;

/// Distinct rows a routed call holds: row r repeats row r % uniq_rows's codes and ids.
pub const uniq_rows = 19;

fn none(_: c.Shape) usize {
    return 0;
}

/// A routed problem's device tables: ids, slot_of, and the grouping (offs, pair_tok, pair_of, active, nactive).
pub const Tables = struct {
    routes: xr.Routes,
    grouped: xr.Grouped,

    pub fn init(a: std.mem.Allocator, m: usize, topk: usize, nids: usize, nslots: usize, uniq: usize, rng: std.Random) !Tables {
        const routes = try xr.Routes.init(a, m, topk, nids, nslots, uniq, rng);
        errdefer routes.deinit(a);
        return .{ .routes = routes, .grouped = try xr.Grouped.init(a, routes) };
    }

    pub fn deinit(t: Tables, a: std.mem.Allocator) void {
        t.routes.deinit(a);
        t.grouped.deinit(a);
    }
};

/// tf_dsv41_rel_group: shape (rows, nids, topk); every word equal to the host's (slot, row, k) bucketing.
pub const Group = struct {
    rig: Rig,
    ops: kx.Ops,
    name: []const u8,
    /// Slots the rank holds out of nids (nids: every expert).
    held: enum { all, quarter },
    served: []const c.Shape,
    t: ?Tables = null,
    args: kx.Group = undefined,
    want: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn kernel(self: *Group) g.Kernel {
        return .{ .spec = .{ .name = self.name, .accum = .none, .depth = none, .out = .raw, .nan = .{ .stated = "integer ids: unheld or out-of-range ids are skipped" }, .tiles = .{ .{ .size = 1024 }, .{}, .{} }, .served = self.served }, .ctx = self, .vt = &vt };
    }

    const vt = blk: {
        var v = g.hooks(Group);
        v.release = release;
        break :blk v;
    };

    fn release(ctx: *anyopaque) void {
        const self: *Group = @ptrCast(@alignCast(ctx));
        if (self.t) |t| t.deinit(self.rig.gpu.gpa);
        self.t = null;
        self.want.deinit(self.rig.gpu.gpa);
        self.want = .empty;
        self.rig.finishGates();
    }

    pub fn deinit(self: *Group) void {
        release(self);
    }

    pub fn prepare(self: *Group, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        const a = self.rig.gpu.gpa;
        if (self.t) |t| t.deinit(a);
        self.t = null;
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k > shape.n) return false;
        var prng = std.Random.DefaultPrng.init(seed);
        const nslots = if (self.held == .all) shape.n else shape.n / 4;
        const t = try Tables.init(a, shape.rows, shape.k, shape.n, nslots, shape.rows, prng.random());
        self.t = t;
        const pairs = shape.rows * shape.k;
        self.args = .{
            .ids = try self.rig.input(std.mem.sliceAsBytes(t.routes.ids)),
            .slot_of = try self.rig.input(std.mem.sliceAsBytes(t.routes.slot_of)),
            .offs = try self.rig.output(4 * (nslots + 1)),
            .pair_tok = try self.rig.output(4 * pairs),
            .pair_of = try self.rig.output(4 * pairs),
            .active = try self.rig.output(4 * (nslots + 1)),
            .nactive = try self.rig.output(4),
            .m = @intCast(shape.rows),
            .topk = @intCast(shape.k),
            .nslots = @intCast(nslots),
            .nids = @intCast(shape.n),
        };
        return true;
    }

    pub fn run(self: *Group) !void {
        try self.ops.group(self.args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Group) usize {
        const t = self.t.?;
        return 4 * (2 * (t.routes.nslots + 1) + 2 * t.routes.m * t.routes.topk + 1);
    }

    pub fn reference(_: *Group, _: []f64, _: []f64) !void {
        return error.CheckedByBytes;
    }

    pub fn check(self: *Group, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const gr = self.t.?.grouped;
        self.want.clearRetainingCapacity();
        for ([_][]const i32{ gr.offs, gr.pair_tok, gr.pair_of, gr.active, &.{gr.nactive} }) |part| try self.want.appendSlice(a, std.mem.sliceAsBytes(part));
        const got = try self.rig.outputBytes();
        const at = std.mem.indexOfDiff(u8, got, self.want.items) orelse return null;
        return std.fmt.bufPrint(&self.msg, "grouping word {d} is {d} where the host's is {d}", .{ at / 4, std.mem.bytesAsValue(i32, got[at / 4 * 4 ..][0..4]).*, std.mem.bytesAsValue(i32, self.want.items[at / 4 * 4 ..][0..4]).* }) catch "differs";
    }
};

/// tf_dsv41_rel_gateup (rows, n, k): `sums` against float64, `v` in its sums' SwiGLU range, `codes` by bytes.
pub const GateUp = struct {
    rig: Rig,
    ops: kx.Ops,
    name: []const u8,
    part: enum { sums, v, codes },
    topk: usize,
    nids: usize,
    nslots: usize,
    served: []const c.Shape,
    t: ?Tables = null,
    x: []u8 = &.{},
    xs: []u8 = &.{},
    w: [4][]u8 = .{ &.{}, &.{}, &.{}, &.{} },
    args: kx.GateUp = undefined,
    shape: c.Shape = undefined,
    uniq: usize = 0,
    want: std.ArrayList(u8) = .empty,
    vbits: std.ArrayList(u16) = .empty,
    msg: [200]u8 = undefined,

    fn depth(shape: c.Shape) usize {
        return 2 * (shape.k / 32 + 32);
    }

    pub fn kernel(self: *GateUp) g.Kernel {
        const tiles: [3]c.Tile = .{ .{ .size = 16 }, .{ .size = 64, .step = 32, .min = 32 }, .{ .size = 512, .step = 32, .min = 32 } };
        if (self.part == .sums) return .{ .spec = .{ .name = self.name, .accum = .f32, .depth = depth, .out = .f32, .tiles = tiles, .served = self.served }, .ctx = self, .vt = &sums_vt };
        return .{ .spec = .{ .name = self.name, .accum = .none, .depth = none, .out = .raw, .nan = .{ .stated = "v takes its FP32 steps' class; a non-finite v makes its 32-column block NaN" }, .tiles = tiles, .served = self.served }, .ctx = self, .vt = &check_vt };
    }

    const sums_vt = blk: {
        var v = g.hooks(GateUp);
        v.release = release;
        v.check = null;
        break :blk v;
    };
    const check_vt = blk: {
        var v = g.hooks(GateUp);
        v.release = release;
        v.simulate = null;
        break :blk v;
    };

    fn free(self: *GateUp) void {
        const a = self.rig.gpu.gpa;
        if (self.t) |t| t.deinit(a);
        self.t = null;
        a.free(self.x);
        a.free(self.xs);
        for (&self.w) |*w| {
            a.free(w.*);
            w.* = &.{};
        }
        self.x = &.{};
        self.xs = &.{};
    }

    fn release(ctx: *anyopaque) void {
        const self: *GateUp = @ptrCast(@alignCast(ctx));
        self.free();
        self.want.deinit(self.rig.gpu.gpa);
        self.want = .empty;
        self.vbits.deinit(self.rig.gpu.gpa);
        self.vbits = .empty;
        self.rig.finishGates();
    }

    pub fn deinit(self: *GateUp) void {
        release(self);
    }

    pub fn prepare(self: *GateUp, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = self.rig.gpu.gpa;
        self.free();
        self.rig.reset();
        if (shape.rows == 0 or shape.n % 32 != 0 or shape.k % 32 != 0 or shape.n == 0 or shape.k == 0) return false;
        self.shape = shape;
        const m = shape.rows;
        const n = shape.n;
        const k = shape.k;
        self.uniq = @min(m, uniq_rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const t = try Tables.init(a, m, self.topk, self.nids, self.nslots, self.uniq, rng);
        self.t = t;
        if (t.grouped.pairs == 0) return false;
        self.x = try a.alloc(u8, self.uniq * k);
        self.xs = try a.alloc(u8, self.uniq * k / 32);
        xr.fillCodes(self.x, self.xs, k, fill, rng);
        for (0..2) |i| {
            self.w[2 * i] = try a.alloc(u8, self.nslots * n * k / 2);
            self.w[2 * i + 1] = try a.alloc(u8, self.nslots * n * k / 32);
            xr.fillPacked(self.w[2 * i], self.w[2 * i + 1], k, fill, rng);
        }
        if (fill == .nonfinite) {
            self.x[k / 3] = 0x7f;
            if (self.uniq > 1) self.xs[k / 32 + 3] = 255;
            self.w[1][3 * (k / 32)] = 255;
        }
        const xb = try a.alloc(u8, m * k + m * k / 32);
        defer a.free(xb);
        for (0..m) |r| {
            @memcpy(xb[r * k ..][0..k], self.x[(r % self.uniq) * k ..][0..k]);
            @memcpy(xb[m * k + r * k / 32 ..][0 .. k / 32], self.xs[(r % self.uniq) * k / 32 ..][0 .. k / 32]);
        }
        const xd = try self.rig.input(xb);
        const gr = t.grouped;
        const pn = gr.pairs * n;
        self.args = .{
            .x = xd,
            .x_ld = @intCast(k),
            .xs = xd + m * k,
            .xs_ld = @intCast(k / 32),
            .w1 = try self.rig.input(self.w[0]),
            .s1 = try self.rig.input(self.w[1]),
            .w3 = try self.rig.input(self.w[2]),
            .s3 = try self.rig.input(self.w[3]),
            .wts = try self.rig.input(std.mem.sliceAsBytes(t.routes.wts)),
            .pair_tok = try self.rig.input(std.mem.sliceAsBytes(gr.pair_tok)),
            .offs = try self.rig.input(std.mem.sliceAsBytes(gr.offs)),
            .active = try self.rig.input(std.mem.sliceAsBytes(gr.active)),
            .nactive = try self.rig.input(std.mem.asBytes(&gr.nactive)),
            .codes = 0,
            .scales = 0,
            .n = @intCast(n),
            .k = @intCast(k),
            .topk = @intCast(self.topk),
        };
        const out = self.part;
        self.args.gsum = if (out == .sums) try self.rig.output(4 * pn) else try self.rig.scratch(4 * pn);
        self.args.usum = if (out == .sums) try self.rig.output(4 * pn) else try self.rig.scratch(4 * pn);
        self.args.v = if (out == .v) try self.rig.output(2 * pn) else try self.rig.scratch(2 * pn);
        self.args.codes = if (out == .codes) try self.rig.output(pn) else try self.rig.scratch(pn);
        self.args.scales = if (out == .codes) try self.rig.output(pn / 32) else try self.rig.scratch(pn / 32);
        return true;
    }

    pub fn run(self: *GateUp) !void {
        try self.ops.gateUp(self.args, self.shape.rows, self.nslots, .served);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *GateUp) usize {
        const pn = self.t.?.grouped.pairs * self.shape.n;
        return switch (self.part) {
            .sums => 2 * pn,
            .v => 2 * pn,
            .codes => pn + pn / 32,
        };
    }

    /// The dots (gate, up) of distinct keys (row % uniq, k) over every column; grouped pairs repeat them.
    const Job = struct {
        s: *GateUp,
        keys: [2]usize,
        want: []f64,
        mag: []f64,
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
            const t = s.t.?;
            const n = s.shape.n;
            const k = s.shape.k;
            const terms = try s.rig.gpu.gpa.alloc(f64, k);
            defer s.rig.gpu.gpa.free(terms);
            for (job.keys[0]..job.keys[1]) |key| {
                const slot = t.routes.slot(key);
                if (slot < 0) continue;
                for (0..n) |col| for (0..2) |b| {
                    const d = xr.dot(s.x, s.xs, key / s.topk, s.w[2 * b], s.w[2 * b + 1], @as(usize, @intCast(slot)) * n + col, k, terms, job.acc, job.order);
                    job.want[(key * 2 + b) * n + col] = d[0];
                    job.mag[(key * 2 + b) * n + col] = d[1];
                };
            }
        }
    };

    fn spread(self: *GateUp, want: []f64, mag: []f64, acc: ?c.Accum, order: g.Order) !void {
        const a = self.rig.gpu.gpa;
        const t = self.t.?;
        const n = self.shape.n;
        const keys = self.uniq * self.topk;
        const cache = try a.alloc(f64, 4 * keys * n);
        defer a.free(cache);
        var jobs: [threads.maximum]Job = undefined;
        const jn = threads.count(threads.maximum, keys);
        for (jobs[0..jn], 0..) |*job, i| job.* = .{ .s = self, .keys = .{ keys * i / jn, keys * (i + 1) / jn }, .want = cache[0 .. 2 * keys * n], .mag = cache[2 * keys * n ..], .acc = acc, .order = order };
        threads.run(Job, jobs[0..jn], .normal);
        for (jobs[0..jn]) |job| if (job.failed) return error.OutOfMemory;
        const pn = t.grouped.pairs * n;
        for (0..t.grouped.pairs) |j| {
            const tok: usize = @intCast(t.grouped.pair_tok[j]);
            const key = ((tok / self.topk) % self.uniq) * self.topk + tok % self.topk;
            for (0..2) |b| for (0..n) |col| {
                want[b * pn + j * n + col] = cache[(key * 2 + b) * n + col];
                mag[b * pn + j * n + col] = cache[2 * keys * n + (key * 2 + b) * n + col];
            };
        }
    }

    pub fn reference(self: *GateUp, want: []f64, mag: []f64) !void {
        return self.spread(want, mag, null, .sequential);
    }

    pub fn simulate(self: *GateUp, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.rig.gpu.gpa.alloc(f64, out.len);
        defer self.rig.gpu.gpa.free(mag);
        return self.spread(out, mag, acc, order);
    }

    pub fn check(self: *GateUp, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const t = self.t.?;
        const n = self.shape.n;
        const pairs = t.grouped.pairs;
        const pn = pairs * n;
        try self.vbits.resize(a, pn);
        @memcpy(std.mem.sliceAsBytes(self.vbits.items), try self.rig.bytes(self.args.v, 2 * pn));
        if (self.part == .codes) {
            const want = try glue.expected(a, self.vbits.items, pairs, n, false, &self.want);
            return glue.verdict(&self.msg, try self.rig.outputBytes(), want, pn);
        }
        const sums = try a.alloc(f32, 2 * pn);
        defer a.free(sums);
        @memcpy(std.mem.sliceAsBytes(sums[0..pn]), try self.rig.bytes(self.args.gsum, 4 * pn));
        @memcpy(std.mem.sliceAsBytes(sums[pn..]), try self.rig.bytes(self.args.usum, 4 * pn));
        for (0..pairs) |j| {
            const route = t.routes.wts[@intCast(t.grouped.pair_tok[j])];
            for (0..n) |col| {
                const i = j * n + col;
                const got = self.vbits.items[i];
                const ok = if (xr.swigluRange(sums[i], sums[pn + i], route, true)) |range| xr.inSpan(got, range) else c.classOf(c.fromBf16(got)) == c.classOf(xr.swigluIeee(sums[i], sums[pn + i], route, true));
                if (!ok) return std.fmt.bufPrint(&self.msg, "pair {d} column {d}: v {e} outside the SwiGLU range of g {e}, u {e}", .{ j, col, c.fromBf16(got), sums[i], sums[pn + i] }) catch "v outside its range";
            }
        }
        return null;
    }
};
