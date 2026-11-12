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

const score_rule = "picks by fp32 sigmoid(logit) + bias, larger first, ties to the lower id, a NaN never picked; weights the score";

/// A routing's fixed form; shape: rows, experts (n), top-k (k). score: sigmoid + bias ranked (plant: the logit form).
pub const Route = struct { name: []const u8, shared: bool, normalize: bool = false, scale: f32 = 1, warp: bool, score: bool = false, shared_slots: usize = 1, plant: bool = false, groups: usize = 1, keep: usize = 1, experts: usize = 0, high: bool = false }; // high: the highest of equal groups kept (planted)

const nemotron: []const c.Shape = &.{ .{ .rows = 16, .n = 128, .k = 6 }, .{ .rows = 300, .n = 128, .k = 6 } };
const bolt: []const c.Shape = &.{ .{ .rows = 16, .n = 496, .k = 9 }, .{ .rows = 300, .n = 496, .k = 9 } };
const ling: []const c.Shape = &.{ .{ .rows = 16, .n = 512, .k = 8 }, .{ .rows = 300, .n = 512, .k = 8 } };
const groups256: []const c.Shape = &.{ .{ .rows = 16, .n = 256, .k = 8 }, .{ .rows = 300, .n = 256, .k = 8 } };

pub const routes = [_]Route{
    .{ .name = "cuda moe top-k tf_moe_topk_sigmoid_warp (Kolibri: 384 experts, top 6 + shared)", .shared = true, .warp = true },
    .{ .name = "cuda moe top-k tf_moe_topk_sigmoid_warp (256 experts, top 8 normalized x2.5)", .shared = false, .normalize = true, .scale = 2.5, .warp = true },
    .{ .name = "cuda moe top-k tf_moe_topk_sigmoid (block form, 513 to 4,096 experts)", .shared = true, .warp = false },
    .{ .name = "cuda moe top-k tf_moe_topk_sigmoid_score_warp_128 (Nemotron: 128 experts, top 6 normalized x2.5 + 2 shared)", .shared = true, .shared_slots = 2, .normalize = true, .scale = 2.5, .warp = true, .score = true },
    .{ .name = "cuda moe top-k tf_moe_topk_sigmoid_score_warp (Bolt: 496 experts, top 9 normalized x2.5)", .shared = false, .normalize = true, .scale = 2.5, .warp = true, .score = true },
    .{ .name = "control: score top-k ranking logit + bias (tf_moe_topk_sigmoid_warp)", .shared = false, .normalize = true, .scale = 2.5, .warp = true, .score = true, .plant = true },
    .{ .name = "cuda moe top-k tf_moe_topk_sigmoid_score_groups (Ling-3.1: 512 experts, 4 of 8 groups, top 8 normalized x2.5 + shared)", .shared = true, .normalize = true, .scale = 2.5, .warp = true, .score = true, .groups = 8, .keep = 4, .experts = 512 },
    .{ .name = "cuda moe top-k tf_moe_topk_sigmoid_score_groups (256 experts, 4 of 8 groups, top 8 normalized x2.5 + shared)", .shared = true, .normalize = true, .scale = 2.5, .warp = true, .score = true, .groups = 8, .keep = 4, .experts = 256 },
    .{ .name = "control: group-limited top-k ranking every group (tf_moe_topk_sigmoid_score_warp)", .shared = true, .normalize = true, .scale = 2.5, .warp = true, .score = true, .plant = true, .groups = 8, .keep = 4, .experts = 512 },
    .{ .name = "control: group-limited top-k keeping the highest of equal groups (tf_ctl_moe_topk_score_groups_high)", .shared = true, .normalize = true, .scale = 2.5, .warp = true, .score = true, .groups = 8, .keep = 4, .experts = 512, .high = true },
};

pub fn topkSpec(r: Route) c.Spec {
    const served: []const c.Shape = if (r.groups > 1) (if (r.experts == 512) ling else groups256) else if (r.score) (if (r.shared or r.plant) nemotron else bolt) else if (!r.warp) &.{ .{ .rows = 16, .n = 4096, .k = 64 }, .{ .rows = 33, .n = 513, .k = 6 } } else if (r.normalize) &.{ .{ .rows = 16, .n = 256, .k = 8 }, .{ .rows = 300, .n = 256, .k = 8 } } else &.{ .{ .rows = 16, .n = 384, .k = 6 }, .{ .rows = 300, .n = 384, .k = 6 } };
    const tiles: [3]c.Tile = if (r.warp) .{ .{ .size = 8 }, .{ .size = 32, .min = 8 }, .{} } else .{ .{}, .{ .size = 256, .min = 513 }, .{} };
    return .{ .name = r.name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = if (r.score) score_rule else pick_rule }, .tiles = tiles, .served = served };
}

pub const TopK = struct {
    rig: rig_.Rig,
    route: Route,
    g: *const glue.MoeGlue,
    ctl: ?glue.MoeGlue = null, // high: g with its group kernel swapped for the planted one
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    logits: std.ArrayList(f32) = .empty,
    bias: std.ArrayList(f32) = .empty,
    dev: [4]u64 = @splat(0),
    msg: [200]u8 = undefined,
    fill: g.Fill = .random,
    near: usize = 0, // the last check's rows skipped as near-tied group cuts
    ties: usize = 0, // and its rows whose group cut is an exact tie, checked

    pub fn deinit(self: *TopK) void {
        self.logits.deinit(self.rig.gpu.gpa);
        self.bias.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *TopK) g.Kernel {
        return .{ .spec = topkSpec(self.route), .ctx = self, .vt = &vt, .control = if (self.route.plant or self.route.high) .shapes else null };
    }

    const vt = g.hooks(TopK);

    fn fns(self: *const TopK) *const glue.MoeGlue {
        return if (self.ctl) |*m| m else self.g;
    }

    fn slots(self: *const TopK) usize {
        return self.shape.k + if (self.route.shared) self.route.shared_slots else 0;
    }

    fn form(self: *const TopK) glue.Route {
        const r = self.route;
        const grouped = r.plant and r.groups > 1; // the group plant runs the ungrouped score form
        return .{ .experts = self.shape.n, .topk = self.shape.k, .shared = r.shared, .normalize = r.normalize, .scale = r.scale, .score = r.score and (!r.plant or grouped), .shared_slots = r.shared_slots, .groups = if (r.plant) 1 else r.groups, .keep = if (r.plant) 1 else r.keep };
    }

    pub fn prepare(self: *TopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        const warp = shape.n <= 512 and shape.k <= 32;
        if (warp != self.route.warp or shape.rows == 0 or shape.k == 0 or shape.k > shape.n or shape.n > glue.most_experts or shape.k > glue.most_topk) return false;
        const gr = self.route.groups;
        if (gr > 1 and (shape.n % gr != 0 or shape.n / gr < 2 or shape.k > self.route.keep * (shape.n / gr))) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        self.fill = fill;
        if (self.route.high and self.ctl == null) {
            var m = self.g.*;
            m.topk_groups_fn = try self.g.mod.function("tf_ctl_moe_topk_score_groups_high");
            self.ctl = m;
        }
        const ne = shape.n;
        const gs = ne / gr; // bias repeats every group, so a group copied from another ties it exactly
        try self.logits.resize(a, shape.rows * ne);
        try self.bias.resize(a, ne);
        for (self.logits.items, 0..) |*l, i| {
            l.* = rng.floatNorm(f32) * 2;
            if (i % 7 == 3) l.* = self.logits.items[i - 1]; // equal neighbours: the tie rule decides
        }
        for (self.bias.items, 0..) |*b, i| b.* = if (i % gs % 5 == 0) 0.25 else 0;
        if (gr > 1) tieRows(self.logits.items, shape.rows, ne, gr, self.route.keep);
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
        try self.fns().topk(self.rig.s, self.dev[0], self.dev[1], self.form(), self.dev[2], self.dev[3], self.shape.rows);
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
    pub fn check(self: *TopK, at: g.Gate) !?[]const u8 {
        if (self.route.score) {
            if (try self.checkScore()) |why| return why;
            if (self.route.groups == 1) return null;
            const s = self.shape;
            if (!g.quiet) std.debug.print("INFO {t} {s}: {d}x{d}x{d}, {t} fill: {d} of {d} rows skipped (near-tied group cut), {d} exact group ties checked\n", .{ at, self.route.name, s.rows, s.n, s.k, self.fill, self.near, s.rows, self.ties });
            return self.cutRows();
        }
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

    /// The score form's picks against float64 sigmoid + bias (near-ties within 2^-20 may swap), weights within 2^-20.
    fn checkScore(self: *TopK) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const out = try self.rig.outputBytes();
        const ns = self.slots();
        const n = self.shape.rows * ns;
        const ne = self.shape.n;
        const k_ = self.shape.k;
        const sel = try a.alloc(f64, ne);
        defer a.free(sel);
        const score = try a.alloc(f64, ne);
        defer a.free(score);
        const taken = try a.alloc(bool, ne);
        defer a.free(taken);
        self.near = 0;
        self.ties = 0;
        for (0..self.shape.rows) |r| {
            for (sel, score, 0..) |*v, *p, e| {
                p.* = 1 / (1 + @exp(-@as(f64, self.logits.items[r * ne + e])));
                v.* = if (std.math.isNan(p.*)) -std.math.inf(f64) else p.* + self.bias.items[e];
            }
            if (self.route.groups > 1) switch (keepGroups(sel, self.route.groups, self.route.keep)) {
                .near => { // fp32 may rank these groups either way
                    self.near += 1;
                    continue;
                },
                .exact => self.ties += 1,
                .clear => {},
            };
            @memset(taken, false);
            var w64: [64]f64 = undefined;
            var sum: f64 = 0;
            for (0..k_) |k| {
                const id: usize = std.mem.readInt(u32, out[(r * ns + k) * 4 ..][0..4], .little);
                if (id >= ne or taken[id]) return try std.fmt.bufPrint(&self.msg, "row {d} pick {d}: expert {d} out of range or repeated", .{ r, k, id });
                var best: f64 = -std.math.inf(f64);
                for (sel, taken) |v, t| if (!t) {
                    best = @max(best, v);
                };
                if (!(sel[id] >= best - 0x1p-20 * (@abs(best) + 1)) and !(best == -std.math.inf(f64) and sel[id] == best))
                    return try std.fmt.bufPrint(&self.msg, "row {d} pick {d}: expert {d} scores {d:.6}, an unpicked one {d:.6}", .{ r, k, id, sel[id], best });
                for (0..id) |e| if (!taken[e] and sel[e] == sel[id])
                    return try std.fmt.bufPrint(&self.msg, "row {d} pick {d}: expert {d} ahead of tied expert {d}", .{ r, k, id, e });
                taken[id] = true;
                w64[k] = score[id];
                sum += score[id];
            }
            for (0..ns) |k| {
                const id = std.mem.readInt(u32, out[(r * ns + k) * 4 ..][0..4], .little);
                const got: f64 = @as(f32, @bitCast(std.mem.readInt(u32, out[(n + r * ns + k) * 4 ..][0..4], .little)));
                if (k >= k_ and id != ne + k - k_) return try std.fmt.bufPrint(&self.msg, "row {d}: shared slot {d} holds {d}", .{ r, k, id });
                const want = if (k >= k_) 1 else (if (self.route.normalize) w64[k] / (sum + 1e-20) else w64[k]) * self.route.scale;
                if (!(@abs(got - want) <= @abs(want) * 0x1p-20 + 1e-30)) return try std.fmt.bufPrint(&self.msg, "row {d} slot {d}: weight {d:.8} where float64 gives {d:.8}", .{ r, k, got, want });
            }
        }
        return null;
    }

    /// The same rows as two calls (the first half, then the rest): every byte equal to the one call's.
    fn cutRows(self: *TopK) !?[]const u8 {
        const s = self.shape;
        if (s.rows < 2) return null;
        const a = self.rig.gpu.gpa;
        const whole = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(whole);
        const row = self.slots() * 4;
        for (self.dev[2..4]) |p| try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = p, .len = s.rows * row }).fill8(0xff, self.rig.s.handle);
        const m = s.rows / 2;
        try self.fns().topk(self.rig.s, self.dev[0], self.dev[1], self.form(), self.dev[2], self.dev[3], m);
        try self.fns().topk(self.rig.s, self.dev[0] + m * s.n * 4, self.dev[1], self.form(), self.dev[2] + m * row, self.dev[3] + m * row, s.rows - m);
        const at = std.mem.indexOfDiff(u8, whole, try self.rig.outputBytes()) orelse return null;
        return try std.fmt.bufPrint(&self.msg, "rows {d} and {d} as two calls: byte {d} differs from one call of {d}", .{ m, s.rows - m, at, s.rows });
    }
};

/// Rows with exact group ties: odd rows copy group 0 into every group; rows 3 mod 4 lift the last keep - 1 by 3.
fn tieRows(l: []f32, n_rows: usize, ne: usize, groups: usize, keep: usize) void {
    const gs = ne / groups;
    var r: usize = 1;
    while (r < n_rows) : (r += 2) {
        const row = l[r * ne ..][0..ne];
        for (gs..ne) |e| row[e] = row[e % gs] + @as(f32, if (r % 4 == 3 and e / gs >= groups + 1 - keep) 3 else 0);
    }
}

const Cut = enum { clear, exact, near };

/// float64 group-limited selection: experts outside the `keep` best groups (lower group on equal sums) go to -inf.
fn keepGroups(sel: []f64, groups: usize, keep: usize) Cut {
    const gs = sel.len / groups;
    var sums: [glue.most_groups]f64 = undefined;
    for (sums[0..groups], 0..) |*s, gi| {
        var a: f64 = -std.math.inf(f64);
        var b: f64 = -std.math.inf(f64);
        for (sel[gi * gs ..][0..gs]) |v| {
            if (v > a) {
                b = a;
                a = v;
            } else if (v > b) b = v;
        }
        s.* = a + b;
    }
    var kept: u32 = 0;
    for (0..keep) |_| {
        var best: usize = groups;
        for (sums[0..groups], 0..) |s, gi| if ((kept >> @intCast(gi)) & 1 == 0 and (best == groups or s > sums[best])) {
            best = gi;
        };
        kept |= @as(u32, 1) << @intCast(best);
    }
    var cut: Cut = .clear; // each kept group against each dropped one: equal sums, or near ones fp32 may swap
    for (sums[0..groups], 0..) |s, gi| {
        if ((kept >> @intCast(gi)) & 1 == 0) continue;
        for (sums[0..groups], 0..) |t, gj| {
            if ((kept >> @intCast(gj)) & 1 == 1) continue;
            if (s == t) cut = .exact else if (!(s - t > 0x1p-18 * (@abs(s) + 1))) return .near;
        }
    }
    for (sel, 0..) |*v, e| if ((kept >> @intCast(e / gs)) & 1 == 0) {
        v.* = -std.math.inf(f64);
    };
    return cut;
}

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

/// Shape: rows, dims (n), slots (k): Kolibri's 6 + 1 over 2,560, 8 + 1 over 7,168, Bolt's latent sum of 9 over 1,024.
pub fn combineSpec(x: Combo) c.Spec {
    return .{ .name = x.name, .accum = .f32, .depth = slotChain, .out = x.elem, .tiles = .{ .{}, .{ .size = 256 }, .{ .size = 8 } }, .served = &.{ .{ .rows = 16, .n = 2560, .k = 7 }, .{ .rows = 300, .n = 2560, .k = 7 }, .{ .rows = 9, .n = 7168, .k = 9 }, .{ .rows = 16, .n = 1024, .k = 9 }, .{ .rows = 300, .n = 1024, .k = 9 } } };
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
