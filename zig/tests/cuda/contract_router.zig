//! Nemotron's routers under the contract on CUDA: fp32 logits in K slices, then each row's top-k by its stated rule.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const glue = nemotron.glue;

/// One fma chain a K slice (two a bf16 word pair).
fn sliceChain(s: c.Shape) usize {
    return s.k / glue.routerShape(s.k).sk + 2;
}

pub fn logitsSpec() c.Spec {
    return .{ .name = "cuda nemotron router logits tf_nemo_router", .accum = .f32, .depth = sliceChain, .out = .f32, .tiles = .{ .{ .size = 16 }, .{ .size = 16 }, .{} }, .served = &.{ .{ .rows = 16, .n = 128, .k = 2688 }, .{ .rows = 33, .n = 128, .k = 2688 } } };
}

/// x [rows, D] bf16 times the router's W [E, D] bf16: fp32 partials [slices][rows][E].
pub const Logits = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Logits) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Logits) g.Kernel {
        return .{ .spec = logitsSpec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Logits);

    pub fn prepare(self: *Logits, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % 64 != 0 or shape.n > 256 or shape.n == 0 or shape.rows == 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, glue.routerShape(shape.k).sk);
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
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.out = try self.rig.output(d.outputs() * 4);
        return true;
    }

    pub fn run(self: *Logits) !void {
        const d = self.d.?;
        const shape = glue.routerShape(d.shape.k);
        var a: cuda.Args = .{};
        for ([_]u64{ self.x, self.w, self.out }) |p| a.add(p);
        for ([_]usize{ d.shape.rows, d.shape.k, d.shape.n, shape.sk }) |v| a.add(@as(c_int, @intCast(v)));
        const grid: cuda.Dim3 = .{ .x = @intCast((d.shape.rows + 15) / 16), .y = @intCast((d.shape.n + 15) / 16), .z = @intCast(shape.sk) };
        try cuda.launch.launch(self.f, .{ .grid = grid, .block = .{ .x = 256 }, .shared = shape.shared }, self.rig.s, &a);
        try self.rig.s.synchronize();
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

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "a NaN logit is never picked ahead of a number; ties go to the lower expert; two shared slots follow the top-k";
const top_k = 6;
const slots = top_k + 2;
const scaling: f32 = 2.5;

pub fn topkSpec() c.Spec {
    return .{ .name = "cuda nemotron top-k tf_nemo_topk", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = rule }, .tiles = .{ .{}, .{ .size = 32, .step = 32, .min = 32 }, .{} }, .served = &.{ .{ .rows = 16, .n = 128, .k = 6 }, .{ .rows = 33, .n = 256, .k = 6 } } };
}

/// Each row's slices summed in order, sigmoid plus bias ranked; shape: rows, experts (n) and K slices (k).
pub const TopK = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    shape: c.Shape = undefined,
    part: []f32 = &.{},
    bias: []f32 = &.{},
    in: u64 = 0,
    b: u64 = 0,
    ids: u64 = 0,
    wts: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *TopK) void {
        self.rig.gpu.gpa.free(self.part);
        self.rig.gpu.gpa.free(self.bias);
        self.rig.deinit();
    }

    pub fn kernel(self: *TopK) g.Kernel {
        return .{ .spec = topkSpec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(TopK);

    pub fn prepare(self: *TopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n < slots or shape.n > 256 or shape.k == 0 or shape.rows == 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        a.free(self.part);
        self.part = &.{};
        a.free(self.bias);
        self.bias = &.{};
        self.part = try a.alloc(f32, shape.k * shape.rows * shape.n);
        self.bias = try a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.part) |*p| p.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 16 - 2; // sixteenths: some ties
        for (self.bias) |*p| p.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 8))) / 64;
        if (fill == .nonfinite) {
            const n = shape.n;
            const r = shape.rows;
            const at = [_][3]usize{ .{ 0, 0, 3 }, .{ 0, 1, 5 }, .{ 0, 2, 7 }, .{ 0, 3, 2 }, .{ shape.k - 1, 3, 2 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |p, v| if (p[1] < r and !(p[0] == shape.k - 1 and shape.k == 1 and p[1] == 3)) {
                self.part[(p[0] * r + p[1]) * n + p[2]] = v;
            };
        }
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.part));
        self.b = try self.rig.input(std.mem.sliceAsBytes(self.bias));
        self.ids = try self.rig.output(shape.rows * slots * 4);
        self.wts = try self.rig.output(shape.rows * slots * 4);
        return true;
    }

    pub fn run(self: *TopK) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ self.in, self.b, self.ids, self.wts }) |p| a.add(p);
        a.add(@as(c_int, @intCast(self.shape.rows)));
        a.add(scaling);
        for ([_]usize{ self.shape.n, self.shape.k, top_k, slots, 1 }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = 1, .z = 1 }, .block = .{ .x = 32 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *TopK) usize {
        return self.shape.rows * slots * 8;
    }

    pub fn reference(_: *TopK, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Each row's picks against float64 scores (near-ties within 2^-20 may swap), its weights, its shared slots.
    pub fn check(self: *TopK, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const n = self.shape.n;
        const ids = try a.dupe(u8, try self.rig.bytes(self.ids, self.shape.rows * slots * 4));
        defer a.free(ids);
        const wts = try self.rig.bytes(self.wts, self.shape.rows * slots * 4);
        const sel = try a.alloc(f64, n);
        defer a.free(sel);
        const score = try a.alloc(f64, n);
        defer a.free(score);
        for (0..self.shape.rows) |r| {
            for (sel, score, 0..) |*v, *p, e| {
                var logit: f64 = 0;
                for (0..self.shape.k) |s| logit += self.part[(s * self.shape.rows + r) * n + e];
                p.* = 1 / (1 + @exp(-logit));
                v.* = if (std.math.isNan(p.* + self.bias[e])) -std.math.inf(f64) else p.* + self.bias[e];
            }
            var taken: u256 = 0;
            var total: f64 = 0;
            for (0..top_k) |k| {
                const id: usize = @intCast(std.mem.readInt(i32, ids[(r * slots + k) * 4 ..][0..4], .little));
                if (id >= n or taken >> @intCast(id) & 1 != 0) return try std.fmt.bufPrint(&self.msg, "row {d} pick {d}: expert {d} out of range or repeated", .{ r, k, id });
                var best: f64 = -std.math.inf(f64);
                for (sel, 0..) |v, e| if (taken >> @intCast(e) & 1 == 0) {
                    best = @max(best, v);
                };
                const tol = 0x1p-20 * (@abs(best) + 1);
                if (!(sel[id] >= best - tol) and !(best == -std.math.inf(f64) and sel[id] == best))
                    return try std.fmt.bufPrint(&self.msg, "row {d} pick {d}: expert {d} scores {d:.6}, an unpicked one {d:.6}", .{ r, k, id, sel[id], best });
                for (0..id) |e| if (taken >> @intCast(e) & 1 == 0 and sel[e] == sel[id])
                    return try std.fmt.bufPrint(&self.msg, "row {d} pick {d}: expert {d} ahead of tied expert {d}", .{ r, k, id, e });
                taken |= @as(u256, 1) << @intCast(id);
                total += if (std.math.isNan(score[id])) 0 else score[id];
            }
            for (0..slots) |k| {
                const id = std.mem.readInt(i32, ids[(r * slots + k) * 4 ..][0..4], .little);
                const w: f64 = @as(f32, @bitCast(std.mem.readInt(u32, wts[(r * slots + k) * 4 ..][0..4], .little)));
                const want: f64 = if (k >= top_k) 1 else score[@intCast(id)] / (total + 1e-20) * scaling;
                if (k >= top_k and id != @as(i32, @intCast(n + k - top_k))) return try std.fmt.bufPrint(&self.msg, "row {d} shared slot {d} holds {d}", .{ r, k, id });
                const ok = if (std.math.isFinite(want)) @abs(w - want) <= 0x1p-18 * @abs(want) + 1e-30 else c.classOf(w) == c.classOf(want);
                if (!ok) return try std.fmt.bufPrint(&self.msg, "row {d} slot {d}: weight {d:.7} where float64 gives {d:.7}", .{ r, k, w, want });
            }
        }
        return null;
    }
};

const latent_rule = "tf_super_topk's bytes, and its host twin's picks: a NaN never ahead of a number, ties to the lower " ++
    "expert, every pick past the numbers expert 0";
const latent_picks = 22;
const latent_shares = 2;

/// Super's 512 experts and the 1,024 the kernel holds, summed from 8 slices at decode-sized calls; warp edges too.
pub fn latentSpec() c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 1, .n = 512, .k = 8 }, .{ .rows = 16, .n = 512, .k = 8 }, .{ .rows = 16, .n = 1024, .k = 8 } };
    return .{ .name = "cuda nemotron latent top-k tf_super_topk_few", .accum = .none, .depth = noSums, .out = .raw, .replaces = "tf_super_topk", .nan = .{ .stated = latent_rule }, .tiles = .{ .{}, .{ .size = 32, .min = latent_picks }, .{} }, .served = served };
}

/// The latent router's picks, a thread an expert: pair ids by pick and share, weights by pick; shape as TopK's.
pub const LatentTopK = struct {
    rig: rig_.Rig,
    fns: *const nemotron.latent.Fns,
    shape: c.Shape = undefined,
    part: []f32 = &.{},
    bias: []f32 = &.{},
    in: u64 = 0,
    b: u64 = 0,
    ids: u64 = 0,
    wts: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *LatentTopK) void {
        self.rig.gpu.gpa.free(self.part);
        self.rig.gpu.gpa.free(self.bias);
        self.rig.deinit();
    }

    pub fn kernel(self: *LatentTopK) g.Kernel {
        return .{ .spec = latentSpec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(LatentTopK);

    fn idBytes(self: *const LatentTopK) usize {
        return self.shape.rows * latent_picks * latent_shares * 4;
    }

    fn outBytes(self: *const LatentTopK) usize {
        return self.idBytes() + self.shape.rows * latent_picks * 4;
    }

    /// Sixteenths of logits (ties), biases in 64ths; non-finite: NaN and infinities in rows 0-3, numbers in 12 experts.
    pub fn prepare(self: *LatentTopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n < latent_picks or shape.n > nemotron.latent.max_experts or shape.k == 0 or shape.rows == 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        a.free(self.part);
        self.part = &.{};
        a.free(self.bias);
        self.bias = &.{};
        self.part = try a.alloc(f32, shape.k * shape.rows * shape.n);
        self.bias = try a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.part) |*p| p.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 16 - 2;
        for (self.bias) |*p| p.* = @as(f32, @floatFromInt(rng.uintLessThan(u32, 8))) / 64;
        if (fill == .nonfinite) {
            const n = shape.n;
            const r = shape.rows;
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };
            for (put, 0..) |v, row| if (row < r) for (0..4) |e| {
                self.part[row * n + 3 * e + row] = v;
            };
            @memset(self.bias[12..], -std.math.inf(f32));
        }
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.part));
        self.b = try self.rig.input(std.mem.sliceAsBytes(self.bias));
        self.ids = try self.rig.output(self.idBytes());
        self.wts = try self.rig.output(self.outBytes() - self.idBytes());
        self.base = try self.rig.scratch(self.outBytes());
        return true;
    }

    pub fn run(self: *LatentTopK) !void {
        try self.launch(true, self.ids, self.wts);
        try self.rig.s.synchronize();
    }

    fn launch(self: *LatentTopK, few: bool, ids: u64, wts: u64) !void {
        const x = self.shape;
        try self.fns.picks(self.rig.s, few, self.in, self.b, ids, wts, x.rows, scaling, x.n, x.k, latent_picks, latent_shares, true);
    }

    /// tf_super_topk on the same inputs, ids then weights as the outputs gather.
    pub fn replaced(self: *LatentTopK) ![]const u8 {
        try self.launch(false, self.base, self.base + self.idBytes());
        return self.rig.bytes(self.base, self.outBytes());
    }

    pub fn count(self: *LatentTopK) usize {
        return self.outBytes();
    }

    pub fn reference(_: *LatentTopK, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// tf_super_topk's bytes, then the host twin's ids and weights (a NaN weight for a NaN).
    pub fn check(self: *LatentTopK, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const x = self.shape;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (std.mem.indexOfDiff(u8, got, try self.replaced())) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from tf_super_topk's", .{ at, got.len });
        const idx = try a.alloc(i32, x.rows * latent_picks * latent_shares);
        defer a.free(idx);
        const wt = try a.alloc(f32, x.rows * latent_picks);
        defer a.free(wt);
        nemotron.glue_ref.superTopk(self.part, self.bias, idx, wt, x.rows, scaling, x.n, x.k, latent_picks, latent_shares, true);
        if (std.mem.indexOfDiff(u8, got[0..self.idBytes()], std.mem.sliceAsBytes(idx))) |at| return try std.fmt.bufPrint(&self.msg, "id byte {d} differs from the host twin's", .{at});
        for (wt, 0..) |w, i| {
            const v: f32 = @bitCast(std.mem.readInt(u32, got[self.idBytes() + 4 * i ..][0..4], .little));
            if (v != w and !(std.math.isNan(v) and std.math.isNan(w))) return try std.fmt.bufPrint(&self.msg, "weight {d} is {e}, the host twin's {e}", .{ i, v, w });
        }
        return null;
    }
};
