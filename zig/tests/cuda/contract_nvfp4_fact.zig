//! NVFP4 prompt tiles with the decode form's sums under the contract: each replaces its decode form bit for bit.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const fx = cuda.experts.Nvfp4Experts;
const grouped = cuda.grouped;

/// SwiGLU gate-up, fp32 down, and a control: tf_nvfp4_prompt_swiglu posing as a replacement, which gate 3 refuses.
pub const Kind = enum { up, down, control };

const experts = 4;
const slots = 2; // pairs a token row, both routed
const rule = "as the decode form it replaces gives them: NaN and infinite global scales, a NaN block scale, NaN and infinite inputs";

/// Each 16-input block's mma (16 products) and its fma into the running sum, blocks in ascending K.
fn depth(s: c.Shape) usize {
    return s.k / 16 + 16;
}

/// Flash Next's routed experts: 2,560 dims, 640 wide.
pub fn spec(kind: Kind) c.Spec {
    // a plan item of 64 pairs (32 token rows) by a CTA's columns (gate-up 4 blocks of 32, down 8), 64 inputs a stage
    const up_tiles = [3]c.Tile{ .{ .size = 64 / slots }, .{ .size = 128, .step = 32, .min = 32 }, .{ .size = 64, .step = 32, .min = 32 } };
    const down_tiles = [3]c.Tile{ .{ .size = 64 / slots }, .{ .size = 256, .step = 32, .min = 32 }, .{ .size = 64, .step = 32, .min = 32 } };
    return switch (kind) {
        .up => .{ .name = "cuda nvfp4 experts prompt gate-up tf_nvfp4_fact_swiglu (decode sums)", .accum = .f32, .depth = depth, .ops = 1, .mids = 3, .out = .bf16, .replaces = "nvfp4_expert_kernel<2, 2, 4>", .nan = .{ .stated = rule }, .tiles = up_tiles, .served = &.{ .{ .rows = 130, .n = 640, .k = 2560 }, .{ .rows = 1, .n = 640, .k = 2560 } }, .sums = .{ .bf16 = 30, .why = "a bf16 sum that collapses moves the output by about itself, and the epilogue's three bf16 roundings already allow 3 u of it: at most 1 / (3 u) = 85x is provable" } },
        .control => .{ .name = "control: scaled-code prompt tile tf_nvfp4_prompt_swiglu posing as the decode gate-up", .accum = .f32, .depth = depth, .ops = 1, .mids = 3, .out = .bf16, .replaces = "nvfp4_expert_kernel<2, 2, 4>", .nan = .{ .stated = rule }, .tiles = up_tiles, .served = &.{.{ .rows = 130, .n = 640, .k = 2560 }} },
        .down => .{ .name = "cuda nvfp4 experts prompt down tf_nvfp4_fact_down_f32 (decode sums)", .accum = .f32, .depth = depth, .ops = 1, .out = .f32, .replaces = "nvfp4_expert_kernel<1, 0, 4>", .nan = .{ .stated = rule }, .tiles = down_tiles, .served = &.{ .{ .rows = 130, .n = 2560, .k = 640 }, .{ .rows = 1, .n = 2560, .k = 640 } } },
    };
}

pub const Experts = struct {
    rig: rig_.Rig,
    kind: Kind,
    k: *const fx,
    router: grouped.Router,
    shape: c.Shape = undefined,
    x: []u16 = &.{}, // bf16 rows: a token's for up, a pair's for down
    codes: []u8 = &.{}, // [experts][parts][n][k/2]
    scales: []u8 = &.{}, // [experts][parts][n][k/16] e4m3
    globals: [experts * 2]f32 = undefined, // [experts][parts]
    picks: []i32 = &.{},
    allow_extra: []f64 = &.{}, // up: the SwiGLU roundings' share of each output's bound
    wd: u64 = 0,
    sd: u64 = 0,
    xd: u64 = 0,
    plan: grouped.Plan = undefined, // 16 pairs an item: the decode form's
    wide: grouped.Plan = undefined, // promptTile pairs an item: the fact tiles'
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Experts) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Experts) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.codes);
        a.free(self.scales);
        a.free(self.picks);
        a.free(self.allow_extra);
        self.x = &.{};
        self.codes = &.{};
        self.scales = &.{};
        self.picks = &.{};
        self.allow_extra = &.{};
    }

    pub fn kernel(self: *Experts) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt, .control = if (self.kind == .control) .shapes else null };
    }

    const vt = g.hooks(Experts);

    fn parts(self: *const Experts) usize {
        return if (self.kind != .down) 2 else 1;
    }

    fn pairs(self: *const Experts) usize {
        return self.shape.rows * slots;
    }

    fn xRows(self: *const Experts) usize {
        return if (self.kind != .down) self.shape.rows else self.pairs();
    }

    fn row(self: *const Experts, pair: usize) usize {
        return if (self.kind != .down) pair / slots else pair;
    }

    fn route(self: *Experts, ids: u64, tile: usize, zeros: []const u8) !grouped.Plan {
        const blocks = (self.pairs() + 1023) / 1024;
        const p: grouped.Plan = .{ .members = try self.rig.input(zeros[0 .. self.pairs() * 4]), .items = try self.rig.input(zeros[0 .. grouped.maxItems(self.pairs(), experts, tile) * 12]), .counts = try self.rig.input(zeros[0..8]), .rank = try self.rig.scratch(self.pairs() * 4), .hist = try self.rig.scratch(blocks * experts * 4) };
        try self.router.route(self.rig.s, ids, self.pairs(), experts, tile, p);
        return p;
    }

    fn codesOf(self: *const Experts, e: usize, part: usize) []u8 {
        const per = self.shape.n * self.shape.k / 2;
        return self.codes[(e * self.parts() + part) * per ..][0..per];
    }

    fn scalesOf(self: *const Experts, e: usize, part: usize) []u8 {
        const per = self.shape.n * self.shape.k / 16;
        return self.scales[(e * self.parts() + part) * per ..][0..per];
    }

    /// Random codes, scales, globals and inputs; non-finite: planted values; stress: integer sums, mirrored halves.
    pub fn prepare(self: *Experts, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u16, self.xRows() * k);
        self.codes = try a.alloc(u8, experts * self.parts() * n * k / 2);
        self.scales = try a.alloc(u8, experts * self.parts() * n * k / 16);
        self.picks = try a.alloc(i32, self.pairs());
        self.allow_extra = try a.alloc(f64, self.count());
        rng.bytes(self.codes);
        for (self.scales) |*s| s.* = 0x20 + rng.uintLessThan(u8, 0x28);
        for (&self.globals) |*v| v.* = 0.005 + rng.float(f32) * 0.03;
        for (self.x) |*v| v.* = c.toBf16((rng.float(f32) - 0.5) * 2);
        for (self.picks, 0..) |*p, i| p.* = @intCast(if (i < experts) i else rng.uintLessThan(usize, experts));
        switch (fill) {
            .random => {},
            .nonfinite => {
                self.globals[1 * self.parts()] = std.math.nan(f32);
                self.globals[2 * self.parts()] = std.math.inf(f32);
                self.scales[0] = 0x7f;
                self.x[(self.xRows() - 1) * k] = c.toBf16(std.math.nan(f32));
                if (self.xRows() > 1) self.x[(self.xRows() - 2) * k + 1] = c.toBf16(std.math.inf(f32));
            },
            .stress => self.stress(rng),
        }
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
        // pack on the host as the loader does: [E][n/32][k/32][parts][144] int32
        const per = fx.bytes(1, n, k, self.parts()) / 4;
        const words = try a.alloc(u32, experts * per);
        defer a.free(words);
        for (0..experts) |e| {
            const o = words[e * per ..][0..per];
            if (self.kind != .down) fx.packGateUp(o, .{ self.codesOf(e, 0), self.scalesOf(e, 0) }, .{ self.codesOf(e, 1), self.scalesOf(e, 1) }, n, k) else fx.packOne(o, self.codesOf(e, 0), self.scalesOf(e, 0), n, k);
        }
        self.wd = try self.rig.input(std.mem.sliceAsBytes(words));
        self.sd = try self.rig.input(std.mem.sliceAsBytes(self.globals[0 .. experts * self.parts()]));
        const ids = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        const zeros = try a.alloc(u8, @max(self.pairs() * 4, grouped.maxItems(self.pairs(), experts, 16) * 12, 8));
        defer a.free(zeros);
        @memset(zeros, 0);
        self.plan = try self.route(ids, 16, zeros);
        self.wide = try self.route(ids, cuda.experts.promptTile(.nvfp4), zeros);
        try self.rig.s.synchronize();
        self.out = try self.rig.output(self.outBytes());
        self.base = try self.rig.scratch(self.outBytes());
        return true;
    }

    /// Gate 4: bf16-integer inputs, codes 4 and 6 at scale 1, K's halves mirrored: fp32 sums exact in any order.
    fn stress(self: *Experts, rng: std.Random) void {
        const k = self.shape.k;
        @memset(self.scales, 0x38);
        @memset(&self.globals, 1);
        for (0..self.xRows()) |r| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(128 + rng.uintLessThan(u32, 128));
            self.x[r * k + i] = c.toBf16(v);
            self.x[r * k + k / 2 + i] = c.toBf16(if (i == 0) v - 1 else v);
        };
        for (0..experts * self.parts() * self.shape.n) |j| for (0..k / 2) |i| {
            const v: u8 = if (rng.boolean()) 6 else 7; // e2m1 4 and 6
            put(self.codes[j * k / 2 ..], i, v);
            put(self.codes[j * k / 2 ..], k / 2 + i, v | 8);
        };
    }

    fn put(codes: []u8, i: usize, v: u8) void {
        const b = &codes[i / 2];
        b.* = if (i % 2 == 0) (b.* & 0xf0) | v else (b.* & 0x0f) | v << 4;
    }

    fn weight(self: *const Experts, e: usize, part: usize, j: usize, i: usize) f64 {
        const k = self.shape.k;
        const b = self.codesOf(e, part)[j * k / 2 + i / 2];
        const nib: u4 = @truncate(if (i % 2 == 0) b & 0xf else b >> 4);
        return core.quant.nvfp4.e2m1(nib) * core.quant.fp8.e4m3(self.scalesOf(e, part)[j * k / 16 + i / 16]);
    }

    /// Pair p's products with column j of its expert's part, in input order (the global scale applies after the sum).
    fn terms(self: *const Experts, p: usize, part: usize, j: usize, t: []f64) void {
        const e: usize = @intCast(self.picks[p]);
        const xr = self.x[self.row(p) * self.shape.k ..][0..self.shape.k];
        for (t, 0..) |*o, i| o.* = @as(f64, c.fromBf16(xr[i])) * self.weight(e, part, j, i);
    }

    fn outBytes(self: *const Experts) usize {
        return self.count() * c.size(spec(self.kind).out);
    }

    pub fn count(self: *const Experts) usize {
        return self.pairs() * self.shape.n;
    }

    fn layer(self: *const Experts) cuda.experts.Layer {
        const up = self.kind != .down;
        return .{ .format = .nvfp4, .up = self.wd, .down = self.wd, .up_scale = self.sd, .down_scale = self.sd, .width = if (up) self.shape.n else self.shape.k, .dims = if (up) self.shape.k else self.shape.n, .experts = experts };
    }

    fn launch(self: *Experts, fact: bool, out: u64) !void {
        const up = self.kind != .down;
        const in: cuda.experts.Rows = .{ .x = self.xd, .stride = self.shape.k, .slots = if (up) slots else 0 };
        const epi: cuda.experts.Epi = if (up) .up else .down_f32;
        if (fact and self.kind == .control) {
            try self.k.runPrompt(self.rig.s, epi, in, self.layer(), self.wide, out, grouped.maxItems(self.pairs(), experts, cuda.experts.promptTile(.nvfp4)), -1);
        } else if (fact) {
            try self.k.runFact(self.rig.s, epi, in, self.layer(), self.wide, out, grouped.maxItems(self.pairs(), experts, cuda.experts.promptTile(.nvfp4)), -1);
        } else try self.k.run(self.rig.s, epi, in, self.layer(), self.plan, out, grouped.maxItems(self.pairs(), experts, 16), -1);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Experts) !void {
        try self.launch(true, self.out);
    }

    pub fn replaced(self: *Experts) ![]const u8 {
        try self.launch(false, self.base);
        return self.rig.bytes(self.base, self.outBytes());
    }

    /// Gate 1: the decode form's bytes, and some output non-finite (the planted values reach it).
    pub fn check(self: *Experts, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from {s}'s", .{ at, got.len, spec(self.kind).replaces.? });
        const out = spec(self.kind).out;
        for (0..got.len / c.size(out)) |i| if (c.classOf(c.read(out, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    fn silu(v: f64) f64 {
        return v / (1 + @exp(-v));
    }

    /// The epilogue on pair p's sums: global scales, then for up the kernel's SwiGLU with its bf16 roundings.
    fn epilogue(self: *const Experts, p: usize, s0: f64, s1: f64) f64 {
        const e: usize = @intCast(self.picks[p]);
        if (self.kind == .down) return s0 * self.globals[e];
        const gv = c.fromBf16(c.toBf16(@floatCast(s0 * self.globals[e * 2])));
        const uv = c.fromBf16(c.toBf16(@floatCast(s1 * self.globals[e * 2 + 1])));
        return c.fromBf16(c.toBf16(@floatCast(silu(gv)))) * uv;
    }

    pub fn reference(self: *Experts, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        const t = try a.alloc(f64, self.shape.k);
        defer a.free(t);
        const n = self.shape.n;
        for (0..self.pairs()) |p| for (0..n) |j| {
            const e: usize = @intCast(self.picks[p]);
            const o = p * n + j;
            var sum = [2]f64{ 0, 0 };
            var abs = [2]f64{ 0, 0 };
            for (0..self.parts()) |part| {
                self.terms(p, part, j, t);
                for (t) |v| {
                    sum[part] += v;
                    abs[part] += @abs(v);
                }
            }
            if (self.kind == .down) {
                const gl = self.globals[e];
                want[o] = sum[0] * gl;
                mag[o] = @abs(gl) * abs[0];
                self.allow_extra[o] = 0;
                continue;
            }
            const gs = sum[0] * self.globals[e * 2];
            const us = sum[1] * self.globals[e * 2 + 1];
            want[o] = silu(gs) * us;
            // d silu / dg is at most 1.1: a sum's error moves the output by its slope times the other factor
            mag[o] = 1.1 * @abs(us) * @abs(self.globals[e * 2]) * abs[0] + @abs(silu(gs)) * @abs(self.globals[e * 2 + 1]) * abs[1];
            // the kernel rounds g, u and silu(g) to bf16 before the product: each moves it by half a bf16 ulp
            self.allow_extra[o] = 0x1p-8 * (1.1 * @abs(gs) * @abs(us) + 2 * @abs(silu(gs)) * @abs(us)) * 1.01;
        };
    }

    pub fn bounds(self: *Experts, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.allow_extra) |*l, x| l.* += x;
    }

    pub fn simulate(self: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const t = try a.alloc(f64, self.shape.k);
        defer a.free(t);
        const n = self.shape.n;
        for (0..self.pairs()) |p| for (0..n) |j| {
            var s = [2]f64{ 0, 0 };
            for (0..self.parts()) |part| {
                self.terms(p, part, j, t);
                s[part] = contract.rows.sumIn(acc, order, t);
            }
            out[p * n + j] = self.epilogue(p, s[0], s[1]);
        };
    }
};
