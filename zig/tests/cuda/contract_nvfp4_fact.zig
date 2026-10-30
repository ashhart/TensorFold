//! NVFP4 prompt tiles and decode waves with the decode form's sums under the contract: each replaces it bit for bit.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const fx = cuda.experts.Nvfp4Experts;
const grouped = cuda.grouped;

/// Fact up/down, a control, experts_waves' entries, the TF_NVFP4_DECODE_WARPS=1 CTAs, two controls cut to one wave.
pub const Kind = enum { up, down, control, up_waves, down_waves, up_half, up_one, down_one, down_bf16_one, relu2_one, up_waves_one, down_waves_one, up_half_one, cut_waves, cut_waves_one };

/// experts_waves.zig's entry a kind launches, or null for the prompt-tile and classic kinds.
fn wavesEntry(kind: Kind) ?usize {
    return switch (kind) {
        .up_waves, .up_waves_one, .cut_waves => 4,
        .down_waves, .down_waves_one, .cut_waves_one => 5,
        .up_half, .up_half_one => 6,
        else => null,
    };
}

fn isDown(kind: Kind) bool {
    return switch (kind) {
        .down, .down_waves, .down_one, .down_bf16_one, .down_waves_one, .cut_waves_one => true,
        else => false,
    };
}

/// Kinds that run the group's 1-warp CTAs; the baseline is always its 4-warp classic kernel.
fn isOne(kind: Kind) bool {
    return switch (kind) {
        .up_one, .down_one, .down_bf16_one, .relu2_one, .up_waves_one, .down_waves_one, .up_half_one, .cut_waves_one => true,
        else => false,
    };
}

fn epiOf(kind: Kind) cuda.experts.Epi {
    return if (kind == .down_bf16_one) .down_bf16 else if (isDown(kind)) .down_f32 else .up;
}

const experts = 4;
const slots = 2; // pairs a token row, both routed
const waves_rows = 1024; // the waves kinds' two-wave shape: cut_waves and cut_waves_one go red only there
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
    // a decode item of 16 pairs (8 token rows) by a warp's 32 columns (half: 16 of them) over all K, 32 inputs a step
    const unit = [3]c.Tile{ .{ .size = 16 / slots }, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 32, .step = 32, .min = 32 } };
    const why = "a bf16 sum that collapses moves the output by about itself, and the epilogue's three bf16 roundings already allow 3 u of it: at most 1 / (3 u) = 85x is provable";
    // 1,024 rows: 128+ items of 16 pairs, 2,560+ gate-up and 10,240+ down units: two waves below 2,560 resident warps
    const up_served = &[_]c.Shape{ .{ .rows = 1, .n = 640, .k = 2560 }, .{ .rows = 4, .n = 640, .k = 2560 }, .{ .rows = 16, .n = 640, .k = 2560 }, .{ .rows = waves_rows, .n = 640, .k = 2560 } };
    const down_served = &[_]c.Shape{ .{ .rows = 1, .n = 2560, .k = 640 }, .{ .rows = 4, .n = 2560, .k = 640 }, .{ .rows = 16, .n = 2560, .k = 640 }, .{ .rows = waves_rows, .n = 2560, .k = 640 } };
    const why2 = "the bf16 rounding before the square is allowed in the bound (2 u of the output), so a short sum stands out less than in SwiGLU: fp16 and bf16 sums miss by 64x";
    const cut = "dropped past the first wave: gate 3 at the two-wave shape refuses it";
    return switch (kind) {
        .up_one => .{ .name = "cuda nvfp4 experts decode gate-up nvfp4_expert_kernel<2, 2, 1> (TF_NVFP4_DECODE_WARPS=1)", .accum = .f32, .depth = depth, .ops = 1, .mids = 3, .out = .bf16, .replaces = "nvfp4_expert_kernel<2, 2, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = up_served, .sums = .{ .bf16 = 30, .why = why } },
        .up_waves_one => .{ .name = "cuda nvfp4 experts decode gate-up nvfp4_expert_waves<2, 2, 1> (TF_NVFP4_DECODE_WARPS=1)", .accum = .f32, .depth = depth, .ops = 1, .mids = 3, .out = .bf16, .replaces = "nvfp4_expert_kernel<2, 2, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = up_served, .sums = .{ .bf16 = 30, .why = why } },
        .up_half_one => .{ .name = "cuda nvfp4 experts decode gate-up nvfp4_expert_half<2, 2, 1> (TF_NVFP4_DECODE_WARPS=1)", .accum = .f32, .depth = depth, .ops = 1, .mids = 3, .out = .bf16, .replaces = "nvfp4_expert_kernel<2, 2, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = up_served, .sums = .{ .bf16 = 30, .why = why } },
        .relu2_one => .{ .name = "cuda nvfp4 experts decode up relu^2 nvfp4_expert_kernel<1, 1, 1> (TF_NVFP4_DECODE_WARPS=1)", .accum = .f32, .depth = depth, .ops = 1, .mids = 1, .out = .bf16, .replaces = "nvfp4_expert_kernel<1, 1, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = up_served, .sums = .{ .f16 = 50, .bf16 = 30, .why = why2 } },
        .down_one => .{ .name = "cuda nvfp4 experts decode down nvfp4_expert_kernel<1, 0, 1> (TF_NVFP4_DECODE_WARPS=1)", .accum = .f32, .depth = depth, .ops = 1, .out = .f32, .replaces = "nvfp4_expert_kernel<1, 0, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = down_served },
        .down_waves_one => .{ .name = "cuda nvfp4 experts decode down nvfp4_expert_waves<1, 0, 1> (TF_NVFP4_DECODE_WARPS=1)", .accum = .f32, .depth = depth, .ops = 1, .out = .f32, .replaces = "nvfp4_expert_kernel<1, 0, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = down_served },
        .down_bf16_one => .{ .name = "cuda nvfp4 experts decode down bf16 nvfp4_expert_kernel<1, 3, 1> (TF_NVFP4_DECODE_WARPS=1)", .accum = .f32, .depth = depth, .ops = 1, .out = .bf16, .replaces = "nvfp4_expert_kernel<1, 3, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = down_served, .sums = .{ .bf16 = 30, .why = why } },
        .cut_waves => .{ .name = "control: nvfp4_expert_waves<2, 2, 4> with its items " ++ cut, .accum = .f32, .depth = depth, .ops = 1, .mids = 3, .out = .bf16, .replaces = "nvfp4_expert_kernel<2, 2, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = up_served },
        .cut_waves_one => .{ .name = "control: nvfp4_expert_waves<1, 0, 1> with its items " ++ cut, .accum = .f32, .depth = depth, .ops = 1, .out = .f32, .replaces = "nvfp4_expert_kernel<1, 0, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = down_served },
        .up_waves => .{ .name = "cuda nvfp4 experts decode gate-up nvfp4_expert_waves<2, 2, 4> (equal waves)", .accum = .f32, .depth = depth, .ops = 1, .mids = 3, .out = .bf16, .replaces = "nvfp4_expert_kernel<2, 2, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = up_served, .sums = .{ .bf16 = 30, .why = why } },
        .up_half => .{ .name = "cuda nvfp4 experts decode gate-up nvfp4_expert_half<2, 2, 4> (two warps a unit)", .accum = .f32, .depth = depth, .ops = 1, .mids = 3, .out = .bf16, .replaces = "nvfp4_expert_kernel<2, 2, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = up_served, .sums = .{ .bf16 = 30, .why = why } },
        .down_waves => .{ .name = "cuda nvfp4 experts decode down nvfp4_expert_waves<1, 0, 4> (equal waves)", .accum = .f32, .depth = depth, .ops = 1, .out = .f32, .replaces = "nvfp4_expert_kernel<1, 0, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = down_served },
        .up => .{ .name = "cuda nvfp4 experts prompt gate-up tf_nvfp4_fact_swiglu (decode sums)", .accum = .f32, .depth = depth, .ops = 1, .mids = 3, .out = .bf16, .replaces = "nvfp4_expert_kernel<2, 2, 4>", .nan = .{ .stated = rule }, .tiles = up_tiles, .served = &.{ .{ .rows = 130, .n = 640, .k = 2560 }, .{ .rows = 1, .n = 640, .k = 2560 } }, .sums = .{ .bf16 = 30, .why = "a bf16 sum that collapses moves the output by about itself, and the epilogue's three bf16 roundings already allow 3 u of it: at most 1 / (3 u) = 85x is provable" } },
        .control => .{ .name = "control: scaled-code prompt tile tf_nvfp4_prompt_swiglu posing as the decode gate-up", .accum = .f32, .depth = depth, .ops = 1, .mids = 3, .out = .bf16, .replaces = "nvfp4_expert_kernel<2, 2, 4>", .nan = .{ .stated = rule }, .tiles = up_tiles, .served = &.{.{ .rows = 130, .n = 640, .k = 2560 }} },
        .down => .{ .name = "cuda nvfp4 experts prompt down tf_nvfp4_fact_down_f32 (decode sums)", .accum = .f32, .depth = depth, .ops = 1, .out = .f32, .replaces = "nvfp4_expert_kernel<1, 0, 4>", .nan = .{ .stated = rule }, .tiles = down_tiles, .served = &.{ .{ .rows = 130, .n = 2560, .k = 640 }, .{ .rows = 1, .n = 2560, .k = 640 } } },
    };
}

pub const Experts = struct {
    rig: rig_.Rig,
    kind: Kind,
    k: *const fx, // Group's four: the baseline, and the kernels under test unless isOne
    k1: *const fx, // Group's one
    tile: ?*const cuda.experts_tile.Tile = null,
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
        var entry = spec(self.kind);
        if (self.tile != null) {
            entry.name = switch (self.kind) {
                .up => "cuda nvfp4 K32 tile gate-up (decode sums)",
                .down => "cuda nvfp4 K32 tile fp32 down (decode sums)",
                .control => "control: scaled-code prompt posing as the K32 tile",
                else => unreachable,
            };
            entry.tiles[1].size = 128;
        }
        return .{ .spec = entry, .ctx = self, .vt = &vt, .control = switch (self.kind) {
            .control, .cut_waves, .cut_waves_one => .shapes,
            else => null,
        } };
    }

    const vt = g.hooks(Experts);

    fn parts(self: *const Experts) usize {
        return if (isDown(self.kind) or self.kind == .relu2_one) 1 else 2;
    }

    fn pairs(self: *const Experts) usize {
        return self.shape.rows * slots;
    }

    fn xRows(self: *const Experts) usize {
        return if (!isDown(self.kind)) self.shape.rows else self.pairs();
    }

    fn row(self: *const Experts, pair: usize) usize {
        return if (!isDown(self.kind)) pair / slots else pair;
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
            if (self.parts() == 2) fx.packGateUp(o, .{ self.codesOf(e, 0), self.scalesOf(e, 0) }, .{ self.codesOf(e, 1), self.scalesOf(e, 1) }, n, k) else fx.packOne(o, self.codesOf(e, 0), self.scalesOf(e, 0), n, k);
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
        const up = !isDown(self.kind);
        return .{ .format = .nvfp4, .up = self.wd, .down = self.wd, .up_scale = self.sd, .down_scale = self.sd, .width = if (up) self.shape.n else self.shape.k, .dims = if (up) self.shape.k else self.shape.n, .experts = experts, .relu2 = self.kind == .relu2_one };
    }

    fn launch(self: *Experts, fact: bool, out: u64) !void {
        const up = !isDown(self.kind);
        const in: cuda.experts.Rows = .{ .x = self.xd, .stride = self.shape.k, .slots = if (up) slots else 0 };
        const epi = epiOf(self.kind);
        const items = grouped.maxItems(self.pairs(), experts, 16);
        const k = if (isOne(self.kind)) self.k1 else self.k;
        if (fact and wavesEntry(self.kind) != null) {
            const plan = if (self.kind == .cut_waves or self.kind == .cut_waves_one) try self.firstWave(k, items) else self.plan;
            try cuda.experts_waves.entry(k, wavesEntry(self.kind).?, self.rig.s, in, self.layer(), plan, out, items, -1);
        } else if (fact and isOne(self.kind)) {
            try k.run(self.rig.s, epi, in, self.layer(), self.plan, out, items, -1);
        } else if (fact and self.kind == .control) {
            try self.k.runPrompt(self.rig.s, epi, in, self.layer(), self.wide, out, grouped.maxItems(self.pairs(), experts, cuda.experts.promptTile(.nvfp4)), -1);
        } else if (fact and self.tile != null) {
            try self.tile.?.run(self.rig.s, epi, in, self.layer(), self.wide, out, grouped.maxItems(self.pairs(), experts, cuda.experts.promptTile(.nvfp4)), -1);
        } else if (fact) {
            try self.k.runFact(self.rig.s, epi, in, self.layer(), self.wide, out, grouped.maxItems(self.pairs(), experts, cuda.experts.promptTile(.nvfp4)), -1);
        } else try self.k.run(self.rig.s, epi, in, self.layer(), self.plan, out, items, -1);
        try self.rig.s.synchronize();
    }

    /// The plan with its item count cut to one wave of the entry's grid (unchanged where one wave covers it).
    fn firstWave(self: *Experts, k: *const fx, items: usize) !grouped.Plan {
        const fi = wavesEntry(self.kind).?;
        const nb = self.shape.n / fx.cols;
        const grid = @min((items * nb + k.warps - 1) / k.warps, k.resident[fi]);
        try self.rig.s.synchronize();
        var counts: [2]i32 = undefined;
        @memcpy(std.mem.asBytes(&counts), try self.rig.bytes(self.plan.counts, 8));
        counts[0] = @min(counts[0], @as(i32, @intCast(grid * k.warps / nb)));
        var p = self.plan;
        p.counts = try self.rig.input(std.mem.asBytes(&counts));
        return p;
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
        if (isDown(self.kind)) return s0 * self.globals[e];
        if (self.kind == .relu2_one) {
            const r = @max(c.fromBf16(c.toBf16(@floatCast(s0 * self.globals[e]))), 0);
            return r * r;
        }
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
            if (isDown(self.kind)) {
                const gl = self.globals[e];
                want[o] = sum[0] * gl;
                mag[o] = @abs(gl) * abs[0];
                self.allow_extra[o] = 0;
                continue;
            }
            if (self.kind == .relu2_one) {
                const r = @max(sum[0] * self.globals[e], 0);
                want[o] = r * r;
                mag[o] = 2 * r * @abs(self.globals[e]) * abs[0];
                // the kernel rounds the scaled sum to bf16 before squaring: half a bf16 ulp, doubled by the square
                self.allow_extra[o] = 0x1p-8 * 2 * r * r * 1.01;
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
