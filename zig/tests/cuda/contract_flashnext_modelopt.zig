//! NVIDIA's NVFP4 Flash Next checkpoint: BF16 dense and shared projections and FP8 MTP-head expert downs, gates 1-4.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const Dense = core.contract.dense.Dense;
const q = cuda.qmmf;
const ex = cuda.experts;
const grouped = cuda.grouped;

fn depth(s: c.Shape) usize {
    return s.k / 16 + s.k / 128 + 24;
}

/// Stress weights: integers 1 to 7 over the first half of each row and their negatives over the second.
fn stress(rng: std.Random, n: usize, k: usize, out: []f32) void {
    for (0..n) |j| for (0..k / 2) |i| {
        const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
        out[j * k + i] = v;
        out[j * k + k / 2 + i] = -v;
    };
}

// The dense and shared layers at decode rows: DeltaNet qkvz+ab, attention qkv+indexer, both outs, shared gate+up, down.
const bf16_served = [_]c.Shape{
    .{ .rows = 1, .n = 16480, .k = 2560 },  .{ .rows = 16, .n = 16480, .k = 2560 },
    .{ .rows = 1, .n = 13952, .k = 2560 },  .{ .rows = 64, .n = 13952, .k = 2560 },
    .{ .rows = 1, .n = 2560, .k = 6144 },   .{ .rows = 16, .n = 2560, .k = 6144 },
    .{ .rows = 4, .n = 1280, .k = 2560 },   .{ .rows = 64, .n = 1280, .k = 2560 },
    .{ .rows = 1, .n = 2560, .k = 640 },    .{ .rows = 64, .n = 2560, .k = 640 },
};

/// The qmmf lane's BF16-weight product (16-, 32- and 64-row tiles, K slices as the lane plans them) as Flash Next calls it.
pub const Bf16 = struct {
    rig: Rig,
    lane: *const q.Lane,
    format: q.Out,
    control: bool = false,
    d: ?Dense = null,
    w: q.Weight = undefined,
    x: u64 = 0,
    out: u64 = 0,
    part: u64 = 0,

    const vt = g.hooks(Bf16);

    pub fn deinit(self: *Bf16) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Bf16) g.Kernel {
        const name = if (self.control) "control: Flash Next qmmf BF16 weights scaled 1.5x" else switch (self.format) {
            .bf16 => "cuda qmmf BF16 weights, bf16 out (Flash Next dense and shared layers)",
            .f32 => "cuda qmmf BF16 weights, fp32 out (Flash Next dense and shared layers)",
        };
        return .{ .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null, .spec = .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 1,
            .out = if (self.format == .f32) .f32 else .bf16,
            .tiles = .{ .{ .size = 16 }, .{ .size = 64 }, .{ .size = 64, .step = 64, .min = 64 } },
            .served = &bf16_served,
        } };
    }

    pub fn prepare(self: *Bf16, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 256);
        const vals = try a.alloc(f32, shape.n * shape.k);
        defer a.free(vals);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xb16d);
        const rng = prng.random();
        if (fill == .stress) stress(rng, shape.n, shape.k, vals) else for (vals) |*v| {
            v.* = rng.float(f32) * 4 - 2;
        }
        const bits = try a.alloc(u16, vals.len);
        defer a.free(bits);
        for (bits, vals, d.w) |*b, v, *w| {
            b.* = c.toBf16(v);
            w.* = c.fromBf16(b.*);
        }
        const npad = (shape.n + 127) / 128 * 128;
        const tiled = try a.alloc(u16, npad * shape.k);
        defer a.free(tiled);
        q.packBf16(tiled, bits, shape.n, shape.k);
        self.w = cuda.modelopt.bf16Weight(try self.rig.input(std.mem.sliceAsBytes(tiled)), shape.n, shape.k);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.out = try self.rig.output(d.outputs() * @as(usize, if (self.format == .f32) 4 else 2));
        const parts = self.lane.partBytes(shape.rows, self.w);
        self.part = if (parts == 0) 0 else try self.rig.scratch(parts);
        return true;
    }

    pub fn run(self: *Bf16) !void {
        const s = self.d.?.shape;
        var w = self.w;
        if (self.control) w.scale = 1.5;
        try self.lane.matmulOut(self.rig.s, self.x, s.k, s.rows, w, self.out, self.format, self.part);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Bf16) usize {
        return self.d.?.outputs();
    }
    pub fn reference(self: *Bf16, want: []f64, mag: []f64) !void {
        try self.d.?.reference(want, mag);
    }
    pub fn simulate(self: *Bf16, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};

const experts = 4; // routed experts in the fixture; every pair picks `target`, the rest hold NaN codes
const target = 2;

// The MTP head's routed down (2,560 x 640): pairs are tokens times ten picks, at decode and prompt row counts.
const decode_served = [_]c.Shape{ .{ .rows = 10, .n = 2560, .k = 640 }, .{ .rows = 40, .n = 2560, .k = 640 }, .{ .rows = 160, .n = 2560, .k = 640 } };
const prompt_served = [_]c.Shape{ .{ .rows = 160, .n = 2560, .k = 640 }, .{ .rows = 1280, .n = 2560, .k = 640 } };

/// fp8_experts.cu's down_f32 (decode or prompt kernel) on a grouped plan of one expert's pairs, as the MTP head calls it.
pub const Fp8Down = struct {
    rig: Rig,
    set: *const Set,
    prompt: bool,
    control: bool = false,
    d: ?Dense = null,
    layer: ex.Layer = undefined,
    plan: grouped.Plan = undefined,
    picks: u64 = 0,
    x: u64 = 0,
    out: u64 = 0,

    const vt = g.hooks(Fp8Down);

    pub fn deinit(self: *Fp8Down) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Fp8Down) g.Kernel {
        const name = if (self.control) "control: Flash Next FP8 expert down at 1.5x block scales" else if (self.prompt)
            "cuda fp8_experts prompt down fp32 (Flash Next MTP head)"
        else
            "cuda fp8_experts decode down fp32 (Flash Next MTP head)";
        const rows: usize = if (self.prompt) ex.promptTile(.fp8g) else 16;
        return .{ .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null, .spec = .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 1,
            .out = .f32,
            .tiles = .{ .{ .size = rows }, .{ .size = ex.Fp8Experts.cols, .step = 128, .min = 128 }, .{ .size = 128, .step = 128, .min = 128 } },
            .served = if (self.prompt) &prompt_served else &decode_served,
        } };
    }

    pub fn prepare(self: *Fp8Down, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const n = shape.n;
        const k = shape.k;
        if (shape.rows == 0 or n == 0 or k == 0 or n % 128 != 0 or k % 128 != 0) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 256);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xf8d0);
        const rng = prng.random();
        // E4M3 codes [experts][n][k]: the target's values, every other expert's codes NaN so a misread goes red
        const codes = try a.alloc(u8, experts * n * k);
        defer a.free(codes);
        @memset(codes, 0x7f);
        const mine = codes[target * n * k ..][0 .. n * k];
        if (fill == .stress) {
            const vals = try a.alloc(f32, n * k);
            defer a.free(vals);
            stress(rng, n, k, vals);
            for (mine, vals) |*code, v| code.* = try codeOf(v);
        } else for (mine) |*code| {
            code.* = rng.int(u8);
            if (code.* & 0x7f == 0x7f) code.* ^= 1;
        }
        // f32 block scales [experts][1][n/128][k/128]: 1 for sums the stress inputs keep exact, else 2^-8 to 2^-5
        const blocks = (n / 128) * (k / 128);
        const scales = try a.alloc(f32, experts * blocks);
        defer a.free(scales);
        for (scales) |*s| s.* = if (fill == .stress) 1 else std.math.ldexp(@as(f32, 1), -5 - @as(i32, @intCast(rng.uintLessThan(u32, 4))));
        for (d.w, 0..) |*w, i| w.* = core.quant.fp8.e4m3(mine[i]) * scales[target * blocks + (i / k / 128) * (k / 128) + (i % k) / 128];
        if (self.control) for (scales) |*s| {
            s.* *= 1.5;
        };
        const packed_codes = try a.alloc(u32, experts * n * k / 4);
        defer a.free(packed_codes);
        for (0..experts) |e| ex.Fp8Experts.packOne(packed_codes[e * n * k / 4 ..][0 .. n * k / 4], codes[e * n * k ..][0 .. n * k], n, k);
        const picks = try a.alloc(i32, shape.rows);
        defer a.free(picks);
        @memset(picks, target);
        self.layer = .{
            .format = .fp8g,
            .up = 0,
            .down = try self.rig.input(std.mem.sliceAsBytes(packed_codes)),
            .down_scale = try self.rig.input(std.mem.sliceAsBytes(scales)),
            .width = k,
            .dims = n,
            .experts = experts,
        };
        self.picks = try self.rig.input(std.mem.sliceAsBytes(picks));
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        const pairs = shape.rows;
        const wide = pairs > grouped.small;
        const sizes = [5]usize{ pairs * 4, grouped.maxItems(pairs, experts, 16) * 12, 8, (if (wide) pairs else 1) * 4, (if (wide) (pairs + 1023) / 1024 * experts else 1) * 4 };
        var bufs: [5]u64 = undefined;
        for (sizes, &bufs) |len, *b| b.* = try self.rig.scratch(len);
        self.plan = .{ .members = bufs[0], .items = bufs[1], .counts = bufs[2], .rank = bufs[3], .hist = bufs[4] };
        self.out = try self.rig.output(d.outputs() * 4);
        return true;
    }

    /// The router's plan, then the down call: each pair's fp32 row of `n` outputs at its own row.
    pub fn run(self: *Fp8Down) !void {
        const pairs = self.d.?.shape.rows;
        const tile: usize = if (self.prompt) ex.promptTile(.fp8g) else 16;
        try self.set.router.route(self.rig.s, self.picks, pairs, experts, tile, self.plan);
        const in: ex.Rows = .{ .x = self.x, .stride = self.layer.width };
        try self.set.fp8.run(self.rig.s, .down_f32, in, self.layer, self.plan, self.prompt, self.out, grouped.maxItems(pairs, experts, tile), -1);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Fp8Down) usize {
        return self.d.?.outputs();
    }
    pub fn reference(self: *Fp8Down, want: []f64, mag: []f64) !void {
        try self.d.?.reference(want, mag);
    }
    pub fn simulate(self: *Fp8Down, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};

/// The E4M3 code of an integer the stress weights use (1 to 7 and their negatives).
fn codeOf(v: f32) !u8 {
    for (0..256) |i| if (core.quant.fp8.e4m3(@intCast(i)) == v) return @intCast(i);
    return error.NotE4m3;
}

/// The checkpoint's two kernel families, their kernels and controls appended to `list`.
pub const Set = struct {
    gpa: std.mem.Allocator,
    lane: q.Lane,
    fp8: ex.Fp8Experts,
    mod: cuda.Module,
    router: grouped.Router = undefined,
    bf16: [3]Bf16 = undefined,
    down: [3]Fp8Down = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        const cap = try gpu.ctx.capability();
        const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
        var lane = try q.Lane.load(gpu.d, cap / 10);
        errdefer lane.unload();
        var fp8 = try ex.Fp8Experts.load(gpu.d, sms);
        errdefer fp8.unload();
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
        errdefer mod.unload();
        set.* = .{ .gpa = gpu.gpa, .lane = lane, .fp8 = fp8, .mod = mod };
        set.router = try grouped.Router.resolve(set.mod);
        const r: Rig = .{ .gpu = gpu, .s = s };
        set.bf16 = .{ .{ .rig = r, .lane = &set.lane, .format = .bf16 }, .{ .rig = r, .lane = &set.lane, .format = .f32 }, .{ .rig = r, .lane = &set.lane, .format = .bf16, .control = true } };
        set.down = .{ .{ .rig = r, .set = set, .prompt = false }, .{ .rig = r, .set = set, .prompt = true }, .{ .rig = r, .set = set, .prompt = false, .control = true } };
        for (&set.bf16) |*x| try list.append(a, x.kernel());
        for (&set.down) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.down) |*x| x.deinit();
        for (&set.bf16) |*x| x.deinit();
        set.mod.unload();
        set.fp8.unload();
        set.lane.unload();
        set.gpa.destroy(set);
    }
};

/// These kernels' gates 1-4 and their controls alone (tf-cuda-test flashnext-modelopt-contract).
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6b63, &t);
    std.debug.print("SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "flashnext-modelopt-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
