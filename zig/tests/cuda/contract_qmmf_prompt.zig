//! Four BF16 prompt instances through replacement, nonfinite, alias and FP32-sum gates.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const Dense = core.contract.dense.Dense;
const q = cuda.qmmf;
const cases = @import("qmmf_prompt_tests.zig");

const unsplit = [_]c.Shape{
    .{ .rows = 513, .n = 2560, .k = 640 },
    .{ .rows = 512, .n = 10240, .k = 320 },
    .{ .rows = 512, .n = 129, .k = 512 },
    .{ .rows = 512, .n = 129, .k = 2560 },
    .{ .rows = 513, .n = 8240, .k = 2560 },
    .{ .rows = 1536, .n = 7296, .k = 2560 },
    .{ .rows = 4218, .n = 8240, .k = 2560 },
    .{ .rows = 512, .n = 16480, .k = 2560 },
    .{ .rows = 512, .n = 13952, .k = 2560 },
};
const fused_shapes = [_]c.Shape{
    .{ .rows = 512, .n = 129, .k = 2048 },
    .{ .rows = 512, .n = 129, .k = 6144 },
    .{ .rows = 513, .n = 2560, .k = 3072 },
    .{ .rows = 4218, .n = 2560, .k = 3072 },
    .{ .rows = 512, .n = 2560, .k = 6144 },
    .{ .rows = 513, .n = 640, .k = 2560 },
    .{ .rows = 512, .n = 324, .k = 10240 },
    .{ .rows = 512, .n = 10240, .k = 2560 },
    .{ .rows = 512, .n = 2560, .k = 2560 },
};
const fused_fp32 = fused_shapes ++ [_]c.Shape{
    .{ .rows = 1024, .n = 2560, .k = 2560 }, // MTP fc_hidden: four streams of a 256-token piece
    .{ .rows = 512, .n = 320, .k = 10240 },
    .{ .rows = 1536, .n = 320, .k = 10240 }, // final mixer under .all logits; padded N is 384
};

fn slices(shape: c.Shape, fused: bool) u32 {
    return if (fused) q.splitK(shape.n, shape.k) else 1;
}

fn depth(shape: c.Shape) usize {
    return shape.k / 16 + 24;
}

pub fn spec(out: q.Out, fused: bool, control: bool) c.Spec {
    return .{
        .name = if (control) "control: qmmf BF16 prompt scaled 1.5x" else switch (out) {
            .bf16 => if (fused) "cuda qmmf BF16 prompt 128x128 bf16 fused" else "cuda qmmf BF16 prompt 128x128 bf16 unsplit",
            .f32 => if (fused) "cuda qmmf BF16 prompt 128x128 fp32 fused" else "cuda qmmf BF16 prompt 128x128 fp32 unsplit",
        },
        .accum = .f32,
        .depth = depth,
        .ops = 1,
        .out = if (out == .f32) .f32 else .bf16,
        .replaces = "qmmf BF16 64x64 with identical pinned K slices",
        .tiles = .{ .{ .size = 128 }, .{ .size = 128, .min = 128 }, .{ .size = if (fused) 2048 else 512, .step = 256, .min = if (fused) 2048 else 512 } },
        .served = if (fused) (if (out == .f32) &fused_fp32 else &fused_shapes) else &unsplit,
    };
}

pub const Product = struct {
    rig: Rig,
    lane: *const q.Lane,
    format: q.Out,
    fused: bool,
    control: bool = false,
    d: ?Dense = null,
    w: q.Weight = undefined,
    x: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    part: u64 = 0,

    pub fn deinit(self: *Product) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }
    const vt = g.hooks(Product);
    pub fn kernel(self: *Product) g.Kernel {
        return .{ .spec = spec(self.format, self.fused, self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }
    fn width(self: *const Product) usize {
        return if (self.format == .f32) 4 else 2;
    }
    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        if (shape.rows == 0 or shape.n < 128 or shape.k % 64 != 0 or shape.k < (if (self.fused) @as(usize, 2048) else 320)) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 16);
        const weights = try a.alloc(u16, shape.n * shape.k);
        defer a.free(weights);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xb16);
        const random = prng.random();
        for (weights) |*v| v.* = c.toBf16(if (fill == .stress) @floatFromInt(1 + random.uintLessThan(u32, 7)) else random.float(f32) * 4 - 2);
        if (fill == .stress) for (0..shape.n) |j| for (shape.k / 2..shape.k) |i| {
            weights[j * shape.k + i] = weights[j * shape.k + i - shape.k / 2] | 0x8000;
        };
        for (d.w, weights) |*o, v| o.* = c.fromBf16(v);
        const npad = cuda.modelopt.padded(shape.n);
        const packed_weights = try a.alloc(u16, npad * shape.k);
        defer a.free(packed_weights);
        q.packBf16(packed_weights, weights, shape.n, shape.k);
        const codes = try self.rig.input(std.mem.sliceAsBytes(packed_weights));
        const sk = slices(shape, self.fused);
        self.w = try q.pinned(cuda.modelopt.bf16Weight(codes, shape.n, shape.k), sk);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        const bytes = self.count() * self.width();
        self.out = try self.rig.output(bytes);
        self.base = try self.rig.scratch(bytes);
        const partials = self.lane.partBytes(shape.rows, self.w);
        self.part = if (partials == 0) 0 else try self.rig.scratch(partials);
        if (shape.rows >= q.prompt_rows and (!q.promptEligible(shape.rows, self.w) or self.lane.prompt == null)) return error.PromptPathNotSelected;
        return true;
    }
    pub fn run(self: *Product) !void {
        const shape = self.d.?.shape;
        var w = self.w;
        if (self.control) w.scale = 1.5;
        if (shape.rows >= q.prompt_rows) {
            try self.lane.matmulOut(self.rig.s, self.x, shape.k, shape.rows, w, self.out, self.format, self.part);
        } else {
            // The harness shrinks single-shape gates; force the same instance instead of silently testing decode.
            const band: usize = @max(1, @min((shape.rows + 127) / 128, (12 << 20) / (128 * shape.k * 2)));
            var args: cuda.Args = .{};
            args.add(self.x);
            args.add(w.codes);
            args.add(w.scales);
            args.add(w.scale);
            args.add(self.out);
            args.add(@as(u64, 0));
            for ([_]usize{ shape.rows, shape.n, shape.k, w.sk, w.npad, shape.k, band }) |v| args.add(@as(i32, @intCast(v)));
            try cuda.launch.launch(self.lane.prompt.?[@backingInt(self.format)][@intFromBool(self.fused)], .{
                .grid = .{ .x = @intCast(((shape.rows + 127) / 128) * ((shape.n + 127) / 128)) },
                .block = .{ .x = 256 },
                .shared = q.prompt_shared,
            }, self.rig.s, &args);
        }
        try self.rig.s.synchronize();
    }
    pub fn replaced(self: *Product) ![]const u8 {
        var lane = self.lane.*;
        lane.prompt = null;
        const shape = self.d.?.shape;
        try lane.matmulOut(self.rig.s, self.x, shape.k, shape.rows, self.w, self.base, self.format, self.part);
        return self.rig.bytes(self.base, self.count() * self.width());
    }
    pub fn count(self: *Product) usize {
        return self.d.?.outputs();
    }
    pub fn reference(self: *Product, want: []f64, magnitude: []f64) !void {
        try self.d.?.reference(want, magnitude);
    }
    pub fn simulate(self: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};

pub const Registrations = struct {
    products: [5]Product,
    pub fn init(gpu: check.Gpu, s: cuda.Stream, lane: *const q.Lane) Registrations {
        var r: Registrations = undefined;
        for ([_]q.Out{ .bf16, .f32 }, 0..) |out, oi| for ([_]bool{ false, true }, 0..) |fused, fi| {
            r.products[oi * 2 + fi] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = lane, .format = out, .fused = fused };
        };
        r.products[4] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = lane, .format = .bf16, .fused = true, .control = true };
        return r;
    }
    pub fn deinit(self: *Registrations) void {
        for (&self.products) |*p| p.deinit();
    }
    pub fn append(self: *Registrations, cap: u32, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        if (cap / 10 != 12) return;
        for (&self.products) |*p| try kernels.append(a, p.kernel());
    }
};

pub fn run(gpu: check.Gpu) !void {
    const cap = try gpu.ctx.capability();
    try check.expect(cap / 10 == 12, "qmmf prompt contract requires sm_12x", .{});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var lane = try q.Lane.load(gpu.d, cap / 10);
    defer lane.unload();
    var regs = Registrations.init(gpu, s, &lane);
    defer regs.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try regs.append(cap, gpu.gpa, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0xb16, &tally);
    std.debug.print("SUMMARY qmmf prompt gates 1-4: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{ tally.passed, tally.failed, tally.caught, tally.missed });
    try check.expect(tally.problems() == 0, "qmmf prompt contract failed", .{});
}

test "qmmf prompt contract registers four production instances, served prompts and a replacement control" {
    const a = std.testing.allocator;
    var lane: q.Lane = undefined;
    var regs = Registrations.init(undefined, undefined, &lane);
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    try regs.append(89, a, &kernels);
    try std.testing.expectEqual(@as(usize, 0), kernels.items.len);
    try regs.append(121, a, &kernels);
    try std.testing.expectEqual(@as(usize, 5), kernels.items.len);
    inline for (.{ q.Out.bf16, q.Out.f32 }, 0..) |out, oi| inline for (.{ false, true }, 0..) |fused, fi| {
        const k = kernels.items[oi * 2 + fi];
        try std.testing.expect(k.spec.replaces != null and k.vt.replaced != null);
        try std.testing.expect(k.vt.simulate != null and k.control == null);
        const symbol = q.promptSymbol(out, fused);
        const record = for (core.contract.registry.all_cuda) |r| {
            if (std.mem.eql(u8, r.launch.symbol, symbol)) break r;
        } else return error.MissingPromptRecord;
        try std.testing.expectEqual(q.prompt_shared, record.launch.dynamic);
        try std.testing.expectEqual(@as(u32, 256), record.launch.threads);
        try std.testing.expectEqual(@as(u32, 120), record.launch.min_sm);
        try std.testing.expect(record.launch.opt_in);
        for (k.spec.served) |shape| {
            const sk = slices(shape, fused);
            const w = try q.pinned(cuda.modelopt.bf16Weight(0, shape.n, shape.k), sk);
            try std.testing.expect(shape.rows >= q.prompt_rows and q.promptEligible(shape.rows, w));
            try std.testing.expectEqual(fused, q.planFor(shape.rows, w, true).fused);
        }
    };
    try std.testing.expectEqual(c.Gate.shapes, kernels.items[4].control.?);
}

test "qmmf prompt contract covers current normal operators, actual fused slices and MTP hidden rows" {
    var seen: [9]bool = @splat(false);
    for (cases.normal) |projection| {
        const sk = q.splitK(projection.n, projection.k);
        const s = spec(projection.out, sk > 1, false);
        var found = false;
        for (s.served) |shape| {
            if (shape.n == projection.n and shape.k == projection.k) {
                try std.testing.expectEqual(sk, slices(shape, sk > 1));
                found = true;
            }
        }
        try std.testing.expect(found);
        seen[sk] = true;
    }
    for ([_]usize{ 1, 2, 4, 8 }) |sk| try std.testing.expect(seen[sk]);
    for ([_]usize{ 512, 1024 }) |rows_| {
        var found = false;
        for (spec(.f32, true, false).served) |shape| {
            if (shape.rows == rows_ and shape.n == cases.mtp_hidden.n and shape.k == cases.mtp_hidden.k) found = true;
        }
        try std.testing.expect(found);
    }
    // Vocabulary probes stay in the bounded byte runner, rather than expanding FP64 or repeat gates to full heads.
    for ([_]q.Out{ .bf16, .f32 }) |out| for (spec(out, false, false).served) |shape|
        try std.testing.expect(shape.n < 124160);
}
