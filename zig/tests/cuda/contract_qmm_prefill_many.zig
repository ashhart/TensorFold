//! GB10 affine-4 prompt projections in one grid, byte-equal to one shipped prompt launch per projection.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const gpu_check = @import("check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Affine = core.contract.affine.Affine;
const qlinear = cuda.qlinear;

fn noSums(_: c.Shape) usize {
    return 0;
}

const served = [_]c.Shape{
    .{ .rows = 17, .n = 12288, .k = 5120 },
    .{ .rows = 129, .n = 10240, .k = 5120 },
    .{ .rows = 257, .n = 17408, .k = 5120 },
    .{ .rows = 33, .n = 65, .k = 512 },
};

const rule = "every output byte equals separate qmm_prefill launches on the same BF16 input and affine-4 weights";

pub const Registrations = struct {
    good: Product,
    swapped: Product,

    pub fn init(gpu: gpu_check.Gpu, s: cuda.Stream, affine: *qlinear.Affine4Kernels) Registrations {
        return .{
            .good = .{ .rig = .{ .gpu = gpu, .s = s }, .affine = affine },
            .swapped = .{ .rig = .{ .gpu = gpu, .s = s }, .affine = affine },
        };
    }

    pub fn deinit(self: *Registrations) void {
        self.good.deinit();
        self.swapped.deinit();
    }

    pub fn append(self: *Registrations, cap: u32, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        if (cap != 121) return;
        try kernels.append(a, self.good.kernel());
        try kernels.append(a, swappedKernel(&self.swapped));
    }
};

/// Focused gate run, kept separate so a live GB10 need not retain every contract fixture at once.
pub fn run(gpu: gpu_check.Gpu) !void {
    const cap = try gpu.ctx.capability();
    try gpu_check.expect(cap == 121, "qmm_prefill many contract needs sm_121, got sm_{d}", .{cap});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const kk = cuda.kernels;
    var mods: [4]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack }, 0..) |image, i| {
        mods[i] = try cuda.Module.load(gpu.d, image);
        loaded += 1;
    }
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var affine = try qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, true);
    defer affine.deinit();
    var registrations = Registrations.init(gpu, s, &affine);
    defer registrations.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try registrations.append(cap, gpu.gpa, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x1398, &tally);
    std.debug.print("SUMMARY qmm_prefill_many: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{
        tally.passed, tally.failed, tally.caught, tally.missed,
    });
    try gpu_check.expect(tally.problems() == 0, "qmm_prefill many contract: {d} failures", .{tally.problems()});
}

/// Same-process A-B-A screening of separate launches against the fused launch; not an end-to-end claim.
pub fn bench(gpu: gpu_check.Gpu) !void {
    const cap = try gpu.ctx.capability();
    try gpu_check.expect(cap == 121, "qmm_prefill many benchmark needs sm_121, got sm_{d}", .{cap});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const kk = cuda.kernels;
    var mods: [4]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack }, 0..) |image, i| {
        mods[i] = try cuda.Module.load(gpu.d, image);
        loaded += 1;
    }
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var affine = try qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, true);
    defer affine.deinit();
    var product = Product{ .rig = .{ .gpu = gpu, .s = s }, .affine = &affine };
    defer product.deinit();
    const cases = [_]c.Shape{
        .{ .rows = 17, .n = 12288, .k = 5120 },   .{ .rows = 17, .n = 10240, .k = 5120 },
        .{ .rows = 17, .n = 17408, .k = 5120 },   .{ .rows = 64, .n = 12288, .k = 5120 },
        .{ .rows = 64, .n = 10240, .k = 5120 },   .{ .rows = 128, .n = 12288, .k = 5120 },
        .{ .rows = 128, .n = 10240, .k = 5120 },  .{ .rows = 128, .n = 17408, .k = 5120 },
        .{ .rows = 256, .n = 12288, .k = 5120 },  .{ .rows = 256, .n = 10240, .k = 5120 },
        .{ .rows = 512, .n = 12288, .k = 5120 },  .{ .rows = 512, .n = 10240, .k = 5120 },
        .{ .rows = 512, .n = 17408, .k = 5120 },  .{ .rows = 1024, .n = 12288, .k = 5120 },
        .{ .rows = 1024, .n = 10240, .k = 5120 }, .{ .rows = 2048, .n = 12288, .k = 5120 },
        .{ .rows = 2048, .n = 10240, .k = 5120 }, .{ .rows = 2048, .n = 17408, .k = 5120 },
        .{ .rows = 4096, .n = 12288, .k = 5120 }, .{ .rows = 4096, .n = 10240, .k = 5120 },
        .{ .rows = 4096, .n = 17408, .k = 5120 },
    };
    for (cases, 0..) |shape, i| {
        _ = try product.prepare(shape, .random, 0x1398 + i);
        try product.baselineRun();
        try product.run();
        try gpu_check.expect((try product.check(.shapes)) == null, "qmm_prefill many benchmark parity failed at {d}x{d}x{d}", .{ shape.rows, shape.n, shape.k });
        const a1 = try product.time(false, 7, gpu.io);
        const b = try product.time(true, 7, gpu.io);
        const a2 = try product.time(false, 7, gpu.io);
        const base: f64 = @floatFromInt(@divTrunc(a1 + a2, 2));
        const candidate: f64 = @floatFromInt(b);
        std.debug.print("BENCH qmm_prefill_many rows={d} first_n={d} parts={d}: A1={d:.3} ms B={d:.3} ms A2={d:.3} ms speedup={d:.4}x\n", .{
            shape.rows,                                       shape.n,          product.parts, @as(f64, @floatFromInt(a1)) / std.time.ns_per_ms, candidate / std.time.ns_per_ms,
            @as(f64, @floatFromInt(a2)) / std.time.ns_per_ms, base / candidate,
        });
    }
    const tile_cases = [_]c.Shape{
        .{ .rows = 2048, .n = 12288, .k = 5120 }, .{ .rows = 2048, .n = 10240, .k = 5120 }, .{ .rows = 2048, .n = 17408, .k = 5120 },
        .{ .rows = 4096, .n = 12288, .k = 5120 }, .{ .rows = 4096, .n = 10240, .k = 5120 }, .{ .rows = 4096, .n = 17408, .k = 5120 },
    };
    for (tile_cases, 0..) |shape, i| {
        _ = try product.prepare(shape, .random, 0x21398 + i);
        try product.separateRun(false, false);
        try product.separateRun(true, true);
        try gpu_check.expect((try product.checkOutputs()) == null, "qmm_prefill 2x2 tile parity failed at {d}x{d}x{d}", .{ shape.rows, shape.n, shape.k });
        const a1 = try product.timeSeparate(false, 7, gpu.io);
        const b = try product.timeSeparate(true, 7, gpu.io);
        const a2 = try product.timeSeparate(false, 7, gpu.io);
        const base: f64 = @floatFromInt(@divTrunc(a1 + a2, 2));
        const candidate: f64 = @floatFromInt(b);
        std.debug.print("BENCH qmm_prefill_tile22 rows={d} first_n={d} parts={d}: A1={d:.3} ms B={d:.3} ms A2={d:.3} ms speedup={d:.4}x\n", .{
            shape.rows,                                       shape.n,          product.parts, @as(f64, @floatFromInt(a1)) / std.time.ns_per_ms, candidate / std.time.ns_per_ms,
            @as(f64, @floatFromInt(a2)) / std.time.ns_per_ms, base / candidate,
        });
    }
}

fn widths(shape: c.Shape) []const usize {
    if (shape.k == 512 and shape.n == 65) return &.{ 65, 129 };
    return switch (shape.n) {
        12288 => &.{ 12288, 1024, 1024 },
        10240 => &.{ 10240, 6144, 48, 48 },
        17408 => &.{ 17408, 17408 },
        else => &.{},
    };
}

pub const Product = struct {
    rig: rig_.Rig,
    affine: *qlinear.Affine4Kernels,
    fixtures: [4]?Affine = @splat(null),
    qs: [4]qlinear.Affine4 = undefined,
    outs: [4]u64 = undefined,
    base: u64 = 0,
    parts: usize = 0,
    rows: usize = 0,
    k: usize = 0,
    bytes: usize = 0,
    x: u64 = 0,

    pub fn kernel(self: *Product) g.Kernel {
        return .{ .spec = .{
            .name = "cuda qmm_prefill many bit-exact BF16 launch fusion",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .replaces = "one qmm_prefill launch per projection",
            .nan = .{ .stated = rule },
            .served = &served,
        }, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Product);

    pub fn deinit(self: *Product) void {
        self.drop();
        self.rig.deinit();
    }

    fn drop(self: *Product) void {
        for (&self.fixtures) |*f| if (f.*) |owned| owned.deinit();
        self.fixtures = @splat(null);
        self.parts = 0;
        self.rig.reset();
    }

    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.drop();
        const ns = widths(shape);
        if (ns.len < 2 or ns.len > self.fixtures.len or shape.rows == 0 or shape.k % 64 != 0) return false;
        const a = self.rig.gpu.gpa;
        self.rows = shape.rows;
        self.k = shape.k;
        self.parts = ns.len;
        self.bytes = 0;
        for (ns, 0..) |n, i| {
            const f = try Affine.init(a, .{ .rows = shape.rows, .n = n, .k = shape.k }, 4, 64);
            self.fixtures[i] = f;
            try f.fill(fill, seed + i * 0x101);
            if (i > 0) @memcpy(f.x, self.fixtures[0].?.x);
            try self.pack(i, f);
            self.outs[i] = try self.rig.output(shape.rows * n * 2);
            self.bytes += shape.rows * n * 2;
        }
        self.x = try self.rig.input(std.mem.sliceAsBytes(self.fixtures[0].?.x));
        self.base = try self.rig.scratch(self.bytes);
        return true;
    }

    fn pack(self: *Product, i: usize, f: Affine) !void {
        const a = self.rig.gpu.gpa;
        const shape = f.shape;
        const words = try a.alloc(u32, shape.n * shape.k / 8);
        defer a.free(words);
        f.words(words);
        const lay = qlinear.Affine4.layout(shape.n, shape.k);
        const zeros = try a.alloc(u8, lay.words);
        defer a.free(zeros);
        @memset(zeros, 0);
        const q: qlinear.Affine4 = .{
            .w = try self.rig.input(zeros),
            .s = try self.rig.input(zeros[0..lay.scales]),
            .b = try self.rig.input(zeros[0..lay.scales]),
            .n = shape.n,
            .k = shape.k,
            .npad = lay.npad,
        };
        const src_w = try self.rig.scratch(words.len * 4);
        const src_s = try self.rig.scratch(f.scales.len * 2);
        const src_b = try self.rig.scratch(f.biases.len * 2);
        for ([_]u64{ src_w, src_s, src_b }, [_][]const u8{ std.mem.sliceAsBytes(words), std.mem.sliceAsBytes(f.scales), std.mem.sliceAsBytes(f.biases) }) |p, h| {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = p, .len = h.len };
            try b.upload(0, h);
        }
        try self.affine.pack(self.rig.s, src_w, src_s, src_b, q);
        self.qs[i] = q;
    }

    pub fn run(self: *Product) !void {
        var outs: [4]qlinear.Affine4Kernels.Out = undefined;
        for (outs[0..self.parts], 0..) |*o, i| o.* = .{ .q = self.qs[i], .out = self.outs[i] };
        try self.affine.promptMany(self.rig.s, self.x, outs[0..self.parts], self.rows);
        try self.rig.s.synchronize();
    }

    pub fn runSwapped(self: *Product) !void {
        const rows_t = (self.rows + 127) / 128;
        const band = (12 << 20) / (128 * self.k * 2);
        const group = @max(1, @min(rows_t, band));
        var parts: qlinear.PrefillParts = std.mem.zeroes(qlinear.PrefillParts);
        var sizes: [4]usize = @splat(0);
        var blocks: usize = 0;
        for (self.qs[0..self.parts], self.outs[0..self.parts], 0..) |q, out, i| {
            parts.p[i] = .{
                .w = q.w,
                .scales = q.s,
                .biases = q.b,
                .out = out,
                .n = @intCast(q.n),
                .npad = @intCast(q.npad),
                .first = @intCast(blocks),
            };
            sizes[i] = rows_t * ((q.n + 127) / 128);
            blocks += sizes[i];
        }
        parts.count = @intCast(self.parts);
        find_pair: for (0..self.parts) |i| for (i + 1..self.parts) |j| if (sizes[i] == sizes[j]) {
            std.mem.swap(u64, &parts.p[i].out, &parts.p[j].out);
            break :find_pair;
        };
        var args: cuda.Args = .{};
        args.add(self.x);
        args.add(parts);
        for ([_]usize{ self.rows, self.k, self.k, group }) |v| args.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(self.affine.prefill_many_fn, .{
            .grid = .{ .x = @intCast(blocks) },
            .block = .{ .x = 256 },
            .shared = qlinear.prefill_smem,
        }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn runHalf(self: *Product) !void {
        return self.separateRun(true, true);
    }

    pub fn runHalfSwapped(self: *Product) !void {
        self.affine.half_prompt = true;
        defer self.affine.half_prompt = false;
        var mapped = self.outs;
        find_pair: for (0..self.parts) |i| for (i + 1..self.parts) |j| if (self.qs[i].n == self.qs[j].n) {
            std.mem.swap(u64, &mapped[i], &mapped[j]);
            break :find_pair;
        };
        for (self.qs[0..self.parts], mapped[0..self.parts]) |q, out|
            try self.affine.prompt(self.rig.s, self.x, q, out, self.rows);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Product) ![]const u8 {
        try self.baselineRun();
        return self.rig.bytes(self.base, self.bytes);
    }

    fn baselineRun(self: *Product) !void {
        try self.separateRun(false, false);
    }

    fn separateRun(self: *Product, half: bool, outputs: bool) !void {
        return self.separateRunAt(half, outputs, 0);
    }

    fn separateRunAt(self: *Product, half: bool, outputs: bool, shift: u64) !void {
        self.affine.half_prompt = half;
        defer self.affine.half_prompt = false;
        var at: usize = 0;
        for (self.qs[0..self.parts], 0..) |q, i| {
            try self.affine.prompt(self.rig.s, self.x, q, if (outputs) self.outs[i] + shift else self.base + at, self.rows);
            at += self.rows * q.n * 2;
        }
        try self.rig.s.synchronize();
    }

    fn time(self: *Product, candidate: bool, reps: usize, io: std.Io) !i96 {
        const start = gpu_check.now(io);
        for (0..reps) |_| if (candidate) try self.run() else try self.baselineRun();
        return @divTrunc(gpu_check.now(io) - start, @as(i96, @intCast(reps)));
    }

    fn timeSeparate(self: *Product, half: bool, reps: usize, io: std.Io) !i96 {
        const start = gpu_check.now(io);
        for (0..reps) |_| try self.separateRun(half, false);
        return @divTrunc(gpu_check.now(io) - start, @as(i96, @intCast(reps)));
    }

    fn checkOutputs(self: *Product) !?[]const u8 {
        const got = try self.rig.gpu.gpa.dupe(u8, try self.rig.outputBytes());
        defer self.rig.gpu.gpa.free(got);
        const want = try self.rig.bytes(self.base, self.bytes);
        return if (std.mem.eql(u8, got, want)) null else rule;
    }

    pub fn check(self: *Product, _: c.Gate) !?[]const u8 {
        const got = try self.rig.gpu.gpa.dupe(u8, try self.rig.outputBytes());
        defer self.rig.gpu.gpa.free(got);
        const want = try self.replaced();
        return if (std.mem.eql(u8, got, want)) null else rule;
    }

    pub fn count(self: *Product) usize {
        return self.bytes;
    }

    pub fn reference(_: *Product, _: []f64, _: []f64) !void {}
};

fn swappedKernel(product: *Product) g.Kernel {
    var k = product.kernel();
    k.spec.name = "control: qmm_prefill many swaps two equal-width output bindings";
    // Product.check snapshots bad bytes before the replacement comparison reuses the rig readback buffer.
    k.spec.replaces = null;
    k.vt = &swapped_vt;
    k.control = .shapes;
    return k;
}

const swapped_vt: g.VTable = blk: {
    var vt = g.hooks(Product);
    vt.run = runSwapped;
    break :blk vt;
};

fn runSwapped(ctx: *anyopaque) anyerror!void {
    const product: *Product = @ptrCast(@alignCast(ctx));
    return product.runSwapped();
}
