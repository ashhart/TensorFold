//! ModelOpt NVFP4 prompt projections in one grid, byte-equal to one prompt_lean launch per projection.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const gpu_check = @import("check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const nvfp4 = cuda.nvfp4;

fn noSums(_: c.Shape) usize {
    return 0;
}

const served = [_]c.Shape{
    .{ .rows = 2048, .n = 17408, .k = 5120 },
    .{ .rows = 4096, .n = 17408, .k = 5120 },
    .{ .rows = 129, .n = 17408, .k = 5120 },
    .{ .rows = 1, .n = 17408, .k = 5120 },
    .{ .rows = 33, .n = 130, .k = 256 },
};

const rule = "every output byte equals separate prompt_lean launches on the same BF16 input and NVFP4 weights";

fn widths(shape: c.Shape) []const usize {
    return switch (shape.n) {
        17408 => &.{ 17408, 17408 },
        130 => &.{ 130, 130, 130 },
        else => &.{},
    };
}

/// One NVFP4 projection's host fixture: e2m1 codes, e4m3 scales and the shared bf16 rows, packed as qmm.pack does.
const Fixture = struct {
    a: std.mem.Allocator,
    rows_n: usize,
    k: usize,
    x: []u16,
    codes: []u8,
    scales: []u8,

    const global_scale: f32 = 0.5;

    fn init(a: std.mem.Allocator, shape_rows: usize, n: usize, k: usize) !Fixture {
        return .{
            .a = a,
            .rows_n = shape_rows,
            .k = k,
            .x = try a.alloc(u16, shape_rows * k),
            .codes = try a.alloc(u8, n * k / 2),
            .scales = try a.alloc(u8, n * k / 16),
        };
    }

    fn deinit(f: *Fixture) void {
        f.a.free(f.x);
        f.a.free(f.codes);
        f.a.free(f.scales);
    }

    fn fill(f: *Fixture, mode: rows.Fill, seed: u64) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (f.x, 0..) |*v, i| v.* = if (i % 29 == 0) 0 else c.toBf16(rng.float(f32) * 4 - 2);
        for (f.codes) |*q| q.* = rng.int(u8);
        for (f.scales) |*s| s.* = @intCast(1 + rng.uintLessThan(u8, 0x7E)); // every normal e4m3 magnitude, no NaN byte
        switch (mode) {
            .random => {},
            .nonfinite => {
                const k = f.k;
                const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
                const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (at, put) |p, v| if (p[0] < f.rows_n) {
                    f.x[p[0] * k + p[1]] = c.toBf16(v);
                };
                f.scales[0] = 0x7F; // a NaN block scale: the projections' rows read it identically
            },
            .stress => {
                for (f.x) |*v| v.* = c.toBf16(rng.float(f32) * 2.0e18 + 1.0e17);
                for (f.codes) |*q| q.* = 0xEE; // every nibble -6
                for (f.scales) |*s| s.* = 0x7E; // 448
            },
        }
    }

    /// The words and scale tiles one prompt_lean launch reads, rows past n zero.
    fn packedWords(f: *Fixture, n: usize) ![]u32 {
        const out = try f.a.alloc(u32, nvfp4.padded(n) / 64 * (f.k / nvfp4.group) * 512);
        nvfp4.packWords(out, f.codes, n, f.k);
        return out;
    }

    fn packedScales(f: *Fixture, n: usize) ![]u8 {
        const out = try f.a.alloc(u8, nvfp4.padded(n) * (f.k / 16));
        nvfp4.packScales(out, f.scales, n, f.k);
        return out;
    }
};

pub const Product = struct {
    rig: rig_.Rig,
    prompt: *const nvfp4.Prompt,
    fixtures: [8]?Fixture = @splat(null),
    ws: [8]cuda.qmmf.Weight = undefined,
    outs: [8]u64 = undefined,
    base: u64 = 0,
    parts: usize = 0,
    rows_n: usize = 0,
    k: usize = 0,
    bytes: usize = 0,
    x: u64 = 0,

    pub fn kernel(self: *Product) g.Kernel {
        return .{ .spec = .{
            .name = "cuda qwen3.8 NVFP4 prompt launch fusion",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .replaces = "one prompt_lean launch per projection",
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
        for (&self.fixtures) |*slot| if (slot.*) |*owned| owned.deinit();
        self.fixtures = @splat(null);
        self.parts = 0;
        self.rig.reset();
    }

    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.drop();
        const ns = widths(shape);
        if (ns.len < 2 or ns.len > self.fixtures.len or shape.rows == 0 or shape.k % 64 != 0 or shape.k % 16 != 0) return false;
        const a = self.rig.gpu.gpa;
        self.rows_n = shape.rows;
        self.k = shape.k;
        self.parts = ns.len;
        self.bytes = 0;
        for (ns, 0..) |n, i| {
            var f = try Fixture.init(a, shape.rows, n, shape.k);
            try f.fill(fill, seed + i * 0x101);
            if (i > 0) @memcpy(f.x, self.fixtures[0].?.x); // every member reads the same input rows
            const words = try f.packedWords(n);
            defer a.free(words);
            const tiles = try f.packedScales(n);
            defer a.free(tiles);
            self.fixtures[i] = f;
            self.ws[i] = nvfp4.weight(try self.rig.input(std.mem.sliceAsBytes(words)), try self.rig.input(tiles), Fixture.global_scale, n, shape.k);
            self.outs[i] = try self.rig.output(shape.rows * n * 2);
            self.bytes += shape.rows * n * 2;
        }
        self.x = try self.rig.input(std.mem.sliceAsBytes(self.fixtures[0].?.x));
        self.base = try self.rig.scratch(self.bytes);
        return true;
    }

    fn members(self: *Product, outs: [8]u64) [8]nvfp4.Prompt.ManyMember {
        var ms: [8]nvfp4.Prompt.ManyMember = undefined;
        for (self.ws[0..self.parts], outs[0..self.parts], ms[0..self.parts]) |w, out, *m| m.* = .{ .w = w, .out = out };
        return ms;
    }

    pub fn run(self: *Product) !void {
        try self.prompt.matmulMany(self.rig.s, self.x, self.k, self.rows_n, self.members(self.outs)[0..self.parts], .bf16);
        try self.rig.s.synchronize();
    }

    /// The served pair with two equal-width outputs swapped in the parts: the planted control.
    pub fn runSwapped(self: *Product) !void {
        var outs = self.outs;
        find_pair: for (0..self.parts) |i| for (i + 1..self.parts) |j| if (self.ws[i].n == self.ws[j].n) {
            std.mem.swap(u64, &outs[i], &outs[j]);
            break :find_pair;
        };
        try self.prompt.matmulMany(self.rig.s, self.x, self.k, self.rows_n, self.members(outs)[0..self.parts], .bf16);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Product) ![]const u8 {
        var at: usize = 0;
        for (self.ws[0..self.parts], self.outs[0..self.parts]) |w, _| {
            try self.prompt.matmulTile(self.rig.s, self.x, self.k, self.rows_n, w, self.base + at, .bf16, .lean);
            at += self.rows_n * w.n * 2;
        }
        return self.rig.bytes(self.base, self.bytes);
    }

    pub fn check(self: *Product, _: g.Gate) !?[]const u8 {
        const got = try self.rig.gpu.gpa.dupe(u8, try self.rig.outputBytes());
        defer self.rig.gpu.gpa.free(got);
        const want = try self.replaced();
        return if (std.mem.eql(u8, got, want)) null else rule;
    }

    pub fn count(self: *Product) usize {
        return self.bytes;
    }

    pub fn reference(_: *Product, _: []f64, _: []f64) !void {}

    /// Downloads every fused output, so the fused arm pays the same host copy the lean arm pays.
    fn landed(self: *Product) !void {
        for (self.outs[0..self.parts], self.ws[0..self.parts]) |out, w| {
            _ = try self.rig.bytes(out, self.rows_n * w.n * 2);
        }
    }

    /// One round of an arm: the launches plus the output download both arms pay.
    fn round(self: *Product, fused: bool, io: std.Io) !i96 {
        const start = gpu_check.now(io);
        if (fused) {
            try self.run();
            try self.landed();
        } else _ = try self.replaced();
        return gpu_check.now(io) - start;
    }
};

/// Median of a small sample, sorted in place so one stalled round cannot set the result.
fn median(v: []i96) i96 {
    var i: usize = 1;
    while (i < v.len) : (i += 1) {
        const x = v[i];
        var j = i;
        while (j > 0 and v[j - 1] > x) : (j -= 1) v[j] = v[j - 1];
        v[j] = x;
    }
    return v[v.len / 2];
}

pub const Registrations = struct {
    good: Product,
    swapped: Product,

    pub fn init(gpu: gpu_check.Gpu, s: cuda.Stream, prompt: *const nvfp4.Prompt) Registrations {
        return .{
            .good = .{ .rig = .{ .gpu = gpu, .s = s }, .prompt = prompt },
            .swapped = .{ .rig = .{ .gpu = gpu, .s = s }, .prompt = prompt },
        };
    }

    pub fn deinit(self: *Registrations) void {
        self.good.deinit();
        self.swapped.deinit();
    }

    pub fn append(self: *Registrations, cap: u32, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        if (cap < 120) return;
        try kernels.append(a, self.good.kernel());
        try kernels.append(a, swappedKernel(&self.swapped));
    }
};

fn swappedKernel(product: *Product) g.Kernel {
    var k = product.kernel();
    k.spec.name = "control: NVFP4 prompt many swaps two equal-width output bindings";
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

/// The two commands this module serves: `nvfp4-prompt-many-contract` and `nvfp4-prompt-many-bench`.
pub fn dispatch(gpu: gpu_check.Gpu, cmd: []const u8) !void {
    if (std.mem.endsWith(u8, cmd, "-bench")) return bench(gpu);
    return run(gpu);
}

/// Focused gate run, kept separate so a live GPU need not retain every contract fixture at once.
pub fn run(gpu: gpu_check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prompt = try nvfp4.Prompt.load(gpu.d);
    defer prompt.unload();
    var registrations = Registrations.init(gpu, s, &prompt);
    defer registrations.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    const cap = try gpu.ctx.capability();
    try registrations.append(cap, gpu.gpa, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x1540, &tally);
    std.debug.print("SUMMARY prompt_many: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{
        tally.passed, tally.failed, tally.caught, tally.missed,
    });
    try gpu_check.expect(tally.problems() == 0, "prompt_many contract: {d} failures", .{tally.problems()});
}

/// Interleaved kernel-level screening: one lean round and one fused round per rep, medians per shape.
pub fn bench(gpu: gpu_check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prompt = try nvfp4.Prompt.load(gpu.d);
    defer prompt.unload();
    var product = Product{ .rig = .{ .gpu = gpu, .s = s }, .prompt = &prompt };
    defer product.deinit();
    const cases = [_]c.Shape{
        .{ .rows = 2048, .n = 17408, .k = 5120 },
        .{ .rows = 4096, .n = 17408, .k = 5120 },
        .{ .rows = 8192, .n = 17408, .k = 5120 },
    };
    const rounds = 9;
    for (cases, 0..) |shape, i| {
        _ = try product.prepare(shape, .random, 0x1540 + i);
        _ = try product.round(false, gpu.io); // warm both arms so neither pays the clock ramp alone
        _ = try product.round(true, gpu.io);
        var lean: [rounds]i96 = undefined;
        var fused: [rounds]i96 = undefined;
        for (0..rounds) |r| {
            lean[r] = try product.round(false, gpu.io);
            fused[r] = try product.round(true, gpu.io);
        }
        const a: f64 = @floatFromInt(median(&lean));
        const b: f64 = @floatFromInt(median(&fused));
        std.debug.print("BENCH prompt_many rows={d} parts={d}: lean={d:.3} ms fused={d:.3} ms ratio={d:.4}x median of {d}\n", .{
            shape.rows, product.parts, a / std.time.ns_per_ms, b / std.time.ns_per_ms, a / b, rounds,
        });
    }
}
