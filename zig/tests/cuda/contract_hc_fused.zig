//! BF16 writeback+norm: replacement bytes, guarded repeats, non-finite plants and FP32 sum witnesses.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check_ = @import("check.zig");
const hf = flash.hc_fused;
const rig_ = @import("contract_rig.zig");
const norm_ = @import("contract_hc_normalize.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const d = 2560;
const ss = 4;
const wide = d * ss;
const nc = d / 256;
const slots = 11;

const served_wb = [_]c.Shape{
    .{ .rows = 1, .n = d, .k = 1 },
    .{ .rows = 16, .n = d, .k = 8 },
    .{ .rows = 64, .n = d, .k = 8 },
    .{ .rows = 65, .n = d, .k = 8 },
    .{ .rows = 1536, .n = d, .k = 8 },
    .{ .rows = 16384, .n = d, .k = 8 },
};

fn depth(_: c.Shape) usize {
    return 32;
}

pub const Registrations = struct {
    gpu: check_.Gpu,
    s: cuda.Stream,
    glue: flash.Glue,
    fused: hf.Kernels,
    norms: [2]norm_.Normalize,
    fixtures: [4]Fused = undefined,
    initialized: usize = 0,

    pub fn init(gpu: check_.Gpu, s: cuda.Stream, hc: cuda.Module) !Registrations {
        var glue = try flash.Glue.load(gpu.d);
        errdefer glue.deinit();
        const cols = try hc.function("fn_hc_normalize_cols");
        const old = try hc.function("fn_hc_normalize");
        var norms: [2]norm_.Normalize = undefined;
        for (std.enums.values(norm_.Kind), &norms) |kind, *norm|
            norm.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .cols_fn = cols, .old_fn = old };
        return .{ .gpu = gpu, .s = s, .glue = glue, .fused = try hf.Kernels.resolve(hc), .norms = norms };
    }

    pub fn deinit(self: *Registrations) void {
        for (self.fixtures[0..self.initialized]) |*fixture| fixture.deinit();
        for (&self.norms) |*norm| norm.deinit();
        self.glue.deinit();
    }

    pub fn append(self: *Registrations, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        const modes = [_]hf.Mode{ .sums, .moe, .slices, .slices };
        for (&self.fixtures, modes, 0..) |*fixture, mode, i| {
            fixture.* = .{ .rig = .{ .gpu = self.gpu, .s = self.s }, .fused = &self.fused, .glue = &self.glue, .mode = mode, .control = i == 3 };
            self.initialized += 1;
            try kernels.append(a, fixture.kernel());
        }
        for (&self.norms) |*norm| try kernels.append(a, norm.kernel());
    }
};

pub fn run(gpu: check_.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var hc = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_hc);
    defer hc.unload();
    var registrations = try Registrations.init(gpu, s, hc);
    defer registrations.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try registrations.append(gpu.gpa, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x1754, &tally);
    std.debug.print("SUMMARY hc_writeback_norm: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{
        tally.passed, tally.failed, tally.caught, tally.missed,
    });
    try check_.expect(tally.problems() == 0, "HC writeback and normalize contract: {d} failures", .{tally.problems()});
}

/// The FP32 output prefix is PSS; its BF16 companion outputs follow in the same guarded payload.
pub const Fused = struct {
    rig: rig_.Rig,
    fused: *const hf.Kernels,
    glue: *const flash.Glue,
    mode: hf.Mode = .sums,
    control: bool = false,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    h: []u16 = &.{},
    dh: u64 = 0,
    other: u64 = 0,
    gamma: u64 = 0,
    inj: u64 = 0,
    y: u64 = 0,
    wts: u64 = 0,
    branch: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [180]u8 = undefined,

    pub fn deinit(self: *Fused) void {
        self.rig.deinit();
        if (self.arena) |*a| a.deinit();
    }

    const vt = g.hooks(Fused);

    pub fn kernel(self: *Fused) g.Kernel {
        const name = if (self.control) "control: Flash Next fused writeback with doubled gamma" else switch (self.mode) {
            .sums => "cuda Flash Next fused sums and normalize",
            .moe => "cuda Flash Next fused MoE writeback and normalize",
            .slices => "cuda Flash Next fused slice writeback and normalize",
        };
        return .{ .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null, .spec = .{
            .name = name,
            .accum = .f32,
            .out = .f32,
            .depth = depth,
            .replaces = "fn_hc_writeback then fn_hc_normalize_cols",
            .nan = .{ .stated = "all output bytes equal the existing launches on the same NaN and infinity plants" },
            .served = &served_wb,
            .tiles = .{ .{ .size = 64 }, .{}, .{} },
        } };
    }

    pub fn count(self: *const Fused) usize {
        return self.shape.rows * nc * ss;
    }

    fn prefix(self: *const Fused) usize {
        return self.count() * 4;
    }

    fn bytes(self: *const Fused) usize {
        return self.prefix() + self.shape.rows * 2 * wide * 2;
    }

    pub fn prepare(self: *Fused, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16384 or shape.n != d) return false;
        if (shape.k == 0 or shape.k > 8) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        self.shape = shape;
        self.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.out = try self.rig.output(self.bytes());
        self.base = try self.rig.scratch(self.bytes());
        self.h = try a.alloc(u16, shape.rows * wide);
        const gamma = try a.alloc(f32, wide);
        const twice = try a.alloc(f32, wide);
        const gates = try a.alloc(u16, shape.rows * ss);
        const branch = try a.alloc(f32, if (self.mode == .slices) shape.rows * d * shape.k else 0);
        const y = try a.alloc(f32, if (self.mode == .moe) shape.rows * slots * d else 0);
        const wts = try a.alloc(f32, if (self.mode == .moe) shape.rows * slots else 0);
        for (self.h, 0..) |*v, i| v.* = c.toBf16(if (fill == .stress) @floatFromInt(1 + i % 64) else rng.float(f32) * 2 - 1);
        for (gamma, twice) |*v, *w| {
            v.* = 0.5 + rng.float(f32);
            w.* = v.* * 2;
        }
        for (gates) |*v| v.* = c.toBf16(rng.float(f32) * 2);
        for (branch) |*v| v.* = if (fill == .stress) 0 else rng.float(f32) - 0.5;
        for (y) |*v| v.* = if (fill == .stress) 0 else rng.float(f32) - 0.5;
        for (wts) |*v| v.* = rng.float(f32) * 0.1;
        if (fill == .nonfinite) {
            self.h[0] = c.toBf16(std.math.nan(f32));
            self.h[d] = c.toBf16(std.math.inf(f32));
            self.h[d + 1] = c.toBf16(-std.math.inf(f32));
        }
        self.dh = try self.rig.input(std.mem.sliceAsBytes(self.h));
        self.gamma = try self.rig.input(std.mem.sliceAsBytes(gamma));
        self.other = try self.rig.input(std.mem.sliceAsBytes(twice));
        self.inj = try self.rig.input(std.mem.sliceAsBytes(gates));
        self.branch = try self.rig.input(std.mem.sliceAsBytes(branch));
        self.y = try self.rig.input(std.mem.sliceAsBytes(y));
        self.wts = try self.rig.input(std.mem.sliceAsBytes(wts));
        return true;
    }

    fn copyHidden(self: *Fused, dst: u64) !void {
        const driver = self.rig.gpu.d;
        try driver.check(driver.api.cuMemcpyDtoDAsync_v2(dst, self.dh, self.h.len * 2, self.rig.s.handle), "cuMemcpyDtoDAsync");
    }

    pub fn run(self: *Fused) !void {
        const h = self.out + self.prefix();
        try self.copyHidden(h);
        try self.fused.writeback(self.rig.s, .{ .h = h, .pss = self.out, .branch = self.branch, .inj = self.inj, .y = self.y, .wts = self.wts, .stride = self.shape.rows * d, .slices = self.shape.k, .mode = self.mode }, h + self.h.len * 2, if (self.control) self.other else self.gamma, self.shape.rows);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Fused) ![]const u8 {
        const rows = self.shape.rows;
        const h = self.base + self.prefix();
        try self.copyHidden(h);
        try self.glue.hcWriteback(self.rig.s, h, h, self.base, self.branch, self.inj, self.y, self.wts, rows * d, rows, d, ss, @backingInt(self.mode), 10, slots, self.shape.k);
        try self.glue.hcNormalize(self.rig.s, h, self.base, self.gamma, h + self.h.len * 2, 1e-6, rows, d, ss);
        return self.rig.bytes(self.base, self.bytes());
    }

    pub fn check(self: *Fused, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} differs from the existing chain", .{at});
        var plants: usize = 0;
        for (0..self.count()) |i| if (!std.math.isFinite(c.read(.f32, got, i))) {
            plants += 1;
        };
        if (plants == 0 or plants == self.count()) return "the plants must leave both finite and non-finite FP32 outputs";
        return null;
    }

    fn terms(self: *const Fused, index: usize, out: []f64) void {
        const stream = index % ss;
        const chunk = (index / ss) % nc;
        const row = index / (ss * nc);
        for (out, 0..) |*v, i| {
            const h: f64 = c.fromBf16(self.h[row * wide + stream * d + chunk * 256 + i]);
            v.* = h * h;
        }
    }

    /// Gate 4's branch is zero, so its PSS sums must retain the FP32 square sums.
    pub fn reference(self: *Fused, want: []f64, mag: []f64) !void {
        if (self.fill != .stress) return error.ExpectedStressFill;
        var buf: [256]f64 = undefined;
        const t = buf[0..256];
        for (want, mag, 0..) |*w, *m, i| {
            self.terms(i, t);
            w.* = 0;
            m.* = 0;
            for (t) |v| {
                w.* += v;
                m.* += @abs(v);
            }
        }
    }

    pub fn simulate(self: *Fused, acc: c.Accum, order: g.Order, out: []f64) !void {
        var buf: [256]f64 = undefined;
        const t = buf[0..256];
        for (out, 0..) |*v, i| {
            self.terms(i, t);
            v.* = core.contract.rows.sumIn(acc, order, t);
        }
    }
};

test "fused writeback square stress requires FP32 sums in every simulated order" {
    const a = std.testing.allocator;
    const h = try a.alloc(u16, wide);
    defer a.free(h);
    for (h, 0..) |*v, i| v.* = c.toBf16(@floatFromInt(1 + i % 64));
    var fixture = Fused{ .rig = undefined, .fused = undefined, .glue = undefined, .shape = .{ .rows = 1, .n = d, .k = 8 }, .fill = .stress, .h = h };
    var want: [nc * ss]f64 = undefined;
    var magnitude: [nc * ss]f64 = undefined;
    var got: [nc * ss]f64 = undefined;
    try fixture.reference(&want, &magnitude);
    const spec = fixture.kernel().spec;
    for (std.enums.values(g.Order)) |order| {
        try fixture.simulate(.f32, order, &got);
        try std.testing.expectEqualSlices(f64, &want, &got);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            try fixture.simulate(acc, order, &got);
            var miss: f64 = 0;
            for (got, want) |v, w| miss = @max(miss, if (std.math.isFinite(v)) @abs(v - w) / c.boundExact(spec, w) else std.math.inf(f64));
            try std.testing.expect(miss >= 100);
        }
    }
}

test "HC benchmark rows default to 4218 and stay within the prompt chunk bounds" {
    const parse = @import("flashnext_hc_fused_tests.zig").benchRows;
    try std.testing.expectEqual(@as(usize, 4218), try parse(null));
    for ([_][]const u8{ "1", "1536", "4218", "16384" }, [_]usize{ 1, 1536, 4218, 16384 }) |value, want|
        try std.testing.expectEqual(want, try parse(value));
    for ([_][]const u8{ "0", "16385", "-1", "1536x", "" }) |value|
        try std.testing.expectError(error.BadHcBenchRows, parse(value));
}
