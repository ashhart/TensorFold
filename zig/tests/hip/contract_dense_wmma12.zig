//! dense_wmma12_gemm under the contract on gfx12: BF16 x times raw BF16 weights on the WMMA tile, fp32 and BF16 out.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const Shape = hip.launches.dense.Shape;

/// k products, each exact in fp32's normal range, summed through at most k - 1 fp32 roundings in any order.
fn depth(s: c.Shape) usize {
    return @FieldType(hip.Launcher, "dense").depthWmma12(s.k);
}

/// Qwen3-8B's projections at rows 1, 7, 64, 300 and 2,048, then the blocks' edges and each routing threshold's.
const served = [_]c.Shape{
    .{ .rows = 1, .n = 4096, .k = 4096 },    .{ .rows = 7, .n = 4096, .k = 4096 },     .{ .rows = 64, .n = 4096, .k = 4096 },
    .{ .rows = 300, .n = 4096, .k = 4096 },  .{ .rows = 2048, .n = 4096, .k = 4096 },  .{ .rows = 1, .n = 1024, .k = 4096 },
    .{ .rows = 7, .n = 1024, .k = 4096 },    .{ .rows = 64, .n = 1024, .k = 4096 },    .{ .rows = 300, .n = 1024, .k = 4096 },
    .{ .rows = 2048, .n = 1024, .k = 4096 }, .{ .rows = 1, .n = 12288, .k = 4096 },    .{ .rows = 7, .n = 12288, .k = 4096 },
    .{ .rows = 64, .n = 12288, .k = 4096 },  .{ .rows = 300, .n = 12288, .k = 4096 },  .{ .rows = 2048, .n = 12288, .k = 4096 },
    .{ .rows = 1, .n = 4096, .k = 12288 },   .{ .rows = 7, .n = 4096, .k = 12288 },    .{ .rows = 64, .n = 4096, .k = 12288 },
    .{ .rows = 300, .n = 4096, .k = 12288 }, .{ .rows = 2048, .n = 4096, .k = 12288 }, .{ .rows = 15, .n = 63, .k = 128 },
    .{ .rows = 17, .n = 65, .k = 256 },      .{ .rows = 65, .n = 12288, .k = 4096 },   .{ .rows = 96, .n = 4096, .k = 128 },
    .{ .rows = 97, .n = 4096, .k = 128 },    .{ .rows = 129, .n = 130, .k = 4096 },    .{ .rows = 256, .n = 1024, .k = 128 },
    .{ .rows = 257, .n = 1024, .k = 128 },
};

/// The planted controls: each breaks one thing the served tile relies on, and must go red at its gate.
pub const Plant = enum {
    /// Each 16-deep step summed in BF16 (the small block): red at gate 4.
    bf16_sum,
    /// The big block's operand loads ignoring the stores' swizzle: red at gate 3 against the frozen block.
    unswizzled,
    /// The big block without its K loop's barriers: red at gate 3 against the frozen block.
    unordered,

    pub fn symbol(p: Plant) [:0]const u8 {
        return switch (p) {
            .bf16_sum => "tf_ctl_wmma12_bf16_sum",
            .unswizzled => "tf_ctl_wmma12_unswizzled",
            .unordered => "tf_ctl_wmma12_unordered",
        };
    }

    fn name(p: Plant) []const u8 {
        return switch (p) {
            .bf16_sum => "control: WMMA dense tile summed in bf16",
            .unswizzled => "control: WMMA big block loading unswizzled rows",
            .unordered => "control: WMMA big block without its K-step barriers",
        };
    }

    fn gate(p: Plant) g.Gate {
        return if (p == .bf16_sum) .sums else .shapes;
    }
};

/// One geometry's registration: every served shape runs on it, its own block's edges included.
pub fn spec(wide: bool, geo: Shape, planted: ?Plant) c.Spec {
    const small = geo == .w12_small;
    return .{
        .name = if (planted) |p| p.name() else switch (geo) {
            .w12_small => if (wide) "hip dense_wmma12_gemm<W12Small> (bf16 weights, fp32 out)" else "hip dense_wmma12_gemm<W12Small> (bf16 weights, bf16 out)",
            else => if (wide) "hip dense_wmma12_gemm<W12Big> (bf16 weights, fp32 out)" else "hip dense_wmma12_gemm<W12Big> (bf16 weights, bf16 out)",
        },
        .accum = .f32,
        .depth = depth,
        .out = if (wide) .f32 else .bf16,
        .served = &served,
        .tiles = .{ .{ .size = if (small) 16 else 128 }, .{ .size = if (small) 64 else 128 }, .{ .size = 12288, .step = 128, .min = 128 } },
        // the big block's 64-deep swizzled steps must keep the bytes of the 32-deep block they replace (#2264)
        .replaces = if (small) null else "dense_wmma12_gemm<W12Geo<128, 128, 32, 2, 4>> (tf_ref_wmma12_k32)",
    };
}

/// Launches of the unordered control a shape: gate 3 counts it red if any differs from the frozen block.
const race_runs = 8;

pub const DenseWmma12 = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    /// fp32 out, or BF16 rounded once in the epilogue.
    wide: bool,
    /// The block geometry every shape runs on (both give equal bytes; each is gated on its own).
    geo: Shape,
    /// A planted control (its kernel from the controls' object, its geometry's arguments) in place of the served one.
    planted: ?hip.Function = null,
    plant: Plant = .bf16_sum,
    /// The big block's byte baseline (tf_ref_wmma12_k32, a frozen copy of the block it replaces).
    baseline: ?hip.Function = null,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, wide: bool, geo: Shape) DenseWmma12 {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .wide = wide, .geo = geo };
    }

    pub fn deinit(self: *DenseWmma12) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *DenseWmma12) g.Kernel {
        const plant: ?Plant = if (self.planted != null) self.plant else null;
        return .{ .spec = spec(self.wide, self.geo, plant), .ctx = self, .vt = &vt, .control = if (plant) |p| p.gate() else null };
    }

    const vt = g.hooks(DenseWmma12);

    /// x from the dense family; W as BF16 in [-1, 1), or the stress fill's mirrored integers; only gfx12 serves it.
    pub fn prepare(self: *DenseWmma12, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 128 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.a;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        const bits = try a.alloc(u16, d.w.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xD3);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        if (fill == .nonfinite and shape.n > 5) {
            bits[1 * k + k / 4] = c.toBf16(std.math.nan(f32));
            bits[5 * k] = c.toBf16(std.math.inf(f32));
        }
        for (d.w, bits) |*w, b| w.* = c.fromBf16(b);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.out = try self.rig.output(d.outputs() * @as(usize, if (self.wide) 4 else 2));
        return true;
    }

    pub fn run(self: *DenseWmma12) !void {
        const s = self.d.?.shape;
        const out32: u64 = if (self.wide) self.out else 0;
        const out16: u64 = if (self.wide) 0 else self.out;
        const m: c_int = @intCast(s.rows);
        const n: c_int = @intCast(s.n);
        const k: c_int = @intCast(s.k);
        const h = self.rig.s.handle;
        const f = self.planted orelse {
            try self.l.dense.gemmOn(self.l.r, .wmma12, self.geo, self.x, self.w, out32, out16, m, n, k, h);
            return self.rig.s.synchronize();
        };
        // the unordered control is a race: up to `race_runs` launches, to the first that differs from the baseline
        const runs: usize = if (self.plant == .unordered) race_runs else 1;
        for (0..runs) |_| {
            try self.l.dense.gemmWith(self.l.r, f, self.geo, self.x, self.w, out32, out16, m, n, k, h);
            try self.rig.s.synchronize();
            if (runs == 1) return;
            const out = try self.rig.a.dupe(u8, try self.rig.outputBytes());
            defer self.rig.a.free(out);
            if (!std.mem.eql(u8, out, try self.replaced())) return;
        }
    }

    /// The replaced block's bytes on the prepared inputs: the frozen 32-deep block into its own buffer.
    pub fn replaced(self: *DenseWmma12) ![]const u8 {
        const s = self.d.?.shape;
        const f = self.baseline orelse return error.NoBaseline;
        const len = self.d.?.outputs() * @as(usize, if (self.wide) 4 else 2);
        const base = try self.rig.scratch(len);
        const out32: u64 = if (self.wide) base else 0;
        const out16: u64 = if (self.wide) 0 else base;
        try self.l.dense.gemmWith(self.l.r, f, .w12_big, self.x, self.w, out32, out16, @intCast(s.rows), @intCast(s.n), @intCast(s.k), self.rig.s.handle);
        try self.rig.s.synchronize();
        return self.rig.bytes(base, len);
    }

    /// The design's absolute term: a product below fp32's normal range may round or flush, 2^-126 each, k of them.
    pub fn bounds(self: *DenseWmma12, _: []const f64, _: []const f64, allow: []f64) !void {
        const floor = @as(f64, @floatFromInt(self.d.?.shape.k)) * 0x1p-126 * (1 + 0x1p-8);
        for (allow) |*l| l.* += floor;
    }

    pub fn count(self: *DenseWmma12) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *DenseWmma12, want: []f64, mag: []f64) !void {
        try self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *DenseWmma12, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};
