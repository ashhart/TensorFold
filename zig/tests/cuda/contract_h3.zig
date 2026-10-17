//! MiniMax H3's bf16 CUDA kernels (h3.cu) under the contract, each against its float64 formula (contract_h3_ref.zig).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_h3_ref.zig");
const prep = @import("contract_h3_prep.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

pub const Which = enum { norm, gate_add, gate_mix, swiglu, heads, pool, tile_scores, coarse, attention };

/// The geometry the tests run: two heads, up to 16 tiles of 64 slots.
pub const heads = 2;
pub const tiles = 16;
pub const slots = tiles * 64;

/// A block's sum of squares: each of 256 threads chains over every 256th input, then a shuffle and warp tree.
fn blockChain(s: c.Shape) usize {
    return (s.k + 255) / 256 + 8;
}
fn headChain(_: c.Shape) usize {
    return 32 + 2;
}
fn poolChain(_: c.Shape) usize {
    return 64;
}
/// A lane's four products, then the warp's five-step tree.
fn laneChain(_: c.Shape) usize {
    return 4 + 5 + 2;
}
/// Each lane's share of a row's weights, then the warp tree; the values' sum is one chain over every tile.
fn tileChain(s: c.Shape) usize {
    return s.rows + 6;
}
/// The tensor op's dot over 128 channels, then the value product's sum over every selected key.
fn attnChain(s: c.Shape) usize {
    return 128 + s.k + 4;
}
fn none(_: c.Shape) usize {
    return 0;
}

const soft_why = "softmax weights in [0, 1], rounded to bf16 before the value product: the bf16 output hides the sums' error";
const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn spec(w: Which) c.Spec {
    const norm_sums = c.Sums{ .bf16 = 0, .f16 = 0, .exact = false, .why = norm_why };
    return switch (w) {
        .norm => .{ .name = "cuda h3 modulated norm h3_norm", .accum = .f32, .depth = blockChain, .ops = 8, .mids = 5, .out = .bf16, .in_place = true, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &.{.{ .rows = 16, .n = 5376, .k = 5376 }}, .sums = norm_sums },
        .gate_add => .{ .name = "cuda h3 gated add h3_gate_add", .accum = .none, .depth = none, .ops = 1, .out = .bf16, .in_place = true, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{.{ .rows = 16, .n = 5376, .k = 1 }} },
        .gate_mix => .{ .name = "cuda h3 pooled branch mix h3_gate_mix", .accum = .none, .depth = none, .mids = 1, .out = .bf16, .in_place = true, .tiles = .{ .{ .size = 64 }, .{}, .{} }, .served = &.{.{ .rows = slots, .n = heads * 128, .k = 1 }} },
        .swiglu => .{ .name = "cuda h3 SwiGLU h3_swiglu", .accum = .none, .depth = none, .ops = 4, .mids = 1, .out = .bf16, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{.{ .rows = 16, .n = 14336, .k = 1 }} },
        .heads => .{ .name = "cuda h3 head layout h3_heads", .accum = .f32, .depth = headChain, .ops = 8, .mids = 4, .out = .bf16, .served = &.{.{ .rows = slots, .n = heads * 128, .k = 96 }}, .sums = norm_sums },
        .pool => .{ .name = "cuda h3 tile means h3_pool", .accum = .f32, .depth = poolChain, .ops = 1, .out = .f32, .served = &.{.{ .rows = tiles, .n = 128, .k = 64 }}, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "a mean of at most 64 bf16 values: no sum reaches past fp16's range" } },
        .tile_scores => .{ .name = "cuda h3 tile scores h3_tile_scores", .accum = .f32, .depth = laneChain, .ops = 1, .out = .f32, .tiles = .{ .{}, .{ .size = 8 }, .{} }, .served = &.{.{ .rows = tiles, .n = tiles, .k = 128 }} },
        .coarse => .{ .name = "cuda h3 pooled branch h3_coarse", .accum = .f32, .depth = tileChain, .ops = 6, .out = .f32, .tiles = .{ .{ .size = 32 }, .{}, .{} }, .served = &.{.{ .rows = tiles, .n = 128, .k = tiles }}, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = soft_why } },
        .attention => .{ .name = "cuda h3 routed attention h3_attention", .accum = .f32, .depth = attnChain, .ops = 8, .mids = 1, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256, .step = 64, .min = 64 } }, .served = &.{ .{ .rows = tiles, .n = 128, .k = tiles * 64 }, .{ .rows = tiles, .n = 128, .k = 6 * 64 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = soft_why } },
    };
}

/// The kernel `w` launches: its symbol in the h3 image.
pub fn symbol(w: Which) [:0]const u8 {
    return switch (w) {
        .norm => "h3_norm",
        .gate_add => "h3_gate_add",
        .gate_mix => "h3_gate_mix",
        .swiglu => "h3_swiglu",
        .heads => "h3_heads",
        .pool => "h3_pool",
        .tile_scores => "h3_tile_scores",
        .coarse => "h3_coarse",
        .attention => "h3_attention",
    };
}

pub const H3 = struct {
    rig: rig_.Rig,
    which: Which,
    f: cuda.Function,
    shape: c.Shape = undefined,
    arena: std.heap.ArenaAllocator,
    in: ref.In = .{ .rows = 0, .n = 0, .k = 0 },
    bufs: [12]u64 = undefined,
    nbufs: usize = 0,
    /// The in-place buffer's starting bytes, copied in before every launch.
    fresh: []const u8 = &.{},
    outs: usize = 0,

    pub fn deinit(self: *H3) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *H3) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(H3);

    /// `n` values uniform in [lo, hi), bf16-valued when `half`, some planted non-finite when `plant`.
    pub fn values(self: *H3, n: usize, lo: f64, hi: f64, fill: g.Fill, rng: std.Random, half: bool, plant: bool) ![]f64 {
        const v = try self.arena.allocator().alloc(f64, n);
        for (v, 0..) |*x, i| {
            const r = lo + (hi - lo) * rng.float(f64);
            x.* = switch (fill) {
                .stress => @floatFromInt(@as(i64, @intFromFloat(@round(r * 8)))),
                else => r,
            };
            if (i % 61 == 7 and fill != .stress) x.* *= 6;
            x.* = if (half) ref.bf(x.*) else @as(f32, @floatCast(x.*));
        }
        if (fill == .nonfinite and plant) {
            const at = [_]usize{ n / 3, n / 2, n - 1 };
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (at, put) |p, x| v[p] = x;
        }
        return v;
    }

    /// Upload `v` as bf16 (half) or fp32, as an input.
    pub fn upload(self: *H3, v: []const f64, half: bool) !void {
        try self.bytes(try self.encode(v, half));
    }

    pub fn encode(self: *H3, v: []const f64, half: bool) ![]const u8 {
        const al = self.arena.allocator();
        if (half) {
            const h = try al.alloc(u16, v.len);
            for (h, v) |*o, x| o.* = c.toBf16(@floatCast(x));
            return std.mem.sliceAsBytes(h);
        }
        const f = try al.alloc(f32, v.len);
        for (f, v) |*o, x| o.* = @floatCast(x);
        return std.mem.sliceAsBytes(f);
    }

    pub fn bytes(self: *H3, b: []const u8) !void {
        self.bufs[self.nbufs] = try self.rig.input(b);
        self.nbufs += 1;
    }

    pub fn ints(self: *H3, v: []const i32) !void {
        try self.bytes(std.mem.sliceAsBytes(v));
    }

    pub fn output(self: *H3, len: usize) !void {
        self.bufs[self.nbufs] = try self.rig.output(len);
        self.nbufs += 1;
    }

    pub fn prepare(self: *H3, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.nbufs = 0;
        self.fresh = &.{};
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        return prep.prepare(self, shape, fill, prng.random());
    }

    pub fn run(self: *H3) !void {
        if (self.fresh.len > 0) {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.bufs[0], .len = self.fresh.len };
            try b.upload(0, self.fresh);
        }
        try prep.launch(self);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *H3) usize {
        return self.outs;
    }

    fn compute(self: *H3, s: ref.Sum, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        const scratch = try a.alloc(f64, @max(2 * self.in.n, 1024) + self.in.k);
        defer a.free(scratch);
        switch (self.which) {
            .norm => ref.norm(self.in, s, scratch, want, mag),
            .gate_add => ref.gated(self.in, false, want, mag),
            .gate_mix => ref.gated(self.in, true, want, mag),
            .swiglu => ref.swiglu(self.in, want, mag),
            .heads => ref.heads(self.in, s, scratch, want, mag),
            .pool => ref.pool(self.in, s, scratch, want, mag),
            .tile_scores => ref.scores(self.in, s, scratch, want, mag),
            .coarse => ref.coarse(self.in, s, scratch, want, mag),
            .attention => ref.attention(self.in, s, scratch, want, mag),
        }
    }

    pub fn reference(self: *H3, want: []f64, mag: []f64) !void {
        return self.compute(.{}, want, mag);
    }

    pub fn simulate(self: *H3, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        return self.compute(.{ .acc = acc, .order = order }, out, mag);
    }
};
