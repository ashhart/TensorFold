//! MiniMax H3's bf16 Metal kernels under the contract, each against its own formula in float64 (metal_h3_ref.zig).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const h3k = @import("h3_kernels");
const rig_ = @import("rig.zig");
const ref = @import("metal_h3_ref.zig");

const c = contract.spec;
const g = contract.gates;

pub const Which = enum { rows_in, rows_out, norm, final_norm, gate_add, gate_mix, swiglu, swiglu_gate_first, heads, pool, tile_scores, coarse, attention, modulation };

/// The geometry the test kernels are compiled for: two heads, up to 16 tiles of 64 slots.
const heads = 2;
const tiles = 16;
const slots = tiles * 64;
const defines = std.fmt.comptimePrint("#define H3_HEADS {d}\n#define H3_SLOTS {d}\n#define H3_TILES {d}\n", .{ heads, slots, tiles });

fn kChain(s: c.Shape) usize {
    return s.k + 2;
}
/// A lane's chain over every 32nd input, then one simd_sum tree.
fn laneChain(s: c.Shape) usize {
    return (s.k + 31) / 32 + 6;
}
fn headChain(_: c.Shape) usize {
    return 4 + 6;
}
fn poolChain(_: c.Shape) usize {
    return 64;
}
fn tileChain(s: c.Shape) usize {
    return s.rows + 2;
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
        .rows_in => .{ .name = "metal h3 rows in h3_rows_in", .accum = .f32, .depth = kChain, .ops = 1, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{ .size = 64 }, .{} }, .served = &.{ .{ .rows = 64, .n = 5376, .k = 96 }, .{ .rows = 16, .n = 5376, .k = 32 } } },
        .rows_out => .{ .name = "metal h3 rows out h3_rows_out", .accum = .f32, .depth = laneChain, .ops = 1, .out = .f32, .tiles = .{ .{}, .{}, .{ .size = 32 } }, .served = &.{ .{ .rows = 32, .n = 96, .k = 5376 }, .{ .rows = 16, .n = 32, .k = 5376 } } },
        .norm => .{ .name = "metal h3 modulated norm h3_norm", .accum = .f32, .depth = laneChain, .ops = 8, .mids = 5, .out = .bf16, .in_place = true, .tiles = .{ .{}, .{}, .{ .size = 32 } }, .served = &.{.{ .rows = 16, .n = 5376, .k = 5376 }}, .sums = norm_sums },
        .final_norm => .{ .name = "metal h3 final norm h3_final_norm", .accum = .f32, .depth = laneChain, .ops = 8, .mids = 3, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 32 } }, .served = &.{.{ .rows = 16, .n = 5376, .k = 5376 }}, .sums = norm_sums },
        .gate_add => .{ .name = "metal h3 gated add h3_gate_add", .accum = .none, .depth = none, .mids = 1, .out = .bf16, .in_place = true, .served = &.{.{ .rows = 16, .n = 5376, .k = 1 }} },
        .gate_mix => .{ .name = "metal h3 pooled branch mix h3_gate_mix", .accum = .none, .depth = none, .mids = 1, .out = .bf16, .in_place = true, .tiles = .{ .{ .size = 64 }, .{}, .{} }, .served = &.{.{ .rows = slots, .n = heads * 128, .k = 1 }} },
        .swiglu => .{ .name = "metal h3 SwiGLU h3_swiglu", .accum = .none, .depth = none, .ops = 4, .mids = 2, .out = .bf16, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{.{ .rows = 16, .n = 14336, .k = 1 }} },
        .swiglu_gate_first => .{ .name = "metal h3 SwiGLU gate first h3_swiglu", .accum = .none, .depth = none, .ops = 4, .mids = 2, .out = .bf16, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{.{ .rows = 16, .n = 14336, .k = 1 }} },
        .heads => .{ .name = "metal h3 head layout h3_heads", .accum = .f32, .depth = headChain, .ops = 8, .mids = 2, .out = .bf16, .served = &.{ .{ .rows = slots, .n = heads * 128, .k = 64 }, .{ .rows = slots, .n = heads * 128, .k = 128 } }, .sums = norm_sums },
        .pool => .{ .name = "metal h3 tile means h3_pool", .accum = .f32, .depth = poolChain, .ops = 1, .out = .f32, .served = &.{.{ .rows = tiles, .n = 128, .k = 64 }}, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "a mean of at most 64 bf16 values: no sum reaches past fp16's range" } },
        .tile_scores => .{ .name = "metal h3 tile scores h3_tile_scores", .accum = .f32, .depth = kChain, .ops = 1, .out = .f32, .served = &.{.{ .rows = tiles, .n = tiles, .k = 128 }} },
        .coarse => .{ .name = "metal h3 pooled branch h3_coarse", .accum = .f32, .depth = tileChain, .ops = 6, .out = .f32, .tiles = .{ .{ .size = 4 }, .{}, .{} }, .served = &.{.{ .rows = tiles, .n = 128, .k = tiles }}, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = soft_why } },
        .modulation => .{ .name = "metal h3 AdaLN tables h3_modulation", .accum = .f32, .depth = laneChain, .ops = 1, .out = .bf16, .tiles = .{ .{}, .{ .size = 18 * 32, .step = 18, .min = 18 }, .{ .size = 32 } }, .served = &.{ .{ .rows = 16, .n = 18 * 64, .k = 2688 }, .{ .rows = 3, .n = 18 * 16, .k = 2688 } } },
        .attention => .{ .name = "metal h3 routed attention h3_attention", .accum = .f32, .depth = attnChain, .ops = 8, .mids = 1, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256, .step = 64, .min = 64 } }, .served = &.{ .{ .rows = tiles, .n = 128, .k = tiles * 64 }, .{ .rows = tiles, .n = 128, .k = 6 * 64 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = soft_why } },
    };
}

/// The kernel `w` launches: its function name in h3_kernels' sources.
fn function(w: Which) []const u8 {
    return switch (w) {
        .rows_in => "h3_rows_in",
        .rows_out => "h3_rows_out",
        .norm => "h3_norm",
        .final_norm => "h3_final_norm",
        .gate_add => "h3_gate_add",
        .gate_mix => "h3_gate_mix",
        .swiglu, .swiglu_gate_first => "h3_swiglu",
        .heads => "h3_heads",
        .pool => "h3_pool",
        .tile_scores => "h3_tile_scores",
        .coarse => "h3_coarse",
        .attention => "h3_attention",
        .modulation => "h3_modulation",
    };
}

const Slot = struct { ref: rig_.Ref, len: usize };

pub const H3 = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    arena: std.heap.ArenaAllocator,
    in: ref.In = .{ .rows = 0, .n = 0, .k = 0 },
    bufs: [12]rig_.Ref = undefined,
    nbufs: usize = 0,
    /// The in-place buffer's starting bytes, copied in before every launch.
    fresh: []const u8 = &.{},
    outs: usize = 0,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which) !H3 {
        const src = try std.mem.concat(a, u8, &.{ defines, h3k.source });
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = which, .pipe = try rig_.pipeline(device, src, function(which)), .arena = std.heap.ArenaAllocator.init(a) };
    }

    pub fn deinit(self: *H3) void {
        self.rig.deinit();
        self.pipe.deinit();
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
        const al = self.arena.allocator();
        const bytes = if (half) blk: {
            const h = try al.alloc(u16, v.len);
            for (h, v) |*o, x| o.* = c.toBf16(@floatCast(x));
            break :blk std.mem.sliceAsBytes(h);
        } else blk: {
            const f = try al.alloc(f32, v.len);
            for (f, v) |*o, x| o.* = @floatCast(x);
            break :blk std.mem.sliceAsBytes(f);
        };
        self.bufs[self.nbufs] = try self.rig.input(bytes);
        self.nbufs += 1;
    }

    pub fn ints(self: *H3, v: []const i32) !void {
        self.bufs[self.nbufs] = try self.rig.input(std.mem.sliceAsBytes(v));
        self.nbufs += 1;
    }

    pub fn output(self: *H3, bytes: usize) !void {
        self.bufs[self.nbufs] = try self.rig.output(bytes);
        self.nbufs += 1;
    }

    pub fn prepare(self: *H3, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.nbufs = 0;
        self.fresh = &.{};
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        return @import("metal_h3_prep.zig").prepare(self, shape, fill, prng.random());
    }

    pub fn run(self: *H3) !void {
        if (self.fresh.len > 0) @memcpy(self.rig.bytes(self.bufs[0], self.fresh.len), self.fresh);
        const p = self.rig.begin();
        try @import("metal_h3_prep.zig").launch(self, p.e);
        try self.rig.finish(p);
    }

    pub fn count(self: *H3) usize {
        return self.outs;
    }

    fn compute(self: *H3, s: ref.Sum, want: []f64, mag: []f64) !void {
        const scratch = try self.a.alloc(f64, @max(2 * self.in.n, 1024) + self.in.k);
        defer self.a.free(scratch);
        switch (self.which) {
            .rows_in, .rows_out => ref.dot(self.in, s, scratch, want, mag),
            .norm => ref.norm(self.in, s, scratch, want, mag),
            .final_norm => ref.finalNorm(self.in, s, scratch, want, mag),
            .gate_add => ref.gated(self.in, false, want, mag),
            .gate_mix => ref.gated(self.in, true, want, mag),
            .swiglu => ref.swiglu(self.in, false, want, mag),
            .swiglu_gate_first => ref.swiglu(self.in, true, want, mag),
            .heads => ref.heads(self.in, s, scratch, want, mag),
            .pool => ref.pool(self.in, s, scratch, want, mag),
            .tile_scores => ref.scores(self.in, s, scratch, want, mag),
            .coarse => ref.coarse(self.in, s, scratch, want, mag),
            .attention => ref.attention(self.in, s, scratch, want, mag),
            .modulation => ref.modulation(self.in, s, scratch, want, mag),
        }
    }

    pub fn reference(self: *H3, want: []f64, mag: []f64) !void {
        return self.compute(.{}, want, mag);
    }

    pub fn simulate(self: *H3, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.a.alloc(f64, out.len);
        defer self.a.free(mag);
        return self.compute(.{ .acc = acc, .order = order }, out, mag);
    }
};
