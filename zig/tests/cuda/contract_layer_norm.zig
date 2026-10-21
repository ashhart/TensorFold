//! vit.cu's LayerNorm rows (2048 and 8192 wide) under the contract, and controls each dropping one point.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;
const bfr = core.bf16_round;

const eps: f32 = 1e-5;
const all = 3;

/// A registration: the 2048-wide kernel, the 8192-wide one, or a control on the wide rows keeping all points but one.
pub const Entry = struct { name: []const u8, wide: bool, keep: ?i32 = null };

pub const entries = [_]Entry{
    .{ .name = "cuda vit tf_vit_layer_norm (rows to 2048)", .wide = false },
    .{ .name = "cuda vit tf_vit_layer_norm_wide (rows to 8192)", .wide = true },
    .{ .name = "control: LayerNorm with the variance in one pass (E[x^2] - mean^2)", .wide = true, .keep = all & ~@as(i32, 1) },
    .{ .name = "control: LayerNorm rounding the normalized product before the bias", .wide = true, .keep = all & ~@as(i32, 2) },
};

/// A thread's chain of E adds, eight warps' butterflies, then the warps in order.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

pub fn spec(e: Entry) c.Spec {
    return .{
        .name = e.name,
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 2048, .step = 64, .min = 64 } },
        // the wide kernel: Gemma 4 12B's image embedder (patch_ln1 6912, patch_ln2 and pos_norm 3840, 280 or 70 rows)
        .served = if (e.wide) &.{ .{ .rows = 280, .n = 6912, .k = 6912 }, .{ .rows = 280, .n = 3840, .k = 3840 }, .{ .rows = 70, .n = 3840, .k = 3840 } } else &.{ .{ .rows = 64, .n = 1152, .k = 1152 }, .{ .rows = 64, .n = 2048, .k = 2048 } },
    };
}

pub const LayerNorm = struct {
    rig: rig_.Rig,
    entry: Entry,
    vit: *const cuda.vit.Kernels,
    ctl: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    rows: usize = 0,
    d: usize = 0,
    x: []const u16 = &.{},
    w: []const u16 = &.{},
    b: []const u16 = &.{},
    dev: [4]u64 = undefined,

    pub fn deinit(self: *LayerNorm) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(LayerNorm);

    pub fn kernel(self: *LayerNorm) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.keep != null) .shapes else null };
    }

    /// Rows offset by 0, 64 or 512 over noise in [-1, 1) (one in 64 twenty times wider); stress: zero-sum integers.
    pub fn prepare(self: *LayerNorm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const most: usize = if (self.entry.wide) 8192 else 2048;
        if (shape.k == 0 or shape.k > most or shape.k % 2 != 0 or shape.rows == 0 or shape.n != shape.k) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const d = shape.k;
        const x = try a.alloc(u16, shape.rows * d);
        const w = try a.alloc(u16, d);
        const b = try a.alloc(u16, d);
        for (w) |*v| v.* = bfr.bits(rng.float(f32) * 3 - 0.5);
        for (b) |*v| v.* = bfr.bits(rng.float(f32) * 6 - 3);
        for (0..shape.rows) |r| {
            const row = x[r * d ..][0..d];
            if (fill == .stress) { // integers in pairs v, -v: every sum and square exact in fp32, the mean 0
                for (0..d / 2) |j| {
                    const v: f32 = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 65))) - 32);
                    row[2 * j] = bfr.bits(v);
                    row[2 * j + 1] = bfr.bits(-v);
                }
                continue;
            }
            const offset: f32 = switch (r % 3) { // 512: a one-pass variance loses most of its digits
                0 => 0,
                1 => 64,
                else => 512,
            };
            for (row, 0..) |*v, i| v.* = bfr.bits(offset + (rng.float(f32) * 2 - 1) * (if (i % 64 == 9) @as(f32, 20) else 1));
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, usize, f32 }{ .{ 0, d / 3, std.math.nan(f32) }, .{ 1, d / 2, std.math.inf(f32) }, .{ 2, d - 1, -std.math.inf(f32) } };
            for (planted) |p| if (p[0] < shape.rows) {
                x[p[0] * d + p[1]] = bfr.bits(p[2]);
            };
        }
        self.rows = shape.rows;
        self.d = d;
        self.x = x;
        self.w = w;
        self.b = b;
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(x)), try self.rig.input(std.mem.sliceAsBytes(w)), try self.rig.input(std.mem.sliceAsBytes(b)), try self.rig.output(x.len * 2) };
        return true;
    }

    pub fn run(self: *LayerNorm) !void {
        const s = self.rig.s;
        if (self.ctl) |f| {
            var args: cuda.Args = .{};
            for (self.dev) |p| args.add(p);
            args.add(@as(c_int, @intCast(self.d)));
            args.add(eps);
            args.add(@as(c_int, self.entry.keep.?));
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(self.rows), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
        } else if (self.entry.wide) {
            try self.vit.layerNormWide(s, self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.rows, self.d, eps);
        } else try self.vit.layerNorm(s, self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.rows, self.d, eps);
        try s.synchronize();
    }

    pub fn count(self: *LayerNorm) usize {
        return self.rows * self.d;
    }

    /// Rows in float64 with each sum as `sum` gives it: the mean, then the squares about it.
    fn outputs(self: *LayerNorm, out: []f64, mag: ?[]f64, sum: anytype) !void {
        const a = self.arena.?.allocator();
        const d = self.d;
        const v = try a.alloc(f64, d);
        defer a.free(v);
        const t = try a.alloc(f64, d);
        defer a.free(t);
        for (0..self.rows) |r| {
            var abs: f64 = 0;
            for (v, self.x[r * d ..][0..d]) |*y, h| {
                y.* = bfr.value(h);
                abs += @abs(y.*);
            }
            const mean = sum.of(v) / @as(f64, @floatFromInt(d));
            for (t, v) |*y, x| y.* = (x - mean) * (x - mean);
            const rstd = 1 / @sqrt(sum.of(t) / @as(f64, @floatFromInt(d)) + eps);
            for (0..d) |i| {
                const term = (v[i] - mean) * rstd * bfr.value(self.w[i]);
                out[r * d + i] = term + bfr.value(self.b[i]);
                // the sums' error reaches an output through x - mean (the mean's) and rstd (the squares')
                if (mag) |m| m[r * d + i] = (2 * @abs(v[i] - mean) + abs / @as(f64, @floatFromInt(d))) * rstd * @abs(bfr.value(self.w[i]));
            }
        }
    }

    const Exact = struct {
        fn of(_: Exact, t: []const f64) f64 {
            var s: f64 = 0;
            for (t) |x| s += x;
            return s;
        }
    };

    const Sim = struct {
        acc: c.Accum,
        order: g.Order,
        fn of(s: Sim, t: []const f64) f64 {
            return rows_.sumIn(s.acc, s.order, t);
        }
    };

    pub fn reference(self: *LayerNorm, want: []f64, mag: []f64) !void {
        try self.outputs(want, mag, Exact{});
    }

    pub fn simulate(self: *LayerNorm, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.outputs(out, null, Sim{ .acc = acc, .order = order });
    }
};

test "every control keeps all but one point, each point once" {
    var seen: i32 = 0;
    for (entries) |e| if (e.keep) |k| {
        const dropped = all & ~k;
        try std.testing.expectEqual(@as(u32, 1), @popCount(dropped));
        seen |= dropped;
    };
    try std.testing.expectEqual(@as(i32, all), seen);
}
