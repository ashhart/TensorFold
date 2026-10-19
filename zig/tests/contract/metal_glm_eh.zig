//! GLM-5.3-Flash's eh_proj on the tensor units (glm_rows_bf16_nax) under the contract.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const tf = @import("tensorfold");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;

/// One 8-wide MMA step a k-tile (its products round as they enter the fp32 dot), the tiles in turn the accumulator.
/// The guard the chain's tail carries: k/8 steps at k = 8192.
fn depth(s: c.Shape) usize {
    return s.k / 8 + 8;
}

/// The engine's prompt chunks: every full chunk height (prompt.zig's heights, at 16384 on the roomiest Macs).
/// And the last chunk's tail (absorb = remaining - 1, any positive count): 8 first, so the gates run a whole tile.
const served = [_]c.Shape{
    .{ .rows = 8, .n = 4096, .k = 8192 },
    .{ .rows = 2048, .n = 4096, .k = 8192 },
    .{ .rows = 4096, .n = 4096, .k = 8192 },
    .{ .rows = 6144, .n = 4096, .k = 8192 },
    .{ .rows = 8192, .n = 4096, .k = 8192 },
    .{ .rows = 16384, .n = 4096, .k = 8192 },
    .{ .rows = 1, .n = 4096, .k = 8192 },
};

pub fn spec() c.Spec {
    return .{
        .name = "metal glm eh_proj tensor tile glm_rows_bf16_nax",
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        // Rows: the launcher's ceil-div grid takes any positive count (the last chunk's tail).

        // n moves in 64-column tiles; k is an interior loop with no tiling of its own.
        .tiles = .{ .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 64, .step = 64, .min = 64 }, .{} },
        .served = &served,
    };
}

/// GlmRowDims as glm_rows.metal binds it at buffer 4.
const RowDims = extern struct {
    k: i32,
    n: i32,
    group: i32,
    rows: i32,
    heads: i32 = 1,
    ldx: i32,
    hsx: i32 = 0,
    ldo: i32,
    hso: i32 = 0,
    hsw: i32 = 0,
};

pub const Nax = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    d: ?Dense = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Nax {
        // The engine's own compilation unit for this pipeline: wide.zig's gather_source.

        // (core_row_projection, glue_math, glm_rows and glm_gather concatenated.)
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try rig_.pipeline(device, tf.glm.wide.gather_source, "glm_rows_bf16_nax"),
        };
    }

    pub fn deinit(self: *Nax) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
        self.pipe.deinit();
    }

    pub fn kernel(self: *Nax) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Nax);

    pub fn prepare(self: *Nax, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.n % 64 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const d = try Dense.init(self.a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 2);
        // W as bf16 bits: values in [-1, 1), or gate 4's mirrored integers, half a row the negative of the other half.

        // (The stress sums cancel and the partials stay under 2^24.)
        const bits = try self.a.alloc(u16, shape.n * shape.k);
        defer self.a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6d);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*wv, b| wv.* = c.fromBf16(b);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.out = try self.rig.output(d.outputs() * c.size(spec().out));
        return true;
    }

    pub fn run(self: *Nax) !void {
        const d = self.d.?;
        const p = self.rig.begin();
        const e = p.e;
        e.setPipeline(self.pipe);
        e.setBuffer(self.x.buf, self.x.off, 0);
        e.setBuffer(self.w.buf, self.w.off, 1);
        e.setValue(RowDims{
            .k = @intCast(d.shape.k),
            .n = @intCast(d.shape.n),
            .group = 0,
            .rows = @intCast(d.shape.rows),
            .ldx = @intCast(d.shape.k),
            .ldo = @intCast(d.shape.n),
        }, 4);
        e.setBuffer(self.out.buf, self.out.off, 5);
        e.dispatchGroups(
            mtl.Size.of(@intCast((d.shape.n + 63) / 64), @intCast((d.shape.rows + 7) / 8), 1),
            mtl.Size.of(8 * 32, 1, 1),
        );
        try self.rig.finish(p);
    }

    pub fn count(self: *Nax) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Nax, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Nax, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};
