//! The lane's half tiles (16x32, MXFP8 and BF16) under the contract: the 64-column tile's bits; control: 1.5x scale.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const Rig = @import("contract_rig.zig").Rig;

pub const Kind = enum { mxfp8, bf16, control };

fn depth(shape: c.Shape) usize {
    return shape.k / 16 + 24;
}

/// Flash Next's decode calls whose 64-column grids take half tiles, the column edges, and 16 rows.
const served = [_]c.Shape{
    .{ .rows = 1, .n = 640, .k = 2560 },  .{ .rows = 1, .n = 2560, .k = 640 }, .{ .rows = 16, .n = 640, .k = 2560 },
    .{ .rows = 1, .n = 324, .k = 10240 }, .{ .rows = 7, .n = 512, .k = 2560 }, .{ .rows = 3, .n = 96, .k = 512 },
};

pub fn spec(kind: Kind) c.Spec {
    return .{
        .name = switch (kind) {
            .mxfp8 => "cuda MXFP8 lane half tiles (32 columns, 16 rows)",
            .bf16 => "cuda BF16 lane half tiles (32 columns, 16 rows)",
            .control => "control: MXFP8 half tiles at a 1.5x tensor scale posing as the 64-column tile",
        },
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .replaces = "the 64-column 16-row tile",
        .tiles = .{ .{ .size = 16 }, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 512, .step = 64, .min = 64 } },
        .served = &served,
    };
}

pub const Half = struct {
    rig: Rig,
    lane: *cuda.qmmf.Lane,
    kind: Kind,
    d: ?Dense = null,
    weight: cuda.qmmf.Weight = undefined,
    x: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    part: u64 = 0,

    pub fn deinit(self: *Half) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Half) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt, .control = if (self.kind == .control) .shapes else null };
    }

    const vt = g.hooks(Half);

    fn mode(self: *const Half) cuda.qmmf.Mode {
        return if (self.kind == .bf16) .bf16 else .mxfp8;
    }

    pub fn prepare(self: *Half, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        self.part = 0;
        if (shape.rows == 0 or shape.rows > 16 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 16);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x4a1f);
        const random = prng.random();
        const n = shape.n;
        const k = shape.k;
        if (self.mode() == .bf16) {
            const w = try a.alloc(u16, n * k);
            defer a.free(w);
            for (w, 0..) |*v, i| {
                const f: f32 = if (fill == .stress) @floatFromInt(@as(i32, 1) + @as(i32, @intCast(random.uintLessThan(u32, 8)))) else random.float(f32) * 0.1 - 0.05;
                v.* = c.toBf16(f);
                _ = i;
            }
            if (fill == .stress) for (0..n) |j| for (k / 2..k) |i| {
                w[j * k + i] = w[j * k + i - k / 2] | 0x8000;
            };
            for (d.w, w) |*o, v| o.* = c.fromBf16(v);
            const tiled = try a.alloc(u16, cuda.nvfp4.padded(n) * k);
            defer a.free(tiled);
            cuda.qmmf.packBf16(tiled, w, n, k);
            self.weight = cuda.modelopt.bf16Weight(try self.rig.input(std.mem.sliceAsBytes(tiled)), n, k);
        } else {
            const codes = try a.alloc(u8, n * k);
            defer a.free(codes);
            const scales = try a.alloc(u8, n * k / 32);
            defer a.free(scales);
            for (scales) |*s| s.* = if (fill == .stress) 127 else 123 + random.uintLessThan(u8, 8);
            for (0..n) |j| for (0..k) |i| {
                codes[j * k + i] = if (fill == .stress) 0x38 + random.uintLessThan(u8, 24) else random.uintLessThan(u8, 0x7f);
                if (fill != .stress and random.boolean()) codes[j * k + i] |= 0x80;
            };
            if (fill == .stress) for (0..n) |j| for (k / 2..k) |i| {
                codes[j * k + i] = codes[j * k + i - k / 2] | 0x80;
            };
            for (0..n) |j| for (0..k) |i| {
                const scale = std.math.ldexp(@as(f64, 1), @as(i32, scales[j * (k / 32) + i / 32]) - 127);
                d.w[j * k + i] = core.quant.fp8.e4m3(codes[j * k + i]) * scale;
            };
            const npad = cuda.mxfp8.padded(n);
            const tiled_codes = try a.alloc(u8, npad * k);
            defer a.free(tiled_codes);
            const packed_scales = try a.alloc(u8, npad * k / 32);
            defer a.free(packed_scales);
            cuda.mxfp8.packCodes(tiled_codes, codes, n, k);
            cuda.mxfp8.packScales(packed_scales, scales, n, k);
            self.weight = cuda.mxfp8.weight(try self.rig.input(tiled_codes), try self.rig.input(packed_scales), n, k);
        }
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        const bytes = self.lane.partBytes(shape.rows, self.weight);
        if (bytes > 0) self.part = try self.rig.scratch(bytes);
        self.out = try self.rig.output(shape.rows * n * 2);
        self.base = try self.rig.scratch(shape.rows * n * 2);
        return true;
    }

    fn launch(self: *Half, half: bool, out: u64) !void {
        const d = self.d.?;
        const keep = self.lane.half_below;
        defer self.lane.half_below = keep;
        self.lane.half_below = if (half) std.math.maxInt(usize) else 0;
        var w = self.weight;
        if (half and self.kind == .control) w.scale = 1.5;
        try self.lane.matmul(self.rig.s, self.x, d.shape.k, d.shape.rows, w, out, self.part);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Half) !void {
        try self.launch(true, self.out);
    }

    pub fn replaced(self: *Half) ![]const u8 {
        try self.launch(false, self.base);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    pub fn count(self: *Half) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Half, want: []f64, magnitude: []f64) !void {
        try self.d.?.reference(want, magnitude);
    }

    pub fn simulate(self: *Half, accumulator: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(accumulator, order, out);
    }
};
