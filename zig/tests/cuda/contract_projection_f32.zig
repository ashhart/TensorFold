//! Packed projections (lane FP32 or bf16 out, prompt16 FP8), with independent FP64 products and mirrored stress.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const Rig = @import("contract_rig.zig").Rig;

pub const modes = [_]cuda.qmmf.Mode{ .bf16, .fp8, .fp8g, .mxfp8 };
/// The bf16-out lane modes a ModelOpt Qwen3.8-27B serves: NVFP4 MLPs and head, FP8 mixers, bf16 a and b.
pub const bf16_modes = [_]cuda.qmmf.Mode{ .fp4, .fp8, .bf16 };
/// The 27B's decode and verify rows (1 to 64) at its reductions (5120, 6144, 17408), columns cut to keep FP64 short.
const served_27b = [_]c.Shape{
    .{ .rows = 1, .n = 512, .k = 5120 },
    .{ .rows = 3, .n = 48, .k = 5120 },
    .{ .rows = 16, .n = 1024, .k = 5120 },
    .{ .rows = 33, .n = 640, .k = 6144 },
    .{ .rows = 64, .n = 256, .k = 17408 },
    .{ .rows = 16, .n = 512, .k = 17408 },
};
/// The 27B's prompt chunks past 16 rows on prompt16's FP8 tile.
const served_prompt = [_]c.Shape{
    .{ .rows = 17, .n = 512, .k = 5120 },
    .{ .rows = 128, .n = 1024, .k = 5120 },
    .{ .rows = 300, .n = 512, .k = 6144 },
};
const served = [_]c.Shape{
    .{ .rows = 1, .n = 70, .k = 1024 },
    .{ .rows = 17, .n = 70, .k = 1024 },
    .{ .rows = 33, .n = 70, .k = 1024 },
    .{ .rows = 256, .n = 70, .k = 1024 },
    .{ .rows = 16, .n = 1280, .k = 2560 },
    .{ .rows = 16, .n = 2560, .k = 640 },
    .{ .rows = 7, .n = 2560, .k = 10240 },
    .{ .rows = 7, .n = 13952, .k = 2560 },
};

fn depth(shape: c.Shape) usize {
    return shape.k / 16 + 24;
}

/// One fp32 chain over K in 16-deep MMA steps (prompt16).
fn promptDepth(shape: c.Shape) usize {
    return shape.k / 16 + 16;
}

pub const Product = struct {
    rig: Rig,
    lane: *const cuda.qmmf.Lane,
    mode: cuda.qmmf.Mode,
    clusters: bool,
    out_kind: cuda.qmmf.Out = .f32,
    prompt: ?*const cuda.nvfp4.Prompt = null, // set: prompt16's FP8 tile instead of the lane
    d: ?Dense = null,
    weight: cuda.qmmf.Weight = undefined,
    x: u64 = 0,
    out: u64 = 0,
    part: u64 = 0,

    pub fn deinit(self: *Product) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Product) g.Kernel {
        if (self.prompt != null) return .{ .spec = .{
            .name = "cuda prompt16 FP8 tile 4",
            .accum = .f32,
            .depth = promptDepth,
            .ops = 1,
            .out = .bf16,
            .tiles = .{ .{ .size = 128 }, .{ .size = 128, .step = 64, .min = 64 }, .{ .size = 128, .step = 64, .min = 64 } },
            .served = &served_prompt,
        }, .ctx = self, .vt = &vt };
        const wide = self.out_kind == .f32;
        const name = switch (self.mode) {
            .bf16 => if (!wide) (if (self.clusters) "cuda BF16 bf16 clustered" else "cuda BF16 bf16 reduce") else if (self.clusters) "cuda BF16 fp32 clustered" else "cuda BF16 fp32 reduce",
            .fp8 => if (!wide) (if (self.clusters) "cuda FP8 bf16 clustered" else "cuda FP8 bf16 reduce") else if (self.clusters) "cuda FP8 fp32 clustered" else "cuda FP8 fp32 reduce",
            .fp4 => if (self.clusters) "cuda FP4 bf16 clustered" else "cuda FP4 bf16 reduce",
            .fp8g => if (self.clusters) "cuda block FP8 fp32 clustered" else "cuda block FP8 fp32 reduce",
            .mxfp8 => if (self.clusters) "cuda MXFP8 fp32 clustered" else "cuda MXFP8 fp32 reduce",
        };
        return .{
            .spec = .{
                .name = name,
                .accum = .f32,
                .depth = depth,
                .ops = if (self.mode == .fp4) 2 else 0, // the global scale is not a power of two here
                .out = if (wide) .f32 else .bf16,
                .tiles = .{ .{ .size = 16 }, .{ .size = 64 }, .{ .size = 512, .step = 128, .min = 128 } },
                .served = if (wide) &served else &served_27b,
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    const vt = g.hooks(Product);

    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        self.part = 0;
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 128 != 0) return false;
        const d = try Dense.init(self.rig.gpu.gpa, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 16);
        try self.prepareWeight(shape, fill, seed);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        const bytes = if (self.prompt != null) 0 else self.lane.partBytes(shape.rows, self.weight);
        if (bytes > 0) self.part = try self.rig.scratch(bytes);
        self.out = try self.rig.output(shape.rows * shape.n * @as(usize, if (self.out_kind == .f32 and self.prompt == null) 4 else 2));
        return true;
    }

    fn prepareWeight(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !void {
        var arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const d = self.d.?;
        const n = shape.n;
        const k = shape.k;
        const npad = cuda.fp8.padded(n);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xf32);
        const random = prng.random();
        const codes = try a.alloc(u8, n * k);
        const words = try a.alloc(u16, n * k);
        const ue8 = try a.alloc(u8, n * k / 32);
        for (ue8) |*v| v.* = if (fill == .stress) 127 else 123 + random.uintLessThan(u8, 8);
        const inv = try a.alloc(f32, ((n + 127) / 128) * (k / 128));
        for (inv) |*v| v.* = if (fill == .stress) 1 else std.math.ldexp(@as(f32, 1), @as(i32, random.uintLessThan(u8, 8)) - 4);
        for (0..n) |j| for (0..k) |i| {
            const at = j * k + i;
            codes[at] = if (fill == .stress) 0x38 + random.uintLessThan(u8, 24) else random.uintLessThan(u8, 0x7f);
            if (fill != .stress and random.boolean()) codes[at] |= 0x80;
            words[at] = c.toBf16(if (fill == .stress) @floatFromInt(1 + random.uintLessThan(u8, 7)) else random.float(f32) * 4 - 2);
        };
        if (fill == .stress) for (0..n) |j| for (k / 2..k) |i| {
            codes[j * k + i] = codes[j * k + i - k / 2] | 0x80;
            words[j * k + i] = words[j * k + i - k / 2] | 0x8000;
        };
        if (self.mode == .fp4) return self.prepareFp4(shape, fill, random);
        const scale: f32 = if (fill == .stress) 1 else 0.125;
        for (0..n) |j| for (0..k) |i| {
            const value = core.quant.fp8.e4m3(codes[j * k + i]);
            d.w[j * k + i] = switch (self.mode) {
                .bf16 => c.fromBf16(words[j * k + i]),
                .fp8 => value * scale,
                .fp8g => value * inv[(j / 128) * (k / 128) + i / 128],
                .mxfp8 => value * std.math.ldexp(@as(f64, 1), @as(i32, ue8[j * (k / 32) + i / 32]) - 127),
                else => unreachable,
            };
        };
        if (self.mode == .bf16) {
            const tiles = try a.alloc(u16, npad * k);
            cuda.qmmf.packBf16(tiles, words, n, k);
            self.weight = cuda.modelopt.bf16Weight(try self.rig.input(std.mem.sliceAsBytes(tiles)), n, k);
            return;
        }
        const tiles = try a.alloc(u8, npad * k);
        cuda.fp8.packCodes(tiles, codes, n, k);
        const dw = try self.rig.input(tiles);
        self.weight = switch (self.mode) {
            .fp8 => cuda.modelopt.fp8Weight(dw, scale, n, k),
            .mxfp8 => blk: {
                const scales = try a.alloc(u8, npad * k / 32);
                cuda.mxfp8.packScales(scales, ue8, n, k);
                break :blk cuda.mxfp8.weight(dw, try self.rig.input(scales), n, k);
            },
            .fp8g => blk: {
                const scales = try a.alloc(f32, npad * k / 64);
                cuda.fp8.packScales(scales, inv, n, k);
                break :blk cuda.fp8.weight(dw, try self.rig.input(std.mem.sliceAsBytes(scales)), n, k);
            },
            else => unreachable,
        };
    }

    /// NVFP4 codes and e4m3 block scales (stress: scale 1, codes 1 to 7 mirrored negative) under a global scale.
    fn prepareFp4(self: *Product, shape: c.Shape, fill: g.Fill, random: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const d = self.d.?;
        const n = shape.n;
        const k = shape.k;
        const codes = try a.alloc(u8, n * k / 2);
        defer a.free(codes);
        const scales = try a.alloc(u8, n * k / 16);
        defer a.free(scales);
        const global: f32 = if (fill == .stress) 1 else 0.0123;
        for (scales) |*v| v.* = if (fill == .stress) 0x38 else 0x20 + random.uintLessThan(u8, 0x28);
        for (0..n) |j| for (0..k) |i| {
            const mirror = if (fill == .stress and i >= k / 2) j * k + i - k / 2 else null;
            const nib: u8 = if (mirror) |m| ((if (m % 2 == 0) codes[m / 2] & 0xf else codes[m / 2] >> 4) | 0x8) else if (fill == .stress) 1 + random.uintLessThan(u8, 7) else random.uintLessThan(u8, 16);
            if (i % 2 == 0) codes[(j * k + i) / 2] = nib else codes[(j * k + i) / 2] |= nib << 4;
            d.w[j * k + i] = core.quant.nvfp4.e2m1(nib) * core.quant.fp8.e4m3(scales[j * (k / 16) + i / 16]) * global;
        };
        const words = try a.alloc(u32, cuda.nvfp4.padded(n) / 64 * (k / cuda.nvfp4.group) * 512);
        defer a.free(words);
        cuda.nvfp4.packWords(words, codes, n, k);
        const tiles = try a.alloc(u8, cuda.nvfp4.padded(n) * (k / 16));
        defer a.free(tiles);
        cuda.nvfp4.packScales(tiles, scales, n, k);
        self.weight = cuda.nvfp4.weight(try self.rig.input(std.mem.sliceAsBytes(words)), try self.rig.input(tiles), global, n, k);
    }

    pub fn run(self: *Product) !void {
        const shape = self.d.?.shape;
        if (self.prompt) |p| {
            try p.matmul(self.rig.s, self.x, shape.k, shape.rows, self.weight, self.out);
        } else try self.lane.matmulOut(self.rig.s, self.x, shape.k, shape.rows, self.weight, self.out, self.out_kind, self.part);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Product) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Product, want: []f64, magnitude: []f64) !void {
        try self.d.?.reference(want, magnitude);
    }

    pub fn simulate(self: *Product, accumulator: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(accumulator, order, out);
    }
};
