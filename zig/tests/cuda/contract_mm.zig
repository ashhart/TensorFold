//! Quantized products under the contract on CUDA: qmm_group, lane_gemv's three forms, prompt16 and qmmf FP4 (fp32 too).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const Dense = contract.dense.Dense;
const qlinear = cuda.qlinear;
const nvfp4 = cuda.nvfp4;

pub const Kind = enum { qmm_group, lane_gemv, lane_gemv_split, lane_gemv_seg, prompt16, prompt16_f32, prompt16_lean, prompt16_lean_f32, lane_f32, lane_f32_reduce, verify32_bits, verify32_fp64, verify64_bits, verify64_fp64, verify128_bits, verify128_fp64 };

/// Rows a verify kind's launch holds, 0 for the others.
pub fn verifyTile(kind: Kind) usize {
    return switch (kind) {
        .verify32_bits, .verify32_fp64 => 32,
        .verify64_bits, .verify64_fp64 => 64,
        .verify128_bits, .verify128_fp64 => 128,
        else => 0,
    };
}

fn oneRowBits(kind: Kind) bool {
    return kind == .verify32_bits or kind == .verify64_bits or kind == .verify128_bits;
}

/// A group's 64 products, the producer's 64-input sum, two fmas a group, then the K slices.
fn groupDepth(s: c.Shape) usize {
    return 64 + 64 + 2 * (s.k / 64) + 8;
}

/// One fp32 chain over K in 16-deep MMA steps.
fn mmaChain(s: c.Shape) usize {
    return s.k / 16 + 16;
}

/// A 16-input MMA from zero, one fma a block into the slice's sum, then up to 8 slices added.
fn laneFp4Depth(s: c.Shape) usize {
    return s.k / 16 + 24;
}

/// GLM-5.3-Flash's dense down projection on a rank (#804: 4096 from 6144 or 3072), then a shape reaching each tile.
const fp4_lane = [_]c.Shape{
    .{ .rows = 16, .n = 512, .k = 2560 },  .{ .rows = 16, .n = 4096, .k = 6144 }, .{ .rows = 3, .n = 4096, .k = 3072 },
    .{ .rows = 64, .n = 512, .k = 2560 },  .{ .rows = 16, .n = 2560, .k = 512 },  .{ .rows = 64, .n = 2560, .k = 512 },
    .{ .rows = 256, .n = 512, .k = 2560 },
};
const fp4_tiles = [3]c.Tile{ .{ .size = 16 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 512, .step = 64, .min = 64 } };
const prompt_tiles = [3]c.Tile{ .{ .size = 128 }, .{ .size = 128, .step = 64, .min = 64 }, .{ .size = 128, .step = 64, .min = 64 } };
/// prompt_lean: 128-row tiles, 256 columns, three 64-input stages in its ring.
const lean_tiles = [3]c.Tile{ .{ .size = 128 }, .{ .size = 256, .step = 64, .min = 64 }, .{ .size = 192, .step = 64, .min = 64 } };
/// Partial row and column tiles, odd group counts, and more tiles (3 x 20) than a GB10 has multiprocessors.
const lean_served = &[_]c.Shape{ .{ .rows = 300, .n = 512, .k = 2560 }, .{ .rows = 64, .n = 2560, .k = 512 }, .{ .rows = 300, .n = 5120, .k = 576 }, .{ .rows = 17, .n = 640, .k = 1088 } };

/// The lean tile's kinds, each bit-equal to prompt16's 128 x 128 tile.
pub fn lean(kind: Kind) bool {
    return kind == .prompt16_lean or kind == .prompt16_lean_f32;
}

/// GB10 verify tiles' affine-4 projections (N, K): the 27B's, the 35B's, their drafters', then both vocabulary heads.
pub const projections = [_][2]usize{
    .{ 10240, 5120 }, .{ 6144, 5120 }, .{ 48, 5120 },   .{ 5120, 6144 }, .{ 12288, 5120 }, .{ 1024, 5120 },   .{ 17408, 5120 },
    .{ 5120, 17408 }, .{ 8192, 2048 }, .{ 4096, 2048 }, .{ 32, 2048 },   .{ 2048, 4096 },  .{ 512, 2048 },    .{ 2048, 2048 },
    .{ 5120, 25600 }, .{ 256, 5120 },  .{ 4096, 5120 }, .{ 5120, 4096 }, .{ 1280, 5120 },  .{ 248320, 5120 }, .{ 248320, 2048 },
};
pub const heads = 2;

/// Every projection once at rows in (lo, hi], full and partial tiles in turn; `bits` keeps the heads (no FP64 there).
fn across(comptime lo: usize, comptime hi: usize, comptime turn: usize, comptime bits: bool) [projections.len - if (bits) 0 else heads]c.Shape {
    var out: [projections.len - if (bits) 0 else heads]c.Shape = undefined;
    for (&out, 0..) |*s, i| s.* = .{ .rows = if ((i + turn) % 2 == 0) hi else lo + 1 + (hi - lo - 2) * i / projections.len, .n = projections[i][0], .k = projections[i][1] };
    return out;
}

pub fn spec(kind: Kind) c.Spec {
    const decode = &[_]c.Shape{ .{ .rows = 16, .n = 10304, .k = 2688 }, .{ .rows = 16, .n = 2688, .k = 4096 }, .{ .rows = 3, .n = 4608, .k = 2688 } };
    // Qwen3.5's projections of one input: Qwen3.8-27B's (two K slices, eight, one) and Qwen3.5-35B-A3B's
    const segmented = &[_]c.Shape{ .{ .rows = 16, .n = 10240, .k = 5120 }, .{ .rows = 4, .n = 1024, .k = 5120 }, .{ .rows = 1, .n = 12288, .k = 5120 }, .{ .rows = 16, .n = 8192, .k = 2048 }, .{ .rows = 3, .n = 512, .k = 2048 } };
    const tiles = [3]c.Tile{ .{ .size = 16 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 512, .step = 64, .min = 64 } };
    const wide_edges = [_]c.Shape{ .{ .rows = 16, .n = 512, .k = 5120 }, .{ .rows = 17, .n = 65, .k = 512 }, .{ .rows = 32, .n = 512, .k = 17408 }, .{ .rows = 32, .n = 17408, .k = 512 }, .{ .rows = 32, .n = 5120, .k = 5120 } };
    const wide = comptime &(wide_edges ++ across(16, 32, 0, false));
    const wide_bits = comptime &(wide_edges ++ across(16, 32, 0, true));
    const wide_tiles = [3]c.Tile{ .{ .size = 32 }, .{ .size = 64, .min = 1 }, .{ .size = 512, .step = 64, .min = 64 } };
    // frames past each row-tile edge, ragged panels and long K, then every real projection at full and partial tiles
    const wide64_edges = [_]c.Shape{ .{ .rows = 33, .n = 65, .k = 512 }, .{ .rows = 64, .n = 1024, .k = 5120 }, .{ .rows = 48, .n = 512, .k = 4096 }, .{ .rows = 64, .n = 256, .k = 5120 }, .{ .rows = 64, .n = 512, .k = 17408 } };
    const wide128_edges = [_]c.Shape{ .{ .rows = 65, .n = 65, .k = 512 }, .{ .rows = 128, .n = 1024, .k = 5120 }, .{ .rows = 120, .n = 4096, .k = 5120 }, .{ .rows = 100, .n = 512, .k = 4096 }, .{ .rows = 128, .n = 256, .k = 5120 }, .{ .rows = 112, .n = 512, .k = 25600 }, .{ .rows = 128, .n = 17408, .k = 512 } };
    const wide64 = comptime &(wide64_edges ++ across(32, 64, 1, false));
    const wide64_bits = comptime &(wide64_edges ++ across(32, 64, 1, true));
    const wide128 = comptime &(wide128_edges ++ across(64, 128, 0, false));
    const wide128_bits = comptime &(wide128_edges ++ across(64, 128, 0, true));
    const wide64_tiles = [3]c.Tile{ .{ .size = 64 }, .{ .size = 64, .min = 1 }, .{ .size = 512, .step = 64, .min = 64 } };
    const wide128_tiles = [3]c.Tile{ .{ .size = 128 }, .{ .size = 64, .min = 1 }, .{ .size = 512, .step = 64, .min = 64 } };
    return switch (kind) {
        .verify32_bits => .{ .name = "cuda affine4 verify32 one-row bits", .accum = .f32, .depth = groupDepth, .out = .bf16, .replaces = "the original one-row affine4 decode", .tiles = wide_tiles, .served = wide_bits },
        .verify32_fp64 => .{ .name = "cuda affine4 verify32 FP64", .accum = .f32, .depth = groupDepth, .out = .bf16, .tiles = wide_tiles, .served = wide },
        .verify64_bits => .{ .name = "cuda affine4 verify64 one-row bits", .accum = .f32, .depth = groupDepth, .out = .bf16, .replaces = "the original one-row affine4 decode", .tiles = wide64_tiles, .served = wide64_bits },
        .verify64_fp64 => .{ .name = "cuda affine4 verify64 FP64", .accum = .f32, .depth = groupDepth, .out = .bf16, .tiles = wide64_tiles, .served = wide64 },
        .verify128_bits => .{ .name = "cuda affine4 verify128 one-row bits", .accum = .f32, .depth = groupDepth, .out = .bf16, .replaces = "the original one-row affine4 decode", .tiles = wide128_tiles, .served = wide128_bits },
        .verify128_fp64 => .{ .name = "cuda affine4 verify128 FP64", .accum = .f32, .depth = groupDepth, .out = .bf16, .tiles = wide128_tiles, .served = wide128 },
        .qmm_group => .{ .name = "cuda qmm_group (cluster tile)", .accum = .f32, .depth = groupDepth, .out = .bf16, .tiles = tiles, .served = decode },
        .lane_gemv => .{ .name = "cuda lane_gemv", .accum = .f32, .depth = groupDepth, .out = .bf16, .replaces = "qmm_group's cluster tile", .tiles = tiles, .served = decode },
        .lane_gemv_split => .{ .name = "cuda lane_gemv split-K", .accum = .f32, .depth = groupDepth, .out = .bf16, .replaces = "qmm_group's cluster tile", .tiles = tiles, .served = decode },
        .lane_gemv_seg => .{ .name = "cuda lane_gemv segmented split-K", .accum = .f32, .depth = groupDepth, .out = .bf16, .replaces = "affine-4 decode()", .tiles = tiles, .served = segmented },
        .prompt16 => .{ .name = "cuda prompt16 FP4 tile 4", .accum = .f32, .depth = mmaChain, .ops = 2, .out = .bf16, .tiles = prompt_tiles, .served = &.{ .{ .rows = 300, .n = 512, .k = 2560 }, .{ .rows = 64, .n = 2560, .k = 512 } } },
        .prompt16_f32 => .{ .name = "cuda prompt16 FP4 tile 4, fp32 out", .accum = .f32, .depth = mmaChain, .ops = 2, .out = .f32, .tiles = prompt_tiles, .served = &.{ .{ .rows = 300, .n = 512, .k = 2560 }, .{ .rows = 64, .n = 2560, .k = 512 }, .{ .rows = 64, .n = 4096, .k = 3072 } } },
        .prompt16_lean => .{ .name = "cuda prompt16 FP4 lean tile", .accum = .f32, .depth = mmaChain, .ops = 2, .out = .bf16, .replaces = "prompt16 FP4 tile 4", .tiles = lean_tiles, .served = lean_served },
        .prompt16_lean_f32 => .{ .name = "cuda prompt16 FP4 lean tile, fp32 out", .accum = .f32, .depth = mmaChain, .ops = 2, .out = .f32, .replaces = "prompt16 FP4 tile 4, fp32 out", .tiles = lean_tiles, .served = lean_served },
        .lane_f32 => .{ .name = "cuda qmmf FP4 fp32 out (slices in a cluster)", .accum = .f32, .depth = laneFp4Depth, .ops = 2, .out = .f32, .tiles = fp4_tiles, .served = &fp4_lane },
        .lane_f32_reduce => .{ .name = "cuda qmmf FP4 fp32 out (slices by reduce_kernel<true>)", .accum = .f32, .depth = laneFp4Depth, .ops = 2, .out = .f32, .tiles = fp4_tiles, .served = &fp4_lane },
    };
}

pub const Mm = struct {
    rig: rig_.Rig,
    kind: Kind,
    sm: u32,
    truncate_verify: bool = false, // the device shape controls deliberately omit the second row tile
    affine4: ?*const qlinear.Affine4Kernels = null,
    split: ?*const qlinear.Split = null,
    prompt: ?*const nvfp4.Prompt = null,
    lane: ?*const cuda.qmmf.Lane = null,
    f: ?Affine = null,
    d: ?Dense = null,
    x: u64 = 0,
    xs: u64 = 0,
    q: qlinear.Affine4 = undefined,
    w4: cuda.qmmf.Weight = undefined,
    base: u64 = 0,
    part: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Mm) void {
        if (self.f) |f| f.deinit();
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Mm) ?g.Kernel {
        if (self.sm < 90 and (self.kind == .qmm_group or verifyTile(self.kind) > 0)) return null;
        var declared = spec(self.kind);
        if (self.truncate_verify) declared.name = switch (self.kind) {
            .verify32_bits => "control: verify32 truncated one-row bits",
            .verify64_bits => "control: verify64 truncated one-row bits",
            .verify128_bits => "control: verify128 truncated one-row bits",
            .verify64_fp64 => "control: verify64 truncated FP64",
            .verify128_fp64 => "control: verify128 truncated FP64",
            else => "control: verify32 truncated FP64",
        };
        if (self.sm < 90) {
            if (self.kind != .lane_gemv_seg and !lean(self.kind)) declared.replaces = null; // these run on every GPU
            if (self.kind == .lane_f32) declared.name = "cuda qmmf FP4 fp32 out (slices by reduce_kernel<true>, architecture fallback)";
        }
        return .{ .spec = declared, .ctx = self, .vt = &vt, .control = if (self.truncate_verify) .shapes else null };
    }

    const vt = g.hooks(Mm);

    fn drop(self: *Mm) void {
        if (self.f) |f| f.deinit();
        if (self.d) |d| d.deinit();
        self.f = null;
        self.d = null;
        self.part = 0;
        self.rig.reset();
    }

    pub fn prepare(self: *Mm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.drop();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        return switch (self.kind) {
            .qmm_group, .lane_gemv, .lane_gemv_split, .lane_gemv_seg, .verify32_bits, .verify32_fp64, .verify64_bits, .verify64_fp64, .verify128_bits, .verify128_fp64 => self.prepareAffine(shape, fill, seed),
            .prompt16, .prompt16_f32, .prompt16_lean, .prompt16_lean_f32, .lane_f32, .lane_f32_reduce => self.preparePrompt(shape, fill, seed),
        };
    }

    fn prepareAffine(self: *Mm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const wide = verifyTile(self.kind) > 0;
        if (shape.rows > (if (wide) verifyTile(self.kind) else 16) or (!wide and shape.n % 64 != 0)) return false;
        if (self.kind == .lane_gemv_split and qlinear.splitK(shape.n, shape.k) == 1) return false; // served only with K slices
        const a = self.rig.gpu.gpa;
        const f = try Affine.init(a, shape, 4, 64);
        self.f = f;
        try f.fill(fill, seed);
        const words = try a.alloc(u32, shape.n * shape.k / 8);
        defer a.free(words);
        f.words(words);
        const xs = try a.alloc(f32, shape.rows * (shape.k / 64));
        defer a.free(xs);
        for (xs, 0..) |*v, i| {
            var s: f32 = 0;
            for (f.x[i * 64 ..][0..64]) |h| s += c.fromBf16(h);
            v.* = s;
        }
        self.x = try self.rig.input(std.mem.sliceAsBytes(f.x));
        self.xs = try self.rig.input(std.mem.sliceAsBytes(xs));
        const lay = qlinear.Affine4.layout(shape.n, shape.k);
        const zeros = try a.alloc(u8, lay.words);
        defer a.free(zeros);
        @memset(zeros, 0);
        self.q = .{ .w = try self.rig.input(zeros), .s = try self.rig.input(zeros[0..lay.scales]), .b = try self.rig.input(zeros[0..lay.scales]), .n = shape.n, .k = shape.k, .npad = lay.npad };
        const src_w = try self.rig.scratch(words.len * 4);
        const src_s = try self.rig.scratch(f.scales.len * 2);
        const src_b = try self.rig.scratch(f.biases.len * 2);
        for ([_]u64{ src_w, src_s, src_b }, [_][]const u8{ std.mem.sliceAsBytes(words), std.mem.sliceAsBytes(f.scales), std.mem.sliceAsBytes(f.biases) }) |p, h| {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = p, .len = h.len };
            try b.upload(0, h);
        }
        try self.affine4.?.pack(self.rig.s, src_w, src_s, src_b, self.q);
        try self.rig.s.synchronize();
        const out_bytes = shape.rows * shape.n * 2;
        if (self.kind == .lane_gemv or self.kind == .lane_gemv_split or self.kind == .lane_gemv_seg or oneRowBits(self.kind)) self.base = try self.rig.scratch(out_bytes);
        self.out = try self.rig.output(out_bytes);
        return true;
    }

    fn preparePrompt(self: *Mm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 2);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xf4);
        const rng = prng.random();
        const codes = try a.alloc(u8, n * k / 2);
        defer a.free(codes);
        const scales = try a.alloc(u8, n * k / 16);
        defer a.free(scales);
        const global: f32 = if (fill == .stress) 1 else 0.0123;
        for (scales) |*s| s.* = if (fill == .stress) 0x38 else 0x20 + rng.uintLessThan(u8, 0x28);
        for (0..n) |j| for (0..k) |i| {
            const nib: u8 = if (fill != .stress) rng.uintLessThan(u8, 16) else if (i < k / 2) 1 + rng.uintLessThan(u8, 7) else 0;
            if (i % 2 == 0) codes[(j * k + i) / 2] = nib else codes[(j * k + i) / 2] |= nib << 4;
        };
        if (fill == .stress) for (0..n) |j| for (k / 2..k) |i| {
            const mirror = j * k + i - k / 2;
            const m: u8 = (if (mirror % 2 == 0) codes[mirror / 2] & 0xf else codes[mirror / 2] >> 4) | 0x8;
            if (i % 2 == 0) codes[(j * k + i) / 2] = m else codes[(j * k + i) / 2] |= m << 4;
        };
        for (0..n) |j| for (0..k) |i| {
            const b = codes[(j * k + i) / 2];
            const nib: u8 = if (i % 2 == 0) b & 0xf else b >> 4;
            d.w[j * k + i] = core.quant.nvfp4.e2m1(nib) * core.quant.fp8.e4m3(scales[j * (k / 16) + i / 16]) * global;
        };
        const words = try a.alloc(u32, nvfp4.padded(n) / 64 * (k / nvfp4.group) * 512);
        defer a.free(words);
        nvfp4.packWords(words, codes, n, k);
        const bs = try a.alloc(u8, nvfp4.padded(n) * (k / 16));
        defer a.free(bs);
        nvfp4.packScales(bs, scales, n, k);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w4 = nvfp4.weight(try self.rig.input(std.mem.sliceAsBytes(words)), try self.rig.input(bs), global, n, k);
        if (self.lane) |l| if (l.partBytes(shape.rows, self.w4) > 0) {
            self.part = try self.rig.scratch(l.partBytes(shape.rows, self.w4));
        };
        if (lean(self.kind)) self.base = try self.rig.scratch(shape.rows * n * c.size(spec(self.kind).out));
        self.out = try self.rig.output(shape.rows * n * c.size(spec(self.kind).out));
        return true;
    }

    pub fn run(self: *Mm) !void {
        try self.launch(self.kind, self.out);
        try self.rig.s.synchronize();
    }

    fn launch(self: *Mm, kind: Kind, out: u64) !void {
        const s = self.rig.s;
        switch (kind) {
            .verify32_bits, .verify32_fp64 => {
                const f = self.f.?;
                try self.affine4.?.cluster32(s, self.x, self.xs, self.q, out, if (self.truncate_verify) @min(f.shape.rows, 16) else f.shape.rows);
            },
            .verify64_bits, .verify64_fp64, .verify128_bits, .verify128_fp64 => {
                const f = self.f.?;
                try self.affine4.?.clusterRows(verifyTile(kind), s, self.x, self.xs, self.q, out, if (self.truncate_verify) @min(f.shape.rows, 16) else f.shape.rows);
            },
            .qmm_group => {
                const f = self.f.?;
                try self.affine4.?.cluster(s, self.x, self.xs, self.q, out, f.shape.rows, qlinear.splitK(f.shape.n, f.shape.k));
            },
            .lane_gemv => {
                const f = self.f.?;
                try self.affine4.?.gemv(s, self.x, self.xs, self.q, out, f.shape.rows, qlinear.splitK(f.shape.n, f.shape.k));
            },
            .lane_gemv_split => {
                const f = self.f.?;
                try self.affine4.?.gemvSplit(s, self.x, self.xs, self.q, out, f.shape.rows, qlinear.splitK(f.shape.n, f.shape.k), self.split.?.*);
            },
            .lane_gemv_seg => {
                const f = self.f.?;
                // a second segment checked, so its partials and tickets sit past the first's
                const segs = [_]qlinear.Affine4Kernels.Out{ .{ .q = self.q, .out = self.base }, .{ .q = self.q, .out = out } };
                try self.affine4.?.gemvSeg(s, self.x, self.xs, &segs, f.shape.rows, self.split.?.*);
            },
            .prompt16 => try self.prompt.?.matmulTile(s, self.x, self.d.?.shape.k, self.d.?.shape.rows, self.w4, out, .bf16, .classic),
            .prompt16_f32 => try self.prompt.?.matmulTile(s, self.x, self.d.?.shape.k, self.d.?.shape.rows, self.w4, out, .f32, .classic),
            .prompt16_lean => try self.prompt.?.matmulTile(s, self.x, self.d.?.shape.k, self.d.?.shape.rows, self.w4, out, .bf16, .lean),
            .prompt16_lean_f32 => try self.prompt.?.matmulTile(s, self.x, self.d.?.shape.k, self.d.?.shape.rows, self.w4, out, .f32, .lean),
            .lane_f32, .lane_f32_reduce => {
                const d = self.d.?;
                try self.lane.?.matmulOut(s, self.x, d.shape.k, d.shape.rows, self.w4, out, .f32, self.part);
            },
        }
    }

    /// The replaced kernel on the same inputs: qmm_group's cluster tile (lane_gemv), prompt16's tile (prompt_lean).
    pub fn replaced(self: *Mm) ![]const u8 {
        if (lean(self.kind)) {
            try self.launch(if (self.kind == .prompt16_lean) .prompt16 else .prompt16_f32, self.base);
            try self.rig.s.synchronize();
            return self.rig.bytes(self.base, self.count() * c.size(spec(self.kind).out));
        }
        if (oneRowBits(self.kind)) {
            const f = self.f.?;
            for (0..f.shape.rows) |row| try self.affine4.?.decode(self.rig.s, self.x + row * f.shape.k * 2, self.xs + row * (f.shape.k / 64) * 4, self.q, self.base + row * f.shape.n * 2, 1);
            return self.rig.bytes(self.base, self.count() * 2);
        }
        if (self.kind == .lane_gemv_seg) {
            const f = self.f.?;
            try self.affine4.?.decode(self.rig.s, self.x, self.xs, self.q, self.base, f.shape.rows);
            return self.rig.bytes(self.base, self.count() * 2);
        }
        if (self.kind != .lane_gemv and self.kind != .lane_gemv_split) return error.NotAReplacement;
        try self.launch(.qmm_group, self.base);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    pub fn count(self: *Mm) usize {
        return if (self.f) |f| f.outputs() else self.d.?.outputs();
    }

    pub fn reference(self: *Mm, want: []f64, mag: []f64) !void {
        return if (self.f) |f| f.reference(want, mag) else self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Mm, acc: c.Accum, order: g.Order, out: []f64) !void {
        return if (self.f) |f| f.simulate(acc, order, out) else self.d.?.simulate(acc, order, out);
    }
};
