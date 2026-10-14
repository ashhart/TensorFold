//! Quantized products under the contract on CUDA: qmm_group's cluster tile, lane_gemv (its replacement), prompt16 FP4.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const Dense = contract.dense.Dense;
const qlinear = cuda.qlinear;
const nvfp4 = cuda.nvfp4;

pub const Kind = enum { qmm_group, lane_gemv, prompt16 };

/// A group's 64 products, the producer's 64-input sum, two fmas a group, then the K slices.
fn groupDepth(s: c.Shape) usize {
    return 64 + 64 + 2 * (s.k / 64) + 8;
}

/// One fp32 chain over K in 16-deep MMA steps.
fn mmaChain(s: c.Shape) usize {
    return s.k / 16 + 16;
}

pub fn spec(kind: Kind) c.Spec {
    const decode = &[_]c.Shape{ .{ .rows = 16, .n = 10304, .k = 2688 }, .{ .rows = 16, .n = 2688, .k = 4096 }, .{ .rows = 3, .n = 4608, .k = 2688 } };
    const tiles = [3]c.Tile{ .{ .size = 16 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 512, .step = 64, .min = 64 } };
    return switch (kind) {
        .qmm_group => .{ .name = "cuda qmm_group (cluster tile)", .accum = .f32, .depth = groupDepth, .out = .bf16, .tiles = tiles, .served = decode },
        .lane_gemv => .{ .name = "cuda lane_gemv", .accum = .f32, .depth = groupDepth, .out = .bf16, .replaces = "qmm_group's cluster tile", .tiles = tiles, .served = decode },
        .prompt16 => .{ .name = "cuda prompt16 FP4 tile 4", .accum = .f32, .depth = mmaChain, .out = .bf16, .tiles = .{ .{ .size = 128 }, .{ .size = 128, .step = 64, .min = 64 }, .{ .size = 128, .step = 64, .min = 64 } }, .served = &.{ .{ .rows = 300, .n = 512, .k = 2560 }, .{ .rows = 64, .n = 2560, .k = 512 } } },
    };
}

pub const Mm = struct {
    rig: rig_.Rig,
    kind: Kind,
    affine4: ?*const qlinear.Affine4Kernels = null,
    prompt: ?*const nvfp4.Prompt = null,
    f: ?Affine = null,
    d: ?Dense = null,
    x: u64 = 0,
    xs: u64 = 0,
    q: qlinear.Affine4 = undefined,
    w4: cuda.qmmf.Weight = undefined,
    base: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Mm) void {
        if (self.f) |f| f.deinit();
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Mm) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Mm);

    fn drop(self: *Mm) void {
        if (self.f) |f| f.deinit();
        if (self.d) |d| d.deinit();
        self.f = null;
        self.d = null;
        self.rig.reset();
    }

    pub fn prepare(self: *Mm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.drop();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        return if (self.kind == .prompt16) self.preparePrompt(shape, fill, seed) else self.prepareAffine(shape, fill, seed);
    }

    fn prepareAffine(self: *Mm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows > 16 or shape.n % 64 != 0) return false;
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
        if (self.kind == .lane_gemv) self.base = try self.rig.scratch(out_bytes);
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
        self.out = try self.rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(self: *Mm) !void {
        try self.launch(self.kind, self.out);
        try self.rig.s.synchronize();
    }

    fn launch(self: *Mm, kind: Kind, out: u64) !void {
        const s = self.rig.s;
        switch (kind) {
            .qmm_group => {
                const f = self.f.?;
                try self.affine4.?.cluster(s, self.x, self.xs, self.q, out, f.shape.rows, qlinear.splitK(f.shape.n, f.shape.k));
            },
            .lane_gemv => {
                const f = self.f.?;
                try self.affine4.?.gemv(s, self.x, self.xs, self.q, out, f.shape.rows, qlinear.splitK(f.shape.n, f.shape.k));
            },
            .prompt16 => try self.prompt.?.matmul(s, self.x, self.d.?.shape.k, self.d.?.shape.rows, self.w4, out),
        }
    }

    /// qmm_group's cluster tile on the same inputs: the bits lane_gemv must equal.
    pub fn replaced(self: *Mm) ![]const u8 {
        if (self.kind != .lane_gemv) return error.NotAReplacement;
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
