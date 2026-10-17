//! MiniMax Music 3's m3_rmsnorm and decode GEMVs (bf16, int8 a scale a row, int8 a scale a group) on CUDA.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const Dense = contract.dense.Dense;

/// ceil(k / 512) squares a thread, then a warp's tree and the block's tree of 16 warps.
fn normDepth(s: c.Shape) usize {
    return (s.k + 511) / 512 + 10;
}

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn normSpec() c.Spec {
    return .{ .name = "cuda music3 m3_rmsnorm (bf16 rows, plain gamma)", .accum = .f32, .depth = normDepth, .ops = 8, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 512 } }, .served = &.{ .{ .rows = 2, .n = 4096, .k = 4096 }, .{ .rows = 4, .n = 4096, .k = 4096 }, .{ .rows = 600, .n = 4096, .k = 4096 } }, .sums = .{ .bf16 = 0, .why = norm_why } };
}

pub const Norm = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    r: ?rows.Rows = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, mod: cuda.Module) !Norm {
        return .{ .rig = .{ .gpu = gpu, .s = s }, .f = try mod.function("m3_rmsnorm") };
    }

    pub fn deinit(self: *Norm) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Norm) g.Kernel {
        return .{ .spec = normSpec(), .ctx = self, .vt = &norm_vt };
    }

    const norm_vt = g.hooks(Norm);

    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != shape.k) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const r = try rows.Rows.init(self.rig.gpu.gpa, .rms, shape);
        self.r = r;
        r.fill(fill, seed);
        self.x = try self.rig.input(std.mem.sliceAsBytes(r.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(r.w));
        self.out = try self.rig.output(r.outputs() * c.size(normSpec().out));
        return true;
    }

    pub fn run(self: *Norm) !void {
        const r = self.r.?;
        const k: c_int = @intCast(r.shape.k);
        var a: cuda.Args = .{};
        a.add(self.x);
        a.add(k);
        a.add(self.w);
        a.add(self.out);
        a.add(k);
        a.add(k);
        a.add(r.eps);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(r.shape.rows), .y = 1, .z = 1 }, .block = .{ .x = 512 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Norm) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        return self.r.?.reference(want, mag);
    }

    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(acc, order, out);
    }
};

pub const Kind = enum { bf16, int8, int8_group };

/// A lane's fp32 chain over its K / 32 inputs (a group's partial, then a chunk's fma) and the warp's 5-level tree.
fn gemvDepth(s: c.Shape) usize {
    return s.k / 32 + 16 + 8;
}

fn symbol(kind: Kind) [:0]const u8 {
    return switch (kind) {
        .bf16 => "m3_gemv_bf16_2",
        .int8 => "m3_gemv_i8_2",
        .int8_group => "m3_gemv_g8_2",
    };
}

pub fn gemvSpec(kind: Kind) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 2, .n = 4096, .k = 4096 }, .{ .rows = 2, .n = 4096, .k = 12288 }, .{ .rows = 2, .n = 1024, .k = 4096 } };
    const kt: c.Tile = switch (kind) {
        .bf16 => .{ .size = 256, .step = 8, .min = 8 },
        .int8 => .{ .size = 512, .step = 16, .min = 16 },
        .int8_group => .{ .size = 512, .step = 128, .min = 128 },
    };
    const name = switch (kind) {
        .bf16 => "cuda music3 m3_gemv_bf16_2 (bf16 weights, two rows)",
        .int8 => "cuda music3 m3_gemv_i8_2 (int8 weights, a scale a row)",
        .int8_group => "cuda music3 m3_gemv_g8_2 (int8 weights, a scale a group of 128)",
    };
    return .{ .name = name, .accum = .f32, .depth = gemvDepth, .ops = if (kind == .bf16) 0 else 1, .out = .bf16, .tiles = .{ .{}, .{ .size = 8, .step = 1, .min = 1 }, kt }, .served = served };
}

pub const Gemv = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: cuda.Function,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    sc: u64 = 0,
    out: u64 = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, mod: cuda.Module, kind: Kind) !Gemv {
        return .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .f = try mod.function(symbol(kind)) };
    }

    pub fn deinit(self: *Gemv) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Gemv) g.Kernel {
        return .{ .spec = gemvSpec(self.kind), .ctx = self, .vt = &gemv_vt };
    }

    const gemv_vt = g.hooks(Gemv);

    pub fn prepare(self: *Gemv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const step: usize = switch (self.kind) {
            .bf16 => 8,
            .int8 => 16,
            .int8_group => 128,
        };
        if (shape.rows < 2 or shape.n == 0 or shape.k == 0 or shape.k % step != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 2);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6d33);
        const rng = prng.random();
        const groups = if (self.kind == .int8_group) k / 128 else 1;
        const scales = try a.alloc(f32, n * groups);
        defer a.free(scales);
        // powers of two, so the kernel's scale after the sum equals the scaled terms' sum exactly
        for (scales) |*v| v.* = if (fill == .stress or self.kind == .bf16) 1 else std.math.ldexp(@as(f32, 1), -6 - @as(i32, @intCast(rng.uintLessThan(u32, 3))));
        const q = try a.alloc(i8, n * k);
        defer a.free(q);
        for (0..n) |j| for (0..k) |i| {
            const v: i8 = if (fill == .stress)
                (if (i < k / 2) @as(i8, @intCast(1 + (i * 7 + j) % 6)) else -@as(i8, @intCast(1 + ((i - k / 2) * 7 + j) % 6)))
            else
                @as(i8, @intCast(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127));
            q[j * k + i] = v;
            const s: f64 = scales[j * groups + (if (self.kind == .int8_group) i / 128 else 0)];
            d.w[j * k + i] = @as(f64, @floatFromInt(v)) * s * @as(f64, if (self.kind == .bf16 and fill != .stress) 1.0 / 128.0 else 1.0);
        };
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        if (self.kind == .bf16) {
            const wb = try a.alloc(u16, n * k);
            defer a.free(wb);
            for (wb, d.w) |*o, v| o.* = c.toBf16(@floatCast(v));
            self.w = try self.rig.input(std.mem.sliceAsBytes(wb));
            self.sc = 0;
        } else {
            self.w = try self.rig.input(std.mem.sliceAsBytes(q));
            self.sc = try self.rig.input(std.mem.sliceAsBytes(scales));
        }
        self.out = try self.rig.output(shape.rows * n * 2);
        return true;
    }

    /// The two-row kernel over every pair of rows (an odd last row goes with the one before it, written twice alike).
    pub fn run(self: *Gemv) !void {
        const sh = self.d.?.shape;
        var r0: usize = 0;
        while (r0 < sh.rows) : (r0 += 2) {
            const at = @min(r0, sh.rows - 2);
            var a: cuda.Args = .{};
            a.add(self.w);
            a.add(self.sc);
            a.add(self.x + at * sh.k * 2);
            a.add(@as(c_int, @intCast(sh.k)));
            a.add(@as(c_int, @intCast(sh.k)));
            a.add(@as(c_int, @intCast(sh.n)));
            a.add(self.out + at * sh.n * 2);
            a.add(@as(c_int, @intCast(sh.n)));
            a.add(@as(u64, 0));
            a.add(@as(u64, 0));
            a.add(@as(c_int, 1));
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((sh.n + 7) / 8), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Gemv) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Gemv, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Gemv, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};
