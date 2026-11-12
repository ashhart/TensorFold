const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;

pub fn text(a: std.mem.Allocator, device: mtl.Device, exact: bool, compact: bool) ![]u8 {
    if (compact) {
        const raw = try std.mem.concat(a, u8, &.{ if (exact) "#define TF_NAX_EXACT_B\n" else "", sources.tensor_compact_gemm });
        defer a.free(raw);
        return tf.frags.source(device, a, raw);
    }
    return tf.frags.source(device, a, if (exact) sources.tensor_exact_gemm else "#define TF_NAX_STRICT_FP32\n" ++ sources.core_qwen_vision);
}

fn fullDepth(s: c.Shape) usize {
    return 6 * s.k;
}

fn exactDepth(s: c.Shape) usize {
    return 3 * s.k;
}

const aligned = [_]c.Shape{
    .{ .rows = 1, .n = 128, .k = 256 },
    .{ .rows = 1, .n = 256, .k = 1024 },
    .{ .rows = 1, .n = 512, .k = 256 },
    .{ .rows = 1, .n = 1024, .k = 588 },
    .{ .rows = 1, .n = 1024, .k = 4736 },
    .{ .rows = 1, .n = 2048, .k = 256 },
    .{ .rows = 1, .n = 3072, .k = 1024 },
    .{ .rows = 1, .n = 4096, .k = 1024 },
    .{ .rows = 1, .n = 4736, .k = 1024 },
};
const ragged = [_]c.Shape{
    .{ .rows = 1, .n = 129, .k = 257 },
    .{ .rows = 1, .n = 1, .k = 256 },
    .{ .rows = 1, .n = 4, .k = 256 },
    .{ .rows = 1, .n = 16, .k = 256 },
    .{ .rows = 1, .n = 32, .k = 256 },
    .{ .rows = 1, .n = 64, .k = 256 },
};

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    baseline: mtl.Pipeline,
    name: []const u8,
    exact: bool,
    compact: bool,
    control: bool,
    am: bool,
    an: bool,
    m: usize,
    shape: c.Shape = undefined,
    stress: bool = false,
    x: []f32 = &.{},
    w: []f32 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    params: [16]i32 = @splat(0),

    fn kernel(self: *Probe) g.Kernel {
        return .{
            .spec = .{
                .name = self.name,
                .accum = .f32,
                .depth = if (self.exact) exactDepth else fullDepth,
                .out = .f32,
                .replaces = if (self.exact or self.compact) "full reconstructed FP32 GEMM" else null,
                .served = if (self.an) &aligned else &ragged,
                .tiles = .{ .{}, .{ .size = 128, .step = if (self.an) 128 else 1, .min = if (self.an) 128 else 1 }, .{ .size = 256 } },
            },
            .ctx = self,
            .vt = &vt,
            .control = if (self.control) .shapes else null,
        };
    }

    const vt = g.hooks(Probe);

    fn free(self: *Probe) void {
        self.a.free(self.x);
        self.a.free(self.w);
        self.x = &.{};
        self.w = &.{};
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or (shape.n % 128 == 0) != self.an) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        self.stress = fill == .stress;
        self.x = try self.a.alloc(f32, shape.rows * self.m * shape.k);
        self.w = try self.a.alloc(f32, shape.rows * shape.n * shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (self.stress) {
            for (self.x, 0..) |*v, i| {
                const k = i % shape.k;
                v.* = if (k < 8) @floatFromInt(1024 + 64 * k) else if (k < 16) @floatFromInt(1025 + 64 * (k - 8)) else 0;
            }
            for (self.w, 0..) |*v, i| {
                const k = i % shape.k;
                const col = i / shape.k % shape.n;
                const value: f32 = @floatFromInt(128 + (col + 3 * (k % 8)) % 128);
                v.* = if (k < 8) value else if (k < 16) -value else 0;
            }
        } else {
            for (self.x) |*v| v.* = rng.float(f32) * 4 - 2;
            for (self.w) |*v| {
                const value = rng.float(f32) * 4 - 2;
                v.* = if (self.exact and !self.control) @as(f16, @floatCast(value)) else value;
            }
        }
        if (fill == .nonfinite) {
            const k = shape.k;
            self.x[0] = std.math.nan(f32);
            self.x[k] = std.math.inf(f32);
            self.x[2 * k] = -std.math.inf(f32);
            self.x[3 * k] = std.math.inf(f32);
            self.x[3 * k + @min(1, k - 1)] = -std.math.inf(f32);
            for (0..shape.n) |col| {
                self.w[col * k] = 1;
                self.w[col * k + @min(1, k - 1)] = 1;
            }
        }
        const tn = (shape.n + 127) / 128;
        const tile: usize = if (self.compact) 32 else 64;
        const tm = (self.m + tile - 1) / tile;
        const fields = [_]usize{ self.m, shape.n, shape.k, shape.k, shape.k, shape.n, tn, tm, 2, shape.k / 256, self.m * shape.k, shape.n * shape.k, self.m * shape.n };
        self.params = @splat(0);
        for (fields, 0..) |value, i| self.params[i] = @intCast(value);
        self.ins[0] = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.ins[1] = try self.rig.input(std.mem.sliceAsBytes(self.w));
        self.out = try self.rig.output(self.count() * 4);
        self.base = try self.rig.scratch(self.count() * 4);
        return true;
    }

    fn launch(self: *Probe, baseline: bool) !void {
        const pass = self.rig.begin();
        pass.e.setPipeline(if (baseline) self.baseline else self.pipe);
        for (self.ins, 0..) |input, i| pass.e.setBuffer(input.buf, input.off, i);
        var params = self.params;
        const dst = if (baseline) self.base else self.out;
        pass.e.setBuffer(dst.buf, dst.off, 3);
        const tn = (self.shape.n + 127) / 128;
        const tile: usize = if (self.compact and !baseline) 32 else 64;
        const tm = (self.m + tile - 1) / tile;
        params[7] = @intCast(tm);
        pass.e.setValue(params, 2);
        pass.e.dispatchThreads(mtl.Size.of(tn * 4 * 32, (tm + 3) / 4 * 4, self.shape.rows * 2), mtl.Size.of(32, 4, 2));
        try self.rig.finish(pass);
    }

    pub fn run(self: *Probe) !void {
        try self.launch(false);
    }

    pub fn replaced(self: *Probe) ![]const u8 {
        try self.launch(true);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    pub fn count(self: *Probe) usize {
        return self.shape.rows * self.m * self.shape.n;
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        const rows = if (self.stress) 1 else self.shape.rows * self.m;
        const K = self.shape.k;
        for (0..rows) |row| for (0..self.shape.n) |col| {
            const b = row / self.m;
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (0..if (self.stress) @min(16, K) else K) |k| {
                const term = @as(f64, self.x[row * K + k]) * self.w[(b * self.shape.n + col) * K + k];
                sum += term;
                magnitude += @abs(term);
            }
            want[row * self.shape.n + col] = sum;
            mag[row * self.shape.n + col] = magnitude;
        };
        if (self.stress) for (1..self.shape.rows * self.m) |row| {
            @memcpy(want[row * self.shape.n ..][0..self.shape.n], want[0..self.shape.n]);
            @memcpy(mag[row * self.shape.n ..][0..self.shape.n], mag[0..self.shape.n]);
        };
    }

    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!self.stress) return error.ExpectedGemmStress;
        const terms = try self.a.alloc(f64, self.shape.k);
        defer self.a.free(terms);
        @memset(terms, 0);
        for (0..self.shape.n) |col| {
            for (terms[0..@min(16, self.shape.k)], 0..) |*term, k| term.* = @as(f64, self.x[k]) * self.w[col * self.shape.k + k];
            out[col] = contract.rows.sumIn(acc, order, terms);
        }
        for (1..self.shape.rows * self.m) |row| @memcpy(out[row * self.shape.n ..][0..self.shape.n], out[0..self.shape.n]);
    }

    fn deinit(self: *Probe) void {
        self.free();
        self.rig.deinit();
        self.pipe.deinit();
        self.baseline.deinit();
        self.a.free(self.name);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(self: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        if (!device.tensorUnits()) return;
        const texts = [_][]u8{
            try text(self.a, device, false, false),
            try text(self.a, device, true, false),
            try text(self.a, device, false, true),
            try text(self.a, device, true, true),
        };
        defer for (texts) |source| self.a.free(source);
        var libraries: [4]?mtl.Library = @splat(null);
        defer for (libraries) |maybe| if (maybe) |library| library.deinit();
        for ([_]bool{ false, true }) |compact| for ([_]bool{ false, true }) |am| for ([_]bool{ false, true }) |an| for (0..3) |kind| {
            const exact = kind != 0;
            const control = kind == 2;
            const label = try std.fmt.allocPrint(self.a, "{s}sam3 fp32 gemm {s}{s} nt {d}{d}", .{ if (control) "control: " else "metal ", if (compact) "compact " else "", if (control) "invalid exact flag" else if (exact) "exact" else "full", @intFromBool(am), @intFromBool(an) });
            errdefer self.a.free(label);
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) {
                self.a.free(label);
                continue;
            };
            const slot = @as(usize, @intFromBool(compact)) * 2 + @intFromBool(exact);
            if (libraries[0] == null) libraries[0] = try mtl.Library.fromSource(device, texts[0], mtl.CompileOptions.mlx());
            if (libraries[slot] == null) libraries[slot] = try mtl.Library.fromSource(device, texts[slot], mtl.CompileOptions.mlx());
            var symbol: [64]u8 = undefined;
            const name_ = try std.fmt.bufPrint(&symbol, "qwv_gemm_f32{s}{s}_nt_{d}{d}", .{ if (exact) "_exact" else "", if (compact) "_compact" else "", @intFromBool(am), @intFromBool(an) });
            const pipe = try mtl.Pipeline.init(device, libraries[slot].?, name_, false);
            errdefer pipe.deinit();
            const base_name = try std.fmt.bufPrint(&symbol, "qwv_gemm_f32_nt_{d}{d}", .{ @intFromBool(am and !compact), @intFromBool(an) });
            const baseline = try mtl.Pipeline.init(device, libraries[0].?, base_name, false);
            errdefer baseline.deinit();
            const probe = try self.a.create(Probe);
            errdefer self.a.destroy(probe);
            try self.probes.ensureUnusedCapacity(self.a, 1);
            try kernels.ensureUnusedCapacity(self.a, 1);
            const tile: usize = if (compact) 32 else 64;
            probe.* = .{ .a = self.a, .rig = .{ .a = self.a, .device = device, .queue = queue }, .pipe = pipe, .baseline = baseline, .name = label, .exact = exact, .compact = compact, .control = control, .am = am, .an = an, .m = tile + @intFromBool(!am) };
            self.probes.appendAssumeCapacity(probe);
            kernels.appendAssumeCapacity(probe.kernel());
        };
    }

    pub fn deinit(self: *Suite) void {
        for (self.probes.items) |probe| {
            probe.deinit();
            self.a.destroy(probe);
        }
        self.probes.deinit(self.a);
    }
};
