const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;

const Params = extern struct { h: u32, w: u32, c: u32, n: u32, k: u32, oh: u32, ow: u32, stride: u32, pad: u32, first: u32, rows: u32 };
const name = "metal sam3 deconvolution four outputs";
const unaligned_name = "metal sam3 deconvolution unaligned weights";
const control_name = "control: sam3 deconvolution fp16 sums";

fn depth(shape: c.Shape) usize {
    return shape.k + 1;
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    baseline: mtl.Pipeline,
    control: bool,
    unaligned: bool,
    shape: c.Shape = undefined,
    p: Params = undefined,
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []f32 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    base: rig_.Ref = undefined,

    fn kernel(self: *Probe) g.Kernel {
        return .{
            .spec = .{
                .name = if (self.control) control_name else if (self.unaligned) unaligned_name else name,
                .accum = .f32,
                .depth = depth,
                .out = .f32,
                .replaces = "one-output SAM FP32 deconvolution",
                .served = &.{
                    .{ .rows = 72 * 72, .n = 512, .k = 1024 },
                    .{ .rows = 144 * 144, .n = 256, .k = 512 },
                    .{ .rows = 72 * 72, .n = 64, .k = 256 },
                    .{ .rows = 144 * 144, .n = 32, .k = 64 },
                    .{ .rows = 15, .n = 7, .k = 33 },
                },
                .tiles = .{ .{ .size = 256 }, .{}, .{} },
            },
            .ctx = self,
            .vt = &vt,
            .control = if (self.control) .sums else null,
        };
    }

    const vt = g.hooks(Probe);

    fn free(self: *Probe) void {
        self.a.free(self.x);
        self.a.free(self.w);
        self.a.free(self.b);
        self.x = &.{};
        self.w = &.{};
        self.b = &.{};
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k < 2) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        const width: usize = if (shape.rows == 72 * 72) 72 else if (shape.rows == 144 * 144) 144 else if (shape.rows == 15) 5 else 1;
        self.p = .{ .h = @intCast(shape.rows / width), .w = @intCast(width), .c = @intCast(shape.k), .n = @intCast(shape.n), .k = 2, .oh = @intCast(2 * (shape.rows / width)), .ow = @intCast(2 * width), .stride = 2, .pad = 0, .first = 0, .rows = 0 };
        self.x = try self.a.alloc(f32, shape.rows * shape.k);
        self.w = try self.a.alloc(f32, shape.k * shape.n * 4);
        self.b = try self.a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.x) |*v| v.* = rng.float(f32) * 4 - 2;
        for (self.w) |*v| v.* = rng.float(f32) * 4 - 2;
        for (self.b) |*v| v.* = rng.float(f32) * 4 - 2;
        if (fill == .stress) {
            @memset(self.x, 1);
            @memset(self.b, 0);
            for (0..shape.k) |i| {
                const half = shape.k / 2;
                const value: f32 = if (i < half) 4095 else if (i < 2 * half) -4094 else 0;
                @memset(self.w[i * shape.n * 4 ..][0 .. shape.n * 4], value);
            }
        }
        if (fill == .nonfinite) {
            self.x[0] = std.math.nan(f32);
            self.x[shape.k] = std.math.inf(f32);
            self.x[2 * shape.k] = -std.math.inf(f32);
            self.x[3 * shape.k] = std.math.inf(f32);
            self.x[3 * shape.k + 1] = -std.math.inf(f32);
            for (0..shape.n * 4) |i| {
                self.w[i] = 1;
                self.w[shape.n * 4 + i] = 1;
            }
        }
        for ([_][]const f32{ self.x, self.w, self.b }, &self.ins, 0..) |values, *input, i| {
            const bytes = std.mem.sliceAsBytes(values);
            if (i == 1 and self.unaligned) {
                const shifted = try self.a.alloc(u8, bytes.len + 4);
                defer self.a.free(shifted);
                @memset(shifted[0..4], 0);
                @memcpy(shifted[4..], bytes);
                input.* = try self.rig.input(shifted);
                input.off += 4;
            } else input.* = try self.rig.input(bytes);
        }
        self.out = try self.rig.output(self.count() * 4);
        self.base = try self.rig.scratch(self.count() * 4);
        return true;
    }

    fn launch(self: *Probe, baseline: bool) !void {
        const pass = self.rig.begin();
        pass.e.setPipeline(if (baseline) self.baseline else self.pipe);
        for (self.ins, 0..) |input, i| pass.e.setBuffer(input.buf, input.off, i);
        const output = if (baseline) self.base else self.out;
        pass.e.setBuffer(output.buf, output.off, 3);
        pass.e.setValue(self.p, 4);
        const count_ = if (baseline) self.count() else self.count() / 4;
        pass.e.dispatchThreads(mtl.Size.of(count_, 1, 1), mtl.Size.of(256, 1, 1));
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
        return self.shape.rows * self.shape.n * 4;
    }

    fn location(self: *Probe, i: usize) struct { pixel: usize, weight: usize, col: usize } {
        const col = i % self.shape.n;
        const y = i / self.shape.n / self.p.ow;
        const x = i / self.shape.n % self.p.ow;
        return .{ .pixel = (y / 2 * self.p.w + x / 2), .weight = col * 4 + y % 2 * 2 + x % 2, .col = col };
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*v, *m, i| {
            const at = self.location(i);
            v.* = self.b[at.col];
            m.* = @abs(v.*);
            for (0..self.shape.k) |k_| {
                const term = @as(f64, self.x[at.pixel * self.shape.k + k_]) * self.w[k_ * self.shape.n * 4 + at.weight];
                v.* += term;
                m.* += @abs(term);
            }
        }
    }

    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try self.a.alloc(f64, self.shape.k + 1);
        defer self.a.free(terms);
        for (out, 0..) |*v, i| {
            const at = self.location(i);
            terms[0] = self.b[at.col];
            for (terms[1..], 0..) |*term, k_| term.* = @as(f64, self.x[at.pixel * self.shape.k + k_]) * self.w[k_ * self.shape.n * 4 + at.weight];
            v.* = contract.rows.sumIn(acc, order, terms);
        }
    }

    fn deinit(self: *Probe) void {
        self.free();
        self.rig.deinit();
        self.pipe.deinit();
        self.baseline.deinit();
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [3]?*Probe = @splat(null),

    pub fn collect(self: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (0..3) |i| {
            const label = if (i == 2) control_name else if (i == 1) unaligned_name else name;
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            const original = sources.sam3_deconv ++ sources.sam3_deconv_baseline;
            if (std.mem.count(u8, original, "sum+=x[pixel*p.c+c]*weight;") != 1) return error.ControlTextNotFoundOnce;
            const source = if (i == 2) try std.mem.replaceOwned(u8, self.a, original, "sum+=x[pixel*p.c+c]*weight;", "sum=float4(half4(sum+x[pixel*p.c+c]*weight));") else try self.a.dupe(u8, original);
            defer self.a.free(source);
            const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
            defer lib.deinit();
            const probe = try self.a.create(Probe);
            errdefer self.a.destroy(probe);
            try kernels.ensureUnusedCapacity(self.a, 1);
            const pipe = try mtl.Pipeline.init(device, lib, "im_sam_deconv", false);
            errdefer pipe.deinit();
            const baseline = try mtl.Pipeline.init(device, lib, "im_sam_deconv_baseline", false);
            probe.* = .{ .a = self.a, .rig = .{ .a = self.a, .device = device, .queue = queue }, .pipe = pipe, .baseline = baseline, .control = i == 2, .unaligned = i == 1 };
            self.probes[i] = probe;
            kernels.appendAssumeCapacity(probe.kernel());
        }
    }

    pub fn deinit(self: *Suite) void {
        for (self.probes) |maybe| if (maybe) |probe| {
            probe.deinit();
            self.a.destroy(probe);
        };
    }
};
