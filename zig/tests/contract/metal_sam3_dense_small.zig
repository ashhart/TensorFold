const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
extern "c" fn erf(f64) f64;

fn gelu(value: f64) f64 {
    return 0.5 * value * (1 + erf(value * @sqrt(@as(f64, 0.5))));
}

fn depth(shape: c.Shape) usize {
    return 32 * shape.k + 128;
}

const served = [_]c.Shape{
    .{ .rows = 4, .n = 17, .k = 33 },
    .{ .rows = 3, .n = 17, .k = 33 },
    .{ .rows = 7, .n = 17, .k = 33 },
    .{ .rows = 8, .n = 17, .k = 33 },
    .{ .rows = 9, .n = 17, .k = 33 },
    .{ .rows = 3, .n = 1024, .k = 4096 },
    .{ .rows = 3, .n = 4096, .k = 1024 },
    .{ .rows = 32, .n = 256, .k = 2048 },
    .{ .rows = 1, .n = 1024, .k = 4736 },
    .{ .rows = 1, .n = 4736, .k = 1024 },
};

const device_served = [_]c.Shape{
    .{ .rows = 1, .n = 17, .k = 33 },
    .{ .rows = 63, .n = 65, .k = 64 },
    .{ .rows = 64, .n = 64, .k = 64 },
    .{ .rows = 65, .n = 63, .k = 65 },
    .{ .rows = 65, .n = 17, .k = 4736 },
    .{ .rows = 65, .n = 4736, .k = 33 },
    .{ .rows = 64, .n = 33, .k = 1024 },
    .{ .rows = 65, .n = 1024, .k = 588 },
};

const compensated_served = device_served ++ [_]c.Shape{
    .{ .rows = 65, .n = 256, .k = 2304 },
    .{ .rows = 64, .n = 255, .k = 2303 },
    .{ .rows = 63, .n = 257, .k = 2305 },
};

pub fn deviceText(a: std.mem.Allocator, control: bool) ![]u8 {
    const raw = try sources.tensor_library.source(a, sources.sam3_source);
    defer a.free(raw);
    return if (control) std.mem.replaceOwned(u8, a, raw, "if (domain[0]) return;", "") else a.dupe(u8, raw);
}

pub fn halfText(a: std.mem.Allocator, control: bool) ![]u8 {
    const raw = try deviceText(a, false);
    if (!control) return raw;
    defer a.free(raw);
    const store = "output[long(row + at[1]) * shape.columns + column + at[0]] = value;";
    if (std.mem.count(u8, raw, store) != 1) return error.ControlTextNotFound;
    return std.mem.replaceOwned(u8, a, raw, store, "output[long(row + at[1]) * shape.columns + column + at[0]] = float(half(value));");
}

const Kind = enum { fused, baseline, control, device, device_control, device_nn, device_nn_compensated, device_compensated, device_gelu, device_gelu_control, expansion, expansion_gelu, expansion_control, expansion_residual_control, device_half, device_half_control };
const labels = [_][]const u8{
    "metal sam3 small compensated projection",
    "metal sam3 small compensated projection baseline",
    "control: sam3 small projection drops last K",
    "metal sam3 device projection",
    "control: sam3 device projection omits domain fallback",
    "metal sam3 device projection NN",
    "metal sam3 device projection NN compensated",
    "metal sam3 device projection compensated",
    "metal sam3 device projection GELU",
    "control: sam3 device projection GELU omitted",
    "metal sam3 exact expansion projection",
    "metal sam3 exact expansion projection GELU",
    "control: sam3 exact expansion projection GELU omitted",
    "control: sam3 exact expansion projection omits residual",
    "metal sam3 device projection exact half weight",
    "control: sam3 exact half weight rounds output",
};

const Probe = struct {
    allocator: std.mem.Allocator,
    rig: rig_.Rig,
    baseline_rig: rig_.Rig,
    pipeline: mtl.Pipeline,
    baseline_pipeline: ?mtl.Pipeline = null,
    device_pipelines: [5]mtl.Pipeline = undefined,
    expansion_pipelines: [6]mtl.Pipeline = undefined,
    components: rig_.Ref = undefined,
    residuals: rig_.Ref = undefined,
    baseline_device_pipelines: [5]mtl.Pipeline = undefined,
    baseline_domain: rig_.Ref = undefined,
    kind: Kind,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []f32 = &.{},
    weight: []f32 = &.{},
    bias: []f32 = &.{},
    inputs: [3]rig_.Ref = undefined,
    output: rig_.Ref = undefined,
    baseline_inputs: [3]rig_.Ref = undefined,
    baseline_output: rig_.Ref = undefined,
    domain: rig_.Ref = undefined,

    fn device(self: *Probe) bool {
        return self.kind == .device or self.kind == .device_control or self.kind == .device_compensated or self.kn() or self.activated() or self.expansion() or self.half();
    }

    fn half(self: *const Probe) bool {
        return self.kind == .device_half or self.kind == .device_half_control;
    }

    fn activated(self: *const Probe) bool {
        return self.kind == .device_gelu or self.kind == .device_gelu_control or self.kind == .expansion_gelu or self.kind == .expansion_control;
    }

    fn expansion(self: *const Probe) bool {
        return self.kind == .expansion or self.kind == .expansion_gelu or self.kind == .expansion_control or self.kind == .expansion_residual_control;
    }

    fn kn(self: *Probe) bool {
        return self.kind == .device_nn or self.kind == .device_nn_compensated;
    }

    fn kernel(self: *Probe) g.Kernel {
        return .{
            .spec = .{
                .name = labels[@backingInt(self.kind)],
                .accum = .f32,
                .depth = depth,
                .out = .f32,
                .served = if (self.kind == .device_compensated) &compensated_served else if (self.device()) &device_served else &served,
                .tiles = if (self.device()) .{ .{ .size = 32 }, .{ .size = 64 }, .{ .size = 16 } } else .{ .{ .size = 1 }, .{ .size = 4 }, .{ .size = 32 } },
                .replaces = if (self.kind == .fused) labels[@backingInt(Kind.baseline)] else if (self.kind == .device_gelu) "separate FP32 projection and exact GELU" else if (self.half()) labels[@backingInt(Kind.device)] else null,
            },
            .ctx = self,
            .vt = &vt,
            .control = if (self.kind == .control or self.kind == .device_control or self.kind == .device_gelu_control or self.kind == .expansion_control or self.kind == .expansion_residual_control or self.kind == .device_half_control) .shapes else null,
        };
    }

    const vt = g.hooks(Probe);

    pub fn reset(self: *Probe) void {
        self.rig.reset();
        self.baseline_rig.reset();
        self.allocator.free(self.x);
        self.allocator.free(self.weight);
        self.allocator.free(self.bias);
        self.x = &.{};
        self.weight = &.{};
        self.bias = &.{};
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > (if (self.device()) @as(usize, 129) else 32) or shape.n == 0 or shape.n > 4736 or shape.k == 0 or shape.k > 4736) return false;
        self.reset();
        self.shape = shape;
        self.fill = fill;
        self.x = try self.allocator.alloc(f32, shape.rows * shape.k);
        self.weight = try self.allocator.alloc(f32, shape.n * shape.k);
        self.bias = try self.allocator.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (self.x) |*value| value.* = random.float(f32) * 4 - 2;
        for (self.weight) |*value| value.* = random.float(f32) * 4 - 2;
        if (self.expansion()) for (self.weight) |*value| {
            value.* = @floatCast(@as(f16, @floatCast(value.*)));
        };
        for (self.bias) |*value| value.* = random.float(f32) - 0.5;
        if (fill == .stress) {
            for (self.x, 0..) |*value, index| {
                const k = index % shape.k;
                value.* = if (k < 8) @floatFromInt(1024 + 64 * k) else if (k < 16) @floatFromInt(1025 + 64 * (k - 8)) else 0;
            }
            for (self.weight, 0..) |*value, index| {
                const k = index % shape.k;
                const coefficient: f32 = @floatFromInt(128 + (index / shape.k + 3 * (k % 8)) % 128);
                value.* = if (k < 8) coefficient else if (k < 16) -coefficient else 0;
            }
            @memset(self.bias, 0);
            if (self.activated()) for (self.weight) |*value| {
                value.* = -value.*;
            };
        } else if (fill == .nonfinite) {
            for (0..shape.rows) |row| {
                if (row % 5 == 0) continue;
                self.x[row * shape.k] = switch (row % 5) {
                    1 => std.math.nan(f32),
                    2, 4 => std.math.inf(f32),
                    else => -std.math.inf(f32),
                };
                if (row % 5 == 4 and shape.k > 1) self.x[row * shape.k + 1] = -std.math.inf(f32);
            }
            for (0..shape.n) |column| {
                self.weight[column * shape.k] = if (column % 4 == 1) 0 else 1;
                if (shape.k > 1) self.weight[column * shape.k + 1] = 1;
                if (column % 4 == 2) self.bias[column] = std.math.inf(f32);
                if (column % 4 == 3) self.bias[column] = -std.math.inf(f32);
            }
        } else if (!self.device() or shape.k % 2 != 0) {
            @memset(self.x[0..shape.k], 0);
            self.x[0] = @bitCast(@as(u32, 1));
            self.weight[0] = 0x1p100;
            self.bias[0] = 0;
            if (shape.n > 1) {
                self.weight[shape.k] = 1;
                self.bias[1] = 0;
            }
            if (shape.k > 1) {
                self.x[shape.k - 1] = 1;
                self.weight[shape.k - 1] = 1;
                if (shape.n > 1) self.weight[2 * shape.k - 1] = 0;
            }
            if (shape.rows > 2 and shape.k > 2) {
                @memset(self.x[2 * shape.k ..][0..shape.k], 0);
                self.x[2 * shape.k] = std.math.floatMax(f32);
                self.x[2 * shape.k + 1] = std.math.floatMax(f32);
                for (0..shape.n) |column| self.weight[column * shape.k + 1] = -self.weight[column * shape.k];
            }
        }
        if (self.expansion() and fill == .random and shape.k % 2 == 0) {
            @memset(self.x[0..shape.k], 0);
            self.x[0] = 0x1p-35 + 0x1p-47;
            for (0..shape.n) |column| self.weight[column * shape.k] = 1;
            @memset(self.bias, 0);
        }
        if (self.half()) {
            if (fill == .random and shape.k % 2 != 0) self.weight[0] = 0x1p15;
            if (fill == .random and shape.k % 2 == 0) {
                @memset(self.x[0..shape.k], 0);
                self.x[0] = 1;
                for (0..shape.n) |column| self.weight[column * shape.k] = @floatCast(@as(f16, @bitCast(@as(u16, @intCast(1 + column % 1023)))));
                @memset(self.bias, 0);
                if (shape.rows > 1) {
                    @memset(self.x[shape.k..][0..shape.k], 0);
                    self.x[shape.k] = 1 + 0x1p-20;
                }
            }
            if (fill == .nonfinite and shape.n > 6) {
                self.weight[4 * shape.k] = std.math.inf(f32);
                self.weight[5 * shape.k] = -std.math.inf(f32);
                self.weight[6 * shape.k] = std.math.nan(f32);
            }
            for (self.weight) |*value| value.* = @floatCast(@as(f16, @floatCast(value.*)));
        }
        const upload_weight = if (self.kn()) transposed: {
            const values = try self.allocator.alloc(f32, self.weight.len);
            for (0..shape.n) |col| {
                for (0..shape.k) |k| values[k * shape.n + col] = self.weight[col * shape.k + k];
            }
            break :transposed values;
        } else self.weight;
        defer if (self.kn()) self.allocator.free(upload_weight);
        const weight_input = if (self.half()) half_buffer: {
            const values = try self.allocator.alloc(f16, upload_weight.len);
            defer self.allocator.free(values);
            for (values, upload_weight) |*target, value| target.* = @floatCast(value);
            break :half_buffer try self.rig.input(std.mem.sliceAsBytes(values));
        } else try self.rig.input(std.mem.sliceAsBytes(upload_weight));
        self.inputs = .{ try self.rig.input(std.mem.sliceAsBytes(self.x)), weight_input, try self.rig.input(std.mem.sliceAsBytes(self.bias)) };
        self.output = try self.rig.output(self.count() * 4);
        if (self.device()) self.domain = try self.rig.scratch(if (self.expansion()) 8 * 4 else 7 * 4);
        if (self.expansion()) self.components = try self.rig.scratch((3 * self.x.len + self.weight.len) * 2);
        if (self.expansion()) self.residuals = try self.rig.scratch(self.shape.rows * 36);
        if (self.kind == .fused or (self.activated() and !self.expansion()) or self.half()) {
            self.baseline_inputs = .{ try self.baseline_rig.input(std.mem.sliceAsBytes(self.x)), try self.baseline_rig.input(std.mem.sliceAsBytes(self.weight)), try self.baseline_rig.input(std.mem.sliceAsBytes(self.bias)) };
            self.baseline_output = try self.baseline_rig.output(self.count() * 4);
            if (self.activated() or self.half()) self.baseline_domain = try self.baseline_rig.scratch(7 * 4);
        }
        return true;
    }

    pub fn run(self: *Probe) !void {
        if (self.expansion()) {
            const pass = self.rig.begin();
            const Ref = struct { buffer: mtl.Buffer, offset: usize };
            var refs: [7]Ref = undefined;
            for (self.inputs, refs[0..3]) |input, *ref| ref.* = .{ .buffer = input.buf, .offset = input.off };
            refs[3] = .{ .buffer = self.output.buf, .offset = self.output.off };
            refs[4] = .{ .buffer = self.domain.buf, .offset = self.domain.off };
            refs[5] = .{ .buffer = self.components.buf, .offset = self.components.off };
            refs[6] = .{ .buffer = self.residuals.buf, .offset = self.residuals.off };
            sources.sam3_dense_expansion.encode(mtl, pass.e, self.expansion_pipelines, refs, .{ @intCast(self.shape.rows), @intCast(self.shape.n), @intCast(self.shape.k) });
            return self.rig.finish(pass);
        }
        if (self.device()) {
            const pass = self.rig.begin();
            const Ref = struct { buffer: mtl.Buffer, offset: usize };
            var refs: [5]Ref = undefined;
            for (self.inputs, refs[0..3]) |input, *ref| ref.* = .{ .buffer = input.buf, .offset = input.off };
            refs[3] = .{ .buffer = self.output.buf, .offset = self.output.off };
            refs[4] = .{ .buffer = self.domain.buf, .offset = self.domain.off };
            sources.sam3_dense_device.encode(mtl, pass.e, self.device_pipelines, refs, .{ @intCast(self.shape.rows), @intCast(self.shape.n), @intCast(self.shape.k) }, self.kind != .device_control);
            return self.rig.finish(pass);
        }
        return self.launch(&self.rig, self.pipeline, self.inputs, self.output);
    }

    fn launch(self: *Probe, rig: *rig_.Rig, pipeline: mtl.Pipeline, inputs: [3]rig_.Ref, output: rig_.Ref) !void {
        const pass = rig.begin();
        pass.e.setPipeline(pipeline);
        for (inputs, 0..) |input, index| pass.e.setBuffer(input.buf, input.off, index);
        pass.e.setBuffer(output.buf, output.off, 3);
        pass.e.setValue([3]u32{ @intCast(self.shape.rows), @intCast(self.shape.n), @intCast(self.shape.k) }, 4);
        pass.e.dispatchGroups(mtl.Size.of((self.shape.n + 3) / 4, self.shape.rows, 1), mtl.Size.of(128, 1, 1));
        try rig.finish(pass);
    }

    pub fn replaced(self: *Probe) ![]const u8 {
        if (self.activated() or self.half()) {
            const pass = self.baseline_rig.begin();
            const Ref = struct { buffer: mtl.Buffer, offset: usize };
            var refs: [5]Ref = undefined;
            for (self.baseline_inputs, refs[0..3]) |input, *ref| ref.* = .{ .buffer = input.buf, .offset = input.off };
            refs[3] = .{ .buffer = self.baseline_output.buf, .offset = self.baseline_output.off };
            refs[4] = .{ .buffer = self.baseline_domain.buf, .offset = self.baseline_domain.off };
            sources.sam3_dense_device.encode(mtl, pass.e, self.baseline_device_pipelines, refs, .{ @intCast(self.shape.rows), @intCast(self.shape.n), @intCast(self.shape.k) }, true);
            if (self.activated()) {
                pass.e.setPipeline(self.baseline_pipeline.?);
                pass.e.setBuffer(self.baseline_output.buf, self.baseline_output.off, 0);
                pass.e.setBuffer(self.baseline_output.buf, self.baseline_output.off, 1);
                pass.e.setValue(@as(u32, @intCast(self.count())), 2);
                pass.e.dispatchThreads(mtl.Size.of(self.count(), 1, 1), mtl.Size.of(256, 1, 1));
            }
            try self.baseline_rig.finish(pass);
            if (!try self.baseline_rig.intact()) return error.BaselineGuardsChanged;
            return self.baseline_rig.outputBytes();
        }
        try self.launch(&self.baseline_rig, self.baseline_pipeline.?, self.baseline_inputs, self.baseline_output);
        if (!try self.baseline_rig.intact()) return error.BaselineGuardsChanged;
        return self.baseline_rig.outputBytes();
    }

    pub fn count(self: *Probe) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *Probe, want: []f64, magnitude: []f64) !void {
        for (0..self.shape.rows) |row| for (0..self.shape.n) |column| {
            var value: f64 = self.bias[column];
            var size = @abs(value);
            for (0..self.shape.k) |k| {
                const product = @as(f64, self.x[row * self.shape.k + k]) * self.weight[column * self.shape.k + k];
                value += product;
                size += @abs(product);
            }
            want[row * self.shape.n + column] = if (self.activated()) gelu(value) else value;
            magnitude[row * self.shape.n + column] = if (self.activated()) 1.13 * size else size;
        };
    }

    pub fn intervals(self: *Probe, low: []f64, high: []f64) !void {
        @memset(low, -std.math.floatMax(f64));
        @memset(high, std.math.floatMax(f64));
        if (self.fill != .random or (self.device() and self.shape.k % 2 == 0)) return;
        low[0] = if (self.shape.k > 1) 1 else 0x1p-49;
        high[0] = low[0];
        if (self.shape.n > 1) {
            low[1] = 0x1p-149;
            high[1] = low[1];
        }
        if (self.shape.rows > 2 and self.shape.k > 2) {
            low[2 * self.shape.n] = 0;
            high[2 * self.shape.n] = 0;
        }
        if (self.activated()) for (low, high) |*minimum, *maximum| {
            if (minimum.* == maximum.* and minimum.* >= 0) minimum.* = 0;
        };
    }

    pub fn simulate(self: *Probe, accumulator: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try self.allocator.alloc(f64, self.shape.k + 1);
        defer self.allocator.free(terms);
        for (0..self.shape.rows) |row| for (0..self.shape.n) |column| {
            terms[0] = self.bias[column];
            for (terms[1..], 0..) |*term, k| term.* = @as(f64, self.x[row * self.shape.k + k]) * self.weight[column * self.shape.k + k];
            const value = contract.rows.sumIn(accumulator, order, terms);
            out[row * self.shape.n + column] = if (self.activated()) gelu(value) else value;
        };
    }
};

pub const Suite = struct {
    allocator: std.mem.Allocator,
    probes: [labels.len]?Probe = @splat(null),

    pub fn collect(self: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (&self.probes, 0..) |*entry, index| {
            const label = labels[index];
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            if (index >= 3 and !device.tensorUnits()) continue;
            entry.* = .{
                .allocator = self.allocator,
                .rig = .{ .a = self.allocator, .device = device, .queue = queue },
                .baseline_rig = .{ .a = self.allocator, .device = device, .queue = queue },
                .pipeline = if (index >= 3) undefined else try rig_.pipeline(device, if (index == 1) sources.sam3_dense_small_baseline else sources.sam3_dense_small, if (index == 2) "tf_sam3_dense_small_drop_tail" else "tf_sam3_dense_small"),
                .kind = @fromBackingInt(@intCast(index)),
            };
            if (entry.*) |*probe| {
                if (probe.kind == .fused) probe.baseline_pipeline = try rig_.pipeline(device, sources.sam3_dense_small_baseline, "tf_sam3_dense_small");
                if (probe.expansion()) {
                    const source = try deviceText(self.allocator, false);
                    defer self.allocator.free(source);
                    var names = if (probe.kind == .expansion_gelu) sources.sam3_dense_expansion.gelu_names else sources.sam3_dense_expansion.names;
                    if (probe.kind == .expansion_residual_control) names[4] = "sam_dense_expansion_drop_residual";
                    for (names, &probe.expansion_pipelines) |name, *pipe| pipe.* = try rig_.pipeline(device, source, name);
                } else if (probe.device()) {
                    const source = if (probe.half()) try halfText(self.allocator, probe.kind == .device_half_control) else try deviceText(self.allocator, probe.kind == .device_control);
                    defer self.allocator.free(source);
                    const names = if (probe.half()) sources.sam3_dense_device.half_names else if (probe.kind == .device_gelu) sources.sam3_dense_device.gelu_names else if (probe.kind == .device_compensated) sources.sam3_dense_device.compensated_names else if (probe.kind == .device_nn_compensated) sources.sam3_dense_device.nn_compensated_names else if (probe.kn()) sources.sam3_dense_device.nn_names else sources.sam3_dense_device.names;
                    for (names, &probe.device_pipelines) |name, *pipe| pipe.* = try rig_.pipeline(device, source, name);
                    if (probe.activated()) {
                        for (sources.sam3_dense_device.gelu_linear_names, &probe.baseline_device_pipelines) |name, *pipe| pipe.* = try rig_.pipeline(device, source, name);
                        probe.baseline_pipeline = try rig_.pipeline(device, source, "im_sam_gelu");
                    } else if (probe.half()) {
                        for (sources.sam3_dense_device.names, &probe.baseline_device_pipelines) |name, *pipe| pipe.* = try rig_.pipeline(device, source, name);
                    }
                }
                try kernels.append(self.allocator, probe.kernel());
            }
        }
    }

    pub fn deinit(self: *Suite) void {
        for (&self.probes) |*entry| if (entry.*) |*probe| {
            probe.reset();
            probe.rig.deinit();
            probe.baseline_rig.deinit();
            if (probe.expansion()) {
                for (probe.expansion_pipelines) |pipe| pipe.deinit();
            } else if (probe.device()) {
                for (probe.device_pipelines) |pipe| pipe.deinit();
            } else probe.pipeline.deinit();
            if (probe.baseline_pipeline) |pipeline| pipeline.deinit();
            if ((probe.activated() and !probe.expansion()) or probe.half()) for (probe.baseline_device_pipelines) |pipeline| pipeline.deinit();
        };
    }
};
