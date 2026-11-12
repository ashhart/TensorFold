const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;

fn depth(shape: c.Shape) usize {
    return 32 * shape.k + 128;
}

const served = [_]c.Shape{
    .{ .rows = 4, .n = 17, .k = 33 },
    .{ .rows = 3, .n = 17, .k = 33 },
    .{ .rows = 3, .n = 1024, .k = 4096 },
    .{ .rows = 3, .n = 4096, .k = 1024 },
    .{ .rows = 32, .n = 256, .k = 2048 },
    .{ .rows = 1, .n = 1024, .k = 4736 },
    .{ .rows = 1, .n = 4736, .k = 1024 },
};

const Probe = struct {
    allocator: std.mem.Allocator,
    rig: rig_.Rig,
    pipeline: mtl.Pipeline,
    control: bool,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []f32 = &.{},
    weight: []f32 = &.{},
    bias: []f32 = &.{},
    inputs: [3]rig_.Ref = undefined,
    output: rig_.Ref = undefined,

    fn kernel(self: *Probe) g.Kernel {
        return .{
            .spec = .{
                .name = if (self.control) "control: sam3 small projection drops last K" else "metal sam3 small compensated projection",
                .accum = .f32,
                .depth = depth,
                .out = .f32,
                .served = &served,
                .tiles = .{ .{ .size = 32 }, .{ .size = 4 }, .{ .size = 32 } },
            },
            .ctx = self,
            .vt = &vt,
            .control = if (self.control) .shapes else null,
        };
    }

    const vt = g.hooks(Probe);

    pub fn reset(self: *Probe) void {
        self.rig.reset();
        self.allocator.free(self.x);
        self.allocator.free(self.weight);
        self.allocator.free(self.bias);
        self.x = &.{};
        self.weight = &.{};
        self.bias = &.{};
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 32 or shape.n == 0 or shape.n > 4736 or shape.k == 0 or shape.k > 4736) return false;
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
        } else {
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
        self.inputs = .{ try self.rig.input(std.mem.sliceAsBytes(self.x)), try self.rig.input(std.mem.sliceAsBytes(self.weight)), try self.rig.input(std.mem.sliceAsBytes(self.bias)) };
        self.output = try self.rig.output(self.count() * 4);
        return true;
    }

    pub fn run(self: *Probe) !void {
        const pass = self.rig.begin();
        pass.e.setPipeline(self.pipeline);
        for (self.inputs, 0..) |input, index| pass.e.setBuffer(input.buf, input.off, index);
        pass.e.setBuffer(self.output.buf, self.output.off, 3);
        pass.e.setValue([3]u32{ @intCast(self.shape.rows), @intCast(self.shape.n), @intCast(self.shape.k) }, 4);
        pass.e.dispatchGroups(mtl.Size.of((self.shape.n + 3) / 4, self.shape.rows, 1), mtl.Size.of(128, 1, 1));
        try self.rig.finish(pass);
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
            want[row * self.shape.n + column] = value;
            magnitude[row * self.shape.n + column] = size;
        };
    }

    pub fn intervals(self: *Probe, low: []f64, high: []f64) !void {
        @memset(low, -std.math.floatMax(f64));
        @memset(high, std.math.floatMax(f64));
        if (self.fill != .random) return;
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
    }

    pub fn simulate(self: *Probe, accumulator: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try self.allocator.alloc(f64, self.shape.k + 1);
        defer self.allocator.free(terms);
        for (0..self.shape.rows) |row| for (0..self.shape.n) |column| {
            terms[0] = self.bias[column];
            for (terms[1..], 0..) |*term, k| term.* = @as(f64, self.x[row * self.shape.k + k]) * self.weight[column * self.shape.k + k];
            out[row * self.shape.n + column] = contract.rows.sumIn(accumulator, order, terms);
        };
    }
};

pub const Suite = struct {
    allocator: std.mem.Allocator,
    probes: [2]?Probe = @splat(null),

    pub fn collect(self: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (&self.probes, 0..) |*entry, index| {
            const label = if (index == 0) "metal sam3 small compensated projection" else "control: sam3 small projection drops last K";
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            entry.* = .{ .allocator = self.allocator, .rig = .{ .a = self.allocator, .device = device, .queue = queue }, .pipeline = try rig_.pipeline(device, sources.sam3_dense_small, if (index == 0) "tf_sam3_dense_small" else "tf_sam3_dense_small_drop_tail"), .control = index != 0 };
            if (entry.*) |*probe| try kernels.append(self.allocator, probe.kernel());
        }
    }

    pub fn deinit(self: *Suite) void {
        for (&self.probes) |*entry| if (entry.*) |*probe| {
            probe.reset();
            probe.rig.deinit();
            probe.pipeline.deinit();
        };
    }
};
