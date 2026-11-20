const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const kernel = sources.sam3_dense_planes;
extern "c" fn erf(f64) f64;

fn gelu(value: f64) f64 {
    return 0.5 * value * (1 + erf(value * @sqrt(@as(f64, 0.5))));
}

// FP32 accumulation over the depth in each plane pass, plus the bias and plane recombination.
fn depth(shape: c.Shape) usize {
    return shape.k + 8;
}

const served = [_]c.Shape{
    .{ .rows = 1, .n = 17, .k = 33 },
    .{ .rows = 63, .n = 129, .k = 64 },
    .{ .rows = 65, .n = 127, .k = 65 },
    .{ .rows = 64, .n = 3072, .k = 1024 },
    .{ .rows = 65, .n = 1024, .k = 1024 },
    .{ .rows = 33, .n = 4736, .k = 1024 },
    .{ .rows = 33, .n = 1024, .k = 4736 },
    .{ .rows = 31, .n = 1023, .k = 4735 },
};

pub fn text(a: std.mem.Allocator, control: bool) ![]u8 {
    const raw = try sources.tensor_library.source(a, sources.sam3_source);
    if (!control) return raw;
    defer a.free(raw);
    const middle = "for (ushort e = 0; e < count; ++e) sam_pair_add(sums[e], ldexp(product[e], -11));";
    if (std.mem.count(u8, raw, middle) != 1) return error.ControlTextNotFound;
    return std.mem.replaceOwned(u8, a, raw, middle, "for (ushort e = 0; e < count; ++e) {}");
}

const Kind = enum { plain, compensated, residual, chained, gelu, control };
const labels = [_][]const u8{
    "metal sam3 planes projection",
    "metal sam3 planes projection compensated",
    "metal sam3 planes projection residual",
    "metal sam3 planes projection compensated residual from planes",
    "metal sam3 planes projection GELU planes",
    "control: sam3 planes projection drops the middle plane",
};
const Binding = struct { buffer: mtl.Buffer, offset: usize };

fn bind(ref: rig_.Ref, extra: usize) Binding {
    return .{ .buffer = ref.buf, .offset = ref.off + extra };
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: Kind,
    pipes: [9]mtl.Pipeline = undefined,
    pack: mtl.Pipeline = undefined,
    copy: mtl.Pipeline = undefined,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    weight: []f16 = &.{},
    bias: []f32 = &.{},
    residual: []f32 = &.{},
    refs: [8]rig_.Ref = undefined,

    fn contractKernel(self: *Probe) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .control = if (self.kind == .control) .shapes else null, .spec = .{
            .name = labels[@backingInt(self.kind)],
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .served = &served,
            .tiles = .{ .{ .size = 32 }, .{ .size = 128 }, .{ .size = 64 } },
        } };
    }

    const vt = g.hooks(Probe);

    fn epilogue(self: *const Probe) kernel.Epilogue {
        return switch (self.kind) {
            .residual, .chained => .residual,
            .gelu => .gelu_planes,
            else => .none,
        };
    }

    fn reset(self: *Probe) void {
        self.rig.reset();
        for ([_][]f32{ self.x, self.bias, self.residual }) |values| self.a.free(values);
        self.a.free(self.weight);
        self.x = &.{};
        self.bias = &.{};
        self.residual = &.{};
        self.weight = &.{};
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 129 or shape.n == 0 or shape.n > 4736 or shape.k == 0 or shape.k > 4736) return false;
        self.reset();
        self.shape = shape;
        self.x = try self.a.alloc(f32, shape.rows * shape.k);
        self.weight = try self.a.alloc(f16, shape.n * shape.k);
        self.bias = try self.a.alloc(f32, shape.n);
        self.residual = try self.a.alloc(f32, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (self.x) |*value| value.* = random.float(f32) * 4 - 2;
        for (self.weight) |*value| value.* = @floatCast(random.float(f32) * 4 - 2);
        for (self.bias) |*value| value.* = random.float(f32) - 0.5;
        for (self.residual) |*value| value.* = random.float(f32) * 8 - 4;
        if (fill == .stress) {
            for (self.x, 0..) |*value, index| {
                const k = index % shape.k;
                value.* = if (k < 8) @floatFromInt(1024 + 64 * k) else if (k < 16) @floatFromInt(1025 + 64 * (k - 8)) else 0;
            }
            for (self.weight, 0..) |*value, index| {
                const k = index % shape.k;
                const coefficient: f32 = @floatFromInt(128 + (index / shape.k + 3 * (k % 8)) % 128);
                value.* = @floatCast(if (k < 8) coefficient else if (k < 16) -coefficient else 0);
            }
            @memset(self.bias, 0);
            @memset(self.residual, 0);
        } else if (fill == .nonfinite) {
            for (0..shape.rows) |row| {
                if (row % 5 == 0) continue;
                self.x[row * shape.k] = switch (row % 5) {
                    1 => std.math.nan(f32),
                    2, 4 => std.math.inf(f32),
                    else => -std.math.inf(f32),
                };
            }
            for (0..shape.n) |column| {
                self.weight[column * shape.k] = if (column % 4 == 1) 0 else 1;
                if (column % 4 == 2) self.bias[column] = std.math.inf(f32);
                if (column % 4 == 3) self.bias[column] = -std.math.inf(f32);
            }
        }
        const n = shape.rows * shape.k;
        const outputs = shape.rows * shape.n;
        self.refs[0] = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.refs[1] = try self.rig.input(std.mem.sliceAsBytes(self.weight));
        self.refs[2] = try self.rig.input(std.mem.sliceAsBytes(self.bias));
        self.refs[6] = try self.rig.input(std.mem.sliceAsBytes(self.residual));
        self.refs[3] = try self.rig.output(outputs * 4);
        self.refs[4] = try self.rig.scratch(kernel.planesCount(n) * 2);
        self.refs[5] = try self.rig.scratch(16 * 4);
        self.refs[7] = if (self.kind == .gelu) try self.rig.scratch(kernel.planesCount(outputs) * 2) else try self.rig.scratch(outputs * 4);
        return true;
    }

    pub fn run(self: *Probe) !void {
        const shape = [3]u32{ @intCast(self.shape.rows), @intCast(self.shape.n), @intCast(self.shape.k) };
        const n = self.shape.rows * self.shape.k;
        const outputs = self.shape.rows * self.shape.n;
        const pass = self.rig.begin();
        const e = pass.e;
        if (self.kind == .chained) {
            e.setPipeline(self.pipes[0]);
            for ([_]usize{ 4, 5, 4 }, 0..) |index, slot| e.setBuffer(self.refs[index].buf, self.refs[index].off, slot);
            e.setValue(shape, 3);
            e.setValue([2]u32{ 1, 0 }, 4);
            e.dispatchThreads(mtl.Size.of(16, 1, 1), mtl.Size.of(16, 1, 1));
            e.barrier();
            e.setPipeline(self.pack);
            for ([_]usize{ 0, 1, 2, 4, 5 }, 0..) |index, slot| e.setBuffer(self.refs[index].buf, self.refs[index].off, slot);
            e.setValue(shape, 5);
            e.dispatchThreads(mtl.Size.of(@min(n, 1 << 20), 1, 1), mtl.Size.of(128, 1, 1));
            e.barrier();
        }
        var refs: [8]Binding = undefined;
        for (self.refs, &refs) |ref, *binding| binding.* = bind(ref, 0);
        if (self.kind == .chained) refs[0] = refs[4];
        if (self.epilogue() == .none) refs[7] = refs[3];
        // GELU planes go to scratch and come back through the restore area into the FP32 output.
        if (self.kind == .gelu) refs[3] = refs[7];
        const emitted = refs[7];
        if (self.kind == .gelu) refs[7] = bind(self.refs[3], 0);
        kernel.encode(mtl, e, self.pipes, refs, shape, self.kind != .chained, self.epilogue());
        if (self.kind == .gelu) {
            e.setPipeline(self.pipes[4]);
            e.setBuffer(emitted.buffer, emitted.offset, 0);
            e.setValue([3]u32{ @intCast(outputs), 0, 1 }, 1);
            e.dispatchThreads(mtl.Size.of(outputs, 1, 1), mtl.Size.of(128, 1, 1));
            e.barrier();
            e.setPipeline(self.copy);
            e.setBuffer(emitted.buffer, emitted.offset + kernel.restoreOffset(outputs), 0);
            e.setBuffer(self.refs[3].buf, self.refs[3].off, 1);
            e.setValue(@as(u32, @intCast(outputs)), 2);
            e.dispatchThreads(mtl.Size.of(outputs, 1, 1), mtl.Size.of(128, 1, 1));
        }
        try self.rig.finish(pass);
    }

    pub fn count(self: *Probe) usize {
        return self.shape.rows * self.shape.n;
    }

    fn terms(self: *Probe, row: usize, column: usize, out: []f64) void {
        out[0] = self.bias[column];
        for (out[1..], 0..) |*term, k| term.* = @as(f64, self.x[row * self.shape.k + k]) * self.weight[column * self.shape.k + k];
    }

    fn finish(self: *Probe, row: usize, column: usize, value: f64) f64 {
        return switch (self.kind) {
            .gelu => gelu(value),
            .residual, .chained => self.residual[row * self.shape.n + column] + value,
            else => value,
        };
    }

    pub fn reference(self: *Probe, want: []f64, magnitude: []f64) !void {
        const values = try self.a.alloc(f64, self.shape.k + 1);
        defer self.a.free(values);
        for (0..self.shape.rows) |row| for (0..self.shape.n) |column| {
            self.terms(row, column, values);
            var sum: f64 = 0;
            var size: f64 = 0;
            for (values) |term| {
                sum += term;
                size += @abs(term);
            }
            const at = row * self.shape.n + column;
            want[at] = self.finish(row, column, sum);
            magnitude[at] = if (self.kind == .gelu) 1.13 * size else if (self.kind == .residual or self.kind == .chained) size + @abs(self.residual[at]) else size;
        };
    }

    pub fn simulate(self: *Probe, accumulator: c.Accum, order: g.Order, out: []f64) !void {
        const values = try self.a.alloc(f64, self.shape.k + 1);
        defer self.a.free(values);
        for (0..self.shape.rows) |row| for (0..self.shape.n) |column| {
            self.terms(row, column, values);
            out[row * self.shape.n + column] = self.finish(row, column, contract.rows.sumIn(accumulator, order, values));
        };
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [labels.len]?Probe = @splat(null),

    pub fn collect(self: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        if (!device.tensorUnits()) return;
        for (&self.probes, 0..) |*entry, index| {
            if (only) |needle| if (std.mem.indexOf(u8, labels[index], needle) == null) continue;
            const kind: Kind = @fromBackingInt(@intCast(index));
            const source = try text(self.a, kind == .control);
            defer self.a.free(source);
            entry.* = .{ .a = self.a, .rig = .{ .a = self.a, .device = device, .queue = queue }, .kind = kind };
            const probe = &entry.*.?;
            for (kernel.names, probe.pipes[0..8]) |name, *pipe| pipe.* = try rig_.pipeline(device, source, name);
            const compensated = kind == .compensated or kind == .chained or kind == .gelu;
            const residual = kind == .residual or kind == .chained;
            const gemm = if (compensated) (if (residual) "sam_dense_planes_compensated_residual" else "sam_dense_planes_compensated") else if (residual) "sam_dense_planes_residual" else "sam_dense_planes";
            probe.pipes[8] = try rig_.pipeline(device, source, gemm);
            probe.pack = try rig_.pipeline(device, source, "sam_dense_planes_pack");
            probe.copy = try rig_.pipeline(device, source, "im_sam_copy");
            try kernels.append(self.a, probe.contractKernel());
        }
    }

    pub fn deinit(self: *Suite) void {
        for (&self.probes) |*entry| if (entry.*) |*probe| {
            probe.reset();
            probe.rig.deinit();
            for (probe.pipes) |pipe| pipe.deinit();
            probe.pack.deinit();
            probe.copy.deinit();
        };
    }
};
