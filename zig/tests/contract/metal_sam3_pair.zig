const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const kernel = sources.sam3_dense_pair;
extern "c" fn erf(f64) f64;

fn gelu(value: f64) f64 {
    return 0.5 * value * (1 + erf(value * @sqrt(@as(f64, 0.5))));
}

// 2-plane FP16 (22-bit mantissa, <= 4u/product) + FP32 accumulation across K, plus bias/epilogue.
fn depth(shape: c.Shape) usize {
    return shape.k + 12;
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
    const needle = "const float val = fma(tg[at[1] * BN + at[0]], 0x1p-11f, acc[e]) + bias[c];";
    if (std.mem.count(u8, raw, needle) != 1) return error.ControlTextNotFoundOnce;
    return std.mem.replaceOwned(u8, a, raw, needle, "const float val = acc[e] + bias[c];");
}

const Kind = enum { plain, residual, chained, gelu, control };
const labels = [_][]const u8{
    "metal sam3 pair projection plain",
    "metal sam3 pair projection residual",
    "metal sam3 pair projection residual from pair",
    "metal sam3 pair projection GELU pair",
    "control: sam3 pair projection drops cross planes",
};
const Binding = struct { buffer: mtl.Buffer, offset: usize };

fn bind(ref: rig_.Ref, extra: usize) Binding {
    return .{ .buffer = ref.buf, .offset = ref.off + extra };
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: Kind,
    helpers: [7]mtl.Pipeline = undefined,
    main: mtl.Pipeline = undefined,
    prep_clear: mtl.Pipeline = undefined,
    prep_w: mtl.Pipeline = undefined,
    copy: mtl.Pipeline = undefined,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    weight: []f32 = &.{},
    bias: []f32 = &.{},
    residual: []f32 = &.{},
    // 0=x, 1=w_pair, 2=w_f32, 3=bias, 4=out_f32, 5=x_pair, 6=plan, 7=residual, 8=scratch_or_pair_out
    refs: [9]rig_.Ref = undefined,

    fn contractKernel(self: *Probe) g.Kernel {
        const bm: usize = if (self.epilogue() == .none) 64 else 32;
        return .{ .ctx = self, .vt = &vt, .control = if (self.kind == .control) .shapes else null, .spec = .{
            .name = labels[@backingInt(self.kind)],
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .served = &served,
            .tiles = .{ .{ .size = bm }, .{ .size = 128 }, .{ .size = 64 } },
        } };
    }

    const vt = g.hooks(Probe);

    fn epilogue(self: *const Probe) kernel.Epilogue {
        return switch (self.kind) {
            .residual, .chained => .residual,
            .gelu => .gelu_pair,
            else => .none,
        };
    }

    fn reset(self: *Probe) void {
        self.rig.reset();
        for ([_][]f32{ self.x, self.weight, self.bias, self.residual }) |values| self.a.free(values);
        self.x = &.{};
        self.weight = &.{};
        self.bias = &.{};
        self.residual = &.{};
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 129 or shape.n == 0 or shape.n > 4736 or shape.k == 0 or shape.k > 4736) return false;
        self.reset();
        self.shape = shape;
        self.x = try self.a.alloc(f32, shape.rows * shape.k);
        self.weight = try self.a.alloc(f32, shape.n * shape.k);
        self.bias = try self.a.alloc(f32, shape.n);
        self.residual = try self.a.alloc(f32, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (self.x) |*value| value.* = random.float(f32) * 4 - 2;
        for (self.weight) |*value| value.* = random.float(f32) * 4 - 2;
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
                value.* = if (k < 8) coefficient else if (k < 16) -coefficient else 0;
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
        const nw = shape.n * shape.k;
        const outputs = shape.rows * shape.n;
        self.refs[0] = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.refs[1] = try self.rig.scratch(kernel.weightPairCount(nw) * 2);
        self.refs[2] = try self.rig.input(std.mem.sliceAsBytes(self.weight));
        self.refs[3] = try self.rig.input(std.mem.sliceAsBytes(self.bias));
        self.refs[4] = try self.rig.output(outputs * 4);
        self.refs[5] = try self.rig.scratch(kernel.pairCount(n) * 2);
        self.refs[6] = try self.rig.scratch(16 * 4);
        self.refs[7] = try self.rig.input(std.mem.sliceAsBytes(self.residual));
        self.refs[8] = if (self.kind == .gelu) try self.rig.scratch(kernel.pairCount(outputs) * 2) else try self.rig.scratch(outputs * 4);
        return true;
    }

    pub fn run(self: *Probe) !void {
        const shape = [3]u32{ @intCast(self.shape.rows), @intCast(self.shape.n), @intCast(self.shape.k) };
        const n = self.shape.rows * self.shape.k;
        const nw = self.shape.n * self.shape.k;
        const outputs = self.shape.rows * self.shape.n;
        const pass = self.rig.begin();
        const e = pass.e;
        // Prepare weight pair buffer on GPU.
        e.setPipeline(self.prep_clear);
        e.setBuffer(self.refs[1].buf, self.refs[1].off + 4 * nw, 0);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
        e.barrier();
        e.setPipeline(self.prep_w);
        e.setBuffer(self.refs[2].buf, self.refs[2].off, 0);
        e.setBuffer(self.refs[3].buf, self.refs[3].off, 1);
        e.setBuffer(self.refs[1].buf, self.refs[1].off, 2);
        e.setValue(shape, 3);
        e.dispatchThreads(mtl.Size.of(@min(nw + self.shape.n, 1 << 18), 1, 1), mtl.Size.of(128, 1, 1));
        e.barrier();
        if (self.kind == .chained) {
            const nx: u32 = @intCast(n);
            e.setPipeline(self.helpers[0]);
            e.setBuffer(self.refs[5].buf, self.refs[5].off, 0);
            e.setValue(nx, 1);
            e.dispatchThreads(mtl.Size.of(8, 1, 1), mtl.Size.of(8, 1, 1));
            e.barrier();
            e.setPipeline(self.helpers[1]);
            e.setBuffer(self.refs[0].buf, self.refs[0].off, 0);
            e.setBuffer(self.refs[5].buf, self.refs[5].off, 1);
            e.setValue(nx, 2);
            e.dispatchThreads(mtl.Size.of(@min((n + 3) / 4, 1 << 18), 1, 1), mtl.Size.of(256, 1, 1));
            e.barrier();
        }
        var refs: [9]Binding = undefined;
        for (self.refs, &refs) |ref, *binding| binding.* = bind(ref, 0);
        if (self.kind == .chained) refs[0] = refs[5];
        if (self.kind == .gelu) refs[4] = refs[8];
        kernel.encode(mtl, e, self.helpers, self.main, refs, shape, self.kind != .chained, self.epilogue());
        if (self.kind == .gelu) {
            const out_n: u32 = @intCast(outputs);
            e.setPipeline(self.helpers[3]);
            e.setBuffer(self.refs[8].buf, self.refs[8].off, 0);
            e.setValue(out_n, 1);
            e.dispatchThreads(mtl.Size.of(outputs, 1, 1), mtl.Size.of(128, 1, 1));
            e.barrier();
            e.setPipeline(self.copy);
            e.setBuffer(self.refs[8].buf, self.refs[8].off + kernel.restoreOffset(outputs), 0);
            e.setBuffer(self.refs[4].buf, self.refs[4].off, 1);
            e.setValue(out_n, 2);
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
            for (kernel.helper_names, &probe.helpers) |name, *pipe| pipe.* = try rig_.pipeline(device, source, name);
            probe.main = try rig_.pipeline(device, source, kernel.mainName(probe.epilogue()));
            probe.prep_clear = try rig_.pipeline(device, source, "sam_norm_wide_clear");
            probe.prep_w = try rig_.pipeline(device, source, "sam_dense_pair_prepare_weights");
            probe.copy = try rig_.pipeline(device, source, "sam3_copy");
            try kernels.append(self.a, probe.contractKernel());
        }
    }

    pub fn deinit(self: *Suite) void {
        for (&self.probes) |*entry| if (entry.*) |*probe| {
            probe.reset();
            probe.rig.deinit();
            for (probe.helpers) |pipe| pipe.deinit();
            probe.main.deinit();
            probe.prep_clear.deinit();
            probe.prep_w.deinit();
            probe.copy.deinit();
        };
    }
};
