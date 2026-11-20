const std = @import("std");
const gpu = @import("tensor_metal");
const planes = @import("dense_planes.zig");
const abi = @import("kernel_abi.zig");
const Tensor = gpu.Tensor;
const Size = gpu.mtl.Size;

const Operands = struct {
    x: Tensor,
    w: Tensor,
    bias: Tensor,
    residual: Tensor,
    rows: usize,
    cols: usize,
    depth: usize,

    fn init(g: *gpu.Gpu, rows: usize, cols: usize, depth: usize, seed: u64, tiny: bool) !Operands {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        const self = Operands{ .x = try g.alloc(rows * depth, .f32), .w = try g.alloc(cols * depth, .f16), .bias = try g.alloc(cols, .f32), .residual = try g.alloc(rows * cols, .f32), .rows = rows, .cols = cols, .depth = depth };
        for (self.x.slice(f32, self.x.count), 0..) |*value, i| {
            const unit = random.float(f32) * 4 - 2;
            // Tiny values exercise the subnormal FP16 planes and the 2^-46 quantum.
            value.* = if (tiny and i % 37 == 0) unit * 0x1p-30 else if (tiny and i % 41 == 0) unit * 0x1p-18 else unit;
        }
        for (self.w.slice(f16, self.w.count)) |*value| value.* = @floatCast(random.float(f32) * 0.25 - 0.125);
        for (self.bias.slice(f32, cols)) |*value| value.* = random.float(f32) - 0.5;
        for (self.residual.slice(f32, self.residual.count)) |*value| value.* = random.float(f32) * 8 - 4;
        return self;
    }

    fn deinit(self: Operands) void {
        for ([_]Tensor{ self.x, self.w, self.bias, self.residual }) |t| t.deinit();
    }

    fn reference(self: Operands, row: usize, col: usize) [2]f64 {
        var sum: f64 = self.bias.get(col);
        var size: f64 = @abs(sum);
        for (0..self.depth) |k| {
            const product = @as(f64, self.x.get(row * self.depth + k)) * self.w.get(col * self.depth + k);
            sum += product;
            size += @abs(product);
        }
        return .{ sum, size };
    }
};

fn project(g: *gpu.Gpu, o: Operands, accumulation: planes.Accumulation, epilogue: planes.Epilogue) !Tensor {
    try g.beginBatch();
    errdefer if (g.batch != null) g.endBatch() catch {};
    const out = try planes.run(g, o.x, o.w, o.bias, o.rows, o.cols, o.depth, accumulation, epilogue);
    try g.endBatch();
    return out;
}

fn restored(planes_buffer: Tensor, n: usize, i: usize) f32 {
    const halves = planes_buffer.slice(f16, planes_buffer.count);
    const high: f32 = halves[i];
    if (!std.math.isFinite(high)) return @as([*]const f32, @ptrCast(@alignCast(halves.ptr)))[(3 * n + 1) / 2 + i];
    return (high + @as(f32, halves[i + n]) * 0x1p-11) + @as(f32, halves[i + 2 * n]) * 0x1p-22;
}

// FP16-plane projections: FP64 bounds on ragged shapes, fallbacks, residual and tile epilogues, GELU planes.
pub fn run(g: *gpu.Gpu) !void {
    const o = try Operands.init(g, 65, 129, 257, 7, true);
    defer o.deinit();
    for ([_]planes.Accumulation{ .plain, .compensated }) |accumulation| {
        const out = try project(g, o, accumulation, .none);
        defer out.deinit();
        var worst: f64 = 0;
        for (0..o.rows) |row| for (0..o.cols) |col| {
            const want = o.reference(row, col);
            // FP32 accumulation over the depth plus the 2^-47 quantum of each operand split.
            const bound = @as(f64, @floatFromInt(o.depth + 2)) * 0x1p-24 * want[1] + @as(f64, @floatFromInt(o.depth)) * 0x1p-47 * 0.125;
            worst = @max(worst, @abs(want[0] - out.get(row * o.cols + col)) / bound);
        };
        std.debug.print("SAM planes projection {s}: worst FP64 error {d:.3} of its bound\n", .{ @tagName(accumulation), worst });
        if (!(worst <= 1)) return error.SamPlanesProjection;
        // The compensated residual kernel compensates 128-deep chunks, so only its FP64 bound is shared.
        const residual = try project(g, o, accumulation, .{ .residual = o.residual });
        defer residual.deinit();
        for (0..o.rows) |row| for (0..o.cols) |col| {
            const i = row * o.cols + col;
            const want = o.reference(row, col);
            const total = want[0] + o.residual.get(i);
            const bound = @as(f64, @floatFromInt(o.depth + 2)) * 0x1p-24 * want[1] + @as(f64, @floatFromInt(o.depth)) * 0x1p-47 * 0.125 + 0x1p-24 * @abs(total);
            if (accumulation == .plain and residual.get(i) != o.residual.get(i) + out.get(i)) return error.SamPlanesResidual;
            if (!(@abs(total - residual.get(i)) <= bound)) return error.SamPlanesResidual;
        };
    }
    try checkFallback(g);
    try checkTiles(g);
    try checkGelu(g, o);
    try checkNorm(g);
    std.debug.print("SAM planes projection: fallbacks, residual and tile epilogues, GELU and normalization planes match\n", .{});
}

fn tiles(g: *gpu.Gpu, src: Tensor, dst: Tensor, channels: u32, direction: u32) !void {
    const params = abi.Tiles{ .channels = channels, .grid = 72, .tile = 24, .direction = direction };
    try g.dispatch(abi.tiles, &.{ src, dst }, params, Size.of(src.count, 1, 1), Size.of(256, 1, 1));
}

// Normalization planes hold exactly the planes of the FP32 normalization, in tile order when tiled.
fn checkNorm(g: *gpu.Gpu) !void {
    const o = try Operands.init(g, 5184, 1024, 1, 5, true);
    defer o.deinit();
    const weight = try g.alloc(1024, .f32);
    defer weight.deinit();
    for (weight.slice(f32, 1024), 0..) |*value, i| value.* = 0.5 + @as(f32, @floatFromInt(i % 13)) / 16;
    const bias = o.residual;
    const norm = @import("normalization.zig");
    for ([_]bool{ false, true }) |tiled| {
        try g.beginBatch();
        const float = try norm.run(g, o.residual, weight, view(bias, 1024), 1024, 1e-5);
        defer float.deinit();
        const ordered = try g.allocDevice(float.count, .f32);
        defer ordered.deinit();
        try tiles(g, float, ordered, 1024, 0);
        const emitted = try norm.runPlanes(g, o.residual, weight, view(bias, 1024), 1024, 1e-5, tiled);
        defer emitted.deinit();
        try g.endBatch();
        const want = if (tiled) ordered else float;
        for (0..float.count) |i| {
            const value = want.get(i);
            const got = restored(emitted, float.count, i);
            if (if (@abs(value) >= 0x1p-22) got != value else !(@abs(got - value) <= 0x1p-47)) return error.SamPlanesNormalization;
        }
    }
}

fn view(t: Tensor, count: usize) Tensor {
    return .{ .buffer = t.buffer, .offset = t.offset, .dtype = t.dtype, .count = count, .owned = false };
}

// An input beyond FP16 and a non-finite weight both take the compensated FP32 fallback through the restore area.
fn checkFallback(g: *gpu.Gpu) !void {
    const o = try Operands.init(g, 33, 70, 65, 3, false);
    defer o.deinit();
    o.x.slice(f32, o.x.count)[3] = 70000;
    const out = try project(g, o, .plain, .none);
    defer out.deinit();
    for (0..o.rows) |row| for (0..o.cols) |col| {
        const want = o.reference(row, col);
        if (!(@abs(want[0] - out.get(row * o.cols + col)) <= 0x1p-23 * want[1])) return error.SamPlanesFallback;
    };
    o.w.slice(f16, o.w.count)[5] = std.math.inf(f16);
    const residual = try project(g, o, .plain, .{ .residual = o.residual });
    defer residual.deinit();
    try g.beginBatch();
    const baseline = try @import("dense_device.zig").run(g, o.x, o.w, o.bias, o.rows, o.cols, o.depth, .nk_half);
    defer baseline.deinit();
    try g.endBatch();
    for (0..out.count) |i| {
        const want = o.residual.get(i) + baseline.get(i);
        if (@as(u32, @bitCast(residual.get(i))) != @as(u32, @bitCast(want))) return error.SamPlanesFallback;
    }
}

fn checkTiles(g: *gpu.Gpu) !void {
    const o = try Operands.init(g, 5184, 64, 64, 11, true);
    defer o.deinit();
    const plain = try project(g, o, .plain, .none);
    defer plain.deinit();
    const out = try project(g, o, .plain, .{ .tile_residual = o.residual });
    defer out.deinit();
    const expected = try g.allocDevice(plain.count, .f32);
    defer expected.deinit();
    try g.beginBatch();
    try tiles(g, plain, expected, 64, 1);
    try g.endBatch();
    for (0..out.count) |i| if (out.get(i) != o.residual.get(i) + expected.get(i)) return error.SamPlanesTiles;
}

fn checkGelu(g: *gpu.Gpu, o: Operands) !void {
    const linear = try project(g, o, .compensated, .none);
    defer linear.deinit();
    const activated = try g.allocDevice(linear.count, .f32);
    defer activated.deinit();
    try g.dispatch("im_sam_gelu", &.{ linear, activated }, @as(u32, @intCast(linear.count)), Size.of(linear.count, 1, 1), Size.of(256, 1, 1));
    const emitted = try project(g, o, .compensated, .gelu_planes);
    defer emitted.deinit();
    for (0..linear.count) |i| {
        const want = activated.get(i);
        const got = restored(emitted, linear.count, i);
        if (if (@abs(want) >= 0x1p-22) got != want else !(@abs(got - want) <= 0x1p-47)) return error.SamPlanesGelu;
    }
}
