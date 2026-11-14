const std = @import("std");
const gpu = @import("tensor_metal");
const kernel = @import("sam3_kernels").dense_expansion;

pub fn run(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize, activate: bool) !gpu.Tensor {
    const buffers = try encode(g, x, weight, bias, rows, cols, depth, activate);
    buffers.components.deinit();
    buffers.residuals.deinit();
    buffers.domain.deinit();
    return buffers.output;
}

const Buffers = struct {
    output: gpu.Tensor,
    components: gpu.Tensor,
    residuals: gpu.Tensor,
    domain: gpu.Tensor,

    fn deinit(self: Buffers) void {
        self.components.deinit();
        self.residuals.deinit();
        self.domain.deinit();
        self.output.deinit();
    }
};

pub const Packing = struct { domain_flags: u32, component_mask: u32, operands: usize, residuals: usize, exact: bool };
const SparseRow = extern struct { count: u32, indices: [4]u32, values: [4]f32 };

pub fn inspect(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize, activate: bool) !Packing {
    @setFloatMode(.strict);
    try g.beginBatch();
    errdefer if (g.batch != null) g.endBatch() catch {};
    const buffers = try encode(g, x, weight, bias, rows, cols, depth, activate);
    defer buffers.deinit();
    try g.endBatch();
    const flags = buffers.domain.slice(u32, 8);
    var result = Packing{ .domain_flags = flags[0], .component_mask = flags[7], .operands = x.count + weight.count, .residuals = 0, .exact = flags[0] == 0 };
    if (!result.exact) return result;
    const residuals = buffers.residuals.slice(SparseRow, rows);
    for (residuals) |row| {
        if (row.count > 4) return error.ExpansionResidualOverflow;
        result.residuals += row.count;
    }
    const components = buffers.components.slice(u16, buffers.components.count);
    for (0..x.count + weight.count) |i| {
        const source = if (i < x.count) x.get(i) else weight.get(i - x.count);
        const base = if (i < x.count) i else 3 * x.count + i - x.count;
        const high: f32 = @floatCast(@as(f16, @bitCast(components[base])));
        const reconstructed = if (i < x.count) value: {
            const middle: f32 = @floatCast(@as(f16, @bitCast(components[base + x.count])));
            const low: f32 = @floatCast(@as(f16, @bitCast(components[base + 2 * x.count])));
            var reconstructed = (high + middle * 0x1p-11) + low * 0x1p-22;
            const row = residuals[i / depth];
            for (row.indices[0..row.count], row.values[0..row.count]) |index, residual| {
                if (index >= depth) return error.ExpansionResidualIndex;
                if (index == i % depth) reconstructed += residual;
            }
            break :value reconstructed;
        } else high;
        if (source != 0 and @as(u32, @bitCast(source)) != @as(u32, @bitCast(reconstructed))) return error.ExpansionOperandsChanged;
    }
    return result;
}

fn encode(g: *gpu.Gpu, x: gpu.Tensor, weight: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize, activate: bool) !Buffers {
    if (rows == 0 or cols == 0 or depth == 0 or cols > 4736 or depth > 4736 or
        x.dtype != .f32 or weight.dtype != .f32 or bias.dtype != .f32 or
        x.count != try std.math.mul(usize, rows, depth) or weight.count != try std.math.mul(usize, cols, depth) or bias.count != cols) return error.SamExpansionProjectionShape;
    const output = try g.allocDevice(try std.math.mul(usize, rows, cols), .f32);
    errdefer output.deinit();
    const count = try std.math.add(usize, try std.math.mul(usize, x.count, 3), weight.count);
    const components = try g.allocDevice(count, .f16);
    errdefer components.deinit();
    const domain = try g.allocDevice(8, .f32);
    errdefer domain.deinit();
    const residuals = try g.allocDevice(try std.math.mul(usize, rows, @sizeOf(SparseRow) / 4), .f32);
    errdefer residuals.deinit();
    const shape = [3]u32{ @intCast(std.math.cast(i32, rows) orelse return error.SamExpansionProjectionShape), @intCast(cols), @intCast(depth) };
    const batched = try g.beginOperator();
    defer g.endOperator(batched);
    if (g.operator_observer != null) g.operator_shape = .{ rows, cols, depth };
    const pool = gpu.mtl.objc.Pool.push();
    defer pool.pop();
    var pipes: [6]gpu.mtl.Pipeline = undefined;
    const names = if (activate) kernel.gelu_names else kernel.names;
    for ([_]usize{ 0, 1, 2, 3, 5, 4 }) |i| pipes[i] = try g.pipeline(names[i]);
    const refs = [_]gpu.Tensor{ x, weight, bias, output, domain, components, residuals };
    if (g.operator_observer != null) {
        inline for (0..6) |phase| {
            const pass = g.command();
            g.operator_name = names[phase];
            kernel.encodePhase(gpu.mtl, pass.enc, pipes, refs, shape, phase);
            try g.finish(pass.cb, pass.enc);
        }
    } else {
        const pass = g.command();
        kernel.encode(gpu.mtl, pass.enc, pipes, refs, shape);
        try g.finish(pass.cb, pass.enc);
    }
    return .{ .output = output, .components = components, .domain = domain, .residuals = residuals };
}
