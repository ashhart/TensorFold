const std = @import("std");
const gpu = @import("tensor_metal");
const objc = gpu.mtl.objc;

pub const Backend = struct {
    library: std.DynLib,
    kernel: objc.Id,

    pub fn init(device: gpu.mtl.Device) !Backend {
        var library = try std.DynLib.open("/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders");
        errdefer library.close();
        const raw = objc.msg(objc.Id, objc.class("MPSNDArrayMatrixMultiplication"), "alloc", .{});
        const kernel = objc.msg(?objc.Id, raw, "initWithDevice:sourceCount:", .{ device.id, @as(usize, 3) }) orelse return error.NoMpsMatrixKernel;
        objc.msg(void, kernel, "setOptions:", .{@as(usize, 0)});
        return .{ .library = library, .kernel = kernel };
    }

    pub fn deinit(self: *Backend) void {
        objc.release(self.kernel);
        self.library.close();
    }

    fn array(tensor: gpu.Tensor, dimensions: []const usize, transpose: bool) !objc.Id {
        const descriptor = objc.msg(objc.Id, objc.class("MPSNDArrayDescriptor"), "descriptorWithDataType:dimensionCount:dimensionSizes:", .{ @as(u32, 0x10000020), dimensions.len, dimensions.ptr });
        objc.msg(void, descriptor, "setPreferPackedRows:", .{true});
        if (transpose) objc.msg(void, descriptor, "transposeDimension:withDimension:", .{ @as(usize, 0), @as(usize, 1) });
        const raw = objc.msg(objc.Id, objc.class("MPSNDArray"), "alloc", .{});
        return objc.msg(?objc.Id, raw, "initWithBuffer:offset:descriptor:", .{ tensor.buffer.id, tensor.offset, descriptor }) orelse error.NoMpsArray;
    }

    pub fn run(self: *Backend, g: *gpu.Gpu, x: gpu.Tensor, w: gpu.Tensor, bias: gpu.Tensor, rows: usize, cols: usize, depth: usize) !gpu.Tensor {
        const pool = objc.Pool.push();
        defer pool.pop();
        const out = try g.allocDevice(rows * cols, .f32);
        errdefer out.deinit();
        const left = try array(x, &.{ depth, rows }, false);
        defer objc.release(left);
        const right = try array(w, &.{ depth, cols }, true);
        defer objc.release(right);
        const addend = try array(bias, &.{cols}, false);
        defer objc.release(addend);
        const destination = try array(out, &.{ cols, rows }, false);
        defer objc.release(destination);
        const inputs = [_]objc.Id{ left, right, addend };
        const arrays = objc.msg(objc.Id, objc.class("NSArray"), "arrayWithObjects:count:", .{ inputs[0..].ptr, inputs.len });
        const pass = g.command();
        pass.enc.barrier();
        objc.msg(void, self.kernel, "encodeToCommandEncoder:commandBuffer:sourceArrays:destinationArray:", .{ pass.enc.id, pass.cb.id, arrays, destination });
        pass.enc.barrier();
        try g.finish(pass.cb, pass.enc);
        return out;
    }
};

pub fn check(init: std.process.Init, g: *gpu.Gpu, path: []const u8) !void {
    var backend = try Backend.init(g.device);
    defer backend.deinit();
    const a = init.arena.allocator();
    const Case = struct { shape: [3]usize, fill: []const u8, comparison: []const u8, incorrect_outputs: usize, changed_repeats: usize, maximum_fp64_error: f64 };
    var cases: std.ArrayList(Case) = .empty;
    var failures: usize = 0;
    for ([_][3]usize{ .{ 1, 1, 1 }, .{ 7, 19, 33 }, .{ 32, 128, 256 }, .{ 65, 129, 257 } }) |shape| {
        const rows, const cols, const depth = shape;
        const x = try g.alloc(rows * depth, .f32);
        defer x.deinit();
        const w = try g.alloc(cols * depth, .f32);
        defer w.deinit();
        const bias = try g.alloc(cols, .f32);
        defer bias.deinit();
        for ([_]enum { finite, nonfinite, cancellation, subnormal }{ .finite, .nonfinite, .cancellation, .subnormal }) |fill| {
            for (x.slice(f32, x.count), 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i % 997)) * 0.037);
            for (w.slice(f32, w.count), 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i % 991)) * 0.017);
            for (bias.slice(f32, bias.count), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 256;
            switch (fill) {
                .finite => {},
                .nonfinite => {
                    for (0..rows) |row| x.slice(f32, x.count)[row * depth] = switch (row % 4) {
                        0 => std.math.nan(f32),
                        1 => std.math.inf(f32),
                        2 => -std.math.inf(f32),
                        else => 1,
                    };
                    for (0..cols) |col| w.slice(f32, w.count)[col * depth] = if (col % 2 == 0) 1 else 0;
                },
                .cancellation => {
                    for (x.slice(f32, x.count), 0..) |*v, i| {
                        const k = i % depth;
                        v.* = if (k < 8) @floatFromInt(1024 + 64 * k) else if (k < 16) @floatFromInt(1025 + 64 * (k - 8)) else 0;
                    }
                    for (w.slice(f32, w.count), 0..) |*v, i| {
                        const k = i % depth;
                        const coefficient: f32 = @floatFromInt(128 + (i / depth + 3 * (k % 8)) % 128);
                        v.* = if (k < 8) coefficient else if (k < 16) -coefficient else 0;
                    }
                    @memset(bias.slice(f32, cols), 0);
                },
                .subnormal => {
                    @memset(x.slice(f32, x.count), @bitCast(@as(u32, 1)));
                    @memset(w.slice(f32, w.count), 0x1p100);
                    @memset(bias.slice(f32, cols), 0);
                },
            }
            const actual = try backend.run(g, x, w, bias, rows, cols, depth);
            defer actual.deinit();
            var changed: usize = 0;
            for (0..5) |_| {
                const repeated = try backend.run(g, x, w, bias, rows, cols, depth);
                defer repeated.deinit();
                if (!std.mem.eql(u8, actual.slice(u8, actual.count * 4), repeated.slice(u8, repeated.count * 4))) changed += 1;
            }
            var wrong: usize = 0;
            var maximum: f64 = 0;
            for (0..rows) |row| for (0..cols) |col| {
                var expected: f64 = bias.get(col);
                var magnitude: f64 = @abs(expected);
                for (0..depth) |k| {
                    const term = @as(f64, x.get(row * depth + k)) * w.get(col * depth + k);
                    expected += term;
                    magnitude += @abs(term);
                }
                const value = actual.get(row * cols + col);
                if (std.math.isNan(expected)) {
                    wrong += @intFromBool(!std.math.isNan(value));
                } else if (std.math.isInf(expected)) {
                    wrong += @intFromBool(expected != value);
                } else {
                    const error_ = @abs(expected - value);
                    maximum = @max(maximum, error_);
                    if (fill == .nonfinite) {
                        wrong += @intFromBool(!std.math.isFinite(value));
                        continue;
                    }
                    const bound = if (fill == .finite) (2 * @as(f64, @floatFromInt(depth)) + 1) * std.math.floatEps(f32) * magnitude else 0;
                    wrong += @intFromBool(!std.math.isFinite(value) or error_ > bound);
                }
            };
            std.debug.print("MPSNDArray {d}x{d}x{d} {s}: {d} incorrect outputs, {d} changed repeats, max FP64 error {e}\n", .{ rows, cols, depth, @tagName(fill), wrong, changed, maximum });
            try cases.append(a, .{ .shape = shape, .fill = @tagName(fill), .comparison = if (fill == .nonfinite) "FP64 output classes" else if (fill == .finite) "derived FP32 bound" else "exact FP64", .incorrect_outputs = wrong, .changed_repeats = changed, .maximum_fp64_error = maximum });
            failures += wrong + changed;
        }
    }
    const json = try std.json.Stringify.valueAlloc(a, .{ .backend = "mps_ndarray", .repeat_runs = 5, .failures = failures, .cases = cases.items }, .{ .whitespace = .indent_2 });
    const receipt = try std.Io.Dir.cwd().createFile(init.io, path, .{ .exclusive = true });
    defer receipt.close(init.io);
    try receipt.writeStreamingAll(init.io, json);
    try receipt.writeStreamingAll(init.io, "\n");
    if (failures != 0) return error.MpsProjectionEdgeCasesFailed;
}
