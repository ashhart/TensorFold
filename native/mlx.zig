//! MLX-C ownership boundary. A Scope owns temporary graph handles; persistent
//! weights/cache entries explicitly retain their own handles with `retain`.
const std = @import("std");
pub const c = @import("mlx_c");
pub const Array = c.mlx_array;
pub const empty: Array = .{ .ctx = null };
pub const bf16 = c.MLX_BFLOAT16;
pub const f32t = c.MLX_FLOAT32;
pub const i32t = c.MLX_INT32;
pub var stream: c.mlx_stream = .{ .ctx = null };
// Replaced only by the single-threaded allocation diagnostic; production uses libc.
pub var allocator: std.mem.Allocator = std.heap.c_allocator;
pub var tensor_units = false;
pub var force_simd = false;

fn onError(msg: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    std.debug.print("MLX: {s}\n", .{msg});
}
pub fn check(rc: c_int) !void {
    if (rc != 0) return error.MlxFailure;
}
pub fn init() !void {
    c.mlx_set_error_handler(onError, null, null);
    const dev = c.mlx_device_new_type(c.MLX_GPU, 0);
    defer _ = c.mlx_device_free(dev);
    var info = c.mlx_device_info_new();
    defer _ = c.mlx_device_info_free(info);
    try check(c.mlx_device_info_get(&info, dev));
    var arch: [*c]const u8 = null;
    try check(c.mlx_device_info_get_string(&arch, info, "architecture"));
    const name = std.mem.span(arch);
    const prefix = "applegpu_g";
    if (std.mem.startsWith(u8, name, prefix)) {
        var end: usize = prefix.len;
        while (end < name.len and std.ascii.isDigit(name[end])) : (end += 1) {}
        const generation = std.fmt.parseInt(u32, name[prefix.len..end], 10) catch 0;
        tensor_units = generation >= 17 and !force_simd;
    }
    std.debug.print("Metal: {s}, {s} backend\n", .{ name, if (tensor_units) "tensor" else "SIMD" });
    try check(c.mlx_get_default_stream(&stream, dev));
    var previous: usize = 0;
    try check(c.mlx_set_cache_limit(&previous, 2 * 1024 * 1024 * 1024));
}
pub fn shutdown() void {
    _ = c.mlx_stream_free(stream);
}
pub fn free(a: Array) void {
    if (a.ctx != null) _ = c.mlx_array_free(a);
}
pub fn retain(a: Array) !Array {
    var out = c.mlx_array_new();
    errdefer free(out);
    try check(c.mlx_array_set(&out, a));
    return out;
}
pub fn replace(dst: *Array, src: Array) !void {
    const a = try retain(src);
    free(dst.*);
    dst.* = a;
}
pub fn dim(a: Array, axis: c_int) c_int {
    return c.mlx_array_dim(a, axis);
}
pub fn dtype(a: Array) c.mlx_dtype {
    return c.mlx_array_dtype(a);
}
pub fn shape(a: Array) []const c_int {
    return c.mlx_array_shape(a)[0..c.mlx_array_ndim(a)];
}
pub fn eval(a: Array) !void {
    try check(c.mlx_array_eval(a));
}
pub fn evalMany(arrays: []const Array, async_: bool) !void {
    const v = c.mlx_vector_array_new_data(arrays.ptr, arrays.len);
    defer _ = c.mlx_vector_array_free(v);
    try check(if (async_) c.mlx_async_eval(v) else c.mlx_eval(v));
}
pub fn opt(n: c_int) c.mlx_optional_int {
    return .{ .value = n, .has_value = true };
}

pub const Scope = struct {
    arrays: std.ArrayList(Array) = .empty,
    pub fn deinit(s: *Scope) void {
        for (s.arrays.items) |a| free(a);
        s.arrays.deinit(allocator);
    }
    pub fn own(s: *Scope, a: Array) !Array {
        errdefer free(a);
        if (a.ctx == null) return error.MlxFailure;
        try s.arrays.append(allocator, a);
        return a;
    }
    pub fn result(s: *Scope, rc: c_int, a: Array) !Array {
        check(rc) catch |err| {
            free(a);
            return err;
        };
        return s.own(a);
    }
    pub fn data(s: *Scope, ptr: anytype, dims: []const c_int, dt: c.mlx_dtype) !Array {
        return s.own(c.mlx_array_new_data(@ptrCast(ptr), dims.ptr, @intCast(dims.len), dt));
    }
    pub fn ints(s: *Scope, values: []const i32) !Array {
        return s.data(values.ptr, &.{@intCast(values.len)}, i32t);
    }
    pub fn scalar(s: *Scope, value: f32) !Array {
        return s.data(&value, &.{1}, f32t);
    }
    pub fn zeros(s: *Scope, dims: []const c_int, dt: c.mlx_dtype) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_zeros(&a, dims.ptr, dims.len, dt, stream);
        return s.result(rc, a);
    }
    pub fn reshape(s: *Scope, x: Array, dims: []const c_int) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_reshape(&a, x, dims.ptr, dims.len, stream);
        return s.result(rc, a);
    }
    pub fn transpose(s: *Scope, x: Array, axes: []const c_int) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_transpose_axes(&a, x, axes.ptr, axes.len, stream);
        return s.result(rc, a);
    }
    pub fn cast(s: *Scope, x: Array, dt: c.mlx_dtype) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_astype(&a, x, dt, stream);
        return s.result(rc, a);
    }
    pub fn contiguous(s: *Scope, x: Array) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_contiguous(&a, x, false, stream);
        return s.result(rc, a);
    }
    pub fn slice(s: *Scope, x: Array, axis: usize, start: c_int, end: c_int) !Array {
        var starts: [8]c_int = @splat(0);
        var stops: [8]c_int = undefined;
        const steps: [8]c_int = @splat(1);
        const dims = shape(x);
        @memcpy(stops[0..dims.len], dims);
        starts[axis] = start;
        stops[axis] = end;
        var a = c.mlx_array_new();
        const rc = c.mlx_slice(&a, x, &starts, dims.len, &stops, dims.len, &steps, dims.len, stream);
        return s.result(rc, a);
    }
    pub fn take(s: *Scope, x: Array, ids: Array, axis: c_int) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_take_axis(&a, x, ids, axis, stream);
        return s.result(rc, a);
    }
    pub fn cat(s: *Scope, xs: []const Array, axis: c_int) !Array {
        const v = c.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = c.mlx_vector_array_free(v);
        var a = c.mlx_array_new();
        const rc = c.mlx_concatenate_axis(&a, v, axis, stream);
        return s.result(rc, a);
    }
    pub fn stack(s: *Scope, xs: []const Array, axis: c_int) !Array {
        const v = c.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = c.mlx_vector_array_free(v);
        var a = c.mlx_array_new();
        const rc = c.mlx_stack_axis(&a, v, axis, stream);
        return s.result(rc, a);
    }
    pub fn unary(s: *Scope, comptime op: anytype, x: Array) !Array {
        var a = c.mlx_array_new();
        const rc = op(&a, x, stream);
        return s.result(rc, a);
    }
    pub fn binary(s: *Scope, comptime op: anytype, x: Array, y: Array) !Array {
        var a = c.mlx_array_new();
        const rc = op(&a, x, y, stream);
        return s.result(rc, a);
    }
    pub fn rms(s: *Scope, x: Array, w: Array) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_fast_rms_norm(&a, x, w, 1e-6, stream);
        return s.result(rc, a);
    }
    pub fn rope(s: *Scope, x: Array, positions: Array, dims: c_int) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_fast_rope_dynamic(&a, x, dims, false, .{ .value = 10000000, .has_value = true }, 1, positions, empty, stream);
        return s.result(rc, a);
    }
    pub fn dequant(s: *Scope, w: Array, scales: Array, biases: Array) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_dequantize(&a, w, scales, biases, opt(64), opt(4), "affine", empty, .{ .value = bf16, .has_value = true }, stream);
        return s.result(rc, a);
    }
    pub fn argmax(s: *Scope, x: Array) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_argmax_axis(&a, x, -1, false, stream);
        return s.result(rc, a);
    }
};

pub const Template = struct { name: [:0]const u8, value: union(enum) { int: c_int, dtype: c.mlx_dtype, boolean: bool } };
pub fn ti(name: [:0]const u8, value: c_int) Template {
    return .{ .name = name, .value = .{ .int = value } };
}
pub fn td(name: [:0]const u8, value: c.mlx_dtype) Template {
    return .{ .name = name, .value = .{ .dtype = value } };
}
pub fn tb(name: [:0]const u8, value: bool) Template {
    return .{ .name = name, .value = .{ .boolean = value } };
}
pub const Output = struct { shape: []const c_int, dtype: c.mlx_dtype = bf16 };
pub const Kernels = struct {
    items: std.StringHashMap(c.mlx_fast_metal_kernel),
    pub fn init() Kernels {
        return .{ .items = std.StringHashMap(c.mlx_fast_metal_kernel).init(allocator) };
    }
    pub fn deinit(k: *Kernels) void {
        var it = k.items.valueIterator();
        while (it.next()) |v| c.mlx_fast_metal_kernel_free(v.*);
        k.items.deinit();
    }
    pub fn run(k: *Kernels, s: *Scope, spec: @import("kernel_sources.zig").Spec, inputs: []const Array, templates: []const Template, grid: [3]c_int, group: [3]c_int, outputs: []const Output) ![5]Array {
        if (outputs.len > 5) return error.TooManyKernelOutputs;
        var result_: [5]Array = @splat(empty);
        try k.runInto(s, spec, inputs, templates, grid, group, outputs, result_[0..outputs.len], null);
        return result_;
    }
    pub fn runInto(k: *Kernels, s: *Scope, spec: @import("kernel_sources.zig").Spec, inputs: []const Array, templates: []const Template, grid: [3]c_int, group: [3]c_int, outputs: []const Output, result_: []Array, init_value: ?f32) !void {
        if (result_.len != outputs.len or outputs.len != spec.outputs.len or inputs.len != spec.inputs.len) return error.InvalidKernelArity;
        const entry = try k.items.getOrPut(spec.name);
        if (!entry.found_existing) {
            errdefer _ = k.items.remove(spec.name);
            const ins = c.mlx_vector_string_new();
            defer _ = c.mlx_vector_string_free(ins);
            const outs = c.mlx_vector_string_new();
            defer _ = c.mlx_vector_string_free(outs);
            for (spec.inputs) |n| try check(c.mlx_vector_string_append_value(ins, n));
            for (spec.outputs) |n| try check(c.mlx_vector_string_append_value(outs, n));
            entry.value_ptr.* = c.mlx_fast_metal_kernel_new(spec.name, ins, outs, spec.source, spec.header, spec.contiguous, false);
            if (entry.value_ptr.ctx == null) return error.MlxFailure;
        }
        const cfg = c.mlx_fast_metal_kernel_config_new();
        if (cfg.ctx == null) return error.MlxFailure;
        defer c.mlx_fast_metal_kernel_config_free(cfg);
        if (init_value) |value| try check(c.mlx_fast_metal_kernel_config_set_init_value(cfg, value));
        try check(c.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]));
        try check(c.mlx_fast_metal_kernel_config_set_thread_group(cfg, group[0], group[1], group[2]));
        for (templates) |t| try check(switch (t.value) {
            .int => |v| c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, v),
            .dtype => |v| c.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, t.name, v),
            .boolean => |v| c.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, t.name, v),
        });
        for (outputs) |o| try check(c.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dtype));
        const ins = c.mlx_vector_array_new_data(inputs.ptr, inputs.len);
        defer _ = c.mlx_vector_array_free(ins);
        var outs = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(outs);
        try check(c.mlx_fast_metal_kernel_apply(&outs, entry.value_ptr.*, ins, cfg, stream));
        for (0..outputs.len) |i| {
            var a = c.mlx_array_new();
            const rc = c.mlx_vector_array_get(&a, outs, i);
            result_[i] = try s.result(rc, a);
        }
    }
};
