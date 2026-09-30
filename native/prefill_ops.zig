//! mlx-lm prefill activations, including BF16 intermediates and FP32 gates.
//! Compile the original operation graphs through MLX-C, just as mlx-lm does.
const mx = @import("mlx.zig");
const c = mx.c;
pub const Kind = enum { silu, swiglu, gated, decay, gelu, gelu_tanh, geglu, softcap, clipped_swiglu, deepseek_head, ssm_dt };
pub const Ops = struct {
    closures: [@typeInfo(Kind).@"enum".field_names.len]c.mlx_closure = @splat(.{ .ctx = null }),
    pub fn deinit(o: *Ops) void {
        for (o.closures) |fun| if (fun.ctx != null) {
            _ = c.mlx_closure_free(fun);
        };
        o.* = .{};
    }
    pub fn call(o: *Ops, s: *mx.Scope, comptime kind: Kind, args: []const mx.Array) !mx.Array {
        const slot = &o.closures[@backingInt(kind)];
        if (slot.ctx == null) {
            const fun = c.mlx_closure_new_func(struct {
                fn apply(out: [*c]c.mlx_vector_array, ins: c.mlx_vector_array) callconv(.c) c_int {
                    return graph(kind, out, ins) catch -1;
                }
            }.apply);
            defer _ = c.mlx_closure_free(fun);
            try mx.check(c.mlx_compile(slot, fun, kind != .deepseek_head));
        }
        var prepared: [6]mx.Array = undefined;
        const call_args = if (kind == .deepseek_head) blk: {
            if (args.len != prepared.len) return error.InvalidGraphInputs;
            @memcpy(&prepared, args);
            // HeadHC converts stored parameters before constructing its compiled call.
            for (1..4) |i| prepared[i] = try s.cast(args[i], mx.f32t);
            break :blk prepared[0..];
        } else args;
        const ins = c.mlx_vector_array_new_data(call_args.ptr, call_args.len);
        defer _ = c.mlx_vector_array_free(ins);
        var outs = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(outs);
        try mx.check(c.mlx_closure_apply(&outs, slot.*, ins));
        var result = c.mlx_array_new();
        const rc = c.mlx_vector_array_get(&result, outs, 0);
        return s.result(rc, result);
    }
};
pub fn uncompiled(s: *mx.Scope, comptime kind: Kind, args: []const mx.Array) !mx.Array {
    const ins = c.mlx_vector_array_new_data(args.ptr, args.len);
    defer _ = c.mlx_vector_array_free(ins);
    var outs = c.mlx_vector_array_new();
    defer _ = c.mlx_vector_array_free(outs);
    try mx.check(try graph(kind, &outs, ins));
    var result = c.mlx_array_new();
    const rc = c.mlx_vector_array_get(&result, outs, 0);
    return s.result(rc, result);
}
fn graph(comptime kind: Kind, out: [*c]c.mlx_vector_array, ins: c.mlx_vector_array) !c_int {
    var s = mx.Scope{};
    defer s.deinit();
    var args: [if (kind == .deepseek_head) 6 else if (kind == .ssm_dt) 4 else if (kind == .decay or kind == .clipped_swiglu) 3 else if (kind == .silu or kind == .gelu or kind == .gelu_tanh) 1 else 2]mx.Array = undefined;
    for (&args, 0..) |*a, i| {
        var x = c.mlx_array_new();
        const rc = c.mlx_vector_array_get(&x, ins, i);
        a.* = try s.result(rc, x);
    }
    if (kind == .ssm_dt) {
        const sum = try s.binary(c.mlx_add, try s.cast(args[0], mx.f32t), args[1]);
        const softplus = try s.binary(c.mlx_logaddexp, sum, try s.scalar(0));
        var result = c.mlx_array_new();
        const rc = c.mlx_clip(&result, softplus, args[2], args[3], mx.stream);
        result = try s.result(rc, result);
        return c.mlx_vector_array_set_data(out, &result, 1);
    }
    if (kind == .deepseek_head) {
        const rows = mx.dim(args[0], 0);
        const dims = mx.dim(args[0], 2);
        const streams = try s.cast(args[0], mx.f32t);
        const xf = try s.reshape(streams, &.{ rows, -1 });
        var mean = c.mlx_array_new();
        const rc = c.mlx_mean_axis(&mean, try s.binary(c.mlx_multiply, xf, xf), -1, true, mx.stream);
        mean = try s.result(rc, mean);
        const inv = try s.unary(c.mlx_rsqrt, try s.binary(c.mlx_add, mean, args[4]));
        const mix = try s.binary(c.mlx_matmul, xf, try s.transpose(try s.cast(args[1], mx.f32t), &.{ 1, 0 }));
        const scale = try s.slice(try s.cast(args[3], mx.f32t), 0, 0, 1);
        const pre = try s.binary(c.mlx_add, try s.unary(c.mlx_sigmoid, try s.binary(c.mlx_add, try s.binary(c.mlx_multiply, try s.binary(c.mlx_multiply, mix, inv), scale), try s.cast(args[2], mx.f32t))), args[5]);
        var y = try s.binary(c.mlx_multiply, try s.slice(pre, 1, 0, 1), try s.reshape(try s.slice(streams, 1, 0, 1), &.{ rows, dims }));
        var j: i32 = 1;
        while (j < 4) : (j += 1) y = try s.binary(c.mlx_add, y, try s.binary(c.mlx_multiply, try s.slice(pre, 1, j, j + 1), try s.reshape(try s.slice(streams, 1, j, j + 1), &.{ rows, dims })));
        const result = try s.cast(y, mx.dtype(args[0]));
        return c.mlx_vector_array_set_data(out, &result, 1);
    }
    if (kind == .clipped_swiglu) {
        const limit = try s.cast(args[2], mx.dtype(args[0]));
        const gate = try s.binary(c.mlx_minimum, args[0], limit);
        const up = try s.binary(c.mlx_minimum, try s.binary(c.mlx_maximum, args[1], try s.unary(c.mlx_negative, limit)), limit);
        const result = try s.binary(c.mlx_multiply, try s.binary(c.mlx_multiply, gate, try s.unary(c.mlx_sigmoid, gate)), up);
        return c.mlx_vector_array_set_data(out, &result, 1);
    }
    if (kind == .softcap) {
        const cap = try s.cast(args[1], mx.dtype(args[0]));
        const result = try s.binary(c.mlx_multiply, try s.unary(c.mlx_tanh, try s.binary(c.mlx_divide, args[0], cap)), cap);
        return c.mlx_vector_array_set_data(out, &result, 1);
    }
    if (kind == .gelu or kind == .gelu_tanh or kind == .geglu) {
        const x = args[0];
        const one = try s.cast(try s.scalar(1), mx.dtype(x));
        const half = try s.cast(try s.scalar(0.5), mx.dtype(x));
        const result = if (kind == .gelu) blk: {
            const root = try s.cast(try s.scalar(1.4142135623730951), mx.dtype(x));
            const erf = try s.unary(c.mlx_erf, try s.binary(c.mlx_divide, x, root));
            break :blk try s.binary(c.mlx_divide, try s.binary(c.mlx_multiply, x, try s.binary(c.mlx_add, one, erf)), try s.cast(try s.scalar(2), mx.dtype(x)));
        } else blk: {
            const power = try s.binary(c.mlx_power, x, try s.cast(try s.scalar(3), mx.dtype(x)));
            const cubic = try s.binary(c.mlx_multiply, try s.cast(try s.scalar(0.044715), mx.dtype(x)), power);
            const scaled = try s.binary(c.mlx_multiply, try s.cast(try s.scalar(0.7978845608028654), mx.dtype(x)), try s.binary(c.mlx_add, x, cubic));
            break :blk try s.binary(c.mlx_multiply, try s.binary(c.mlx_multiply, half, x), try s.binary(c.mlx_add, one, try s.unary(c.mlx_tanh, scaled)));
        };
        const activated = if (kind == .geglu) try s.binary(c.mlx_multiply, result, args[1]) else result;
        return c.mlx_vector_array_set_data(out, &activated, 1);
    }
    if (kind != .decay) {
        const x = if (kind == .gated) try s.cast(args[0], mx.f32t) else args[0];
        const silu = try s.binary(c.mlx_multiply, x, try s.unary(c.mlx_sigmoid, x));
        const result = if (kind == .silu) silu else if (kind == .swiglu)
            try s.binary(c.mlx_multiply, silu, args[1])
        else
            try s.cast(try s.binary(c.mlx_multiply, silu, try s.cast(args[1], mx.f32t)), mx.dtype(args[1]));
        return c.mlx_vector_array_set_data(out, &result, 1);
    }
    const sum = try s.binary(c.mlx_add, args[1], args[2]);
    const zero = try s.cast(try s.scalar(0), mx.dtype(sum));
    const softplus = try s.binary(c.mlx_logaddexp, sum, zero);
    const neg_a = try s.unary(c.mlx_negative, try s.unary(c.mlx_exp, try s.cast(args[0], mx.f32t)));
    const result = try s.unary(c.mlx_exp, try s.binary(c.mlx_multiply, neg_a, softplus));
    return c.mlx_vector_array_set_data(out, &result, 1);
}
