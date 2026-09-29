//! mlx-lm prefill activations, including BF16 intermediates and FP32 gates.
//! Compile the original operation graphs through MLX-C, just as mlx-lm does.
const mx = @import("mlx.zig");
const c = mx.c;
pub const Kind = enum { silu, swiglu, gated, decay, gelu, gelu_tanh };
pub const Ops = struct {
    closures: [6]c.mlx_closure = @splat(.{ .ctx = null }),
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
            try mx.check(c.mlx_compile(slot, fun, true));
        }
        const ins = c.mlx_vector_array_new_data(args.ptr, args.len);
        defer _ = c.mlx_vector_array_free(ins);
        var outs = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(outs);
        try mx.check(c.mlx_closure_apply(&outs, slot.*, ins));
        var result = c.mlx_array_new();
        const rc = c.mlx_vector_array_get(&result, outs, 0);
        return s.result(rc, result);
    }
};
fn graph(comptime kind: Kind, out: [*c]c.mlx_vector_array, ins: c.mlx_vector_array) !c_int {
    var s = mx.Scope{};
    defer s.deinit();
    var args: [if (kind == .decay) 3 else if (kind == .silu or kind == .gelu or kind == .gelu_tanh) 1 else 2]mx.Array = undefined;
    for (&args, 0..) |*a, i| {
        var x = c.mlx_array_new();
        const rc = c.mlx_vector_array_get(&x, ins, i);
        a.* = try s.result(rc, x);
    }
    if (kind == .gelu or kind == .gelu_tanh) {
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
        return c.mlx_vector_array_set_data(out, &result, 1);
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
