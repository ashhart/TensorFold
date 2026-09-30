//! mlx-lm's long-prompt Nemotron-H path, sharing caches with the fused lanes.
const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const nemotron = @import("nemotron.zig");
const c = mx.c;
const A = mx.Array;

pub const Route = struct {
    closure: c.mlx_closure = .{ .ctx = null },
    pub fn deinit(r: *Route) void {
        if (r.closure.ctx != null) _ = c.mlx_closure_free(r.closure);
        r.* = .{};
    }
    fn graph(out: [*c]c.mlx_vector_array, inputs: c.mlx_vector_array) callconv(.c) c_int {
        return build(out, inputs) catch -1;
    }
    fn build(out: [*c]c.mlx_vector_array, inputs: c.mlx_vector_array) !c_int {
        var s = mx.Scope{};
        defer s.deinit();
        var args: [2]A = undefined;
        for (&args, 0..) |*a, i| {
            var v = c.mlx_array_new();
            const rc = c.mlx_vector_array_get(&v, inputs, i);
            a.* = try s.result(rc, v);
        }
        const scores = try s.unary(c.mlx_sigmoid, try s.cast(args[0], mx.f32t));
        const biased = try s.binary(c.mlx_add, scores, args[1]);
        var indices = c.mlx_array_new();
        const rc = c.mlx_argpartition_axis(&indices, try s.unary(c.mlx_negative, biased), 5, -1, mx.stream);
        indices = try s.slice(try s.result(rc, indices), 2, 0, 6);
        var chosen = c.mlx_array_new();
        const take_rc = c.mlx_take_along_axis(&chosen, scores, indices, -1, mx.stream);
        chosen = try s.result(take_rc, chosen);
        var total = c.mlx_array_new();
        const sum_rc = c.mlx_sum_axis(&total, chosen, -1, true, mx.stream);
        total = try s.result(sum_rc, total);
        const weights = try s.binary(c.mlx_multiply, try s.binary(c.mlx_divide, chosen, try s.binary(c.mlx_add, total, try s.scalar(1e-20))), try s.scalar(2.5));
        return c.mlx_vector_array_set_data(out, &[_]A{ indices, weights }, 2);
    }
    fn call(r: *Route, s: *mx.Scope, logits: A, bias: A) ![2]A {
        if (r.closure.ctx == null) {
            const fun = c.mlx_closure_new_func(graph);
            defer _ = c.mlx_closure_free(fun);
            try mx.check(c.mlx_compile(&r.closure, fun, false));
        }
        const inputs = c.mlx_vector_array_new_data(&[_]A{ logits, bias }, 2);
        defer _ = c.mlx_vector_array_free(inputs);
        var outputs = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(outputs);
        try mx.check(c.mlx_closure_apply(&outputs, r.closure, inputs));
        var result: [2]A = undefined;
        for (&result, 0..) |*v, i| {
            var a = c.mlx_array_new();
            const rc = c.mlx_vector_array_get(&a, outputs, i);
            v.* = try s.result(rc, a);
        }
        return result;
    }
};
fn linear(m: *nemotron.Model, s: *mx.Scope, base: []const u8, name: []const u8, x: A) !A {
    var buf: [256]u8 = undefined;
    const rows = mx.dim(x, 1);
    const key = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ base, name });
    if (!mx.tensor_units and (std.mem.eql(u8, name, "q_proj") or std.mem.eql(u8, name, "k_proj") or std.mem.eql(u8, name, "v_proj"))) {
        // SIMD installs row kernels on fused QKV; the separate prefill projections stay MLX.
        return m.weights.linear(&m.kernels, s, key, x, false);
    }
    if (!mx.tensor_units and rows <= 128) {
        var parts: [8]A = undefined;
        var count: usize = 0;
        var start: i32 = 0;
        while (start < rows) : (start += 16) {
            parts[count] = try m.weights.linear(&m.kernels, s, key, try s.slice(x, 1, start, @min(start + 16, rows)), true);
            count += 1;
        }
        const joined = try s.cat(parts[0..count], 0);
        return s.reshape(joined, &.{ 1, rows, mx.dim(joined, -1) });
    }
    const out = try m.weights.linear(&m.kernels, s, key, x, rows <= 128);
    return s.reshape(out, &.{ 1, rows, mx.dim(out, -1) });
}
fn expert(m: *nemotron.Model, s: *mx.Scope, base: []const u8, name: []const u8, x: A, indices: A) !A {
    var buf: [256]u8 = undefined;
    const key = try std.fmt.bufPrint(&buf, "{s}.switch_mlp.{s}", .{ base, name });
    const weights = try m.weights.triple(key);
    const format = (try m.weights.format(key)).?;
    var out = c.mlx_array_new();
    const rc = c.mlx_gather_qmm(&out, x, weights[0], weights[1], weights[2], mx.empty, indices, true, mx.opt(format.group_size), mx.opt(format.bits), "affine", true, mx.stream);
    return s.result(rc, out);
}
fn relu2(s: *mx.Scope, x: A) !A {
    const relu = try s.binary(c.mlx_maximum, x, try s.cast(try s.scalar(0), mx.dtype(x)));
    return s.binary(c.mlx_multiply, relu, relu);
}
fn moe(m: *nemotron.Model, s: *mx.Scope, base: []const u8, x: A) !A {
    const rows = mx.dim(x, 1);
    const logits = try s.binary(c.mlx_matmul, x, try s.transpose(try m.weights.field(base, "gate.weight"), &.{ 1, 0 }));
    const selected = try m.prefill_route.call(s, logits, try m.weights.field(base, "gate.e_score_correction_bias"));
    const flat = try s.reshape(selected[0], &.{rows * 6});
    const order = try s.unary(c.mlx_argsort, flat);
    const inverse = try s.unary(c.mlx_argsort, order);
    const row_ids = try s.binary(c.mlx_floor_divide, order, try s.cast(try s.ints(&.{6}), mx.dtype(order)));
    const input = try s.take(try s.reshape(x, &.{ rows, 1, 2688 }), row_ids, 0);
    const ids = try s.take(flat, order, 0);
    const activated = try relu2(s, try expert(m, s, base, "fc1", input, ids));
    const routed = try s.reshape(try s.take(try expert(m, s, base, "fc2", activated, ids), inverse, 0), &.{ 1, rows, 6, 2688 });
    const weighted = try s.binary(c.mlx_multiply, routed, try s.reshape(selected[1], &.{ 1, rows, 6, 1 }));
    var reduced = c.mlx_array_new();
    const rc = c.mlx_sum_axis(&reduced, weighted, -2, false, mx.stream);
    reduced = try s.cast(try s.result(rc, reduced), mx.dtype(routed));
    const shared = try linear(m, s, base, "shared_experts.down_proj", try relu2(s, try linear(m, s, base, "shared_experts.up_proj", x)));
    return s.binary(c.mlx_add, reduced, shared);
}
fn mamba(m: *nemotron.Model, s: *mx.Scope, base: []const u8, x: A, index: usize, record: *@import("model.zig").Cache) !A {
    const rows = mx.dim(x, 1);
    const projected = try linear(m, s, base, "in_proj", x);
    const gate = try s.slice(projected, 2, 0, 4096);
    const input = try s.slice(projected, 2, 4096, 10240);
    const dt = try s.slice(projected, 2, 10240, 10304);
    const old = m.cache[index];
    const previous = if (old.a.ctx != null) try s.reshape(old.a, &.{ 1, 3, 6144 }) else try s.zeros(&.{ 1, 3, 6144 }, mx.dtype(input));
    const padded = try s.cat(&.{ previous, input }, 1);
    record.a = try s.slice(padded, 1, rows, rows + 3);
    // The fused lane loader transposes the same BF16 convolution weights to FP32.
    const weights = try s.reshape(try s.cast(try s.transpose(try m.weights.field(base, "conv1d.weight"), &.{ 1, 0 }), mx.dtype(input)), &.{ 6144, 4, 1 });
    var convolved = c.mlx_array_new();
    const rc = c.mlx_conv1d(&convolved, padded, weights, 1, 0, 1, 6144, mx.stream);
    convolved = try s.binary(c.mlx_add, try s.result(rc, convolved), try m.weights.field(base, "conv1d.bias"));
    convolved = try m.prefill_ops.call(s, .silu, &.{convolved});
    const result = try @import("ssm_prefill.zig").forward(&m.prefill_ops, s, try s.reshape(try s.slice(convolved, 2, 0, 4096), &.{ 1, rows, 64, 64 }), try m.weights.field(base, "A_log"), try s.reshape(try s.slice(convolved, 2, 4096, 5120), &.{ 1, rows, 8, 128 }), try s.reshape(try s.slice(convolved, 2, 5120, 6144), &.{ 1, rows, 8, 128 }), try s.cast(try m.weights.field(base, "D"), mx.dtype(input)), dt, try m.weights.field(base, "dt_bias"), if (old.b.ctx != null) try s.reshape(old.b, &.{ 1, 64, 64, 128 }) else mx.empty, .{ 0, std.math.inf(f32) });
    record.b = result[1];
    const gated = try m.prefill_ops.call(s, .swiglu, &.{ gate, try s.reshape(result[0], &.{ 1, rows, 4096 }) });
    const normed = try s.reshape(try cp.norm(s, try s.reshape(gated, &.{ 1, rows, 8, 512 }), mx.empty, 1e-5), &.{ 1, rows, 4096 });
    return linear(m, s, base, "out_proj", try s.binary(c.mlx_multiply, try m.weights.field(base, "norm.weight"), normed));
}
fn attention(m: *nemotron.Model, s: *mx.Scope, base: []const u8, x: A, index: usize, record: *@import("model.zig").Cache) !A {
    const rows = mx.dim(x, 1);
    const q = try s.transpose(try s.reshape(try linear(m, s, base, "q_proj", x), &.{ 1, rows, 32, 128 }), &.{ 0, 2, 1, 3 });
    var keys = try s.transpose(try s.reshape(try linear(m, s, base, "k_proj", x), &.{ 1, rows, 2, 128 }), &.{ 0, 2, 1, 3 });
    var values = try s.transpose(try s.reshape(try linear(m, s, base, "v_proj", x), &.{ 1, rows, 2, 128 }), &.{ 0, 2, 1, 3 });
    if (m.cache[index].a.ctx != null) {
        keys = try s.cat(&.{ m.cache[index].a, keys }, 2);
        values = try s.cat(&.{ m.cache[index].b, values }, 2);
    }
    record.* = .{ .a = keys, .b = values };
    var attended = c.mlx_array_new();
    const rc = c.mlx_fast_scaled_dot_product_attention(&attended, q, keys, values, 0.08838834764831845, "causal", mx.empty, mx.empty, false, mx.stream);
    attended = try s.result(rc, attended);
    return linear(m, s, base, "o_proj", try s.reshape(try s.transpose(attended, &.{ 0, 2, 1, 3 }), &.{ 1, rows, 4096 }));
}
pub fn forward(m: *nemotron.Model, tokens: []const i32) !nemotron.Pass {
    if (tokens.len < 17 or tokens.len > 2048 or tokens.len > 262144 - m.position) return error.ContextLimitExceeded;
    for (tokens) |token| if (token < 0 or token >= nemotron.Model.vocab) return error.InvalidToken;
    var pass = nemotron.Pass{ .prefilled = true };
    errdefer pass.deinit();
    const s = &pass.scope;
    const rows: i32 = @intCast(tokens.len);
    var h = try s.reshape(try m.weights.embed(s, "backbone.embeddings", tokens), &.{ 1, rows, 2688 });
    var buf: [256]u8 = undefined;
    for (m.kinds, 0..) |kind, index| {
        const weight = try m.weights.get(try std.fmt.bufPrint(&buf, "backbone.layers.{d}.norm.weight", .{index}));
        const x = try cp.norm(s, h, weight, 1e-5);
        const base = try std.fmt.bufPrint(&buf, "backbone.layers.{d}.mixer", .{index});
        const branch = switch (kind) {
            'M' => try mamba(m, s, base, x, index, &pass.records[index]),
            '*' => try attention(m, s, base, x, index, &pass.records[index]),
            'E' => try moe(m, s, base, x),
            else => return error.InvalidLayerKind,
        };
        h = try s.binary(c.mlx_add, h, branch);
        if ((index + 1) % 4 == 0) try mx.eval(h);
    }
    pass.hidden = try s.reshape(try cp.norm(s, h, try m.weights.get("backbone.norm_f.weight"), 1e-5), &.{ rows, 2688 });
    pass.logits = try m.weights.linear(&m.kernels, s, "lm_head", try s.slice(pass.hidden, 0, rows - 1, rows), true);
    try mx.eval(pass.logits);
    return pass;
}

pub fn check(io: std.Io, dir: []const u8, output: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var m = try nemotron.Model.init(io, dir, false);
    defer m.deinit();
    try std.Io.Dir.cwd().createDirPath(io, output);
    var buf: [256]u8 = undefined;
    for ([_]usize{ 17, 255, 16, 257, 2048, 1 }, 0..) |count, step| {
        var tokens: [2048]i32 = undefined;
        for (tokens[0..count], 0..) |*id, j| id.* = 1000 + @mod(m.position + @as(i32, @intCast(j)), 37);
        var pass = try m.prefill(tokens[0..count]);
        defer pass.deinit();
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "hidden-{d}", .{step}), pass.hidden);
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "logits-{d}", .{step}), try pass.scope.slice(pass.logits, 0, mx.dim(pass.logits, 0) - 1, mx.dim(pass.logits, 0)));
        try m.commit(&pass, count);
        for (m.cache, m.kinds, 0..) |cache, kind, i| {
            if (kind == 'E') continue;
            try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "cache-{d}-{d}-0", .{ step, i }), cache.a);
            try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "cache-{d}-{d}-1", .{ step, i }), cache.b);
        }
        std.debug.print("Nemotron prefill cache commit at {d} tokens.\n", .{m.position});
    }
    for (0..4) |step| {
        var pass = try m.forward(&.{@as(i32, @intCast(step)) + 2000});
        defer pass.deinit();
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "continuation-{d}", .{step}), pass.logits);
        try m.commit(&pass, 1);
    }
}
fn save(s: *mx.Scope, dir: []const u8, name: []const u8, value: A) !void {
    const path = try std.fmt.allocPrintSentinel(mx.allocator, "{s}/{s}.npy", .{ dir, name }, 0);
    defer mx.allocator.free(path);
    const out = try s.cast(value, mx.f32t);
    try mx.eval(out);
    try mx.check(c.mlx_save(path, out));
}
