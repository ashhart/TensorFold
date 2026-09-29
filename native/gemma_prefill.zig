const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const gemma = @import("gemma.zig");
const c = mx.c;
const A = mx.Array;

fn linear(s: *mx.Scope, x: A, weights: [3]A, bits: i32) !A {
    var out = c.mlx_array_new();
    const group = @divExact(mx.dim(x, -1), mx.dim(weights[1], -1));
    const rc = c.mlx_quantized_matmul(&out, x, weights[0], weights[1], weights[2], true, mx.opt(group), mx.opt(bits), "affine", mx.stream);
    return s.result(rc, out);
}
fn expert(m: *gemma.Model, s: *mx.Scope, i: usize, name: []const u8, x: A, ids: A, sorted: bool) !A {
    var buf: [160]u8 = undefined;
    const weights = try m.triple(i, try std.fmt.bufPrint(&buf, "experts.switch_glu.{s}", .{name}));
    const group = @divExact(mx.dim(x, -1), mx.dim(weights[1], -1));
    var out = c.mlx_array_new();
    const rc = c.mlx_gather_qmm(&out, x, weights[0], weights[1], weights[2], mx.empty, ids, true, mx.opt(group), mx.opt(4), "affine", sorted, mx.stream);
    return s.result(rc, out);
}
fn moe(m: *gemma.Model, s: *mx.Scope, i: usize, h: A) !A {
    const rows = mx.dim(h, 1);
    const scores = try linear(s, try s.rms(h, try m.weight(i, "router_norm")), try m.triple(i, "router.proj"), 8);
    var partition = c.mlx_array_new();
    const rc = c.mlx_argpartition_axis(&partition, scores, -8, -1, mx.stream);
    partition = try s.result(rc, partition);
    const ids = try s.slice(partition, 2, 120, 128);
    var weights = c.mlx_array_new();
    const take_rc = c.mlx_take_along_axis(&weights, scores, ids, -1, mx.stream);
    weights = try s.result(take_rc, weights);
    var probabilities = c.mlx_array_new();
    const prob_rc = c.mlx_softmax_axis(&probabilities, weights, -1, false, mx.stream);
    weights = try s.binary(c.mlx_multiply, try s.result(prob_rc, probabilities), try s.take(try m.weight(i, "router.per_expert_scale"), ids, 0));
    const normalized = try s.rms(h, try m.weight(i, "pre_feedforward_layernorm_2.weight"));
    const sorted = rows * 8 >= 64;
    var x = try s.reshape(normalized, &.{ 1, rows, 1, 1, 2816 });
    var selected = ids;
    var inverse = mx.empty;
    if (sorted) {
        const flat = try s.reshape(ids, &.{rows * 8});
        const order = try s.unary(c.mlx_argsort, flat);
        inverse = try s.unary(c.mlx_argsort, order);
        const row_ids = try s.binary(c.mlx_floor_divide, order, try s.cast(try s.ints(&.{8}), mx.dtype(order)));
        x = try s.take(try s.reshape(normalized, &.{ rows, 1, 2816 }), row_ids, 0);
        selected = try s.take(flat, order, 0);
    }
    const act = try m.activations.call(s, .geglu, &.{ try expert(m, s, i, "gate_proj", x, selected, sorted), try expert(m, s, i, "up_proj", x, selected, sorted) });
    var y = try expert(m, s, i, "down_proj", act, selected, sorted);
    if (sorted) y = try s.take(y, inverse, 0);
    y = try s.reshape(y, &.{ 1, rows, 8, 2816 });
    const weighted = try s.binary(c.mlx_multiply, y, try s.reshape(weights, &.{ 1, rows, 8, 1 }));
    var out = c.mlx_array_new();
    const sum_rc = c.mlx_sum_axis(&out, weighted, -2, false, mx.stream);
    return s.result(sum_rc, out);
}
fn rope(m: *gemma.Model, s: *mx.Scope, x: A, local: bool) !A {
    var out = c.mlx_array_new();
    const freqs = if (local) mx.empty else try m.weights.get("freq_global");
    const rc = c.mlx_fast_rope(&out, x, if (local) 256 else 512, false, .{ .value = 10000, .has_value = local }, 1, m.position, freqs, mx.stream);
    return s.result(rc, out);
}
fn ordered(s: *mx.Scope, buffer: A, begin: i32, end: i32) !A {
    const slot = @mod(begin, 1152);
    const count = end - begin;
    if (slot + count <= 1152) return s.slice(buffer, 2, slot, slot + count);
    return s.cat(&.{ try s.slice(buffer, 2, slot, 1152), try s.slice(buffer, 2, 0, slot + count - 1152) }, 2);
}
pub fn forward(m: *gemma.Model, tokens: []const i32) !gemma.Pass {
    if (tokens.len == 0 or tokens.len > 2048 or tokens.len > 262144 - m.position) return error.ContextLimitExceeded;
    for (tokens) |id| if (id < 0 or id >= gemma.Model.vocab) return error.InvalidToken;
    var pass = gemma.Pass{ .position = m.position, .generation = m.generation, .rows = tokens.len };
    errdefer pass.deinit();
    const s = &pass.scope;
    const rows: i32 = @intCast(tokens.len);
    var h = try s.reshape(try s.binary(c.mlx_multiply, try m.weights.embed(s, "model.embed_tokens", tokens), try s.cast(try s.scalar(@floatCast(@sqrt(@as(f64, 2816)))), mx.bf16)), &.{ 1, rows, 2816 });
    var masks: [2]A = @splat(mx.empty);
    if (rows > 1 and @min(1023, m.position) + rows > 1024) {
        const earlier = @min(1023, m.position);
        const total = earlier + rows;
        const values = try mx.allocator.alloc(u8, @intCast(rows * total));
        defer mx.allocator.free(values);
        for (0..tokens.len) |r| for (0..@intCast(total)) |j| {
            const query = earlier + @as(i32, @intCast(r));
            const key: i32 = @intCast(j);
            values[r * @as(usize, @intCast(total)) + j] = @intFromBool(key <= query and query - key < 1024);
        };
        masks[0] = try s.data(values.ptr, &.{ rows, total }, c.MLX_BOOL);
    }
    var taps: [32]A = undefined;
    var tap_count: usize = 0;
    for (0..30) |i| {
        const local = i % 6 != 5;
        const heads: i32 = if (local) 8 else 2;
        const dims: i32 = if (local) 256 else 512;
        const x = try s.rms(h, try m.weight(i, "input_layernorm.weight"));
        var q = try s.rms(try s.reshape(try linear(s, x, try m.triple(i, "self_attn.q_proj"), 4), &.{ 1, rows, 16, dims }), try m.weight(i, "self_attn.q_norm.weight"));
        q = try rope(m, s, try s.transpose(q, &.{ 0, 2, 1, 3 }), local);
        const projected = try s.reshape(try linear(s, x, try m.triple(i, "self_attn.k_proj"), 4), &.{ 1, rows, heads, dims });
        const keys = try rope(m, s, try s.transpose(try s.rms(projected, try m.weight(i, "self_attn.k_norm.weight")), &.{ 0, 2, 1, 3 }), local);
        const value = if (local) try s.reshape(try linear(s, x, try m.triple(i, "self_attn.v_proj"), 4), &.{ 1, rows, heads, dims }) else projected;
        const values = try s.transpose(try cp.norm(s, value, mx.empty, 1e-6), &.{ 0, 2, 1, 3 });
        pass.records[i] = .{ .keys = keys, .values = values };
        var all_keys = keys;
        var all_values = values;
        if (m.position > 0) {
            const previous_keys = if (local) try ordered(s, m.cache[i].keys, @max(0, m.position - 1023), m.position) else m.cache[i].keys;
            const previous_values = if (local) try ordered(s, m.cache[i].values, @max(0, m.position - 1023), m.position) else m.cache[i].values;
            all_keys = try s.cat(&.{ previous_keys, keys }, 2);
            all_values = try s.cat(&.{ previous_values, values }, 2);
        }
        const mask = masks[if (local) @as(usize, 0) else 1];
        var out = c.mlx_array_new();
        const rc = c.mlx_fast_scaled_dot_product_attention(&out, q, all_keys, all_values, 1, if (rows > 1 and mask.ctx == null) "causal" else "", mask, mx.empty, false, mx.stream);
        out = try s.result(rc, out);
        out = try linear(s, try s.reshape(try s.transpose(out, &.{ 0, 2, 1, 3 }), &.{ 1, rows, 16 * dims }), try m.triple(i, "self_attn.o_proj"), 4);
        h = try s.binary(c.mlx_add, h, try s.rms(out, try m.weight(i, "post_attention_layernorm.weight")));
        const dense_input = try s.rms(h, try m.weight(i, "pre_feedforward_layernorm.weight"));
        const gate = try linear(s, dense_input, try m.triple(i, "mlp.gate_proj"), 4);
        const up = try linear(s, dense_input, try m.triple(i, "mlp.up_proj"), 4);
        const dense = try linear(s, try m.activations.call(s, .geglu, &.{ gate, up }), try m.triple(i, "mlp.down_proj"), 4);
        const combined = try s.binary(c.mlx_add, try s.rms(dense, try m.weight(i, "post_feedforward_layernorm_1.weight")), try s.rms(try moe(m, s, i, h), try m.weight(i, "post_feedforward_layernorm_2.weight")));
        h = try s.binary(c.mlx_multiply, try s.binary(c.mlx_add, h, try s.rms(combined, try m.weight(i, "post_feedforward_layernorm.weight"))), try m.weight(i, "layer_scalar"));
        if (m.draft) |d| for (d.parsed.value.dflash_config.target_layer_ids) |id| if (id == i) {
            taps[tap_count] = try s.reshape(h, &.{ rows, 2816 });
            tap_count += 1;
        };
        if ((i + 1) % 8 == 0) try mx.evalMany(&.{h}, true);
    }
    pass.hidden = try s.reshape(try s.rms(h, try m.weights.get("model.norm.weight")), &.{ rows, 2816 });
    if (tap_count > 0) pass.taps = try s.cat(taps[0..tap_count], -1);
    pass.logits = try m.activations.call(s, .softcap, &.{ try m.project(s, try s.slice(pass.hidden, 0, rows - 1, rows), try m.weights.triple("model.embed_tokens")), try s.scalar(30) });
    try mx.eval(pass.logits);
    return pass;
}

pub fn check(io: std.Io, dir: []const u8, output: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var m = try gemma.Model.init(io, dir);
    defer m.deinit();
    try std.Io.Dir.cwd().createDirPath(io, output);
    var buf: [256]u8 = undefined;
    for ([_]usize{ 1, 7, 129, 1024, 2048, 3 }, 0..) |count, step| {
        var tokens: [2048]i32 = undefined;
        for (tokens[0..count], 0..) |*id, j| id.* = 1000 + @mod(m.position + @as(i32, @intCast(j)), 37);
        var pass = try m.prefill(tokens[0..count]);
        defer pass.deinit();
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "hidden-{d}", .{step}), pass.hidden);
        try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "logits-{d}", .{step}), pass.logits);
        try m.commit(&pass, count);
        for (m.cache, 0..) |cache, i| {
            try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "keys-{d}-{d}", .{ step, i }), cache.keys);
            try save(&pass.scope, output, try std.fmt.bufPrint(&buf, "values-{d}-{d}", .{ step, i }), cache.values);
        }
        std.debug.print("Gemma prefill verified cache commit at {d} tokens.\n", .{m.position});
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
