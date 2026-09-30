const std = @import("std");
const mx = @import("mlx.zig");
const c = mx.c;
const A = mx.Array;
const src = @import("kernel_sources.zig");
const Weight = @import("flash_ops.zig").Weight;
const mm = @import("flash_prefill_ops.zig").matmul;
const Ops = @import("prefill_ops.zig").Ops;

pub const Config = struct {
    heads: i32,
    kv_heads: i32,
    dims: i32,
    rotary_dims: i32,
    index_heads: i32,
    index_dims: i32,
    top: i32,
    base: f32 = 10000000,
    epsilon: f32 = 1e-6,
    kernel_select: bool = true,
    heads_per_simdgroup: i32 = 0,
};
pub const Weights = struct { q: Weight, k: Weight, v: Weight, out: Weight, index: Weight, q_scale: A, k_scale: A, iq_scale: A, ik_scale: A };
pub const Cache = struct { keys: A = mx.empty, values: A = mx.empty, raw: A = mx.empty, pooled: A = mx.empty, offset: i32 = 0 };
pub const Result = struct { output: A, cache: Cache, queries: A, index_queries: A, attended: A };

pub fn norm(s: *mx.Scope, x: A, scale: A, epsilon: f32) !A {
    const y = try s.cast(x, mx.f32t);
    var mean = c.mlx_array_new();
    const rc = c.mlx_mean_axis(&mean, try s.unary(c.mlx_square, y), -1, true, mx.stream);
    const inv = try s.unary(c.mlx_rsqrt, try s.binary(c.mlx_add, try s.result(rc, mean), try s.scalar(epsilon)));
    return s.cast(try s.binary(c.mlx_multiply, try s.binary(c.mlx_multiply, y, inv), scale), mx.dtype(x));
}
fn rope(s: *mx.Scope, x: A, cfg: Config, offset: i32, scale: f32) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_fast_rope(&out, x, cfg.rotary_dims, false, .{ .has_value = true, .value = cfg.base }, scale, offset, mx.empty, mx.stream);
    return s.result(rc, out);
}
fn range(s: *mx.Scope, begin: i32, end: i32) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_arange(&out, @floatFromInt(begin), @floatFromInt(end), 1, mx.i32t, mx.stream);
    return s.result(rc, out);
}
fn where(s: *mx.Scope, cond: A, yes: A, no: A) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_where(&out, cond, yes, no, mx.stream);
    return s.result(rc, out);
}
fn pool(s: *mx.Scope, cache: *Cache, blocks: i32, w: Weights, cfg: Config) !A {
    const done = if (cache.pooled.ctx != null) mx.dim(cache.pooled, 1) else 0;
    if (blocks > done) {
        const raw = try s.reshape(try s.slice(cache.raw, 1, done * 4, blocks * 4), &.{ 1, blocks - done, 4, cfg.index_dims });
        var mean = c.mlx_array_new();
        const rc = c.mlx_mean_axis(&mean, try s.cast(raw, mx.f32t), -2, false, mx.stream);
        const pooled = try norm(s, try s.cast(try s.result(rc, mean), mx.bf16), w.ik_scale, cfg.epsilon);
        const fresh = try s.reshape(try rope(s, try s.reshape(pooled, &.{ 1, 1, blocks - done, cfg.index_dims }), cfg, done, 4), &.{ 1, blocks - done, cfg.index_dims });
        cache.pooled = if (done == 0) fresh else try s.cat(&.{ cache.pooled, fresh }, 1);
    }
    return s.slice(cache.pooled, 1, 0, blocks);
}
fn rotated(s: *mx.Scope, query: A, w: Weights, cfg: Config, past: i32) !A {
    return rope(s, try s.transpose(try norm(s, query, w.iq_scale, cfg.epsilon), &.{ 0, 2, 1, 3 }), cfg, past, 1);
}
fn scores(ops: *Ops, s: *mx.Scope, query: A, cache: *Cache, w: Weights, cfg: Config, past: i32) !A {
    const rows = mx.dim(query, 1);
    const blocks = @divTrunc(past + rows, 4);
    const pooled = try s.transpose(try s.reshape(try s.cast(try pool(s, cache, blocks, w, cfg), mx.f32t), &.{ blocks, cfg.index_dims }), &.{ 1, 0 });
    const q = try s.reshape(try s.cast(try rotated(s, query, w, cfg, past), mx.f32t), &.{ cfg.index_heads, rows, cfg.index_dims });
    const root = try s.scalar(@sqrt(@as(f32, @floatFromInt(cfg.index_dims))));
    if (rows == 1) {
        var total = mx.empty;
        var head: i32 = 0;
        while (head < cfg.index_heads) : (head += 1) {
            const value = try s.binary(c.mlx_maximum, try s.binary(c.mlx_matmul, try s.reshape(try s.slice(q, 0, head, head + 1), &.{ 1, cfg.index_dims }), pooled), try s.scalar(0));
            total = if (head == 0) value else try s.binary(c.mlx_add, total, value);
        }
        return s.binary(c.mlx_divide, total, root);
    }
    return ops.call(s, .flash_index_sum, &.{ try s.binary(c.mlx_matmul, q, pooled), root });
}
fn mask(s: *mx.Scope, query: A, cache: *Cache, w: Weights, cfg: Config, past: i32) !A {
    const rows = mx.dim(query, 1);
    const keys = past + rows;
    const blocks = @divTrunc(keys, 4);
    if (blocks <= cfg.top) return mx.empty;
    const pooled = try s.transpose(try s.reshape(try s.cast(try pool(s, cache, blocks, w, cfg), mx.f32t), &.{ 1, 1, blocks, cfg.index_dims }), &.{ 0, 1, 3, 2 });
    const q = try s.cast(try rotated(s, query, w, cfg, past), mx.f32t);
    var summed = c.mlx_array_new();
    const rc = c.mlx_sum_axis(&summed, try s.binary(c.mlx_maximum, try s.binary(c.mlx_matmul, q, pooled), try s.scalar(0)), 1, false, mx.stream);
    const sc = try s.binary(c.mlx_divide, try s.result(rc, summed), try s.scalar(@sqrt(@as(f32, @floatFromInt(cfg.index_dims)))));
    const ends = try s.reshape(try range(s, past + 1, keys + 1), &.{ 1, rows, 1 });
    const complete = try s.binary(c.mlx_floor_divide, ends, try s.ints(&.{4}));
    const valid = try s.binary(c.mlx_less, try s.reshape(try range(s, 0, blocks), &.{ 1, 1, blocks }), complete);
    var chosen = c.mlx_array_new();
    const cr = c.mlx_argpartition_axis(&chosen, try where(s, valid, sc, try s.scalar(-std.math.inf(f32))), -cfg.top, -1, mx.stream);
    chosen = try s.slice(try s.result(cr, chosen), 2, blocks - cfg.top, blocks);
    var hits = c.mlx_array_new();
    const hr = c.mlx_put_along_axis(&hits, try s.zeros(&.{ 1, rows, blocks }, c.MLX_BOOL), chosen, try s.cast(try s.ints(&.{1}), c.MLX_BOOL), -1, mx.stream);
    hits = try s.result(hr, hits);
    var picked = c.mlx_array_new();
    const rr = c.mlx_repeat_axis(&picked, hits, 4, -1, mx.stream);
    picked = try s.result(rr, picked);
    if (blocks * 4 < keys) picked = try s.cat(&.{ picked, try s.zeros(&.{ 1, rows, keys - blocks * 4 }, c.MLX_BOOL) }, -1);
    const index = try s.reshape(try range(s, 0, keys), &.{ 1, 1, keys });
    const causal = try s.binary(c.mlx_less, index, ends);
    const tail = try s.binary(c.mlx_logical_and, try s.binary(c.mlx_greater_equal, index, try s.binary(c.mlx_multiply, complete, try s.ints(&.{4}))), causal);
    const sparse = try s.binary(c.mlx_greater, complete, try s.ints(&.{cfg.top}));
    return s.reshape(try where(s, sparse, try s.binary(c.mlx_logical_or, picked, tail), causal), &.{ 1, 1, rows, keys });
}
fn selected(kernels: *mx.Kernels, ops: *Ops, s: *mx.Scope, queries: A, iq: A, cache: *Cache, w: Weights, cfg: Config, past: i32) !A {
    const rows = mx.dim(queries, 2);
    var ends: [2048]i32 = undefined;
    var complete: [2048]i32 = undefined;
    var sparse: [2048]i32 = undefined;
    var counts: [2048]i32 = undefined;
    const n: usize = @intCast(rows);
    for (0..n) |i| {
        ends[i] = past + @as(i32, @intCast(i)) + 1;
        complete[i] = @divTrunc(ends[i], 4);
        sparse[i] = @intFromBool(complete[i] > cfg.top);
        counts[i] = if (sparse[i] != 0) 4 * cfg.top + @mod(ends[i], 4) else ends[i];
    }
    const sc = try scores(ops, s, iq, cache, w, cfg, past);
    const ids = (try kernels.run(s, src.q4_idx_select, &.{ sc, try s.ints(complete[0..n]), try s.ints(ends[0..n]) }, &.{ mx.ti("TOP", cfg.top), mx.ti("KW", 4 * cfg.top + 3) }, .{ 1024 * rows, 1, 1 }, .{ 1024, 1, 1 }, &.{.{ .shape = &.{ rows, 4 * cfg.top + 3 }, .dtype = mx.i32t }}))[0];
    const q = try s.reshape(try s.transpose(queries, &.{ 0, 2, 1, 3 }), &.{ rows, cfg.heads, cfg.dims });
    const group = @divExact(cfg.heads, cfg.kv_heads);
    const gqa = cfg.dims == 256 and group <= 32;
    const preferred = if (cfg.heads_per_simdgroup != 0) cfg.heads_per_simdgroup else if (mx.tensor_units) @as(i32, 1) else 2;
    const hs: i32 = if (preferred == 2 and @mod(group, 2) == 0) 2 else 1;
    const params = [_]mx.Template{ mx.ti("H", cfg.heads), mx.ti("KVH", cfg.kv_heads), mx.ti("D", cfg.dims), mx.ti("P", 4), mx.ti("TK", 16), mx.ti("HS", hs) };
    const partial = try kernels.run(s, if (gqa) src.flash_prefill_gqa else src.q4_attn_parts, &.{ q, cache.keys, cache.values, ids, try s.ints(counts[0..n]), try s.ints(sparse[0..n]), try s.scalar(1 / @sqrt(@as(f32, @floatFromInt(cfg.dims)))) }, params[0..if (gqa) @as(usize, 6) else 4], .{ if (gqa) 32 * @divExact(group, hs) * cfg.kv_heads else 256 * cfg.heads, rows, 4 }, .{ if (gqa) 32 * @divExact(group, hs) else 256, 1, 1 }, &.{ .{ .shape = &.{ rows, cfg.heads, 4, cfg.dims }, .dtype = mx.f32t }, .{ .shape = &.{ rows, cfg.heads, 4, 2 }, .dtype = mx.f32t } });
    const out = (try kernels.run(s, src.q4_attn_merge, &.{ partial[0], partial[1] }, &.{ mx.ti("H", cfg.heads), mx.ti("D", cfg.dims), mx.ti("P", 4) }, .{ cfg.dims, cfg.heads, rows }, .{ cfg.dims, 1, 1 }, &.{.{ .shape = &.{ rows, cfg.heads, cfg.dims } }}))[0];
    return s.reshape(out, &.{ 1, rows, cfg.heads * cfg.dims });
}

pub fn forward(kernels: *mx.Kernels, ops: *Ops, s: *mx.Scope, x: A, w: Weights, cfg: Config, previous: Cache) !Result {
    if (mx.shape(x).len != 3 or mx.dim(x, 0) != 1 or mx.dtype(x) != mx.bf16 or cfg.heads < 1 or cfg.kv_heads < 1 or @mod(cfg.heads, cfg.kv_heads) != 0 or cfg.dims < 32 or cfg.dims > 1024 or @mod(cfg.dims, 32) != 0 or cfg.index_heads < 1 or cfg.index_dims < cfg.rotary_dims or cfg.top < 1 or cfg.rotary_dims < 0 or @mod(cfg.rotary_dims, 2) != 0 or cfg.rotary_dims > cfg.dims) return error.InvalidTensorShape;
    const rows = mx.dim(x, 1);
    if (rows < 1 or rows > 2048 or previous.offset < 0 or previous.offset > 262144 - rows) return error.InvalidTensorShape;
    const past = previous.offset;
    const end = past + rows;
    const qg = try s.reshape(try mm(s, x, w.q), &.{ 1, rows, cfg.heads, 2 * cfg.dims });
    const queries = try rope(s, try s.transpose(try norm(s, try s.slice(qg, 3, 0, cfg.dims), w.q_scale, cfg.epsilon), &.{ 0, 2, 1, 3 }), cfg, past, 1);
    const gate = try s.reshape(try s.slice(qg, 3, cfg.dims, cfg.dims * 2), &.{ 1, rows, cfg.heads * cfg.dims });
    var keys = try rope(s, try s.transpose(try norm(s, try s.reshape(try mm(s, x, w.k), &.{ 1, rows, cfg.kv_heads, cfg.dims }), w.k_scale, cfg.epsilon), &.{ 0, 2, 1, 3 }), cfg, past, 1);
    var values = try s.transpose(try s.reshape(try mm(s, x, w.v), &.{ 1, rows, cfg.kv_heads, cfg.dims }), &.{ 0, 2, 1, 3 });
    const index = try s.reshape(try mm(s, x, w.index), &.{ 1, rows, cfg.index_heads + 1, cfg.index_dims });
    const iq = try s.slice(index, 2, 0, cfg.index_heads);
    var raw = try s.reshape(try s.slice(index, 2, cfg.index_heads, cfg.index_heads + 1), &.{ 1, rows, cfg.index_dims });
    if (past > 0) {
        if (previous.keys.ctx == null or previous.values.ctx == null or previous.raw.ctx == null) return error.InvalidCache;
        keys = try s.cat(&.{ try s.slice(previous.keys, 2, 0, past), keys }, 2);
        values = try s.cat(&.{ try s.slice(previous.values, 2, 0, past), values }, 2);
        raw = try s.cat(&.{ try s.slice(previous.raw, 1, 0, past), raw }, 1);
    }
    var cache = Cache{ .keys = keys, .values = values, .raw = raw, .pooled = previous.pooled, .offset = end };
    const out = if (cfg.kernel_select and end > 4096 and @divTrunc(end, 4) > cfg.top) try selected(kernels, ops, s, queries, iq, &cache, w, cfg, past) else blk: {
        const step = if (end > 8192) @as(i32, 256) else rows;
        var parts: [8]A = undefined;
        var count: usize = 0;
        var start: i32 = 0;
        while (start < rows) : (start += step) {
            const stop = @min(start + step, rows);
            const chosen = try mask(s, try s.slice(iq, 1, start, stop), &cache, w, cfg, past + start);
            var attended = c.mlx_array_new();
            const rc = c.mlx_fast_scaled_dot_product_attention(&attended, try s.slice(queries, 2, start, stop), try s.slice(keys, 2, 0, past + stop), try s.slice(values, 2, 0, past + stop), 1 / @sqrt(@as(f32, @floatFromInt(cfg.dims))), if (chosen.ctx == null and stop - start > 1) "causal" else "", chosen, mx.empty, false, mx.stream);
            parts[count] = try s.result(rc, attended);
            if (step < rows) {
                try mx.evalMany(&.{parts[count]}, true);
                if (count > 0) try mx.eval(parts[count - 1]);
            }
            count += 1;
        }
        const all = if (count == 1) parts[0] else try s.cat(parts[0..count], 2);
        break :blk try s.reshape(try s.transpose(all, &.{ 0, 2, 1, 3 }), &.{ 1, rows, cfg.heads * cfg.dims });
    };
    return .{ .output = try mm(s, try s.binary(c.mlx_multiply, out, try s.unary(c.mlx_sigmoid, gate)), w.out), .cache = cache, .queries = queries, .index_queries = iq, .attended = out };
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var ops = Ops{};
    defer ops.deinit();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/attention.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, weights: []const u8, config: Config, past: i32, previous_pooled: bool, pooled: bool, bits: [5]i32, groups: [5]i32 };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    if (cases.value.len == 0) return error.EmptyFixtures;
    for (cases.value) |case| {
        errdefer std.debug.print("Flash prefill attention fixture failed: {s}\n", .{case.name});
        var store = @import("checkpoint.zig").Store.init(32);
        defer store.deinit();
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.weights }), "", "");
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
        var s = mx.Scope{};
        defer s.deinit();
        var w: Weights = undefined;
        inline for (.{ "q", "k", "v", "out", "index" }, 0..) |key, i| @field(w, key) = .{ .arrays = .{ try store.get(key ++ ".weight"), try store.get(key ++ ".scales"), try store.get(key ++ ".biases") }, .format = .{ .bits = case.bits[i], .group_size = case.groups[i] } };
        inline for (.{ "q_scale", "k_scale", "iq_scale", "ik_scale" }) |key| @field(w, key) = try store.get(key);
        const previous = if (case.past > 0) Cache{ .keys = try store.get("previous.keys"), .values = try store.get("previous.values"), .raw = try store.get("previous.raw"), .pooled = if (case.previous_pooled) try store.get("previous.pooled") else mx.empty, .offset = case.past } else Cache{};
        const result = try forward(&kernels, &ops, &s, try store.get("input"), w, case.config, previous);
        inline for (.{ "queries", "index_queries", "attended", "output" }) |key| {
            errdefer std.debug.print("Mismatch in {s}\n", .{key});
            try @import("variant_checks.zig").equalBits(&s, @field(result, key), try store.get(key));
        }
        inline for (.{ "keys", "values", "raw" }) |key| try @import("variant_checks.zig").equalBits(&s, @field(result.cache, key), try store.get("next." ++ key));
        try std.testing.expectEqual(case.pooled, result.cache.pooled.ctx != null);
        if (case.pooled) try @import("variant_checks.zig").equalBits(&s, result.cache.pooled, try store.get("next.pooled"));
        try std.testing.expectEqual(case.past + mx.dim(try store.get("input"), 1), result.cache.offset);
    }
    std.debug.print("PASS: {d} Flash prefill attention layers, dense/sparse boundaries, GQA variants, indexer pooling and retained KV caches\n", .{cases.value.len});
}
