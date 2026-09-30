const std = @import("std");
const mx = @import("mlx.zig");
const ds = @import("deepseek.zig");
const ops = @import("large_family_ops.zig");
const src = @import("kernel_sources.zig");
const c = mx.c;
const A = mx.Array;

pub const Result = struct { q: A, kv: A, qr: A, projection: A = mx.empty, iq: A = mx.empty, iw: A = mx.empty, attended: A, grouped: A, output: A };

fn range(s: *mx.Scope, lo: i32, hi: i32) !A {
    const values = try mx.allocator.alloc(i32, @intCast(hi - lo));
    defer mx.allocator.free(values);
    for (values, 0..) |*value, i| value.* = lo + @as(i32, @intCast(i));
    return s.ints(values);
}
fn where(s: *mx.Scope, condition: A, yes: A, no: A) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_where(&out, condition, yes, no, mx.stream);
    return s.result(rc, out);
}
fn best(s: *mx.Scope, values: A, top: i32, ascending: bool) !A {
    var ids = c.mlx_array_new();
    const rc = c.mlx_argpartition_axis(&ids, try s.unary(c.mlx_negative, values), top - 1, -1, mx.stream);
    ids = try s.slice(try s.result(rc, ids), 1, 0, top);
    if (!ascending) return ids;
    var sorted = c.mlx_array_new();
    const sr = c.mlx_sort_axis(&sorted, ids, -1, mx.stream);
    return s.result(sr, sorted);
}
fn scores(m: *ds.Model, s: *mx.Scope, q: A, w: A, keys: A) !A {
    const g = m.config.value;
    const dot = try s.binary(c.mlx_matmul, q, try s.transpose(keys, &.{ 1, 0 }));
    const relu = try s.binary(c.mlx_maximum, dot, try s.cast(try s.scalar(0), mx.bf16));
    const scaled = try s.binary(c.mlx_multiply, relu, try s.cast(try s.scalar(1 / @sqrt(@as(f32, @floatFromInt(g.index_head_dim)))), mx.bf16));
    const weighted = try s.binary(c.mlx_multiply, try s.cast(scaled, mx.f32t), try s.reshape(try s.cast(w, mx.f32t), &.{ mx.dim(q, 0), g.index_n_heads, 1 }));
    var out = c.mlx_array_new();
    const rc = c.mlx_sum_axis(&out, weighted, 1, false, mx.stream);
    return s.result(rc, out);
}

pub fn forward(m: *ds.Model, s: *mx.Scope, layer: usize, x: A, entry: *ds.Cache, past: i32) !Result {
    const g = m.config.value;
    if (mx.shape(x).len != 2 or mx.dtype(x) != mx.bf16 or mx.dim(x, 1) != g.hidden_size) return error.InvalidTensorShape;
    const rows = mx.dim(x, 0);
    if (rows < 1 or rows > 2048 or past < 0 or past > 1048576 - rows) return error.ContextLimitExceeded;
    const old_rows = @min(past, g.sliding_window);
    if ((entry.keys.ctx != null) != (old_rows > 0)) return error.InvalidTensorShape;
    if (old_rows > 0 and (!std.mem.eql(i32, mx.shape(entry.keys), &.{ old_rows, g.head_dim }) or mx.dtype(entry.keys) != mx.bf16)) return error.InvalidTensorShape;
    var cache = entry.*;
    const positions = try range(s, past, past + rows);
    const inv = try m.weight(layer, "inv");
    const eps = try s.scalar(g.rms_norm_eps);
    const xp = try ds.Model.stock(s, x, try m.triple(layer, "attn.x_proj"));
    const qr = try @import("checkpoint.zig").norm(s, try s.slice(xp, 1, 0, g.q_lora_rank), try m.weight(layer, "attn.q_norm.weight"), g.rms_norm_eps);
    const kv = try ops.normRope(&m.kernels, s, try s.slice(xp, 1, g.q_lora_rank, g.q_lora_rank + g.head_dim), try m.weight(layer, "attn.kv_norm.weight"), positions, inv, eps, true, false);
    const q = try ops.normRope(&m.kernels, s, try s.reshape(try ds.Model.stock(s, qr, try m.triple(layer, "attn.wq_b")), &.{ rows, g.num_attention_heads, g.head_dim }), null, positions, inv, eps, true, false);
    var result = Result{ .q = q, .kv = kv, .qr = qr, .attended = mx.empty, .grouped = mx.empty, .output = mx.empty };
    const ratio = m.layerRatio(layer);
    const pooled = if (ratio > 0) @divTrunc(past + rows, ratio) else 0;
    if (ratio > 0) result.projection = try @import("deepseek_prefill_compress.zig").forward(m, s, layer, x, &cache, past);
    if (ratio == 4 and pooled > g.index_topk) {
        result.iq = try ops.normRope(&m.kernels, s, try s.reshape(try ds.Model.stock(s, qr, try m.triple(layer, "attn.indexer.wq_b")), &.{ rows, g.index_n_heads, g.index_head_dim }), null, positions, inv, eps, false, false);
        result.iw = try s.binary(c.mlx_multiply, try ds.Model.stock(s, x, try m.triple(layer, "attn.indexer.weights_proj")), try s.cast(try s.scalar(1 / @sqrt(@as(f32, @floatFromInt(g.index_n_heads)))), mx.bf16));
    }
    const lo = @max(0, past - (g.sliding_window - 1));
    const tail = if (past > lo) try s.slice(entry.keys, 0, old_rows - (past - lo), old_rows) else mx.empty;
    const keys = if (tail.ctx != null) try s.cat(&.{ tail, kv }, 0) else kv;
    const all_keys = if (entry.keys.ctx != null) try s.cat(&.{ entry.keys, kv }, 0) else kv;
    cache.keys = try s.contiguous(try s.slice(all_keys, 0, @max(0, mx.dim(all_keys, 0) - g.sliding_window), mx.dim(all_keys, 0)));
    const production = g.num_attention_heads == 64 and g.head_dim == 512;
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(g.head_dim)));
    if (production) {
        const counts = try mx.allocator.alloc(i32, @intCast(rows));
        defer mx.allocator.free(counts);
        const windows = try mx.allocator.alloc(i32, @intCast(2 * rows));
        defer mx.allocator.free(windows);
        for (counts, 0..) |*count, i| {
            const pos = past + @as(i32, @intCast(i));
            count.* = if (ratio > 0) @divTrunc(pos + 1, ratio) else 0;
            windows[2 * i] = @max(lo, pos - g.sliding_window + 1) - lo;
            windows[2 * i + 1] = pos - lo;
        }
        var ids = mx.empty;
        if (result.iq.ctx != null) {
            const step = @max(16, @min(512, @divTrunc(1 << 27, g.index_n_heads * pooled)));
            var parts: std.ArrayList(A) = .empty;
            defer parts.deinit(mx.allocator);
            var start: i32 = 0;
            while (start < rows) : (start += step) {
                const end = @min(rows, start + step);
                const visible = try s.binary(c.mlx_less, try s.reshape(try range(s, 0, pooled), &.{ 1, pooled }), try s.reshape(try s.ints(counts[@intCast(start)..@intCast(end)]), &.{ end - start, 1 }));
                const index_scores = try scores(m, s, try s.slice(result.iq, 0, start, end), try s.slice(result.iw, 0, start, end), cache.ipool);
                try parts.append(mx.allocator, try best(s, try where(s, visible, index_scores, try s.scalar(-std.math.inf(f32))), g.index_topk, true));
            }
            ids = try s.cast(try s.cat(parts.items, 0), mx.i32t);
            for (counts) |*count| count.* = @min(count.*, g.index_topk);
        }
        const inputs = [_]A{ q, if (pooled > 0) cache.pool else try s.slice(keys, 0, 0, 1), if (ids.ctx != null) try s.reshape(ids, &.{-1}) else try s.ints(&.{0}), try s.ints(counts), keys, try s.ints(windows), try s.cast(try m.weight(layer, "attn.attn_sink"), mx.f32t), try s.scalar(scale), try s.ints(&.{ @intFromBool(ids.ctx == null), if (ids.ctx != null) g.index_topk else 1, mx.dim(keys, 0), lo }), inv };
        result.attended = (try m.kernels.run(s, if (rows <= 16) src.ds4_attn_split else src.ds4_attn_rows, &inputs, if (rows <= 16) &.{ mx.ti("S", 4), mx.ti("ROT", 1), mx.ti("PE", g.qk_rope_head_dim) } else &.{ mx.ti("BK", 16), mx.ti("HTG", 8), mx.ti("ROT", 1), mx.ti("PE", g.qk_rope_head_dim) }, .{ 128, rows * @as(i32, if (rows <= 16) 64 else 8), 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ rows, 64, 512 } }}))[0];
    } else {
        const joined = if (pooled > 0) try s.cat(&.{ cache.pool, keys }, 0) else keys;
        const times = try s.reshape(try range(s, lo, past + rows), &.{ 1, -1 });
        var parts: std.ArrayList(A) = .empty;
        defer parts.deinit(mx.allocator);
        var start: i32 = 0;
        while (start < rows) : (start += 512) {
            const end = @min(rows, start + 512);
            const n = end - start;
            const pos = try s.reshape(try range(s, past + start, past + end), &.{ n, 1 });
            var mask = try s.binary(c.mlx_logical_and, try s.binary(c.mlx_less_equal, times, pos), try s.binary(c.mlx_greater, times, try s.binary(c.mlx_subtract, pos, try s.ints(&.{g.sliding_window}))));
            if (pooled > 0) {
                const block_ends = try s.binary(c.mlx_subtract, try s.binary(c.mlx_multiply, try range(s, 1, pooled + 1), try s.ints(&.{ratio})), try s.ints(&.{1}));
                var visible = try s.binary(c.mlx_less_equal, try s.reshape(block_ends, &.{ 1, pooled }), pos);
                if (result.iq.ctx != null) {
                    const index_scores = try scores(m, s, try s.slice(result.iq, 0, start, end), try s.slice(result.iw, 0, start, end), cache.ipool);
                    const ids = try best(s, try where(s, visible, index_scores, try s.scalar(-std.math.inf(f32))), g.index_topk, false);
                    var chosen = c.mlx_array_new();
                    const rc = c.mlx_put_along_axis(&chosen, try s.zeros(&.{ n, pooled }, c.MLX_BOOL), ids, try s.cast(try s.ints(&.{1}), c.MLX_BOOL), -1, mx.stream);
                    visible = try s.binary(c.mlx_logical_and, visible, try s.result(rc, chosen));
                }
                mask = try s.cat(&.{ visible, mask }, 1);
            }
            const query = try s.reshape(try s.transpose(try s.slice(q, 0, start, end), &.{ 1, 0, 2 }), &.{ 1, g.num_attention_heads, n, g.head_dim });
            const key4 = try s.reshape(joined, &.{ 1, 1, mx.dim(joined, 0), g.head_dim });
            var out = c.mlx_array_new();
            const rc = c.mlx_fast_scaled_dot_product_attention(&out, query, key4, key4, scale, "", mask, try s.cast(try m.weight(layer, "attn.attn_sink"), mx.bf16), false, mx.stream);
            try parts.append(mx.allocator, try s.transpose(try s.reshape(try s.result(rc, out), &.{ g.num_attention_heads, n, g.head_dim }), &.{ 1, 0, 2 }));
        }
        result.attended = try s.cat(parts.items, 0);
    }
    const rotated = if (production) result.attended else try ops.normRope(&m.kernels, s, result.attended, null, positions, inv, eps, false, true);
    result.grouped = try s.reshape(rotated, &.{ rows, g.o_groups, -1 });
    const weights = try m.triple(layer, "attn.wo_a");
    const parts = try mx.allocator.alloc(A, @intCast(g.o_groups));
    defer mx.allocator.free(parts);
    for (parts, 0..) |*part, i| {
        const group: i32 = @intCast(i);
        var w: [3]A = undefined;
        for (&w, weights) |*value, tensor| value.* = try s.slice(tensor, 0, group * g.o_lora_rank, (group + 1) * g.o_lora_rank);
        part.* = try ds.Model.stock(s, try s.reshape(try s.slice(result.grouped, 1, group, group + 1), &.{ rows, -1 }), w);
    }
    result.output = try ds.Model.stock(s, try s.cat(parts, 1), try m.triple(layer, "attn.wo_b"));
    entry.* = cache;
    return result;
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/attention.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, past: i32 };
    const Group = struct { checkpoint: []const u8, cases: []const Case };
    const groups = try std.json.parseFromSlice([]const Group, mx.allocator, bytes, .{});
    defer groups.deinit();
    if (groups.value.len == 0) return error.EmptyFixtures;
    var count: usize = 0;
    for (groups.value) |group| {
        var m = try ds.Model.init(io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, group.checkpoint }));
        defer m.deinit();
        var cache = ds.Cache{};
        defer cache.deinit();
        if (group.cases.len == 0) return error.EmptyFixtures;
        for (group.cases) |case| {
            errdefer std.debug.print("DeepSeek prefill attention fixture failed: {s}/{s}\n", .{ group.checkpoint, case.name });
            var store = @import("checkpoint.zig").Store.init(32);
            defer store.deinit();
            try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
            var s = mx.Scope{};
            defer s.deinit();
            var next = cache;
            const result = try forward(&m, &s, 0, try store.get("input"), &next, case.past);
            inline for (comptime std.meta.fieldNames(Result)) |key| {
                errdefer std.debug.print("Mismatch in {s}\n", .{key});
                const actual = @field(result, key);
                try std.testing.expectEqual(store.has(key), actual.ctx != null);
                if (actual.ctx != null) try @import("variant_checks.zig").equalBits(&s, actual, try store.get(key));
            }
            inline for (comptime std.meta.fieldNames(ds.Cache)) |key| {
                errdefer std.debug.print("Mismatch in cache-{s}\n", .{key});
                const actual = @field(next, key);
                try std.testing.expectEqual(store.has("cache-" ++ key), actual.ctx != null);
                if (actual.ctx != null) try @import("variant_checks.zig").equalBits(&s, actual, try store.get("cache-" ++ key));
            }
            const owned = try next.clone();
            cache.deinit();
            cache = owned;
            count += 1;
        }
    }
    std.debug.print("PASS: {d} DeepSeek batched attention cases, causal pools/windows, sparse selection, staged production attention and every cache\n", .{count});
}
