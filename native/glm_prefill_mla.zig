const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const glm = @import("glm.zig");
const c = mx.c;
const A = mx.Array;

pub const Chunk = struct { scores: A = mx.empty, safe_ids: A = mx.empty, attention_scores: A = mx.empty, probabilities: A = mx.empty, output: A = mx.empty };
pub const Result = struct { output: A, q: A, iq: A, iw: A, ql: A, flat: A, chunks: [4]Chunk, count: usize };

fn range(s: *mx.Scope, start: i32, end: i32) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_arange(&out, @floatFromInt(start), @floatFromInt(end), 1, mx.i32t, mx.stream);
    return s.result(rc, out);
}
fn where(s: *mx.Scope, condition: A, yes: A, no: A) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_where(&out, condition, yes, no, mx.stream);
    return s.result(rc, out);
}
fn append(s: *mx.Scope, old: A, value: A) !A {
    return if (old.ctx == null) value else s.cat(&.{ old, value }, 0);
}
fn row(s: *mx.Scope, x: A, index: i32) !A {
    return s.reshape(try s.slice(x, 1, index, index + 1), &.{ mx.dim(x, 0), mx.dim(x, 2) });
}
fn pool(s: *mx.Scope, keys: A, gates: A, ape: A, kp: i32) !A {
    const blocks = @divExact(mx.dim(keys, 0), kp);
    const width = mx.dim(keys, 1);
    const k = try s.reshape(try s.cast(keys, mx.f32t), &.{ blocks, kp, width });
    const logit = try s.binary(c.mlx_add, try s.reshape(try s.cast(gates, mx.f32t), &.{ blocks, kp, width }), try s.cast(ape, mx.f32t));
    var top = try row(s, logit, 0);
    var j: i32 = 1;
    while (j < kp) : (j += 1) top = try s.binary(c.mlx_maximum, top, try row(s, logit, j));
    const e = try s.unary(c.mlx_exp, try s.binary(c.mlx_subtract, logit, try s.reshape(top, &.{ blocks, 1, width })));
    var total = try row(s, e, 0);
    j = 1;
    while (j < kp) : (j += 1) total = try s.binary(c.mlx_add, total, try row(s, e, j));
    var out = try s.binary(c.mlx_multiply, try s.binary(c.mlx_divide, try row(s, e, 0), total), try row(s, k, 0));
    j = 1;
    while (j < kp) : (j += 1) out = try s.binary(c.mlx_add, out, try s.binary(c.mlx_multiply, try s.binary(c.mlx_divide, try row(s, e, j), total), try row(s, k, j)));
    return s.cast(out, mx.bf16);
}

pub fn forward(m: *glm.Model, s: *mx.Scope, layer: usize, x: A, cache: *glm.Cache, past: i32) !Result {
    const g = m.config.value;
    if (mx.shape(x).len != 2 or mx.dtype(x) != mx.bf16 or mx.dim(x, 1) != g.hidden_size) return error.InvalidTensorShape;
    const rows = mx.dim(x, 0);
    if (rows < 1 or rows > 2048 or past < 0 or past > 1048576 - rows) return error.ContextLimitExceeded;
    const kp = g.index_kpool;
    inline for (.{ "keys", "ik", "ig", "pool" }) |key| {
        const a = @field(cache, key);
        const n = if (comptime std.mem.eql(u8, key, "pool")) @divTrunc(past, kp) else past;
        const width = if (comptime std.mem.eql(u8, key, "keys")) g.kv_lora_rank else g.index_head_dim;
        if (a.ctx == null) {
            if (n != 0) return error.InvalidCache;
        } else if (!std.mem.eql(i32, mx.shape(a), &.{ n, width }) or mx.dtype(a) != mx.bf16) return error.InvalidCache;
    }
    var b: [256]u8 = undefined;
    // Prefill must retain separate projections: MLX's batched tiles depend on output width.
    const qr = try cp.norm(s, try m.project(s, layer, "self_attn.q_a_proj", x), try m.weight(layer, "self_attn.q_a_layernorm.weight"), g.rms_norm_eps);
    const q = try s.reshape(try m.project(s, layer, "self_attn.q_b_proj", qr), &.{ rows, g.num_attention_heads, g.qk_nope_head_dim });
    const iq = try s.reshape(try m.project(s, layer, "self_attn.indexer.wq_b", qr), &.{ rows, g.index_n_heads, g.index_head_dim });
    const lat = try cp.norm(s, try m.project(s, layer, "self_attn.kv_a_proj_with_mqa", x), try m.weight(layer, "self_attn.kv_a_layernorm.weight"), g.rms_norm_eps);
    var ik = c.mlx_array_new();
    const ln = c.mlx_fast_layer_norm(&ik, try m.project(s, layer, "self_attn.indexer.wk", x), try m.weight(layer, "self_attn.indexer.k_norm.weight"), try m.weight(layer, "self_attn.indexer.k_norm.bias"), 1e-6, mx.stream);
    ik = try s.result(ln, ik);
    const ig = try s.binary(c.mlx_matmul, x, try s.transpose(try m.weight(layer, "self_attn.indexer.index_kpool_compress_gate"), &.{ 1, 0 }));
    const isc: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(g.index_n_heads))) / @sqrt(@as(f64, @floatFromInt(g.index_head_dim))));
    const iw = try s.binary(c.mlx_multiply, try m.project(s, layer, "self_attn.indexer.weights_proj", x), try s.cast(try s.scalar(isc), mx.bf16));
    var next = cache.*;
    next.keys = try append(s, next.keys, lat);
    next.ik = try append(s, next.ik, ik);
    next.ig = try append(s, next.ig, ig);
    const first = @divTrunc(past, kp);
    const last_block = @divTrunc(past + rows, kp);
    if (last_block > first) next.pool = try append(s, next.pool, try pool(s, try s.slice(next.ik, 0, first * kp, last_block * kp), try s.slice(next.ig, 0, first * kp, last_block * kp), try m.weight(layer, "self_attn.indexer.index_kpool_compress_ape"), kp));
    const absorbed = m.weights.has(try glm.Model.name(&b, layer, "self_attn.embed_q.weight"));
    const ql = try m.qmm(s, try s.transpose(q, &.{ 1, 0, 2 }), try glm.Model.name(&b, layer, if (absorbed) "self_attn.embed_q" else "self_attn.wk"), absorbed, null);
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(g.qk_nope_head_dim))));
    const width = g.index_topk + if (g.index_kpool_always_select_tail) kp - 1 else @as(i32, 0);
    var chunks: [4]Chunk = @splat(.{});
    var outputs: [4]A = undefined;
    var count: usize = 0;
    var begin: i32 = 0;
    while (begin < rows) : (begin += 512) {
        const end = @min(begin + 512, rows);
        const n = end - begin;
        const last = past + end;
        const cq = try s.slice(ql, 1, begin, end);
        var chunk = &chunks[count];
        if (last <= g.index_topk) {
            const keys = try s.reshape(try s.slice(next.keys, 0, 0, last), &.{ 1, 1, last, g.kv_lora_rank });
            var out = c.mlx_array_new();
            const rc = c.mlx_fast_scaled_dot_product_attention(&out, try s.reshape(cq, &.{ 1, g.num_attention_heads, n, g.kv_lora_rank }), keys, keys, scale, "causal", mx.empty, mx.empty, false, mx.stream);
            chunk.output = try s.transpose(try s.reshape(try s.result(rc, out), &.{ g.num_attention_heads, n, g.kv_lora_rank }), &.{ 1, 0, 2 });
        } else {
            const blocks = @divTrunc(last, kp);
            const index = try s.binary(c.mlx_matmul, try s.slice(iq, 0, begin, end), try s.transpose(try s.slice(next.pool, 0, 0, blocks), &.{ 1, 0 }));
            const weighted = try s.binary(c.mlx_multiply, try s.reshape(try s.slice(iw, 0, begin, end), &.{ n, g.index_n_heads, 1 }), try s.binary(c.mlx_maximum, index, try s.cast(try s.scalar(0), mx.bf16)));
            var scores = c.mlx_array_new();
            const rc = c.mlx_sum_axis(&scores, weighted, 1, false, mx.stream);
            chunk.scores = try s.result(rc, scores);
            const pos = try s.reshape(try range(s, past + begin, last), &.{ n, 1 });
            const ends = try s.binary(c.mlx_add, try s.binary(c.mlx_multiply, try s.reshape(try range(s, 0, blocks), &.{ 1, blocks }), try s.ints(&.{kp})), try s.ints(&.{kp - 1}));
            const valid = try s.binary(c.mlx_less_equal, ends, pos);
            const masked = try where(s, valid, chunk.scores, try s.cast(try s.scalar(-1e30), mx.bf16));
            const top = @min(@divTrunc(g.index_topk, kp), blocks);
            var pick = c.mlx_array_new();
            const pc = c.mlx_argpartition_axis(&pick, try s.unary(c.mlx_negative, masked), top - 1, -1, mx.stream);
            pick = try s.slice(try s.result(pc, pick), 1, 0, top);
            var hits = c.mlx_array_new();
            const hc = c.mlx_take_along_axis(&hits, valid, pick, -1, mx.stream);
            hits = try s.result(hc, hits);
            var repeated = c.mlx_array_new();
            const rr = c.mlx_repeat_axis(&repeated, hits, kp, 1, mx.stream);
            repeated = try s.result(rr, repeated);
            const starts = try s.binary(c.mlx_multiply, try s.reshape(try s.cast(pick, mx.i32t), &.{ n, top, 1 }), try s.ints(&.{kp}));
            var ids = try s.reshape(try s.binary(c.mlx_add, starts, try s.reshape(try range(s, 0, kp), &.{ 1, 1, kp })), &.{ n, top * kp });
            const absent = try s.ints(&.{-1});
            ids = try where(s, repeated, ids, absent);
            if (g.index_kpool_always_select_tail and kp > 1) {
                const length = try s.binary(c.mlx_add, pos, try s.ints(&.{1}));
                const rem = try s.binary(c.mlx_remainder, length, try s.ints(&.{kp}));
                const start = try s.binary(c.mlx_subtract, length, rem);
                const tail = try s.binary(c.mlx_add, start, try s.reshape(try range(s, 0, kp - 1), &.{ 1, kp - 1 }));
                ids = try s.cat(&.{ ids, try where(s, try s.binary(c.mlx_less_equal, tail, pos), tail, absent) }, 1);
            }
            const every = try s.reshape(try range(s, 0, width), &.{ 1, width });
            const dense = try s.binary(c.mlx_less, pos, try s.ints(&.{g.index_topk}));
            ids = try where(s, dense, try where(s, try s.binary(c.mlx_less_equal, every, pos), every, absent), ids);
            const valid_sel = try s.binary(c.mlx_greater_equal, ids, try s.ints(&.{0}));
            chunk.safe_ids = try s.reshape(try where(s, valid_sel, ids, try s.ints(&.{0})), &.{-1});
            const keys = try s.reshape(try s.take(try s.slice(next.keys, 0, 0, last), chunk.safe_ids, 0), &.{ n, width, g.kv_lora_rank });
            const scaled = try s.binary(c.mlx_multiply, try s.transpose(cq, &.{ 1, 0, 2 }), try s.cast(try s.scalar(scale), mx.bf16));
            const sc = try s.binary(c.mlx_matmul, scaled, try s.transpose(keys, &.{ 0, 2, 1 }));
            chunk.attention_scores = try where(s, try s.reshape(valid_sel, &.{ n, 1, width }), sc, try s.cast(try s.scalar(-1e30), mx.bf16));
            var probs = c.mlx_array_new();
            const sr = c.mlx_softmax_axis(&probs, chunk.attention_scores, -1, true, mx.stream);
            chunk.probabilities = try s.result(sr, probs);
            chunk.output = try s.binary(c.mlx_matmul, chunk.probabilities, keys);
        }
        outputs[count] = chunk.output;
        count += 1;
    }
    const att = if (count == 1) outputs[0] else try s.cat(outputs[0..count], 0);
    const v = try m.qmm(s, try s.transpose(att, &.{ 1, 0, 2 }), try glm.Model.name(&b, layer, if (absorbed) "self_attn.unembed_out" else "self_attn.wv"), true, null);
    const flat = try s.reshape(try s.transpose(v, &.{ 1, 0, 2 }), &.{ rows, g.num_attention_heads * g.v_head_dim });
    const out = try m.project(s, layer, "self_attn.o_proj", flat);
    cache.* = next;
    return .{ .output = out, .q = q, .iq = iq, .iw = iw, .ql = ql, .flat = flat, .chunks = chunks, .count = count };
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try glm.Model.prepareRuntime();
    try mx.init();
    defer mx.shutdown();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/mla.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, decode: bool, chunks: usize };
    const Group = struct { checkpoint: []const u8, cases: []const Case };
    const groups = try std.json.parseFromSlice([]const Group, mx.allocator, bytes, .{});
    defer groups.deinit();
    if (groups.value.len == 0) return error.EmptyFixtures;
    var count: usize = 0;
    for (groups.value) |group| {
        var m = try glm.Model.init(io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, group.checkpoint }));
        defer m.deinit();
        if (group.cases.len == 0) return error.EmptyFixtures;
        var position: i32 = 0;
        for (group.cases) |case| {
            errdefer std.debug.print("GLM prefill MLA fixture failed: {s}/{s}\n", .{ group.checkpoint, case.name });
            var store = cp.Store.init(64);
            defer store.deinit();
            try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
            var s = mx.Scope{};
            defer s.deinit();
            const x = try store.get("input");
            const equal = @import("variant_checks.zig").equalBits;
            var cache = m.cache[0];
            if (position != 0) inline for (.{ "keys", "ik", "ig", "pool" }) |key| {
                const a = @field(cache, key);
                if (a.ctx != null) try equal(&s, a, try store.get("previous." ++ key));
            };
            const out = if (case.decode) try m.mla(&s, 0, x, &cache, position) else blk: {
                const result = try forward(&m, &s, 0, x, &cache, position);
                inline for (.{ "q", "iq", "iw", "ql", "flat", "output" }) |key| {
                    errdefer std.debug.print("Mismatch in {s}\n", .{key});
                    try equal(&s, @field(result, key), try store.get(key));
                }
                try std.testing.expectEqual(case.chunks, result.count);
                for (result.chunks[0..result.count], 0..) |chunk, j| inline for (comptime std.meta.fieldNames(Chunk)) |key| {
                    const a = @field(chunk, key);
                    const name = try std.fmt.bufPrint(&path, "chunk{d}.{s}", .{ j, key });
                    errdefer std.debug.print("Mismatch in {s}\n", .{name});
                    try std.testing.expectEqual(store.has(name), a.ctx != null);
                    if (a.ctx != null) try equal(&s, a, try store.get(name));
                };
                break :blk result.output;
            };
            try equal(&s, out, try store.get("output"));
            inline for (.{ "keys", "ik", "ig", "pool" }) |key| {
                errdefer std.debug.print("Mismatch in cache {s}\n", .{key});
                const a = @field(cache, key);
                const expected = try store.get("next." ++ key);
                if (a.ctx == null) {
                    try std.testing.expectEqual(@as(usize, 0), c.mlx_array_size(expected));
                } else try equal(&s, a, expected);
            }
            const saved = try cache.clone();
            m.cache[0].deinit();
            m.cache[0] = saved;
            position += mx.dim(x, 0);
            count += 1;
        }
        var s = mx.Scope{};
        defer s.deinit();
        var cache = glm.Cache{};
        const one = try s.zeros(&.{ 1, m.config.value.hidden_size }, mx.bf16);
        try std.testing.expectError(error.InvalidCache, forward(&m, &s, 0, one, &cache, 1));
        try std.testing.expectError(error.ContextLimitExceeded, forward(&m, &s, 0, one, &cache, -1));
        try std.testing.expectError(error.ContextLimitExceeded, forward(&m, &s, 0, one, &cache, 1048576));
        try std.testing.expect(cache.keys.ctx == null);
    }
    std.debug.print("PASS: {d} GLM prefill MLA cases, exact latent/indexer arrays, pooled cache, dense/sparse query chunks and decode continuation\n", .{count});
}
