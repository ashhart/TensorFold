//! DFlash2: target taps -> cached context K/V -> parallel block -> best-first tree.
const std = @import("std");
const mx = @import("mlx.zig");
const lanes = @import("lanes.zig");
const model = @import("model.zig");
const sampling = @import("sampling.zig");
const weights = @import("weights.zig");
const A = mx.Array;
const conv_spec = @import("kernel_sources.zig").Spec{ .name = "dflash_dynamic_conv_v1", .inputs = &.{ "H", "DYNAMIC", "BASE", "dims" }, .outputs = &.{"OUT"}, .source = @embedFile("metal/dynamic_conv.metal"), .header = "", .contiguous = true };
pub const Proposal = struct { tokens: [31]i32 = undefined, parents: [31]i32 = undefined, len: usize = 0 };
pub const Drafter = struct {
    weights: weights.Weights,
    cache: [5]model.Cache = @splat(.{}),
    offset: i32 = 0,
    head: lanes.Linear,
    pred: A,
    succ: A,
    pub fn init(io: std.Io, dir: []const u8, target: *model.Model) !Drafter {
        var w = weights.Weights.init();
        errdefer w.deinit();
        try w.loadDraft(io, dir);
        var s = mx.Scope{};
        defer s.deinit();
        const head = try target.weights.linear("lm_head");
        const q = try s.cat(&.{ try s.slice(head.weight, 0, 0, 98304), try s.slice(head.weight, 0, 248032, 248320) }, 0);
        const sb = try s.cat(&.{ try s.slice(head.sb, 1, 0, 98304), try s.slice(head.sb, 1, 248032, 248320) }, 1);
        const pred = try s.cast(try w.get("candidate_selector.predecessor_codebook"), mx.f32t);
        const succ = try s.cast(try w.get("candidate_selector.successor_codebook"), mx.f32t);
        try mx.evalMany(&.{ q, sb, pred, succ }, false);
        const qown = try mx.retain(q);
        errdefer mx.free(qown);
        const sbown = try mx.retain(sb);
        errdefer mx.free(sbown);
        const pown = try mx.retain(pred);
        errdefer mx.free(pown);
        const sc = if (mx.tensor_units) mx.empty else try mx.retain(try s.cat(&.{ try s.slice(head.scales, 0, 0, 98304), try s.slice(head.scales, 0, 248032, 248320) }, 0));
        errdefer mx.free(sc);
        const bi = if (mx.tensor_units) mx.empty else try mx.retain(try s.cat(&.{ try s.slice(head.biases, 0, 0, 98304), try s.slice(head.biases, 0, 248032, 248320) }, 0));
        errdefer mx.free(bi);
        return .{ .weights = w, .head = .{ .weight = qown, .sb = sbown, .n = 98592, .k = 5120, .tiled = mx.tensor_units, .scales = sc, .biases = bi }, .pred = pown, .succ = try mx.retain(succ) };
    }
    pub fn reset(d: *Drafter) void {
        for (&d.cache) |*v| v.deinit();
        d.offset = 0;
    }
    pub fn deinit(d: *Drafter) void {
        d.reset();
        d.weights.deinit();
        d.head.deinit();
        mx.free(d.pred);
        mx.free(d.succ);
    }
    fn get(d: *Drafter, i: usize, suffix: []const u8) !A {
        var buf: [160]u8 = undefined;
        return d.weights.get(try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, suffix }));
    }
    fn project(d: *Drafter, k: *mx.Kernels, s: *mx.Scope, i: usize, suffix: []const u8, x: A) !A {
        var buf: [160]u8 = undefined;
        return (try d.weights.linear(try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, suffix }))).apply(k, s, .{ .x = x });
    }
    /// Only committed target rows enter the drafter cache; rejected siblings never do.
    pub fn absorb(d: *Drafter, target: *model.Model, p: *model.Pass, rows: []const i32) !void {
        const s = &p.scope;
        const k = &target.kernels;
        const ids = try s.ints(rows);
        var taps: [5]A = undefined;
        for (p.taps, 0..) |v, i| taps[i] = try s.take(v, ids, 1);
        const ctx = try s.rms(try (try d.weights.linear("fc")).apply(k, s, .{ .x = try s.cat(&taps, -1) }), try d.weights.get("hidden_norm.weight"));
        const count: i32 = @intCast(rows.len);
        var positions: [128]i32 = undefined;
        for (rows, 0..) |_, j| positions[j] = d.offset + @as(i32, @intCast(j));
        const pos = try s.ints(positions[0..rows.len]);
        var next: [5]model.Cache = @splat(.{});
        errdefer for (&next) |*v| v.deinit();
        for (0..5) |i| {
            var keys = try s.rms(try s.reshape(try d.project(k, s, i, "self_attn.k_proj", ctx), &.{ 1, count, 8, 128 }), try d.get(i, "self_attn.k_norm.weight"));
            keys = try s.transpose(try s.rope(try s.transpose(keys, &.{ 1, 2, 0, 3 }), pos, 128), &.{ 2, 1, 0, 3 });
            var values = try s.transpose(try s.reshape(try d.project(k, s, i, "self_attn.v_proj", ctx), &.{ 1, count, 8, 128 }), &.{ 0, 2, 1, 3 });
            if (d.cache[i].a.ctx != null) {
                keys = try s.cat(&.{ d.cache[i].a, keys }, 2);
                values = try s.cat(&.{ d.cache[i].b, values }, 2);
            }
            const n = mx.dim(keys, 2);
            if (n > 2047) {
                keys = try s.slice(keys, 2, n - 2047, n);
                values = try s.slice(values, 2, n - 2047, n);
            }
            next[i].a = try mx.retain(try s.contiguous(keys));
            next[i].b = try mx.retain(try s.contiguous(values));
        }
        var arrays: [10]A = undefined;
        for (next, 0..) |v, i| {
            arrays[i * 2] = v.a;
            arrays[i * 2 + 1] = v.b;
        }
        try mx.evalMany(&arrays, false);
        for (&d.cache) |*v| v.deinit();
        d.cache = next;
        d.offset += count;
    }
    fn convolve(d: *Drafter, k: *mx.Kernels, s: *mx.Scope, i: usize, prefix: []const u8, h: A, dynamic: A, part: i32) !A {
        var buf: [160]u8 = undefined;
        const base = try d.get(i, try std.fmt.bufPrint(&buf, "{s}.base_kernel", .{prefix}));
        const n = mx.dim(h, 1);
        return (try k.run(s, conv_spec, &.{ h, dynamic, base, try s.ints(&.{n}) }, &.{ mx.ti("N", 5120), mx.ti("PART", part) }, .{ n * 5120, 1, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ 1, n, 5120 } }}))[0];
    }
    fn prepare(d: *Drafter, k: *mx.Kernels, s: *mx.Scope, i: usize, prefix: []const u8, x: A) !struct { x: A, dynamic: A } {
        var buf: [160]u8 = undefined;
        const dynamic = try d.project(k, s, i, try std.fmt.bufPrint(&buf, "{s}.kernel_projection", .{prefix}), x);
        return .{ .x = try d.convolve(k, s, i, prefix, x, dynamic, 0), .dynamic = dynamic };
    }
    fn attention(d: *Drafter, k: *mx.Kernels, s: *mx.Scope, i: usize, x: A) !A {
        const n = mx.dim(x, 1);
        var positions: [16]i32 = undefined;
        for (0..@intCast(n)) |j| positions[j] = d.offset + @as(i32, @intCast(j));
        const pos = try s.ints(positions[0..@intCast(n)]);
        var q = try s.rms(try s.reshape(try d.project(k, s, i, "self_attn.q_proj", x), &.{ 1, n, 32, 128 }), try d.get(i, "self_attn.q_norm.weight"));
        q = try s.transpose(try s.rope(try s.transpose(q, &.{ 1, 2, 0, 3 }), pos, 128), &.{ 2, 1, 0, 3 });
        var keys = try s.rms(try s.reshape(try d.project(k, s, i, "self_attn.k_proj", x), &.{ 1, n, 8, 128 }), try d.get(i, "self_attn.k_norm.weight"));
        keys = try s.transpose(try s.rope(try s.transpose(keys, &.{ 1, 2, 0, 3 }), pos, 128), &.{ 2, 1, 0, 3 });
        const values = try s.transpose(try s.reshape(try d.project(k, s, i, "self_attn.v_proj", x), &.{ 1, n, 8, 128 }), &.{ 0, 2, 1, 3 });
        const ctx = mx.dim(d.cache[i].a, 2);
        const total = ctx + n;
        const mask = try mx.allocator.alloc(u8, @intCast(n * total));
        defer mx.allocator.free(mask);
        for (0..@intCast(n)) |row| for (0..@intCast(total)) |col| {
            mask[row * @as(usize, @intCast(total)) + col] = @intFromBool(col >= ctx or @as(i32, @intCast(row)) + ctx - @as(i32, @intCast(col)) < 2048);
        };
        const mask_array = try s.data(mask.ptr, &.{ n, total }, mx.c.MLX_BOOL);
        var out = mx.c.mlx_array_new();
        const rc = mx.c.mlx_fast_scaled_dot_product_attention(&out, q, try s.cat(&.{ d.cache[i].a, keys }, 2), try s.cat(&.{ d.cache[i].b, values }, 2), 0.08838834764831845, "", mask_array, mx.empty, false, mx.stream);
        out = try s.result(rc, out);
        return d.project(k, s, i, "self_attn.o_proj", try s.reshape(try s.transpose(out, &.{ 0, 2, 1, 3 }), &.{ 1, n, 4096 }));
    }
    pub fn propose(d: *Drafter, target: *model.Model, anchor: i32, budget: usize, settings: sampling.Sampling) !Proposal {
        if (budget == 0 or d.cache[0].a.ctx == null) return .{};
        const n: usize = @min(16, budget + 1);
        var block: [16]i32 = @splat(248070);
        block[0] = anchor;
        var s = mx.Scope{};
        defer s.deinit();
        const k = &target.kernels;
        var h = try target.weights.embed(&s, block[0..n]);
        for (0..5) |i| {
            const pre = try d.prepare(k, &s, i, "attention_conv", try s.rms(h, try d.get(i, "input_layernorm.weight")));
            const attended = try d.attention(k, &s, i, pre.x);
            h = try s.binary(mx.c.mlx_add, h, try d.convolve(k, &s, i, "attention_conv", attended, pre.dynamic, 1));
            const post = try d.prepare(k, &s, i, "mlp_conv", try s.rms(h, try d.get(i, "post_attention_layernorm.weight")));
            const gate = try d.project(k, &s, i, "mlp.gate_proj", post.x);
            const up = try d.project(k, &s, i, "mlp.up_proj", post.x);
            const act = try s.binary(mx.c.mlx_multiply, try s.binary(mx.c.mlx_multiply, gate, try s.unary(mx.c.mlx_sigmoid, gate)), up);
            const down = try d.project(k, &s, i, "mlp.down_proj", act);
            h = try s.binary(mx.c.mlx_add, h, try d.convolve(k, &s, i, "mlp_conv", down, post.dynamic, 1));
            try mx.evalMany(&.{h}, true);
        }
        const hidden = try s.rms(try s.slice(h, 1, 1, @intCast(n)), try d.weights.get("norm.weight"));
        const logits = try d.head.apply(k, &s, .{ .x = hidden });
        const ranked = try @import("gpu_sampling.zig").topk(k, &s, logits, 16);
        const projection = try s.cast(try (try d.weights.linear("candidate_selector.hidden_projection")).apply(k, &s, .{ .x = hidden }), mx.f32t);
        try mx.evalMany(&.{ ranked[0], ranked[1], projection }, false);
        const indices = mx.c.mlx_array_data_int32(ranked[0]);
        const values = mx.c.mlx_array_data_float32(ranked[1]);
        var candidates: [15][16]sampling.Candidate = undefined;
        for (0..n - 1) |row| {
            for (0..16) |j| {
                const id = indices[row * 16 + j];
                candidates[row][j] = .{ .id = if (id < 98304) id else id - 98304 + 248032, .value = values[row * 16 + j] };
            }
        }
        return d.search(candidates[0 .. n - 1], mx.c.mlx_array_data_float32(projection)[0 .. (n - 1) * 256], anchor, @min(budget, 15), settings);
    }
    fn search(d: *Drafter, candidates: []const [16]sampling.Candidate, hproj: []const f32, anchor: i32, budget: usize, settings: sampling.Sampling) Proposal {
        const Node = struct { score: f64, depth: usize, token: i32, parent: i32 };
        var queue: [64]Node = undefined;
        var len: usize = 0;
        var result = Proposal{};
        var parent: i32 = -1;
        var predecessor = anchor;
        var depth: usize = 0;
        var path_score: f64 = 0;
        const pred = mx.c.mlx_array_data_float32(d.pred);
        const succ = mx.c.mlx_array_data_float32(d.succ);
        while (true) {
            if (depth < candidates.len) {
                var scores: [16]f64 = undefined;
                var max: f64 = -std.math.inf(f64);
                const temp = if (settings.temperature > 0) @max(settings.temperature, 1e-6) else 1;
                for (candidates[depth], 0..) |candidate, j| {
                    var edge: f64 = 0;
                    for (0..256) |r| edge += @as(f64, pred[@as(usize, @intCast(predecessor)) * 256 + r]) * @as(f64, hproj[depth * 256 + r]) * @as(f64, succ[@as(usize, @intCast(candidate.id)) * 256 + r]);
                    const noise = if (settings.temperature > 0) sampling.noise(settings.seed, @intCast(d.offset + 1 + @as(i32, @intCast(depth))), @intCast(candidate.id)) else 0;
                    scores[j] = (candidate.value / temp + 0.6 * edge / temp + 0.7 * noise) / 1.5;
                    max = @max(max, scores[j]);
                }
                var sum: f64 = 0;
                for (scores) |score| sum += @exp(score - max);
                const normalizer = max + @log(sum);
                var chosen: [16]bool = @splat(false);
                for (0..4) |_| {
                    var best: usize = 0;
                    var best_score: f64 = -std.math.inf(f64);
                    for (scores, 0..) |score, j| if (!chosen[j] and score > best_score) {
                        best = j;
                        best_score = score;
                    };
                    chosen[best] = true;
                    queue[len] = .{ .score = path_score + scores[best] - normalizer, .depth = depth, .token = candidates[depth][best].id, .parent = parent };
                    len += 1;
                }
            }
            if (len == 0 or result.len >= budget) break;
            var best: usize = 0;
            for (queue[0..len], 0..) |node, j| if (node.score > queue[best].score) {
                best = j;
            };
            const node = queue[best];
            len -= 1;
            queue[best] = queue[len];
            result.tokens[result.len] = node.token;
            result.parents[result.len] = node.parent;
            parent = @intCast(result.len);
            predecessor = node.token;
            depth = node.depth + 1;
            path_score = node.score;
            result.len += 1;
        }
        return result;
    }
};
