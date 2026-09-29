const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const src = @import("kernel_sources.zig");
const ops = @import("large_family_ops.zig");
const A = mx.Array;
const c = mx.c;
const Config = @import("deepseek.zig").Config;
const DraftConfig = struct {
    dspark_block_size: i32,
    dspark_noise_token_id: i32,
    dspark_target_layer_ids: []const usize,
    dspark_markov_rank: i32,
    fn validate(g: DraftConfig, layers: usize, vocab: i32) !void {
        if (g.dspark_block_size < 1 or g.dspark_block_size > 16 or g.dspark_noise_token_id < 0 or g.dspark_noise_token_id >= vocab or g.dspark_markov_rank < 1 or g.dspark_markov_rank > 4096 or g.dspark_target_layer_ids.len < 1 or g.dspark_target_layer_ids.len > layers) return error.InvalidDraftConfig;
        for (g.dspark_target_layer_ids, 0..) |layer, i| if (layer >= layers or (i > 0 and layer <= g.dspark_target_layer_ids[i - 1])) return error.InvalidDraftConfig;
    }
};

pub const Draft = struct {
    weights: cp.Store,
    config: std.json.Parsed(DraftConfig),
    base: Config,
    keys: []A,
    position: i32 = 0,
    activations: @import("prefill_ops.zig").Ops = .{},

    pub fn init(io: std.Io, dir: []const u8, base: Config) !Draft {
        var path: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        const config = try std.json.parseFromSlice(DraftConfig, mx.allocator, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        errdefer config.deinit();
        const g = config.value;
        try g.validate(base.num_hidden_layers, base.vocab_size);
        var weights = cp.Store.init(64);
        errdefer weights.deinit();
        try weights.loadFile(io, try std.fmt.bufPrint(&path, "{s}/dspark.safetensors", .{dir}), "", "");
        var count: usize = 0;
        var it = weights.arrays.keyIterator();
        while (it.next()) |key| if (std.mem.endsWith(u8, key.*, ".attn_norm.weight")) {
            count += 1;
        };
        if (count < 1 or count > 16) return error.InvalidDraftCheckpoint;
        for (0..count) |i| {
            const layer = base.num_hidden_layers + 1 + i;
            if (layer < base.compress_ratios.len and base.compress_ratios[layer] != 0) return error.UnsupportedCompressionRatio;
        }
        const keys = try mx.allocator.alloc(A, count);
        @memset(keys, mx.empty);
        var d = Draft{ .weights = weights, .config = config, .base = base, .keys = keys };
        weights = cp.Store.init(64);
        errdefer d.weights.deinit();
        errdefer mx.allocator.free(keys);
        var s = mx.Scope{};
        defer s.deinit();
        for (0..count) |i| {
            try d.validateLayer(i);
            try d.stack(&s, i, "attn.x_proj", &.{ "attn.wq_a", "attn.wkv" });
            try d.stack(&s, i, "ffn.shared_gate_up", &.{ "ffn.shared_experts.gate_proj", "ffn.shared_experts.up_proj" });
            const router = try s.contiguous(try s.transpose(try s.cast(try d.weight(i, "ffn.gate.weight"), mx.f32t), &.{ 1, 0 }));
            try d.put(i, "router", router);
        }
        const checks = @import("deepseek.zig").Model;
        try checks.expectQ(try d.triple(0, "main_proj"), base.hidden_size, @as(i32, @intCast(g.dspark_target_layer_ids.len)) * base.hidden_size);
        try checks.expectTensor(try d.weight(0, "main_norm.weight"), &.{base.hidden_size}, mx.bf16);
        try checks.expectFloat(try d.weight(count - 1, "hc_head.fn"), &.{ 4, 4 * base.hidden_size });
        try checks.expectFloat(try d.weight(count - 1, "hc_head.base"), &.{4});
        try checks.expectFloat(try d.weight(count - 1, "hc_head.scale"), &.{1});
        try checks.expectTensor(try d.weight(count - 1, "norm.weight"), &.{base.hidden_size}, mx.bf16);
        for ([_][]const u8{ "markov_head.markov_w1.weight", "markov_head.markov_w2.weight" }) |key| try checks.expectTensor(try d.weight(count - 1, key), &.{ base.vocab_size, g.dspark_markov_rank }, mx.bf16);
        const half = @divExact(base.qk_rope_head_dim, 2);
        const values = try mx.allocator.alloc(f32, @intCast(half));
        defer mx.allocator.free(values);
        for (values, 0..) |*v, j| v.* = @floatFromInt(j * 2);
        const powers = try s.binary(c.mlx_divide, try s.data(values.ptr, &.{half}, mx.f32t), try s.scalar(@floatFromInt(base.qk_rope_head_dim)));
        try d.weights.put("inv", try s.binary(c.mlx_divide, try s.scalar(1), try s.binary(c.mlx_power, try s.scalar(base.rope_theta), powers)));
        return d;
    }
    fn validateLayer(d: *Draft, i: usize) !void {
        const g = d.base;
        const checks = @import("deepseek.zig").Model;
        var buf: [256]u8 = undefined;
        for ([_][]const u8{ "attn", "ffn" }) |kind| {
            try checks.expectFloat(try d.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.fn", .{kind})), &.{ 24, 4 * g.hidden_size });
            try checks.expectFloat(try d.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.base", .{kind})), &.{24});
            try checks.expectFloat(try d.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.scale", .{kind})), &.{3});
            try checks.expectTensor(try d.weight(i, try std.fmt.bufPrint(&buf, "{s}_norm.weight", .{kind})), &.{g.hidden_size}, mx.bf16);
        }
        for ([_]struct { []const u8, i32, i32 }{
            .{ "attn.wq_a", g.q_lora_rank, g.hidden_size }, .{ "attn.wq_b", g.num_attention_heads * g.head_dim, g.q_lora_rank }, .{ "attn.wkv", g.head_dim, g.hidden_size }, .{ "attn.wo_a", g.o_groups * g.o_lora_rank, @divExact(g.num_attention_heads * g.head_dim, g.o_groups) }, .{ "attn.wo_b", g.hidden_size, g.o_groups * g.o_lora_rank }, .{ "ffn.shared_experts.gate_proj", g.moe_intermediate_size, g.hidden_size }, .{ "ffn.shared_experts.up_proj", g.moe_intermediate_size, g.hidden_size }, .{ "ffn.shared_experts.down_proj", g.hidden_size, g.moe_intermediate_size },
        }) |shape| try checks.expectQ(try d.triple(i, shape[0]), shape[1], shape[2]);
        try checks.expectTensor(try d.weight(i, "attn.q_norm.weight"), &.{g.q_lora_rank}, mx.bf16);
        try checks.expectTensor(try d.weight(i, "attn.kv_norm.weight"), &.{g.head_dim}, mx.bf16);
        try checks.expectFloat(try d.weight(i, "attn.attn_sink"), &.{g.num_attention_heads});
        try checks.expectFloat(try d.weight(i, "ffn.gate.weight"), &.{ g.n_routed_experts, g.hidden_size });
        try checks.expectFloat(try d.weight(i, "ffn.gate.e_score_correction_bias"), &.{g.n_routed_experts});
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |key, j| {
            const n = if (j == 2) g.hidden_size else g.moe_intermediate_size;
            const k = if (j == 2) g.moe_intermediate_size else g.hidden_size;
            try checks.expectTensor(try d.weight(i, try std.fmt.bufPrint(&buf, "ffn.switch_mlp.{s}.weight", .{key})), &.{ g.n_routed_experts, n, @divExact(k, 8) }, c.MLX_UINT32);
            try checks.expectTensor(try d.weight(i, try std.fmt.bufPrint(&buf, "ffn.switch_mlp.{s}.scales", .{key})), &.{ g.n_routed_experts, n, @divExact(k, 32) }, c.MLX_UINT8);
        }
    }
    pub fn deinit(d: *Draft) void {
        d.reset();
        mx.allocator.free(d.keys);
        d.weights.deinit();
        d.config.deinit();
        d.activations.deinit();
    }
    pub fn reset(d: *Draft) void {
        for (d.keys) |*key| {
            mx.free(key.*);
            key.* = mx.empty;
        }
        d.position = 0;
    }
    fn weight(d: *Draft, i: usize, field: []const u8) !A {
        var buf: [256]u8 = undefined;
        return d.weights.get(try std.fmt.bufPrint(&buf, "dspark.{d}.{s}", .{ i, field }));
    }
    fn triple(d: *Draft, i: usize, field: []const u8) ![3]A {
        var buf: [256]u8 = undefined;
        return d.weights.triple(try std.fmt.bufPrint(&buf, "dspark.{d}.{s}", .{ i, field }));
    }
    fn put(d: *Draft, i: usize, field: []const u8, value: A) !void {
        var buf: [256]u8 = undefined;
        try d.weights.put(try std.fmt.bufPrint(&buf, "dspark.{d}.{s}", .{ i, field }), value);
    }
    fn stack(d: *Draft, s: *mx.Scope, i: usize, dest: []const u8, members: []const []const u8) !void {
        var buf: [256]u8 = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }) |field| {
            var values: [2]A = undefined;
            for (members, &values) |member, *value| value.* = try d.weight(i, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ member, field }));
            try d.put(i, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ dest, field }), try s.cat(&values, 0));
        }
    }
    fn qmm(s: *mx.Scope, x: A, w: [3]A) !A {
        var out = c.mlx_array_new();
        const rc = c.mlx_quantized_matmul(&out, x, w[0], w[1], w[2], true, mx.opt(64), mx.opt(4), "affine", mx.stream);
        return s.result(rc, out);
    }
    fn project(d: *Draft, s: *mx.Scope, i: usize, key: []const u8, x: A) !A {
        return qmm(s, x, try d.triple(i, key));
    }
    fn positions(d: *Draft, s: *mx.Scope, rows: i32) !A {
        const ids = try mx.allocator.alloc(i32, @intCast(rows));
        defer mx.allocator.free(ids);
        for (ids, 0..) |*id, i| id.* = d.position + @as(i32, @intCast(i));
        return s.ints(ids);
    }
    pub fn absorb(d: *Draft, kernels: *mx.Kernels, taps: A) !void {
        const g = d.base;
        if (mx.shape(taps).len != 2 or mx.dim(taps, 1) != @as(i32, @intCast(d.config.value.dspark_target_layer_ids.len)) * g.hidden_size or mx.dtype(taps) != mx.bf16) return error.InvalidDraftTaps;
        const rows = mx.dim(taps, 0);
        if (rows < 1 or rows > 65536 or d.position > 1048576 - rows) return error.ContextLimitExceeded;
        var s = mx.Scope{};
        defer s.deinit();
        const x = try cp.norm(&s, try d.project(&s, 0, "main_proj", taps), try d.weight(0, "main_norm.weight"), g.rms_norm_eps);
        const at = try d.positions(&s, rows);
        const next = try mx.allocator.alloc(A, d.keys.len);
        defer mx.allocator.free(next);
        @memset(next, mx.empty);
        errdefer for (next) |a| mx.free(a);
        for (d.keys, next, 0..) |old, *key, i| {
            const kv = try ops.normRope(kernels, &s, try d.project(&s, i, "attn.wkv", x), try d.weight(i, "attn.kv_norm.weight"), at, try d.weights.get("inv"), try s.scalar(g.rms_norm_eps), true, false);
            var all = if (old.ctx == null) kv else try s.cat(&.{ old, kv }, 0);
            if (mx.dim(all, 0) > g.sliding_window) all = try s.contiguous(try s.slice(all, 0, mx.dim(all, 0) - g.sliding_window, mx.dim(all, 0)));
            key.* = try mx.retain(all);
        }
        try mx.evalMany(next, false);
        for (d.keys, next) |*old, new| {
            mx.free(old.*);
            old.* = new;
        }
        d.position += rows;
    }
    fn hc(d: *Draft, kernels: *mx.Kernels, s: *mx.Scope, i: usize, kind: []const u8, x: A) ![3]A {
        const g = d.base;
        const rows = mx.dim(x, 0);
        var buf: [256]u8 = undefined;
        const z = try cp.norm(s, try s.reshape(try s.cast(x, mx.f32t), &.{ rows, 4 * g.hidden_size }), mx.empty, g.rms_norm_eps);
        const fnw = try s.cast(try d.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.fn", .{kind})), mx.f32t);
        const mix = try s.binary(c.mlx_matmul, z, try s.transpose(fnw, &.{ 1, 0 }));
        const scale = try s.cast(try d.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.scale", .{kind})), mx.f32t);
        const base = try s.cast(try d.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.base", .{kind})), mx.f32t);
        const out = try kernels.run(s, src.glm_hc_split, &.{ x, mix, scale, base }, &.{ mx.td("T", mx.bf16), mx.ti("HC", 4), mx.ti("ITERS", g.hc_sinkhorn_iters), mx.ti("D", g.hidden_size), mx.ti("EPS_INT", @intFromFloat(@round(g.hc_eps / 1e-9))) }, .{ 256 * rows, 1, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ rows, g.hidden_size } }, .{ .shape = &.{ rows, 4 }, .dtype = mx.f32t }, .{ .shape = &.{ rows, 4, 4 }, .dtype = mx.f32t } });
        return .{ out[0], out[1], out[2] };
    }
    fn expand(s: *mx.Scope, x: A, branch: A, post: A, comb: A) !A {
        const rows = mx.dim(x, 0);
        const a = try s.binary(c.mlx_multiply, try s.reshape(post, &.{ rows, 4, 1 }), try s.reshape(try s.cast(branch, mx.f32t), &.{ rows, 1, -1 }));
        const b = try s.binary(c.mlx_matmul, try s.transpose(comb, &.{ 0, 2, 1 }), try s.cast(x, mx.f32t));
        return s.cast(try s.binary(c.mlx_add, a, b), mx.bf16);
    }
    fn attention(d: *Draft, kernels: *mx.Kernels, s: *mx.Scope, i: usize, x: A) !A {
        const g = d.base;
        const rows = mx.dim(x, 0);
        const at = try d.positions(s, rows);
        const inv = try d.weights.get("inv");
        const eps = try s.scalar(g.rms_norm_eps);
        const xp = try d.project(s, i, "attn.x_proj", x);
        const qr = try cp.norm(s, try s.slice(xp, 1, 0, g.q_lora_rank), try d.weight(i, "attn.q_norm.weight"), g.rms_norm_eps);
        const kv = try ops.normRope(kernels, s, try s.slice(xp, 1, g.q_lora_rank, g.q_lora_rank + g.head_dim), try d.weight(i, "attn.kv_norm.weight"), at, inv, eps, true, false);
        const q = try ops.normRope(kernels, s, try s.reshape(try d.project(s, i, "attn.wq_b", qr), &.{ rows, g.num_attention_heads, g.head_dim }), null, at, inv, eps, true, false);
        const keys = if (d.keys[i].ctx == null) kv else try s.cat(&.{ d.keys[i], kv }, 0);
        const key4 = try s.reshape(keys, &.{ 1, 1, mx.dim(keys, 0), g.head_dim });
        var out = c.mlx_array_new();
        const rc = c.mlx_fast_scaled_dot_product_attention(&out, try s.reshape(try s.transpose(q, &.{ 1, 0, 2 }), &.{ 1, g.num_attention_heads, rows, g.head_dim }), key4, key4, 1 / @sqrt(@as(f32, @floatFromInt(g.head_dim))), "", mx.empty, try s.cast(try d.weight(i, "attn.attn_sink"), mx.bf16), false, mx.stream);
        out = try s.result(rc, out);
        const roped = try ops.normRope(kernels, s, try s.transpose(try s.reshape(out, &.{ g.num_attention_heads, rows, g.head_dim }), &.{ 1, 0, 2 }), null, at, inv, eps, false, true);
        const grouped = try s.reshape(roped, &.{ rows, g.o_groups, -1 });
        const parts = try mx.allocator.alloc(A, @intCast(g.o_groups));
        defer mx.allocator.free(parts);
        const w = try d.triple(i, "attn.wo_a");
        for (parts, 0..) |*part, j| {
            const group: i32 = @intCast(j);
            var weights: [3]A = undefined;
            for (&weights, w) |*dst, tensor| dst.* = try s.slice(tensor, 0, group * g.o_lora_rank, (group + 1) * g.o_lora_rank);
            part.* = try qmm(s, try s.reshape(try s.slice(grouped, 1, group, group + 1), &.{ rows, -1 }), weights);
        }
        return d.project(s, i, "attn.wo_b", try s.cat(parts, 1));
    }
    fn swiglu(d: *Draft, s: *mx.Scope, gate_: A, up_: A) !A {
        var gate = gate_;
        var up = up_;
        if (d.base.swiglu_limit != 0) {
            const limit = try s.cast(try s.scalar(d.base.swiglu_limit), mx.bf16);
            gate = try s.binary(c.mlx_minimum, gate, limit);
            up = try s.binary(c.mlx_minimum, try s.binary(c.mlx_maximum, up, try s.unary(c.mlx_negative, limit)), limit);
        }
        return s.binary(c.mlx_multiply, try d.activations.call(s, .silu, &.{gate}), up);
    }
    fn expert(d: *Draft, s: *mx.Scope, i: usize, key: []const u8, x: A, ids: A, sorted: bool) !A {
        var buf: [256]u8 = undefined;
        const w = try d.weight(i, try std.fmt.bufPrint(&buf, "ffn.switch_mlp.{s}.weight", .{key}));
        const scales = try d.weight(i, try std.fmt.bufPrint(&buf, "ffn.switch_mlp.{s}.scales", .{key}));
        var out = c.mlx_array_new();
        const rc = c.mlx_gather_qmm(&out, x, w, scales, mx.empty, mx.empty, ids, true, mx.opt(32), mx.opt(4), "mxfp4", sorted, mx.stream);
        return s.result(rc, out);
    }
    fn moe(d: *Draft, s: *mx.Scope, i: usize, x: A) !A {
        const g = d.base;
        const rows = mx.dim(x, 0);
        const top = g.num_experts_per_tok;
        const scores = try s.unary(c.mlx_sqrt, try s.binary(c.mlx_logaddexp, try s.binary(c.mlx_matmul, try s.cast(x, mx.f32t), try d.weight(i, "router")), try s.scalar(0)));
        const biased = try s.binary(c.mlx_add, scores, try s.cast(try d.weight(i, "ffn.gate.e_score_correction_bias"), mx.f32t));
        var partition = c.mlx_array_new();
        const rc = c.mlx_argpartition_axis(&partition, try s.unary(c.mlx_negative, biased), top - 1, -1, mx.stream);
        partition = try s.result(rc, partition);
        var ids = c.mlx_array_new();
        const sort_rc = c.mlx_sort_axis(&ids, try s.slice(partition, 1, 0, top), -1, mx.stream);
        ids = try s.result(sort_rc, ids);
        var weights = c.mlx_array_new();
        const take_rc = c.mlx_take_along_axis(&weights, scores, ids, -1, mx.stream);
        weights = try s.result(take_rc, weights);
        var total = try s.slice(weights, 1, 0, 1);
        var j: i32 = 1;
        while (j < top) : (j += 1) total = try s.binary(c.mlx_add, total, try s.slice(weights, 1, j, j + 1));
        weights = try s.binary(c.mlx_multiply, try s.binary(c.mlx_divide, weights, try s.binary(c.mlx_add, total, try s.scalar(1e-20))), try s.scalar(g.routed_scaling_factor));
        const sorted = rows * top >= 64;
        var input = try s.reshape(x, &.{ rows, 1, 1, g.hidden_size });
        var expert_ids = ids;
        var inverse = mx.empty;
        if (sorted) {
            const flat = try s.reshape(ids, &.{rows * top});
            const order = try s.unary(c.mlx_argsort, flat);
            inverse = try s.unary(c.mlx_argsort, order);
            const row_ids = try s.binary(c.mlx_floor_divide, order, try s.cast(try s.ints(&.{top}), mx.dtype(order)));
            input = try s.take(try s.reshape(x, &.{ rows, 1, g.hidden_size }), row_ids, 0);
            expert_ids = try s.take(flat, order, 0);
        }
        const act = try d.swiglu(s, try d.expert(s, i, "gate_proj", input, expert_ids, sorted), try d.expert(s, i, "up_proj", input, expert_ids, sorted));
        var expert_out = try d.expert(s, i, "down_proj", act, expert_ids, sorted);
        if (sorted) expert_out = try s.take(expert_out, inverse, 0);
        const y = try s.reshape(try s.cast(expert_out, mx.f32t), &.{ rows, top, g.hidden_size });
        var acc = try s.binary(c.mlx_multiply, try s.slice(weights, 1, 0, 1), try s.reshape(try s.slice(y, 1, 0, 1), &.{ rows, g.hidden_size }));
        j = 1;
        while (j < top) : (j += 1) acc = try s.binary(c.mlx_add, acc, try s.binary(c.mlx_multiply, try s.slice(weights, 1, j, j + 1), try s.reshape(try s.slice(y, 1, j, j + 1), &.{ rows, g.hidden_size })));
        const gu = try d.project(s, i, "ffn.shared_gate_up", x);
        const width = g.moe_intermediate_size;
        const shared = try d.project(s, i, "ffn.shared_experts.down_proj", try d.swiglu(s, try s.slice(gu, 1, 0, width), try s.slice(gu, 1, width, 2 * width)));
        return s.binary(c.mlx_add, try s.cast(acc, mx.bf16), shared);
    }
    pub fn logits(d: *Draft, model: anytype, s: *mx.Scope, first: i32) !A {
        const g = d.base;
        const rows = d.config.value.dspark_block_size;
        var ids: [16]i32 = @splat(d.config.value.dspark_noise_token_id);
        ids[0] = first;
        const h = try model.weights.embed(s, "model.embed_tokens", ids[0..@intCast(rows)]);
        var x = try s.stack(&.{ h, h, h, h }, 1);
        for (0..d.keys.len) |i| {
            const a = try d.hc(&model.kernels, s, i, "attn", x);
            const ax = try cp.norm(s, a[0], try d.weight(i, "attn_norm.weight"), g.rms_norm_eps);
            x = try expand(s, x, try d.attention(&model.kernels, s, i, ax), a[1], a[2]);
            const f = try d.hc(&model.kernels, s, i, "ffn", x);
            const fx = try cp.norm(s, f[0], try d.weight(i, "ffn_norm.weight"), g.rms_norm_eps);
            x = try expand(s, x, try d.moe(s, i, fx), f[1], f[2]);
        }
        const last = d.keys.len - 1;
        const head = try @import("prefill_ops.zig").uncompiled(s, .deepseek_head, &.{ x, try d.weight(last, "hc_head.fn"), try d.weight(last, "hc_head.base"), try d.weight(last, "hc_head.scale"), try s.scalar(g.rms_norm_eps), try s.scalar(g.hc_eps) });
        const normed = try cp.norm(s, head, try d.weight(last, "norm.weight"), g.rms_norm_eps);
        return model.dispatch.apply(&model.kernels, s, normed, .{ .weights = try model.weights.triple("lm_head") });
    }
    pub fn draw(d: *Draft, model: anytype, first: i32, output: []i32, settings: @import("sampling.zig").Sampling) !void {
        if (first < 0 or first >= d.base.vocab_size or output.len > d.config.value.dspark_block_size) return error.InvalidDraftBudget;
        if (output.len == 0) return;
        var s = mx.Scope{};
        defer s.deinit();
        const logits_ = try d.logits(model, &s, first);
        const last = d.keys.len - 1;
        const markov_in = try d.weight(last, "markov_head.markov_w1.weight");
        const markov_out = try s.transpose(try d.weight(last, "markov_head.markov_w2.weight"), &.{ 1, 0 });
        var previous = first;
        for (output, 0..) |*token, j| {
            const bias = try s.binary(c.mlx_matmul, try s.take(markov_in, try s.ints(&.{previous}), 0), markov_out);
            const logit = try s.binary(c.mlx_add, try s.cast(try s.slice(logits_, 0, @intCast(j), @intCast(j + 1)), mx.f32t), try s.cast(bias, mx.f32t));
            const sampled = try @import("sampling.zig").rows(&model.kernels, &s, logit, &.{model.position + @as(i32, @intCast(j)) + 1}, settings);
            defer mx.allocator.free(sampled);
            token.* = sampled[0];
            previous = token.*;
        }
    }
};

test "DSpark rejects invalid blocks, noise tokens, tap order and Markov rank" {
    const valid = DraftConfig{ .dspark_block_size = 4, .dspark_noise_token_id = 250, .dspark_target_layer_ids = &.{ 2, 3, 4 }, .dspark_markov_rank = 16 };
    try valid.validate(5, 256);
    for ([_]i32{ 0, 17 }) |size| {
        var bad = valid;
        bad.dspark_block_size = size;
        try std.testing.expectError(error.InvalidDraftConfig, bad.validate(5, 256));
    }
    for ([_][]const usize{ &.{}, &.{ 2, 2 }, &.{ 3, 2 }, &.{5} }) |taps| {
        var bad = valid;
        bad.dspark_target_layer_ids = taps;
        try std.testing.expectError(error.InvalidDraftConfig, bad.validate(5, 256));
    }
    var bad = valid;
    bad.dspark_noise_token_id = 256;
    try std.testing.expectError(error.InvalidDraftConfig, bad.validate(5, 256));
    bad = valid;
    bad.dspark_markov_rank = 0;
    try std.testing.expectError(error.InvalidDraftConfig, bad.validate(5, 256));
}
