const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const Model = @import("deepseek.zig").Model;
const c = mx.c;
const A = mx.Array;

pub const Result = struct { logits: A, scores: A, ids: A, weights: A, experts: A, routed: A, shared: A, output: A };

fn expert(m: *Model, s: *mx.Scope, layer: usize, key: []const u8, x: A, ids: A, sorted: bool) !A {
    var b: [256]u8 = undefined;
    const w = try m.weight(layer, try std.fmt.bufPrint(&b, "ffn.switch_mlp.{s}.weight", .{key}));
    const scales = try m.weight(layer, try std.fmt.bufPrint(&b, "ffn.switch_mlp.{s}.scales", .{key}));
    var out = c.mlx_array_new();
    const rc = c.mlx_gather_qmm(&out, x, w, scales, mx.empty, mx.empty, ids, true, mx.opt(32), mx.opt(4), "mxfp4", sorted, mx.stream);
    return s.result(rc, out);
}

fn activate(m: *Model, s: *mx.Scope, gate: A, up: A) !A {
    const limit = m.config.value.swiglu_limit;
    var g = gate;
    var u = up;
    if (limit != 0) {
        const bound = try s.cast(try s.scalar(limit), mx.bf16);
        g = try s.binary(c.mlx_minimum, g, bound);
        u = try s.binary(c.mlx_minimum, try s.binary(c.mlx_maximum, u, try s.unary(c.mlx_negative, bound)), bound);
    }
    return s.binary(c.mlx_multiply, try m.activations.call(s, .silu, &.{g}), u);
}

pub fn forward(m: *Model, s: *mx.Scope, layer: usize, x: A, tokens: []const i32) !Result {
    const g = m.config.value;
    if (mx.shape(x).len != 2 or mx.dtype(x) != mx.bf16 or mx.dim(x, 1) != g.hidden_size) return error.InvalidTensorShape;
    const rows = mx.dim(x, 0);
    if (rows < 1 or rows > 2048 or tokens.len != rows) return error.InvalidTensorShape;
    for (tokens) |token| if (token < 0 or token >= m.vocab) return error.InvalidToken;
    const top = g.num_experts_per_tok;
    const logits = try s.binary(c.mlx_matmul, try s.cast(x, mx.f32t), try m.weight(layer, "router"));
    const scores = try s.unary(c.mlx_sqrt, try s.binary(c.mlx_logaddexp, logits, try s.scalar(0)));
    var ids = c.mlx_array_new();
    if (layer < g.num_hash_layers) {
        mx.free(ids);
        ids = try s.take(try s.cast(try m.weight(layer, "ffn.gate.tid2eid"), mx.i32t), try s.ints(tokens), 0);
    } else {
        const biased = try s.binary(c.mlx_add, scores, try s.cast(try m.weight(layer, "ffn.gate.e_score_correction_bias"), mx.f32t));
        const rc = c.mlx_argpartition_axis(&ids, try s.unary(c.mlx_negative, biased), top - 1, -1, mx.stream);
        ids = try s.slice(try s.result(rc, ids), 1, 0, top);
    }
    var ascending = c.mlx_array_new();
    const sort_rc = c.mlx_sort_axis(&ascending, ids, -1, mx.stream);
    ids = try s.result(sort_rc, ascending);
    var weights = c.mlx_array_new();
    const wr = c.mlx_take_along_axis(&weights, scores, ids, -1, mx.stream);
    weights = try s.result(wr, weights);
    var sum = try s.slice(weights, 1, 0, 1);
    var j: i32 = 1;
    while (j < top) : (j += 1) sum = try s.binary(c.mlx_add, sum, try s.slice(weights, 1, j, j + 1));
    weights = try s.binary(c.mlx_multiply, try s.binary(c.mlx_divide, weights, try s.binary(c.mlx_add, sum, try s.scalar(1e-20))), try s.scalar(g.routed_scaling_factor));
    const sorted = rows * top >= 64;
    var input = try s.reshape(x, &.{ rows, 1, 1, g.hidden_size });
    var indices = ids;
    var inverse = mx.empty;
    if (sorted) {
        const flat = try s.reshape(ids, &.{-1});
        const order = try s.unary(c.mlx_argsort, flat);
        inverse = try s.unary(c.mlx_argsort, order);
        const row_ids = try s.binary(c.mlx_floor_divide, order, try s.cast(try s.ints(&.{top}), mx.dtype(order)));
        input = try s.take(try s.reshape(x, &.{ rows, 1, g.hidden_size }), row_ids, 0);
        indices = try s.take(flat, order, 0);
    }
    const gate = try expert(m, s, layer, "gate_proj", input, indices, sorted);
    const up = try expert(m, s, layer, "up_proj", input, indices, sorted);
    var y = try expert(m, s, layer, "down_proj", try activate(m, s, gate, up), indices, sorted);
    if (sorted) y = try s.take(y, inverse, 0);
    y = try s.reshape(y, &.{ rows, top, g.hidden_size });
    const yf = try s.cast(y, mx.f32t);
    var acc = try s.binary(c.mlx_multiply, try s.slice(weights, 1, 0, 1), try s.reshape(try s.slice(yf, 1, 0, 1), &.{ rows, g.hidden_size }));
    j = 1;
    while (j < top) : (j += 1) acc = try s.binary(c.mlx_add, acc, try s.binary(c.mlx_multiply, try s.slice(weights, 1, j, j + 1), try s.reshape(try s.slice(yf, 1, j, j + 1), &.{ rows, g.hidden_size })));
    const routed = try s.cast(acc, mx.bf16);
    const gu = if (rows > 16) try m.project(s, layer, "ffn.shared_gate_up", x) else try Model.stock(s, x, try m.triple(layer, "ffn.shared_gate_up"));
    const width = @divExact(mx.dim(gu, 1), 2);
    const act = try activate(m, s, try s.slice(gu, 1, 0, width), try s.slice(gu, 1, width, 2 * width));
    const shared = if (rows > 16) try m.project(s, layer, "ffn.shared_experts.down_proj", act) else try Model.stock(s, act, try m.triple(layer, "ffn.shared_experts.down_proj"));
    return .{ .logits = logits, .scores = scores, .ids = ids, .weights = weights, .experts = y, .routed = routed, .shared = shared, .output = try s.binary(c.mlx_add, routed, shared) };
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/moe.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Group = struct { checkpoint: []const u8, cases: []const []const u8 };
    const groups = try std.json.parseFromSlice([]const Group, mx.allocator, bytes, .{});
    defer groups.deinit();
    if (groups.value.len == 0) return error.EmptyFixtures;
    var count: usize = 0;
    for (groups.value) |group| {
        var m = try Model.init(io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, group.checkpoint }));
        defer m.deinit();
        if (group.cases.len == 0) return error.EmptyFixtures;
        for (group.cases) |case| {
            errdefer std.debug.print("DeepSeek prefill MoE fixture failed: {s}/{s}\n", .{ group.checkpoint, case });
            var store = cp.Store.init(32);
            defer store.deinit();
            try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case }), "", "");
            var s = mx.Scope{};
            defer s.deinit();
            const tokens = try mx.allocator.alloc(i32, @intCast(mx.dim(try store.get("input"), 0)));
            defer mx.allocator.free(tokens);
            for (tokens, 0..) |*token, i| token.* = @intCast((i * 7 + 3) % @as(usize, @intCast(m.vocab)));
            const result = try forward(&m, &s, 0, try store.get("input"), tokens);
            inline for (comptime std.meta.fieldNames(Result)) |key| {
                errdefer std.debug.print("Mismatch in {s}\n", .{key});
                try @import("variant_checks.zig").equalBits(&s, @field(result, key), try store.get(key));
            }
            count += 1;
        }
    }
    std.debug.print("PASS: {d} DeepSeek batched MoE cases, hash/score routing, sorted MXFP4 experts, clipping and shared experts\n", .{count});
}
