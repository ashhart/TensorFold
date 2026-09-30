const std = @import("std");
const mx = @import("mlx.zig");
const glm = @import("glm.zig");
const cp = @import("checkpoint.zig");
const c = mx.c;
const A = mx.Array;

pub const Result = struct { output: A, logits: A, ids: A, weights: A, experts: A, routed: A, shared: A = mx.empty };

fn expert(m: *glm.Model, s: *mx.Scope, layer: usize, key: []const u8, x: A, ids: A, sorted: bool) !A {
    var b: [256]u8 = undefined;
    const name = try glm.Model.name(&b, layer, key);
    const w = @import("flash_ops.zig").Weight{ .arrays = try m.weights.triple(name), .format = m.formats.get(name) orelse return error.UnsupportedQuantization };
    if (sorted and w.format.bits == 4 and @mod(w.format.group_size, 32) == 0 and @divTrunc(mx.dim(x, 0), mx.dim(w.arrays[0], 0)) >= 4 and try m.kernels.flash_prefill.tiles(&m.kernels)) {
        const y = try @import("flash_prefill_mm.zig").gatherSorted(&m.kernels, s, try s.reshape(x, &.{ mx.dim(x, 0), mx.dim(x, -1) }), w, try s.cast(ids, c.MLX_UINT32), null);
        return s.reshape(y, &.{ mx.dim(y, 0), 1, mx.dim(y, 1) });
    }
    var out = c.mlx_array_new();
    const rc = c.mlx_gather_qmm(&out, x, w.arrays[0], w.arrays[1], w.arrays[2], mx.empty, ids, true, mx.opt(w.format.group_size), mx.opt(w.format.bits), "affine", sorted, mx.stream);
    return s.result(rc, out);
}

pub fn forward(m: *glm.Model, s: *mx.Scope, layer: usize, x: A) !Result {
    const g = m.config.value;
    if (mx.shape(x).len != 2 or mx.dtype(x) != mx.bf16 or mx.dim(x, 1) != g.hidden_size) return error.InvalidTensorShape;
    const rows = mx.dim(x, 0);
    if (rows < 1 or rows > 2048) return error.InvalidTensorShape;
    const top = g.num_experts_per_tok;
    const logits = try s.binary(c.mlx_matmul, try s.cast(x, mx.f32t), try m.weight(layer, "mlp.router"));
    const scores = try s.unary(c.mlx_sigmoid, logits);
    var ids = c.mlx_array_new();
    const ir = c.mlx_argpartition_axis(&ids, try s.unary(c.mlx_negative, try s.binary(c.mlx_add, scores, try s.cast(try m.weight(layer, "mlp.gate.e_score_correction_bias"), mx.f32t))), top - 1, -1, mx.stream);
    ids = try s.slice(try s.result(ir, ids), 1, 0, top);
    var weights = c.mlx_array_new();
    const wr = c.mlx_take_along_axis(&weights, scores, ids, -1, mx.stream);
    weights = try s.result(wr, weights);
    if (top > 1 and g.norm_topk_prob) {
        var sum = c.mlx_array_new();
        const sr = c.mlx_sum_axis(&sum, weights, -1, true, mx.stream);
        weights = try s.binary(c.mlx_divide, weights, try s.result(sr, sum));
    }
    weights = try s.binary(c.mlx_multiply, weights, try s.scalar(g.routed_scaling_factor));
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
    const gate = try expert(m, s, layer, "mlp.switch_mlp.gate_proj", input, indices, sorted);
    const up = try expert(m, s, layer, "mlp.switch_mlp.up_proj", input, indices, sorted);
    var y = try expert(m, s, layer, "mlp.switch_mlp.down_proj", try m.activation(s, gate, up), indices, sorted);
    if (sorted) y = try s.take(y, inverse, 0);
    y = try s.reshape(y, &.{ rows, top, g.hidden_size });
    const yf = try s.cast(y, mx.f32t);
    var total = try s.binary(c.mlx_multiply, try s.slice(weights, 1, 0, 1), try s.reshape(try s.slice(yf, 1, 0, 1), &.{ rows, g.hidden_size }));
    var j: i32 = 1;
    while (j < top) : (j += 1) total = try s.binary(c.mlx_add, total, try s.binary(c.mlx_multiply, try s.slice(weights, 1, j, j + 1), try s.reshape(try s.slice(yf, 1, j, j + 1), &.{ rows, g.hidden_size })));
    const routed = try s.cast(total, mx.bf16);
    var b: [256]u8 = undefined;
    const shared = if (m.weights.has(try glm.Model.name(&b, layer, "mlp.shared_experts.gate_proj.weight"))) try m.dense(s, layer, "mlp.shared_experts", x) else mx.empty;
    return .{ .output = if (shared.ctx != null) try s.binary(c.mlx_add, routed, shared) else routed, .logits = logits, .ids = ids, .weights = weights, .experts = y, .routed = routed, .shared = shared };
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try glm.Model.prepareRuntime();
    try mx.init();
    defer mx.shutdown();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/moe.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, tiles: bool };
    const Group = struct { checkpoint: []const u8, cases: []const Case };
    const groups = try std.json.parseFromSlice([]const Group, mx.allocator, bytes, .{});
    defer groups.deinit();
    if (groups.value.len == 0) return error.EmptyFixtures;
    var count: usize = 0;
    for (groups.value) |group| {
        var m = try glm.Model.init(io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, group.checkpoint }));
        defer m.deinit();
        if (group.cases.len == 0) return error.EmptyFixtures;
        for (group.cases) |case| {
            errdefer std.debug.print("GLM prefill MoE fixture failed: {s}/{s}\n", .{ group.checkpoint, case.name });
            m.kernels.flash_prefill.decision = case.tiles;
            var store = cp.Store.init(32);
            defer store.deinit();
            try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
            var s = mx.Scope{};
            defer s.deinit();
            const result = try forward(&m, &s, 0, try store.get("input"));
            inline for (comptime std.meta.fieldNames(Result)) |key| {
                errdefer std.debug.print("Mismatch in {s}\n", .{key});
                const a = @field(result, key);
                try std.testing.expectEqual(store.has(key), a.ctx != null);
                if (a.ctx != null) try @import("variant_checks.zig").equalBits(&s, a, try store.get(key));
            }
            count += 1;
        }
    }
    std.debug.print("PASS: {d} GLM prefill MoE cases, exact routing, sorted/custom gathers, mixed quantization, clipping and shared experts\n", .{count});
}
