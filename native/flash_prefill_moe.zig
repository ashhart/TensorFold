const std = @import("std");
const mx = @import("mlx.zig");
const A = mx.Array;
const c = mx.c;
const Weight = @import("flash_ops.zig").Weight;
const mm = @import("flash_prefill_ops.zig").matmul;
const Ops = @import("prefill_ops.zig").Ops;

pub const Weights = struct {
    router: A,
    gate: Weight,
    up: Weight,
    down: Weight,
    shared_gate: Weight,
    shared_up: Weight,
    shared_down: Weight,
    shared_route: Weight,
};
pub const Result = struct { output: A, ids: A, weights: A, routed: A, shared: A };

fn gather(s: *mx.Scope, x: A, w: Weight, ids: A, sorted: bool) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_gather_qmm(&out, x, w.arrays[0], w.arrays[1], w.arrays[2], mx.empty, ids, true, mx.opt(w.format.group_size), mx.opt(w.format.bits), "affine", sorted, mx.stream);
    return s.result(rc, out);
}

fn expert(s: *mx.Scope, x: A, w: Weight, ids: A, sorted: bool, bounded: bool) !A {
    if (!bounded or mx.dim(x, 0) <= 32768) return gather(s, x, w, ids, sorted);
    // Upstream balances slices to avoid MLX's 16-bit sorted-gather row offsets.
    const rows = mx.dim(x, 0);
    const chunks = @divTrunc(rows + 32767, 32768);
    const size = @divTrunc(rows + chunks - 1, chunks);
    var parts: std.ArrayList(A) = .empty;
    defer parts.deinit(mx.allocator);
    var start: i32 = 0;
    while (start < rows) : (start += size) try parts.append(mx.allocator, try gather(s, try s.slice(x, 0, start, @min(start + size, rows)), w, try s.slice(ids, 0, start, @min(start + size, rows)), true));
    return s.cat(parts.items, 0);
}

pub fn forward(ops: *Ops, s: *mx.Scope, x: A, w: Weights, top: i32) !Result {
    if (mx.shape(x).len != 3 or mx.dtype(x) != mx.bf16 or mx.shape(w.router).len != 2 or mx.dtype(w.router) != mx.bf16) return error.InvalidTensorShape;
    const batch = mx.dim(x, 0);
    const rows = mx.dim(x, 1);
    const dims = mx.dim(x, 2);
    const experts = mx.dim(w.router, 0);
    if (batch < 1 or rows < 1 or rows > 2048 or top < 1 or top > experts or mx.dim(w.router, 1) != dims) return error.InvalidTensorShape;
    const g = try w.gate.geometry(3);
    const u = try w.up.geometry(3);
    const d = try w.down.geometry(3);
    if (g.k != dims or !std.meta.eql(g, u) or d.k != g.n or d.n != dims) return error.InvalidTensorShape;
    for ([_]Weight{ w.gate, w.up, w.down }) |weight| if (mx.dim(weight.arrays[0], 0) != experts) return error.InvalidTensorShape;
    const sg = try w.shared_gate.geometry(2);
    const su = try w.shared_up.geometry(2);
    const sd = try w.shared_down.geometry(2);
    const sr = try w.shared_route.geometry(2);
    if (sg.k != dims or !std.meta.eql(sg, su) or sd.k != sg.n or sd.n != dims or sr.k != dims or sr.n != 1) return error.InvalidTensorShape;
    const logits = try s.binary(c.mlx_matmul, x, try s.transpose(w.router, &.{ 1, 0 }));
    var probs = c.mlx_array_new();
    const pr = c.mlx_softmax_axis(&probs, logits, -1, true, mx.stream);
    probs = try s.result(pr, probs);
    var ids = c.mlx_array_new();
    const ir = c.mlx_argpartition_axis(&ids, probs, -top, -1, mx.stream);
    ids = try s.slice(try s.result(ir, ids), 2, experts - top, experts);
    var weights = c.mlx_array_new();
    const wr = c.mlx_take_along_axis(&weights, probs, ids, -1, mx.stream);
    weights = try s.result(wr, weights);
    var sum = c.mlx_array_new();
    const sr_sum = c.mlx_sum_axis(&sum, weights, -1, true, mx.stream);
    weights = try s.binary(c.mlx_divide, weights, try s.result(sr_sum, sum));
    const optimized = batch == 1 and rows >= 64 and w.gate.format.bits == 4 and w.up.format.bits == 4 and w.down.format.bits == 4 and w.gate.format.group_size == 32 and w.up.format.group_size == 32 and w.down.format.group_size == 32;
    const sorted = batch * rows * top >= 64;
    var input = try s.reshape(x, &.{ batch, rows, 1, 1, dims });
    var indices = ids;
    var inverse = mx.empty;
    if (sorted) {
        const flat = try s.reshape(ids, &.{-1});
        const order = try s.unary(c.mlx_argsort, flat);
        inverse = try s.unary(c.mlx_argsort, order);
        const row_ids = try s.binary(c.mlx_floor_divide, order, try s.cast(try s.ints(&.{top}), mx.dtype(order)));
        input = try s.take(try s.reshape(x, &.{ batch * rows, 1, dims }), row_ids, 0);
        indices = try s.take(flat, order, 0);
    }
    const gate = try expert(s, input, w.gate, indices, sorted, optimized);
    const up = try expert(s, input, w.up, indices, sorted, optimized);
    const act = try ops.call(s, .swiglu, &.{ gate, up });
    var routed = try expert(s, act, w.down, indices, sorted, optimized);
    if (sorted) routed = try s.take(routed, inverse, 0);
    routed = try s.reshape(routed, &.{ batch, rows, top, dims });
    const weighted = try s.binary(c.mlx_multiply, routed, try s.reshape(weights, &.{ batch, rows, top, 1 }));
    var reduced = c.mlx_array_new();
    const rr = c.mlx_sum_axis(&reduced, weighted, -2, false, mx.stream);
    reduced = try s.result(rr, reduced);
    const sx = if (optimized) try s.reshape(x, &.{ rows, dims }) else x;
    const shared_gate = try ops.call(s, .silu, &.{try mm(s, sx, w.shared_gate)});
    const shared_act = try s.binary(c.mlx_multiply, shared_gate, try mm(s, sx, w.shared_up));
    const shared = try s.reshape(try s.binary(c.mlx_multiply, try mm(s, shared_act, w.shared_down), try s.unary(c.mlx_sigmoid, try mm(s, sx, w.shared_route))), &.{ batch, rows, dims });
    return .{ .output = try s.binary(c.mlx_add, reduced, shared), .ids = ids, .weights = weights, .routed = reduced, .shared = shared };
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var ops = Ops{};
    defer ops.deinit();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/moe.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, weights: []const u8, top: i32, bits: [7]i32, groups: [7]i32 };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    if (cases.value.len == 0) return error.EmptyFixtures;
    for (cases.value) |case| {
        errdefer std.debug.print("Flash prefill MoE fixture failed: {s}\n", .{case.name});
        var store = @import("checkpoint.zig").Store.init(32);
        defer store.deinit();
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.weights }), "", "");
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
        var s = mx.Scope{};
        defer s.deinit();
        var w: Weights = undefined;
        w.router = try store.get("router");
        inline for (.{ "gate", "up", "down", "shared_gate", "shared_up", "shared_down", "shared_route" }, 0..) |key, i| @field(w, key) = .{ .arrays = .{ try store.get(key ++ ".weight"), try store.get(key ++ ".scales"), try store.get(key ++ ".biases") }, .format = .{ .bits = case.bits[i], .group_size = case.groups[i] } };
        const result = try forward(&ops, &s, try store.get("input"), w, case.top);
        inline for (.{ "output", "ids", "weights", "routed", "shared" }) |key| {
            errdefer std.debug.print("Mismatch in {s}\n", .{key});
            try @import("variant_checks.zig").equalBits(&s, @field(result, key), try store.get(key));
        }
    }
    std.debug.print("PASS: {d} Flash prefill MoE layers, exact routing and BF16 expert reductions, shared gates and bounded sorted gathers\n", .{cases.value.len});
}
