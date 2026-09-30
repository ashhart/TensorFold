const std = @import("std");
const mx = @import("mlx.zig");
const c = mx.c;
const A = mx.Array;
const Weight = @import("flash_ops.zig").Weight;
const mm = @import("flash_prefill_ops.zig").matmul;
const Ops = @import("prefill_ops.zig").Ops;
pub const Weights = struct { key: Weight, value: Weight, key_scale: A, query_scale: A, conv_scale: A, conv: A };
pub const Result = struct { branch: A, tail: A, gated: A, normed: A };

fn norm(s: *mx.Scope, x: A, scale: A, streams: i32) !A {
    const batch = mx.dim(x, 0);
    const rows = mx.dim(x, 1);
    const dims = @divExact(mx.dim(x, 2), streams);
    return s.reshape(try @import("flash_prefill_attention.zig").norm(s, try s.reshape(x, &.{ batch, rows, streams, dims }), try s.reshape(scale, &.{ streams, dims }), 1e-6), mx.shape(x));
}
pub fn forward(ops: *Ops, s: *mx.Scope, h: A, embedding: A, w: Weights, previous: A, streams: i32, dilation: i32) !Result {
    if (mx.shape(h).len != 3 or mx.shape(embedding).len != 3 or mx.dtype(h) != mx.bf16 or mx.dtype(embedding) != mx.bf16 or streams < 1 or dilation < 1) return error.InvalidTensorShape;
    const batch = mx.dim(h, 0);
    const rows = mx.dim(h, 1);
    const wide = mx.dim(h, 2);
    if (batch < 1 or rows < 1 or rows > 2048 or wide < 1 or @mod(wide, streams) != 0 or mx.dim(embedding, 0) != batch or mx.dim(embedding, 1) != rows or mx.shape(w.conv).len != 3 or mx.dim(w.conv, 0) != wide or mx.dim(w.conv, 1) < 1 or mx.dim(w.conv, 2) != 1 or mx.dtype(w.conv) != mx.bf16) return error.InvalidTensorShape;
    const dims = @divExact(wide, streams);
    const tail = (mx.dim(w.conv, 1) - 1) * dilation;
    for ([_]A{ w.key_scale, w.query_scale, w.conv_scale }) |scale| if (mx.c.mlx_array_size(scale) != wide or mx.dtype(scale) != mx.f32t) return error.InvalidTensorShape;
    if (previous.ctx != null and (!std.mem.eql(i32, mx.shape(previous), &.{ batch, tail, wide }) or mx.dtype(previous) != mx.bf16)) return error.InvalidTensorShape;
    const keys = try s.reshape(try norm(s, try mm(s, embedding, w.key), w.key_scale, streams), &.{ batch, rows, streams, dims });
    const values = try mm(s, embedding, w.value);
    const queries = try s.reshape(try norm(s, h, w.query_scale, streams), mx.shape(keys));
    var gate = c.mlx_array_new();
    const gr = c.mlx_sum_axis(&gate, try s.binary(c.mlx_multiply, keys, queries), -1, true, mx.stream);
    gate = try s.binary(c.mlx_divide, try s.result(gr, gate), try s.cast(try s.scalar(@sqrt(@as(f32, @floatFromInt(dims)))), mx.bf16));
    const sign = try s.unary(c.mlx_sign, gate);
    const root = try s.unary(c.mlx_sqrt, try s.binary(c.mlx_maximum, try s.unary(c.mlx_abs, gate), try s.cast(try s.scalar(1e-6), mx.bf16)));
    gate = try s.binary(c.mlx_multiply, sign, root);
    const gated = try s.reshape(try s.binary(c.mlx_multiply, try s.unary(c.mlx_sigmoid, gate), try s.reshape(values, &.{ batch, rows, 1, dims })), mx.shape(h));
    const normed = try norm(s, gated, w.conv_scale, streams);
    const seq = try s.cat(&.{ if (previous.ctx != null) previous else try s.zeros(&.{ batch, tail, wide }, mx.bf16), normed }, 1);
    const next_tail = try s.contiguous(try s.slice(seq, 1, rows, rows + tail));
    var conv = c.mlx_array_new();
    const cr = c.mlx_conv1d(&conv, seq, w.conv, 1, 0, dilation, wide, mx.stream);
    const branch = try s.binary(c.mlx_add, gated, try ops.call(s, .silu, &.{try s.result(cr, conv)}));
    return .{ .branch = branch, .tail = next_tail, .gated = gated, .normed = normed };
}

pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var ops = Ops{};
    defer ops.deinit();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/ple.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, weights: []const u8, streams: i32, dilation: i32, cached: bool, bits: [2]i32, groups: [2]i32 };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    if (cases.value.len == 0) return error.EmptyFixtures;
    for (cases.value) |case| {
        errdefer std.debug.print("Flash prefill PLE fixture failed: {s}\n", .{case.name});
        var store = @import("checkpoint.zig").Store.init(32);
        defer store.deinit();
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.weights }), "", "");
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
        var s = mx.Scope{};
        defer s.deinit();
        var w: Weights = undefined;
        inline for (.{ "key", "value" }, 0..) |key, i| @field(w, key) = .{ .arrays = .{ try store.get(key ++ ".weight"), try store.get(key ++ ".scales"), try store.get(key ++ ".biases") }, .format = .{ .bits = case.bits[i], .group_size = case.groups[i] } };
        inline for (.{ "key_scale", "query_scale", "conv_scale", "conv" }) |key| @field(w, key) = try store.get(key);
        const result = try forward(&ops, &s, try store.get("h"), try store.get("embedding"), w, if (case.cached) try store.get("previous") else mx.empty, case.streams, case.dilation);
        inline for (.{ "branch", "tail", "gated", "normed" }) |key| {
            errdefer std.debug.print("Mismatch in {s}\n", .{key});
            try @import("variant_checks.zig").equalBits(&s, @field(result, key), try store.get(key));
        }
    }
    std.debug.print("PASS: {d} Flash prefill PLE projection/gating/convolution cases and retained-tail continuation\n", .{cases.value.len});
}
