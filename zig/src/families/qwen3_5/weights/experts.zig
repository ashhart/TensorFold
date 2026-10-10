//! The Python loader's `_routed`, `_experts` and `_affine_side`: router rows and the stacked affine experts.

const std = @import("std");
const quant = @import("core").quant;
const table = @import("table.zig");
const convert = quant.convert;
const config = @import("config.zig");
const host = @import("host.zig");
const projection = @import("projection.zig");

const Tensor = table.Tensor;
const Table = table.Table;
const DType = table.DType;
pub const Error = projection.Error;

fn refuse(key: []const u8, what: []const u8) error{UnexpectedTensor} {
    std.log.err("{s}: {s}", .{ key, what });
    return error.UnexpectedTensor;
}

/// `_router_rows`: one router's rows [R, D] as fp32, a float tensor as stored or an affine group table unpacked.
fn routerRows(a: std.mem.Allocator, t: *const Table, key: []const u8) Error!struct { data: []f32, rows: usize, cols: usize } {
    var buf: [256]u8 = undefined;
    var other: [256]u8 = undefined;
    const weight = try t.get(projection.join(&buf, &.{ key, ".weight" }));
    if (convert.isFloat(weight.dtype)) {
        if (weight.rank != 2) return refuse(key, "router weight must be (rows, hidden)");
        const wide = try convert.float32(a, weight);
        const out = try a.alloc(f32, weight.numel());
        for (out, 0..) |*o, i| o.* = convert.load(.f32, wide.bytes, i);
        return .{ .data = out, .rows = weight.shape[0], .cols = weight.shape[1] };
    }
    const group = (try t.width(key)).group;
    const scale = try t.get(projection.join(&buf, &.{ key, ".scales" }));
    const bias = try t.get(projection.join(&other, &.{ key, ".biases" }));
    const data = try quant.mlx.dequant(a, weight, scale, bias, group);
    return .{ .data = data, .rows = weight.shape[0], .cols = scale.shape[1] * group };
}

/// The routed stack and the shared expert's tensor for one suffix, as `_shared_last` checks them.
const Pair = struct { mine: Tensor, one: Tensor };

fn pair(t: *const Table, stack_key: []const u8, shared: []const u8, suffix: []const u8) Error!Pair {
    var b1: [256]u8 = undefined;
    var b2: [256]u8 = undefined;
    var mine = try t.get(projection.join(&b1, &.{ stack_key, suffix }));
    var one = try t.get(projection.join(&b2, &.{ shared, suffix }));
    if (mine.dtype == .u32) mine.dtype = .i32;
    if (one.dtype == .u32) one.dtype = .i32;
    if (one.rank == mine.rank and one.shape[0] == 1) {
        one.rank -= 1;
        one.shape = .{ one.shape[1], one.shape[2], one.shape[3], one.shape[4], 1 };
    }
    if (mine.rank != one.rank + 1 or !std.mem.eql(usize, mine.shape[1..mine.rank], one.shape[0..one.rank])) return refuse(stack_key, "does not stack with the shared expert");
    return .{ .mine = mine, .one = one };
}

/// `fused` words, scales and biases of `srcs` stacked per expert, each expert's slices side by side along N.
fn stack(a: std.mem.Allocator, srcs: []const Pair, dtype: DType) Error!Tensor {
    const count = srcs[0].mine.shape[0] + 1;
    const rest = srcs[0].mine.shape[2];
    var rows: usize = 0;
    for (srcs) |s| rows += s.mine.shape[1];
    const size = dtype.size();
    const out = try a.alloc(u8, count * rows * rest * size);
    var at: usize = 0;
    for (0..count) |e| for (srcs) |s| {
        const n = s.mine.shape[1] * rest;
        const src = if (e + 1 < count) s.mine.bytes[e * n * s.mine.dtype.size() ..][0 .. n * s.mine.dtype.size()] else s.one.bytes;
        if (s.mine.dtype == dtype) {
            @memcpy(out[at..][0..src.len], src);
        } else for (0..n) |i| {
            std.mem.writeInt(u32, out[at + 4 * i ..][0..4], @bitCast(convert.load(s.mine.dtype, src, i)), .little);
        }
        at += n * size;
    };
    return .{ .dtype = dtype, .rank = 3, .shape = .{ count, rows, rest, 1, 1 }, .bytes = out };
}

/// The dtype the group tables of `srcs` share, or fp32 when they differ or the kernels do not read them as stored.
fn tableDtype(srcs: []const Pair, other: []const Pair) DType {
    const first = srcs[0].mine.dtype;
    for ([_][]const Pair{ srcs, other }) |list| for (list) |s| {
        if (s.mine.dtype != first or s.one.dtype != first) return .f32;
    };
    return if (convert.tableDtype(first)) first else .f32;
}

/// One projection's tensors, checked as `_affine_side` does against (bits, group).
const Part = struct { words: Pair, scales: Pair, biases: Pair };

fn part(t: *const Table, stack_key: []const u8, shared: []const u8, w: config.Width) Error!Part {
    const p: Part = .{
        .words = try pair(t, stack_key, shared, ".weight"),
        .scales = try pair(t, stack_key, shared, ".scales"),
        .biases = try pair(t, stack_key, shared, ".biases"),
    };
    const words, const scale, const bias = .{ p.words.mine, p.scales.mine, p.biases.mine };
    if (words.dtype != .i32 or scale.rank != 3 or !std.mem.eql(usize, scale.shape[0..3], bias.shape[0..3])) return refuse(stack_key, "not an MLX affine expert stack");
    const k = scale.shape[2] * w.group;
    if (words.shape[2] != k * w.bits / 32 or !std.mem.eql(usize, words.shape[0..2], scale.shape[0..2])) return refuse(stack_key, "packed stack does not fit K, bits and group");
    return p;
}

fn side(a: std.mem.Allocator, parts: []const Part, w: config.Width) Error!host.Projection {
    var words: [2]Pair = undefined;
    var scales: [2]Pair = undefined;
    var biases: [2]Pair = undefined;
    for (parts, 0..) |p, i| {
        words[i] = p.words;
        scales[i] = p.scales;
        biases[i] = p.biases;
    }
    const n = parts.len;
    const dtype = tableDtype(scales[0..n], biases[0..n]);
    return .{ .mlx = .{
        .words = try stack(a, words[0..n], .i32),
        .scales = try stack(a, scales[0..n], dtype),
        .biases = try stack(a, biases[0..n], dtype),
        .bits = w.bits,
        .group = w.group,
    } };
}

/// `_experts`: a layer's E + 1 affine experts, the shared one last, gate and up fused per expert.
pub fn experts(a: std.mem.Allocator, t: *const Table, prefix: []const u8) Error!host.Experts {
    var b: [4][256]u8 = undefined;
    const mine = projection.join(&b[0], &.{ prefix, "switch_mlp." });
    const shared = projection.join(&b[1], &.{ prefix, "shared_expert." });
    if (!t.has(projection.join(&b[2], &.{ mine, "up_proj.weight" }))) {
        if (t.has(projection.join(&b[2], &.{ mine, "up_proj.qweight" }))) return error.UnsupportedQuantization;
        _ = try t.get(projection.join(&b[2], &.{ mine, "up_proj.weight" }));
    }
    const gated = t.has(projection.join(&b[2], &.{ mine, "gate_proj.weight" }));
    var names: [3][2][256]u8 = undefined;
    const keys = [3][]const u8{ "up_proj", "gate_proj", "down_proj" };
    var routed_keys: [3][]const u8 = undefined;
    var shared_keys: [3][]const u8 = undefined;
    for (keys, 0..) |k, i| {
        routed_keys[i] = projection.join(&names[i][0], &.{ mine, k });
        shared_keys[i] = projection.join(&names[i][1], &.{ shared, k });
    }
    const up_w = try t.width(routed_keys[0]);
    const down_w = try t.width(routed_keys[2]);
    if (gated and !std.meta.eql(try t.width(routed_keys[1]), up_w)) return refuse(mine, "gate and up differ in width");
    const up = try part(t, routed_keys[0], shared_keys[0], up_w);
    const down = try part(t, routed_keys[2], shared_keys[2], down_w);
    var fused_parts: [2]Part = undefined;
    var n: usize = 0;
    if (gated) {
        fused_parts[0] = try part(t, routed_keys[1], shared_keys[1], up_w);
        n = 1;
        if (!std.mem.eql(usize, fused_parts[0].words.mine.shape[0..3], up.words.mine.shape[0..3])) return refuse(mine, "gate and up differ in shape");
    }
    fused_parts[n] = up;
    n += 1;
    const fused = try side(a, fused_parts[0..n], up_w);
    const d = try side(a, &.{down}, down_w);
    return .{
        .fused = fused,
        .gated = gated,
        .down = d,
        .width = up.words.mine.shape[1],
        .dims = down.words.mine.shape[1],
        .count = up.words.mine.shape[0] + 1,
    };
}

/// `_routed`: router rows [E + 1, D] in bf16 (the shared expert's gate row last), widened once, and the experts.
pub fn routed(a: std.mem.Allocator, t: *const Table, prefix: []const u8, spec: config.Spec) Error!host.Routed {
    var b: [2][256]u8 = undefined;
    const gate = try routerRows(a, t, projection.join(&b[0], &.{ prefix, "gate" }));
    const shared_gate = try routerRows(a, t, projection.join(&b[1], &.{ prefix, "shared_expert_gate" }));
    const rows = gate.rows + shared_gate.rows;
    if (gate.cols != shared_gate.cols or rows != spec.experts + 1 or gate.cols != spec.hidden) return refuse(prefix, "router rows do not fit E + 1 by hidden");
    const router = try a.alloc(u8, rows * gate.cols * 2);
    const rows32 = try a.alloc(u8, rows * gate.cols * 4);
    var at: usize = 0;
    for ([_][]f32{ gate.data, shared_gate.data }) |part_rows| for (part_rows) |v| {
        const half = convert.bf16(v);
        std.mem.writeInt(u16, router[2 * at ..][0..2], half, .little);
        std.mem.writeInt(u32, rows32[4 * at ..][0..4], @as(u32, half) << 16, .little);
        at += 1;
    };
    const ex = try experts(a, t, prefix);
    if (ex.count != rows) return refuse(prefix, "router and experts disagree on E + 1");
    return .{
        .router = .{ .dtype = .bf16, .rank = 2, .shape = .{ rows, gate.cols, 1, 1, 1 }, .bytes = router },
        .rows32 = .{ .dtype = .f32, .rank = 2, .shape = .{ rows, gate.cols, 1, 1, 1 }, .bytes = rows32 },
        .experts = ex,
        .top_k = spec.top_k,
    };
}
