//! A MoE layer's router rows and its affine experts stacked with the shared expert last, any table with has/get/width.

const std = @import("std");
const types = @import("types.zig");
const convert = @import("convert.zig");
const mlx = @import("mlx.zig");
const slice = @import("slice.zig");
const quant = @import("quant.zig");

const Tensor = types.Tensor;
const DType = types.DType;
const Allocator = std.mem.Allocator;
const refuse = types.refuse;
const join = types.join;

/// A layer's E + 1 experts, shared last: `fused` is gate then up along N per expert, or up alone when not `gated`.
pub const Experts = struct {
    fused: quant.Host,
    gated: bool,
    down: quant.Host,
    /// The expert MLP's inner width (rows of the gate, and of the up).
    width: usize,
    /// The model width (rows of the down).
    dims: usize,
    /// E + 1.
    count: usize,
};

/// The router rows [E + 1, D] with the shared gate last, in bf16 and widened to `rows32`; `remap` is a rank's own ids.
pub const Routed = struct { router: Tensor, rows32: Tensor, experts: Experts, top_k: usize, remap: ?Tensor = null };

/// What the router's rows are checked against: the routed experts, the model width, the experts a token takes.
pub const Shape = struct { experts: usize, hidden: usize, top_k: usize };

/// `_router_rows`: one router's rows [R, D] as fp32, a float tensor as stored or an affine group table unpacked.
fn routerRows(a: Allocator, t: anytype, key: []const u8) !struct { data: []f32, rows: usize, cols: usize } {
    var buf: [256]u8 = undefined;
    var other: [256]u8 = undefined;
    const weight = try t.get(join(&buf, &.{ key, ".weight" }));
    if (convert.isFloat(weight.dtype)) {
        if (weight.rank != 2) return refuse(key, "router weight must be (rows, hidden)");
        const wide = try convert.float32(a, weight);
        const out = try a.alloc(f32, weight.numel());
        for (out, 0..) |*o, i| o.* = convert.load(.f32, wide.bytes, i);
        return .{ .data = out, .rows = weight.shape[0], .cols = weight.shape[1] };
    }
    const group = (try t.width(key)).group;
    const scale = try t.get(join(&buf, &.{ key, ".scales" }));
    const bias = try t.get(join(&other, &.{ key, ".biases" }));
    const data = try mlx.dequant(a, weight, scale, bias, group);
    return .{ .data = data, .rows = weight.shape[0], .cols = scale.shape[1] * group };
}

/// The routed stack and the shared expert's tensor for one suffix, as `_shared_last` checks them.
const Pair = struct { mine: Tensor, one: Tensor };

fn pair(t: anytype, stack_key: []const u8, shared: []const u8, suffix: []const u8) !Pair {
    var b1: [256]u8 = undefined;
    var b2: [256]u8 = undefined;
    var mine = try t.get(join(&b1, &.{ stack_key, suffix }));
    var one = try t.get(join(&b2, &.{ shared, suffix }));
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
fn stack(a: Allocator, srcs: []const Pair, dtype: DType) !Tensor {
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

fn part(t: anytype, stack_key: []const u8, shared: []const u8, w: mlx.Width) !Part {
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

fn side(a: Allocator, parts: []const Part, w: mlx.Width) !quant.Host {
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
pub fn experts(a: Allocator, t: anytype, prefix: []const u8) !Experts {
    var b: [4][256]u8 = undefined;
    const mine = join(&b[0], &.{ prefix, "switch_mlp." });
    const shared = join(&b[1], &.{ prefix, "shared_expert." });
    if (!t.has(join(&b[2], &.{ mine, "up_proj.weight" }))) {
        if (t.has(join(&b[2], &.{ mine, "up_proj.qweight" }))) return error.UnsupportedQuantization;
        _ = try t.get(join(&b[2], &.{ mine, "up_proj.weight" }));
    }
    const gated = t.has(join(&b[2], &.{ mine, "gate_proj.weight" }));
    var names: [3][2][256]u8 = undefined;
    const keys = [3][]const u8{ "up_proj", "gate_proj", "down_proj" };
    var routed_keys: [3][]const u8 = undefined;
    var shared_keys: [3][]const u8 = undefined;
    for (keys, 0..) |k, i| {
        routed_keys[i] = join(&names[i][0], &.{ mine, k });
        shared_keys[i] = join(&names[i][1], &.{ shared, k });
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
pub fn routed(a: Allocator, t: anytype, prefix: []const u8, spec: Shape) !Routed {
    var b: [2][256]u8 = undefined;
    const gate = try routerRows(a, t, join(&b[0], &.{ prefix, "gate" }));
    const shared_gate = try routerRows(a, t, join(&b[1], &.{ prefix, "shared_expert_gate" }));
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

/// A rank's contiguous routed experts, the shared one on rank 0, and the remap to its own ids (-1: another rank's).
pub fn sliceRank(a: Allocator, x: *Routed, r: slice.Rank) slice.Error!void {
    const total = x.experts.count - 1;
    const part_n = try slice.even(total, r.world, "experts");
    x.experts.fused = try quant.sliceStack(a, x.experts.fused, r, part_n);
    x.experts.down = try quant.sliceStack(a, x.experts.down, r, part_n);
    x.experts.count = part_n + @intFromBool(r.rank == 0);
    const remap = try a.alloc(i32, total + 1);
    @memset(remap, -1);
    for (0..part_n) |i| remap[r.rank * part_n + i] = @intCast(i);
    if (r.rank == 0) remap[total] = @intCast(part_n);
    x.remap = .{ .dtype = .i32, .rank = 1, .shape = .{ total + 1, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(remap) };
}

const Fake = struct {
    map: std.StringHashMapUnmanaged(Tensor) = .empty,

    pub fn has(f: *const Fake, key: []const u8) bool {
        return f.map.contains(key);
    }

    pub fn get(f: *const Fake, key: []const u8) !Tensor {
        return f.map.get(key) orelse error.MissingTensor;
    }

    pub fn width(_: *const Fake, _: []const u8) !mlx.Width {
        return .{ .bits = 4, .group = 32 };
    }
};

test "two routed experts and the shared one stack in order, and a rank keeps its own with the remap" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{};
    // expert e's word w of row n is 100 e + 10 n + w; the shared expert's are 900 + 10 n + w
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |name| {
        const words = try a.alloc(u32, 2 * 2 * 4);
        for (words, 0..) |*x, i| x.* = @intCast(100 * (i / 8) + 10 * ((i / 4) % 2) + i % 4);
        const one = try a.alloc(u32, 2 * 4);
        for (one, 0..) |*x, i| x.* = @intCast(900 + 10 * (i / 4) + i % 4);
        const tables = try a.alloc(u16, 2 * 2);
        @memset(tables, 0x3f80);
        const shared_tables = try a.alloc(u16, 2);
        @memset(shared_tables, 0x3f80);
        for ([_][]const u8{ "weight", "scales", "biases" }) |suffix| {
            const stacked = std.mem.eql(u8, suffix, "weight");
            try f.map.put(a, try std.fmt.allocPrint(a, "mlp.switch_mlp.{s}.{s}", .{ name, suffix }), if (stacked)
                .{ .dtype = .u32, .rank = 3, .shape = .{ 2, 2, 4, 1, 1 }, .bytes = std.mem.sliceAsBytes(words) }
            else
                .{ .dtype = .bf16, .rank = 3, .shape = .{ 2, 2, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(tables) });
            try f.map.put(a, try std.fmt.allocPrint(a, "mlp.shared_expert.{s}.{s}", .{ name, suffix }), if (stacked)
                .{ .dtype = .u32, .rank = 2, .shape = .{ 2, 4, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(one) }
            else
                .{ .dtype = .bf16, .rank = 2, .shape = .{ 2, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(shared_tables) });
        }
    }
    const ex = try experts(a, &f, "mlp.");
    try std.testing.expect(ex.gated);
    try std.testing.expectEqual(@as(usize, 3), ex.count);
    var fused: [48]u32 = undefined;
    @memcpy(std.mem.sliceAsBytes(&fused), ex.fused.mlx.words.bytes);
    // expert 1: its gate rows, then its up rows; the shared expert last
    try std.testing.expectEqualSlices(u32, &.{ 100, 101, 102, 103, 110, 111, 112, 113, 100, 101 }, fused[16..26]);
    try std.testing.expectEqual(@as(u32, 900), fused[32]);
    var r: Routed = .{ .router = undefined, .rows32 = undefined, .experts = ex, .top_k = 1 };
    try sliceRank(a, &r, .{ .rank = 1, .world = 2 });
    try std.testing.expectEqual(@as(usize, 1), r.experts.count);
    var remap: [3]i32 = undefined;
    @memcpy(std.mem.sliceAsBytes(&remap), r.remap.?.bytes);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, -1 }, &remap);
}
