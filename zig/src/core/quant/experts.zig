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

/// The shared expert's own projections, at its own (bits, group): gate and up joined along N, then its down.
pub const Separate = struct { gate_up: mlx.Host, down: mlx.Host };

/// A layer's experts, shared last in `fused`/`down`, or beside them in `separate` at another width (then E, not E+1).
pub const Experts = struct {
    fused: quant.Host,
    gated: bool,
    down: quant.Host,
    /// The expert MLP's inner width (rows of the gate, and of the up).
    width: usize,
    /// The model width (rows of the down).
    dims: usize,
    /// E + 1 when the shared expert is in the stack, E when it runs beside it.
    count: usize,
    /// The shared expert at its own width; null when it stacks in `fused` and `down`.
    separate: ?Separate = null,
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

/// The routed stack's and the shared expert's tensor for one suffix; `same` says they share (bits, group).
const Pair = struct { mine: Tensor, one: Tensor };

fn pair(t: anytype, stack_key: []const u8, shared: []const u8, suffix: []const u8, same: bool) !Pair {
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
    const inner = one.rank -| 1;
    if (mine.rank != one.rank + 1 or !std.mem.eql(usize, mine.shape[1 .. 1 + inner], one.shape[0..inner]))
        return refuse(stack_key, "does not stack with the shared expert");
    if (same and mine.shape[one.rank] != one.shape[inner])
        return refuse(stack_key, "does not stack with the shared expert");
    return .{ .mine = mine, .one = one };
}

/// Whether the shared expert of `key` packs at the stack's width `w`, so its tensors stack into it.
fn sameWidth(t: anytype, key: []const u8, w: mlx.Width) !bool {
    const one = try t.width(key);
    return one.bits == w.bits and one.group == w.group;
}

/// `srcs` stacked per expert along N; `with_shared` appends the shared expert as expert E, else routed experts only.
fn stack(a: Allocator, srcs: []const Pair, dtype: DType, comptime with_shared: bool) !Tensor {
    const count = if (with_shared)
        std.math.add(usize, srcs[0].mine.shape[0], 1) catch return error.UnexpectedTensor
    else
        srcs[0].mine.shape[0];
    const rest = srcs[0].mine.shape[2];
    var rows: usize = 0;
    for (srcs) |s| rows = std.math.add(usize, rows, s.mine.shape[1]) catch return error.UnexpectedTensor;
    _ = std.math.cast(u32, rows) orelse return error.UnexpectedTensor;
    const size = dtype.size();
    const values = try mlx.geometry.product(try mlx.geometry.product(count, rows), rest);
    const out = try a.alloc(u8, try mlx.geometry.product(values, size));
    var at: usize = 0;
    for (0..count) |e| for (srcs) |s| {
        const n = s.mine.shape[1] * rest;
        const shared_row = with_shared and e + 1 == count;
        const tensor = if (!shared_row) s.mine else s.one;
        const src = if (!shared_row)
            tensor.bytes[e * n * tensor.dtype.size() ..][0 .. n * tensor.dtype.size()]
        else
            tensor.bytes;
        if (tensor.dtype == dtype) {
            @memcpy(out[at..][0..src.len], src);
        } else try convert.widenInto(out[at..][0 .. n * size], .{
            .dtype = tensor.dtype,
            .rank = 1,
            .shape = .{ n, 1, 1, 1, 1 },
            .bytes = src,
        });
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

fn part(t: anytype, stack_key: []const u8, shared: []const u8, w: mlx.Width, same: bool) !Part {
    const p: Part = .{
        .words = try pair(t, stack_key, shared, ".weight", same),
        .scales = try pair(t, stack_key, shared, ".scales", same),
        .biases = try pair(t, stack_key, shared, ".biases", same),
    };
    _ = mlx.geometry.shape(p.words.mine, p.scales.mine, p.biases.mine, w.bits, w.group, true) catch
        return refuse(stack_key, "packed stack has incompatible or unrepresentable geometry");
    return p;
}

/// Input columns of an MLX host: its group tables' columns times the group.
fn cols(h: mlx.Host) usize {
    return h.scales.shape[1] * h.group;
}

/// The routed stack's input columns: its gate/up tables' columns times the stack's group.
fn colsStack(p: Part, w: mlx.Width) usize {
    return p.scales.mine.shape[2] * w.group;
}

/// A projection's (rows, columns) as the fallback's preflight reads them from the group tables.
const Side = struct { rows: usize, columns: usize };

/// One affine side as the fallback checks it itself: (rows, columns) of a device-readable packed projection.
fn sideShape(t: anytype, key: []const u8) ?Side {
    var b: [256]u8 = undefined;
    var o: [256]u8 = undefined;
    const w = t.width(key) catch return null;
    const words = t.get(join(&b, &.{ key, ".weight" })) catch return null;
    const scales = t.get(join(&b, &.{ key, ".scales" })) catch return null;
    const biases = t.get(join(&o, &.{ key, ".biases" })) catch return null;
    if (mlx.geometry.shape(words, scales, biases, w.bits, w.group, false)) |_| {} else |_| return null;
    return .{ .rows = words.shape[0], .columns = scales.shape[1] * w.group };
}

/// The shared expert at its own widths; a side no format reads, or widths off the model's, refuses unloaded.
fn separate(a: Allocator, t: anytype, keys: []const []const u8, width: usize, dims: usize) !Separate {
    var sides: [3]?Side = undefined;
    for (keys, 0..) |key, i| sides[i] = sideShape(t, key);
    for (sides) |maybe| if (maybe == null) return error.UnexpectedTensor;
    const g = sides[1].?;
    const u = sides[0].?;
    const d = sides[2].?;
    // gate and up one shape and the model's inputs; the down back from its gate's rows to the model width
    if (g.rows != u.rows or g.columns != u.columns or g.columns != width) return error.UnexpectedTensor;
    if (d.rows != dims or d.columns != g.rows) return error.UnexpectedTensor;
    const gate = try mlx.read(a, t, keys[1]);
    const up = try mlx.read(a, t, keys[0]);
    const down = try mlx.read(a, t, keys[2]);
    if (gate.bits != up.bits or gate.group != up.group) return refuse(keys[1], "shared expert's gate and up differ");
    return .{ .gate_up = try joinHost(a, gate, up), .down = down };
}

/// Two rank-2 hosts of one width joined along N (gate rows first, then up), as the experts' fused side is.
fn joinHost(a: Allocator, gate: mlx.Host, up: mlx.Host) !mlx.Host {
    return .{
        .words = try stackHosts(a, gate.words, up.words),
        .scales = try stackHosts(a, gate.scales, up.scales),
        .biases = try stackHosts(a, gate.biases, up.biases),
        .bits = gate.bits,
        .group = gate.group,
    };
}

/// Two rank-2 tensors of the same row length concatenated along N.
fn stackHosts(a: Allocator, top: Tensor, under: Tensor) !Tensor {
    if (top.dtype != under.dtype or top.rank != 2 or under.rank != 2 or top.shape[1] != under.shape[1])
        return error.UnexpectedTensor;
    const bytes = try a.alloc(u8, top.bytes.len + under.bytes.len);
    @memcpy(bytes[0..top.bytes.len], top.bytes);
    @memcpy(bytes[top.bytes.len..], under.bytes);
    return .{
        .dtype = top.dtype,
        .rank = 2,
        .shape = .{ top.shape[0] + under.shape[0], top.shape[1], 1, 1, 1 },
        .bytes = bytes,
    };
}

/// The stacked host tensors of `parts`: the routed experts, and the shared one last when `with_shared` is set.
fn side(a: Allocator, parts: []const Part, w: mlx.Width, comptime with_shared: bool) !quant.Host {
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
        .words = try stack(a, words[0..n], .i32, with_shared),
        .scales = try stack(a, scales[0..n], dtype, with_shared),
        .biases = try stack(a, biases[0..n], dtype, with_shared),
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
    // one width throughout stacks the shared expert in as expert E; another leaves it beside the stack
    var joined = try sameWidth(t, shared_keys[0], up_w);
    if (joined and gated) joined = try sameWidth(t, shared_keys[1], up_w);
    if (joined) joined = try sameWidth(t, shared_keys[2], down_w);
    const up = try part(t, routed_keys[0], shared_keys[0], up_w, joined);
    const down = try part(t, routed_keys[2], shared_keys[2], down_w, joined);
    var fused_parts: [2]Part = .{ up, up };
    var n: usize = 1;
    if (gated) {
        fused_parts[0] = try part(t, routed_keys[1], shared_keys[1], up_w, joined);
        n = 2;
        if (!std.mem.eql(usize, fused_parts[0].words.mine.shape[0..3], up.words.mine.shape[0..3]))
            return refuse(mine, "gate and up differ in shape");
    }
    if (!joined) return .{
        .fused = try side(a, fused_parts[0..n], up_w, false),
        .gated = gated,
        .down = try side(a, &.{down}, down_w, false),
        .width = up.words.mine.shape[1],
        .dims = down.words.mine.shape[1],
        .count = up.words.mine.shape[0],
        .separate = try separate(a, t, shared_keys[0..3], colsStack(up, up_w), down.words.mine.shape[1]),
    };
    const fused = try side(a, fused_parts[0..n], up_w, true);
    const d = try side(a, &.{down}, down_w, true);
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
    if (gate.cols != shared_gate.cols or rows != spec.experts + 1 or gate.cols != spec.hidden)
        return refuse(prefix, "router rows do not fit E + 1 by hidden");
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
    // a shared expert beside the stack has no router row of its own to count against E + 1: the gate row is its own
    if (ex.count != rows and !(ex.count + 1 == rows and ex.separate != null))
        return refuse(prefix, "router and experts disagree on E + 1");
    return .{
        .router = .{ .dtype = .bf16, .rank = 2, .shape = .{ rows, gate.cols, 1, 1, 1 }, .bytes = router },
        .rows32 = .{ .dtype = .f32, .rank = 2, .shape = .{ rows, gate.cols, 1, 1, 1 }, .bytes = rows32 },
        .experts = ex,
        .top_k = spec.top_k,
    };
}

/// A rank's contiguous routed experts, the shared one on rank 0, and the remap to its own ids (-1: another rank's).
pub fn sliceRank(a: Allocator, x: *Routed, r: slice.Rank) slice.Error!void {
    // a shared expert beside the stack is not part of the stack's count, so the stack holds E routed experts there
    const total = x.experts.count - @intFromBool(x.experts.separate == null);
    const part_n = try slice.even(total, r.world, "experts");
    x.experts.fused = try quant.sliceStack(a, x.experts.fused, r, part_n);
    x.experts.down = try quant.sliceStack(a, x.experts.down, r, part_n);
    x.experts.count = part_n + @intFromBool(r.rank == 0 and x.experts.separate == null);
    // the shared expert beside the stack belongs to rank 0 alone, where it is run beside that rank's launch
    if (x.experts.separate) |s| x.experts.separate = if (r.rank == 0) s else null;
    const remap = try a.alloc(i32, total + 1);
    @memset(remap, -1);
    for (0..part_n) |i| remap[r.rank * part_n + i] = @intCast(i);
    // the shared expert is this rank's last stacked expert only when it stacks; beside the stack it is -1 (no item)
    if (r.rank == 0 and x.experts.separate == null) remap[total] = @intCast(part_n);
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
                .{
                    .dtype = .bf16,
                    .rank = 2,
                    .shape = .{ 2, 1, 1, 1, 1 },
                    .bytes = std.mem.sliceAsBytes(shared_tables),
                });
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

/// A table whose projections answer their own (bits, group), so a mixed-width checkpoint can be described.
const Mixed = struct {
    map: std.StringHashMapUnmanaged(Tensor) = .empty,
    widths: std.StringHashMapUnmanaged(mlx.Width) = .empty,

    pub fn has(m: *const Mixed, key: []const u8) bool {
        return m.map.contains(key);
    }

    pub fn get(m: *const Mixed, key: []const u8) !Tensor {
        return m.map.get(key) orelse error.MissingTensor;
    }

    pub fn width(m: *const Mixed, key: []const u8) !mlx.Width {
        return m.widths.get(key) orelse .{ .bits = 4, .group = 32 };
    }
};

/// One affine projection of `rows` x `columns` at `w` from `seed`; `stacked` gives the routed stack shapes.
fn affineInto(
    a: Allocator,
    f: *Mixed,
    key: []const u8,
    rows: usize,
    columns: usize,
    w: mlx.Width,
    seed: usize,
    stacked: bool,
) !void {
    const words_cols = columns * w.bits / 32;
    const words = try a.alloc(u32, rows * words_cols);
    for (words, 0..) |*x, i| x.* = @intCast(seed + i);
    // a stacked projection is (E, N, K words), a lone one (N, K words); its tables add K / group
    const rank: u8 = 2 + @as(u8, @intFromBool(stacked));
    const shape = [5]usize{ if (stacked) 2 else rows, if (stacked) rows / 2 else words_cols, words_cols, 1, 1 };
    const name = try std.fmt.allocPrint(a, "{s}.weight", .{key});
    try f.map.put(a, name, wordTensor(rank, shape, std.mem.sliceAsBytes(words)));
    const tables = try a.alloc(u16, rows * (columns / w.group));
    @memset(tables, 0x3f80);
    const tshape = [5]usize{ shape[0], if (stacked) rows / 2 else columns / w.group, columns / w.group, 1, 1 };
    for ([_][]const u8{ "scales", "biases" }) |suffix|
        try f.map.put(a, try std.fmt.allocPrint(a, "{s}.{s}", .{ key, suffix }), .{ .dtype = .bf16, .rank = rank, .shape = tshape, .bytes = std.mem.sliceAsBytes(tables) });
    if (w.bits != 4 or w.group != 32) try f.widths.put(a, key, w);
}

/// A packed int32 tensor of `shape`'s first three dimensions over `bytes`.
fn wordTensor(rank: u8, shape: [5]usize, bytes: []const u8) Tensor {
    return .{ .dtype = .u32, .rank = rank, .shape = shape, .bytes = bytes };
}

/// Two routed experts of one width, stacked (2, 64, ...): gate and up of 64 inputs, down of the hidden 64.
fn routedInto(a: Allocator, f: *Mixed, w: mlx.Width) !void {
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |name, i| {
        try affineInto(a, f, try std.fmt.allocPrint(a, "mlp.switch_mlp.{s}", .{name}), 2 * 64, 64, w, 1000 * i, true);
    }
}

/// The router's E rows and the shared gate's row, both over the model's width of 64.
fn routerInto(a: Allocator, f: *Mixed) !void {
    const g = try a.alloc(f32, 2 * 64);
    for (g, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 64));
    try f.map.put(a, "mlp.gate.weight", .{ .dtype = .f32, .rank = 2, .shape = .{ 2, 64, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(g) });
    const s = try a.alloc(f32, 64);
    @memset(s, 0.5);
    try f.map.put(a, "mlp.shared_expert_gate.weight", .{ .dtype = .f32, .rank = 2, .shape = .{ 1, 64, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(s) });
}

// A layer whose shared expert is 8-bit g64 over 4-bit g32 routed experts: it keeps its own width beside the stack.
test "a shared expert of another width than the routed stack keeps its own and runs beside it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Mixed = .{};
    try routedInto(a, &f, .{ .bits = 4, .group = 32 });
    const wide: mlx.Width = .{ .bits = 8, .group = 64 };
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |name, i| {
        const key = try std.fmt.allocPrint(a, "mlp.shared_expert.{s}", .{name});
        try affineInto(a, &f, key, 64, 64, wide, 500 + 2000 * i, false);
    }
    try routerInto(a, &f);
    const ex = try experts(a, &f, "mlp.");
    // the stack holds the two routed experts alone, at their own width; the shared one is separate, at 8-bit g64
    try std.testing.expectEqual(@as(usize, 2), ex.count);
    try std.testing.expectEqual(@as(u8, 4), ex.fused.mlx.bits);
    try std.testing.expectEqual(@as(u16, 32), ex.down.mlx.group);
    const s = ex.separate orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u8, 8), s.gate_up.bits);
    try std.testing.expectEqual(@as(u16, 64), s.gate_up.group);
    try std.testing.expectEqual(@as(u8, 8), s.down.bits);
    // gate then up along N: 128 rows of 16 words, the gate's last row and the up's first back to back
    var join_at: [2]u32 = undefined;
    @memcpy(std.mem.sliceAsBytes(&join_at), s.gate_up.words.bytes[4 * (16 * 64 - 1) ..][0..8]);
    try std.testing.expectEqualSlices(u32, &.{ 1523, 2500 }, &join_at);
    try std.testing.expectEqual(@as(usize, 128 * 16 * 4), s.gate_up.words.bytes.len);
    try std.testing.expectEqual(@as(usize, 64 * 16 * 4), s.down.words.bytes.len);
    // the routed stack holds gate and up of experts 0 and 1 only: 2 * 128 rows of 8 words at 4-bit, no shared row
    try std.testing.expectEqual(@as(usize, 2 * (2 * 64) * 8 * 4), ex.fused.mlx.words.bytes.len);
    try std.testing.expectEqual(@as(u32, 1000 + 1023), lastU32(ex.fused.mlx.words));
    try std.testing.expectEqual(@as(u8, 4), ex.down.mlx.bits);
    // rank 0 keeps the shared expert beside its own, rank 1 has none; the router's rows still count E + 1
    var r = try routed(a, &f, "mlp.", .{ .experts = 2, .hidden = 64, .top_k = 1 });
    try sliceRank(a, &r, .{ .rank = 1, .world = 2 });
    try std.testing.expectEqual(@as(usize, 1), r.experts.count);
    try std.testing.expectEqual(@as(?Separate, null), r.experts.separate);
    var remap: [3]i32 = undefined;
    @memcpy(std.mem.sliceAsBytes(&remap), r.remap.?.bytes);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, -1 }, &remap);
    var first: Routed = try routed(a, &f, "mlp.", .{ .experts = 2, .hidden = 64, .top_k = 1 });
    try sliceRank(a, &first, .{ .rank = 0, .world = 2 });
    try std.testing.expect(first.experts.separate != null);
}

/// The last u32 of a host's packed words.
fn lastU32(t: Tensor) u32 {
    var v: u32 = undefined;
    @memcpy(std.mem.asBytes(&v), t.bytes[t.bytes.len - 4 ..]);
    return v;
}

// A shared expert at a width the device formats cannot represent (7-bit codes) is still refused, not run beside.
test "a shared expert at a width no format represents still refuses" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Mixed = .{};
    try routedInto(a, &f, .{ .bits = 4, .group = 32 });
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |name| {
        const key = try std.fmt.allocPrint(a, "mlp.shared_expert.{s}", .{name});
        try affineInto(a, &f, key, 64, 64, .{ .bits = 7, .group = 64 }, 500, false);
    }
    try routerInto(a, &f);
    try std.testing.expectError(error.UnexpectedTensor, experts(a, &f, "mlp."));
}

// A shared expert whose input width differs from the model's is refused too: its rows cannot multiply `x`.
test "a shared expert of the wrong shape still refuses" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Mixed = .{};
    try routedInto(a, &f, .{ .bits = 4, .group = 32 });
    const narrow: mlx.Width = .{ .bits = 8, .group = 32 };
    for ([_][]const u8{ "gate_proj", "up_proj" }) |name| {
        const key = try std.fmt.allocPrint(a, "mlp.shared_expert.{s}", .{name});
        try affineInto(a, &f, key, 64, 32, narrow, 500, false);
    }
    try affineInto(a, &f, "mlp.shared_expert.down_proj", 64, 64, narrow, 900, false);
    try routerInto(a, &f);
    try std.testing.expectError(error.UnexpectedTensor, experts(a, &f, "mlp."));
}
