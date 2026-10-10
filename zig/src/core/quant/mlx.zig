//! MLX affine weights: i32 words of `bits`-wide codes, a scale and bias a group of K; host and device side.

const std = @import("std");
const types = @import("types.zig");
const slice = @import("slice.zig");
const convert = @import("convert.zig");

const Tensor = types.Tensor;
const Allocator = std.mem.Allocator;
const join = types.join;
const refuse = types.refuse;

pub const id: types.Format = .mlx;
pub const decoder: types.Decoder = .mlx;

pub const bit_widths = [_]u8{ 2, 3, 4, 5, 6, 8 };

/// One affine width: groups of `group` weights share a scale and a bias, `bits` per weight.
pub const Width = struct { bits: u8, group: u16 };

/// The config's `quantization` table: the global width and the per-tensor overrides (mixed-width conversions).
pub const Config = struct {
    global: Width,
    /// A per-tensor entry; `null` marks one this format refuses, which fails when the tensor is looked up.
    overrides: std.StringHashMapUnmanaged(?Width) = .empty,

    /// The tensor's own width when the table names it as an object, else the global one.
    pub fn width(q: Config, key: []const u8) error{UnsupportedQuantization}!Width {
        if (q.overrides.get(key)) |w| return w orelse error.UnsupportedQuantization;
        return q.global;
    }
};

/// The affine width an object of the config names, or null when it is not one the kernels run.
fn affineWidth(o: std.json.ObjectMap) ?Width {
    const bits = intOrNull(o.get("bits")) orelse return null;
    const group = intOrNull(o.get("group_size")) orelse return null;
    if (o.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return null;
    if (std.mem.indexOfScalar(i64, &.{ 2, 3, 4, 5, 6, 8 }, bits) == null) return null;
    if (std.mem.indexOfScalar(i64, &.{ 32, 64, 128 }, group) == null) return null;
    return .{ .bits = @intCast(bits), .group = @intCast(group) };
}

fn intOrNull(v: ?std.json.Value) ?i64 {
    const value = v orelse return null;
    return switch (value) {
        .integer => |i| i,
        .float => |f| if (f == @floor(f) and @abs(f) < 1e15) @as(i64, @intFromFloat(f)) else null,
        else => null,
    };
}

/// The checkpoint's `quantization` object when it is MLX affine (`mode` absent or "affine"), else null.
pub fn detect(a: Allocator, src: types.Sources) Allocator.Error!?Config {
    const obj = src.quantization;
    var q: Config = .{ .global = affineWidth(obj) orelse return null };
    var it = obj.iterator();
    while (it.next()) |kv| if (kv.value_ptr.* == .object) try q.overrides.put(a, kv.key_ptr.*, affineWidth(kv.value_ptr.object));
    return q;
}

/// An affine projection as stored: i32 words [N, K * bits / 32], tables [N, K / group]; a stack leads with E + 1.
pub const Host = struct { words: Tensor, scales: Tensor, biases: Tensor, bits: u8, group: u16 };

/// Whether `key` holds an MLX affine projection (a table of scales beside its words).
pub fn matches(t: anytype, key: []const u8) bool {
    var buf: [256]u8 = undefined;
    return t.has(join(&buf, &.{ key, ".scales" }));
}

/// The fp32 scale and bias tables when the pair is not one of the dtypes the kernels read as stored.
pub fn tables(a: Allocator, scale: Tensor, bias: Tensor) convert.Error![2]Tensor {
    if (scale.dtype == bias.dtype and convert.tableDtype(scale.dtype)) return .{ scale, bias };
    return .{ try convert.float32(a, scale), try convert.float32(a, bias) };
}

/// `_packed`: an affine projection as stored (group tables widened to fp32 only when not fp32, bf16 or fp16 pairs).
pub fn read(a: Allocator, t: anytype, key: []const u8) !Host {
    var buf: [256]u8 = undefined;
    var other: [256]u8 = undefined;
    var words = try t.get(join(&buf, &.{ key, ".weight" }));
    const w = try t.width(key);
    if ((words.dtype != .u32 and words.dtype != .i32) or words.rank != 2) return refuse(key, "weight is not packed int32 words");
    words.dtype = .i32;
    const scale = try t.get(join(&buf, &.{ key, ".scales" }));
    const bias = try t.get(join(&other, &.{ key, ".biases" }));
    if (scale.rank != 2 or bias.rank != 2 or !std.mem.eql(usize, scale.shape[0..2], bias.shape[0..2])) return refuse(key, "scale and bias must share shape (N, K / group)");
    const k = scale.shape[1] * w.group;
    if (words.shape[1] != k * w.bits / 32 or words.shape[0] != scale.shape[0]) return refuse(key, "packed shape does not match K, bits and group");
    const pair = try tables(a, scale, bias);
    return .{ .words = words, .scales = pair[0], .biases = pair[1], .bits = w.bits, .group = w.group };
}

/// Output rows N of a projection.
pub fn rows(h: Host) usize {
    return h.words.shape[0];
}

/// The bytes of the three tensors.
pub fn bytes(h: Host) usize {
    return h.words.bytes.len + h.scales.bytes.len + h.biases.bytes.len;
}

/// `_halves`: a fused [embedding | hidden] projection split in two along K, words and group tables alike.
pub fn halves(a: Allocator, p: Host) ![2]Host {
    const groups = p.scales.shape[1];
    const half = groups * p.group / 2;
    if (groups % 2 != 0 or half * p.bits % 32 != 0) return refuse("fc", "does not split on a word boundary");
    const word = p.words.shape[1] / 2;
    const tab = groups / 2;
    var out: [2]Host = undefined;
    for (&out, 0..) |*o, i| {
        const w = if (i == 0) [2]usize{ 0, word } else [2]usize{ word, p.words.shape[1] };
        const g = if (i == 0) [2]usize{ 0, tab } else [2]usize{ tab, groups };
        o.* = .{
            .words = try slice.takeCols(a, p.words, w[0], w[1]),
            .scales = try slice.takeCols(a, p.scales, g[0], g[1]),
            .biases = try slice.takeCols(a, p.biases, g[0], g[1]),
            .bits = p.bits,
            .group = p.group,
        };
    }
    return out;
}

/// `_rows`: output rows `spans` of a projection, the share of a column-split one.
pub fn sliceRows(a: Allocator, p: Host, spans: []const slice.Span) slice.Error!Host {
    return .{ .words = try slice.takeRows(a, p.words, spans), .scales = try slice.takeRows(a, p.scales, spans), .biases = try slice.takeRows(a, p.biases, spans), .bits = p.bits, .group = p.group };
}

/// `_cols`: one rank's whole input groups of a row-split projection, whose fp32 outputs the ranks sum.
pub fn sliceCols(a: Allocator, p: Host, r: slice.Rank, what: []const u8) slice.Error!Host {
    const groups = try slice.even(p.scales.shape[1], r.world, what);
    const words = groups * p.group * p.bits / 32;
    return .{
        .words = try slice.takeCols(a, p.words, r.rank * words, (r.rank + 1) * words),
        .scales = try slice.takeCols(a, p.scales, r.rank * groups, (r.rank + 1) * groups),
        .biases = try slice.takeCols(a, p.biases, r.rank * groups, (r.rank + 1) * groups),
        .bits = p.bits,
        .group = p.group,
    };
}

/// A stack of E + 1 experts cut to a rank's `part` of them (the shared one last, kept on rank 0).
pub fn sliceStack(a: Allocator, p: Host, r: slice.Rank, part: usize) slice.Error!Host {
    return .{ .words = try slice.takeExperts(a, p.words, r, part), .scales = try slice.takeExperts(a, p.scales, r, part), .biases = try slice.takeExperts(a, p.biases, r, part), .bits = p.bits, .group = p.group };
}

/// An uploaded projection (or stack of them): the same tensors on the device.
pub const Device = struct { words: types.Buf, scales: types.Buf, biases: types.Buf, bits: u8, group: u16 };

/// The checkpoint's words and tables on the device as stored: this format keeps its layout.
pub fn upload(u: types.Uploader, h: Host) !Device {
    return .{ .words = try u.tensor(h.words), .scales = try u.tensor(h.scales), .biases = try u.tensor(h.biases), .bits = h.bits, .group = h.group };
}

/// What the kernels read of an MLX projection: the three addresses, the tables' type and the width.
pub const Matrix = struct { words: u64, scale: u64, bias: u64, tables: types.Tables, bits: u8, group: u16 };

pub const View = struct { n: u32, k: u32, matrix: Matrix };

fn tableKind(dtype: types.DType) error{UnsupportedTables}!types.Tables {
    return switch (dtype) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        else => error.UnsupportedTables,
    };
}

/// The projection's (N, K) and addresses; a stack (E + 1, N, ...) reads as one expert's shape at the stack's addresses.
pub fn view(d: Device, stacked: bool) error{UnsupportedTables}!View {
    const lead: usize = @intFromBool(stacked);
    const n = d.words.dim(lead);
    const k = d.words.dim(lead + 1) * 32 / d.bits;
    const matrix: Matrix = .{ .words = d.words.ptr, .scale = d.scales.ptr, .bias = d.biases.ptr, .tables = try tableKind(d.scales.dtype), .bits = d.bits, .group = d.group };
    return .{ .n = @intCast(n), .k = @intCast(k), .matrix = matrix };
}

fn table64(dtype: types.DType, data: []const u8, i: usize) f64 {
    return @as(f64, convert.load(dtype, data, i));
}

/// Weights of a projection: its rows (a stack's leading dimensions folded) by K.
pub fn elements(h: Host) usize {
    var n_rows: usize = 1;
    for (h.words.shape[0 .. h.words.rank - 1]) |d| n_rows *= d;
    return n_rows * h.scales.shape[h.scales.rank - 1] * h.group;
}

/// fp64 dequantization of every row of `h` (leading dimensions folded), scale * code + bias, into `out` [rows, K].
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const rank = h.words.rank;
    var n_rows: usize = 1;
    for (h.words.shape[0 .. rank - 1]) |d| n_rows *= d;
    const groups = h.scales.shape[h.scales.rank - 1];
    const k = groups * h.group;
    const words_row = h.words.shape[rank - 1];
    if (out.len != n_rows * k or words_row * 32 != k * h.bits) return error.UnexpectedTensor;
    const mask: u64 = (@as(u64, 1) << @intCast(h.bits)) - 1;
    for (0..n_rows) |r| for (0..k) |c| {
        const bit = c * h.bits;
        const at = r * words_row + (bit >> 5);
        const lo: u64 = std.mem.readInt(u32, h.words.bytes[4 * at ..][0..4], .little);
        const hi: u64 = if ((bit & 31) + h.bits > 32) std.mem.readInt(u32, h.words.bytes[4 * (at + 1) ..][0..4], .little) else 0;
        const code: f64 = @floatFromInt(((lo | hi << 32) >> @intCast(bit & 31)) & mask);
        const g = r * groups + c / h.group;
        out[r * k + c] = code * table64(h.scales.dtype, h.scales.bytes, g) + table64(h.biases.dtype, h.biases.bytes, g);
    };
}

/// MLX affine words and tables as fp32 [R, K] the way torch computes them (product rounded, then sum), widths 2, 4, 8.
pub fn dequant(a: Allocator, words: Tensor, scale: Tensor, bias: Tensor, group: usize) convert.Error![]f32 {
    if (words.rank != 2 or scale.rank != 2 or bias.rank != 2 or !std.mem.eql(usize, scale.shape[0..2], bias.shape[0..2])) return error.UnexpectedTensor;
    if (words.dtype != .u32 and words.dtype != .i32) return error.UnexpectedTensor;
    if (!convert.isFloat(scale.dtype) or scale.dtype == .f64 or !convert.isFloat(bias.dtype) or bias.dtype == .f64) return error.UnexpectedTensor;
    const n_rows = words.shape[0];
    const k = scale.shape[1] * group;
    if (k == 0 or 32 * words.shape[1] / k == 0) return error.UnexpectedTensor;
    const bits = 32 * words.shape[1] / k;
    if (bits != 2 and bits != 4 and bits != 8) return error.UnsupportedQuantization;
    const per = 32 / bits;
    if (words.shape[1] * per != k or scale.shape[0] != n_rows) return error.UnexpectedTensor;
    const out = try a.alloc(f32, n_rows * k);
    const mask: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
    for (0..n_rows) |r| for (0..k) |c| {
        const word = std.mem.readInt(u32, words.bytes[4 * (r * words.shape[1] + c / per) ..][0..4], .little);
        const code: f32 = @floatFromInt((word >> @intCast(bits * (c % per))) & mask);
        const g = r * scale.shape[1] + c / group;
        const s = convert.load(scale.dtype, scale.bytes, g);
        const b = convert.load(bias.dtype, bias.bytes, g);
        // torch multiplies, then adds: two roundings
        const prod = code * s;
        out[r * k + c] = prod + b;
    };
    return out;
}

fn hostOf(words: []const u32, scales: []const u16, biases: []const u16, rows_n: usize, bits: u8, group: u16) Host {
    const groups = scales.len / rows_n;
    return .{
        .words = .{ .dtype = .i32, .rank = 2, .shape = .{ rows_n, words.len / rows_n, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(words) },
        .scales = .{ .dtype = .bf16, .rank = 2, .shape = .{ rows_n, groups, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(scales) },
        .biases = .{ .dtype = .bf16, .rank = 2, .shape = .{ rows_n, groups, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(biases) },
        .bits = bits,
        .group = group,
    };
}

test "the reference unpacks low code first, across words for 3 bits" {
    var four = [_]u32{0x76543210};
    var scale = [_]u16{0x4000}; // bf16 2.0
    var bias = [_]u16{0xBF80}; // bf16 -1.0
    var out4: [8]f64 = undefined;
    try reference(hostOf(&four, &scale, &bias, 1, 4, 8), &out4);
    try std.testing.expectEqualSlices(f64, &.{ -1, 1, 3, 5, 7, 9, 11, 13 }, &out4);
    // 32 codes of 3 bits: code i is i % 8, packed little-endian over three words
    var three: [3]u32 = @splat(0);
    for (0..32) |i| {
        const bit = i * 3;
        const v: u64 = (i % 8) << @intCast(bit & 31);
        three[bit >> 5] |= @truncate(v);
        if ((bit & 31) + 3 > 32) three[(bit >> 5) + 1] |= @truncate(v >> 32);
    }
    var scale3 = [_]u16{0x3F80}; // 1.0
    var bias3 = [_]u16{0x0000};
    var out3: [32]f64 = undefined;
    try reference(hostOf(&three, &scale3, &bias3, 1, 3, 32), &out3);
    for (out3, 0..) |v, i| try std.testing.expectEqual(@as(f64, @floatFromInt(i % 8)), v);
}

test "dequant unpacks 4-bit words low nibble first" {
    var words = [_]u32{0x76543210};
    var scale = [_]u16{0x4000};
    var bias = [_]u16{0xBF80};
    const h = hostOf(&words, &scale, &bias, 1, 4, 8);
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const out = try dequant(arena.allocator(), h.words, h.scales, h.biases, 8);
    try std.testing.expectEqualSlices(f32, &.{ -1, 1, 3, 5, 7, 9, 11, 13 }, out);
}

test "a projection splits its words and group tables on the same boundary" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var words: [2 * 4]u32 = undefined;
    for (&words, 0..) |*w, i| w.* = @intCast(i);
    var scales: [2 * 2]u16 = .{ 10, 11, 12, 13 };
    const p = hostOf(&words, &scales, &scales, 2, 4, 32);
    const pair = try halves(arena.allocator(), p);
    try std.testing.expectEqual(@as(usize, 2), pair[0].words.shape[1]);
    try std.testing.expectEqual(@as(usize, 1), pair[1].scales.shape[1]);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 6, 7 }, @alignCast(std.mem.bytesAsSlice(u32, pair[1].words.bytes)));
    try std.testing.expectEqualSlices(u16, &.{ 11, 13 }, @alignCast(std.mem.bytesAsSlice(u16, pair[1].scales.bytes)));
}

test "a column split keeps whole groups and a row split its own rows" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var words: [4 * 4]u32 = undefined;
    for (&words, 0..) |*w, i| w.* = @intCast(i);
    var scales: [4 * 2]u16 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    // eight bits: four words a row hold sixteen codes, two groups of eight
    const p = hostOf(&words, &scales, &scales, 4, 8, 8);
    const c = try sliceCols(a, p, .{ .rank = 1, .world = 2 }, "test");
    try std.testing.expectEqual(@as(usize, 2), c.words.shape[1]);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 6, 7, 10, 11, 14, 15 }, @alignCast(std.mem.bytesAsSlice(u32, c.words.bytes)));
    try std.testing.expectEqualSlices(u16, &.{ 1, 3, 5, 7 }, @alignCast(std.mem.bytesAsSlice(u16, c.scales.bytes)));
    const q = try sliceRows(a, p, &.{ .{ .from = 1, .to = 2 }, .{ .from = 3, .to = 4 } });
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 6, 7, 12, 13, 14, 15 }, @alignCast(std.mem.bytesAsSlice(u32, q.words.bytes)));
}

test "the config's per-tensor widths override the global one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text =
        \\{"group_size": 64, "bits": 4, "mode": "affine",
        \\ "layers.0.mlp.gate": {"group_size": 32, "bits": 8},
        \\ "vision_tower.x": {"group_size": 64, "bits": 7}}
    ;
    const obj = (try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{})).object;
    const q = (try detect(a, .{ .root = obj, .quantization = obj })).?;
    try std.testing.expectEqual(Width{ .bits = 8, .group = 32 }, try q.width("layers.0.mlp.gate"));
    try std.testing.expectEqual(Width{ .bits = 4, .group = 64 }, try q.width("layers.1.mlp.gate"));
    try std.testing.expectError(error.UnsupportedQuantization, q.width("vision_tower.x"));
    const gptq = (try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"group_size\": 64, \"bits\": 4, \"mode\": \"gptq\"}", .{})).object;
    try std.testing.expect((try detect(a, .{ .root = gptq, .quantization = gptq })) == null);
}
