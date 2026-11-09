//! ModelOpt NVFP4 as stored: e2m1 codes, an E4M3 scale every 16 inputs, one global scale, and the f64 reference.

const std = @import("std");
const types = @import("types.zig");
const slice = @import("slice.zig");
const convert = @import("convert.zig");
const fp8 = @import("fp8.zig");

const Tensor = types.Tensor;
const Allocator = std.mem.Allocator;

pub const id: types.Format = .nvfp4;
pub const decoder: types.Decoder = .nvfp4;

/// Inputs one e4m3 scale covers.
pub const group = 16;

/// A family admits a ModelOpt checkpoint through core/modelopt.zig, so this format is never detected.
pub const Config = void;

pub fn detect(a: Allocator, src: types.Sources) Allocator.Error!?Config {
    _ = a;
    _ = src;
    return null;
}

/// An e2m1 nibble's value.
pub fn e2m1(n: u8) f64 {
    const mag = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    return (if (n & 8 != 0) @as(f64, -1) else 1) * mag[n & 7];
}

/// As stored: `weight` u8 [..., N, K / 2] (even columns low nibbles), `scale` BF16 [..., N, K / 16], one global scale.
pub const Host = struct { weight: Tensor, scale: Tensor, global: []const f32 };

fn name(buf: []u8, key: []const u8, suffix: []const u8) ?[]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ key, suffix }) catch null;
}

fn refuse(key: []const u8, what: []const u8) error{UnexpectedTensor} {
    return types.refuse(key, what);
}

/// Whether `key` holds ModelOpt's NVFP4 tensors: U8 codes, a group table and the second (global) scale.
pub fn matches(t: anytype, key: []const u8) bool {
    var b: [3][320]u8 = undefined;
    const w = t.get(name(&b[0], key, ".weight") orelse return false) catch return false;
    _ = t.get(name(&b[1], key, ".weight_scale") orelse return false) catch return false;
    if (!t.has(name(&b[2], key, ".weight_scale_2") orelse return false)) return false;
    return w.dtype == .u8 and w.rank == 2;
}

/// Why these tensors cannot be this format, or null when they can: checked without a log, as `bf16.unfit` is.
pub fn unfit(w: Tensor, s: Tensor, g: Tensor) ?[]const u8 {
    if (w.dtype != .u8 or w.rank != 2) return "an NVFP4 weight is U8 [N, K/2]";
    const n = w.shape[0];
    const k = 2 * w.shape[1];
    if (k % 32 != 0) return "an NVFP4 weight's K is a multiple of 32 (eights of codes, groups of 16)";
    if (s.rank != 2 or s.shape[0] != n or s.shape[1] * group != k) return "NVFP4 scales are [N, K/16]";
    if (g.numel() != 1 or g.dtype != .f32) return "NVFP4's second scale is one F32";
    const global = convert.load(.f32, g.bytes, 0);
    if (!std.math.isFinite(global) or global == 0) return "NVFP4's second scale is zero, infinite or nan";
    switch (s.dtype) {
        .f8_e4m3 => for (0..s.numel()) |i| {
            if (s.bytes[i] & 0x7f == 0x7f) return "an NVFP4 scale is E4M3 nan";
        },
        else => return "NVFP4 scales are E4M3, which is what ModelOpt writes and the 16-bit table holds",
    }
    return null;
}

/// `key`'s projection as stored, its group table widened; any other dtype, rank or value is refused by name.
pub fn read(a: Allocator, t: anytype, key: []const u8) !Host {
    var b: [3][320]u8 = undefined;
    const wn = name(&b[0], key, ".weight") orelse return error.UnexpectedTensor;
    const sn = name(&b[1], key, ".weight_scale") orelse return error.UnexpectedTensor;
    const gn = name(&b[2], key, ".weight_scale_2") orelse return error.UnexpectedTensor;
    const w = try t.get(wn);
    const s = try t.get(sn);
    const g = try t.get(gn);
    if (unfit(w, s, g)) |why| return refuse(wn, why);
    const one = try a.alloc(f32, 1);
    one[0] = convert.load(.f32, g.bytes, 0);
    return .{ .weight = w, .scale = try widened(a, s), .global = one };
}

/// The group table as the kernel reads it: E4M3's values as BF16 (exact: three mantissa bits fit seven).
fn widened(a: Allocator, s: Tensor) !Tensor {
    const out = try a.alloc(u16, s.numel());
    for (out, 0..) |*o, i| {
        const value: f32 = @floatCast(fp8.e4m3(s.bytes[i]));
        o.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
    return .{ .dtype = .bf16, .rank = s.rank, .shape = s.shape, .bytes = std.mem.sliceAsBytes(out) };
}

pub fn rows(h: Host) usize {
    return h.weight.shape[0];
}

pub fn bytes(h: Host) usize {
    return h.weight.bytes.len + h.scale.bytes.len + @sizeOf(f32) * h.global.len;
}

/// Inputs of the projection.
fn inputs(h: Host) usize {
    return 2 * h.weight.shape[1];
}

/// Columns [from, to) of K, copied: a cut on an even input (a byte of codes) and a group boundary.
fn cols(a: Allocator, p: Host, from: usize, to: usize) slice.Error!Host {
    if (from % 2 != 0 or to % 2 != 0 or from % group != 0 or to % group != 0) return error.UnevenSplit;
    return .{
        .weight = try slice.takeCols(a, p.weight, from / 2, to / 2),
        .scale = try slice.takeCols(a, p.scale, from / group, to / group),
        .global = p.global,
    };
}

/// Both halves along K, copied (K a multiple of 32).
pub fn halves(a: Allocator, p: Host) slice.Error![2]Host {
    const k = inputs(p);
    if (k % 32 != 0) return error.UnevenSplit;
    return .{ try cols(a, p, 0, k / 2), try cols(a, p, k / 2, k) };
}

pub fn sliceRows(a: Allocator, p: Host, spans: []const slice.Span) slice.Error!Host {
    return .{ .weight = try slice.takeRows(a, p.weight, spans), .scale = try slice.takeRows(a, p.scale, spans), .global = p.global };
}

pub fn sliceCols(a: Allocator, p: Host, r: slice.Rank, what: []const u8) slice.Error!Host {
    const width = try slice.even(inputs(p), r.world, what);
    return cols(a, p, r.rank * width, (r.rank + 1) * width);
}

/// A rank's `part` experts of a stack and the shared one last on rank 0, their globals cut the same way.
pub fn sliceStack(a: Allocator, p: Host, r: slice.Rank, part: usize) slice.Error!Host {
    if (p.global.len != p.weight.shape[0]) return error.UnevenSplit; // one global an expert, the shared one last
    const first = r.rank * part;
    const global = if (r.rank != 0) p.global[first..][0..part] else blk: {
        const out = try a.alloc(f32, part + 1);
        @memcpy(out[0..part], p.global[first..][0..part]);
        out[part] = p.global[p.global.len - 1];
        break :blk out;
    };
    return .{
        .weight = try slice.takeExperts(a, p.weight, r, part),
        .scale = try slice.takeExperts(a, p.scale, r, part),
        .global = global,
    };
}

/// An uploaded projection: the checkpoint's codes and group table on the device, the global scale a number.
pub const Device = struct { weight: types.Buf, scale: types.Buf, global: f32 };

/// The tensors as stored: this format keeps its layout, as MLX's does.
pub fn upload(u: types.Uploader, h: Host) !Device {
    if (h.global.len != 1) return error.UnexpectedTensor; // a stack's experts carry a global each
    return .{ .weight = try u.tensor(h.weight), .scale = try u.tensor(h.scale), .global = h.global[0] };
}

/// What the kernels read of an NVFP4 projection: the codes' address, the group table's and the global scale.
pub const Matrix = struct { codes: u64, scale: u64, tables: types.Tables, global: f32 };

pub const View = struct { n: u32, k: u32, matrix: Matrix };

fn tableKind(dtype: types.DType) error{UnsupportedTables}!types.Tables {
    return switch (dtype) {
        .f32 => .f32,
        .bf16 => .bf16,
        .f16 => .f16,
        else => error.UnsupportedTables,
    };
}

/// The projection's (N, K) and addresses; a stack is refused: its experts carry a global scale each.
pub fn view(d: Device, stacked: bool) error{UnsupportedTables}!View {
    if (stacked or d.weight.dtype != .u8 or d.weight.rank != 2) return error.UnsupportedTables;
    if (d.scale.rank != 2 or d.scale.shape[0] != d.weight.shape[0] or d.scale.shape[1] * group != 2 * d.weight.shape[1]) return error.UnsupportedTables;
    return .{
        .n = @intCast(d.weight.shape[0]),
        .k = @intCast(2 * d.weight.shape[1]),
        .matrix = .{ .codes = d.weight.ptr, .scale = d.scale.ptr, .tables = try tableKind(d.scale.dtype), .global = d.global },
    };
}

/// Weights of a projection: its rows (a stack's leading dimensions folded) by K.
pub fn elements(h: Host) usize {
    return h.weight.numel() * 2;
}

/// A group table entry as fp64, in whichever form the host holds it.
fn scaleOf(h: Host, i: usize) f64 {
    return switch (h.scale.dtype) {
        .f8_e4m3 => fp8.e4m3(h.scale.bytes[i]),
        else => convert.load(h.scale.dtype, h.scale.bytes, i),
    };
}

/// fp64 dequantization of every row of `h` (a stack's leading dimensions folded), code * scale * global, into `out`.
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const rank = h.weight.rank;
    if (rank < 2 or rank > h.weight.shape.len or h.scale.rank != rank or h.weight.dtype != .u8) return error.UnexpectedTensor;
    if (h.scale.dtype != .f8_e4m3 and !convert.tableDtype(h.scale.dtype)) return error.UnexpectedTensor;
    const n = h.weight.shape[rank - 2];
    const k = h.weight.shape[rank - 1] * 2;
    if (n == 0 or k == 0) return error.UnexpectedTensor;
    for (0..rank - 2) |d| if (h.weight.shape[d] == 0 or h.weight.shape[d] != h.scale.shape[d]) return error.UnexpectedTensor;
    if (h.scale.shape[rank - 2] != n or h.scale.shape[rank - 1] * group != k or out.len != elements(h)) return error.UnexpectedTensor;
    const stack = elements(h) / (n * k);
    if (h.global.len != stack) return error.UnexpectedTensor;
    for (0..stack) |e| for (0..n) |r| for (0..k) |c| {
        const row = e * n + r;
        const byte = h.weight.bytes[row * (k / 2) + c / 2];
        const nib: u8 = if (c & 1 == 0) byte & 0xF else byte >> 4;
        out[row * k + c] = e2m1(nib) * scaleOf(h, row * (k / group) + c / group) * h.global[e];
    };
}

fn parsed(a: Allocator, text: []const u8) !std.json.ObjectMap {
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{})).object;
}

test "an NVFP4 stack against hand-decoded values" {
    // two experts of [1, 32]: two scale groups a row; byte 0 holds columns 0 (low) and 1 (high)
    var codes: [2 * 16]u8 = @splat(0);
    codes[0] = 0xF2; // column 0: e2m1 2 -> 1.0; column 1: e2m1 0xF -> -6
    codes[8] = 0x07; // column 16: 6
    codes[16 + 15] = 0x30; // expert 1, column 31: e2m1 3 -> 1.5
    const scales = [_]u8{ 0x38, 0x40, 0x38, 0x30 }; // 1, 2 | 1, 0.5
    const global = [_]f32{ 0.25, 2 };
    const h: Host = .{
        .weight = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 1, 16, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 2, 1, 2, 1, 1 }, .bytes = &scales },
        .global = &global,
    };
    var out: [64]f64 = undefined;
    try reference(h, &out);
    try std.testing.expectEqual(@as(f64, 0.25), out[0]);
    try std.testing.expectEqual(@as(f64, -1.5), out[1]);
    try std.testing.expectEqual(@as(f64, 3), out[16]);
    try std.testing.expectEqual(@as(f64, 1.5), out[32 + 31]);
    try std.testing.expectEqual(@as(f64, 0), out[2]);
}

test "a reference refuses mismatched expert stacks" {
    const codes: [16]u8 = @splat(0x22);
    const scales: [2]u8 = @splat(0x38);
    const global = [_]f32{ 1, 1 };
    const h: Host = .{
        .weight = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 1, 8, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = &scales },
        .global = &global,
    };
    var out: [32]f64 = undefined;
    try std.testing.expectError(error.UnexpectedTensor, reference(h, &out));
}

const Fake = struct {
    w: Tensor,
    s: Tensor,
    g: Tensor,

    pub fn get(f: *const Fake, key: []const u8) !Tensor {
        if (std.mem.endsWith(u8, key, ".weight")) return f.w;
        if (std.mem.endsWith(u8, key, ".weight_scale_2")) return f.g;
        if (std.mem.endsWith(u8, key, ".weight_scale")) return f.s;
        return error.MissingTensor;
    }

    pub fn has(f: *const Fake, key: []const u8) bool {
        _ = f;
        return std.mem.endsWith(u8, key, ".weight_scale_2");
    }
};

test "a projection is read as stored, its E4M3 table widened exactly; a bad scale is unfit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var codes: [4 * 16]u8 = @splat(0);
    codes[0] = 0x21; // row 0, inputs 0 and 1: e2m1 1 (+0.5) and 2 (+1.0)
    const scales = [_]u8{ 0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x40 }; // [4, 2]: 1s, then 2 in the last row
    const g = [_]f32{0.5};
    const t: Fake = .{
        .w = .{ .dtype = .u8, .rank = 2, .shape = .{ 4, 16, 1, 1, 1 }, .bytes = &codes },
        .s = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 4, 2, 1, 1, 1 }, .bytes = &scales },
        .g = .{ .dtype = .f32, .rank = 1, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&g) },
    };
    try std.testing.expect(matches(&t, "x"));
    try std.testing.expect(unfit(t.w, t.s, t.g) == null);
    const h = try read(a, &t, "x");
    try std.testing.expectEqual(types.DType.bf16, h.scale.dtype);
    try std.testing.expectEqual(@as(f32, 0.5), h.global[0]);
    try std.testing.expectEqual(@as(usize, 4), rows(h));
    try std.testing.expectEqual(@as(usize, 64 + 8 * 2 + 4), bytes(h));
    var out: [128]f64 = undefined;
    try reference(h, &out);
    try std.testing.expectEqual(@as(f64, 0.25), out[0]);
    try std.testing.expectEqual(@as(f64, 0.5), out[1]);
    try std.testing.expectEqual(@as(f64, 0), out[2]);

    var nan = scales;
    nan[7] = 0x7f;
    try std.testing.expect(unfit(t.w, .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 4, 2, 1, 1, 1 }, .bytes = &nan }, t.g) != null);
    const zero = [_]f32{0};
    try std.testing.expect(unfit(t.w, t.s, .{ .dtype = .f32, .rank = 1, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&zero) }) != null);
    var f32_table = [_]f32{ 1.0, 1.0 };
    try std.testing.expect(unfit(t.w, .{ .dtype = .f32, .rank = 2, .shape = .{ 4, 2, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&f32_table) }, t.g) != null);
    var short = t.w; // K = 16, but K is a multiple of 32
    short.shape = .{ 4, 8, 1, 1, 1 };
    try std.testing.expect(unfit(short, t.s, t.g) != null);
}

test "halves cut the codes on a byte and the scales on a group" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var codes: [2 * 32]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @truncate(i);
    const scales = [_]u8{ 0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x38 }; // [2, 4]
    const g = [_]f32{1};
    const h: Host = .{
        .weight = .{ .dtype = .u8, .rank = 2, .shape = .{ 2, 32, 1, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 2, 4, 1, 1, 1 }, .bytes = &scales },
        .global = &g,
    };
    const pair = try halves(a, h);
    try std.testing.expectEqualSlices(u8, codes[0..16], pair[0].weight.bytes[0..16]);
    try std.testing.expectEqualSlices(u8, codes[32..48], pair[0].weight.bytes[16..32]);
    try std.testing.expectEqualSlices(u8, codes[16..32], pair[1].weight.bytes[0..16]);
    try std.testing.expectEqualSlices(u8, codes[48..64], pair[1].weight.bytes[16..32]);
    try std.testing.expectEqual(@as(usize, 2), pair[0].scale.shape[1]);
    try std.testing.expectEqual(@as(usize, 64), elements(pair[0]));
}

test "a view is the codes' and table's addresses and the global scale; a stack is refused" {
    const d: Device = .{
        .weight = .{ .ptr = 4096, .len = 64, .dtype = .u8, .rank = 2, .shape = .{ 4, 16, 1, 1, 1 } },
        .scale = .{ .ptr = 8192, .len = 16, .dtype = .bf16, .rank = 2, .shape = .{ 4, 2, 1, 1, 1 } },
        .global = 0.25,
    };
    const v = try view(d, false);
    try std.testing.expectEqual(@as(u32, 4), v.n);
    try std.testing.expectEqual(@as(u32, 32), v.k);
    try std.testing.expectEqual(@as(u64, 4096), v.matrix.codes);
    try std.testing.expectEqual(@as(u64, 8192), v.matrix.scale);
    try std.testing.expectEqual(types.Tables.bf16, v.matrix.tables);
    try std.testing.expectEqual(@as(f32, 0.25), v.matrix.global);
    try std.testing.expectError(error.UnsupportedTables, view(d, true));
    var narrow = d;
    narrow.scale.shape = .{ 4, 1, 1, 1, 1 }; // K/16 of 1 does not cover K = 32
    try std.testing.expectError(error.UnsupportedTables, view(narrow, false));
}

test "the config is never detected: a family admits a ModelOpt checkpoint itself" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const src: types.Sources = .{ .root = try parsed(arena.allocator(), "{\"quantization_config\": {\"quant_algo\": \"NVFP4\"}}"), .quantization = .empty };
    try std.testing.expect(try detect(arena.allocator(), src) == null);
}

test "a stack of experts is refused: a global scale is one a projection" {
    var codes: [2 * 8 * 16]u8 = @splat(0);
    const scales: [2 * 8 * 2]u8 = @splat(0x38);
    const g = [_]f32{ 1, 1 };
    const w: Tensor = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 8, 16, 1, 1 }, .bytes = &codes };
    const s: Tensor = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 2, 8, 2, 1, 1 }, .bytes = &scales };
    const gg: Tensor = .{ .dtype = .f32, .rank = 1, .shape = .{ 2, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&g) };
    try std.testing.expect(unfit(w, s, gg) != null);
}
