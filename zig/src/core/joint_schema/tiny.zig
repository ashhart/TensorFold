//! The tiny head of tools/clef_head_golden.py: counter-generated BF16 weights and inputs, and a checkpoint writer.
const std = @import("std");
const Config = @import("config.zig").Config;
const tensors = @import("tensors.zig");

pub const cfg: Config = .{ .hidden_size = 128, .width = 32, .routing_layers = 2, .layers = 4, .heads = 4, .feedforward = 128 };
pub const vocab = 40;
pub const rows = 48;
pub const zero_token_row = 29;
/// tools/clef_head_golden.py's outputs for this head: FP64 and torch BF16 logits, and the near-zero cases.
pub const fixture = @embedFile("fixtures/clef_tiny.json");

/// SplitMix64's output for state x.
pub fn splitmix64(x: u64) u64 {
    var z = x +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

pub fn fnv1a(name: []const u8) u32 {
    var h: u32 = 0x811C9DC5;
    for (name) |b| h = (h ^ b) *% 0x01000193;
    return h;
}

pub const Recipe = struct { levels: u64, exp: i32, offset: f64 };

/// offset + k 2^-exp with k in [-levels, levels]: the i-th value of the stream `name`.
pub fn grid(name: []const u8, i: u64, r: Recipe) f64 {
    const k: i64 = @as(i64, @intCast(splitmix64((@as(u64, fnv1a(name)) << 32) ^ i) % (2 * r.levels + 1))) - @as(i64, @intCast(r.levels));
    return r.offset + std.math.ldexp(@as(f64, @floatFromInt(k)), -r.exp);
}

/// A head tensor's recipe from its name and shape alone, as the golden tool's `recipe`.
pub fn recipe(name: []const u8, shape: []const usize) Recipe {
    const module_end = std.mem.lastIndexOfScalar(u8, name, '.') orelse name.len;
    const module_start = if (std.mem.lastIndexOfScalar(u8, name[0..module_end], '.')) |d| d + 1 else 0;
    if (std.mem.indexOf(u8, name[module_start..module_end], "norm") != null)
        return if (std.mem.endsWith(u8, name, ".weight")) .{ .levels = 31, .exp = 7, .offset = 1 } else .{ .levels = 31, .exp = 8, .offset = 0 };
    if (std.mem.eql(u8, name, "type_embedding.weight")) return .{ .levels = 127, .exp = 8, .offset = 0 };
    if (std.mem.endsWith(u8, name, "bias")) return .{ .levels = 127, .exp = 10, .offset = 0 };
    const bit_length: i32 = std.math.log2_int(usize, shape[shape.len - 1]) + 1;
    return .{ .levels = 127, .exp = @divFloor(bit_length + 11, 2), .offset = 0 };
}

pub const scalars = [_]struct { []const u8, f64 }{ .{ "prior_logit_scale", 2.296875 }, .{ "joint_logit_scale", 2.296875 }, .{ "residual_gate", -2.0 } };

/// Element i of the head tensor `name`.
pub fn value(name: []const u8, shape: []const usize, i: usize) f64 {
    for (scalars) |s| if (std.mem.eql(u8, s[0], name)) return s[1];
    return grid(name, i, recipe(name, shape));
}

/// An exactly representable value's BF16 bits.
pub fn bits(v: f64) u16 {
    const f: f32 = @floatCast(v);
    return @intCast(@as(u32, @bitCast(f)) >> 16);
}

/// The record's final-norm rows (rows × hidden), exact BF16 values.
pub fn hidden(a: std.mem.Allocator) ![]f64 {
    const out = try a.alloc(f64, rows * cfg.hidden_size);
    for (out, 0..) |*v, i| v.* = grid("hidden", i, .{ .levels = 127, .exp = 4, .offset = 0 });
    return out;
}

/// The output-embedding table as BF16 bytes: row 0 zero, the rest generated.
pub fn table(a: std.mem.Allocator) ![]u8 {
    const h: usize = cfg.hidden_size;
    const out = try a.alloc(u8, vocab * h * 2);
    for (0..vocab * h) |i| {
        const v = if (i < h) 0 else grid("lm_head", i - h, .{ .levels = 127, .exp = 7, .offset = 0 });
        std.mem.writeInt(u16, out[2 * i ..][0..2], bits(v), .little);
    }
    return out;
}

pub fn ids(a: std.mem.Allocator) ![]u32 {
    const out = try a.alloc(u32, rows);
    for (out, 0..) |*v, i| v.* = @intCast(1 + splitmix64((@as(u64, fnv1a("ids")) << 32) ^ i) % (vocab - 1));
    out[zero_token_row] = 0;
    return out;
}

/// How `write` departs from the generated head, for the loader's refusals.
pub const Edit = union(enum) {
    none,
    drop: []const u8, // the tensor left out
    extra: []const u8, // a tensor added under this name
    dtype: []const u8, // the tensor stored as F32
    shape: []const u8, // the tensor's last dimension one longer (a scalar becomes [2])
    config: []const u8, // the config file's text
};

/// `joint_head_config.json` and `joint_head.safetensors` for the tiny head (or `c`) in `dir`, with `edit` applied.
pub fn write(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, c: Config, edit: Edit) !void {
    const text = if (edit == .config) edit.config else try std.fmt.allocPrint(a, "{{\"hidden_size\": {d}, \"width\": {d}, \"routing_layers\": {d}, \"layers\": {d}, \"heads\": {d}, \"feedforward\": {d}}}", .{ c.hidden_size, c.width, c.routing_layers, c.layers, c.heads, c.feedforward });
    try dir.writeFile(io, .{ .sub_path = "joint_head_config.json", .data = text });
    var listing: tensors.Listing = .{ .a = a };
    _ = try tensors.fill(a, c, &listing);
    if (edit == .extra) {
        try listing.names.append(a, edit.extra);
        try listing.shapes.append(a, &.{2});
    }
    var header: std.ArrayList(u8) = .empty;
    var data: std.ArrayList(u8) = .empty;
    try header.appendSlice(a, "{\"__metadata__\":{\"format\":\"pt\"}");
    for (listing.names.items, listing.shapes.items) |name, given| {
        if (edit == .drop and std.mem.eql(u8, edit.drop, name)) continue;
        const f32s = edit == .dtype and std.mem.eql(u8, edit.dtype, name);
        var shape = try a.dupe(usize, given);
        if (edit == .shape and std.mem.eql(u8, edit.shape, name)) shape = if (shape.len == 0) try a.dupe(usize, &.{2}) else blk: {
            shape[shape.len - 1] += 1;
            break :blk shape;
        };
        var n: usize = 1;
        for (shape) |d| n *= d;
        const begin = data.items.len;
        for (0..n) |i| {
            const v = value(name, given, i);
            if (f32s) try data.print(a, "{s}", .{std.mem.asBytes(&@as(f32, @floatCast(v)))}) else try data.print(a, "{s}", .{std.mem.asBytes(&bits(v))});
        }
        try header.print(a, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ name, if (f32s) "F32" else "BF16" });
        for (shape, 0..) |d, i| try header.print(a, "{s}{d}", .{ if (i == 0) "" else ",", d });
        try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ begin, data.items.len });
    }
    try header.append(a, '}');
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, header.items.len, .little);
    try dir.writeFile(io, .{ .sub_path = "joint_head.safetensors", .data = try std.mem.concat(a, u8, &.{ &len, header.items, data.items }) });
}
