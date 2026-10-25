//! A synthetic safetensors file exercises the real read-only checkpoint and PLE constant admission.
const std = @import("std");
const cfg = @import("config.zig");
const weights = @import("weights.zig");
const ngram = @import("hybrid_ngram");
const Item = struct { name: []const u8, shape: weights.Shape, bytes: []const u8, dtype: []const u8 };
const Sink = struct {
    a: std.mem.Allocator,
    c: *const cfg.Config,
    broken: bool,
    nonzero: bool = false,
    f32_vectors: bool = false,
    list: std.ArrayList(Item) = .empty,
    fn item(s: *@This(), stem: []const u8, tail: []const u8, shape: weights.Shape, dtype: []const u8, bytes: []const u8) !void {
        try s.list.append(s.a, .{ .name = try std.fmt.allocPrint(s.a, "{s}.{s}", .{ stem, tail }), .shape = shape, .dtype = dtype, .bytes = bytes });
    }
    fn packedMatrix(s: *@This(), name: []const u8, w: weights.Want) !void {
        const n = w.shape.dims[0];
        const k = w.shape.dims[1];
        const f = w.recipe.format;
        const mlx = f == .mlx4 or f == .mlx8;
        const bytes = try s.a.alloc(u8, n * k / (if (f == .mlx4 or f == .nvfp4) @as(usize, 2) else 1));
        @memset(bytes, if (f == .mlx4) 0x11 else if (f == .mlx8) 1 else if (f == .fp8) 0x38 else 0x22);
        var shape = w.shape;
        shape.dims[1] = if (mlx) k / (if (f == .mlx4) @as(usize, 8) else 4) else if (f == .nvfp4) k / 2 else k;
        try s.item(name, "weight", shape, if (mlx) "U32" else if (f == .fp8) "F8_E4M3" else "U8", bytes);
        if (mlx) {
            const sc = try s.a.alloc(u8, n * (k / w.recipe.group) * 2);
            const bi = try s.a.alloc(u8, sc.len);
            for (0..sc.len / 2) |i| {
                std.mem.writeInt(u16, sc[2 * i ..][0..2], 0x3f80, .little);
                std.mem.writeInt(u16, bi[2 * i ..][0..2], 0xbf00, .little);
            }
            shape.dims[1] = k / w.recipe.group;
            try s.item(name, "scales", shape, "BF16", sc);
            try s.item(name, "biases", shape, "BF16", bi);
        } else {
            if (f == .nvfp4) {
                const sc = try s.a.alloc(u8, n * (k / 16));
                @memset(sc, 0x38);
                shape.dims[1] = k / 16;
                try s.item(name, "weight_scale", shape, "F8_E4M3", sc);
            }
            const global = try s.a.alloc(u8, 4);
            std.mem.writeInt(u32, global[0..4], @bitCast(@as(f32, 0.5)), .little);
            try s.item(name, if (f == .fp8) "weight_scale" else "weight_scale_2", .{ .rank = 0 }, "F32", global);
        }
    }
    pub fn take(s: *@This(), name: []const u8, w: weights.Want) !void {
        var count: usize = 1;
        for (w.shape.dims[0..w.shape.rank]) |n| count *= n;
        const integers = w.role == .integer_vector;
        const matrix = w.role == .matrix or w.role == .embedding;
        if (matrix and w.recipe.format != .bf16) return s.packedMatrix(name, w);
        const f32_vector = s.f32_vectors and !matrix and !integers;
        const bytes = try s.a.alloc(u8, count * (if (integers) @as(usize, 8) else if (f32_vector) @as(usize, 4) else 2));
        @memset(bytes, 0);
        if (integers) {
            var prime: u64 = s.c.ngram_vocab - 1;
            var offset: i64 = 0;
            for (0..count) |i| {
                prime = try ngram.nextPrime(prime);
                const value: i64 = if (std.mem.endsWith(u8, name, "layer_multipliers"))
                    try ngram.multiplier(s.c.seed, 0, i, s.c.vocab)
                else if (std.mem.endsWith(u8, name, "ngram_heads_offsets")) offset else @intCast(prime);
                std.mem.writeInt(i64, bytes[i * 8 ..][0..8], value + @as(i64, if (s.broken and std.mem.endsWith(u8, name, "layer_multipliers") and i == 0) 1 else 0), .little);
                offset += @as(i64, @intCast(prime));
            }
        } else if (s.nonzero) {
            const key = try std.fmt.allocPrint(s.a, "{s}{s}", .{ name, if (matrix) ".weight" else "" });
            const direct = std.mem.endsWith(u8, key, ".linear_attn.norm.weight");
            const norm = !direct and (std.mem.endsWith(u8, key, "norm.weight") or
                std.mem.endsWith(u8, key, "norm_key.weight") or std.mem.endsWith(u8, key, "norm_query.weight") or
                std.mem.endsWith(u8, key, "norm_conv.weight"));
            const values = try @import("../../../tests/qwen4/math.zig").source(s.a, key, count, norm, direct);
            for (values, 0..) |v, i| {
                if (f32_vector) std.mem.writeInt(u32, bytes[4 * i ..][0..4], @bitCast(@as(f32, @bitCast(@as(u32, v) << 16))), .little) else std.mem.writeInt(u16, bytes[2 * i ..][0..2], v, .little);
            }
        }
        try s.list.append(s.a, .{ .name = try std.fmt.allocPrint(s.a, "{s}{s}", .{ name, if (matrix) ".weight" else "" }), .shape = w.shape, .bytes = bytes, .dtype = if (integers) "I64" else if (f32_vector) "F32" else "BF16" });
    }
};
pub fn write(tmp: std.testing.TmpDir, a: std.mem.Allocator, c: *const cfg.Config, broken: bool) !void {
    return writeOptions(tmp, a, c, broken, false, false);
}
pub fn writeNonzero(tmp: std.testing.TmpDir, a: std.mem.Allocator, c: *const cfg.Config) !void {
    return writeOptions(tmp, a, c, false, true, false);
}
pub fn writeF32Vectors(tmp: std.testing.TmpDir, a: std.mem.Allocator, c: *const cfg.Config) !void {
    return writeOptions(tmp, a, c, false, true, true);
}
fn writeOptions(tmp: std.testing.TmpDir, a: std.mem.Allocator, c: *const cfg.Config, broken: bool, nonzero: bool, f32_vectors: bool) !void {
    var sink: Sink = .{ .a = a, .c = c, .broken = broken, .nonzero = nonzero, .f32_vectors = f32_vectors };
    var p: cfg.Problem = .{};
    try weights.walk(c, .causal, &sink, &p);
    var header: std.Io.Writer.Allocating = .init(a);
    try header.writer.writeAll("{");
    var at: usize = 0;
    for (sink.list.items, 0..) |item, i| {
        try header.writer.print("{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (i == 0) "" else ",", item.name, item.dtype });
        for (item.shape.dims[0..item.shape.rank], 0..) |n, j| try header.writer.print("{s}{d}", .{ if (j == 0) "" else ",", n });
        try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + item.bytes.len });
        at += item.bytes.len;
    }
    try header.writer.writeAll("}");
    const h = header.written();
    const image = try a.alloc(u8, 8 + h.len + at);
    std.mem.writeInt(u64, image[0..8], h.len, .little);
    @memcpy(image[8..][0..h.len], h);
    at = 8 + h.len;
    for (sink.list.items) |item| {
        @memcpy(image[at..][0..item.bytes.len], item.bytes);
        at += item.bytes.len;
    }
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = image });
}
