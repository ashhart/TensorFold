//! FP8 n-gram rows: every E4M3 code against an f64 reference, refusals, and a synthetic FP8 checkpoint through Ngram.
const std = @import("std");
const core = @import("core");
const ple = @import("cuda_ple.zig");
const Table = @import("cuda_ple_fp8.zig").Table;
const a = std.testing.allocator;
const io = std.testing.io;
const base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";

fn tensor(dtype: core.safetensors.DType, rows: usize, cols: usize, bytes: []const u8) core.safetensors.Tensor {
    return .{ .dtype = dtype, .rank = 2, .shape = .{ rows, cols, 1, 1, 1 }, .bytes = bytes };
}

fn e4m3(byte: u8) f64 {
    const mag = byte & 127;
    if (mag == 127) return std.math.nan(f64);
    const e: i32 = @as(i32, mag >> 3) - 7;
    const value = if (mag < 8) @as(f64, @floatFromInt(mag)) / 512 else std.math.ldexp(1 + @as(f64, @floatFromInt(mag & 7)) / 8, e);
    return if (byte & 128 != 0) -value else value;
}

fn rounded(value: f64) u16 {
    const v: f32 = @floatCast(value);
    const bits: u32 = @bitCast(v);
    const high = bits >> 16;
    const low = bits & 65535;
    return @intCast(high + @intFromBool(low > 32768 or (low == 32768 and high % 2 == 1)));
}

fn expectValue(code: u8, scale: f32, actual: u16) !void {
    if (code & 127 == 127) {
        try std.testing.expect(actual & 0x7fc0 == 0x7fc0);
    } else try std.testing.expectEqual(rounded(e4m3(code) * scale), actual);
}

test "FP8 rows: every E4M3 code times a BF16 scale rounds once to BF16 in reordered repeated gathers" {
    var codes: [256]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast(i);
    for ([_]u32{ 0xBE53, 0x3B8D, 0x4241 }) |bits| {
        const scale: f32 = @bitCast(bits << 16);
        const t = try Table.init(tensor(.f8_e4m3, 16, 16, &codes), scale);
        var ids: [18]usize = undefined;
        for (0..16) |i| ids[i] = 15 - i;
        ids[16] = 3;
        ids[17] = 3;
        var out: [18 * 16]u16 = undefined;
        try t.gather(&ids, &out);
        for (ids, 0..) |id, row| for (0..16) |col| try expectValue(codes[id * 16 + col], scale, out[row * 16 + col]);
    }
}

test "FP8 rows: a bad row or destination refuses before writing and maps are never aliased" {
    var storage: [64]u8 align(8) = @splat(0x38);
    const t = try Table.init(tensor(.f8_e4m3, 2, 8, storage[0..16]), 0.5);
    var out: [16]u16 = @splat(0xbeef);
    try std.testing.expectError(error.BadFp8Rows, t.gather(&.{ 0, 2 }, &out));
    for (out) |v| try std.testing.expectEqual(@as(u16, 0xbeef), v);
    try std.testing.expectError(error.BadFp8Rows, t.gather(&.{0}, &out));
    try std.testing.expectError(error.AliasedFp8Rows, t.gather(&.{0}, std.mem.bytesAsSlice(u16, storage[0..16])));
    var ids: [2]usize = .{ 0, 0 };
    try std.testing.expectError(error.AliasedFp8Rows, t.gather(ids[0..1], std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(&ids))));
    try t.gather(&.{}, &.{});
}

test "FP8 rows: invalid dtype shape extent and scale cannot create a borrowed view" {
    const codes: [16]u8 = @splat(0);
    try std.testing.expectError(error.BadFp8Table, Table.init(tensor(.u8, 2, 8, &codes), 1));
    try std.testing.expectError(error.BadFp8Table, Table.init(tensor(.f8_e4m3, 2, 7, &codes), 1));
    try std.testing.expectError(error.BadFp8Table, Table.init(tensor(.f8_e4m3, 0, 16, codes[0..0]), 1));
    try std.testing.expectError(error.BadFp8Table, Table.init(tensor(.f8_e4m3, 2, 8, &codes), std.math.nan(f32)));
    try std.testing.expectError(error.BadFp8Table, Table.init(tensor(.f8_e4m3, std.math.maxInt(usize), 8, &codes), 1));
}

const scale_bits: u16 = 0xBE53;

fn codeAt(id: usize, col: usize) u8 {
    return @intCast((id * 7 + col * 13) % 256);
}

fn fixture(dir: std.Io.Dir, include_scale: bool) !void {
    const config =
        \\{"text_config":{"num_hidden_layers":2,"hidden_size":32,"vocab_size":100,"ple_layer_ids":[2],
        \\ "ngram_size":2,"heads_per_ngram":2,"ngram_vocab_size_base":13,"make_ngram_vocab_size_divisible_by":2,
        \\ "split_ngram_parts":2,"ple_embed_dim":32,"eos_token_id":99}}
    ;
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
    const h = try ple.Hash.init(.{ .vocab = 100, .ngram = 2, .per = 2, .base = 13, .divisor = 2, .shards = 2, .seed = 1234, .eos = 99, .ple_dim = 32, .ple_index = 0 });
    var header: std.Io.Writer.Allocating = .init(a);
    defer header.deinit();
    var index: std.Io.Writer.Allocating = .init(a);
    defer index.deinit();
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(a);
    try header.writer.writeByte('{');
    try index.writer.writeAll("{\"weight_map\":{");
    for (0..2) |shard| {
        const rows: usize = @intCast(h.starts[shard + 1] - h.starts[shard]);
        const begin = data.items.len;
        for (0..rows) |row| {
            const id: usize = @intCast(h.starts[shard] + @as(i64, @intCast(row)));
            for (0..16) |col| try data.append(a, codeAt(id, col));
        }
        if (shard > 0) {
            try header.writer.writeByte(',');
            try index.writer.writeByte(',');
        }
        try header.writer.print("\"{s}.shard_{d}.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},16],\"data_offsets\":[{d},{d}]}}", .{ base, shard, rows, begin, data.items.len });
        try index.writer.print("\"{s}.shard_{d}.weight\":\"table.safetensors\"", .{ base, shard });
    }
    if (include_scale) {
        const begin = data.items.len;
        try data.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u16, scale_bits)));
        try header.writer.print(",\"{s}.weight_scale\":{{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[{d},{d}]}}", .{ base, begin, data.items.len });
        try index.writer.print(",\"{s}.weight_scale\":\"table.safetensors\"", .{base});
    }
    try header.writer.writeByte('}');
    try index.writer.writeAll("}}");
    var image: std.Io.Writer.Allocating = .init(a);
    defer image.deinit();
    try image.writer.writeAll(&std.mem.toBytes(std.mem.nativeToLittle(u64, header.written().len)));
    try image.writer.writeAll(header.written());
    try image.writer.writeAll(data.items);
    try dir.writeFile(io, .{ .sub_path = "table.safetensors", .data = image.written() });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
}

test "FP8 PLE rows: mapped and positional reads equal the reference dequantization" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture(tmp.dir, true);
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    var g = try ple.Ngram.init(a, io, path);
    defer g.deinit();
    try std.testing.expectEqual(ple.Format.fp8, g.format);
    const scale: f32 = @bitCast(@as(u32, scale_bits) << 16);
    try std.testing.expectEqual(scale, g.scale);
    const tokens = [_]u32{ 7, 7, 12, 7, 3, 9, 12, 7, 1, 44, 63, 98 };
    var mapped: [tokens.len * 32]u16 = undefined;
    var positional: [tokens.len * 32]u16 = undefined;
    var mapped_xs: [tokens.len]f32 = undefined;
    var positional_xs: [tokens.len]f32 = undefined;
    var first = [_]i64{99};
    var second = first;
    g.read_mode = .mmap;
    try g.rows(&first, &tokens, &mapped, &mapped_xs);
    g.read_mode = .pread;
    try g.rows(&second, tokens[0..5], positional[0 .. 5 * 32], positional_xs[0..5]);
    try g.rows(&second, tokens[5..], positional[5 * 32 ..], positional_xs[5..]);
    try std.testing.expectEqualSlices(u16, &mapped, &positional);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&mapped_xs), std.mem.sliceAsBytes(&positional_xs));
    try std.testing.expectEqualSlices(i64, &first, &second);
    var hist = [_]i64{99};
    for (tokens, 0..) |token, t| {
        var ids: [2]i64 = undefined;
        try ple.lastIds(g.hash, &hist, token, &ids);
        hist = .{token};
        for (ids, 0..) |id, head| for (0..16) |col| {
            try expectValue(codeAt(@intCast(id), col), scale, mapped[t * 32 + head * 16 + col]);
        };
    }
}

test "FP8 PLE requires its table scale" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture(tmp.dir, false);
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    try std.testing.expectError(error.MissingTensor, ple.Ngram.init(a, io, path));
}
