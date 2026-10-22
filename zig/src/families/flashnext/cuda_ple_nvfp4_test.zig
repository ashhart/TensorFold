//! Synthetic mapped shards exercise the production gather, threading and chunk-history boundaries.
const std = @import("std");
const core = @import("core");
const ple = @import("cuda_ple.zig");
const a = std.testing.allocator;
const io = std.testing.io;
const base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";

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
        for ([_][]const u8{ "weight", "weight_scale" }, 0..) |part, component| {
            const cols: usize = if (component == 0) 8 else 1;
            const dtype = if (component == 0) "U8" else "F8_E4M3";
            const begin = data.items.len;
            for (0..rows) |row| {
                const id: usize = @intCast(h.starts[shard] + @as(i64, @intCast(row)));
                const code: u8 = if (component == 0) @intCast(id % 16 + ((id + 3) % 16) * 16) else 0x38;
                try data.appendNTimes(a, code, cols);
            }
            if (begin > 0) {
                try header.writer.writeByte(',');
                try index.writer.writeByte(',');
            }
            try header.writer.print("\"{s}.shard_{d}.{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ base, shard, part, dtype, rows, cols, begin, data.items.len });
            try index.writer.print("\"{s}.shard_{d}.{s}\":\"table.safetensors\"", .{ base, shard, part });
        }
    }
    if (include_scale) {
        const begin = data.items.len;
        const bytes = std.mem.toBytes(@as(u32, @bitCast(@as(f32, 0.375))));
        try data.appendSlice(a, &bytes);
        try header.writer.print(",\"{s}.weight_scale_2\":{{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[{d},{d}]}}", .{ base, begin, data.items.len });
        try index.writer.print(",\"{s}.weight_scale_2\":\"table.safetensors\"", .{base});
    }
    try header.writer.writeByte('}');
    try index.writer.writeAll("}}");
    var image: std.Io.Writer.Allocating = .init(a);
    defer image.deinit();
    const size = std.mem.toBytes(std.mem.nativeToLittle(u64, header.written().len));
    try image.writer.writeAll(&size);
    try image.writer.writeAll(header.written());
    try image.writer.writeAll(data.items);
    try dir.writeFile(io, .{ .sub_path = "table.safetensors", .data = image.written() });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
}

test "NVFP4 mapped PLE rows equal token decode across prompt chunks and threaded gathers" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture(tmp.dir, true);
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    var g = try ple.Ngram.init(a, io, path);
    defer g.deinit();
    try std.testing.expectEqual(@as(usize, 1), g.files.items.len);
    var tokens: [64]u32 = undefined;
    for (&tokens, 0..) |*v, i| v.* = @intCast((i * 7) % 100);
    var full: [64 * 32]u16 = undefined;
    var xs_full: [64]f32 = undefined;
    var history = [_]i64{99};
    try g.rows(&history, &tokens, &full, &xs_full);
    var chunked: [64 * 32]u16 = undefined;
    var xs_chunked: [64]f32 = undefined;
    var chunk_hist = [_]i64{99};
    try g.rows(&chunk_hist, tokens[0..17], chunked[0 .. 17 * 32], xs_chunked[0..17]);
    try g.rows(&chunk_hist, tokens[17..], chunked[17 * 32 ..], xs_chunked[17..]);
    try std.testing.expectEqualSlices(u16, &full, &chunked);
    try std.testing.expectEqualSlices(f32, &xs_full, &xs_chunked);
    var single_hist = [_]i64{99};
    const values = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    for (tokens, 0..) |token, row| {
        var ids: [2]i64 = undefined;
        try ple.lastIds(g.hash, &single_hist, token, &ids);
        var one: [32]u16 = undefined;
        var xs: [1]f32 = undefined;
        try g.rows(&single_hist, &.{token}, &one, &xs);
        try std.testing.expectEqualSlices(u16, full[row * 32 ..][0..32], &one);
        for (ids, 0..) |id, head| for (0..16) |col| {
            const code: usize = @intCast(@mod(id + @as(i64, if (col % 2 == 0) 0 else 3), 16));
            const v = values[code % 8] * @as(f32, if (code < 8) 1 else -1) * 0.375;
            try std.testing.expectEqual(core.quant.convert.bf16(v), one[head * 16 + col]);
        };
    }
    try std.testing.expectEqualSlices(i64, &history, &single_hist);
    try std.testing.expectEqualSlices(i64, &history, &chunk_hist);
}

test "NVFP4 PLE requires its global table scale" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture(tmp.dir, false);
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    try std.testing.expectError(error.MissingTensor, ple.Ngram.init(a, io, path));
}

test "PLE gather: NVFP4 pool and caller preserve every output byte at partition edges" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture(tmp.dir, true);
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    var g = try ple.Ngram.init(a, io, path);
    defer g.deinit();
    var tokens: [513]u32 = undefined;
    for (&tokens, 0..) |*v, i| v.* = @intCast((i * 11 + 3) % 100);
    var expected: [513 * 32]u16 = undefined;
    var actual: [513 * 32]u16 = undefined;
    var want_xs: [513]f32 = undefined;
    var got_xs: [513]f32 = undefined;
    for ([_]usize{ 0, 1, 3, 4, 5, 16, 17, 63, 64, 129, 513 }) |n| {
        var want_history = [_]i64{99};
        var got_history = want_history;
        {
            const parked = g.pool;
            g.pool = null;
            defer g.pool = parked;
            try g.rows(&want_history, tokens[0..n], expected[0 .. n * 32], want_xs[0..n]);
        }
        @memset(&actual, 0xdead);
        try g.rows(&got_history, tokens[0..n], actual[0 .. n * 32], got_xs[0..n]);
        try std.testing.expectEqualSlices(u16, expected[0 .. n * 32], actual[0 .. n * 32]);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_xs[0..n]), std.mem.sliceAsBytes(got_xs[0..n]));
        try std.testing.expectEqualSlices(i64, &want_history, &got_history);
        for (actual[n * 32 ..]) |v| try std.testing.expectEqual(@as(u16, 0xdead), v);
    }
}
