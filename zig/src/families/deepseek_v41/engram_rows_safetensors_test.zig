//! The streamed Engram reader against the resident one: the same release bytes, row for row.

const std = @import("std");
const testing = std.testing;
const sr = @import("engram_rows_safetensors.zig");
const f = @import("engram_rows.zig");

const layer: u32 = 1;
const row_count: u64 = 300;
const shard = "model-00047-of-00048.safetensors";
const row_bytes: u64 = f.values + f.values / 32;
const weight_bytes: u64 = row_count * f.values;

/// Deterministic bytes, distinct per row and column, so a wrong offset or a swapped tensor shows up.
fn weightByte(r: u64, j: u64) u8 {
    return @truncate((r * 37 + j * 11 + 5) % 251 + 1);
}
fn scaleByte(r: u64, j: u64) u8 {
    return @truncate(120 + (r * 7 + j * 3) % 8);
}

/// The release's layout: one layer a shard, weight tensor first, scale tensor after it.
fn writeShard(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, scale_dtype: []const u8) !void {
    const data = try a.alloc(u8, @intCast(weight_bytes + row_count * 8));
    defer a.free(data);
    var r: u64 = 0;
    while (r < row_count) : (r += 1) {
        var j: u64 = 0;
        while (j < f.values) : (j += 1) data[r * f.values + j] = weightByte(r, j);
        j = 0;
        while (j < 8) : (j += 1) data[@intCast(weight_bytes + r * 8 + j)] = scaleByte(r, j);
    }
    const header = try std.fmt.allocPrint(a, "{{\"layers.{d}.engram.embed.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}," ++
        "\"layers.{d}.engram.embed.scale\":{{\"dtype\":\"{s}\",\"shape\":[{d},8],\"data_offsets\":[{d},{d}]}}}}", .{ layer, row_count, f.values, weight_bytes, layer, scale_dtype, row_count, weight_bytes, weight_bytes + row_count * 8 });
    defer a.free(header);

    // One buffer: the 8-byte header length, the JSON header, then the two tensors.
    const blob = try a.alloc(u8, 8 + header.len + data.len);
    defer a.free(blob);
    var head: [8]u8 = undefined;
    std.mem.writeInt(u64, &head, header.len, .little);
    @memcpy(blob[0..8], &head);
    @memcpy(blob[8..][0..header.len], header);
    @memcpy(blob[8 + header.len ..][0..data.len], data);
    try dir.writeFile(io, .{ .sub_path = shard, .data = blob });
}

/// The resident path's view of the same bytes: the oracle a streamed row must equal.
fn oracle(a: std.mem.Allocator) !f.Table {
    const weights = try a.alloc(u8, @intCast(weight_bytes));
    const scales = try a.alloc(u8, @intCast(row_count * 8));
    var r: u64 = 0;
    while (r < row_count) : (r += 1) {
        var j: u64 = 0;
        while (j < f.values) : (j += 1) weights[r * f.values + j] = weightByte(r, j);
        j = 0;
        while (j < 8) : (j += 1) scales[@intCast(r * 8 + j)] = scaleByte(r, j);
    }
    return .{ .weights = weights, .scales = scales, .rows = row_count };
}

test "safetensors rows: a streamed row decodes to the resident table's own values, and repeats come from the cache" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, testing.io, tmp.dir, "F8_E8M0");
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);

    const rows = try sr.Rows.open(a, testing.io, dir, 0, 1, &.{layer}, 64);
    defer rows.close();

    const table = try oracle(a);
    defer a.free(table.weights);
    defer a.free(table.scales);

    // Unsorted, with a repeat and the row at each end of the shard.
    const ids = [_]i64{ 7, 0, 199, 7, 42, 299, 128, 199 };
    var out: [ids.len * f.values]u16 = undefined;
    try rows.rows(0, &ids, f.values, &out);

    for (ids, 0..) |id, i| {
        var want: [f.values]u16 = undefined;
        try table.row(@intCast(id), &want);
        try testing.expectEqualSlices(u16, &want, out[i * f.values ..][0..f.values]);
    }
    // Six distinct rows miss; the two repeats hit.
    try testing.expectEqual(@as(u64, 6), rows.stats.misses);
    try testing.expectEqual(@as(u64, 2), rows.stats.hits);
    try testing.expectEqual(@as(u64, 12), rows.stats.reads);
    try testing.expectEqual(@as(u64, 6 * row_bytes), rows.stats.bytes);

    // The same window again: every row is resident, so nothing reaches the shard and the shard is not touched.
    try rows.rows(0, &ids, f.values, &out);
    try testing.expectEqual(@as(u64, 10), rows.stats.hits);
    try testing.expectEqual(@as(u64, 6), rows.stats.misses);
    try testing.expectEqual(@as(u64, 12), rows.stats.reads);
}

test "safetensors rows: the GPU's u32 ids take the same path as the i64 ones" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, testing.io, tmp.dir, "F8_E8M0");
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);

    const rows = try sr.Rows.open(a, testing.io, dir, 0, 1, &.{layer}, 8);
    defer rows.close();
    const table = try oracle(a);
    defer a.free(table.weights);
    defer a.free(table.scales);

    const ids = [_]u32{ 3, 250, 17, 3 };
    var out: [ids.len * f.values]u16 = undefined;
    try rows.rowsU32(0, &ids, f.values, &out);
    for (ids, 0..) |id, i| {
        var want: [f.values]u16 = undefined;
        try table.row(id, &want);
        try testing.expectEqualSlices(u16, &want, out[i * f.values ..][0..f.values]);
    }
}

test "safetensors rows: a cache smaller than the window reports every row a miss, and no row is served stale" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, testing.io, tmp.dir, "F8_E8M0");
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);

    const rows = try sr.Rows.open(a, testing.io, dir, 0, 1, &.{layer}, 8);
    defer rows.close();
    const table = try oracle(a);
    defer a.free(table.weights);
    defer a.free(table.scales);

    var ids: [40]i64 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i * 7 % row_count);
    var out: [ids.len * f.values]u16 = undefined;
    try rows.rows(0, &ids, f.values, &out);
    try testing.expectEqual(@as(u64, 0), rows.stats.hits);
    try testing.expectEqual(@as(u64, 40), rows.stats.misses);
    for (ids, 0..) |id, i| {
        var want: [f.values]u16 = undefined;
        try table.row(@intCast(id), &want);
        try testing.expectEqualSlices(u16, &want, out[i * f.values ..][0..f.values]);
    }
}

test "safetensors rows: an id past the shard is refused, not read" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, testing.io, tmp.dir, "F8_E8M0");
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);

    const rows = try sr.Rows.open(a, testing.io, dir, 0, 1, &.{layer}, 4);
    defer rows.close();
    var out: [2 * f.values]u16 = undefined;
    try testing.expectError(error.EngramRowOutsideShard, rows.rows(0, &.{ 0, @intCast(row_count) }, f.values, &out));
}

test "safetensors rows: a scale tensor that is not UE8M0 is refused at open" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, testing.io, tmp.dir, "F8_E4M3");
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);

    try testing.expectError(error.UnexpectedEngramTensor, sr.Rows.open(a, testing.io, dir, 0, 1, &.{layer}, 4));
}

test "safetensors rows: an unknown Engram layer names itself rather than reading a shard it does not have" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, testing.io, tmp.dir, "F8_E8M0");
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);

    try testing.expectError(error.UnknownEngramLayer, sr.Rows.open(a, testing.io, dir, 0, 1, &.{2}, 4));
}

test "safetensors rows: the mapped shard serves exactly what the positional reads serve" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(a, testing.io, tmp.dir, "F8_E8M0");
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);

    const mapped = try sr.Rows.openWithMap(a, testing.io, dir, 0, 1, &.{layer}, 64, true);
    defer mapped.close();
    const positional = try sr.Rows.openWithMap(a, testing.io, dir, 0, 1, &.{layer}, 64, false);
    defer positional.close();
    try testing.expect(mapped.mapped());
    try testing.expect(!positional.mapped());

    // Unsorted, with repeats, over the whole shard.
    var ids: [600]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 173 + 7) % row_count);
    var out_map: [ids.len * f.values]u16 = undefined;
    var out_pos: [ids.len * f.values]u16 = undefined;

    try mapped.rowsU32(0, &ids, f.values, &out_map);
    try positional.rowsU32(0, &ids, f.values, &out_pos);
    try testing.expectEqualSlices(u16, &out_map, &out_pos);

    // The threaded bulk path too: the same bytes, whichever worker decodes them.
    try mapped.rowsU32Parallel(0, &ids, f.values, &out_map, 4);
    try positional.rowsU32Parallel(0, &ids, f.values, &out_pos, 4);
    try testing.expectEqualSlices(u16, &out_map, &out_pos);

    // The mapped gather is the same rows, and it asks the kernel for no syscall a row.
    try testing.expectEqual(@as(u64, 0), mapped.stats.syscalls);
    try testing.expect(positional.stats.syscalls > 0);

    // And every row is the resident table's own values.
    const table = try oracle(a);
    defer a.free(table.weights);
    defer a.free(table.scales);
    for (ids, 0..) |id, i| {
        var want: [f.values]u16 = undefined;
        try table.row(id, &want);
        try testing.expectEqualSlices(u16, &want, out_map[i * f.values ..][0..f.values]);
    }
}
