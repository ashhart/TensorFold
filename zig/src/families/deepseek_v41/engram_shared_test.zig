//! Independent hash/layout and FP64-ideal row equations adapted from @akol1's #826 reference.
const std = @import("std");
const t = std.testing;
const hash = @import("engram_hash.zig");
const eh = @import("engram_host.zig");
const er = @import("engram_rows.zig");
const portable = @import("engram_rows_portable.zig");
const fx = @import("engram_fixture.zig");
fn e4Ideal(code: u8) f64 {
    const mag = code & 127;
    if (mag == 127) return std.math.nan(f64);
    const exponent = mag >> 3;
    const mantissa = mag & 7;
    const value = if (exponent == 0) @as(f64, @floatFromInt(mantissa)) / 512 else std.math.pow(f64, 2, @as(f64, @floatFromInt(exponent)) - 7) * (1 + @as(f64, @floatFromInt(mantissa)) / 8);
    return if (code & 128 != 0) -value else value;
}
fn bf16Ideal(x: f64) u16 {
    const rounded: f32 = @floatCast(x);
    if (std.math.isNan(rounded)) return 0xffff;
    const bits: u32 = @bitCast(rounded);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
test "all 65536 row codes agree with the independent FP64 ideal through the BF16 contract" {
    const lut = try t.allocator.create([65536]u16);
    defer t.allocator.destroy(lut);
    er.fillTable(lut);
    for (0..256) |scale| for (0..256) |code| {
        const multiplier = if (scale == 255) std.math.nan(f64) else std.math.pow(f64, 2, @as(f64, @floatFromInt(scale)) - 127);
        try t.expectEqual(bf16Ideal(e4Ideal(@intCast(code)) * multiplier), lut[scale * 256 + code]);
    };
}
test "canonical history and chain lookback use one hash for both released layers and DEAD boundaries" {
    const a = t.allocator;
    const bytes = try a.alloc(u8, hash.raw_vocab * 4);
    defer a.free(bytes);
    for (0..hash.raw_vocab) |id| std.mem.writeInt(u32, bytes[id * 4 ..][0..4], @intCast(id % hash.compressed_vocab), .little);
    const map: hash.TokenMap = .{ .bytes = bytes }; // Synthetic map: no claim of the pinned tokenizer checksum.
    var layers: [2]u32 = undefined;
    const host = try eh.Host.fromTokenMap(map, &layers);
    const ids = [_]u32{ 20, 800, 2, 129279, 99, 100, 200 };
    var got: [ids.len * 2 * 24]i64 = undefined;
    host.hash(&ids, &.{}, &got);
    for (hash.layers, 0..) |meta, li| {
        var history: hash.History = .{};
        var out: [ids.len][24]u32 = undefined;
        try hash.hashChunk(&history, map, meta, ids[0..3], null, out[0..3]);
        const saved = history.snapshot();
        try hash.hashChunk(&history, map, meta, ids[3..], null, out[3..]);
        for (out, 0..) |row, i| for (row, 0..) |value, col| try t.expectEqual(@as(i64, value), got[(i * 2 + li) * 24 + col]);
        history.rollback(saved);
        const masked = try history.push(hash.dead, meta);
        var expected: [24]u32 = undefined;
        hash.hashValues(&[_]u32{ 2, 2, 2, 2 }, &meta.multipliers, &meta.primes, &meta.offsets, 8, &expected);
        try t.expectEqual(expected, masked);
        const before = history;
        try t.expectError(error.InvalidEngramToken, hash.hashChunk(&history, map, meta, &.{hash.raw_vocab}, null, out[0..1]));
        try t.expectEqualDeep(before, history);
    }
}
test "packed shard rows and resident separate arrays agree, including LRU eviction" {
    const a = t.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var unused: [32]u8 = undefined;
    const fixture = try fx.parseFx(arena.allocator(), &unused);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try fx.writeShard(a, t.io, tmp.dir, fixture);
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(dir);
    const rows = try portable.Rows.openWithIo(a, t.io, dir, 0, 2, &.{1}, 4);
    defer rows.close();
    const packed_rows = try tmp.dir.readFileAlloc(t.io, "engram-l1-r0of2.bin", a, .limited(1 << 24));
    defer a.free(packed_rows);
    const weight = try a.alloc(u8, @as(usize, @intCast(fixture.rows)) * 271);
    defer a.free(weight);
    const scales = try a.alloc(u8, @as(usize, @intCast(fixture.rows)) * 13);
    defer a.free(scales);
    @memset(weight, 0xdd);
    @memset(scales, 0xdd);
    for (0..fixture.rows) |i| {
        const record = packed_rows[er.header_bytes + i * er.row_bytes ..][0..er.row_bytes];
        @memcpy(weight[i * 271 ..][0..256], record[0..256]);
        @memcpy(scales[i * 13 ..][0..8], record[256..264]);
    }
    const resident: er.Table = .{ .weights = weight, .scales = scales, .rows = @intCast(fixture.rows), .weight_stride = 271, .scale_stride = 13 };
    for (fixture.sets) |set| {
        const out = try a.alloc(u16, set.idx.len * 256);
        defer a.free(out);
        try rows.source().vtable.rows(rows.source().ptr, 0, set.idx, 256, out);
        for (set.idx, 0..) |id, j| {
            var resident_row: [256]u16 = undefined;
            try resident.row(@intCast(@as(u64, @intCast(id)) - fixture.lo), &resident_row);
            try t.expectEqualSlices(u16, &resident_row, out[j * 256 ..][0..256]);
        }
        try t.expectEqual(set.sha, fx.sha(out));
        try rows.issue(0, set.idx);
        try rows.rows(0, set.idx, 256, out);
        try t.expectEqual(set.sha, fx.sha(out));
    }
    var output: [256]u16 = undefined;
    try t.expectError(error.EngramRowOutsideShard, rows.rows(0, &.{@intCast(fixture.lo - 1)}, 256, &output));
}
test "legacy metadata supports unaligned little endian bytes and rejects every truncated prefix" {
    var blob: [24 + 4 + 16 + 16 + 1]u8 = @splat(0);
    const b = blob[1..];
    @memcpy(b[0..8], "DSV41EH1");
    std.mem.writeInt(u32, b[8..12], 1, .little);
    std.mem.writeInt(u32, b[12..16], 2, .little);
    std.mem.writeInt(u32, b[16..20], 4, .little);
    std.mem.writeInt(u32, b[24..28], 1, .little);
    std.mem.writeInt(i64, b[28..36], 3, .little);
    std.mem.writeInt(i64, b[36..44], 5, .little);
    std.mem.writeInt(u32, b[52..56], 2, .little);
    var layers: [4]u32 = undefined;
    for (0..b.len) |n| try t.expectError(error.BadEngramHost, eh.Host.loadLegacyFixture(b[0..n], 1, 2, 31, &layers));
    const host = try eh.Host.loadLegacyFixture(b, 1, 2, 31, &layers);
    try t.expectEqual(@as(i64, 2), host.pad_id);
    try t.expectError(error.InvalidEngramMetadata, hash.TokenMap.parse(b));
}
