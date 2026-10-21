//! Deterministic malformed-header sweeps through the core and cluster checkpoint entry points.
const std = @import("std");
const st = @import("core").safetensors;
const ck = @import("cluster_checkpoint");
const a = std.testing.allocator;

fn refused(result: anytype) !void {
    if (result) |_| return error.MalformedHeaderAccepted else |_| {}
}

fn image(arena: std.mem.Allocator, json: []const u8, payload: usize) ![]u8 {
    const bytes = try arena.alloc(u8, 8 + json.len + payload);
    std.mem.writeInt(u64, bytes[0..8], json.len, .little);
    @memcpy(bytes[8..][0..json.len], json);
    @memset(bytes[8 + json.len ..], 0);
    return bytes;
}

test "safetensors shared: every truncated header and payload is refused by both entry points" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    const json = "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2,8],\"data_offsets\":[0,32]}}";
    const bytes = try image(mem, json, 32);
    for (0..json.len) |end| try refused(st.parseHeader(mem, json[0..end], 32));
    for (0..bytes.len) |end| {
        try refused(ck.parseHeader(mem, 0, bytes[0..end]));
        if (end >= 8 + json.len) try refused(st.parseHeader(mem, json, end - 8 - json.len));
    }
    _ = try ck.parseHeader(mem, 0, bytes);
    _ = try st.parseHeader(mem, json, 32);
}

test "safetensors shared: inconsistent ranges and oversized dimensions have one validation policy" {
    const cases = [_][]const u8{
        "[]",
        "{\"w\":0}",
        "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,3]}}",
        "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[1,5]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[1,0]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[-1],\"data_offsets\":[0,0]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[18446744073709551615,2],\"data_offsets\":[0,0]}}",
        "{\"w\":{\"dtype\":\"BF16\",\"shape\":[16777216,16777216,32768,0],\"data_offsets\":[0,0]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[18446744073709551616],\"data_offsets\":[0,0]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[18446744073709551615,18446744073709551615]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[1,1,1,1,1,1],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"UNKNOWN\",\"shape\":[],\"data_offsets\":[0,1]}}",
    };
    for (cases) |json| {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        try refused(st.parseHeader(arena.allocator(), json, 4));
        try refused(ck.parseHeader(arena.allocator(), 0, try image(arena.allocator(), json, 4)));
    }
}

test "safetensors shared: valid dtype and shape sweeps preserve tensor bytes names order and roles" {
    const dtypes = [_][]const u8{ "BOOL", "U8", "I8", "U16", "I16", "BF16", "F16", "U32", "I32", "F32", "U64", "I64", "F64", "F8_E4M3", "F8_E5M2", "F8_E8M0" };
    for (dtypes) |dt| for (0..9) |rows| for (0..5) |rank| {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const mem = arena.allocator();
        const shape = [_]u64{ rows, 2, 1, 1 };
        const bytes = try ck.image(mem, &.{.{ .name = "model.layers.3.mlp.experts.7.down_proj.weight", .dtype = dt, .shape = shape[0..rank], .fill = 42 }});
        const n: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
        const core = try st.parseHeader(mem, bytes[8..][0..n], bytes.len - 8 - n);
        const cluster = try ck.parseHeader(mem, 5, bytes);
        try std.testing.expectEqual(@as(usize, 1), cluster.len);
        const c = cluster[0];
        const entry = core.values()[0];
        try std.testing.expectEqualStrings(core.keys()[0], c.name);
        try std.testing.expectEqual(@as(u64, 8 + n + entry.begin), c.start);
        try std.testing.expectEqual(@as(u64, entry.end - entry.begin), c.bytes);
        try std.testing.expectEqual(ck.dtypeOf(dt).?, c.dtype);
        try std.testing.expectEqual(@as(u32, 5), c.file);
        try std.testing.expectEqual(entry.rank, c.rank);
        for (entry.shape[0..entry.rank], c.shape[0..c.rank]) |x, y| try std.testing.expectEqual(@as(u64, x), y);
        try std.testing.expectEqual(@as(?u32, 7), c.class.expert);
        for (bytes[c.start..][0..c.bytes]) |b| try std.testing.expectEqual(@as(u8, 42), b);
    };
}

test "safetensors shared: length prefix extremes and file reads refuse before header allocation" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = "" });
    const path = try tmp.dir.realPathFileAlloc(std.testing.io, "model.safetensors", a);
    defer a.free(path);
    for ([_]u64{ 1, 8, 64 << 20, 1 << 30, std.math.maxInt(u64) }) |n| {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, n, .little);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = &bytes });
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
        try std.testing.expectError(error.BadSafetensors, st.File.open(failing.allocator(), std.testing.io, path));
        try std.testing.expectError(error.BadHeader, ck.parseHeader(failing.allocator(), 0, &bytes));
        try std.testing.expect(!failing.has_induced_failure);
    }
}

test "safetensors shared: header-only disk loading keeps names offsets and file bounds" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = try ck.image(mem, &.{.{ .name = "w", .dtype = "BF16", .shape = &.{2}, .fill = 42 }});
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = bytes });
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", mem);
    const loaded = try ck.load(std.testing.io, mem, dir);
    try std.testing.expectEqual(@as(usize, 1), loaded.tensors.len);
    const n: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    const copied = try ck.parseHeader(mem, 0, bytes);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = bytes[0 .. bytes.len - 1] });
    const path = try std.fs.path.join(mem, &.{ dir, "model.safetensors" });
    try std.testing.expectError(error.BadSafetensors, st.File.open(mem, std.testing.io, path));
    try std.testing.expectError(error.BadHeader, ck.load(std.testing.io, mem, dir));
    @memset(bytes[8..][0..n], ' ');
    try std.testing.expectEqualStrings("w", copied[0].name);
    try std.testing.expectEqual(@as(u64, bytes.len), loaded.files[0].size);
    try std.testing.expectEqual(@as(u64, 8 + n), loaded.tensors[0].start);
    try std.testing.expectEqual(@as(u64, 4), loaded.tensors[0].bytes);
}
