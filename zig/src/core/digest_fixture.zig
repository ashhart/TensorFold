//! Local deterministic checkpoint fixtures and order-independent cache record comparisons for CPU checks.
const std = @import("std");
const cache = @import("digest_cache.zig");

pub fn write(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, files: usize, payload: usize) !void {
    if (payload < 2 or payload % 2 != 0) return error.BadFixtureSize;
    const header = try std.fmt.allocPrint(a, "{{\"dense\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}},\"m.layers.1.mixer.experts.0.up_proj.weight\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}}}", .{ payload / 2, payload / 2, payload / 2, payload / 2, payload });
    defer a.free(header);
    const bytes = try a.alloc(u8, 8 + header.len + payload);
    defer a.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    var index: std.Io.Writer.Allocating = .init(a);
    defer index.deinit();
    var manifest: std.Io.Writer.Allocating = .init(a);
    defer manifest.deinit();
    try index.writer.writeAll("{\"weight_map\":{");
    try manifest.writer.writeAll("{\"files\":{");
    for (0..files) |i| {
        const name = try std.fmt.allocPrint(a, "model-{d}.safetensors", .{i});
        defer a.free(name);
        for (bytes[8 + header.len ..], 0..) |*b, at| b.* = @truncate(at *% 17 +% i);
        try dir.writeFile(io, .{ .sub_path = name, .data = bytes });
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
        const comma = if (i == 0) "" else ",";
        try index.writer.print("{s}\"tensor-{d}\":\"{s}\"", .{ comma, i, name });
        try manifest.writer.print("{s}\"{s}\":{{\"sha256\":\"{s}\"}}", .{ comma, name, &std.fmt.bytesToHex(hash, .lower) });
    }
    try index.writer.writeAll("}}");
    try manifest.writer.writeAll("}}");
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
    try dir.writeFile(io, .{ .sub_path = "tensorfold-export-manifest.json", .data = manifest.written() });
}

pub fn equalRecords(a: std.mem.Allocator, io: std.Io, first: []const u8, second: []const u8, count: usize) !void {
    var sets: [2]std.ArrayList(cache.Record) = undefined;
    for ([_][]const u8{ first, second }, &sets, 0..) |root, *set, i| {
        errdefer for (sets[0..i]) |*done| done.deinit(a);
        const path = try std.fs.path.join(a, &.{ root, "index.bin" });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(cache.max_bytes));
        defer a.free(bytes);
        set.* = try cache.decode(a, bytes);
    }
    defer for (&sets) |*set| set.deinit(a);
    if (sets[0].items.len != count or sets[1].items.len != count) return error.DifferentRecordCount;
    for (sets[0].items) |one| {
        var found = false;
        for (sets[1].items) |two| if (one.key.equal(two.key)) {
            if (!std.meta.eql(one.value, two.value)) return error.DifferentDigest;
            found = true;
        };
        if (!found) return error.MissingRecord;
    }
}
