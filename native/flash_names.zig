const std = @import("std");

pub fn accepts(raw: []const u8, drafts: bool) bool {
    if (std.mem.startsWith(u8, raw, "language_model.")) {
        if (std.mem.indexOf(u8, raw, ".mtp.") != null or std.mem.startsWith(u8, raw, "language_model.mtp"))
            return drafts and std.mem.startsWith(u8, raw, "language_model.mtp.");
        return true;
    }
    return drafts and std.mem.startsWith(u8, raw, "mtp.");
}

pub fn normalize(buffer: []u8, raw: []const u8) ![]const u8 {
    const name = if (std.mem.startsWith(u8, raw, "language_model.")) raw[15..] else raw;
    const alternate = "ngram_embedding.shards.";
    if (std.mem.indexOf(u8, name, alternate)) |at|
        return std.fmt.bufPrint(buffer, "{s}ngram_embedding.shard_{s}", .{ name[0..at], name[at + alternate.len ..] });
    return name;
}

pub fn resolve(map: anytype, buffer: []u8, name: []const u8) ![]const u8 {
    if (map.contains(name)) return name;
    if (std.mem.startsWith(u8, name, "language_model.mtp.") and map.contains(name[15..])) return name[15..];
    const original = "ngram_embedding.shard_";
    if (std.mem.indexOf(u8, name, original)) |at| {
        const alternate = try std.fmt.bufPrint(buffer, "{s}ngram_embedding.shards.{s}", .{ name[0..at], name[at + original.len ..] });
        if (map.contains(alternate)) return alternate;
    }
    return error.MissingWeight;
}

pub fn pleBase(map: anytype, buffer: []u8, shard: usize) ![]const u8 {
    for ([_][]const u8{ "shard_", "shards." }) |style| {
        const key = try std.fmt.bufPrint(buffer, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.{s}{d}.weight", .{ style, shard });
        if (map.contains(key)) return key[0 .. key.len - ".weight".len];
    }
    return error.MissingPLEWeight;
}

pub fn check(io: std.Io, dir: []const u8) !void {
    const mx = @import("mlx.zig");
    const cp = @import("checkpoint.zig");
    try mx.init();
    defer mx.shutdown();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/cases.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, indexed: bool, scale_error: bool = false, ple_error: ?[]const u8 = null };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    var arrays: usize = 0;
    for (cases.value) |case| {
        var folder_buffer: [4096]u8 = undefined;
        const folder = try std.fmt.bufPrint(&folder_buffer, "{s}/{s}", .{ dir, case.name });
        if (case.ple_error) |name| {
            const expected_error: anyerror = if (std.mem.eql(u8, name, "missing")) error.MissingPLEWeight else if (std.mem.eql(u8, name, "split")) error.SplitPLEWeight else if (std.mem.eql(u8, name, "format")) error.MixedPLEFormats else error.DuplicateWeight;
            var empty = cp.Store.init(32);
            defer empty.deinit();
            try std.testing.expectError(expected_error, @import("ple_tables.zig").Tables.checkAliases(io, folder, &empty));
            continue;
        }
        for ([_]bool{ false, true }) |drafts| {
            var actual = cp.Store.init(32);
            defer actual.deinit();
            actual.flash_drafts = drafts;
            if (case.scale_error) {
                try std.testing.expectError(error.UnsupportedPLEScale, actual.load(io, folder, "language_model."));
                continue;
            }
            try actual.load(io, folder, "language_model.");
            var expected = cp.Store.init(32);
            defer expected.deinit();
            try expected.loadFile(io, try std.fmt.bufPrint(&path, "{s}/expected-{d}.safetensors", .{ folder, @intFromBool(drafts) }), "", "");
            try std.testing.expectEqual(expected.arrays.count(), actual.arrays.count());
            var it = expected.arrays.iterator();
            while (it.next()) |entry| {
                var s = mx.Scope{};
                defer s.deinit();
                const loaded = try actual.get(entry.key_ptr.*);
                try std.testing.expectEqual(mx.dtype(entry.value_ptr.*), mx.dtype(loaded));
                try std.testing.expectEqualSlices(i32, mx.shape(entry.value_ptr.*), mx.shape(loaded));
                try mx.evalMany(&.{ entry.value_ptr.*, loaded }, false);
                const size = mx.c.mlx_array_size(loaded);
                if (mx.dtype(loaded) == mx.c.MLX_UINT32) {
                    try std.testing.expectEqualSlices(u32, mx.c.mlx_array_data_uint32(entry.value_ptr.*)[0..size], mx.c.mlx_array_data_uint32(loaded)[0..size]);
                } else if (mx.dtype(loaded) == mx.c.MLX_INT64) {
                    try std.testing.expectEqualSlices(i64, mx.c.mlx_array_data_int64(entry.value_ptr.*)[0..size], mx.c.mlx_array_data_int64(loaded)[0..size]);
                } else try @import("sampling_checks.zig").equal(&s, entry.value_ptr.*, loaded);
                arrays += 1;
            }
            if (drafts) {
                var ple = cp.Store.init(32);
                defer ple.deinit();
                try ple.loadFile(io, try std.fmt.bufPrint(&path, "{s}/ple.safetensors", .{folder}), "", "");
                try @import("ple_tables.zig").Tables.checkAliases(io, folder, &ple);
            }
        }
    }
    std.debug.print("PASS: {d} Flash checkpoint layouts/rejections, {d} sanitized tensors, both MTP prefixes and bounded/resident PLE aliases match upstream\n", .{ cases.value.len, arrays });
}

test "Flash checkpoint aliases preserve PLE and MTP identities" {
    var buffer: [512]u8 = undefined;
    for ([_][]const u8{ "language_model.mtp.fc_hidden.weight", "mtp.fc_hidden.weight" }) |name| {
        try std.testing.expect(accepts(name, true));
        try std.testing.expect(!accepts(name, false));
        try std.testing.expectEqualStrings("mtp.fc_hidden.weight", try normalize(&buffer, name));
    }
    for ([_][]const u8{ "visual.weight", "other.mtp.weight", "language_model.other.mtp.weight", "language_model.mtp_unused.weight" }) |name|
        try std.testing.expect(!accepts(name, true));
    try std.testing.expect(accepts("language_model.model.norm.weight", false));
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
        \\{"mtp.fc_hidden.weight":"mtp.safetensors","language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.127.weight":"ple.safetensors"}
    , .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("mtp.fc_hidden.weight", try resolve(parsed.value.object, &buffer, "language_model.mtp.fc_hidden.weight"));
    const ple = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight";
    const alternate = try resolve(parsed.value.object, &buffer, ple);
    var normalized: [512]u8 = undefined;
    try std.testing.expectEqualStrings(ple[15..], try normalize(&normalized, alternate));
    try std.testing.expectEqualStrings("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.127", try pleBase(parsed.value.object, &buffer, 127));
    try std.testing.expectError(error.MissingPLEWeight, pleBase(parsed.value.object, &buffer, 0));
    try std.testing.expectError(error.MissingWeight, resolve(parsed.value.object, &buffer, "language_model.mtp.missing.weight"));
}
