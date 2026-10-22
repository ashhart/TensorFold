const std = @import("std");
const ck = @import("checkpoint.zig");
const edit = @import("shard_edit.zig");
const originals = @import("originals_files.zig");

fn list(dir: []const u8) !void {
    const files = try ck.shardFiles(std.testing.allocator, std.testing.io, dir);
    ck.freeShardFiles(std.testing.allocator, files);
}

test "security: malformed checkpoint indexes return an error without trapping" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const invalid = [_][]const u8{
        "[]",                          "null",                          "{}", "{\"weight_map\":[]}", "{\"weight_map\":null}",
        "{\"weight_map\":{\"a\":42}}", "{\"weight_map\":{\"a\":null}}",
    };
    for (invalid) |json| {
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = json });
        try std.testing.expectError(error.BadIndex, list(dir));
    }
}

test "security: checkpoint shard names cannot escape their model folder" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const invalid = [_][]const u8{
        "../outside.safetensors",     "/tmp/outside.safetensors", "sub/../outside.safetensors",
        "..\\outside.safetensors",    "C:\\outside.safetensors",  "C:outside.safetensors",
        "model.safetensors\x00extra", "",                         ".",
        "..",
    };
    for (invalid) |name| {
        const json = try std.json.Stringify.valueAlloc(a, .{ .weight_map = .{ .a = name } }, .{});
        defer a.free(json);
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = json });
        try std.testing.expectError(error.BadIndex, list(dir));
    }
}

test "security: a checkpoint shard may sit in a folder inside its model folder" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "sub");
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const shard = try std.fs.path.join(a, &.{ dir, "sub", "model.safetensors" });
    defer a.free(shard);
    try edit.create(a, shard, &.{}, &.{.{ .name = "a", .dtype = .u8, .shape = &.{4}, .bytes = "here" }});
    try tmp.dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"weight_map\":{\"a\":\"sub/model.safetensors\"}}",
    });
    const files = try ck.shardFiles(a, io, dir);
    defer ck.freeShardFiles(a, files);
    try std.testing.expectEqual(@as(usize, 1), files.len);
    try std.testing.expectEqualStrings("sub/model.safetensors", files[0][dir.len + 1 ..]);
    var model = try ck.Checkpoint.openModel(a, io, dir);
    defer model.close();
    try std.testing.expectEqualStrings("here", (try model.get("a")).bytes);
}

test "security: an unreadable checkpoint index is not silently ignored" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "unused" });
    try tmp.dir.createDirPath(io, "model.safetensors.index.json");
    if (list(dir)) |_| return error.TestUnexpectedResult else |_| {}
}

test "security: a checkpoint index cannot redirect Living Weights into another folder" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "model");
    const outside = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(outside);
    const dir = try std.fs.path.join(a, &.{ outside, "model" });
    defer a.free(dir);
    const victim = try std.fs.path.join(a, &.{ outside, "victim.safetensors" });
    defer a.free(victim);
    try edit.create(a, victim, &.{}, &.{.{ .name = "a.weight", .dtype = .u8, .shape = &.{4}, .bytes = "safe" }});
    const before = try tmp.dir.readFileAlloc(io, "victim.safetensors", a, .limited(4096));
    defer a.free(before);
    try tmp.dir.writeFile(io, .{
        .sub_path = "model/model.safetensors.index.json",
        .data = "{\"weight_map\":{\"a.weight\":\"../victim.safetensors\"}}",
    });
    const changed = edit.overwrite(a, io, dir, &.{.{ .name = "a.weight", .dtype = .u8, .shape = &.{4}, .bytes = "evil" }});
    const after = try tmp.dir.readFileAlloc(io, "victim.safetensors", a, .limited(4096));
    defer a.free(after);
    try std.testing.expectEqualSlices(u8, before, after);
    try std.testing.expectError(error.BadIndex, changed);
}

test "security: Living Weights refuse an index with a shard in a folder before writing to it" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "sub");
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const shard = try std.fs.path.join(a, &.{ dir, "sub", "model.safetensors" });
    defer a.free(shard);
    try edit.create(a, shard, &.{}, &.{.{ .name = "a.weight", .dtype = .u8, .shape = &.{4}, .bytes = "safe" }});
    const before = try tmp.dir.readFileAlloc(io, "sub/model.safetensors", a, .limited(4096));
    defer a.free(before);
    try tmp.dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"weight_map\":{\"a.weight\":\"sub/model.safetensors\"}}",
    });
    const evil: []const edit.Replacement = &.{.{ .name = "a.weight", .dtype = .u8, .shape = &.{4}, .bytes = "evil" }};
    try std.testing.expectError(error.BadIndex, edit.overwrite(a, io, dir, evil));
    try std.testing.expectError(error.BadIndex, edit.grow(a, io, dir, evil));
    try std.testing.expectError(error.BadIndex, edit.bake(a, io, dir, evil));
    const after = try tmp.dir.readFileAlloc(io, "sub/model.safetensors", a, .limited(4096));
    defer a.free(after);
    try std.testing.expectEqualSlices(u8, before, after);
}

test "security: Living Weights keep to the model folder's own shards, so a linked folder cannot redirect them" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "model");
    try tmp.dir.createDirPath(io, "outside");
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const dir = try std.fs.path.join(a, &.{ root, "model" });
    defer a.free(dir);
    const victim = try std.fs.path.join(a, &.{ root, "outside", "victim.safetensors" });
    defer a.free(victim);
    try edit.create(a, victim, &.{}, &.{.{ .name = "a.weight", .dtype = .u8, .shape = &.{4}, .bytes = "safe" }});
    const before = try tmp.dir.readFileAlloc(io, "outside/victim.safetensors", a, .limited(4096));
    defer a.free(before);
    try tmp.dir.symLink(io, "../outside", "model/sub", .{ .is_directory = true });
    try tmp.dir.writeFile(io, .{
        .sub_path = "model/model.safetensors.index.json",
        .data = "{\"weight_map\":{\"a.weight\":\"sub/victim.safetensors\"}}",
    });
    const changed = edit.overwrite(a, io, dir, &.{.{ .name = "a.weight", .dtype = .u8, .shape = &.{4}, .bytes = "evil" }});
    const after = try tmp.dir.readFileAlloc(io, "outside/victim.safetensors", a, .limited(4096));
    defer a.free(after);
    try std.testing.expectEqualSlices(u8, before, after);
    try std.testing.expectError(error.BadIndex, changed);
}

test "security: shard and snapshot atomic writes cannot follow preexisting temporary symlinks" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "outside", .data = "preserve unrelated content" });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    try tmp.dir.symLink(io, "outside", "config.json.tmp", .{});
    try edit.setConfig(a, io, dir, &.{.{ "learned", 1 }});
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("preserve unrelated content", try tmp.dir.readFile(io, "outside", &buf));
    try std.testing.expectEqual(@as(?i64, 1), try edit.configInt(a, io, dir, "learned"));
    const target = try std.fs.path.join(a, &.{ dir, "snapshot.json" });
    defer a.free(target);
    try tmp.dir.symLink(io, "outside", "snapshot.json.tmp", .{});
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    try originals.writeWhole(arena.allocator(), target, "snapshot");
    try std.testing.expectEqualStrings("preserve unrelated content", try tmp.dir.readFile(io, "outside", &buf));
    try std.testing.expectEqualStrings("snapshot", try tmp.dir.readFile(io, "snapshot.json", &buf));
}
