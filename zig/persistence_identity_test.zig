//! Real checkpoint metadata and persisted namespace checks; no model, device or shader is opened.
const std = @import("std");
const disk = @import("src/native/disk_identity.zig");
const glm = @import("src/families/glm/load_plan.zig");
const pack = @import("src/families/flashnext/pack_io.zig");
const Imprint = @import("engine_api").prompt_imprint.Imprint;
const binary = @import("sys").file_identity;
const t = std.testing;
const io = t.io;
const Kind = enum { glm, flashnext };
const config = "{}";
const index = "{\"weight_map\":{\"unprobed.weight\":\"model.safetensors\"}}";
const header = "{\"unprobed.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";

fn legacy(a: std.mem.Allocator, dir: []const u8, kind: Kind) !u64 {
    if (kind == .glm) return glm.modelHash(a, dir, config);
    const text = try pack.sourceIdentity(a, io, dir);
    defer a.free(text);
    return std.hash.Wyhash.hash(0x6964, text);
}

fn allocatingScope(a: std.mem.Allocator, dir: []const u8) !void {
    _ = try disk.scoped(a, io, dir, 7);
}

fn changedWeightMiss(kind: Kind) !void {
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const root = try std.fs.path.join(a, &.{ dir, "states" });
    defer a.free(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    var shard: [8 + header.len + 4]u8 = @splat(0);
    std.mem.writeInt(u64, shard[0..8], header.len, .little);
    @memcpy(shard[8..][0..header.len], header);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &shard });
    const old = try legacy(a, dir, kind);
    if (kind == .glm) try t.checkAllAllocationFailures(a, allocatingScope, .{dir});
    const before = try disk.scoped(a, io, dir, old);
    try t.expectEqual(before, try disk.scoped(a, io, dir, old));
    const tokens = [_]u32{ 7, 11, 13 };
    const key = Imprint.keyOf(&tokens);
    {
        var state = try Imprint.open(a, root, before, 1 << 20);
        defer state.deinit();
        try state.add(key, 2, &tokens, &.{}, 4);
        try t.expect(state.best(&tokens, &.{}, false, 0) != null);
    }
    // An unvisited tensor can change without changing a fixed probe, config, index or shard header.
    try std.Io.sleep(io, .fromMilliseconds(5), .awake);
    shard[shard.len - 1] = 1;
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &shard });
    try t.expectEqual(old, try legacy(a, dir, kind));
    const after = try disk.scoped(a, io, dir, old);
    var fresh = try Imprint.open(a, root, after, 1 << 20);
    defer fresh.deinit();
    try t.expectEqual(@as(usize, 0), fresh.metas.items.len);
    try t.expect(fresh.best(&tokens, &.{}, false, 0) == null);
    var preserved = try Imprint.open(a, root, before, 1 << 20);
    defer preserved.deinit();
    try t.expect(preserved.best(&tokens, &.{}, false, 0) != null);
}

test "persistence identity: GLM same-header weight rewrite misses the old serving namespace" {
    try changedWeightMiss(.glm);
}

test "persistence identity: Flash Next same-header weight rewrite misses the old serving namespace" {
    try changedWeightMiss(.flashnext);
}

test {
    _ = @import("src/native/variant_identity.zig");
}

test "persistence identity: changed serving binary bytes move an otherwise identical namespace" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const path = try std.fs.path.join(a, &.{ dir, "server" });
    defer a.free(path);
    try tmp.dir.writeFile(io, .{ .sub_path = "server", .data = "compiled code one" });
    const first = try binary.hash(io, path, 1024);
    try std.testing.expectEqual(first, try binary.hash(io, path, 1024));
    try std.testing.expectError(error.IdentityFileTooLarge, binary.hash(io, path, 1));
    var before = std.hash.Wyhash.init(17);
    binary.bind(&before, first);
    const root = try std.fs.path.join(a, &.{ dir, "kept" });
    defer a.free(root);
    const ids = [_]u32{ 5, 7, 11 };
    var kept = try Imprint.open(a, root, before.final(), 1 << 20);
    defer kept.deinit();
    try kept.add(Imprint.keyOf(&ids), 2, &ids, &.{}, 4);
    try tmp.dir.writeFile(io, .{ .sub_path = "server", .data = "compiled code two" });
    var after = std.hash.Wyhash.init(17);
    binary.bind(&after, try binary.hash(io, path, 1024));
    try std.testing.expect(before.final() != after.final());
    var next = try Imprint.open(a, root, after.final(), 1 << 20);
    defer next.deinit();
    try t.expect(next.best(&ids, &.{}, false, 0) == null);
    try t.expect(kept.best(&ids, &.{}, false, 0) != null);
}

test "persistence identity: file fingerprints cover every block and agree with the whole-file SHA256" {
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const path = try std.fs.path.join(a, &.{ dir, "code" });
    defer a.free(path);
    const data = try a.alloc(u8, 2 * 32768 + 17);
    defer a.free(data);
    for (data, 0..) |*value, i| value.* = @truncate(i *% 131);
    try tmp.dir.writeFile(io, .{ .sub_path = "code", .data = data });
    var expected: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &expected, .{});
    try t.expectEqual(expected, try binary.hash(io, path, data.len));
    data[32768 + 100] ^= 1;
    try tmp.dir.writeFile(io, .{ .sub_path = "code", .data = data });
    try t.expect(!std.mem.eql(u8, &expected, &(try binary.hash(io, path, data.len))));
}

test "persistence identity: tokenizer and template edits reuse only identical canonical token prefixes" {
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = "first tokenizer" });
    try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = "first template" });
    const identity = try disk.scoped(a, io, dir, 1);
    const root = try std.fs.path.join(a, &.{ dir, "states" });
    defer a.free(root);
    var kept = try Imprint.open(a, root, identity, 1 << 20);
    defer kept.deinit();
    const ids = [_]u32{ 5, 7, 11 };
    try kept.add(Imprint.keyOf(&ids), 2, &ids, &.{1}, 4);
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = "other tokenizer" });
    try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = "other template" });
    try t.expectEqual(identity, try disk.scoped(a, io, dir, 1));
    try t.expect(kept.best(&ids, &.{ 1, 2 }, true, 0) != null);
    try t.expect(kept.best(&.{ 5, 9, 11 }, &.{ 1, 2 }, true, 0) == null);
    try t.expect(kept.best(&.{ 5, 7, 13 }, &.{ 1, 2 }, true, 0) == null);
    try t.expect(kept.best(&ids, &.{ 0, 2 }, true, 0) == null);
}

test {
    _ = @import("persistence_policy_test.zig");
}
