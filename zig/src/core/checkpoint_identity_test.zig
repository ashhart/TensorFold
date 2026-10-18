//! Existing checkpoint identity fixtures retained from Nemotron and #900, with shared-cache controls.
const std = @import("std");
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;
const production = @import("checkpoint_identity.zig");
const identity = production.identity;
const Content = production.Content;

test {
    _ = @import("checkpoint_parallel_test.zig");
}

/// A test checkpoint: config.json, a dense and a 1 MiB expert tensor, a manifest of `manifest_sha` (else the shard's).
pub fn write(io: Io, dir: Io.Dir, sub: []const u8, config: []const u8, dense: u8, flip: ?usize, manifest_sha: ?[32]u8) !void {
    const gpa = std.testing.allocator;
    const mib = 1 << 20;
    const head = "{\"m.layers.0.d.weight\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}," ++
        "\"m.layers.1.mixer.experts.0.up_proj.weight\":{\"dtype\":\"U8\",\"shape\":[1048576],\"data_offsets\":[4,1048580]}}";
    const bytes = try gpa.alloc(u8, 8 + head.len + 4 + mib);
    defer gpa.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], head.len, .little);
    @memcpy(bytes[8..][0..head.len], head);
    const body = bytes[8 + head.len ..];
    for (body, 0..) |*b, i| b.* = @truncate(i *% 13);
    body[0] = dense;
    if (flip) |at| body[4 + at] ^= 1;
    try dir.createDirPath(io, sub);
    var d = try dir.openDir(io, sub, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "config.json", .data = config });
    try d.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });
    var sha: [32]u8 = undefined;
    Sha256.hash(bytes, &sha, .{});
    const m = try std.fmt.allocPrint(gpa, "{{\"files\": {{\"model.safetensors\": {{\"sha256\": \"{s}\"}}}}}}", .{&std.fmt.bytesToHex(manifest_sha orelse sha, .lower)});
    defer gpa.free(m);
    try d.writeFile(io, .{ .sub_path = "tensorfold-export-manifest.json", .data = m });
}

fn contentOf(io: Io, path: []const u8, split: bool) ![32]u8 {
    const c = try Content.start(std.testing.allocator, io, path, split);
    return c.finish();
}

test "every byte counts: one bit at 8 KiB into a 1 MiB expert payload changes the digest, or fails a split's manifest" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = "{\"hidden_size\": 8, \"tensorfold_tp\": {\"rank\": 0, \"world\": 2}}";
    try write(io, tmp.dir, "a", cfg, 9, null, null);
    try write(io, tmp.dir, "b", "{\"hidden_size\": 8, \"tensorfold_tp\": {\"rank\": 1, \"world\": 2}}", 9, null, null);
    try write(io, tmp.dir, "c", cfg, 9, 8192, null); // the expert byte past every old sample, manifest updated
    try write(io, tmp.dir, "d", cfg, 10, null, null); // a dense byte
    var good: [32]u8 = undefined;
    Sha256.hash("not this shard", &good, .{});
    try write(io, tmp.dir, "e", cfg, 9, 8192, good); // the expert byte, the manifest still the old file's
    var path: [5][]u8 = undefined;
    for ([_][]const u8{ "a", "b", "c", "d", "e" }, &path) |s, *p| p.* = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, s });
    defer for (path) |p| gpa.free(p);
    // before loading: the split's rank leaves the identity alone, its bytes are not read
    try std.testing.expectEqual(try identity(gpa, io, path[0]), try identity(gpa, io, path[1]));
    try std.testing.expectEqual(try identity(gpa, io, path[0]), try identity(gpa, io, path[2]));
    // whole: every byte, so the unsampled expert byte changes the digest
    try std.testing.expect(!std.mem.eql(u8, &try contentOf(io, path[0], false), &try contentOf(io, path[2], false)));
    // split: the halves share their non-expert bytes; a dense byte differs; a damaged expert byte fails the manifest
    try std.testing.expectEqual(try contentOf(io, path[0], true), try contentOf(io, path[1], true));
    try std.testing.expect(!std.mem.eql(u8, &try contentOf(io, path[0], true), &try contentOf(io, path[3], true)));
    try std.testing.expectError(error.ShardBytesDiffer, contentOf(io, path[4], true));
}

test "a Hugging Face snapshot's shard is checked against its blob name: the right sha256 passes, another refuses" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(io, tmp.dir, "src", "{\"hidden_size\": 8}", 9, null, null);
    const bytes = try tmp.dir.readFileAlloc(io, "src/model.safetensors", gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    var sha: [32]u8 = undefined;
    Sha256.hash(bytes, &sha, .{});
    var other: [32]u8 = undefined;
    Sha256.hash("another revision", &other, .{});
    for ([_][]const u8{ "good", "bad" }, [_][32]u8{ sha, other }) |repo, name| {
        var buf: [128]u8 = undefined;
        const blobs = try tmp.dir.createDirPathOpen(io, try std.fmt.bufPrint(&buf, "{s}/blobs", .{repo}), .{});
        defer blobs.close(io);
        const hexname = std.fmt.bytesToHex(name, .lower);
        try blobs.writeFile(io, .{ .sub_path = &hexname, .data = bytes });
        const snap = try tmp.dir.createDirPathOpen(io, try std.fmt.bufPrint(&buf, "{s}/snapshots/r1", .{repo}), .{});
        defer snap.close(io);
        try snap.writeFile(io, .{ .sub_path = "config.json", .data = "{\"hidden_size\": 8}" });
        snap.symLink(io, try std.fmt.bufPrint(&buf, "../../blobs/{s}", .{&hexname}), "model.safetensors", .{}) catch |e|
            return if (e == error.PermissionDenied and @import("builtin").os.tag == .windows) error.SkipZigTest else e; // symlinks need privilege on Windows
    }
    const good = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/good/snapshots/r1", .{tmp.sub_path});
    defer gpa.free(good);
    const bad = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/bad/snapshots/r1", .{tmp.sub_path});
    defer gpa.free(bad);
    _ = try contentOf(io, good, false);
    try std.testing.expectError(error.ShardBytesDiffer, contentOf(io, bad, false));
}

test "the identity reads headers, not tensor bytes or the folder's name: another tensor name changes it" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = "{\"hidden_size\": 8}";
    try write(io, tmp.dir, "a", cfg, 9, null, null);
    try write(io, tmp.dir, "b", cfg, 10, null, null); // another dense byte under the same header, in another folder
    try write(io, tmp.dir, "n", cfg, 9, null, null);
    const bytes = try tmp.dir.readFileAlloc(io, "n/model.safetensors", gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    bytes[10] = 'n'; // the first tensor's name: "m.layers.0.d.weight" becomes "n.layers.0.d.weight"
    try tmp.dir.writeFile(io, .{ .sub_path = "n/model.safetensors", .data = bytes });
    var path: [3][]u8 = undefined;
    for ([_][]const u8{ "a", "b", "n" }, &path) |s, *p| p.* = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, s });
    defer for (path) |p| gpa.free(p);
    const a = try identity(gpa, io, path[0]);
    try std.testing.expectEqual(a, try identity(gpa, io, path[1]));
    try std.testing.expect(!std.mem.eql(u8, &a, &try identity(gpa, io, path[2])));
}

fn fixturePaths(a: std.mem.Allocator, tmp: *const std.testing.TmpDir, sub: []const u8) ![]u8 {
    const root = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(root);
    return std.fs.path.join(a, &.{ root, sub });
}

test "whole and split consumers share cached file digests and still check manifests" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(io, tmp.dir, "model", "{\"hidden_size\":8}", 9, null, null);
    const path = try fixturePaths(a, &tmp, "model");
    defer a.free(path);
    const cache = try fixturePaths(a, &tmp, "cache");
    defer a.free(cache);
    const cold = try (try Content.startWithCache(a, io, path, false, cache)).finishWithStats();
    try std.testing.expect(cold.bytes_read > 1 << 20);
    try std.testing.expectEqual(@as(usize, 0), cold.cache_hits);
    const warm = try (try Content.startWithCache(a, io, path, false, cache)).finishWithStats();
    try std.testing.expectEqual(cold.digest, warm.digest);
    try std.testing.expectEqual(@as(u64, 0), warm.bytes_read);
    try std.testing.expectEqual(@as(usize, 1), warm.cache_hits);
    const split = try (try Content.startWithCache(a, io, path, true, cache)).finishWithStats();
    try std.testing.expectEqual(@as(u64, 0), split.bytes_read);
    try std.testing.expectEqual(@as(usize, 1), split.cache_hits);
    try std.testing.expectEqual(try contentOf(io, path, true), split.digest);
    try tmp.dir.writeFile(io, .{ .sub_path = "model/tensorfold-export-manifest.json", .data = "{\"files\":{\"model.safetensors\":{\"sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\"}}}" });
    try std.testing.expectError(error.ShardBytesDiffer, (try Content.startWithCache(a, io, path, true, cache)).finish());
}

test "full checkpoint identity includes payloads and detects changed files at ready" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(io, tmp.dir, "model", "{\"hidden_size\":8}", 9, null, null);
    const path = try fixturePaths(a, &tmp, "model");
    defer a.free(path);
    const cache = try fixturePaths(a, &tmp, "cache");
    defer a.free(cache);
    const first = try production.fullWithCache(a, io, path, cache);
    const c = try Content.startWithCache(a, io, path, false, cache);
    c.wait();
    try write(io, tmp.dir, "model", "{\"hidden_size\":8}", 10, null, null);
    try std.testing.expectError(error.DigestFileChanged, c.finish());
    const next = try production.fullWithCache(a, io, path, cache);
    try std.testing.expect(!std.mem.eql(u8, &first, &next));
}

test "a corrupt shared digest cache refuses the real content consumer" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(io, tmp.dir, "model", "{}", 9, null, null);
    const path = try fixturePaths(a, &tmp, "model");
    defer a.free(path);
    const cache = try fixturePaths(a, &tmp, "cache");
    defer a.free(cache);
    _ = try (try Content.startWithCache(a, io, path, false, cache)).finish();
    try tmp.dir.writeFile(io, .{ .sub_path = "cache/index.bin", .data = "broken" });
    try std.testing.expectError(error.CorruptDigestCache, (try Content.startWithCache(a, io, path, false, cache)).finish());
}
