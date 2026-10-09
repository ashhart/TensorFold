//! The production cache lifecycle with tiny prepared pack files and injected interruption.
const std = @import("std");
const Io = std.Io;
const cache = @import("pack_cache.zig");
const pio = @import("pack_io.zig");
const a = std.testing.allocator;
const io = std.testing.io;

const Builder = struct {
    identity: []const u8,
    calls: usize = 0,
    fail: bool = false,
    forbidden_final: ?[]const u8 = null,

    fn unpublished(self: *Builder, actual_io: Io) !void {
        const path = self.forbidden_final orelse return;
        if (Io.Dir.cwd().openDir(actual_io, path, .{})) |dir| {
            dir.close(actual_io);
            return error.HalfPackPublished;
        } else |err| if (err != error.FileNotFound) return err;
    }

    pub fn build(self: *Builder, gpa: std.mem.Allocator, actual_io: Io, _: []const u8, dir: []const u8, _: []const u8) !void {
        self.calls += 1;
        try self.unpublished(actual_io);
        for ([_][]const u8{ "pack.safetensors", "pack_mlx.safetensors", "pack_mtp_mlx.safetensors" }) |name| {
            var out = pio.Out.init(gpa);
            defer out.deinit();
            const bytes = try gpa.dupe(u8, "test");
            try out.put("weights", "U8", &.{4}, bytes);
            const path = try std.fs.path.join(gpa, &.{ dir, name });
            defer gpa.free(path);
            try out.write(gpa, actual_io, path, self.identity);
            try self.unpublished(actual_io);
            if (self.fail) return error.StoppedBuild;
        }
    }
};

const Paths = struct {
    model: []u8,
    root: []u8,
    fn init(tmp: std.testing.TmpDir) !Paths {
        try tmp.dir.createDir(io, "model", .default_dir);
        const base = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(base);
        const model = try std.fs.path.join(a, &.{ base, "model" });
        errdefer a.free(model);
        return .{ .model = model, .root = try std.fs.path.join(a, &.{ base, "cache" }) };
    }
    fn deinit(self: Paths) void {
        a.free(self.model);
        a.free(self.root);
    }
};

test "read-only checkpoint builds and reuses packs without writing in the model directory" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try Paths.init(tmp);
    defer paths.deinit();
    var model = try tmp.dir.openDir(io, "model", .{ .iterate = true });
    defer model.close(io);
    try model.setPermissions(io, .fromMode(0o555));
    defer model.setPermissions(io, .fromMode(0o755)) catch {};
    var builder: Builder = .{ .identity = "checkpoint one" };
    const dir = try cache.ensure(a, io, paths.model, paths.root, builder.identity, &builder);
    defer a.free(dir);
    try std.testing.expect(std.mem.startsWith(u8, dir, paths.root));
    const reused = try cache.ensure(a, io, paths.model, paths.root, builder.identity, &builder);
    defer a.free(reused);
    try std.testing.expectEqualStrings(dir, reused);
    try std.testing.expectEqual(@as(usize, 1), builder.calls);
    try std.testing.expectError(error.FileNotFound, model.openDir(io, "zig-pack", .{}));
}

test "an interrupted build publishes no half pack and the next open rebuilds" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try Paths.init(tmp);
    defer paths.deinit();
    var builder: Builder = .{ .identity = "checkpoint one", .fail = true };
    const final = try cache.directory(a, paths.root, builder.identity);
    defer a.free(final);
    builder.forbidden_final = final;
    try std.testing.expectError(error.StoppedBuild, cache.ensure(a, io, paths.model, paths.root, builder.identity, &builder));
    try std.testing.expectError(error.FileNotFound, tmp.dir.openDir(io, "model/zig-pack", .{}));
    var root = try tmp.dir.openDir(io, "cache", .{ .iterate = true });
    defer root.close(io);
    var it = root.iterate();
    while (try it.next(io)) |entry| try std.testing.expect(entry.kind != .directory);
    builder.fail = false;
    const dir = try cache.ensure(a, io, paths.model, paths.root, builder.identity, &builder);
    defer a.free(dir);
    try std.testing.expect(try pio.packsReady(a, io, dir, builder.identity));
}

fn writeSource(model: []const u8, bytes: []const u8) ![]u8 {
    const index_path = try std.fs.path.join(a, &.{ model, "model.safetensors.index.json" });
    defer a.free(index_path);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = index_path, .data = "{\"weight_map\":{\"weights\":\"model.safetensors\"}}" });
    const shard_path = try std.fs.path.join(a, &.{ model, "model.safetensors" });
    defer a.free(shard_path);
    var out = pio.Out.init(a);
    defer out.deinit();
    try out.put("weights", "U8", &.{bytes.len}, try a.dupe(u8, bytes));
    try out.write(a, io, shard_path, "source");
    return pio.sourceIdentity(a, io, model);
}

test "a checkpoint header or size change gets a new immutable pack directory" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try Paths.init(tmp);
    defer paths.deinit();
    const old_identity = try writeSource(paths.model, "four");
    defer a.free(old_identity);
    var builder: Builder = .{ .identity = old_identity };
    const first = try cache.ensure(a, io, paths.model, paths.root, builder.identity, &builder);
    defer a.free(first);
    const new_identity = try writeSource(paths.model, "changed size");
    defer a.free(new_identity);
    builder.identity = new_identity;
    const second = try cache.ensure(a, io, paths.model, paths.root, builder.identity, &builder);
    defer a.free(second);
    try std.testing.expect(!std.mem.eql(u8, first, second));
    try std.testing.expect(try cache.ready(a, io, first, old_identity));
    try std.testing.expect(try cache.ready(a, io, second, new_identity));
}

test "matching complete legacy packs are read without creating a writable cache" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try Paths.init(tmp);
    defer paths.deinit();
    try tmp.dir.createDir(io, "model/zig-pack", .default_dir);
    const legacy = try std.fs.path.join(a, &.{ paths.model, "zig-pack" });
    defer a.free(legacy);
    var builder: Builder = .{ .identity = "checkpoint one" };
    try builder.build(a, io, paths.model, legacy, "");
    var model = try tmp.dir.openDir(io, "model", .{ .iterate = true });
    defer model.close(io);
    try model.setPermissions(io, .fromMode(0o555));
    defer model.setPermissions(io, .fromMode(0o755)) catch {};
    builder.fail = true;
    const dir = try cache.ensure(a, io, paths.model, paths.root, builder.identity, &builder);
    defer a.free(dir);
    try std.testing.expectEqualStrings(legacy, dir);
    try std.testing.expectEqual(@as(usize, 1), builder.calls);
    try std.testing.expectError(error.FileNotFound, tmp.dir.openDir(io, "cache", .{}));
}

test "incomplete and mismatched legacy packs remain untouched and rebuild outside the model" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try Paths.init(tmp);
    defer paths.deinit();
    try tmp.dir.createDir(io, "model/zig-pack", .default_dir);
    const legacy = try std.fs.path.join(a, &.{ paths.model, "zig-pack" });
    defer a.free(legacy);
    var builder: Builder = .{ .identity = "old checkpoint" };
    try builder.build(a, io, paths.model, legacy, "");
    try tmp.dir.deleteFile(io, "model/zig-pack/pack_mtp_mlx.safetensors");
    builder.identity = "new checkpoint";
    const dir = try cache.ensure(a, io, paths.model, paths.root, builder.identity, &builder);
    defer a.free(dir);
    try std.testing.expect(std.mem.startsWith(u8, dir, paths.root));
    try std.testing.expect(try pio.packsReady(a, io, legacy, "old checkpoint"));
    try std.testing.expectError(error.FileNotFound, tmp.dir.openFile(io, "model/zig-pack/pack_mtp_mlx.safetensors", .{}));
}

test "cache admission rejects malformed or unmarked metadata and a truncated tensor body" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const headers = [_][]const u8{
        "[]",
        "{\"__metadata__\":[],\"w\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
        "{\"__metadata__\":{\"tf_source\":\"checkpoint\"},\"w\":{\"dtype\":\"U8\",\"shape\":[5],\"data_offsets\":[0,5]}}",
    };
    const base = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(base);
    const path = try std.fs.path.join(a, &.{ base, "pack" });
    defer a.free(path);
    for (headers) |head| {
        const image = try a.alloc(u8, 8 + head.len + 4);
        defer a.free(image);
        std.mem.writeInt(u64, image[0..8], head.len, .little);
        @memcpy(image[8..][0..head.len], head);
        @memset(image[8 + head.len ..], 0);
        try tmp.dir.writeFile(io, .{ .sub_path = "pack", .data = image });
        try std.testing.expect(!try pio.packReady(a, io, path, "checkpoint"));
    }
}

test "an incomplete cache build is refused and only owned stale temporary directories are removed" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try Paths.init(tmp);
    defer paths.deinit();
    const dir = try cache.directory(a, paths.root, "checkpoint one");
    defer a.free(dir);
    const stale = try std.fmt.allocPrint(a, "{s}.part-00000000000000000000000000000000", .{dir});
    defer a.free(stale);
    const foreign = try std.fmt.allocPrint(a, "{s}.part-unowned", .{dir});
    defer a.free(foreign);
    try Io.Dir.cwd().createDirPath(io, stale);
    try Io.Dir.cwd().createDirPath(io, foreign);
    const leftover = try std.fs.path.join(a, &.{ stale, "draft_vocab.txt" });
    defer a.free(leftover);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = leftover, .data = "interrupted" });
    const Incomplete = struct {
        pub fn build(_: @This(), _: std.mem.Allocator, _: Io, _: []const u8, _: []const u8, _: []const u8) !void {}
    };
    try std.testing.expectError(error.IncompletePackBuild, cache.ensure(a, io, paths.model, paths.root, "checkpoint one", Incomplete{}));
    try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().openDir(io, stale, .{}));
    try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().openDir(io, dir, .{}));
    var kept = try Io.Dir.cwd().openDir(io, foreign, .{});
    kept.close(io);
}

test "a truncated current cache is rebuilt under its identity without reusing a partial file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try Paths.init(tmp);
    defer paths.deinit();
    var builder: Builder = .{ .identity = "checkpoint one" };
    const first = try cache.ensure(a, io, paths.model, paths.root, builder.identity, &builder);
    defer a.free(first);
    const path = try std.fs.path.join(a, &.{ first, "pack_mtp_mlx.safetensors" });
    defer a.free(path);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "partial" });
    const second = try cache.ensure(a, io, paths.model, paths.root, builder.identity, &builder);
    defer a.free(second);
    try std.testing.expectEqualStrings(first, second);
    try std.testing.expectEqual(@as(usize, 2), builder.calls);
    try std.testing.expect(try cache.ready(a, io, second, builder.identity));
}

test "concurrent cache builders publish one complete directory" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try Paths.init(tmp);
    defer paths.deinit();
    var builder: Builder = .{ .identity = "checkpoint one" };
    var go: std.atomic.Value(u32) = .init(0);
    const Worker = struct {
        paths: *const Paths,
        builder: *Builder,
        go: *std.atomic.Value(u32),
        dir: ?[]u8 = null,
        err: ?anyerror = null,
        fn run(self: *@This()) void {
            _ = self.go.fetchAdd(1, .acq_rel);
            while (self.go.load(.acquire) != 2) std.Thread.yield() catch {};
            self.dir = cache.ensure(a, io, self.paths.model, self.paths.root, self.builder.identity, self.builder) catch |err| {
                self.err = err;
                return;
            };
        }
    };
    var workers = [_]Worker{ .{ .paths = &paths, .builder = &builder, .go = &go }, .{ .paths = &paths, .builder = &builder, .go = &go } };
    const first = try std.Thread.spawn(.{}, Worker.run, .{&workers[0]});
    const second = std.Thread.spawn(.{}, Worker.run, .{&workers[1]}) catch |err| {
        go.store(2, .release);
        first.join();
        return err;
    };
    first.join();
    second.join();
    defer for (workers) |worker| if (worker.dir) |dir| a.free(dir);
    for (workers) |worker| if (worker.err) |err| return err;
    try std.testing.expectEqualStrings(workers[0].dir.?, workers[1].dir.?);
    try std.testing.expectEqual(@as(usize, 1), builder.calls);
    try std.testing.expect(try cache.ready(a, io, workers[0].dir.?, builder.identity));
}

test "cache readiness reads bounded metadata rather than copying the weight body" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const body_len = 64 << 20;
    const header = "{\"__metadata__\":{\"tf_source\":\"checkpoint\"},\"w\":{\"dtype\":\"U8\",\"shape\":[67108864],\"data_offsets\":[0,67108864]}}";
    const file = try tmp.dir.createFile(io, "sparse-pack", .{});
    defer file.close(io);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, header.len, .little);
    try file.writeStreamingAll(io, &len);
    try file.writeStreamingAll(io, header);
    try file.setLength(io, 8 + header.len + body_len);
    const base = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(base);
    const path = try std.fs.path.join(a, &.{ base, "sparse-pack" });
    defer a.free(path);
    var storage: [16 << 10]u8 = undefined;
    var bounded: std.heap.FixedBufferAllocator = .init(&storage);
    try std.testing.expect(try pio.packReady(bounded.allocator(), io, path, "checkpoint"));
}
