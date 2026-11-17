//! The `models` command: the checkpoints on disk that a registered Zig family can serve.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");

/// One listed checkpoint: the facts a line prints, and the fields of a `--json` entry.
const Listed = struct { repo: []const u8, family: []const u8, model_type: []const u8 };

/// The cache checkpoints a Zig family serves, as lines or as one JSON document. Returns the process exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, json: bool) !u8 {
    const root = try hub.cacheDir(a, env, override);
    const found = (try find(a, io, root)) orelse {
        if (json) return printJson(out, &.{});
        try out.print("no checkpoints on disk: the Hugging Face cache at {s} does not exist yet; run: tensorfold pull OWNER/NAME\n", .{root});
        return 0;
    };
    var listed: std.ArrayList(Listed) = .empty;
    for (found) |f| try listed.append(a, f.listed);
    if (json) return printJson(out, listed.items);
    for (listed.items) |m| try out.print("{s}: {s} ({s})\n", .{ m.repo, m.family, m.model_type });
    if (listed.items.len == 0) try out.print("no checkpoints on disk that a registered Zig family can serve; run: tensorfold pull OWNER/NAME\n", .{});
    return 0;
}

const Found = struct { listed: Listed, snapshot: []const u8 };

/// The cache's checkpoints a Zig family serves, with their snapshot folders; null when the cache does not exist.
fn find(a: Allocator, io: std.Io, root: []const u8) !?[]const Found {
    var dir = std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    var found: std.ArrayList(Found) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory) continue;
        const repo = (repoFromDirName(a, entry.name) catch continue) orelse continue;
        const snapshot = (try hub.cachedSnapshot(a, io, root, repo)) orelse continue;
        const model_type = hub.modelType(a, io, snapshot);
        const family = hub.family(model_type) orelse continue;
        try found.append(a, .{ .listed = .{ .repo = repo, .family = family.title, .model_type = family.model_type }, .snapshot = snapshot });
    }
    return found.items;
}

/// A checkpoint `models` lists, with its snapshot's size, for `tensorfold device`'s fit against this machine.
pub const OnDisk = struct { repo: []const u8, family: []const u8, model_type: []const u8, bytes: u64 };

pub fn onDisk(a: Allocator, io: std.Io, env: ?*const std.process.Environ.Map, override: ?[]const u8) ![]const OnDisk {
    const found = (try find(a, io, try hub.cacheDir(a, env, override))) orelse return &.{};
    const out = try a.alloc(OnDisk, found.len);
    for (out, found) |*o, f| o.* = .{ .repo = f.listed.repo, .family = f.listed.family, .model_type = f.listed.model_type, .bytes = try hub.sizeOf(a, io, f.snapshot) };
    return out;
}

/// `{"schema": 1, "models": [...]}`: the same checkpoints the lines print, one object each, for programs.
fn printJson(out: *std.Io.Writer, listed: []const Listed) !u8 {
    try std.json.Stringify.value(.{ .schema = 1, .models = listed }, .{}, out);
    try out.writeByte('\n');
    return 0;
}

/// models--Org--Name back to Org/Name: the first double dash separates them; other entries do not name a repo.
fn repoFromDirName(a: Allocator, name: []const u8) !?[]const u8 {
    const prefix = "models--";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const rest = name[prefix.len..];
    const sep = std.mem.indexOf(u8, rest, "--") orelse return null;
    if (sep == 0 or sep + 2 >= rest.len) return null;
    return try std.fmt.allocPrint(a, "{s}/{s}", .{ rest[0..sep], rest[sep + 2 ..] });
}

/// A cache with one checkpoint a Zig family serves (Org/Flash) and one none does (Org/Draft); the caller deletes it.
fn syntheticCache(a: Allocator, io: std.Io) ![]const u8 {
    const root = try std.fmt.allocPrint(a, ".tf-models-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    const w = std.Io.Dir.cwd();
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1" }));
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs/main" }), .data = "rev1" });
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1/config.json" }), .data = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}}" });
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Draft/snapshots/rev2" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Draft/snapshots/rev2/config.json" }), .data = "{\"model_type\": \"gemma3n\"}" });
    return root;
}

test "models lists serve-able checkpoints and skips families Zig cannot serve" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try syntheticCache(a, io);
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};

    var out: std.Io.Writer.Allocating = .init(a);
    const code = try run(a, io, &out.writer, null, root, false);
    try std.testing.expectEqual(@as(u8, 0), code);
    const listed = out.written();
    try std.testing.expect(std.mem.indexOf(u8, listed, "Org/Flash: Qwen3.8 Flash Next (qwen4_exp)") != null);
    try std.testing.expect(std.mem.indexOf(u8, listed, "Org/Draft") == null);
}

test "models --json prints the listed checkpoints as one document, and an absent cache as none" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try syntheticCache(a, io);
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};

    var out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &out.writer, null, root, true));
    var parsed = try std.json.parseFromSlice(std.json.Value, a, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 1), parsed.value.object.get("schema").?.integer);
    const models = parsed.value.object.get("models").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), models.len);
    try std.testing.expectEqualStrings("Org/Flash", models[0].object.get("repo").?.string);
    try std.testing.expectEqualStrings("Qwen3.8 Flash Next", models[0].object.get("family").?.string);
    try std.testing.expectEqualStrings("qwen4_exp", models[0].object.get("model_type").?.string);

    var none: std.Io.Writer.Allocating = .init(a);
    const missing = try std.fs.path.join(a, &.{ root, "missing" });
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &none.writer, null, missing, true));
    try std.testing.expectEqualStrings("{\"schema\":1,\"models\":[]}\n", none.written());
}

test "onDisk sizes the checkpoints models lists, and an absent cache has none" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try syntheticCache(a, io);
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const found = try onDisk(a, io, null, root);
    try std.testing.expectEqual(@as(usize, 1), found.len);
    try std.testing.expectEqualStrings("Org/Flash", found[0].repo);
    try std.testing.expectEqualStrings("qwen4_exp", found[0].model_type);
    try std.testing.expect(found[0].bytes > 0); // its config.json
    try std.testing.expectEqual(@as(usize, 0), (try onDisk(a, io, null, try std.fs.path.join(a, &.{ root, "missing" }))).len);
}

fn has(text: []const u8, needle: []const u8) bool {
    if (std.mem.indexOf(u8, text, needle) != null) return true;
    std.debug.print("missing \"{s}\" in:\n{s}\n", .{ needle, text });
    return false;
}
