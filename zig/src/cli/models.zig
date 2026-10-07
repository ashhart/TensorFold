//! The `models` command: the checkpoints on disk that a registered Zig family can serve.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");

/// Lists each cache checkpoint whose model_type a Zig family serves. Returns the process exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8) !u8 {
    const root = try hub.cacheDir(a, env, override);
    var dir = std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true }) catch {
        try out.print("no checkpoints on disk: the Hugging Face cache at {s} does not exist yet; run: tensorfold pull OWNER/NAME\n", .{root});
        return 0;
    };
    defer dir.close(io);
    var found: usize = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory) continue;
        const repo = (repoFromDirName(a, entry.name) catch continue) orelse continue;
        const snapshot = (try hub.cachedSnapshot(a, io, root, repo)) orelse continue;
        const model_type = hub.modelType(a, io, snapshot);
        const family = hub.family(model_type) orelse continue;
        try out.print("{s}: {s} ({s})\n", .{ repo, family.title, family.model_type });
        found += 1;
    }
    if (found == 0) try out.print("no checkpoints on disk that a registered Zig family can serve; run: tensorfold pull OWNER/NAME\n", .{});
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

test "models lists serve-able checkpoints and skips families Zig cannot serve" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try std.fmt.allocPrint(a, ".tf-models-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const w = std.Io.Dir.cwd();
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1" }));
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs/main" }), .data = "rev1" });
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1/config.json" }), .data = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}}" });
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Draft/snapshots/rev2" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Draft/snapshots/rev2/config.json" }), .data = "{\"model_type\": \"gemma4\"}" });

    var out: std.Io.Writer.Allocating = .init(a);
    const code = try run(a, io, &out.writer, null, root);
    try std.testing.expectEqual(@as(u8, 0), code);
    const listed = out.written();
    try std.testing.expect(std.mem.indexOf(u8, listed, "Org/Flash: Qwen3.8 Flash Next (qwen4_exp)") != null);
    try std.testing.expect(std.mem.indexOf(u8, listed, "Org/Draft") == null);
}

fn has(text: []const u8, needle: []const u8) bool {
    if (std.mem.indexOf(u8, text, needle) != null) return true;
    std.debug.print("missing \"{s}\" in:\n{s}\n", .{ needle, text });
    return false;
}
