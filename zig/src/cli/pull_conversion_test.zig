//! Conversion pull admission and symbolic ref publication through the production helpers, without HTTP.
const std = @import("std");
const pull = @import("pull.zig");
const fixture = @import("tfq3_test.zig");
const tfq3 = @import("tfq3.zig");

fn admit(a: std.mem.Allocator, config: []const u8, admission: pull.Admission) !bool {
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    return pull.admitConfig(a, &out.writer, &err.writer, .{}, "Org/Model", config, admission);
}

test "conversion pull: generic sources bypass only inference-family admission" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const generic = "{\"model_type\":\"generic_architecture\"}";
    try std.testing.expect(try admit(a, generic, .conversion));
    try std.testing.expect(!try admit(a, generic, .inference));
    for ([_][]const u8{ "not json", "[]", "null" }) |config| {
        try std.testing.expect(!try admit(a, config, .conversion));
        try std.testing.expect(!try admit(a, config, .inference));
    }
    try std.testing.expect(try admit(a, "{\"model_type\":\"qwen3_5\"}", .inference));
}

test "conversion pull: a symbolic ref resolves after publication and is revoked before repair" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const repo = try std.fs.path.join(a, &.{ root, "models--Org--Model" });
    const refs = try std.fs.path.join(a, &.{ repo, "refs" });
    const sha = "abcdef0123456789abcdef0123456789abcdef01";
    const snap = try std.fs.path.join(a, &.{ repo, "snapshots", sha });
    try std.Io.Dir.cwd().createDirPath(io, refs);
    try std.Io.Dir.cwd().createDirPath(io, snap);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ snap, "config.json" }), .data = "{\"model_type\":\"generic_architecture\"}" });
    try pull.publishRefs(a, io, refs, "release", sha);
    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    defer env.deinit();
    var cap: fixture.Capture = .{};
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    const ctx = fixture.makeCtx(a, &env, root, &cap, &out.writer, &err.writer);
    try std.testing.expectEqual(@as(u8, 0), try tfq3.runWith(&ctx, &.{ "Org/Model@release", "--no-pull" }));
    try std.testing.expectEqualStrings(snap, cap.argv[2]);
    for ([_][]const u8{ "main", "release" }) |name| {
        const path = try std.fs.path.join(a, &.{ refs, name });
        try std.testing.expectEqualStrings(sha, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(128)));
    }
    try pull.revokeRefs(a, io, refs, "release", sha);
    for ([_][]const u8{ "main", "release" }) |name| {
        const f = std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ refs, name }), .{});
        try std.testing.expectError(error.FileNotFound, f);
    }
    try pull.publishRefs(a, io, refs, "main", sha);
    try std.testing.expectError(error.BadRevision, pull.publishRefs(a, io, refs, "../bad", sha));
    try std.testing.expectError(error.BadRevision, pull.publishRefs(a, io, refs, "release", "../bad"));
}
