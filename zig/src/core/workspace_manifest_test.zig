const std = @import("std");
const disk = @import("workspace_manifest_io.zig");
const wm = @import("workspace_manifest.zig");

const a = std.testing.allocator;
const io = std.testing.io;
const id: wm.Identity = .{ .model = "model-sha256", .template = "chat-v3", .cache = "kv-geometry-v7" };

const Fixture = struct {
    tmp: std.testing.TmpDir,
    store: wm.Store,

    fn init() !Fixture {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.createDirPath(io, "repo");
        try tmp.dir.createDirPath(io, "state");
        try tmp.dir.setFilePermissions(io, "state", .fromMode(0o700), .{});
        const root = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(root);
        const repo = try std.fs.path.join(a, &.{ root, "repo" });
        defer a.free(repo);
        const state = try std.fs.path.join(a, &.{ root, "state" });
        defer a.free(state);
        return .{ .tmp = tmp, .store = try wm.Store.init(a, io, repo, state) };
    }

    fn deinit(f: *Fixture) void {
        f.store.deinit();
        f.tmp.cleanup();
    }

    fn write(f: *Fixture, relative: []const u8, bytes: []const u8) !void {
        const path = try std.fs.path.join(a, &.{ "repo", relative });
        defer a.free(path);
        try f.tmp.dir.writeFile(io, .{ .sub_path = path, .data = bytes });
    }

    fn remove(f: *Fixture, relative: []const u8) !void {
        const path = try std.fs.path.join(a, &.{ "repo", relative });
        defer a.free(path);
        try f.tmp.dir.deleteFile(io, path);
    }
};

fn expectStateEmpty(f: *Fixture) !void {
    var dir = try f.tmp.dir.openDir(io, "state", .{ .iterate = true });
    defer dir.close(io);
    var entries = dir.iterate();
    try std.testing.expect((try entries.next(io)) == null);
}

fn cancelled(_: *anyopaque) bool {
    return true;
}

fn stop() wm.Stop {
    const Context = struct {
        var value: u8 = 0;
    };
    return .{ .ptr = &Context.value, .check = cancelled };
}

fn source(path: []const u8, tokens: []const u32, bytes: []const u8) wm.Source {
    var sha256: wm.Digest = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &sha256, .{});
    return .{ .path = path, .tokens = tokens, .size = @intCast(bytes.len), .sha256 = sha256 };
}

fn cancelledAfterFirst(ptr: *anyopaque) bool {
    const checks: *usize = @ptrCast(@alignCast(ptr));
    checks.* += 1;
    return checks.* > 1;
}

test "workspace manifest cold and warm reuse exact ordered prefixes with stable provenance" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("b.txt", "bravo");
    try f.write("a.txt", "alpha");
    var cold = try f.store.cold(id, &.{1}, &.{
        source("b.txt", &.{ 4, 5 }, "bravo"),
        source("a.txt", &.{ 2, 3 }, "alpha"),
    }, null);
    defer cold.deinit();
    try std.testing.expectEqual(@as(u32, 0), cold.reused_tokens);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5 }, cold.manifest.prompt);
    try std.testing.expectEqualStrings("a.txt", cold.manifest.files[0].path);
    try std.testing.expectEqualStrings("b.txt", cold.manifest.files[1].path);
    try std.testing.expectEqual(@as(u32, 3), cold.manifest.files[0].prompt_end);
    try std.testing.expectEqual(@as(u32, 5), cold.manifest.files[1].prompt_end);
    var alpha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("alpha", &alpha, .{});
    try std.testing.expectEqualSlices(u8, &alpha, &cold.manifest.files[0].sha256);
    const provenance = cold.manifest.provenance;
    const workspace_id = cold.manifest.workspaceId();
    const namespace_id = cold.manifest.namespaceId();
    var warm = try f.store.warm(id, null);
    defer warm.deinit();
    try std.testing.expectEqual(@as(u32, 5), warm.reused_tokens);
    try std.testing.expectEqual(@as(u32, 0), warm.changed_files);
    try std.testing.expectEqualSlices(u8, &provenance, &warm.manifest.provenance);
    try std.testing.expectEqualSlices(u8, &workspace_id, &warm.manifest.workspaceId());
    try std.testing.expectEqualSlices(u8, &namespace_id, &warm.manifest.namespaceId());
}

test "workspace manifest one edit reuses only the exact unchanged prefix" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    try f.write("b.txt", "bravo");
    try f.write("c.txt", "charlie");
    var cold = try f.store.cold(id, &.{10}, &.{
        source("a.txt", &.{ 1, 2 }, "alpha"),
        source("b.txt", &.{ 3, 4 }, "bravo"),
        source("c.txt", &.{ 5, 6 }, "charlie"),
    }, null);
    defer cold.deinit();
    const old_provenance = cold.manifest.provenance;
    try f.write("b.txt", "changed");
    var edited = try f.store.edit(id, &.{10}, &.{
        source("a.txt", &.{ 1, 2 }, "alpha"),
        source("b.txt", &.{9}, "changed"),
        source("c.txt", &.{ 5, 6 }, "charlie"),
    }, null);
    defer edited.deinit();
    try std.testing.expectEqual(@as(u32, 3), edited.reused_tokens);
    try std.testing.expectEqual(@as(u32, 1), edited.changed_files);
    try std.testing.expectEqualSlices(u32, &.{ 10, 1, 2, 9, 5, 6 }, edited.manifest.prompt);
    try std.testing.expect(!std.mem.eql(u8, &old_provenance, &edited.manifest.provenance));
    var warm = try f.store.warm(id, null);
    defer warm.deinit();
    try std.testing.expectEqual(@as(u32, 6), warm.reused_tokens);
}

test "workspace manifest stale and missing files fail closed" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    try f.write("b.txt", "bravo");
    var cold = try f.store.cold(id, &.{1}, &.{
        source("a.txt", &.{2}, "alpha"),
        source("b.txt", &.{3}, "bravo"),
    }, null);
    defer cold.deinit();
    try f.write("a.txt", "Alpha");
    try std.testing.expectError(error.WorkspaceFileChanged, f.store.warm(id, null));
    try f.write("a.txt", "alpha");
    var restored = try f.store.warm(id, null);
    restored.deinit();
    try f.remove("b.txt");
    try std.testing.expectError(error.WorkspaceFileMissing, f.store.warm(id, null));
}

test "workspace manifest cold rejects a source binding from different same-length bytes" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    const sources = [_]wm.Source{source("a.txt", &.{2}, "Alpha")};
    try std.testing.expectError(error.WorkspaceFileChanged, f.store.cold(id, &.{1}, &sources, null));
    try expectStateEmpty(&f);
}

test "workspace manifest namespaces cannot cross model or workspace roots" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    var cold = try f.store.cold(id, &.{1}, &.{source("a.txt", &.{2}, "alpha")}, null);
    defer cold.deinit();
    const other_model: wm.Identity = .{ .model = "other-model", .template = id.template, .cache = id.cache };
    try std.testing.expectError(error.NamespaceMismatch, f.store.warm(other_model, null));
    try f.tmp.dir.createDirPath(io, "other-repo");
    const root = try f.tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const other_repo = try std.fs.path.join(a, &.{ root, "other-repo" });
    defer a.free(other_repo);
    const state = try std.fs.path.join(a, &.{ root, "state" });
    defer a.free(state);
    var other = try wm.Store.init(a, io, other_repo, state);
    defer other.deinit();
    try std.testing.expectError(error.NamespaceMismatch, other.warm(id, null));
}

test "workspace manifest provenance depends on content and identity not state location" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    const sources = [_]wm.Source{source("a.txt", &.{2}, "alpha")};
    var first = try f.store.cold(id, &.{1}, &sources, null);
    defer first.deinit();
    try f.tmp.dir.createDirPath(io, "second-state");
    const root = try f.tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const second_state = try std.fs.path.join(a, &.{ root, "second-state" });
    try f.tmp.dir.setFilePermissions(io, "second-state", .fromMode(0o700), .{});
    defer a.free(second_state);
    var second_store = try wm.Store.init(a, io, f.store.workspace, second_state);
    defer second_store.deinit();
    var second = try second_store.cold(id, &.{1}, &sources, null);
    defer second.deinit();
    try std.testing.expectEqualSlices(u8, &first.manifest.provenance, &second.manifest.provenance);
    const other_cache: wm.Identity = .{ .model = id.model, .template = id.template, .cache = "other-cache" };
    try std.testing.expectError(error.NamespaceMismatch, second_store.warm(other_cache, null));
}

test "workspace manifest rejects relative traversal and escaping symlinks" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var f = try Fixture.init();
    defer f.deinit();
    try f.tmp.dir.writeFile(io, .{ .sub_path = "outside.txt", .data = "outside" });
    try f.tmp.dir.symLink(io, "../outside.txt", "repo/link.txt", .{});
    try f.tmp.dir.createDirPath(io, "outside-dir");
    try f.tmp.dir.writeFile(io, .{ .sub_path = "outside-dir/file.txt", .data = "outside" });
    try f.tmp.dir.symLink(io, "../outside-dir", "repo/nested", .{ .is_directory = true });
    try std.testing.expectError(error.WorkspacePathRefused, f.store.cold(id, &.{1}, &.{source("nested/file.txt", &.{2}, "outside")}, null));
    try std.testing.expectError(error.WorkspacePathRefused, f.store.cold(id, &.{1}, &.{source("link.txt", &.{2}, "outside")}, null));
    const invalid = [_][]const u8{ "../outside.txt", "/absolute", "sub/../outside.txt", "C:outside.txt", "", "sub//file" };
    for (invalid) |path| {
        try std.testing.expectError(error.InvalidRelativePath, f.store.cold(id, &.{1}, &.{source(path, &.{2}, "unused")}, null));
    }
    try expectStateEmpty(&f);
}

test "workspace manifest warm refuses a file replaced by an escaping symlink" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    try f.tmp.dir.writeFile(io, .{ .sub_path = "outside.txt", .data = "outside" });
    var cold = try f.store.cold(id, &.{1}, &.{source("a.txt", &.{2}, "alpha")}, null);
    defer cold.deinit();
    try f.remove("a.txt");
    try f.tmp.dir.symLink(io, "../outside.txt", "repo/a.txt", .{});
    try std.testing.expectError(error.WorkspacePathRefused, f.store.warm(id, null));
}

test "workspace manifest cancellation leaves neither publication nor temporary files" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    const sources = [_]wm.Source{source("a.txt", &.{2}, "alpha")};
    try std.testing.expectError(error.Cancelled, f.store.cold(id, &.{1}, &sources, stop()));
    try expectStateEmpty(&f);
    var captured = try disk.capture(a, io, f.store.workspace, f.store.workspace_dir, id, &.{1}, &sources, null);
    defer captured.deinit();
    var checks: usize = 0;
    const trip: wm.Stop = .{ .ptr = &checks, .check = cancelledAfterFirst };
    try std.testing.expectError(error.Cancelled, disk.publish(a, io, f.store.state_dir, &captured, false, trip));
    try std.testing.expectEqual(@as(usize, 2), checks);
    try expectStateEmpty(&f);
}

test "workspace manifest corruption is not admitted as warm state" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    var cold = try f.store.cold(id, &.{1}, &.{source("a.txt", &.{2}, "alpha")}, null);
    defer cold.deinit();
    var dir = try f.tmp.dir.openDir(io, "state", .{});
    defer dir.close(io);
    const bytes = try dir.readFileAlloc(io, wm.manifest_name, a, .limited(1 << 20));
    defer a.free(bytes);
    bytes[12] ^= 1;
    try dir.writeFile(io, .{ .sub_path = wm.manifest_name, .data = bytes });
    try std.testing.expectError(error.ManifestCorrupt, f.store.warm(id, null));
}

test "workspace source reads bind exact bytes and digest for cold and edit verification" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    const text = try f.store.readSource(a, "a.txt", null);
    defer a.free(text.bytes);
    try std.testing.expectEqualStrings("alpha", text.bytes);
    var expected: wm.Digest = undefined;
    std.crypto.hash.sha2.Sha256.hash("alpha", &expected, .{});
    try std.testing.expectEqualSlices(u8, &expected, &text.sha256);
    const bound: wm.Source = .{
        .path = "a.txt",
        .tokens = &.{2},
        .size = @intCast(text.bytes.len),
        .sha256 = text.sha256,
    };
    var cold = try f.store.cold(id, &.{1}, &.{bound}, null);
    defer cold.deinit();
    var edited = try f.store.edit(id, &.{1}, &.{bound}, null);
    defer edited.deinit();
    try f.write("a.txt", "Alpha");
    try std.testing.expectError(error.WorkspaceFileChanged, f.store.warm(id, null));
}

test "workspace source reads refuse traversal nonregular files and symlinks" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.tmp.dir.createDirPath(io, "repo/dir");
    try std.testing.expectError(error.InvalidRelativePath, f.store.readSource(a, "../outside.txt", null));
    try std.testing.expectError(error.WorkspaceFileNotRegular, f.store.readSource(a, "dir", null));
    if (@import("builtin").os.tag != .windows) {
        try f.tmp.dir.writeFile(io, .{ .sub_path = "outside.txt", .data = "outside" });
        try f.tmp.dir.symLink(io, "../outside.txt", "repo/link.txt", .{});
        try std.testing.expectError(error.WorkspacePathRefused, f.store.readSource(a, "link.txt", null));
    }
}

test "workspace source reads honor cancellation" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    try std.testing.expectError(error.Cancelled, f.store.readSource(a, "a.txt", stop()));
}

test "workspace manifest stays bound across ancestor substitution" {
    var f = try Fixture.init();
    defer f.deinit();
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    try f.tmp.dir.createDirPath(io, "container/repo");
    try f.tmp.dir.createDirPath(io, "container/state");
    try f.tmp.dir.setFilePermissions(io, "container/state", .fromMode(0o700), .{});
    try f.tmp.dir.createDirPath(io, "outside/repo");
    try f.tmp.dir.createDirPath(io, "outside/state");
    try f.tmp.dir.writeFile(io, .{ .sub_path = "container/repo/secret.txt", .data = "original-secret" });
    try f.tmp.dir.writeFile(io, .{ .sub_path = "outside/repo/secret.txt", .data = "outside-secret" });
    const root = try f.tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const workspace = try std.fs.path.join(a, &.{ root, "container/repo" });
    defer a.free(workspace);
    const state = try std.fs.path.join(a, &.{ root, "container/state" });
    defer a.free(state);
    var store = try wm.Store.init(a, io, workspace, state);
    defer store.deinit();
    try f.tmp.dir.rename("container", f.tmp.dir, "original-container", io);
    try f.tmp.dir.symLink(io, "outside", "container", .{});
    const text = try store.readSource(a, "secret.txt", null);
    defer a.free(text.bytes);
    try std.testing.expectEqualStrings("original-secret", text.bytes);
    var cold = try store.cold(id, &.{1}, &.{source("secret.txt", &.{2}, "original-secret")}, null);
    defer cold.deinit();
    var warm = try store.warm(id, null);
    defer warm.deinit();
    try std.testing.expectEqualSlices(u32, cold.manifest.prompt, warm.manifest.prompt);
    var outside = try f.tmp.dir.openDir(io, "outside/state", .{});
    defer outside.close(io);
    try std.testing.expectError(error.FileNotFound, outside.statFile(io, wm.manifest_name, .{}));
}

test "workspace manifest aggregate source bytes refuse before reading or publication" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    try f.write("b.txt", "bravo");
    var sources = [_]wm.Source{ source("a.txt", &.{2}, "alpha"), source("b.txt", &.{3}, "bravo") };
    sources[0].size = wm.max_source_bytes;
    sources[1].size = 1;
    try std.testing.expectError(error.ManifestTooLarge, f.store.cold(id, &.{1}, &sources, null));
    try expectStateEmpty(&f);
    sources[0].size = 5;
    sources[1].size = 5;
    var cold = try f.store.cold(id, &.{1}, &sources, null);
    defer cold.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, cold.manifest.prompt);
    var state = try f.tmp.dir.openDir(io, "state", .{});
    defer state.close(io);
    const published = try state.statFile(io, wm.manifest_name, .{ .follow_symlinks = false });
    try std.testing.expectEqual(@as(u32, 0), published.permissions.toMode() & 0o077);
}

test "workspace manifest empty UTF-8 sources retain lexical order and warm IDs" {
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "");
    try f.write("b.txt", "alpha");
    try f.write("c.txt", "");
    const sources = [_]wm.Source{ source("c.txt", &.{}, ""), source("b.txt", &.{ 2, 3 }, "alpha"), source("a.txt", &.{}, "") };
    var cold = try f.store.cold(id, &.{1}, &sources, null);
    defer cold.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, cold.manifest.prompt);
    try std.testing.expectEqualStrings("a.txt", cold.manifest.files[0].path);
    try std.testing.expectEqualStrings("b.txt", cold.manifest.files[1].path);
    try std.testing.expectEqualStrings("c.txt", cold.manifest.files[2].path);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 3 }, &.{ cold.manifest.files[0].prompt_end, cold.manifest.files[1].prompt_end, cold.manifest.files[2].prompt_end });
    var warm = try f.store.warm(id, null);
    defer warm.deinit();
    try std.testing.expectEqualSlices(u32, cold.manifest.prompt, warm.manifest.prompt);
    try f.write("b.txt", "bravo");
    const edited_sources = [_]wm.Source{ source("a.txt", &.{}, ""), source("b.txt", &.{4}, "bravo"), source("c.txt", &.{}, "") };
    var edited = try f.store.edit(id, &.{1}, &edited_sources, null);
    defer edited.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 1, 4 }, edited.manifest.prompt);
    try std.testing.expectEqual(@as(u32, 1), edited.reused_tokens);
    try std.testing.expectEqual(@as(u32, 1), edited.changed_files);
    try std.testing.expectError(error.EmptySourcePrompt, f.store.cold(id, &.{}, &.{source("a.txt", &.{}, "")}, null));
}

test "workspace manifest FIFO sources and state refuse without an open wait" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var f = try Fixture.init();
    defer f.deinit();
    const root = try f.tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const fifo = try std.fs.path.join(a, &.{ root, "repo/fifo" });
    defer a.free(fifo);
    const fifo_z = try a.dupeSentinel(u8, fifo, 0);
    defer a.free(fifo_z);
    try std.testing.expectEqual(@as(c_int, 0), mkfifo(fifo_z, 0o600));
    try std.testing.expectError(error.WorkspaceFileNotRegular, f.store.readSource(a, "fifo", null));
    try f.write("a.txt", "alpha");
    var cold = try f.store.cold(id, &.{1}, &.{source("a.txt", &.{2}, "alpha")}, null);
    defer cold.deinit();
    try f.tmp.dir.deleteFile(io, "state/workspace-manifest.bin");
    const state_fifo = try std.fs.path.join(a, &.{ root, "state/workspace-manifest.bin" });
    defer a.free(state_fifo);
    const state_fifo_z = try a.dupeSentinel(u8, state_fifo, 0);
    defer a.free(state_fifo_z);
    try std.testing.expectEqual(@as(c_int, 0), mkfifo(state_fifo_z, 0o600));
    try std.testing.expectError(error.ManifestRefused, f.store.warm(id, null));
}

test "workspace manifest private state is revalidated before load and atomic replacement" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var f = try Fixture.init();
    defer f.deinit();
    try f.write("a.txt", "alpha");
    var cold = try f.store.cold(id, &.{1}, &.{source("a.txt", &.{2}, "alpha")}, null);
    defer cold.deinit();
    try f.write("a.txt", "bravo");
    var next = try disk.capture(a, io, f.store.workspace, f.store.workspace_dir, id, &.{1}, &.{source("a.txt", &.{3}, "bravo")}, null);
    defer next.deinit();
    try f.tmp.dir.setFilePermissions(io, "state", .fromMode(0o755), .{});
    defer f.tmp.dir.setFilePermissions(io, "state", .fromMode(0o700), .{}) catch {};
    try std.testing.expectError(error.InsecureWorkspaceState, f.store.warm(id, null));
    try std.testing.expectError(error.InsecureWorkspaceState, f.store.commit(&next, true, null));
    try f.tmp.dir.setFilePermissions(io, "state", .fromMode(0o700), .{});
    try f.write("a.txt", "alpha");
    var warm = try f.store.warm(id, null);
    defer warm.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, warm.manifest.prompt);
    var state = try f.tmp.dir.openDir(io, "state", .{ .iterate = true });
    defer state.close(io);
    var entries = state.iterate();
    try std.testing.expectEqualStrings(wm.manifest_name, (try entries.next(io)).?.name);
    try std.testing.expect((try entries.next(io)) == null);
}

extern "c" fn mkfifo(path: [*:0]const u8, mode: std.posix.mode_t) c_int;
