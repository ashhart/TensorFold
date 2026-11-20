const std = @import("std");
const codec = @import("workspace_manifest_codec.zig");
const types = @import("workspace_manifest_types.zig");

const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

fn openWorkspace(io: Io, workspace: []const u8) !Io.Dir {
    return Io.Dir.cwd().openDir(io, workspace, .{ .follow_symlinks = false }) catch |err| switch (err) {
        error.FileNotFound => error.WorkspaceFileMissing,
        error.Canceled => error.Canceled,
        else => error.WorkspacePathRefused,
    };
}

fn openRelative(io: Io, root: Io.Dir, path: []const u8) !Io.File {
    try types.validateRelativePath(path);
    const slash = std.mem.lastIndexOfScalar(u8, path, '/');
    var current = root;
    var owned: ?Io.Dir = null;
    defer if (owned) |dir| dir.close(io);
    if (slash) |at| {
        var parts = std.mem.splitScalar(u8, path[0..at], '/');
        while (parts.next()) |part| {
            const next = current.openDir(io, part, .{ .follow_symlinks = false }) catch |err| switch (err) {
                error.FileNotFound => return error.WorkspaceFileMissing,
                error.Canceled => return error.Canceled,
                else => return error.WorkspacePathRefused,
            };
            if (owned) |dir| dir.close(io);
            owned = next;
            current = next;
        }
    }
    const name = if (slash) |at| path[at + 1 ..] else path;
    return current.openFile(io, name, .{ .follow_symlinks = false, .resolve_beneath = true }) catch |err| switch (err) {
        error.FileNotFound => error.WorkspaceFileMissing,
        error.Canceled => error.Canceled,
        else => error.WorkspacePathRefused,
    };
}

fn sameVersion(a: Io.File.Stat, b: Io.File.Stat) bool {
    return a.inode == b.inode and a.kind == b.kind and a.size == b.size and
        a.mtime.toNanoseconds() == b.mtime.toNanoseconds() and
        a.ctime.toNanoseconds() == b.ctime.toNanoseconds();
}

fn digestFile(io: Io, root: Io.Dir, relative: []const u8, expected: ?*const types.File, out: *types.File, stop: ?types.Stop) !void {
    try types.checkStop(stop);
    const file = try openRelative(io, root, relative);
    defer file.close(io);
    const before = try file.stat(io);
    if (before.kind != .file) return error.WorkspaceFileNotRegular;
    if (before.size > types.max_file_bytes) return error.WorkspaceFileTooLarge;
    if (expected) |old| if (old.size != before.size) return error.WorkspaceFileChanged;
    var hash = Sha256.init(.{});
    var buffer: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;
    while (offset < before.size) {
        try types.checkStop(stop);
        const wanted: usize = @intCast(@min(buffer.len, before.size - offset));
        const n = try file.readPositionalAll(io, buffer[0..wanted], offset);
        if (n == 0) return error.WorkspaceFileChanged;
        hash.update(buffer[0..n]);
        offset += n;
    }
    var extra: [1]u8 = undefined;
    if (try file.readPositionalAll(io, &extra, before.size) != 0) return error.WorkspaceFileChanged;
    const after = try file.stat(io);
    if (!sameVersion(before, after)) return error.WorkspaceFileChanged;
    const current = try openRelative(io, root, relative);
    defer current.close(io);
    if (!sameVersion(after, try current.stat(io))) return error.WorkspaceFileChanged;
    const digest = hash.finalResult();
    if (expected) |old| if (!std.mem.eql(u8, &old.sha256, &digest)) return error.WorkspaceFileChanged;
    out.size = before.size;
    out.sha256 = digest;
}

pub fn verify(io: Io, workspace: []const u8, m: *const types.Manifest, stop: ?types.Stop) !void {
    if (!std.mem.eql(u8, workspace, m.workspace)) return error.NamespaceMismatch;
    const root = try openWorkspace(io, workspace);
    defer root.close(io);
    for (m.files) |*f| {
        var current: types.File = undefined;
        try digestFile(io, root, f.path, f, &current, stop);
    }
}

pub fn capture(
    gpa: std.mem.Allocator,
    io: Io,
    workspace: []const u8,
    id: types.Identity,
    base_prompt: []const u32,
    sources: []const types.Source,
    stop: ?types.Stop,
) !types.Manifest {
    try types.checkStop(stop);
    try types.validateIdentity(id);
    if (!std.fs.path.isAbsolute(workspace) or sources.len == 0 or sources.len > types.max_files) return error.InvalidWorkspace;
    const ordered = try gpa.dupe(types.Source, sources);
    defer gpa.free(ordered);
    std.mem.sort(types.Source, ordered, {}, struct {
        fn less(_: void, a: types.Source, b: types.Source) bool {
            return std.mem.order(u8, a.path, b.path) == .lt;
        }
    }.less);
    var token_count = base_prompt.len;
    for (ordered, 0..) |source, i| {
        try types.validateRelativePath(source.path);
        if (i > 0 and std.mem.eql(u8, ordered[i - 1].path, source.path)) return error.DuplicateWorkspaceFile;
        if (source.tokens.len == 0) return error.EmptySourcePrompt;
        token_count = std.math.add(usize, token_count, source.tokens.len) catch return error.ManifestTooLarge;
        if (token_count > types.max_prompt_tokens) return error.ManifestTooLarge;
    }
    const owned_workspace = try gpa.dupeSentinel(u8, workspace, 0);
    errdefer gpa.free(owned_workspace);
    const model = try gpa.dupe(u8, id.model);
    errdefer gpa.free(model);
    const template = try gpa.dupe(u8, id.template);
    errdefer gpa.free(template);
    const cache = try gpa.dupe(u8, id.cache);
    errdefer gpa.free(cache);
    const files = try gpa.alloc(types.File, ordered.len);
    var initialized: usize = 0;
    errdefer {
        for (files[0..initialized]) |f| gpa.free(f.path);
        gpa.free(files);
    }
    const prompt = try gpa.alloc(u32, token_count);
    errdefer gpa.free(prompt);
    @memcpy(prompt[0..base_prompt.len], base_prompt);
    var prompt_at = base_prompt.len;
    const root = try openWorkspace(io, workspace);
    defer root.close(io);
    for (ordered, files) |source, *f| {
        f.path = try gpa.dupe(u8, source.path);
        initialized += 1;
        @memcpy(prompt[prompt_at..][0..source.tokens.len], source.tokens);
        prompt_at += source.tokens.len;
        f.prompt_end = @intCast(prompt_at);
        try digestFile(io, root, source.path, null, f, stop);
    }
    return .{
        .gpa = gpa,
        .workspace = owned_workspace,
        .model = model,
        .template = template,
        .cache = cache,
        .files = files,
        .base_prefix_end = @intCast(base_prompt.len),
        .prompt = prompt,
        .provenance = undefined,
    };
}

pub fn load(gpa: std.mem.Allocator, io: Io, state: []const u8) !types.Manifest {
    const dir = Io.Dir.cwd().openDir(io, state, .{ .follow_symlinks = false }) catch return error.ManifestRefused;
    defer dir.close(io);
    const file = dir.openFile(io, types.manifest_name, .{ .follow_symlinks = false, .resolve_beneath = true }) catch |err| switch (err) {
        error.FileNotFound => return error.ManifestMissing,
        else => return error.ManifestRefused,
    };
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.ManifestRefused;
    if (stat.size > types.max_manifest_bytes) return error.ManifestCorrupt;
    const bytes = try gpa.alloc(u8, @intCast(stat.size));
    defer gpa.free(bytes);
    const n = try file.readPositionalAll(io, bytes, 0);
    if (n != bytes.len) return error.ManifestCorrupt;
    var extra: [1]u8 = undefined;
    if (try file.readPositionalAll(io, &extra, bytes.len) != 0) return error.ManifestCorrupt;
    return codec.decode(gpa, bytes);
}

pub fn publish(gpa: std.mem.Allocator, io: Io, state: []const u8, m: *types.Manifest, replace_existing: bool, stop: ?types.Stop) !void {
    try types.checkStop(stop);
    const bytes = try codec.encode(m);
    defer gpa.free(bytes);
    m.provenance = bytes[bytes.len - 32 ..][0..32].*;
    const dir = Io.Dir.cwd().openDir(io, state, .{ .follow_symlinks = false }) catch return error.ManifestRefused;
    defer dir.close(io);
    var atomic = try dir.createFileAtomic(io, types.manifest_name, .{ .replace = replace_existing });
    defer atomic.deinit(io);
    try atomic.file.writePositionalAll(io, bytes, 0);
    try atomic.file.sync(io);
    try types.checkStop(stop);
    if (replace_existing) try atomic.replace(io) else try atomic.link(io);
}
