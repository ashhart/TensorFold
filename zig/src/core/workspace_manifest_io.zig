const std = @import("std");
const builtin = @import("builtin");
const codec = @import("workspace_manifest_codec.zig");
const types = @import("workspace_manifest_types.zig");

const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const SourceText = struct {
    bytes: []u8,
    sha256: types.Digest,
};

fn openRegular(io: Io, root: Io.Dir, name: []const u8) !Io.File {
    const stat = root.statFile(io, name, .{ .follow_symlinks = false }) catch |err| switch (err) {
        error.FileNotFound => return error.WorkspaceFileMissing,
        error.Canceled => return error.Canceled,
        else => return error.WorkspacePathRefused,
    };
    if (stat.kind == .sym_link) return error.WorkspacePathRefused;
    if (stat.kind != .file) return error.WorkspaceFileNotRegular;
    if (builtin.os.tag == .windows or builtin.os.tag == .wasi) return root.openFile(io, name, .{ .allow_directory = false, .follow_symlinks = false, .resolve_beneath = true }) catch |err| switch (err) {
        error.FileNotFound => error.WorkspaceFileMissing,
        error.Canceled => error.Canceled,
        else => error.WorkspacePathRefused,
    };
    var flags: std.posix.O = .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .NONBLOCK = true };
    if (@hasField(std.posix.O, "CLOEXEC")) flags.CLOEXEC = true;
    if (@hasField(std.posix.O, "RESOLVE_BENEATH")) flags.RESOLVE_BENEATH = true;
    const fd = std.posix.openat(root.handle, name, flags, 0) catch |err| switch (err) {
        error.FileNotFound => return error.WorkspaceFileMissing,
        else => return error.WorkspacePathRefused,
    };
    return .{ .handle = fd, .flags = .{ .nonblocking = false } };
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
    return openRegular(io, current, name);
}

fn sameVersion(a: Io.File.Stat, b: Io.File.Stat) bool {
    return a.inode == b.inode and a.kind == b.kind and a.size == b.size and
        a.mtime.toNanoseconds() == b.mtime.toNanoseconds() and
        a.ctime.toNanoseconds() == b.ctime.toNanoseconds();
}

fn readFile(
    gpa: ?std.mem.Allocator,
    io: Io,
    root: Io.Dir,
    relative: []const u8,
    expected: ?*const types.File,
    byte_limit: u64,
    out: *types.File,
    stop: ?types.Stop,
) !?[]u8 {
    try types.checkStop(stop);
    const file = try openRelative(io, root, relative);
    defer file.close(io);
    const before = try file.stat(io);
    if (before.kind != .file) return error.WorkspaceFileNotRegular;
    if (before.size > @min(types.max_file_bytes, byte_limit)) return error.WorkspaceFileTooLarge;
    if (expected) |old| if (old.size != before.size) return error.WorkspaceFileChanged;
    const bytes = if (gpa) |allocator| try allocator.alloc(u8, @intCast(before.size)) else null;
    errdefer if (bytes) |value| gpa.?.free(value);
    var hash = Sha256.init(.{});
    var buffer: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;
    while (offset < before.size) {
        try types.checkStop(stop);
        const wanted: usize = @intCast(@min(buffer.len, before.size - offset));
        const target = if (bytes) |value| value[@intCast(offset)..][0..wanted] else buffer[0..wanted];
        const n = try file.readPositionalAll(io, target, offset);
        if (n == 0) return error.WorkspaceFileChanged;
        hash.update(target[0..n]);
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
    return bytes;
}

fn digestFile(io: Io, root: Io.Dir, relative: []const u8, expected: ?*const types.File, out: *types.File, stop: ?types.Stop) !void {
    _ = try readFile(null, io, root, relative, expected, types.max_file_bytes, out, stop);
}

pub fn readSource(gpa: std.mem.Allocator, io: Io, root: Io.Dir, relative: []const u8, stop: ?types.Stop) !SourceText {
    var file: types.File = undefined;
    const bytes = (try readFile(gpa, io, root, relative, null, types.max_file_bytes, &file, stop)).?;
    return .{ .bytes = bytes, .sha256 = file.sha256 };
}

pub fn readSourceBudget(gpa: std.mem.Allocator, io: Io, root: Io.Dir, relative: []const u8, remaining_bytes: u64, stop: ?types.Stop) !SourceText {
    var file: types.File = undefined;
    const bytes = (try readFile(gpa, io, root, relative, null, remaining_bytes, &file, stop)).?;
    return .{ .bytes = bytes, .sha256 = file.sha256 };
}

pub fn verify(io: Io, workspace: []const u8, root: Io.Dir, m: *const types.Manifest, stop: ?types.Stop) !void {
    if (!std.mem.eql(u8, workspace, m.workspace)) return error.NamespaceMismatch;
    for (m.files) |*f| {
        var current: types.File = undefined;
        try digestFile(io, root, f.path, f, &current, stop);
    }
}

pub fn capture(
    gpa: std.mem.Allocator,
    io: Io,
    workspace: []const u8,
    root: Io.Dir,
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
    var byte_count: u64 = 0;
    for (ordered, 0..) |source, i| {
        try types.validateRelativePath(source.path);
        if (i > 0 and std.mem.eql(u8, ordered[i - 1].path, source.path)) return error.DuplicateWorkspaceFile;
        byte_count = std.math.add(u64, byte_count, source.size) catch return error.ManifestTooLarge;
        if (byte_count > types.max_source_bytes) return error.ManifestTooLarge;
        token_count = std.math.add(usize, token_count, source.tokens.len) catch return error.ManifestTooLarge;
        if (token_count > types.max_prompt_tokens) return error.ManifestTooLarge;
    }
    if (token_count == 0) return error.EmptySourcePrompt;
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
    for (ordered, files) |source, *f| {
        f.path = try gpa.dupe(u8, source.path);
        initialized += 1;
        @memcpy(prompt[prompt_at..][0..source.tokens.len], source.tokens);
        prompt_at += source.tokens.len;
        f.prompt_end = @intCast(prompt_at);
        f.size = source.size;
        f.sha256 = source.sha256;
        try digestFile(io, root, source.path, f, f, stop);
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

pub fn load(gpa: std.mem.Allocator, io: Io, dir: Io.Dir) !types.Manifest {
    const file = openRegular(io, dir, types.manifest_name) catch |err| switch (err) {
        error.WorkspaceFileMissing => return error.ManifestMissing,
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

pub fn publish(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, m: *types.Manifest, replace_existing: bool, stop: ?types.Stop) !void {
    try types.checkStop(stop);
    const bytes = try codec.encode(m);
    defer gpa.free(bytes);
    m.provenance = bytes[bytes.len - 32 ..][0..32].*;
    var atomic = try dir.createFileAtomic(io, types.manifest_name, .{ .replace = replace_existing, .permissions = .fromMode(0o600) });
    defer atomic.deinit(io);
    try atomic.file.writePositionalAll(io, bytes, 0);
    try atomic.file.sync(io);
    try types.checkStop(stop);
    if (replace_existing) try atomic.replace(io) else try atomic.link(io);
}
