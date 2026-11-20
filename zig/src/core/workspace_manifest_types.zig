const std = @import("std");

pub const Digest = [32]u8;
pub const manifest_name = "workspace-manifest.bin";
pub const max_files = 4096;
pub const max_file_bytes = 1 << 30;
pub const max_prompt_tokens = 1 << 24;
pub const max_identity_bytes = 4096;
pub const max_relative_path_bytes = 4096;
pub const max_manifest_bytes = 256 << 20;

pub const Stop = struct {
    ptr: *anyopaque,
    check: *const fn (*anyopaque) bool,
};

pub const Identity = struct {
    model: []const u8,
    template: []const u8,
    cache: []const u8,
};

pub const Source = struct {
    path: []const u8,
    tokens: []const u32,
};

pub const File = struct {
    path: []u8,
    size: u64,
    sha256: Digest,
    prompt_end: u32,
};

pub const Manifest = struct {
    gpa: std.mem.Allocator,
    workspace: [:0]u8,
    model: []u8,
    template: []u8,
    cache: []u8,
    files: []File,
    base_prefix_end: u32,
    prompt: []u32,
    provenance: Digest,

    pub fn deinit(m: *Manifest) void {
        for (m.files) |f| m.gpa.free(f.path);
        m.gpa.free(m.files);
        m.gpa.free(m.prompt);
        m.gpa.free(m.cache);
        m.gpa.free(m.template);
        m.gpa.free(m.model);
        m.gpa.free(m.workspace);
        m.* = undefined;
    }

    pub fn workspaceId(m: *const Manifest) Digest {
        return partsDigest(&.{m.workspace});
    }

    pub fn namespaceId(m: *const Manifest) Digest {
        return partsDigest(&.{ m.workspace, m.model, m.template, m.cache });
    }

    pub fn identityMatches(m: *const Manifest, workspace: []const u8, id: Identity) bool {
        return std.mem.eql(u8, m.workspace, workspace) and
            std.mem.eql(u8, m.model, id.model) and
            std.mem.eql(u8, m.template, id.template) and
            std.mem.eql(u8, m.cache, id.cache);
    }
};

pub const Revision = struct {
    manifest: Manifest,
    reused_tokens: u32,
    changed_files: u32,

    pub fn deinit(r: *Revision) void {
        r.manifest.deinit();
    }
};

pub fn checkStop(stop: ?Stop) !void {
    if (stop) |s| if (s.check(s.ptr)) return error.Cancelled;
}

pub fn validateIdentity(id: Identity) !void {
    for ([_][]const u8{ id.model, id.template, id.cache }) |part| {
        if (part.len == 0 or part.len > max_identity_bytes) return error.InvalidIdentity;
    }
}

pub fn validateRelativePath(path: []const u8) !void {
    if (path.len == 0 or path.len > max_relative_path_bytes or std.fs.path.isAbsolute(path)) return error.InvalidRelativePath;
    if (std.mem.indexOfAny(u8, path, "\x00\\:") != null) return error.InvalidRelativePath;
    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part| {
        if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return error.InvalidRelativePath;
    }
}

pub fn validateStructure(m: *const Manifest) !void {
    if (!std.fs.path.isAbsolute(m.workspace) or std.mem.indexOfScalar(u8, m.workspace, 0) != null) return error.InvalidWorkspace;
    try validateIdentity(.{ .model = m.model, .template = m.template, .cache = m.cache });
    if (m.files.len == 0 or m.files.len > max_files or m.prompt.len == 0 or m.prompt.len > max_prompt_tokens) return error.InvalidManifest;
    if (m.base_prefix_end >= m.prompt.len) return error.InvalidPromptPrefixes;
    var previous_path: []const u8 = "";
    var previous_end = m.base_prefix_end;
    for (m.files, 0..) |f, i| {
        try validateRelativePath(f.path);
        if (i > 0 and std.mem.order(u8, previous_path, f.path) != .lt) return error.UnorderedWorkspaceFiles;
        if (f.size > max_file_bytes or f.prompt_end <= previous_end or f.prompt_end > m.prompt.len) return error.InvalidManifest;
        previous_path = f.path;
        previous_end = f.prompt_end;
    }
    if (previous_end != m.prompt.len) return error.InvalidPromptPrefixes;
}

fn partsDigest(parts: []const []const u8) Digest {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("tensorfold/workspace-manifest/v1");
    for (parts) |part| {
        var length: [8]u8 = undefined;
        std.mem.writeInt(u64, &length, part.len, .little);
        hash.update(&length);
        hash.update(part);
    }
    return hash.finalResult();
}
