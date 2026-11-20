const std = @import("std");
const disk = @import("workspace_manifest_io.zig");
const types = @import("workspace_manifest_types.zig");

pub const Digest = types.Digest;
pub const File = types.File;
pub const Identity = types.Identity;
pub const Manifest = types.Manifest;
pub const Revision = types.Revision;
pub const Source = types.Source;
pub const Stop = types.Stop;
pub const manifest_name = types.manifest_name;
pub const SourceText = disk.SourceText;

pub const Store = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    workspace: [:0]u8,
    state: [:0]u8,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, workspace: []const u8, state: []const u8) !Store {
        const canonical_workspace = try canonicalDirectory(gpa, io, workspace, error.InvalidWorkspace);
        errdefer gpa.free(canonical_workspace);
        const canonical_state = try canonicalDirectory(gpa, io, state, error.InvalidStateRoot);
        errdefer gpa.free(canonical_state);
        return .{ .gpa = gpa, .io = io, .workspace = canonical_workspace, .state = canonical_state };
    }

    pub fn deinit(s: *Store) void {
        s.gpa.free(s.state);
        s.gpa.free(s.workspace);
        s.* = undefined;
    }

    pub fn readSource(s: *Store, gpa: std.mem.Allocator, path: []const u8, stop: ?Stop) !SourceText {
        return disk.readSource(gpa, s.io, s.workspace, path, stop);
    }
    pub fn cold(s: *Store, id: Identity, base_prompt: []const u32, sources: []const Source, stop: ?Stop) !Revision {
        var m = try disk.capture(s.gpa, s.io, s.workspace, id, base_prompt, sources, stop);
        errdefer m.deinit();
        try disk.verify(s.io, s.workspace, &m, stop);
        try disk.publish(s.gpa, s.io, s.state, &m, false, stop);
        return .{ .manifest = m, .reused_tokens = 0, .changed_files = 0 };
    }

    pub fn warm(s: *Store, id: Identity, stop: ?Stop) !Revision {
        var m = try disk.load(s.gpa, s.io, s.state);
        errdefer m.deinit();
        if (!m.identityMatches(s.workspace, id)) return error.NamespaceMismatch;
        try disk.verify(s.io, s.workspace, &m, stop);
        return .{ .manifest = m, .reused_tokens = @intCast(m.prompt.len), .changed_files = 0 };
    }

    pub fn edit(s: *Store, id: Identity, base_prompt: []const u32, sources: []const Source, stop: ?Stop) !Revision {
        var old = try disk.load(s.gpa, s.io, s.state);
        defer old.deinit();
        if (!old.identityMatches(s.workspace, id)) return error.NamespaceMismatch;
        var m = try disk.capture(s.gpa, s.io, s.workspace, id, base_prompt, sources, stop);
        errdefer m.deinit();
        const reused = exactPrefix(&old, &m);
        const changed = changedFiles(&old, &m);
        try disk.verify(s.io, s.workspace, &m, stop);
        try disk.publish(s.gpa, s.io, s.state, &m, true, stop);
        return .{ .manifest = m, .reused_tokens = reused, .changed_files = changed };
    }
};

fn canonicalDirectory(gpa: std.mem.Allocator, io: std.Io, path: []const u8, comptime failure: anyerror) ![:0]u8 {
    const canonical = std.Io.Dir.cwd().realPathFileAlloc(io, path, gpa) catch return failure;
    errdefer gpa.free(canonical);
    const dir = std.Io.Dir.cwd().openDir(io, canonical, .{ .follow_symlinks = false }) catch return failure;
    dir.close(io);
    return canonical;
}

fn exactPrefix(old: *const Manifest, new: *const Manifest) u32 {
    const count = @min(old.prompt.len, new.prompt.len);
    var common: usize = 0;
    while (common < count and old.prompt[common] == new.prompt[common]) : (common += 1) {}
    var reused: u32 = if (old.base_prefix_end <= common) old.base_prefix_end else 0;
    for (old.files) |f| {
        if (f.prompt_end > common) break;
        reused = f.prompt_end;
    }
    return reused;
}

fn changedFiles(old: *const Manifest, new: *const Manifest) u32 {
    var i: usize = 0;
    var j: usize = 0;
    var changed: u32 = 0;
    while (i < old.files.len or j < new.files.len) {
        if (i == old.files.len) {
            changed += 1;
            j += 1;
            continue;
        }
        if (j == new.files.len) {
            changed += 1;
            i += 1;
            continue;
        }
        const a = old.files[i];
        const b = new.files[j];
        switch (std.mem.order(u8, a.path, b.path)) {
            .lt => {
                changed += 1;
                i += 1;
            },
            .gt => {
                changed += 1;
                j += 1;
            },
            .eq => {
                if (a.size != b.size or !std.mem.eql(u8, &a.sha256, &b.sha256)) changed += 1;
                i += 1;
                j += 1;
            },
        }
    }
    return changed;
}
