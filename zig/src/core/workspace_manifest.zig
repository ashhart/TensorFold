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
pub const max_prompt_tokens = types.max_prompt_tokens;
pub const max_source_bytes = types.max_source_bytes;
pub const SourceText = disk.SourceText;

pub const Store = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    workspace: [:0]u8,
    state: [:0]u8,
    workspace_dir: std.Io.Dir,
    state_dir: std.Io.Dir,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, workspace: []const u8, state: []const u8) !Store {
        var pinned_workspace = try pinnedDirectory(gpa, io, workspace, error.InvalidWorkspace);
        errdefer pinned_workspace.deinit(gpa, io);
        var pinned_state = try pinnedDirectory(gpa, io, state, error.InvalidStateRoot);
        errdefer pinned_state.deinit(gpa, io);
        try validateState(io, pinned_state.dir);
        return .{
            .gpa = gpa,
            .io = io,
            .workspace = pinned_workspace.path,
            .state = pinned_state.path,
            .workspace_dir = pinned_workspace.dir,
            .state_dir = pinned_state.dir,
        };
    }

    pub fn deinit(s: *Store) void {
        s.workspace_dir.close(s.io);
        s.state_dir.close(s.io);
        s.gpa.free(s.state);
        s.gpa.free(s.workspace);
        s.* = undefined;
    }

    pub fn readSource(s: *Store, gpa: std.mem.Allocator, path: []const u8, stop: ?Stop) !SourceText {
        return disk.readSource(gpa, s.io, s.workspace_dir, path, stop);
    }
    pub fn readSourceBudget(s: *Store, gpa: std.mem.Allocator, path: []const u8, remaining_bytes: u64, stop: ?Stop) !SourceText {
        return disk.readSourceBudget(gpa, s.io, s.workspace_dir, path, remaining_bytes, stop);
    }
    pub fn cold(s: *Store, id: Identity, base_prompt: []const u32, sources: []const Source, stop: ?Stop) !Revision {
        var revision = try s.prepareCold(id, base_prompt, sources, stop);
        errdefer revision.deinit();
        try s.commit(&revision.manifest, false, stop);
        return revision;
    }
    pub fn prepareCold(s: *Store, id: Identity, base_prompt: []const u32, sources: []const Source, stop: ?Stop) !Revision {
        var m = try disk.capture(s.gpa, s.io, s.workspace, s.workspace_dir, id, base_prompt, sources, stop);
        errdefer m.deinit();
        try disk.verify(s.io, s.workspace, s.workspace_dir, &m, stop);
        return .{ .manifest = m, .reused_tokens = 0, .changed_files = 0 };
    }

    pub fn warm(s: *Store, id: Identity, stop: ?Stop) !Revision {
        try validateState(s.io, s.state_dir);
        var m = try disk.load(s.gpa, s.io, s.state_dir);
        errdefer m.deinit();
        if (!m.identityMatches(s.workspace, id)) return error.NamespaceMismatch;
        try disk.verify(s.io, s.workspace, s.workspace_dir, &m, stop);
        return .{ .manifest = m, .reused_tokens = @intCast(m.prompt.len), .changed_files = 0 };
    }

    pub fn edit(s: *Store, id: Identity, base_prompt: []const u32, sources: []const Source, stop: ?Stop) !Revision {
        var revision = try s.prepareEdit(id, base_prompt, sources, stop);
        errdefer revision.deinit();
        try s.commit(&revision.manifest, true, stop);
        return revision;
    }
    pub fn prepareEdit(s: *Store, id: Identity, base_prompt: []const u32, sources: []const Source, stop: ?Stop) !Revision {
        try validateState(s.io, s.state_dir);
        var old = try disk.load(s.gpa, s.io, s.state_dir);
        defer old.deinit();
        if (!old.identityMatches(s.workspace, id)) return error.NamespaceMismatch;
        var m = try disk.capture(s.gpa, s.io, s.workspace, s.workspace_dir, id, base_prompt, sources, stop);
        errdefer m.deinit();
        const reused = exactPrefix(&old, &m);
        const changed = changedFiles(&old, &m);
        try disk.verify(s.io, s.workspace, s.workspace_dir, &m, stop);
        return .{ .manifest = m, .reused_tokens = reused, .changed_files = changed };
    }

    pub fn commit(s: *Store, m: *Manifest, replace_existing: bool, stop: ?Stop) !void {
        try validateState(s.io, s.state_dir);
        return disk.publish(s.gpa, s.io, s.state_dir, m, replace_existing, stop);
    }
};

const PinnedDirectory = struct {
    path: [:0]u8,
    dir: std.Io.Dir,

    fn deinit(p: *PinnedDirectory, gpa: std.mem.Allocator, io: std.Io) void {
        p.dir.close(io);
        gpa.free(p.path);
    }
};

fn validateState(io: std.Io, dir: std.Io.Dir) !void {
    const stat = try dir.stat(io);
    if (stat.kind != .directory or stat.permissions.toMode() & 0o077 != 0) return error.InsecureWorkspaceState;
}

fn pinnedDirectory(gpa: std.mem.Allocator, io: std.Io, path: []const u8, comptime failure: anyerror) !PinnedDirectory {
    const canonical = std.Io.Dir.cwd().realPathFileAlloc(io, path, gpa) catch return failure;
    errdefer gpa.free(canonical);
    const dir = std.Io.Dir.cwd().openDir(io, canonical, .{ .follow_symlinks = false }) catch return failure;
    return .{ .path = canonical, .dir = dir };
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
