//! The `pull` command: a Hugging Face checkpoint into the same cache 0.6.6 uses, resumable and verified.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");
const ranged = @import("pull_parts.zig");
const single = @import("pull_file.zig");
const Entry = single.Entry;
const blobName = single.blobName;
const fileStat = single.fileStat;
const downloadRetrying = single.downloadRetrying;
const linkIntoSnapshot = single.linkIntoSnapshot;

pub const GiB: u64 = 1 << 30;

/// Where files switch to shared pieces, and how big a piece is (tests use small ones).
pub const Sizes = struct { threshold: u64 = ranged.threshold, piece: u64 = ranged.piece };

/// A large LFS file on the shared queue, named by its sha256 once its bytes check out.
const Big = struct { entry: Entry, blob: []const u8, file: ranged.File };

/// Downloads ``repo[@revision]`` into the cache and prints where it landed. Returns the exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, err_out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, spec: []const u8) !u8 {
    return runWith(a, io, out, err_out, env, override, spec, .{});
}

pub fn runWith(a: Allocator, io: std.Io, out: *std.Io.Writer, err_out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, spec: []const u8, sizes: Sizes) !u8 {
    const at = std.mem.indexOfScalar(u8, spec, '@');
    const repo = if (at) |i| spec[0..i] else spec;
    const revision = if (at) |i| spec[i + 1 ..] else "main";
    if (!hub.isRepoIdLike(repo)) {
        try err_out.print("{s} is not a Hugging Face repo id (owner/name)\n", .{spec});
        return 2;
    }
    const endpoint = envValue(env, "HF_ENDPOINT") orelse "https://huggingface.co";
    const root = try hub.cacheDir(a, env, override);
    var client: std.http.Client = .{ .allocator = a, .io = io };
    defer client.deinit();

    const sha = resolveRevision(a, &client, out, endpoint, repo, revision) catch |e| switch (e) {
        error.HubStatus => return 1,
        error.BadRevision => {
            try err_out.print("{s}@{s}: not a revision Hugging Face names (a branch, tag or 40-hex commit)\n", .{ repo, revision });
            return 1;
        },
        else => {
            try err_out.print("{s}@{s}: the hub is unreachable ({t}); check the network or HF_ENDPOINT\n", .{ repo, revision, e });
            return 1;
        },
    };

    const entries = listTree(a, &client, endpoint, repo, sha) catch |e| switch (e) {
        error.HubStatus, error.NoConfig => return 1,
        else => {
            try err_out.print("{s}@{s}: the hub's file tree is unreachable ({t})\n", .{ repo, revision, e });
            return 1;
        },
    };

    // The family check runs on config.json before any weight moves.
    var config: ?Entry = null;
    for (entries) |e| if (std.mem.eql(u8, e.path, "config.json")) {
        config = e;
        break;
    };
    if (config == null) {
        try err_out.print("{s}: the hub tree has no config.json; refusing\n", .{repo});
        return 1;
    }
    const config_bytes = try fetchOne(a, &client, endpoint, repo, sha, "config.json");
    const kind = configKind(a, config_bytes);
    const model_type = kind.model_type;
    if (kind.drafter) {
        try out.print("{s}: DFlash2 drafter ({s}), for serve --drafter\n", .{ repo, model_type });
    } else if (hub.family(model_type)) |family| {
        try out.print("{s}: {s} ({s})\n", .{ repo, family.title, family.model_type });
    } else {
        try err_out.print("{s}: no registered Zig family serves model_type {s}; the 0.6 line may: tensorfold@0.6\n", .{ repo, model_type });
        return 1;
    }

    const repo_dir = try std.fs.path.join(a, &.{ root, try hub.repoDirName(a, repo) });
    const blobs_dir = try std.fs.path.join(a, &.{ repo_dir, "blobs" });
    const snapshot_dir = try std.fs.path.join(a, &.{ repo_dir, "snapshots", sha });
    const refs_dir = try std.fs.path.join(a, &.{ repo_dir, "refs" });
    const w = std.Io.Dir.cwd();
    for ([_][]const u8{ blobs_dir, snapshot_dir, refs_dir }) |d| try w.createDirPath(io, d);

    var big: std.ArrayList(Big) = .empty;
    for (entries) |e| {
        const blob_name = try blobName(a, e);
        const blob_path = try std.fs.path.join(a, &.{ blobs_dir, blob_name });
        if (e.sha256 != null or e.git_sha1 != null) blob_exists: {
            const size = fileStat(io, blob_path) catch break :blob_exists;
            if (size != e.size) break :blob_exists;
            try out.print("  {s}: cached\n", .{e.path});
            try linkIntoSnapshot(a, io, snapshot_dir, e.path, blob_name);
            continue;
        }
        const url = try std.fmt.allocPrint(a, "{s}/{s}/resolve/{s}/{s}", .{ endpoint, repo, sha, e.path });
        if (e.sha256 != null and e.size >= sizes.threshold) {
            try big.append(a, .{ .entry = e, .blob = blob_name, .file = .{ .url = url, .path = try std.fmt.allocPrint(a, "{s}.ranges", .{blob_path}), .size = e.size } });
            continue;
        }
        const got = downloadRetrying(a, io, &client, out, url, blobs_dir, e) catch |err| {
            try err_out.print("  {s}: the download failed ({t})\n", .{ e.path, err });
            return 1;
        };
        try linkIntoSnapshot(a, io, snapshot_dir, e.path, got);
    }
    if (!try fetchBig(a, io, &client, out, err_out, blobs_dir, snapshot_dir, big.items, sizes.piece)) return 1;
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ refs_dir, "main" }), .data = sha });
    try out.print("stored {s}@{s} at {s}\n", .{ repo, sha, snapshot_dir });
    return 0;
}

/// The large files as pieces on one queue; each matches its sha256 before it becomes a blob, else it comes once more.
fn fetchBig(a: Allocator, io: std.Io, client: *std.http.Client, out: *std.Io.Writer, err_out: *std.Io.Writer, blobs_dir: []const u8, snapshot_dir: []const u8, big: []const Big, piece: u64) !bool {
    if (big.len == 0) return true;
    const files = try a.alloc(ranged.File, big.len);
    for (files, big) |*f, b| f.* = b.file;
    ranged.fetch(a, io, files, piece, out) catch |err| switch (err) {
        error.RangeUnsupported => {
            for (big) |b| {
                ranged.discard(a, io, b.file, piece); // the hub sent whole files: one stream each
                const got = downloadRetrying(a, io, client, out, b.file.url, blobs_dir, b.entry) catch |e| {
                    try err_out.print("  {s}: the download failed ({t})\n", .{ b.entry.path, e });
                    return false;
                };
                try linkIntoSnapshot(a, io, snapshot_dir, b.entry.path, got);
            }
            return true;
        },
        else => {
            try err_out.print("  the download failed ({t}); pull again to fetch only the missing pieces\n", .{err});
            return false;
        },
    };
    for (big) |b| {
        if (!try matches(a, io, b)) {
            ranged.discard(a, io, b.file, piece);
            ranged.fetch(a, io, &.{b.file}, piece, out) catch |err| {
                try err_out.print("  {s}: the download failed ({t})\n", .{ b.entry.path, err });
                return false;
            };
            if (!try matches(a, io, b)) {
                ranged.discard(a, io, b.file, piece);
                try err_out.print("  {s}: the bytes do not match the hub's sha256 twice; refusing\n", .{b.entry.path});
                return false;
            }
        }
        try std.Io.Dir.renameAbsolute(b.file.path, try std.fs.path.join(a, &.{ blobs_dir, b.blob }), io);
        try out.print("  {s}: {d:.2} MiB in {d} pieces\n", .{ b.entry.path, @as(f64, @floatFromInt(b.entry.size)) / (1 << 20), ranged.count(b.entry.size, piece) });
        try linkIntoSnapshot(a, io, snapshot_dir, b.entry.path, b.blob);
    }
    return true;
}

fn matches(a: Allocator, io: std.Io, b: Big) !bool {
    return std.mem.eql(u8, &(try ranged.digest(a, io, b.file.path, b.file.size)), &b.entry.sha256.?);
}

fn envValue(env: ?*const std.process.Environ.Map, key: []const u8) ?[]const u8 {
    return if (env) |m| m.get(key) else null;
}

/// The commit the hub serves for ``repo@revision``.
fn resolveRevision(a: Allocator, client: *std.http.Client, out: *std.Io.Writer, endpoint: []const u8, repo: []const u8, revision: []const u8) ![]const u8 {
    if (!revisionOk(revision)) return error.BadRevision;
    const url = try std.fmt.allocPrint(a, "{s}/api/models/{s}/revision/{s}", .{ endpoint, repo, revision });
    const body = try getJson(a, client, url);
    try out.print("[tensorfold] downloading {s}@{s} from Hugging Face\n", .{ repo, revision });
    var parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch return error.BadHubJson;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadHubJson;
    const sha = parsed.value.object.get("sha") orelse return error.BadHubJson;
    if (sha != .string or sha.string.len == 0) return error.BadHubJson;
    // The sha names the snapshot dir and is written to refs/main: only a 40-hex git sha is a path.
    if (!hexSha(sha.string)) return error.BadHubJson;
    return try a.dupe(u8, sha.string);
}

/// A git sha: 40 hex characters, the only strings ever joined into a cache path.
fn hexSha(text: []const u8) bool {
    if (text.len != 40) return false;
    for (text) |ch| {
        const ok = (ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f') or (ch >= 'A' and ch <= 'F');
        if (!ok) return false;
    }
    return true;
}

/// A branch, tag or sha: the characters a URL's path segment may hold, and no dots.
fn revisionOk(text: []const u8) bool {
    if (text.len == 0 or std.mem.indexOf(u8, text, "..") != null) return false;
    for (text) |ch| {
        const ok = std.ascii.isAlphanumeric(ch) or ch == '-' or ch == '_' or ch == '.';
        if (!ok) return false;
    }
    return true;
}

/// The revision's files: path, size and the sha256 the hub states for LFS objects.
fn listTree(a: Allocator, client: *std.http.Client, endpoint: []const u8, repo: []const u8, sha: []const u8) ![]Entry {
    const url = try std.fmt.allocPrint(a, "{s}/api/models/{s}/tree/{s}?recursive=true", .{ endpoint, repo, sha });
    const body = try getJson(a, client, url);
    var parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch return error.BadHubJson;
    defer parsed.deinit();
    if (parsed.value != .array) return error.BadHubJson;
    var entries: std.ArrayList(Entry) = .empty;
    for (parsed.value.array.items) |item| {
        if (item != .object) continue;
        const o = item.object;
        const kind = o.get("type") orelse continue;
        if (kind != .string or !std.mem.eql(u8, kind.string, "file")) continue;
        const path_v = o.get("path") orelse continue;
        if (path_v != .string) continue;
        const size_v = o.get("size");
        const size: u64 = if (size_v != null and size_v.? == .integer and size_v.?.integer >= 0) @intCast(size_v.?.integer) else 0;
        var sha256: ?[32]u8 = null;
        if (o.get("lfs")) |lfs| {
            if (lfs == .object) {
                if (lfs.object.get("oid")) |oid| {
                    if (oid == .string) sha256 = lfsDigest(oid.string);
                }
            }
        }
        var git_sha1: ?[20]u8 = null; // an LFS entry's own oid hashes its pointer file, not the bytes
        if (sha256 == null) if (o.get("oid")) |oid| if (oid == .string) {
            git_sha1 = parseHex(20, oid.string);
        };
        try entries.append(a, .{ .path = try a.dupe(u8, path_v.string), .size = size, .sha256 = sha256, .git_sha1 = git_sha1 });
    }
    if (entries.items.len == 0) return error.NoConfig;
    return entries.items;
}

/// An LFS oid's sha256: the hub's tree sends bare hex, pointer files say "sha256:" first.
fn lfsDigest(oid: []const u8) ?[32]u8 {
    return parseHex(32, if (std.mem.startsWith(u8, oid, "sha256:")) oid[7..] else oid);
}

/// config.json's model_type from raw bytes.
/// config.json's model_type, copied out of the parse, and whether it is a DFlash2 drafter (`serve --drafter`).
fn configKind(a: Allocator, bytes: []const u8) struct { model_type: []const u8, drafter: bool } {
    var parsed = std.json.parseFromSlice(std.json.Value, a, bytes, .{}) catch return .{ .model_type = "unknown", .drafter = false };
    defer parsed.deinit();
    if (parsed.value != .object) return .{ .model_type = "unknown", .drafter = false };
    const t = parsed.value.object.get("model_type");
    const model_type = if (t != null and t.? == .string) a.dupe(u8, t.?.string) catch "unknown" else "unknown";
    return .{ .model_type = model_type, .drafter = parsed.value.object.get("dflash_config") != null };
}

fn parseHex(comptime n: usize, text: []const u8) ?[n]u8 {
    if (text.len != 2 * n) return null;
    var out: [n]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return null;
    return out;
}

/// One small file fetched whole into memory.
fn fetchOne(a: Allocator, client: *std.http.Client, endpoint: []const u8, repo: []const u8, sha: []const u8, path: []const u8) ![]u8 {
    const url = try std.fmt.allocPrint(a, "{s}/{s}/resolve/{s}/{s}", .{ endpoint, repo, sha, path });
    var body: std.Io.Writer.Allocating = .init(a);
    const result = client.fetch(.{ .location = .{ .url = url }, .response_writer = &body.writer }) catch return error.HubUnreachable;
    if (result.status != .ok) return error.HubStatus;
    return body.toOwnedSlice();
}

fn getJson(a: Allocator, client: *std.http.Client, url: []const u8) ![]u8 {
    var body: std.Io.Writer.Allocating = .init(a);
    const result = client.fetch(.{ .location = .{ .url = url }, .response_writer = &body.writer }) catch return error.HubUnreachable;
    if (result.status != .ok) return error.HubStatus;
    return body.toOwnedSlice();
}

test "an LFS oid names its sha256 with or without the sha256: prefix" {
    const want: [32]u8 = @splat(0xa5);
    const hex = std.fmt.bytesToHex(want, .lower);
    try std.testing.expectEqualSlices(u8, &want, &lfsDigest(&hex).?);
    try std.testing.expectEqualSlices(u8, &want, &lfsDigest("sha256:" ++ hex).?);
    try std.testing.expect(lfsDigest("sha1:abc") == null and lfsDigest(hex[1..]) == null);
}
