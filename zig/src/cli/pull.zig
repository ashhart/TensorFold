//! The `pull` command: a Hugging Face checkpoint into the same cache 0.6.6 uses, resumable and verified.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");
const Auth = @import("pull_auth.zig").Auth;
const ranged = @import("pull_parts.zig");
const single = @import("pull_file.zig");
const Entry = single.Entry;
const blobName = single.blobName;
const downloadRetrying = single.downloadRetrying;
const linkIntoSnapshot = single.linkIntoSnapshot;
const revisionOk = hub.revisionOk;
const route = @import("pull_route.zig");

pub const GiB: u64 = 1 << 30;
const revision_json_bytes = 1 << 20;
const tree_json_bytes = 16 << 20;
const config_json_bytes = 1 << 20;
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
    const auth = try Auth.init(a, io, env, err_out) orelse return 1;
    const endpoint = auth.endpoint;
    if (!try routes(a, io, err_out, env)) return 2;
    const root = try hub.cacheDir(a, env, override);
    const w = std.Io.Dir.cwd();
    try w.createDirPath(io, root);
    const mutation = @import("sys").folder_lock.hold(a, root, true) catch |e| {
        try err_out.print("tensorfold pull: cache {s} is busy or cannot be locked ({t}); retry after other pulls or removals finish\n", .{ root, e });
        return 1;
    };
    defer mutation.release();
    var client = @import("pull_http.zig").client(a, io);
    defer client.deinit();

    const sha = resolveRevision(a, &client, out, auth, repo, revision) catch |e| switch (e) {
        error.HubStatus => return 1,
        error.HubDenied => return auth.refuse(err_out, repo),
        error.BadRevision => {
            try err_out.print("{s}@{s}: not a revision Hugging Face names (a branch, tag or 40-hex commit)\n", .{ repo, revision });
            return 1;
        },
        error.HubTooLarge => {
            try err_out.print("{s}@{s}: revision metadata exceeds its decompressed byte cap\n", .{ repo, revision });
            return 1;
        },
        else => {
            try err_out.print("{s}@{s}: the hub at {s} is unreachable: ", .{ repo, revision, endpoint });
            try route.explain(io, err_out, e);
            try err_out.writeAll("; check the network, HF_ENDPOINT or the proxy settings\n");
            return 1;
        },
    };

    const entries = listTree(a, &client, auth, repo, sha) catch |e| switch (e) {
        error.HubStatus, error.NoConfig => return 1,
        error.HubDenied => return auth.refuse(err_out, repo),
        error.HubTooLarge => {
            try err_out.print("{s}@{s}: file tree exceeds its decompressed byte cap\n", .{ repo, revision });
            return 1;
        },
        else => {
            try err_out.print("{s}@{s}: the hub's file tree is unreachable: ", .{ repo, revision });
            try route.explain(io, err_out, e);
            try err_out.writeAll("\n");
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
    const config_bytes = fetchOne(a, &client, auth, repo, sha, "config.json") catch |e| switch (e) {
        error.HubDenied => return auth.refuse(err_out, repo),
        else => return e,
    };
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
    for ([_][]const u8{ blobs_dir, snapshot_dir, refs_dir }) |d| try w.createDirPath(io, d);
    // A failed same-revision repair must not leave its old completion marker behind.
    const ref_path = try std.fs.path.join(a, &.{ refs_dir, "main" });
    var ref_buf: [128]u8 = undefined;
    const published: ?[]const u8 = w.readFile(io, ref_path, &ref_buf) catch null;
    if (published) |text| if (std.mem.eql(u8, std.mem.trim(u8, text, " \r\n"), sha))
        try w.deleteFile(io, ref_path);

    var big: std.ArrayList(Big) = .empty;
    for (entries) |e| {
        const blob_name = blobName(a, e) catch |err| switch (err) {
            error.SnapshotPath => {
                try err_out.print("  {s}: a hub path must stay inside the snapshot and not use a name TensorFold reserves; refusing\n", .{e.path});
                return 1;
            },
            else => return err,
        };
        const blob_path = try std.fs.path.join(a, &.{ blobs_dir, blob_name });
        if (e.sha256 != null or e.git_sha1 != null) blob_exists: {
            if (!(single.cachedMatches(io, blob_path, e) catch false)) {
                std.Io.Dir.cwd().deleteFile(io, blob_path) catch {};
                break :blob_exists;
            }
            try out.print("  {s}: cached\n", .{e.path});
            try linkIntoSnapshot(a, io, snapshot_dir, e.path, blob_name);
            continue;
        }
        const url = try std.fmt.allocPrint(a, "{s}/{s}/resolve/{s}/{s}", .{ auth.endpoint, repo, sha, e.path });
        if (e.sha256 != null and e.size >= sizes.threshold) {
            try big.append(a, .{ .entry = e, .blob = blob_name, .file = .{ .url = url, .path = try std.fmt.allocPrint(a, "{s}.ranges", .{blob_path}), .size = e.size } });
            continue;
        }
        const got = downloadRetrying(a, io, &client, auth, out, url, blobs_dir, e) catch |err| {
            if (err == error.HubDenied) return auth.refuse(err_out, repo);
            try failed(io, err_out, e.path, err);
            return 1;
        };
        try linkIntoSnapshot(a, io, snapshot_dir, e.path, got);
    }
    const fetched = fetchBig(a, io, &client, auth, out, err_out, blobs_dir, snapshot_dir, big.items, sizes.piece) catch |e| switch (e) {
        error.HubDenied => return auth.refuse(err_out, repo),
        else => return e,
    };
    if (!fetched) return 1;
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ refs_dir, "main" }), .data = sha });
    try out.print("stored {s}@{s} at {s}\n", .{ repo, sha, snapshot_dir });
    return 0;
}

/// The large files as pieces on one queue; each matches its sha256 before it becomes a blob, else it comes once more.
fn fetchBig(a: Allocator, io: std.Io, client: *std.http.Client, auth: Auth, out: *std.Io.Writer, err_out: *std.Io.Writer, blobs_dir: []const u8, snapshot_dir: []const u8, big: []const Big, piece: u64) !bool {
    if (big.len == 0) return true;
    const files = try a.alloc(ranged.File, big.len);
    for (files, big) |*f, b| f.* = b.file;
    ranged.fetch(a, io, files, piece, auth, out) catch |err| switch (err) {
        error.RangeUnsupported => {
            for (big) |b| {
                ranged.discard(a, io, b.file, piece); // the hub sent whole files: one stream each
                const got = downloadRetrying(a, io, client, auth, out, b.file.url, blobs_dir, b.entry) catch |e| {
                    if (e == error.HubDenied) return e;
                    try failed(io, err_out, b.entry.path, e);
                    return false;
                };
                try linkIntoSnapshot(a, io, snapshot_dir, b.entry.path, got);
            }
            return true;
        },
        error.HubDenied => return err,
        else => {
            try err_out.writeAll("  the download failed: ");
            try route.explain(io, err_out, err);
            try err_out.writeAll("; pull again to fetch only the missing pieces\n");
            return false;
        },
    };
    for (big) |b| {
        if (!try matches(a, io, b)) {
            ranged.discard(a, io, b.file, piece);
            ranged.fetch(a, io, &.{b.file}, piece, auth, out) catch |err| {
                if (err == error.HubDenied) return err;
                try failed(io, err_out, b.entry.path, err);
                return false;
            };
            if (!try matches(a, io, b)) {
                ranged.discard(a, io, b.file, piece);
                try err_out.print("  {s}: the bytes do not match the hub's sha256 twice; refusing\n", .{b.entry.path});
                return false;
            }
        }
        try std.Io.Dir.cwd().rename(b.file.path, std.Io.Dir.cwd(), try std.fs.path.join(a, &.{ blobs_dir, b.blob }), io);
        try out.print("  {s}: {d:.2} MiB in {d} pieces\n", .{ b.entry.path, @as(f64, @floatFromInt(b.entry.size)) / (1 << 20), ranged.count(b.entry.size, piece) });
        try linkIntoSnapshot(a, io, snapshot_dir, b.entry.path, b.blob);
    }
    return true;
}

fn matches(a: Allocator, io: std.Io, b: Big) !bool {
    return std.mem.eql(u8, &(try ranged.digest(a, io, b.file.path, b.file.size)), &b.entry.sha256.?);
}

/// Reads the proxy and timeout settings; false (after saying why, naming only the variable) when one is unusable.
pub fn routes(a: Allocator, io: std.Io, err_out: *std.Io.Writer, env: ?*const std.process.Environ.Map) !bool {
    route.reset(io);
    route.configure(a, env) catch |e| {
        try err_out.writeAll("tensorfold pull: ");
        try route.refusal(err_out, e);
        try err_out.writeAll("\n");
        return false;
    };
    if (route.settings.skipped.len > 0) try err_out.print("tensorfold pull: {s} names a socks or https:// proxy, which pull cannot use; connecting directly\n", .{route.settings.skipped});
    return true;
}

/// One file's failure: a stall names its timeout, its resumes and the proxy in use; others what the connect met.
pub fn failed(io: std.Io, err_out: *std.Io.Writer, path: []const u8, err: anyerror) !void {
    if (err == error.Stalled) {
        try err_out.print("  {s}: no bytes arrived for {d} s ({s}), and {d} range resumes stalled too; ", .{ path, route.settings.stall_s, route.stall_name, single.max_resumes });
        try route.proxyInUse(err_out);
        try err_out.writeAll("; pull again to resume\n");
        return;
    }
    try err_out.print("  {s}: the download failed: ", .{path});
    try route.explain(io, err_out, err);
    try err_out.writeAll("\n");
}

pub fn envValue(env: ?*const std.process.Environ.Map, key: []const u8) ?[]const u8 {
    return if (env) |m| m.get(key) else null;
}

/// The commit the hub serves for ``repo@revision``.
pub fn resolveRevision(a: Allocator, client: *std.http.Client, out: *std.Io.Writer, auth: Auth, repo: []const u8, revision: []const u8) ![]const u8 {
    if (!revisionOk(revision)) return error.BadRevision;
    const url = try std.fmt.allocPrint(a, "{s}/api/models/{s}/revision/{s}", .{ auth.endpoint, repo, revision });
    const body = try getJson(a, client, auth, url, revision_json_bytes);
    try out.print("[tensorfold] downloading {s}@{s} from Hugging Face\n", .{ repo, revision });
    var parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch return error.BadHubJson;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadHubJson;
    const sha = parsed.value.object.get("sha") orelse return error.BadHubJson;
    if (sha != .string or sha.string.len == 0) return error.BadHubJson;
    // The sha names the snapshot dir and is written to refs/main: only a 40-hex git sha is a path.
    if (!hub.hexSha(sha.string)) return error.BadHubJson;
    return try a.dupe(u8, sha.string);
}

/// The revision's files: path, size and the sha256 the hub states for LFS objects.
pub fn listTree(a: Allocator, client: *std.http.Client, auth: Auth, repo: []const u8, sha: []const u8) ![]Entry {
    const url = try std.fmt.allocPrint(a, "{s}/api/models/{s}/tree/{s}?recursive=true", .{ auth.endpoint, repo, sha });
    const body = try getJson(a, client, auth, url, tree_json_bytes);
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

/// config.json's model_type, copied from the raw bytes' parse, and whether it is a DFlash2 drafter (`serve --drafter`).
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
fn fetchOne(a: Allocator, client: *std.http.Client, auth: Auth, repo: []const u8, sha: []const u8, path: []const u8) ![]u8 {
    return getJson(a, client, auth, try std.fmt.allocPrint(a, "{s}/{s}/resolve/{s}/{s}", .{ auth.endpoint, repo, sha, path }), config_json_bytes);
}

/// One bounded hub answer, with its decompressed bytes capped independently of the transfer encoding.
fn getJson(a: Allocator, client: *std.http.Client, auth: Auth, url: []const u8, limit: usize) ![]u8 {
    const uri = std.Uri.parse(url) catch return error.HubUnreachable;
    var req: std.http.Client.Request = undefined;
    var redirect_buf: [8 << 10]u8 = undefined;
    var response = auth.send(client, &req, .GET, uri, .{}, &redirect_buf) catch |e| return if (e == error.HubDenied) e else error.HubUnreachable;
    defer req.deinit();
    if (response.head.status != .ok) return error.HubStatus;
    const window: []u8 = if (response.head.content_encoding == .identity) &.{} else try a.alloc(u8, std.compress.flate.max_window_len);
    var transfer: [64]u8 = undefined;
    var decompress: std.http.Decompress = undefined;
    const reader = response.readerDecompressing(&transfer, &decompress, window);
    return readCapped(a, reader, limit);
}

fn readCapped(a: Allocator, reader: anytype, limit: usize) ![]u8 {
    var body: std.ArrayList(u8) = .empty;
    errdefer body.deinit(a);
    var chunk: [16 << 10]u8 = undefined;
    while (true) {
        const n = reader.readSliceShort(&chunk) catch return error.HubUnreachable;
        if (n == 0) return body.toOwnedSlice(a);
        if (n > limit -| body.items.len) return error.HubTooLarge;
        try body.appendSlice(a, chunk[0..n]);
    }
}

test "an LFS oid names its sha256 with or without the sha256: prefix" {
    const want: [32]u8 = @splat(0xa5);
    const hex = std.fmt.bytesToHex(want, .lower);
    try std.testing.expectEqualSlices(u8, &want, &lfsDigest(&hex).?);
    try std.testing.expectEqualSlices(u8, &want, &lfsDigest("sha256:" ++ hex).?);
    try std.testing.expect(lfsDigest("sha1:abc") == null and lfsDigest(hex[1..]) == null);
}

test "a revision is one URL path segment, and only a 40-hex sha names a snapshot" {
    for ([_][]const u8{ "main", "v1.0", "refs-pr-1", "my_branch", "0123456789abcdef0123456789abcdef01234567" }) |r| try std.testing.expect(revisionOk(r));
    for ([_][]const u8{ "", "..", "../main", "a/b", "main?x=1", "a%2Fb", "x..y" }) |r| try std.testing.expect(!revisionOk(r));
    try std.testing.expect(hub.hexSha("0123456789abcdefABCDEF0123456789abcdef01"));
    for ([_][]const u8{ "", "main", "0123456789abcdef0123456789abcdef0123456", "0123456789abcdef0123456789abcdef012345678", "../456789abcdef0123456789abcdef0123456789" }) |s| try std.testing.expect(!hub.hexSha(s));
}

test "security: identity bodies fail only above each Hub metadata cap" {
    for ([_]usize{ 1 << 20, 16 << 20 }) |limit| {
        const body = try std.testing.allocator.alloc(u8, limit + 1);
        defer std.testing.allocator.free(body);
        @memset(body, 'A');
        for ([_]usize{ limit - 1, limit }) |length| {
            var reader: std.Io.Reader = .fixed(body[0..length]);
            const got = try readCapped(std.testing.allocator, &reader, limit);
            defer std.testing.allocator.free(got);
            try std.testing.expectEqualSlices(u8, body[0..length], got);
        }
        var reader: std.Io.Reader = .fixed(body);
        try std.testing.expectError(error.HubTooLarge, readCapped(std.testing.allocator, &reader, limit));
    }
}
