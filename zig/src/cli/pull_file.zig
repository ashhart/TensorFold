//! One file from the hub into the blob store: range resume, size and sha256 checks, retries, and its snapshot link.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// One file the hub's tree lists for the revision: LFS files carry their sha256, small files their git blob sha1.
pub const Entry = struct { path: []const u8, size: u64, sha256: ?[32]u8 = null, git_sha1: ?[20]u8 = null };

const attempts = 5;

/// The blob's cache name, as huggingface_hub names it: the LFS sha256, else the git blob sha1, else the path.
pub fn blobName(a: Allocator, e: Entry) ![]const u8 {
    if (e.sha256) |digest| return try std.fmt.allocPrint(a, "{s}", .{std.fmt.bytesToHex(digest, .lower)});
    if (e.git_sha1) |digest| return try std.fmt.allocPrint(a, "{s}", .{std.fmt.bytesToHex(digest, .lower)});
    const safe = try a.dupe(u8, e.path);
    for (safe) |*ch| if (ch.* == '/') {
        ch.* = '_';
    };
    return safe;
}

fn partialPath(a: Allocator, blobs_dir: []const u8, e: Entry) ![]const u8 {
    return std.fmt.allocPrint(a, "{s}.incomplete", .{try std.fs.path.join(a, &.{ blobs_dir, try blobName(a, e) })});
}

/// `download` again after a reset or a short body, waiting longer each time; a sha256 mismatch starts over once.
pub fn downloadRetrying(a: Allocator, io: std.Io, client: *std.http.Client, out: *std.Io.Writer, url: []const u8, blobs_dir: []const u8, e: Entry) ![]const u8 {
    var wait_ms: i64 = 250;
    var restarted = false;
    var attempt: u32 = 1;
    while (true) : (attempt += 1) {
        if (download(a, io, client, out, url, blobs_dir, e)) |name| return name else |err| {
            if (attempt == attempts or (err == error.ShaMismatch and restarted)) return err;
            if (err == error.ShaMismatch) {
                restarted = true;
                std.Io.Dir.cwd().deleteFile(io, try partialPath(a, blobs_dir, e)) catch {};
            }
            std.Io.sleep(io, .fromMilliseconds(wait_ms), .awake) catch {};
            wait_ms *= 2;
        }
    }
}

/// Downloads one file into ``blobs`` with HTTP range resume, verifies size and sha256, and returns the blob name.
fn download(a: Allocator, io: std.Io, client: *std.http.Client, out: *std.Io.Writer, url: []const u8, blobs_dir: []const u8, e: Entry) ![]const u8 {
    const final_name = try blobName(a, e);
    const final_path = try std.fs.path.join(a, &.{ blobs_dir, final_name });
    const partial_path = try partialPath(a, blobs_dir, e);
    const w = std.Io.Dir.cwd();
    var resume_from: u64 = 0;
    if (e.sha256 != null) {
        if (fileStat(io, partial_path)) |size| resume_from = size else |_| {}
    }
    if (resume_from >= e.size) resume_from = 0;

    var headers: [1]std.http.Header = undefined;
    var range_buf: [32]u8 = undefined;
    var extra: []const std.http.Header = &.{};
    if (resume_from > 0) {
        const range = try std.fmt.bufPrint(&range_buf, "bytes={d}-", .{resume_from});
        headers[0] = .{ .name = "Range", .value = range };
        extra = &headers;
    }
    const uri = std.Uri.parse(url) catch return error.BadUrl;
    var req = try client.request(.GET, uri, .{ .extra_headers = extra, .headers = .{ .accept_encoding = .{ .override = "identity" } } }); // the file's own bytes: sizes, digests and ranges count them
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buf: [8 << 10]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);
    const status = response.head.status;
    if (status != .ok and status != .partial_content) return error.HubStatus;
    var restart = false;
    if (resume_from > 0 and status != .partial_content) {
        restart = true; // the hub ignored the range; start over
        resume_from = 0;
    }
    // Read access too: a resumed blob feeds its on-disk prefix into the same digest.
    const file = try w.createFile(io, partial_path, .{ .read = true, .truncate = restart });
    defer file.close(io);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    const check_git = e.sha256 == null and e.git_sha1 != null;
    var git = std.crypto.hash.Sha1.init(.{}); // git's blob id: sha1 of "blob SIZE\0" and the bytes
    var git_head: [32]u8 = undefined;
    if (check_git) git.update(try std.fmt.bufPrint(&git_head, "blob {d}\x00", .{e.size}));
    if (resume_from > 0) {
        // The prefix already on disk feeds the same digest so the final check spans the whole file.
        // It is read in chunks: its length is capped by a hub-named size, never allocated whole.
        var prefix_buf: [64 << 10]u8 = undefined;
        var at: u64 = 0;
        while (at < resume_from) {
            const want: usize = @intCast(@min(prefix_buf.len, resume_from - at));
            const read = file.readPositionalAll(io, prefix_buf[0..want], at) catch 0;
            if (read == 0) break;
            hash.update(prefix_buf[0..read]);
            at += read;
        }
    }
    var reader_buf: [64 << 10]u8 = undefined;
    var r = response.reader(&reader_buf);
    var chunk: [32 << 10]u8 = undefined;
    var offset = resume_from;
    while (true) {
        const n = r.readSliceShort(&chunk) catch return error.ReadFailed;
        if (n == 0) break;
        hash.update(chunk[0..n]);
        if (check_git) git.update(chunk[0..n]);
        try file.writePositionalAll(io, chunk[0..n], offset);
        offset += n;
    }
    const total = offset;
    if (total != e.size) return error.SizeMismatch;
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    if (e.sha256) |want| {
        if (!std.mem.eql(u8, &digest, &want)) return error.ShaMismatch;
    } else if (e.git_sha1) |want| {
        if (!std.mem.eql(u8, &git.finalResult(), &want)) return error.ShaMismatch;
    }
    try file.setLength(io, total);
    file.sync(io) catch {};
    try std.Io.Dir.renameAbsolute(partial_path, final_path, io);
    try out.print("  {s}: {d:.2} MiB\n", .{ e.path, @as(f64, @floatFromInt(total)) / (1 << 20) });
    return final_name;
}

pub fn fileStat(io: std.Io, path: []const u8) !u64 {
    const st = try std.Io.Dir.cwd().statFile(io, path, .{});
    return st.size;
}

/// A snapshot's path a hub tree names: relative, and no traversal segments.
fn snapshotPathOk(path: []const u8) bool {
    if (path.len == 0 or path[0] == '/') return false;
    var it = std.mem.tokenizeScalar(u8, path, '/');
    while (it.next()) |c| {
        if (std.mem.eql(u8, c, ".") or std.mem.eql(u8, c, "..")) return false;
    }
    return true;
}

/// The snapshot's path points at the blob, the way huggingface_hub's cache links them.
pub fn linkIntoSnapshot(a: Allocator, io: std.Io, snapshot_dir: []const u8, path: []const u8, blob_name: []const u8) !void {
    if (!snapshotPathOk(path)) return error.SnapshotPath;
    const link_path = try std.fs.path.join(a, &.{ snapshot_dir, path });
    if (std.fs.path.dirname(link_path)) |parent| try std.Io.Dir.cwd().createDirPath(io, parent);
    std.Io.Dir.cwd().deleteFile(io, link_path) catch {};
    const target = try std.fs.path.join(a, &.{ "..", "..", "blobs", blob_name });
    std.Io.Dir.cwd().symLink(io, target, link_path, .{}) catch return error.SymLinkFailed;
}
