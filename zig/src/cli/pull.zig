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
const Sizes = struct { threshold: u64 = ranged.threshold, piece: u64 = ranged.piece };

/// A large LFS file on the shared queue, named by its sha256 once its bytes check out.
const Big = struct { entry: Entry, blob: []const u8, file: ranged.File };

/// Downloads ``repo[@revision]`` into the cache and prints where it landed. Returns the exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, err_out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, spec: []const u8) !u8 {
    return runWith(a, io, out, err_out, env, override, spec, .{});
}

fn runWith(a: Allocator, io: std.Io, out: *std.Io.Writer, err_out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, spec: []const u8, sizes: Sizes) !u8 {
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
    const model_type = modelTypeOf(a, config_bytes);
    if (hub.family(model_type)) |family| {
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
        if (e.sha256 != null) blob_exists: {
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
    const url = try std.fmt.allocPrint(a, "{s}/api/models/{s}/revision/{s}", .{ endpoint, repo, revision });
    const body = try getJson(a, client, url);
    try out.print("[tensorfold] downloading {s}@{s} from Hugging Face\n", .{ repo, revision });
    var parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch return error.BadHubJson;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadHubJson;
    const sha = parsed.value.object.get("sha") orelse return error.BadHubJson;
    if (sha != .string or sha.string.len == 0) return error.BadHubJson;
    return try a.dupe(u8, sha.string);
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
        try entries.append(a, .{ .path = try a.dupe(u8, path_v.string), .size = size, .sha256 = sha256 });
    }
    if (entries.items.len == 0) return error.NoConfig;
    return entries.items;
}

/// An LFS oid's sha256: the hub's tree sends bare hex, pointer files say "sha256:" first.
fn lfsDigest(oid: []const u8) ?[32]u8 {
    return parseHex32(if (std.mem.startsWith(u8, oid, "sha256:")) oid[7..] else oid);
}

/// config.json's model_type from raw bytes.
fn modelTypeOf(a: Allocator, bytes: []const u8) []const u8 {
    var parsed = std.json.parseFromSlice(std.json.Value, a, bytes, .{}) catch return "unknown";
    defer parsed.deinit();
    if (parsed.value != .object) return "unknown";
    const t = parsed.value.object.get("model_type") orelse return "unknown";
    return if (t == .string) t.string else "unknown";
}

fn parseHex32(text: []const u8) ?[32]u8 {
    if (text.len != 64) return null;
    var out: [32]u8 = undefined;
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

/// A fake hub on 127.0.0.1: revision and tree APIs, ranged resolve endpoints, a body cut once, a repo ignoring ranges.
const FakeHub = struct {
    server_fd: std.posix.socket_t,
    stop: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    weight_requests: std.atomic.Value(u32) = .init(0),
    flaky_requests: std.atomic.Value(u32) = .init(0),
    range_starts: std.atomic.Value(u64) = .init(0),
    weights: []const u8,
    config: []const u8,
    weights_sha_hex: []const u8,
    port: u16,

    const weights_path = "/Org/Flash/resolve/rev1sha/weights.safetensors";
    const config_path = "/Org/Flash/resolve/rev1sha/config.json";
    const flaky_body = "{\"chat_template\": \"{{ messages }}\"}";

    fn open() !FakeHub {
        const posix = std.posix;
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const fd: posix.socket_t = @intCast(rc);
        errdefer _ = posix.system.close(fd);
        const one: c_int = 1;
        try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one));
        var addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, 0), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
        if (posix.errno(posix.system.bind(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.BindFailed;
        if (posix.errno(posix.system.listen(fd, 32)) != .SUCCESS) return error.ListenFailed;
        var bound: posix.sockaddr.in = undefined;
        var len: posix.socklen_t = @sizeOf(posix.sockaddr.in);
        if (posix.errno(posix.system.getsockname(fd, @ptrCast(&bound), &len)) != .SUCCESS) return error.BindFailed;
        const port = std.mem.bigToNative(u16, bound.port);
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(weights_body);
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        return .{
            .server_fd = fd,
            .weights = weights_body,
            .config = config_body,
            .weights_sha_hex = try std.fmt.allocPrint(std.testing.allocator, "{s}", .{std.fmt.bytesToHex(digest, .lower)}),
            .port = port,
        };
    }

    const config_body = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}, \"max_position_embeddings\": 262144}";
    var weights_body_storage: [5000]u8 = @splat(0xA5);
    const weights_body: []const u8 = &weights_body_storage;

    fn serve(fake: *FakeHub) void {
        var conns: [64]std.Thread = undefined;
        var n: usize = 0;
        defer for (conns[0..n]) |t| t.join();
        while (!fake.stop.load(.acquire)) {
            var fds = [_]std.posix.pollfd{.{ .fd = fake.server_fd, .events = std.posix.POLL.IN, .revents = 0 }};
            const ready = std.posix.poll(&fds, 50) catch return;
            if (ready == 0) continue;
            const rc = std.posix.system.accept(fake.server_fd, null, null);
            if (std.posix.errno(rc) != .SUCCESS) return;
            const conn_fd: std.posix.socket_t = @intCast(rc);
            if (n == conns.len) {
                _ = std.posix.system.close(conn_fd);
                continue;
            }
            conns[n] = std.Thread.spawn(.{}, handle, .{ fake, conn_fd }) catch {
                _ = std.posix.system.close(conn_fd);
                continue;
            };
            n += 1;
        }
    }

    fn handle(fake: *FakeHub, fd: std.posix.socket_t) void {
        defer _ = std.posix.system.close(fd);
        var buf: [8192]u8 = undefined;
        var end: usize = 0;
        while (true) {
            const head_end = std.mem.indexOf(u8, buf[0..end], "\r\n\r\n") orelse {
                if (end == buf.len) return;
                var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
                const ready = std.posix.poll(&fds, 50) catch return;
                if (ready == 0) {
                    if (fake.stop.load(.acquire)) return;
                    continue;
                }
                const n = std.posix.read(fd, buf[end..]) catch return;
                if (n == 0) return;
                end += n;
                continue;
            };
            if (!fake.respondOne(fd, buf[0..head_end])) return;
            const rest = end - (head_end + 4);
            std.mem.copyForwards(u8, buf[0..rest], buf[head_end + 4 .. end]);
            end = rest;
        }
    }

    /// Answers one request; false closes the connection.
    fn respondOne(fake: *FakeHub, fd: std.posix.socket_t, head: []const u8) bool {
        const line_end = std.mem.indexOf(u8, head, "\r\n") orelse return false;
        var parts = std.mem.tokenizeScalar(u8, head[0..line_end], ' ');
        const method = parts.next() orelse return false;
        if (!std.mem.eql(u8, method, "GET")) return false;
        const raw_path = parts.next() orelse return false;
        const path = if (std.mem.indexOfScalar(u8, raw_path, '?')) |q| raw_path[0..q] else raw_path;
        var range_start: ?u64 = null;
        var range_end: ?u64 = null;
        var it = std.mem.splitSequence(u8, head[line_end + 2 ..], "\r\n");
        while (it.next()) |h| {
            const colon = std.mem.indexOfScalar(u8, h, ':') orelse continue;
            if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, h[0..colon], " "), "range")) {
                const value = std.mem.trim(u8, h[colon + 1 ..], " ");
                if (std.mem.startsWith(u8, value, "bytes=")) {
                    const spec = std.mem.trimStart(u8, value[6..], " ");
                    const dash = std.mem.indexOfScalar(u8, spec, '-') orelse spec.len;
                    range_start = std.fmt.parseInt(u64, spec[0..dash], 10) catch null;
                    if (dash + 1 < spec.len) range_end = std.fmt.parseInt(u64, spec[dash + 1 ..], 10) catch null;
                }
            }
        }
        if (std.mem.eql(u8, path, "/api/models/Org/Flash/revision/main")) {
            respond(fd, "{\"sha\": \"rev1sha\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Whole/revision/main")) {
            respond(fd, "{\"sha\": \"wholesha\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Draft/revision/main")) {
            respond(fd, "{\"sha\": \"draftsha\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Draft/tree/draftsha")) {
            respond(fd, "[{\"type\": \"file\", \"path\": \"config.json\", \"size\": 27}]");
        } else if (std.mem.eql(u8, path, "/Org/Draft/resolve/draftsha/config.json")) {
            respond(fd, "{\"model_type\": \"gemma4\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Flash/tree/rev1sha") or std.mem.eql(u8, path, "/api/models/Org/Whole/tree/wholesha")) {
            // The hub's tree sends LFS oids as bare hex.
            const tree = std.fmt.allocPrint(std.testing.allocator, "[{{\"type\": \"file\", \"path\": \"config.json\", \"size\": {d}}}, {{\"type\": \"file\", \"path\": \"weights.safetensors\", \"size\": {d}, \"lfs\": {{\"oid\": \"{s}\", \"size\": {d}}}}}, {{\"type\": \"file\", \"path\": \"flaky.json\", \"size\": {d}}}]", .{ fake.config.len, fake.weights.len, fake.weights_sha_hex, fake.weights.len, flaky_body.len }) catch return false;
            defer std.testing.allocator.free(tree);
            respond(fd, tree);
        } else if (std.mem.eql(u8, path, config_path) or std.mem.eql(u8, path, "/Org/Whole/resolve/wholesha/config.json")) {
            respond(fd, fake.config);
        } else if (std.mem.endsWith(u8, path, "/flaky.json")) {
            if (fake.flaky_requests.fetchAdd(1, .monotonic) > 0) {
                respond(fd, flaky_body);
                return true;
            }
            var head_buf: [128]u8 = undefined; // the first answer stops halfway and drops the connection
            sendAll(fd, std.fmt.bufPrint(&head_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n", .{flaky_body.len}) catch return false);
            sendAll(fd, flaky_body[0 .. flaky_body.len / 2]);
            return false;
        } else if (std.mem.eql(u8, path, "/Org/Whole/resolve/wholesha/weights.safetensors")) {
            _ = fake.weight_requests.fetchAdd(1, .monotonic);
            whole(fd, fake.weights);
        } else if (std.mem.eql(u8, path, weights_path)) {
            _ = fake.weight_requests.fetchAdd(1, .monotonic);
            const from = range_start orelse {
                whole(fd, fake.weights);
                return true;
            };
            fake.range_starts.store(from, .monotonic);
            if (from >= fake.weights.len) {
                respond(fd, "");
                return true;
            }
            const stop = if (range_end) |last| @min(last + 1, fake.weights.len) else fake.weights.len;
            const body = fake.weights[from..stop];
            var head_buf: [256]u8 = undefined;
            const head_text = std.fmt.bufPrint(&head_buf, "HTTP/1.1 206 Partial Content\r\nContent-Type: application/octet-stream\r\nContent-Length: {d}\r\nContent-Range: bytes {d}-{d}/{d}\r\nConnection: keep-alive\r\n\r\n", .{ body.len, from, stop - 1, fake.weights.len }) catch return false;
            sendAll(fd, head_text);
            sendAll(fd, body);
        } else {
            respond(fd, "{}");
        }
        return true;
    }

    fn whole(fd: std.posix.socket_t, body: []const u8) void {
        var head_buf: [128]u8 = undefined;
        const head_text = std.fmt.bufPrint(&head_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n", .{body.len}) catch return;
        sendAll(fd, head_text);
        sendAll(fd, body);
    }

    fn respond(fd: std.posix.socket_t, body: []const u8) void {
        var head_buf: [128]u8 = undefined;
        const head_text = std.fmt.bufPrint(&head_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n", .{body.len}) catch return;
        sendAll(fd, head_text);
        sendAll(fd, body);
    }

    fn sendAll(fd: std.posix.socket_t, bytes: []const u8) void {
        var sent: usize = 0;
        while (sent < bytes.len) {
            const n = std.posix.system.write(fd, bytes[sent..].ptr, bytes.len - sent);
            if (std.posix.errno(n) != .SUCCESS) return;
            sent += @intCast(n);
        }
    }

    fn start(fake: *FakeHub) void {
        fake.thread = std.Thread.spawn(.{}, serve, .{fake}) catch null;
    }

    fn endpoint(fake: *FakeHub, a: Allocator) ![]const u8 {
        return std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{fake.port});
    }

    fn stopServer(fake: *FakeHub) void {
        fake.stop.store(true, .release);
        if (fake.thread) |t| t.join();
        _ = std.posix.system.close(fake.server_fd);
        std.testing.allocator.free(fake.weights_sha_hex);
    }
};

fn testEnv(a: Allocator, endpoint_url: []const u8) !std.process.Environ.Map {
    var env: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    try env.put("HF_ENDPOINT", endpoint_url);
    return env;
}

test "an LFS oid names its sha256 with or without the sha256: prefix" {
    const want: [32]u8 = @splat(0xa5);
    const hex = std.fmt.bytesToHex(want, .lower);
    try std.testing.expectEqualSlices(u8, &want, &lfsDigest(&hex).?);
    try std.testing.expectEqualSlices(u8, &want, &lfsDigest("sha256:" ++ hex).?);
    try std.testing.expect(lfsDigest("sha1:abc") == null and lfsDigest(hex[1..]) == null);
}

test "pull downloads, verifies, retries a cut body, resumes and refuses a family Zig cannot serve" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake_hub = FakeHub.open() catch {
        return error.SkipZigTest;
    };
    fake_hub.start();
    defer fake_hub.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try testEnv(a, try fake_hub.endpoint(a));

    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
    const snapshot = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1sha" });
    const config_link = try std.fs.path.join(a, &.{ snapshot, "config.json" });
    const weights_link = try std.fs.path.join(a, &.{ snapshot, "weights.safetensors" });
    const config_read = try std.Io.Dir.cwd().readFileAlloc(io, config_link, a, .limited(1 << 20));
    try std.testing.expectEqualStrings(FakeHub.config_body, config_read);
    const weights_read = try std.Io.Dir.cwd().readFileAlloc(io, weights_link, a, .limited(1 << 20));
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, weights_read);
    const ref = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs/main" }), a, .limited(64));
    try std.testing.expectEqualStrings("rev1sha", std.mem.trim(u8, ref, " \r\n"));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "weights.safetensors") != null);
    try std.testing.expectEqual(@as(u32, 1), fake_hub.weight_requests.load(.monotonic));
    // The blob is named by the hub's bare-hex oid, and the body cut short once came down on the retry.
    try std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", fake_hub.weights_sha_hex }), .{});
    try std.testing.expectEqual(@as(u32, 2), fake_hub.flaky_requests.load(.monotonic));
    try std.testing.expectEqualStrings(FakeHub.flaky_body, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ snapshot, "flaky.json" }), a, .limited(1 << 20)));

    // A second pull skips the verified blob and touches the weights endpoint no more.
    var out2: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &out2.writer, &err_out.writer, &env, root, "Org/Flash"));
    try std.testing.expectEqual(@as(u32, 1), fake_hub.weight_requests.load(.monotonic));
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "weights.safetensors: cached") != null);

    // A partial blob resumes from its own length.
    const blob_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", fake_hub.weights_sha_hex });
    std.Io.Dir.cwd().deleteFile(io, blob_path) catch {};
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}.incomplete", .{blob_path}), .data = FakeHub.weights_body[0..2000] });
    var out3: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &out3.writer, &err_out.writer, &env, root, "Org/Flash"));
    try std.testing.expectEqual(@as(u64, 2000), fake_hub.range_starts.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 2), fake_hub.weight_requests.load(.monotonic));
    const resumed = try std.Io.Dir.cwd().readFileAlloc(io, weights_link, a, .limited(1 << 20));
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, resumed);

    // A family Zig cannot serve is refused with the 0.6 line, after the config check only.
    var refused_out: std.Io.Writer.Allocating = .init(a);
    var refused_err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try run(a, io, &refused_out.writer, &refused_err.writer, &env, root, "Org/Draft"));
    try std.testing.expect(std.mem.indexOf(u8, refused_err.written(), "model_type gemma4") != null);
    try std.testing.expect(std.mem.indexOf(u8, refused_err.written(), "tensorfold@0.6") != null);

    // A bad repo id is usage.
    var bad_err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 2), try run(a, io, &out.writer, &bad_err.writer, &env, root, "no-slash"));
}

test "large files come down as shared pieces, and whole files when the hub ignores ranges" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake_hub = FakeHub.open() catch return error.SkipZigTest;
    fake_hub.start();
    defer fake_hub.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-pieces-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const env = try testEnv(a, try fake_hub.endpoint(a));
    const small: Sizes = .{ .threshold = 4096, .piece = 1250 };

    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try runWith(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash", small));
    try std.testing.expectEqual(@as(u32, 4), fake_hub.weight_requests.load(.monotonic));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "weights.safetensors: 0.00 MiB in 4 pieces") != null);
    const flash = try std.fs.path.join(a, &.{ root, "models--Org--Flash" });
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ flash, "snapshots/rev1sha/weights.safetensors" }), a, .limited(1 << 20)));
    const blob = try std.fs.path.join(a, &.{ flash, "blobs", fake_hub.weights_sha_hex });
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, try std.fmt.allocPrint(a, "{s}.ranges.part0", .{blob}), .{}));

    // A repo whose weights ignore ranges: the pieces give way to one stream, and no piece file is left behind.
    var out2: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try runWith(a, io, &out2.writer, &err_out.writer, &env, root, "Org/Whole", small));
    const whole = try std.fs.path.join(a, &.{ root, "models--Org--Whole" });
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ whole, "snapshots/wholesha/weights.safetensors" }), a, .limited(1 << 20)));
    const whole_blob = try std.fs.path.join(a, &.{ whole, "blobs", fake_hub.weights_sha_hex });
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, try std.fmt.allocPrint(a, "{s}.ranges", .{whole_blob}), .{}));
}

test "shared pieces fill every byte across files, restart only unmarked pieces and refuse a server without ranges" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake_hub = FakeHub.open() catch return error.SkipZigTest;
    fake_hub.start();
    defer fake_hub.stopServer();
    const root = try std.fmt.allocPrint(a, ".tf-pull-ranges-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    try std.Io.Dir.cwd().createDirPath(io, root);
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const url = try std.fmt.allocPrint(a, "{s}{s}", .{ try fake_hub.endpoint(a), FakeHub.weights_path });
    const size = FakeHub.weights_body.len;
    const files = [_]ranged.File{
        .{ .url = url, .path = try std.fs.path.join(a, &.{ root, "first.ranges" }), .size = size },
        .{ .url = url, .path = try std.fs.path.join(a, &.{ root, "second.ranges" }), .size = size },
    };

    try ranged.fetch(a, io, &files, 1250, null);
    try std.testing.expectEqual(@as(u32, 8), fake_hub.weight_requests.load(.monotonic));
    for (files) |f| try std.testing.expectEqualSlices(u8, FakeHub.weights_body, try std.Io.Dir.cwd().readFileAlloc(io, f.path, a, .limited(1 << 20)));

    // Pieces 0 and 2 already marked done: only 1 and 3 are fetched again.
    const first = files[0];
    try std.Io.Dir.cwd().deleteFile(io, first.path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = first.path, .data = FakeHub.weights_body });
    for ([_]u32{ 0, 2 }) |i| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}.part{d}", .{ first.path, i }), .data = "" });
    try ranged.fetch(a, io, &.{first}, 1250, null);
    try std.testing.expectEqual(@as(u32, 10), fake_hub.weight_requests.load(.monotonic));
    try std.testing.expectEqualSlices(u8, &(try ranged.digest(a, io, first.path, size)), &(try hexBytes(fake_hub.weights_sha_hex)));

    // A server that answers a range with the whole file (the config endpoint here) is refused for pieces.
    const config_url = try std.fmt.allocPrint(a, "{s}{s}", .{ try fake_hub.endpoint(a), FakeHub.config_path });
    try std.testing.expectError(error.RangeUnsupported, ranged.fetch(a, io, &.{.{ .url = config_url, .path = try std.fs.path.join(a, &.{ root, "config.ranges" }), .size = FakeHub.config_body.len }}, 50, null));
}

fn hexBytes(text: []const u8) ![32]u8 {
    var out: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&out, text);
    return out;
}
