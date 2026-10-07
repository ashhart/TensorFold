//! The `pull` command: a Hugging Face checkpoint into the same cache 0.6.6 uses, resumable and verified.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");

pub const GiB: u64 = 1 << 30;

/// One file the hub's tree lists for the revision.
const Entry = struct { path: []const u8, size: u64, sha256: ?[32]u8 = null };

/// Downloads ``repo[@revision]`` into the cache and prints where it landed. Returns the exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, err_out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, spec: []const u8) !u8 {
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
        const got = download(a, io, &client, out, url, blobs_dir, e) catch |err| {
            try err_out.print("  {s}: the download failed ({t})\n", .{ e.path, err });
            return 1;
        };
        try linkIntoSnapshot(a, io, snapshot_dir, e.path, got);
    }
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ refs_dir, "main" }), .data = sha });
    try out.print("stored {s}@{s} at {s}\n", .{ repo, sha, snapshot_dir });
    return 0;
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
                    if (oid == .string and std.mem.startsWith(u8, oid.string, "sha256:")) {
                        sha256 = parseHex32(oid.string[7..]) orelse null;
                    }
                }
            }
        }
        try entries.append(a, .{ .path = try a.dupe(u8, path_v.string), .size = size, .sha256 = sha256 });
    }
    if (entries.items.len == 0) return error.NoConfig;
    return entries.items;
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

/// The blob's final cache name: the LFS sha256 when the hub states one, else the sha256 of the bytes.
fn blobName(a: Allocator, e: Entry) ![]const u8 {
    if (e.sha256) |digest| return try std.fmt.allocPrint(a, "{s}", .{std.fmt.bytesToHex(digest, .lower)});
    const safe = try a.dupe(u8, e.path);
    for (safe) |*ch| if (ch.* == '/') {
        ch.* = '_';
    };
    return safe;
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

/// Downloads one file into ``blobs`` with HTTP range resume, verifies size and sha256, and returns the blob name.
fn download(a: Allocator, io: std.Io, client: *std.http.Client, out: *std.Io.Writer, url: []const u8, blobs_dir: []const u8, e: Entry) ![]const u8 {
    const final_name = try blobName(a, e);
    const final_path = try std.fs.path.join(a, &.{ blobs_dir, final_name });
    const partial_path = try std.fmt.allocPrint(a, "{s}.incomplete", .{final_path});
    const w = std.Io.Dir.cwd();
    var resume_from: u64 = 0;
    if (e.sha256 != null) {
        if (fileStat(io, partial_path)) |size| resume_from = size else |_| {}
    }
    if (resume_from > e.size) resume_from = 0;

    var headers: [1]std.http.Header = undefined;
    var range_buf: [32]u8 = undefined;
    var extra: []const std.http.Header = &.{};
    if (resume_from > 0) {
        const range = try std.fmt.bufPrint(&range_buf, "bytes={d}-", .{resume_from});
        headers[0] = .{ .name = "Range", .value = range };
        extra = &headers;
    }
    const uri = std.Uri.parse(url) catch return error.BadUrl;
    var req = try client.request(.GET, uri, .{ .extra_headers = extra });
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
    if (resume_from > 0) {
        // The prefix already on disk feeds the same digest so the final check spans the whole file.
        const prefix = try a.alloc(u8, @intCast(resume_from));
        const read = file.readPositionalAll(io, prefix, 0) catch 0;
        hash.update(prefix[0..read]);
    }
    var reader_buf: [64 << 10]u8 = undefined;
    var r = response.reader(&reader_buf);
    var chunk: [32 << 10]u8 = undefined;
    var offset = resume_from;
    while (true) {
        const n = r.readSliceShort(&chunk) catch return error.ReadFailed;
        if (n == 0) break;
        hash.update(chunk[0..n]);
        try file.writePositionalAll(io, chunk[0..n], offset);
        offset += n;
    }
    const total = offset;
    if (total != e.size) return error.SizeMismatch;
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    if (e.sha256) |want| {
        if (!std.mem.eql(u8, &digest, &want)) return error.ShaMismatch;
    }
    try file.setLength(io, total);
    file.sync(io) catch {};
    try std.Io.Dir.renameAbsolute(partial_path, final_path, io);
    try out.print("  {s}: {d:.2} MiB\n", .{ e.path, @as(f64, @floatFromInt(total)) / (1 << 20) });
    return final_name;
}

fn fileStat(io: std.Io, path: []const u8) !u64 {
    const st = try std.Io.Dir.cwd().statFile(io, path, .{});
    return st.size;
}

/// The snapshot's path points at the blob, the way huggingface_hub's cache links them.
fn linkIntoSnapshot(a: Allocator, io: std.Io, snapshot_dir: []const u8, path: []const u8, blob_name: []const u8) !void {
    const link_path = try std.fs.path.join(a, &.{ snapshot_dir, path });
    if (std.fs.path.dirname(link_path)) |parent| try std.Io.Dir.cwd().createDirPath(io, parent);
    std.Io.Dir.cwd().deleteFile(io, link_path) catch {};
    const target = try std.fs.path.join(a, &.{ "..", "..", "blobs", blob_name });
    std.Io.Dir.cwd().symLink(io, target, link_path, .{}) catch return error.SymLinkFailed;
}

/// A fake hub on 127.0.0.1: the revision and tree APIs, and resolve endpoints with range support.
const FakeHub = struct {
    io: std.Io,
    server_fd: std.posix.socket_t,
    stop: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    weight_requests: std.atomic.Value(u32) = .init(0),
    range_starts: std.atomic.Value(u64) = .init(0),
    weights: []const u8,
    config: []const u8,
    weights_sha_hex: []const u8,
    port: u16,
    current_fd: std.posix.socket_t = -1,
    log: [16][]const u8 = @splat(""),
    log_len: usize = 0,
    log_a: Allocator = undefined,

    const weights_path = "/Org/Flash/resolve/rev1sha/weights.safetensors";
    const config_path = "/Org/Flash/resolve/rev1sha/config.json";

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
        if (posix.errno(posix.system.listen(fd, 8)) != .SUCCESS) return error.ListenFailed;
        var bound: posix.sockaddr.in = undefined;
        var len: posix.socklen_t = @sizeOf(posix.sockaddr.in);
        if (posix.errno(posix.system.getsockname(fd, @ptrCast(&bound), &len)) != .SUCCESS) return error.BindFailed;
        const port = std.mem.bigToNative(u16, bound.port);
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(weights_body);
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        return .{
            .io = std.testing.io,
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
        while (!fake.stop.load(.acquire)) {
            var fds = [_]std.posix.pollfd{.{ .fd = fake.server_fd, .events = std.posix.POLL.IN, .revents = 0 }};
            const ready = std.posix.poll(&fds, 50) catch return;
            if (ready == 0) continue;
            const rc = std.posix.system.accept(fake.server_fd, null, null);
            if (std.posix.errno(rc) != .SUCCESS) return;
            const conn_fd: std.posix.socket_t = @intCast(rc);
            fake.current_fd = conn_fd;
            fake.handle(conn_fd);
            _ = std.posix.system.close(conn_fd);
        }
    }

    fn handle(fake: *FakeHub, fd: std.posix.socket_t) void {
        var buf: [8192]u8 = undefined;
        var end: usize = 0;
        while (true) {
            const head_end = std.mem.indexOf(u8, buf[0..end], "\r\n\r\n") orelse {
                if (end == buf.len) return;
                const n = std.posix.read(fd, buf[end..]) catch return;
                if (n == 0) return;
                end += n;
                continue;
            };
            fake.respondOne(buf[0..head_end]);
            const rest = end - (head_end + 4);
            std.mem.copyForwards(u8, buf[0..rest], buf[head_end + 4 .. end]);
            end = rest;
        }
    }

    fn respondOne(fake: *FakeHub, head: []const u8) void {
        const fd = fake.current_fd;
        if (fake.log_len < fake.log.len) {
            const line_end_i = std.mem.indexOfScalar(u8, head, '\r') orelse head.len;
            fake.log[fake.log_len] = fake.log_a.dupe(u8, head[0..line_end_i]) catch head[0..line_end_i];
            fake.log_len += 1;
        }
        const line_end = std.mem.indexOf(u8, head, "\r\n") orelse return;
        var parts = std.mem.tokenizeScalar(u8, head[0..line_end], ' ');
        const method = parts.next() orelse return;
        if (!std.mem.eql(u8, method, "GET")) return;
        const raw_path = parts.next() orelse return;
        const path = if (std.mem.indexOfScalar(u8, raw_path, '?')) |q| raw_path[0..q] else raw_path;
        var range_start: ?u64 = null;
        var it = std.mem.splitSequence(u8, head[line_end + 2 ..], "\r\n");
        while (it.next()) |h| {
            const colon = std.mem.indexOfScalar(u8, h, ':') orelse continue;
            if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, h[0..colon], " "), "range")) {
                const value = std.mem.trim(u8, h[colon + 1 ..], " ");
                if (std.mem.startsWith(u8, value, "bytes=")) {
                    const spec = std.mem.trimStart(u8, value[6..], " ");
                    const dash = std.mem.indexOfScalar(u8, spec, '-') orelse spec.len;
                    range_start = std.fmt.parseInt(u64, spec[0..dash], 10) catch null;
                }
            }
        }
        if (std.mem.eql(u8, path, "/api/models/Org/Flash/revision/main")) {
            respond(fd, "{\"sha\": \"rev1sha\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Draft/revision/main")) {
            respond(fd, "{\"sha\": \"draftsha\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Draft/tree/draftsha")) {
            respond(fd, "[{\"type\": \"file\", \"path\": \"config.json\", \"size\": 27}]");
        } else if (std.mem.eql(u8, path, "/Org/Draft/resolve/draftsha/config.json")) {
            respond(fd, "{\"model_type\": \"gemma4\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Flash/tree/rev1sha")) {
            const tree = std.fmt.allocPrint(std.testing.allocator, "[{{\"type\": \"file\", \"path\": \"config.json\", \"size\": {d}}}, {{\"type\": \"file\", \"path\": \"weights.safetensors\", \"size\": {d}, \"lfs\": {{\"oid\": \"sha256:{s}\", \"size\": {d}}}}}]", .{ fake.config.len, fake.weights.len, fake.weights_sha_hex, fake.weights.len }) catch return;
            defer std.testing.allocator.free(tree);
            respond(fd, tree);
        } else if (std.mem.eql(u8, path, config_path)) {
            respond(fd, fake.config);
        } else if (std.mem.eql(u8, path, weights_path)) {
            _ = fake.weight_requests.fetchAdd(1, .monotonic);
            if (range_start) |from| {
                fake.range_starts.store(from, .monotonic);
                if (from >= fake.weights.len) {
                    respond(fd, "");
                    return;
                }
                const body = fake.weights[from..];
                var head_buf: [256]u8 = undefined;
                const head_text = std.fmt.bufPrint(&head_buf, "HTTP/1.1 206 Partial Content\r\nContent-Type: application/octet-stream\r\nContent-Length: {d}\r\nContent-Range: bytes {d}-{d}/{d}\r\nConnection: keep-alive\r\n\r\n", .{ body.len, from, fake.weights.len - 1, fake.weights.len }) catch return;
                sendAll(fd, head_text);
                sendAll(fd, body);
            } else {
                var head_buf: [128]u8 = undefined;
                const head_text = std.fmt.bufPrint(&head_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n", .{fake.weights.len}) catch return;
                sendAll(fd, head_text);
                sendAll(fd, fake.weights);
            }
        } else {
            respond(fd, "{}");
        }
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

test "pull downloads, verifies, resumes and refuses a family Zig cannot serve" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    var fake_hub = FakeHub.open() catch {
        return error.SkipZigTest;
    };
    fake_hub.log_a = a;
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
