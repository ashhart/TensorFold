//! `pull` against a fake hub on 127.0.0.1: downloads, retries, resume, shared pieces and the whole-file fallback.
const std = @import("std");
const Allocator = std.mem.Allocator;
const pull = @import("pull.zig");
const ranged = @import("pull_parts.zig");

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
    config_git_hex: [40]u8,
    flaky_git_hex: [40]u8,
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
            .config_git_hex = gitHex(config_body),
            .flaky_git_hex = gitHex(flaky_body),
            .port = port,
        };
    }

    fn gitHex(body: []const u8) [40]u8 {
        var git = std.crypto.hash.Sha1.init(.{});
        var head: [32]u8 = undefined;
        git.update(std.fmt.bufPrint(&head, "blob {d}\x00", .{body.len}) catch unreachable);
        git.update(body);
        return std.fmt.bytesToHex(git.finalResult(), .lower);
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
            const tree = std.fmt.allocPrint(std.testing.allocator, "[{{\"type\": \"file\", \"oid\": \"{s}\", \"path\": \"config.json\", \"size\": {d}}}, {{\"type\": \"file\", \"oid\": \"1111111111111111111111111111111111111111\", \"path\": \"weights.safetensors\", \"size\": {d}, \"lfs\": {{\"oid\": \"{s}\", \"size\": {d}}}}}, {{\"type\": \"file\", \"oid\": \"{s}\", \"path\": \"flaky.json\", \"size\": {d}}}]", .{ &fake.config_git_hex, fake.config.len, fake.weights.len, fake.weights_sha_hex, fake.weights.len, &fake.flaky_git_hex, flaky_body.len }) catch return false;
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
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash"));
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
    // Small files are blobs named by their git blob sha1, as huggingface_hub names them.
    try std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", &fake_hub.config_git_hex }), .{});
    try std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", &fake_hub.flaky_git_hex }), .{});
    try std.testing.expectEqualStrings(FakeHub.flaky_body, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ snapshot, "flaky.json" }), a, .limited(1 << 20)));

    // A second pull skips the verified blob and touches the weights endpoint no more.
    var out2: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out2.writer, &err_out.writer, &env, root, "Org/Flash"));
    try std.testing.expectEqual(@as(u32, 1), fake_hub.weight_requests.load(.monotonic));
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "weights.safetensors: cached") != null);
    try std.testing.expect(std.mem.indexOf(u8, out2.written(), "flaky.json: cached") != null);
    try std.testing.expectEqual(@as(u32, 2), fake_hub.flaky_requests.load(.monotonic));

    // A partial blob resumes from its own length.
    const blob_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/blobs", fake_hub.weights_sha_hex });
    std.Io.Dir.cwd().deleteFile(io, blob_path) catch {};
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}.incomplete", .{blob_path}), .data = FakeHub.weights_body[0..2000] });
    var out3: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.run(a, io, &out3.writer, &err_out.writer, &env, root, "Org/Flash"));
    try std.testing.expectEqual(@as(u64, 2000), fake_hub.range_starts.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 2), fake_hub.weight_requests.load(.monotonic));
    const resumed = try std.Io.Dir.cwd().readFileAlloc(io, weights_link, a, .limited(1 << 20));
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, resumed);

    // A family Zig cannot serve is refused with the 0.6 line, after the config check only.
    var refused_out: std.Io.Writer.Allocating = .init(a);
    var refused_err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 1), try pull.run(a, io, &refused_out.writer, &refused_err.writer, &env, root, "Org/Draft"));
    try std.testing.expect(std.mem.indexOf(u8, refused_err.written(), "model_type gemma4") != null);
    try std.testing.expect(std.mem.indexOf(u8, refused_err.written(), "tensorfold@0.6") != null);

    // A bad repo id is usage.
    var bad_err: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 2), try pull.run(a, io, &out.writer, &bad_err.writer, &env, root, "no-slash"));
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
    const small: pull.Sizes = .{ .threshold = 4096, .piece = 1250 };

    var out: std.Io.Writer.Allocating = .init(a);
    var err_out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.runWith(a, io, &out.writer, &err_out.writer, &env, root, "Org/Flash", small));
    try std.testing.expectEqual(@as(u32, 4), fake_hub.weight_requests.load(.monotonic));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "weights.safetensors: 0.00 MiB in 4 pieces") != null);
    const flash = try std.fs.path.join(a, &.{ root, "models--Org--Flash" });
    try std.testing.expectEqualSlices(u8, FakeHub.weights_body, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ flash, "snapshots/rev1sha/weights.safetensors" }), a, .limited(1 << 20)));
    const blob = try std.fs.path.join(a, &.{ flash, "blobs", fake_hub.weights_sha_hex });
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, try std.fmt.allocPrint(a, "{s}.ranges.part0", .{blob}), .{}));

    // A repo whose weights ignore ranges: the pieces give way to one stream, and no piece file is left behind.
    var out2: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try pull.runWith(a, io, &out2.writer, &err_out.writer, &env, root, "Org/Whole", small));
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
