//! Fake hub transport shared by checkpoint-download host tests.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// A fake hub on 127.0.0.1: revision and tree APIs, ranged resolve endpoints, a body cut once, a repo ignoring ranges.
pub const FakeHub = struct {
    allocator: Allocator = std.heap.page_allocator, // the fake server owns its worker allocations
    server: std.Io.net.Server,
    io: std.Io,
    threaded: *std.Io.Threaded,
    stop: std.atomic.Value(bool) = .init(false),
    live: std.ArrayList(std.Io.net.Stream) = .empty,
    live_lock: std.Io.Mutex = .init,
    thread: ?std.Thread = null,
    weight_requests: std.atomic.Value(u32) = .init(0),
    flaky_requests: std.atomic.Value(u32) = .init(0),
    range_starts: std.atomic.Value(u64) = .init(0),
    weights: []const u8,
    config: []const u8,
    nested: []const u8,
    weights_sha_hex: []const u8,
    nested_sha_hex: []const u8,
    config_git_hex: [40]u8,
    flaky_git_hex: [40]u8,
    port: u16,
    /// Routes a test puts before the fake's own: the answer's keep-alive, or null when the request is not theirs.
    route: ?*const fn (fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) ?bool = null,
    pub const weights_path = "/Org/Flash/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1/weights.safetensors";
    pub const config_path = "/Org/Flash/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1/config.json";
    const nested_path = "/Org/Flash/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1/tensors/model-00001-of-00001.safetensors";
    pub const flaky_body = "{\"chat_template\": \"{{ messages }}\"}";
    pub const drafter_config = "{\"model_type\": \"qwen3\", \"architectures\": [\"DFlash2DraftModel\"], \"dflash_config\": {\"block_size\": 8}}";
    pub const lilicorr_config = "{\"model_type\": \"qwen3\", \"architectures\": [\"LiLiCorrDraftModel\"], \"dflash_config\": {\"projector_type\": \"lilicorr\", \"lilicorr_enabled\": true}}";
    pub const transcribe_drafter_config = "{\"model_type\": \"nemotron_parse\", \"architectures\": [\"DFlash2DraftModel\"], \"dflash_config\": {\"block_size\": 16}}";
    pub const lilicorr_sha = "fffffffffffffffffffffffffffffffffffffff6";
    pub const transcribe_drafter_sha = "9999999999999999999999999999999999999997";
    const lilicorr_weights_path = "/Org/LiLiCorr/resolve/fffffffffffffffffffffffffffffffffffffff6/weights.safetensors";

    var nested_body_storage: [1234]u8 = @splat(0x5A);
    pub const nested_body: []const u8 = &nested_body_storage;

    pub fn open() !FakeHub {
        const a = std.heap.page_allocator;
        const threaded = try a.create(std.Io.Threaded);
        errdefer a.destroy(threaded);
        threaded.* = std.Io.Threaded.init(a, .{});
        errdefer threaded.deinit();
        const io = threaded.io();
        const loopback: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
        var server = try loopback.listen(io, .{});
        errdefer server.deinit(io);
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(weights_body);
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        var nested_hash = std.crypto.hash.sha2.Sha256.init(.{});
        nested_hash.update(nested_body);
        var nested_digest: [32]u8 = undefined;
        nested_hash.final(&nested_digest);
        const weights_sha_hex = try std.fmt.allocPrint(a, "{s}", .{std.fmt.bytesToHex(digest, .lower)});
        errdefer a.free(weights_sha_hex);
        const nested_sha_hex = try std.fmt.allocPrint(a, "{s}", .{std.fmt.bytesToHex(nested_digest, .lower)});
        errdefer a.free(nested_sha_hex);
        return .{
            .allocator = a,
            .server = server,
            .io = io,
            .threaded = threaded,
            .weights = weights_body,
            .config = config_body,
            .nested = nested_body,
            .weights_sha_hex = weights_sha_hex,
            .nested_sha_hex = nested_sha_hex,
            .config_git_hex = gitHex(config_body),
            .flaky_git_hex = gitHex(flaky_body),
            .port = server.socket.address.getPort(),
        };
    }

    fn gitHex(body: []const u8) [40]u8 {
        var git = std.crypto.hash.Sha1.init(.{});
        var head: [32]u8 = undefined;
        git.update(std.fmt.bufPrint(&head, "blob {d}\x00", .{body.len}) catch unreachable);
        git.update(body);
        return std.fmt.bytesToHex(git.finalResult(), .lower);
    }

    pub const config_body = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}, \"max_position_embeddings\": 262144}";
    var weights_body_storage: [5000]u8 = @splat(0xA5);
    pub const weights_body: []const u8 = &weights_body_storage;

    fn serve(fake: *FakeHub) void {
        var conns: [64]std.Thread = undefined;
        var n: usize = 0;
        defer for (conns[0..n]) |t| t.join();
        while (!fake.stop.load(.acquire)) {
            const conn = fake.server.accept(fake.io) catch return;
            if (fake.stop.load(.acquire)) {
                conn.close(fake.io);
                return;
            }
            if (n == conns.len) {
                conn.close(fake.io);
                continue;
            }
            fake.live_lock.lockUncancelable(fake.io);
            fake.live.append(fake.allocator, conn) catch {};
            fake.live_lock.unlock(fake.io);
            conns[n] = std.Thread.spawn(.{}, handle, .{ fake, conn }) catch {
                fake.forget(conn);
                conn.close(fake.io);
                continue;
            };
            n += 1;
        }
    }

    fn handle(fake: *FakeHub, conn: std.Io.net.Stream) void {
        defer {
            fake.forget(conn);
            conn.close(fake.io);
        }
        var buf: [8192]u8 = undefined;
        var end: usize = 0;
        var read_buf: [4096]u8 = undefined;
        var reader = conn.reader(fake.io, &read_buf);
        while (true) {
            const head_end = std.mem.indexOf(u8, buf[0..end], "\r\n\r\n") orelse {
                if (end == buf.len) return;
                var vecs = [_][]u8{buf[end..]};
                const n = reader.interface.readVec(&vecs) catch return;
                if (n == 0) return;
                end += n;
                continue;
            };
            if (!fake.respondOne(conn, buf[0..head_end])) return;
            const rest = end - (head_end + 4);
            std.mem.copyForwards(u8, buf[0..rest], buf[head_end + 4 .. end]);
            end = rest;
        }
    }

    /// Answers one request; false closes the connection.
    fn respondOne(fake: *FakeHub, conn: std.Io.net.Stream, head: []const u8) bool {
        if (fake.route) |r| if (r(fake, conn, head)) |keep| return keep;
        var tree_buf: [256]u8 = undefined;
        const line_end = std.mem.indexOf(u8, head, "\r\n") orelse return false;
        var parts = std.mem.tokenizeScalar(u8, head[0..line_end], ' ');
        const method = parts.next() orelse return false;
        if (std.mem.eql(u8, method, "HEAD")) {
            // The hub’s link is already the storage link, so no redirect is needed.
            fake.sendAll(conn, "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n");
            return true;
        }
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
            fake.respond(conn, "{\"sha\": \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Whole/revision/main")) {
            fake.respond(conn, "{\"sha\": \"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Draft/revision/main")) {
            fake.respond(conn, "{\"sha\": \"ccccccccccccccccccccccccccccccccccccccc3\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Draft/tree/ccccccccccccccccccccccccccccccccccccccc3")) {
            fake.respond(conn, "[{\"type\": \"file\", \"path\": \"config.json\", \"size\": 27}]");
        } else if (std.mem.eql(u8, path, "/Org/Draft/resolve/ccccccccccccccccccccccccccccccccccccccc3/config.json")) {
            fake.respond(conn, "{\"model_type\": \"gemma3n\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/DFlash/revision/main")) {
            fake.respond(conn, "{\"sha\": \"ddddddddddddddddddddddddddddddddddddddd4\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/DFlash/tree/ddddddddddddddddddddddddddddddddddddddd4")) {
            fake.respond(conn, std.fmt.bufPrint(&tree_buf, "[{{\"type\": \"file\", \"path\": \"config.json\", \"size\": {d}}}]", .{drafter_config.len}) catch return false);
        } else if (std.mem.eql(u8, path, "/Org/DFlash/resolve/ddddddddddddddddddddddddddddddddddddddd4/config.json")) {
            fake.respond(conn, drafter_config);
        } else if (std.mem.eql(u8, path, "/api/models/Org/LiLiCorr/revision/main")) {
            fake.respond(conn, "{\"sha\": \"" ++ lilicorr_sha ++ "\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/LiLiCorr/tree/" ++ lilicorr_sha)) {
            fake.respond(conn, std.fmt.bufPrint(&tree_buf, "[{{\"type\": \"file\", \"path\": \"config.json\", \"size\": {d}}}, {{\"type\": \"file\", \"path\": \"weights.safetensors\", \"size\": {d}, \"lfs\": {{\"oid\": \"{s}\", \"size\": {d}}}}}]", .{ lilicorr_config.len, fake.weights.len, fake.weights_sha_hex, fake.weights.len }) catch return false);
        } else if (std.mem.eql(u8, path, "/Org/LiLiCorr/resolve/" ++ lilicorr_sha ++ "/config.json")) {
            fake.respond(conn, lilicorr_config);
        } else if (std.mem.eql(u8, path, lilicorr_weights_path)) {
            fake.weight_requests.store(fake.weight_requests.load(.monotonic) + 1, .monotonic);
            fake.whole(conn, fake.weights);
        } else if (std.mem.eql(u8, path, "/api/models/Org/TranscribeDraft/revision/main")) {
            fake.respond(conn, "{\"sha\": \"" ++ transcribe_drafter_sha ++ "\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/TranscribeDraft/tree/" ++ transcribe_drafter_sha)) {
            fake.respond(conn, std.fmt.bufPrint(&tree_buf, "[{{\"type\": \"file\", \"path\": \"config.json\", \"size\": {d}}}]", .{transcribe_drafter_config.len}) catch return false);
        } else if (std.mem.eql(u8, path, "/Org/TranscribeDraft/resolve/" ++ transcribe_drafter_sha ++ "/config.json")) {
            fake.respond(conn, transcribe_drafter_config);
        } else if (std.mem.eql(u8, path, "/api/models/Org/Reserved/revision/main")) {
            fake.respond(conn, "{\"sha\": \"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee5\"}");
        } else if (std.mem.eql(u8, path, "/api/models/Org/Reserved/tree/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee5")) {
            // A planted name that belongs to TensorFold, not to a hub tree.
            fake.respond(conn, std.fmt.bufPrint(&tree_buf, "[{{\"type\": \"file\", \"path\": \".tensorfold-save-pending\", \"size\": 3}}, {{\"type\": \"file\", \"path\": \"config.json\", \"size\": {d}}}]", .{config_body.len}) catch return false);
        } else if (std.mem.eql(u8, path, "/Org/Reserved/resolve/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee5/config.json")) {
            fake.respond(conn, config_body);
        } else if (std.mem.eql(u8, path, "/api/models/Org/Flash/tree/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1") or std.mem.eql(u8, path, "/api/models/Org/Whole/tree/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2")) {
            // The hub's tree sends LFS oids as bare hex.
            const tree = std.fmt.allocPrint(fake.allocator, "[{{\"type\": \"file\", \"oid\": \"{s}\", \"path\": \"config.json\", \"size\": {d}}}, {{\"type\": \"file\", \"oid\": \"1111111111111111111111111111111111111111\", \"path\": \"weights.safetensors\", \"size\": {d}, \"lfs\": {{\"oid\": \"{s}\", \"size\": {d}}}}}, {{\"type\": \"file\", \"oid\": \"{s}\", \"path\": \"flaky.json\", \"size\": {d}}}, {{\"type\": \"file\", \"oid\": \"3333333333333333333333333333333333333333\", \"path\": \"tensors/model-00001-of-00001.safetensors\", \"size\": {d}, \"lfs\": {{\"oid\": \"{s}\", \"size\": {d}}}}}]", .{ &fake.config_git_hex, fake.config.len, fake.weights.len, fake.weights_sha_hex, fake.weights.len, &fake.flaky_git_hex, flaky_body.len, fake.nested.len, fake.nested_sha_hex, fake.nested.len }) catch return false;
            defer fake.allocator.free(tree);
            fake.respond(conn, tree);
        } else if (std.mem.eql(u8, path, config_path) or std.mem.eql(u8, path, "/Org/Whole/resolve/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2/config.json")) {
            fake.respond(conn, fake.config);
        } else if (std.mem.endsWith(u8, path, "/flaky.json")) {
            if (fake.flaky_requests.fetchAdd(1, .monotonic) > 0) {
                fake.respond(conn, flaky_body);
                return true;
            }
            var head_buf: [128]u8 = undefined; // the first answer stops halfway and drops the connection
            fake.sendAll(conn, std.fmt.bufPrint(&head_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n", .{flaky_body.len}) catch return false);
            fake.sendAll(conn, flaky_body[0 .. flaky_body.len / 2]);
            return false;
        } else if (std.mem.eql(u8, path, "/Org/Whole/resolve/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2/weights.safetensors")) {
            _ = fake.weight_requests.fetchAdd(1, .monotonic);
            fake.whole(conn, fake.weights);
            return false;
        } else if (std.mem.eql(u8, path, nested_path) or std.mem.eql(u8, path, "/Org/Whole/resolve/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2/tensors/model-00001-of-00001.safetensors")) {
            fake.whole(conn, fake.nested);
            return false;
        } else if (std.mem.eql(u8, path, weights_path)) {
            _ = fake.weight_requests.fetchAdd(1, .monotonic);
            const from = range_start orelse {
                fake.whole(conn, fake.weights);
                return true;
            };
            fake.range_starts.store(from, .monotonic);
            if (from >= fake.weights.len) {
                fake.respond(conn, "");
                return true;
            }
            const stop = if (range_end) |last| @min(last + 1, fake.weights.len) else fake.weights.len;
            const body = fake.weights[from..stop];
            var head_buf: [256]u8 = undefined;
            const head_text = std.fmt.bufPrint(&head_buf, "HTTP/1.1 206 Partial Content\r\nContent-Type: application/octet-stream\r\nContent-Length: {d}\r\nContent-Range: bytes {d}-{d}/{d}\r\nConnection: keep-alive\r\n\r\n", .{ body.len, from, stop - 1, fake.weights.len }) catch return false;
            fake.sendAll(conn, head_text);
            fake.sendAll(conn, body);
        } else {
            fake.respond(conn, "{}");
        }
        return true;
    }

    fn whole(fake: *FakeHub, conn: std.Io.net.Stream, body: []const u8) void {
        var head_buf: [128]u8 = undefined;
        const head_text = std.fmt.bufPrint(&head_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n", .{body.len}) catch return;
        fake.sendAll(conn, head_text);
        fake.sendAll(conn, body);
    }

    pub fn respond(fake: *FakeHub, conn: std.Io.net.Stream, body: []const u8) void {
        var head_buf: [128]u8 = undefined;
        const head_text = std.fmt.bufPrint(&head_buf, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n", .{body.len}) catch return;
        fake.sendAll(conn, head_text);
        fake.sendAll(conn, body);
    }

    pub fn sendAll(fake: *FakeHub, conn: std.Io.net.Stream, bytes: []const u8) void {
        var buf: [4096]u8 = undefined;
        var w = conn.writer(fake.io, &buf);
        w.interface.writeAll(bytes) catch return;
        w.interface.flush() catch return;
    }

    pub fn start(fake: *FakeHub) void {
        fake.thread = std.Thread.spawn(.{}, serve, .{fake}) catch null;
    }

    pub fn endpoint(fake: *FakeHub, a: Allocator) ![]const u8 {
        return std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{fake.port});
    }

    fn forget(fake: *FakeHub, conn: std.Io.net.Stream) void {
        fake.live_lock.lockUncancelable(fake.io);
        defer fake.live_lock.unlock(fake.io);
        for (fake.live.items, 0..) |c, i| {
            if (c.socket.handle == conn.socket.handle) {
                _ = fake.live.swapRemove(i);
                return;
            }
        }
    }

    pub fn stopServer(fake: *FakeHub) void {
        fake.stop.store(true, .release);
        fake.live_lock.lockUncancelable(fake.io);
        for (fake.live.items) |conn| conn.shutdown(fake.io, .both) catch {};
        fake.live_lock.unlock(fake.io);
        // one connection of our own unblocks accept, and is closed on arrival
        if (fake.server.socket.address.connect(fake.io, .{ .mode = .stream }) catch null) |wake| wake.close(fake.io);
        if (fake.thread) |t| t.join();
        fake.server.deinit(fake.io);
        fake.live.deinit(fake.allocator);
        fake.threaded.deinit();
        fake.allocator.destroy(fake.threaded);
        fake.allocator.free(fake.weights_sha_hex);
        fake.allocator.free(fake.nested_sha_hex);
    }
};
