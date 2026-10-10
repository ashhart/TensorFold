//! Frozen server goldens for `zig build test-golden`, with the ten differences that are still known.
const std = @import("std");
const posix = std.posix;
const Io = std.Io;
const wire = @import("wire.zig");
const Reply = wire.Reply;

const context_tokens = "1024";

const Known = struct { group: []const u8, name: []const u8 };
const known_diffs = [_]Known{
    .{ .group = "errors", .name = "format-json" },
    .{ .group = "errors", .name = "stream-structured" },
    .{ .group = "anthropic", .name = "refuse-format" },
    .{ .group = "cancel", .name = "metrics-after" },
    .{ .group = "metrics-open", .name = "metrics" },
    .{ .group = "metrics-open", .name = "v1-metrics" },
    .{ .group = "keys", .name = "metrics-counted" },
    .{ .group = "tools", .name = "two_tools-stream" },
    .{ .group = "tools", .name = "two_tools-single-stream" },
    .{ .group = "anthropic", .name = "tool-use-stream" },
};

fn isKnown(group: []const u8, name: []const u8) bool {
    for (known_diffs) |k| {
        if (std.mem.eql(u8, k.group, group) and std.mem.eql(u8, k.name, name)) return true;
    }
    return false;
}

const Conn = struct {
    fd: posix.socket_t,
    buf: [8 << 10]u8 = undefined,
    start: usize = 0,
    end: usize = 0,

    fn connect(port: u16) !Conn {
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const fd: posix.socket_t = @intCast(rc);
        errdefer _ = posix.system.close(fd);
        // sin_addr is network byte order: on the wire 127.0.0.1 is bytes 7f 00 00 01, which a little-endian u32 stores as 0x0100007f
        const addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = 1 << 24 | 127 };
        const crc = posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in));
        if (posix.errno(crc) != .SUCCESS) return error.ConnectFailed;
        // a stalled exchange errors in ten seconds instead of hanging the run
        const timeout: posix.timeval = .{ .sec = 10, .usec = 0 };
        posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.RCVTIMEO, std.mem.asBytes(&timeout)) catch {};
        return .{ .fd = fd };
    }

    fn close(self: *Conn) void {
        _ = posix.system.close(self.fd);
    }

    fn send(self: *Conn, bytes: []const u8) !void {
        var at: usize = 0;
        while (at < bytes.len) {
            const n = posix.system.write(self.fd, bytes[at..].ptr, bytes.len - at);
            switch (posix.errno(n)) {
                .SUCCESS => at += @intCast(n),
                else => return error.WriteFailed,
            }
        }
    }

    fn halfClose(self: *Conn) void {
        _ = posix.system.shutdown(self.fd, posix.SHUT.WR);
    }

    fn fill(self: *Conn) !usize {
        return posix.read(self.fd, self.buf[0..]) catch |e| switch (e) {
            error.WouldBlock => return 0,
            else => return e,
        };
    }

    fn buffered(self: *Conn) ?u8 {
        if (self.start < self.end) return self.buf[self.start];
        return null;
    }

    /// One line's bytes without the trailing newline; null at a clean EOF before any byte.
    fn readLine(self: *Conn, a: std.mem.Allocator) !?[]u8 {
        var out: std.ArrayList(u8) = .empty;
        while (true) {
            if (self.buffered()) |b| {
                self.start += 1;
                if (b == '\n') return try out.toOwnedSlice(a);
                try out.append(a, b);
                continue;
            }
            const got = try self.fill();
            if (got == 0) {
                if (out.items.len == 0) return null;
                return try out.toOwnedSlice(a);
            }
            self.start = 0;
            self.end = got;
        }
    }

    fn readExact(self: *Conn, a: std.mem.Allocator, n: usize) ![]u8 {
        var out = try a.alloc(u8, n);
        var at: usize = 0;
        while (at < n) {
            if (self.buffered()) |b| {
                out[at] = b;
                self.start += 1;
                at += 1;
                continue;
            }
            const got = try self.fill();
            if (got == 0) return error.TruncatedReply;
            self.start = 0;
            self.end = got;
        }
        return out;
    }

    fn readToEof(self: *Conn, a: std.mem.Allocator) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        while (true) {
            if (self.buffered()) |b| {
                self.start += 1;
                try out.append(a, b);
                continue;
            }
            const got = try self.fill();
            if (got == 0) return try out.toOwnedSlice(a);
            self.start = 0;
            self.end = got;
        }
    }

    /// The closed probe: an event within 300 ms says the server kept the connection or tore it down; silence says open. poll, because a second setsockopt can fail on a reset socket.
    fn probeClosed(self: *Conn) ![]const u8 {
        var pfd: [1]posix.pollfd = .{.{ .fd = self.fd, .events = posix.POLL.IN, .revents = 0 }};
        const n = posix.poll(&pfd, 300) catch return "open";
        if (n == 0) return "open";
        const got = posix.read(self.fd, &self.buf) catch return "closed";
        if (got == 0) return "closed";
        self.start = 0;
        self.end = got;
        return "data";
    }
};

fn parseReply(a: std.mem.Allocator, conn: *Conn, skip_body: bool) !Reply {
    var r: Reply = .{ .status = "", .headers = .empty, .body = "", .interim = "" };
    if (conn.start >= conn.end) {
        const got = conn.fill() catch return r;
        if (got == 0) return r; // a connection closed with no bytes: an empty reply
        conn.start = 0;
        conn.end = got;
    }
    if (conn.buf[conn.start] != 'H') {
        // an HTTP/0.9 reply: its exact bytes to EOF are the body
        r.body = try conn.readToEof(a);
        return r;
    }
    var first = (try conn.readLine(a)) orelse return r;
    var interim: std.ArrayList(u8) = .empty;
    while (std.mem.startsWith(u8, first, "HTTP/1.1 100")) {
        try interim.appendSlice(a, first);
        try interim.append(a, '\n');
        while (true) {
            const line = (try conn.readLine(a)) orelse return error.TruncatedReply;
            try interim.appendSlice(a, line);
            try interim.append(a, '\n');
            if (std.mem.eql(u8, std.mem.trimEnd(u8, line, "\r"), "")) break;
        }
        first = (try conn.readLine(a)) orelse return error.TruncatedReply;
    }
    r.interim = try interim.toOwnedSlice(a);
    if (!std.mem.startsWith(u8, first, "HTTP/")) {
        const rest = try conn.readToEof(a);
        r.body = try std.mem.concat(a, u8, &.{ first, "\n", rest });
        return r;
    }
    r.status = std.mem.trimEnd(u8, first, "\r");
    while (true) {
        const line = (try conn.readLine(a)) orelse return error.TruncatedReply;
        if (std.mem.eql(u8, std.mem.trimEnd(u8, line, "\r"), "")) break;
        const at = std.mem.indexOfScalar(u8, line, ':') orelse return error.BadHeader;
        try r.headers.append(a, .{
            .name = try a.dupe(u8, std.mem.trim(u8, line[0..at], " ")),
            .value = try a.dupe(u8, std.mem.trim(u8, line[at + 1 ..], " \r")),
        });
    }
    var length: ?usize = null;
    for (r.headers.items) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, "content-length")) length = std.fmt.parseInt(usize, h.value, 10) catch null;
    }
    r.body = if (skip_body) "" else if (length) |n| try conn.readExact(a, n) else try conn.readToEof(a);
    return r;
}

const Verdict = enum { equal, differ, no_golden };

const Outcome = struct { group: []const u8, name: []const u8, verdict: Verdict, why: []const []const u8 = &.{} };

var outcomes: std.ArrayList(Outcome) = .empty;

/// A case name as a file name every OS checks out: Windows' reserved characters, controls and '%' become %XX.
fn fileName(a: std.mem.Allocator, name: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (name) |c| {
        if (c < 0x20 or std.mem.indexOfScalar(u8, "<>:\"|?*\\%", c) != null) {
            try out.appendSlice(a, try std.fmt.allocPrint(a, "%{X:0>2}", .{c}));
        } else try out.append(a, c);
    }
    return out.toOwnedSlice(a);
}

fn runCase(a: std.mem.Allocator, io: Io, golden_dir: []const u8, port: u16, case: std.json.Value) !void {
    const group = case.object.get("group").?.string;
    const name = case.object.get("name").?.string;
    const opts = case.object.get("opts").?.object;
    const half_close = if (opts.get("half_close")) |v| v == .bool and v.bool else false;
    const leave = if (opts.get("leave_after_events")) |v| if (v == .integer) @as(u32, @intCast(v.integer)) else null else null;

    var conn = try Conn.connect(port);
    defer conn.close();
    var replies: std.ArrayList(Reply) = .empty;
    var closed: []const u8 = "";
    var why: std.ArrayList([]const u8) = .empty;
    var failed = false;

    outer: for (case.object.get("raws").?.array.items) |raw| {
        const obj = raw.object;
        var bytes: []const u8 = "";
        if (obj.get("b64")) |b| {
            const dec = std.base64.standard.Decoder;
            const n = try dec.calcSizeForSlice(b.string);
            const buf = try a.alloc(u8, n);
            try dec.decode(buf, b.string);
            bytes = buf;
        } else if (obj.get("tmpl_b64")) |b| {
            const dec = std.base64.standard.Decoder;
            const n = try dec.calcSizeForSlice(b.string);
            var buf = try a.alloc(u8, n);
            try dec.decode(buf, b.string);
            // each ZZIDkZZ marker takes reply k's id, then the declared length follows the marker, not the id
            var out: std.ArrayList(u8) = .empty;
            var at: usize = 0;
            while (at < buf.len) {
                if (at + 7 <= buf.len and std.mem.eql(u8, buf[at..][0..4], "ZZID") and buf[at + 5] == 'Z' and buf[at + 6] == 'Z' and std.ascii.isDigit(buf[at + 4])) {
                    const rid = try wire.replyId(a, &replies.items[buf[at + 4] - '0']);
                    try out.appendSlice(a, rid);
                    at += 7;
                    continue;
                }
                try out.append(a, buf[at]);
                at += 1;
            }
            buf = try out.toOwnedSlice(a);
            // a template with a body declares the marker length, not the id's: rewrite it
            const split = std.mem.indexOf(u8, buf, "\r\n\r\n") orelse return error.NoBody;
            const body_len = buf.len - (split + 4);
            if (std.mem.indexOf(u8, buf[0..split], "Content-Length: ")) |cl_at| {
                const cl_end = std.mem.indexOfPos(u8, buf[0 .. split + 2], cl_at, "\r\n") orelse return error.NoLength;
                buf = try std.fmt.allocPrint(a, "{s}Content-Length: {d}{s}", .{ buf[0..cl_at], body_len, buf[cl_end..] });
            }
            bytes = buf;
        }
        if (bytes.len != 0) try conn.send(bytes);
        if (leave) |n| {
            var seen: std.ArrayList(u8) = .empty;
            while (std.mem.count(u8, seen.items, "data: ") < n) {
                var buf: [4096]u8 = undefined;
                const got = posix.read(conn.fd, &buf) catch break;
                if (got == 0) break;
                try seen.appendSlice(a, buf[0..got]);
            }
            closed = "left";
            break :outer;
        }
        if (half_close) conn.halfClose();
        const parsed = parseReply(a, &conn, std.mem.startsWith(u8, bytes, "HEAD ")) catch |e| {
            try why.append(a, try std.fmt.allocPrint(a, "reply read failed: {s}", .{@errorName(e)}));
            failed = true;
            break :outer;
        };
        replies.append(a, parsed) catch return error.OutOfMemory;
    }
    if (closed.len == 0) closed = conn.probeClosed() catch "open";

    // the golden comparison
    var path_buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}/{s}.json", .{ golden_dir, group, try fileName(a, name) });
    const golden_text = Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28)) catch {
        try outcomes.append(a, .{ .group = group, .name = name, .verdict = .no_golden });
        return;
    };
    const golden = try std.json.parseFromSliceLeaky(std.json.Value, a, golden_text, .{});
    const g = golden.object;

    if (!failed) {
        const g_closed = g.get("closed").?.string;
        if (!std.mem.eql(u8, g_closed, closed)) try why.append(a, try std.fmt.allocPrint(a, "connection after: golden {s}, zig {s}", .{ g_closed, closed }));
        const g_replies = g.get("replies").?.array;
        if (g_replies.items.len != replies.items.len) try why.append(a, try std.fmt.allocPrint(a, "replies: golden {d}, zig {d}", .{ g_replies.items.len, replies.items.len }));
        for (g_replies.items, replies.items, 0..) |gv, zr, i| {
            if (!wire.framed(&zr) and !std.mem.startsWith(u8, zr.status, "HTTP/1.1 501")) {
                try why.append(a, try std.fmt.allocPrint(a, "reply {d}: zig Content-Length does not match its body", .{i}));
            }
            const n = try wire.normalize(a, &zr);
            const go = gv.object;
            if (!std.mem.eql(u8, go.get("status").?.string, n.status)) try why.append(a, try std.fmt.allocPrint(a, "reply {d} status: golden {s}, zig {s}", .{ i, go.get("status").?.string, n.status }));
            if (!std.mem.eql(u8, go.get("body").?.string, n.body)) {
                const gb = go.get("body").?.string;
                var at: usize = 0;
                while (at < gb.len and at < n.body.len and gb[at] == n.body[at]) at += 1;
                try why.append(a, try std.fmt.allocPrint(a, "reply {d} body differs at {d}: golden [{s}] zig [{s}]", .{ i, at, gb[at..@min(gb.len, at + 60)], n.body[at..@min(n.body.len, at + 60)] }));
            }
            if (!std.mem.eql(u8, go.get("interim").?.string, n.interim)) try why.append(a, try std.fmt.allocPrint(a, "reply {d} interim differs", .{i}));
            var head_ok = true;
            const gh = go.get("headers").?.array;
            if (gh.items.len * 0 == 0) {
                var golden_head: std.ArrayList(u8) = .empty;
                for (gh.items) |pair| {
                    try golden_head.appendSlice(a, pair.array.items[0].string);
                    try golden_head.appendSlice(a, ": ");
                    try golden_head.appendSlice(a, pair.array.items[1].string);
                    try golden_head.append(a, '\n');
                }
                head_ok = std.mem.eql(u8, golden_head.items, n.headers);
            }
            if (!head_ok) try why.append(a, try std.fmt.allocPrint(a, "reply {d} headers differ", .{i}));
        }
    }

    const verdict: Verdict = if (why.items.len == 0) .equal else .differ;
    try outcomes.append(a, .{ .group = group, .name = name, .verdict = verdict, .why = try why.toOwnedSlice(a) });
}

const Server = struct {
    child: *std.process.Child,
    port: u16,
    log_path: ?[]const u8 = null,
    key_path: ?[]const u8 = null,
};

fn spawnServer(a: std.mem.Allocator, io: Io, binary: []const u8, fixtures: []const u8, args: []const []const u8, env: std.json.ObjectMap, tmp_dir: []const u8, key_content: ?[]const u8) !Server {
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(a, &.{ binary, "serve", fixtures, "--name", "fake-model", "--alias", "fake-alias", "--port", "0", "--max-tokens", "64", "--parallel", "4" });
    var log_path: ?[]const u8 = null;
    var key_path: ?[]const u8 = null;
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "@KEYFILE@")) {
            if (key_path == null) key_path = try std.fmt.allocPrint(a, "{s}/keys", .{tmp_dir});
            try argv.append(a, key_path.?);
        } else try argv.append(a, arg);
    }
    var map = std.process.Environ.Map.init(a);
    try map.put("TENSORFOLD_NO_LIVE", "1");
    try map.put("TF_FAKE_CONTEXT", context_tokens);
    var it = env.iterator();
    while (it.next()) |kv| {
        var value: []const u8 = kv.value_ptr.*.string;
        if (std.mem.eql(u8, kv.key_ptr.*, "TENSORFOLD_REQUEST_LOG")) {
            log_path = try std.fmt.allocPrint(a, "{s}/zig.jsonl", .{tmp_dir});
            value = log_path.?;
        }
        if (std.mem.eql(u8, value, "@KEYFILE@")) {
            if (key_path == null) key_path = try std.fmt.allocPrint(a, "{s}/keys", .{tmp_dir});
            value = key_path.?;
        }
        try map.put(kv.key_ptr.*, value);
    }
    // the server reads its key file at startup: it exists before the spawn
    if (key_path) |kp| {
        if (key_content) |content| {
            var file = try Io.Dir.cwd().createFile(io, kp, .{});
            defer file.close(io);
            var wbuf: [4096]u8 = undefined;
            var fw = file.writerStreaming(io, &wbuf);
            try fw.interface.writeAll(content);
            try fw.interface.flush();
            const kpz = try std.fmt.allocPrintSentinel(a, "{s}", .{kp}, 0);
            _ = std.c.fchmodat(posix.AT.FDCWD, kpz, 0o600, 0);
        }
    }
    const child = try a.create(std.process.Child);
    child.* = try std.process.spawn(io, .{
        .argv = argv.items,
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .inherit,
        .environ_map = &map,
    });
    // the startup lines until the port
    var rbuf: [4096]u8 = undefined;
    var reader = child.stdout.?.reader(io, &rbuf);
    while (true) {
        const line = reader.interface.takeDelimiterExclusive('\n') catch return error.NoPortLine;
        if (std.mem.startsWith(u8, line, "PORT ")) {
            const port = try std.fmt.parseInt(u16, std.mem.trim(u8, line[5..], " \r"), 10);
            return .{ .child = child, .port = port, .log_path = log_path, .key_path = key_path };
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) return error.ExpectedArgs;
    const binary = args[1];
    const cases_path = args[2];
    const golden_dir = args[3];
    const fixtures = args[4];
    const tmp_dir = try std.fmt.allocPrint(init.arena.allocator(), ".zig-tmp-golden-{d}", .{std.c.getpid()});
    Io.Dir.cwd().createDir(io, tmp_dir, .default_dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, tmp_dir) catch {};

    const plan_text = try Io.Dir.cwd().readFileAlloc(io, cases_path, init.arena.allocator(), .limited(1 << 28));
    const plan = try std.json.parseFromSliceLeaky(std.json.Value, init.arena.allocator(), plan_text, .{});

    var server: ?Server = null;
    defer if (server) |*s| if (s.child.id) |pid| {
        posix.kill(pid, posix.SIG.TERM) catch {};
    };

    const steps = plan.object.get("steps").?.array.items;
    for (steps) |step| {
        const obj = step.object;
        if (server) |*s| if (s.child.id) |pid| {
            posix.kill(pid, posix.SIG.TERM) catch {};
        };
        var argv: std.ArrayList([]const u8) = .empty;
        for (obj.get("args").?.array.items) |v| try argv.append(init.arena.allocator(), v.string);
        server = try spawnServer(init.arena.allocator(), io, binary, fixtures, argv.items, obj.get("env").?.object, tmp_dir, if (obj.get("key_file")) |kf| kf.string else null);
        if (obj.get("cases")) |cases| {
            for (cases.array.items) |case| try runCase(init.arena.allocator(), io, golden_dir, server.?.port, case);
        }
        for (obj.get("then").?.array.items) |then| {
            const t = then.object;
            const kind = t.get("kind").?.string;
            const s = &server.?;
            if (std.mem.eql(u8, kind, "cases")) {
                for (t.get("cases").?.array.items) |case| try runCase(init.arena.allocator(), io, golden_dir, s.port, case);
            } else if (std.mem.eql(u8, kind, "sleep")) {
                std.Io.sleep(io, .fromMilliseconds(@intFromFloat(t.get("seconds").?.float * 1000.0)), .awake) catch {};
            } else if (std.mem.eql(u8, kind, "sigterm")) {
                posix.kill(s.child.id.?, posix.SIG.TERM) catch {};
                const term = s.child.wait(io) catch return error.WaitFailed;
                try outcomes.append(init.arena.allocator(), .{
                    .group = "lifecycle",
                    .name = "sigterm-exit-0",
                    .verdict = if (term == .exited and term.exited == 0) .equal else .differ,
                    .why = if (term == .exited and term.exited == 0) &.{} else try init.arena.allocator().dupe([]const u8, &.{try std.fmt.allocPrint(init.arena.allocator(), "fake_serve did not exit 0 on SIGTERM", .{})}),
                });
            } else if (std.mem.eql(u8, kind, "compare_log")) {
                const golden_log = try std.fmt.allocPrint(init.arena.allocator(), "{s}/{s}", .{ golden_dir, t.get("golden").?.string });
                const want = Io.Dir.cwd().readFileAlloc(io, golden_log, init.arena.allocator(), .limited(1 << 28)) catch "";
                const got = Io.Dir.cwd().readFileAlloc(io, s.log_path.?, init.arena.allocator(), .limited(1 << 28)) catch "";
                try outcomes.append(init.arena.allocator(), .{
                    .group = "lifecycle",
                    .name = "request-log",
                    .verdict = if (std.mem.eql(u8, want, got)) .equal else .differ,
                    .why = if (std.mem.eql(u8, want, got)) &.{} else &.{"TENSORFOLD_REQUEST_LOG lines differ from the frozen ones"},
                });
            } else if (std.mem.eql(u8, kind, "rewrite_key")) {
                var file = try Io.Dir.cwd().createFile(io, s.key_path.?, .{});
                defer file.close(io);
                var wbuf: [4096]u8 = undefined;
                var fw = file.writerStreaming(io, &wbuf);
                try fw.interface.writeAll(t.get("content").?.string);
                try fw.interface.flush();
                posix.kill(s.child.id.?, posix.SIG.HUP) catch {}; // the server rereads the key file, as the reference does
            } else if (std.mem.eql(u8, kind, "sighup")) {
                posix.kill(s.child.id.?, posix.SIG.HUP) catch {};
            } else if (std.mem.eql(u8, kind, "terminate")) {
                posix.kill(s.child.id.?, posix.SIG.TERM) catch {};
                _ = s.child.wait(io) catch {};
                server = null;
            } else return error.UnknownStep;
        }
    }

    // the verdicts by group, then the total against the frozen answers
    const GroupCounts = struct { name: []const u8, equal: usize = 0, differ: usize = 0, known: usize = 0 };
    var groups: std.ArrayList(GroupCounts) = .empty;
    defer groups.deinit(gpa);
    var failed: usize = 0;
    for (outcomes.items) |r| {
        var g: ?*GroupCounts = null;
        for (groups.items) |*gc| {
            if (std.mem.eql(u8, gc.name, r.group)) g = gc;
        }
        if (g == null) groups.append(gpa, .{ .name = r.group }) catch return error.OutOfMemory;
        g = &groups.items[groups.items.len - 1];
        switch (r.verdict) {
            .equal => g.?.equal += 1,
            .differ => {
                if (isKnown(r.group, r.name)) g.?.known += 1 else {
                    g.?.differ += 1;
                    failed += 1;
                }
            },
            .no_golden => {
                g.?.differ += 1;
                failed += 1;
            },
        }
        if (r.verdict != .equal) {
            const known = r.verdict == .differ and isKnown(r.group, r.name);
            std.debug.print("{s} {s}/{s}\n", .{ if (known) "KNOWN-DIFFER" else if (r.verdict == .no_golden) "NO-GOLDEN" else "MISMATCH", r.group, r.name });
            for (r.why, 0..) |w, i| {
                if (i == 3) break;
                std.debug.print("  {s}\n", .{w});
            }
        }
    }
    for (groups.items) |gc| {
        std.debug.print("{s:12} {d:4} equal  {d:3} differ  {d:2} known\n", .{ gc.name, gc.equal, gc.differ, gc.known });
    }
    var total: usize = 0;
    for (outcomes.items) |r| {
        if (r.verdict == .equal) total += 1;
    }
    std.debug.print("{d}/{d} equal to the frozen Python answers, {d} unexpected differences\n", .{ total, outcomes.items.len, failed });
    if (failed != 0) return error.UnexpectedDifferences;
}
