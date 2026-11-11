//! Execution witness fixtures: a fake engine with or without a hook, a checkpoint's text files, a socket server.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const server = @import("server.zig");
const auth = @import("auth.zig");
const cli = @import("cli.zig");
const listener = @import("listener.zig");
const witness = @import("witness.zig");
const parts = @import("witness_parts.zig");
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const posix = @import("sys").posix;
const t = std.testing;
const Allocator = std.mem.Allocator;
const Facts = api.execution_witness.Facts;

/// A fake engine: one token a reply; its hook (when `hooked`) attests `facts`.
pub const Engine = struct {
    hooked: bool = true,
    facts: Facts = .{ .known = .{ .path = "test/fake", .digest = @splat(7) } },
    calls: std.atomic.Value(u32) = .init(0),
    /// A witness whose epoch moves inside submit, as a reload between admission and the end would move it.
    bump: ?*witness.State = null,

    pub fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn cast(ptr: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ptr));
    }
    fn info(ptr: *anyopaque) api.Info {
        const e = cast(ptr);
        return .{ .context_window = 4096, .witness = if (e.hooked) .{ .ctx = e, .read = read } else null };
    }
    fn read(ptr: *anyopaque) Facts {
        return cast(ptr).facts;
    }
    fn submit(ptr: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        _ = cast(ptr).calls.fetchAdd(1, .monotonic);
        if (cast(ptr).bump) |s| s.bump();
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{'x'} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// A checkpoint folder with the text files a load reads; `path` is its absolute path.
pub const Checkpoint = struct {
    tmp: t.TmpDir,
    path: [:0]u8,

    pub fn init() !Checkpoint {
        var tmp = t.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3_5\"}" });
        try tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = "{\"model\":{\"vocab\":{\"a\":0}}}" });
        try tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = "{{ messages }}" });
        return .{ .tmp = tmp, .path = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator) };
    }
    pub fn write(c: *Checkpoint, name: []const u8, data: []const u8) !void {
        try c.tmp.dir.writeFile(t.io, .{ .sub_path = name, .data = data });
    }
    /// A rewrite one second later, so a coarse-tick filesystem cannot give it the earlier stamp.
    pub fn rewrite(c: *Checkpoint, name: []const u8, data: []const u8) !void {
        const before = (try c.tmp.dir.statFile(t.io, name, .{})).mtime;
        try c.write(name, data);
        const f = try c.tmp.dir.openFile(t.io, name, .{ .mode = .read_write });
        defer f.close(t.io);
        try f.setTimestamps(t.io, .{ .modify_timestamp = .{ .new = before.addDuration(.fromSeconds(1)) } });
    }
    /// A small stand-in for the running binary, so a test's hash does not wait on the size of the test executable.
    pub fn binary(c: *Checkpoint) !parts.Binary {
        try c.write("tensorfold-native", "a server build");
        const path = try std.fs.path.join(t.allocator, &.{ c.path, "tensorfold-native" });
        defer t.allocator.free(path);
        return parts.Binary.openPath(t.io, path);
    }
    pub fn deinit(c: *Checkpoint) void {
        t.allocator.free(c.path);
        c.tmp.cleanup();
    }
};

/// Waits for the hashing thread's one publication, then reads.
pub fn settled(state: *witness.State, srv: *server.Server) witness.Snapshot {
    var waited: usize = 0;
    while (waited < 10000) : (waited += 1) {
        state.mutex.lockUncancelable(t.io);
        const done = state.hashed != .pending;
        state.mutex.unlock(t.io);
        if (done) break;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    return state.read(srv);
}

/// A server over real sockets; `on` attaches a witness over `checkpoint`'s text, not yet started.
pub const Harness = struct {
    srv: *server.Server = undefined,
    listener: listener.Listener = undefined,
    accept: std.Thread = undefined,
    stop: std.atomic.Value(bool) = .init(false),
    text: ?*parts.Text = null,
    binary: ?parts.Binary = null,
    state: ?*witness.State = null,

    pub fn start(h: *Harness, engine: *Engine, keys: ?*auth.Store, checkpoint: ?*Checkpoint, args: cli.Args) !void {
        h.srv = try server.Server.init(t.allocator, t.io, engine.engine(), ByteText.model(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .keep_warm_s = 0 }, keys);
        errdefer h.srv.deinit();
        if (checkpoint) |c| {
            h.text = try parts.Text.stamp(t.allocator, t.io, c.path, null);
            h.binary = try c.binary();
            h.state = try witness.State.init(t.allocator, t.io, h.srv, args, h.binary, h.text);
            h.srv.witness = h.state;
        }
        h.listener = try listener.Listener.open(.{ .ip4 = .loopback(0) });
        errdefer h.listener.close();
        h.accept = try std.Thread.spawn(.{}, server.Server.serve, .{ h.srv, h.listener, &h.stop });
    }
    pub fn deinit(h: *Harness) void {
        h.stop.store(true, .release);
        h.accept.join();
        var waited: usize = 0;
        while (h.srv.open_connections.load(.acquire) > 0) : (waited += 1) {
            if (waited == 5000) @panic("witness fixture did not quiesce");
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        h.listener.close();
        if (h.state) |s| s.deinit();
        if (h.text) |x| x.deinit();
        if (h.binary) |b| b.close(t.io);
        h.srv.deinit();
    }
    pub fn ask(h: *Harness, a: Allocator, method: []const u8, path: []const u8, key: []const u8, host: []const u8, data: []const u8) ![]const u8 {
        const header = if (key.len > 0) try std.fmt.allocPrint(a, "Authorization: Bearer {s}\r\n", .{key}) else "";
        const raw = try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: {s}\r\n{s}Content-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ method, path, host, header, data.len, data });
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.listener.port()) };
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&address, &storage);
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.Socket;
        const fd: posix.socket_t = @intCast(rc);
        defer _ = posix.system.close(fd);
        if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) return error.Connect;
        var sent: usize = 0;
        while (sent < raw.len) {
            const n = posix.system.write(fd, raw[sent..].ptr, raw.len - sent);
            if (posix.errno(n) != .SUCCESS or n == 0) return error.Write;
            sent += @intCast(n);
        }
        var out: std.ArrayList(u8) = .empty;
        var buffer: [4096]u8 = undefined;
        while (true) {
            var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
            if (try posix.poll(&fds, 5000) == 0) return error.Timeout;
            const n = posix.system.read(fd, &buffer, buffer.len);
            if (posix.errno(n) != .SUCCESS) return error.Read;
            if (n == 0) return out.items;
            try out.appendSlice(a, buffer[0..@intCast(n)]);
        }
    }
};

pub fn expectStatus(reply: []const u8, code: u16) !void {
    var prefix: [32]u8 = undefined;
    if (!std.mem.startsWith(u8, reply, try std.fmt.bufPrint(&prefix, "HTTP/1.1 {d} ", .{code}))) {
        std.debug.print("expected {d}, got: {s}\n", .{ code, reply[0..@min(reply.len, 300)] });
        return error.UnexpectedStatus;
    }
}

pub fn body(reply: []const u8) ![]const u8 {
    const split = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.NoBody;
    return reply[split + 4 ..];
}

pub fn payload(a: Allocator, reply: []const u8) !json.Value {
    return switch (try json.parse(a, try body(reply))) {
        .ok => |v| v,
        .err => error.InvalidJson,
    };
}
