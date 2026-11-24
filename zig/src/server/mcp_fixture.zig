//! The MCP tests' world: a temporary home and cache, fake engines, sessions fed lines, and their output read back.
const std = @import("std");
const json = @import("json");
const cli = @import("cli.zig");
const mcp = @import("mcp.zig");
const rpc = @import("mcp_rpc.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const version = "0.0.0-test";
pub const sha = "0123456789abcdef0123456789abcdef01234567";
pub const lightning = "{\"model_type\": \"nemotron_h\", \"quantization\": {\"bits\": 4, \"group_size\": 64}, \"max_position_embeddings\": 8192}";

pub fn engines(_: Allocator) cli.Engines {
    return .{ .version = version, .families = &.{.{ .model_type = "nemotron_h", .formats = &.{} }} };
}

/// A temporary home whose Hugging Face cache is ``hub/``, and the environment naming them.
pub const Home = struct {
    tmp: std.testing.TmpDir,
    dir: []const u8,
    env: std.process.Environ.Map,

    pub fn init(a: Allocator) !Home {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir = try a.dupe(u8, buf[0..try tmp.dir.realPath(std.testing.io, &buf)]);
        var env: std.process.Environ.Map = .init(a);
        try env.put("PATH", "/usr/bin:/bin");
        try env.put("HOME", dir);
        try env.put("HF_HUB_CACHE", try std.fs.path.join(a, &.{ dir, "hub" }));
        return .{ .tmp = tmp, .dir = dir, .env = env };
    }

    pub fn deinit(h: *Home) void {
        h.tmp.cleanup();
    }

    pub fn context(h: *const Home) mcp.Context {
        return .{ .io = std.testing.io, .gpa = std.testing.allocator, .environ = &h.env, .home = h.dir, .version = version, .engines = &engines, .drain_s = 10 };
    }

    /// ``repo`` cached at ``sha`` with ``config`` and a small weights file; returns its snapshot directory.
    pub fn checkpoint(h: *const Home, a: Allocator, repo: []const u8, config: []const u8) ![]const u8 {
        const io = std.testing.io;
        const slash = std.mem.indexOfScalar(u8, repo, '/').?;
        const root = try std.fmt.allocPrint(a, "{s}/hub/models--{s}--{s}", .{ h.dir, repo[0..slash], repo[slash + 1 ..] });
        const snapshot = try std.fs.path.join(a, &.{ root, "snapshots", sha });
        const cwd = std.Io.Dir.cwd();
        const weights: [4096]u8 = @splat('x');
        try cwd.createDirPath(io, snapshot);
        try cwd.createDirPath(io, try std.fs.path.join(a, &.{ root, "refs" }));
        try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "refs", "main" }), .data = sha });
        try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ snapshot, "config.json" }), .data = config });
        try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ snapshot, "model.safetensors" }), .data = &weights });
        return snapshot;
    }

    pub fn path(h: *const Home, a: Allocator, name: []const u8) ![]const u8 {
        return std.fs.path.join(a, &.{ h.dir, name });
    }
};

/// ``input`` through ``mcp.run`` as stdin brings it, to its end; the lines written, each one JSON object.
pub fn exchange(a: Allocator, ctx: *const mcp.Context, input: []const u8) ![]Value {
    var fixed: std.Io.Reader = .fixed(input);
    var stdin: std.Io.Reader.Limited = .init(&fixed, .unlimited, try a.alloc(u8, rpc.max_line));
    var out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try mcp.run(ctx, &stdin.interface, &out.writer));
    return lines(a, out.written());
}

/// Every line of ``bytes`` parsed: each a whole JSON object, and the text ends at a line's end.
pub fn lines(a: Allocator, bytes: []const u8) ![]Value {
    var out: std.ArrayList(Value) = .empty;
    var it = std.mem.splitScalar(u8, bytes, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const v = switch (try json.parse(a, line)) {
            .ok => |v| v,
            .err => return error.TestUnexpectedResult,
        };
        try std.testing.expect(v == .object);
        try out.append(a, v);
    }
    try std.testing.expect(bytes.len == 0 or bytes[bytes.len - 1] == '\n');
    return out.items;
}

/// The answer to request ``id`` among ``got``, or null.
pub fn find(got: []const Value, id: i64) ?Value {
    for (got) |v| if (v.get("id")) |x| if (x.int64() == id) return v;
    return null;
}

/// The JSON-RPC error code an answer carries, or null for a result.
pub fn code(v: Value) ?i64 {
    return (v.get("error") orelse return null).get("code").?.int64();
}

/// A tool answer's text and whether it is a tool error.
pub fn text(v: Value) []const u8 {
    const content = v.get("result").?.get("content").?.array;
    return content[0].strField("text").?;
}

pub fn isError(v: Value) bool {
    return v.get("result").?.get("isError").?.bool;
}

pub fn has(haystack: []const u8, needle: []const u8) bool {
    if (std.mem.indexOf(u8, haystack, needle) != null) return true;
    std.debug.print("missing \"{s}\" in:\n{s}\n", .{ needle, haystack });
    return false;
}

/// A session fed line by line, for calls that answer later from their own threads; it must not move once set up.
pub const Live = struct {
    out: std.Io.Writer.Allocating,
    wire: rpc.Wire,
    session: mcp.Session,
    arena: std.heap.ArenaAllocator,

    pub fn setUp(l: *Live, ctx: *const mcp.Context) void {
        l.out = .init(std.testing.allocator);
        l.wire = .{ .out = &l.out.writer, .io = ctx.io };
        l.session = .init(ctx, &l.wire);
        l.arena = .init(std.testing.allocator);
    }

    /// The session's end, as stdin's end brings it, and everything freed.
    pub fn tearDown(l: *Live) void {
        l.session.finish();
        l.session.deinit();
        l.arena.deinit();
        l.out.deinit();
    }

    pub fn send(l: *Live, line: []const u8) !void {
        _ = l.arena.reset(.retain_capacity);
        try l.session.handle(l.arena.allocator(), line);
    }

    /// Every line written so far, read under the wire's lock.
    pub fn written(l: *Live, a: Allocator) ![]Value {
        l.wire.mutex.lockUncancelable(l.wire.io);
        const copy = a.dupe(u8, l.out.written());
        l.wire.mutex.unlock(l.wire.io);
        return lines(a, try copy);
    }

    /// The answer to request ``id``, waited for up to 20 s.
    pub fn answer(l: *Live, a: Allocator, id: i64) !Value {
        var waited: u32 = 0;
        while (waited < 20_000) : (waited += 20) {
            if (find(try l.written(a), id)) |v| return v;
            std.testing.io.sleep(.fromMilliseconds(20), .awake) catch {};
        }
        return error.TestTimedOut;
    }

    /// Waits up to 20 s for every running call to finish.
    pub fn idle(l: *Live) !void {
        var waited: u32 = 0;
        while (waited < 20_000) : (waited += 20) {
            const busy = for (l.session.calls.items) |c| {
                if (!c.done.load(.acquire)) break true;
            } else false;
            if (!busy) return;
            std.testing.io.sleep(.fromMilliseconds(20), .awake) catch {};
        }
        return error.TestTimedOut;
    }
};

/// Waits up to 10 s for ``path`` to exist, as a stand-in child writes its files.
pub fn appears(path: []const u8) !void {
    var waited: u32 = 0;
    while (waited < 10_000) : (waited += 20) {
        if (std.Io.Dir.cwd().access(std.testing.io, path, .{})) |_| return else |_| {}
        std.testing.io.sleep(.fromMilliseconds(20), .awake) catch {};
    }
    return error.TestTimedOut;
}
