//! NCCL's own INFO log during init, captured into a private file when the user set no NCCL_DEBUG*, and parsed.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const nvml = @import("p2p_nvml.zig");

/// The variables a capture sets; any of them set by the user leaves NCCL's logging alone.
pub const vars = [_][:0]const u8{ "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE" };
pub const subsystems = "INIT,P2P,SHM,NET";
const file_name = "nccl.log";

/// The environment as a capture reads and writes it (the process's, or a test's).
pub const Env = struct {
    get: *const fn ([*:0]const u8) ?[*:0]const u8,
    set: *const fn ([*:0]const u8, [*:0]const u8) void,
    unset: *const fn ([*:0]const u8) void,
};

pub const process: Env = .{ .get = Process.get, .set = Process.set, .unset = Process.unset };

const Process = struct {
    fn get(name: [*:0]const u8) ?[*:0]const u8 {
        return platform.c.getenv(name);
    }
    fn set(name: [*:0]const u8, value: [*:0]const u8) void {
        _ = platform.env.setenv(name, value, 0);
    }
    fn unset(name: [*:0]const u8) void {
        _ = platform.env.unsetenv(name);
    }
};

extern "c" fn mkdtemp(template: [*:0]u8) ?[*:0]u8;

pub const State = enum { off, user, unavailable, on };

/// A private temp directory NCCL logs into while it initializes; `restore` hands the environment back.
pub const Capture = struct {
    state: State = .off,
    dir: [224:0]u8 = @splat(0),
    dir_len: usize = 0,
    env_set: bool = false,

    /// Sets NCCL_DEBUG=INFO, the subsystems and a private NCCL_DEBUG_FILE, unless the user set any of them.
    pub fn begin(env: Env) Capture {
        for (vars) |v| if (env.get(v) != null) return .{ .state = .user };
        if (comptime builtin.os.tag == .windows) return .{ .state = .unavailable };
        var c: Capture = .{ .state = .unavailable };
        const tmp = if (env.get("TMPDIR")) |t| std.mem.span(t) else "/tmp";
        const base = std.mem.trimEnd(u8, tmp, "/");
        // NCCL expands % in NCCL_DEBUG_FILE (%h host, %p pid): a path holding one would not be ours
        if (std.mem.indexOfScalar(u8, base, '%') != null) return c;
        const made = std.mem.printSentinel(&c.dir, "{s}/tf-p2p-XXXXXX", .{if (base.len == 0) "/tmp" else base}, 0) catch return c;
        if (mkdtemp(c.dir[0..made.len :0].ptr) == null) return c;
        c.dir_len = made.len;
        var file: [256:0]u8 = @splat(0);
        const log_path = c.path(&file) orelse {
            _ = std.c.rmdir(&c.dir);
            return c;
        };
        env.set("NCCL_DEBUG", "INFO");
        env.set("NCCL_DEBUG_SUBSYS", subsystems);
        env.set("NCCL_DEBUG_FILE", log_path);
        c.env_set = true;
        c.state = .on;
        return c;
    }

    fn path(c: *const Capture, buf: *[256:0]u8) ?[:0]const u8 {
        return std.mem.printSentinel(buf, "{s}/" ++ file_name, .{c.dir[0..c.dir_len]}, 0) catch null;
    }

    /// Unsets what `begin` set; NCCL read it at its first message and keeps its own copy.
    pub fn restore(c: *Capture, env: Env) void {
        if (!c.env_set) return;
        for (vars) |v| env.unset(v);
        c.env_set = false;
    }

    /// The log's first `buf.len` bytes so far (empty when NCCL wrote nothing or the capture is off).
    pub fn read(c: *const Capture, buf: []u8) []const u8 {
        if (c.state != .on) return buf[0..0];
        // begin never captures on Windows, whose C library has no POSIX open for this to reference
        return if (comptime builtin.os.tag == .windows) buf[0..0] else c.readFile(buf);
    }

    fn readFile(c: *const Capture, buf: []u8) []const u8 {
        var name: [256:0]u8 = @splat(0);
        const p = c.path(&name) orelse return buf[0..0];
        const fd = std.c.open(p, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) return buf[0..0];
        defer _ = std.c.close(fd);
        var n: usize = 0;
        while (n < buf.len) {
            const got = std.c.read(fd, buf[n..].ptr, buf.len - n);
            if (got <= 0) break;
            n += @intCast(got);
        }
        return buf[0..n];
    }

    /// Removes the log and its directory; NCCL's open handle, if any, keeps writing to the unlinked file.
    pub fn remove(c: *Capture) void {
        if (c.state != .on) return;
        if (comptime builtin.os.tag != .windows) {
            var name: [256:0]u8 = @splat(0);
            if (c.path(&name)) |p| _ = std.c.unlink(p);
            _ = std.c.rmdir(&c.dir);
        }
        c.state = .off;
    }
};

pub const Kind = enum { p2p, shm, net };

/// One way this rank and `peer` connected, with how many channel connections NCCL logged for it.
pub const Link = struct {
    peer: u32,
    kind: Kind,
    read: bool = false,
    net: nvml.Text(16) = .{},
    connections: u32 = 1,

    fn same(a: Link, b: Link) bool {
        return a.peer == b.peer and a.kind == b.kind and a.read == b.read and std.mem.eql(u8, a.net.slice(), b.net.slice());
    }
};

pub const max_links = 32;

pub const Summary = struct {
    links: [max_links]Link = undefined,
    count: usize = 0,

    pub fn list(s: *const Summary) []const Link {
        return s.links[0..s.count];
    }

    fn add(s: *Summary, l: Link) void {
        for (s.links[0..s.count]) |*x| if (x.same(l)) {
            x.connections += 1;
            return;
        };
        if (s.count == max_links) return;
        s.links[s.count] = l;
        s.count += 1;
    }
};

/// NCCL's "Channel NN[/C] : A[d] -> B[d] [send|receive] via P2P|SHM|NET/..." lines involving rank `me`.
pub fn parse(text: []const u8, me: u32) Summary {
    var s: Summary = .{};
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| if (parseLine(line, me)) |l| s.add(l);
    return s;
}

fn parseLine(line: []const u8, me: u32) ?Link {
    const tag = "NCCL INFO Channel ";
    const at = std.mem.indexOf(u8, line, tag) orelse return null;
    const rest = line[at + tag.len ..];
    const ends = rest[(std.mem.indexOf(u8, rest, " : ") orelse return null) + 3 ..];
    const arrow = std.mem.indexOf(u8, ends, " -> ") orelse return null;
    const via = std.mem.indexOf(u8, ends, " via ") orelse return null;
    if (via < arrow) return null;
    const from = rankOf(ends[0..arrow]) orelse return null;
    const to = rankOf(ends[arrow + 4 .. via]) orelse return null;
    const how = std.mem.trimEnd(u8, ends[via + 5 ..], " \r");
    const peer = if (from == me) to else if (to == me) from else return null;
    if (std.mem.startsWith(u8, how, "P2P/")) return .{ .peer = peer, .kind = .p2p, .read = std.mem.indexOf(u8, how, "/read") != null };
    if (std.mem.startsWith(u8, how, "SHM/")) return .{ .peer = peer, .kind = .shm };
    if (!std.mem.startsWith(u8, how, "NET/")) return null;
    const net = how[4..];
    return .{ .peer = peer, .kind = .net, .net = .of(net[0 .. std.mem.indexOfScalar(u8, net, '/') orelse net.len]) };
}

fn rankOf(text: []const u8) ?u32 {
    const t = std.mem.trim(u8, text, " ");
    return std.fmt.parseInt(u32, t[0 .. std.mem.indexOfScalar(u8, t, '[') orelse t.len], 10) catch null;
}

/// "rank 1 P2P read (2 connections); rank 2 NET/IB (4 connections)", or nothing for an empty summary.
pub fn render(w: *std.Io.Writer, s: *const Summary) !void {
    for (s.list(), 0..) |l, i| {
        try w.print("{s}rank {d} ", .{ if (i > 0) "; " else "", l.peer });
        switch (l.kind) {
            .p2p => try w.print("P2P {s}", .{if (l.read) "read" else "write"}),
            .shm => try w.writeAll("SHM"),
            .net => try w.print("NET/{s}", .{l.net.slice()}),
        }
        try w.print(" ({d} connection{s})", .{ l.connections, if (l.connections == 1) "" else "s" });
    }
}
