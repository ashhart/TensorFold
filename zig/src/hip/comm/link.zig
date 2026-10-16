//! The ranks' TCP link: rank 0 listens, the others connect once, every rank's hello is checked, then rank 0 commands.

const std = @import("std");
const posix = @import("sys").posix;
const rccl = @import("rccl.zig");
const agree_ = @import("../agree.zig");
const Policy = @import("../policy.zig").Policy;
const Identity = agree_.Identity;
const Field = agree_.Field;
const Fault = agree_.Fault;

pub const Error = error{
    SocketFailed,
    BindFailed,
    ConnectFailed,
    LinkClosed,
    BadRank,
    RanksDiffer,
} || std.mem.Allocator.Error;

pub const max_world = 16;

/// The hello's first word, and its layout's version: a rank of another build is refused before its fields are read.
const magic: u32 = 0x4c484654;
const layout: u32 = 2;

/// Milliseconds rank 0 keeps telling late ranks that the group was refused before it gives up on them.
const linger_ms = 5000;

/// What a rank says when it joins: its place, its GPU and RCCL, and what it runs.
pub const Hello = extern struct {
    magic: u32 = magic,
    layout: u32 = layout,
    rank: u32,
    world: u32,
    gpu: u32,
    rccl: u32,
    ident: Identity,
};

/// Rank 0's answer to a hello: `field` 0 when the group formed, else the differing field + 1 and its rank.
const Answer = extern struct { magic: u32 = magic, field: u32 = 0, rank: u32 = 0, pad: u32 = 0 };

/// The first field where `h` differs from rank 0's `lead`, given the ranks already `taken`; null when it may join.
pub fn differs(lead: Hello, h: Hello, taken: []const bool) ?Field {
    if (h.magic != magic or h.layout != layout) return .layout;
    if (h.world != lead.world) return .world;
    if (h.rank == 0 or h.rank >= lead.world or taken[h.rank]) return .rank;
    if (h.gpu != lead.gpu) return .gpu;
    if (h.rccl != lead.rccl) return .rccl;
    if (!std.mem.eql(u8, &h.ident.program, &lead.ident.program)) return .program;
    if (!std.mem.eql(u8, &h.ident.config, &lead.ident.config)) return .config;
    if (!std.mem.eql(u8, &h.ident.weights, &lead.ident.weights)) return .weights;
    return null;
}

pub const Link = struct {
    rank: usize,
    world: usize,
    /// Rank 0: the follower of rank r + 1 at index r. A follower: rank 0's socket at index 0.
    fds: [max_world - 1]posix.socket_t = undefined,

    /// Rank 0 listens on `host:port`, answers once every hello matches its own; `fault` names a rank that does not.
    pub fn open(
        io: std.Io,
        host: []const u8,
        port: u16,
        id: rccl.UniqueId,
        hello: Hello,
        policy: [Policy.word_count]u32,
        fault: *Fault,
    ) Error!struct { Link, rccl.UniqueId, [Policy.word_count]u32 } {
        const rank = hello.rank;
        const world = hello.world;
        if (world < 2 or world > max_world or rank >= world) return error.BadRank;
        const ip = std.Io.net.IpAddress.parse(host, port) catch return error.ConnectFailed;
        if (ip != .ip4) return error.ConnectFailed;
        var addr: posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.readInt(u32, &ip.ip4.bytes, .little) };
        var link: Link = .{ .rank = rank, .world = world };
        if (rank == 0) {
            try link.lead(&addr, hello, fault);
            for (link.fds[0 .. world - 1]) |fd| {
                try writeAll(fd, std.mem.asBytes(&Answer{}));
                try writeAll(fd, std.mem.asBytes(&id));
                try writeAll(fd, std.mem.sliceAsBytes(&policy));
            }
            return .{ link, id, policy };
        }
        var tries: usize = 0;
        while (true) : (tries += 1) {
            const fd = try socket();
            if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) == .SUCCESS) {
                link.fds[0] = fd;
                break;
            }
            _ = posix.system.close(fd);
            if (tries >= 6000) return error.ConnectFailed;
            std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
        }
        errdefer _ = posix.system.close(link.fds[0]);
        try writeAll(link.fds[0], std.mem.asBytes(&hello));
        var answer: Answer = undefined;
        try readAll(link.fds[0], std.mem.asBytes(&answer));
        if (answer.magic != magic) {
            fault.* = .{ .field = .layout, .rank = 0 };
            return error.RanksDiffer;
        }
        if (answer.field != 0) {
            const known = answer.field - 1 < @typeInfo(Field).@"enum".field_names.len;
            fault.* = .{
                .field = if (known) @fromBackingInt(@intCast(answer.field - 1)) else .layout,
                .rank = answer.rank,
            };
            return error.RanksDiffer;
        }
        var got: rccl.UniqueId = undefined;
        try readAll(link.fds[0], std.mem.asBytes(&got));
        var theirs: [Policy.word_count]u32 = undefined;
        try readAll(link.fds[0], std.mem.sliceAsBytes(&theirs));
        return .{ link, got, theirs };
    }

    /// Rank 0 takes every hello; at the first that differs, every rank in (and any arriving soon after) is told.
    fn lead(l: *Link, addr: *posix.sockaddr.in, hello: Hello, fault: *Fault) Error!void {
        const server = try socket();
        defer _ = posix.system.close(server);
        const one: c_int = 1;
        posix.setsockopt(
            server,
            posix.SOL.SOCKET,
            posix.SO.REUSEADDR,
            std.mem.asBytes(&one),
        ) catch return error.SocketFailed;
        if (posix.errno(posix.system.bind(
            server,
            @ptrCast(addr),
            @sizeOf(posix.sockaddr.in),
        )) != .SUCCESS) return error.BindFailed;
        if (posix.errno(posix.system.listen(server, 16)) != .SUCCESS) return error.BindFailed;
        var taken: [max_world]bool = @splat(false);
        taken[0] = true;
        var refused: ?Fault = null;
        var arrived: usize = 0;
        while (arrived < l.world - 1) : (arrived += 1) {
            const fd = (try accept(server, if (refused == null) -1 else linger_ms)) orelse break;
            var h = std.mem.zeroes(Hello);
            const field = if (readHello(fd, &h)) |_| differs(hello, h, &taken) else |_| Field.layout;
            if (refused == null) {
                if (field) |f| {
                    refused = .{ .field = f, .rank = h.rank };
                    for (1..l.world) |r| if (taken[r]) tell(l.fds[r - 1], refused.?);
                } else {
                    taken[h.rank] = true;
                    l.fds[h.rank - 1] = fd;
                    continue;
                }
            }
            tell(fd, refused.?);
        }
        if (refused) |f| {
            fault.* = f;
            return error.RanksDiffer;
        }
    }

    pub fn close(l: *Link) void {
        const n = if (l.rank == 0) l.world - 1 else 1;
        for (l.fds[0..n]) |fd| _ = posix.system.close(fd);
    }

    /// Rank 0: `words` to every follower, behind their count.
    pub fn send(l: *const Link, words: []const u32) Error!void {
        const n: u32 = @intCast(words.len);
        for (l.fds[0 .. l.world - 1]) |fd| {
            try writeAll(fd, std.mem.asBytes(&n));
            try writeAll(fd, std.mem.sliceAsBytes(words));
        }
    }

    /// A follower: rank 0's next message, appended to `out` (cleared first).
    pub fn recv(l: *const Link, gpa: std.mem.Allocator, out: *std.ArrayList(u32)) Error!void {
        var n: u32 = undefined;
        try readAll(l.fds[0], std.mem.asBytes(&n));
        out.clearRetainingCapacity();
        try out.resize(gpa, n);
        try readAll(l.fds[0], std.mem.sliceAsBytes(out.items));
    }

    /// A startup step: each follower's outcome to rank 0, rank 0's verdict to all; a rank whose link closed refuses.
    pub fn agree(l: *const Link, mine: agree_.Step, how: agree_.How) Error!agree_.Verdict {
        if (l.rank != 0) {
            try writeAll(l.fds[0], std.mem.asBytes(&mine));
            var v: agree_.Verdict = undefined;
            try readAll(l.fds[0], std.mem.asBytes(&v));
            return v;
        }
        var steps: [max_world]agree_.Step = undefined;
        steps[0] = mine;
        for (1..l.world) |r| readAll(l.fds[r - 1], std.mem.asBytes(&steps[r])) catch {
            steps[r] = .refuse("its link to rank 0 closed");
        };
        const v = agree_.combine(steps[0..l.world], how);
        for (l.fds[0 .. l.world - 1]) |fd| writeAll(fd, std.mem.asBytes(&v)) catch {};
        return v;
    }
};

/// A rank's hello, its layout checked before the rest is read.
fn readHello(fd: posix.socket_t, h: *Hello) Error!void {
    const bytes = std.mem.asBytes(h);
    try readAll(fd, bytes[0..8]);
    if (h.magic != magic or h.layout != layout) return;
    try readAll(fd, bytes[8..]);
}

/// Tells a rank that the group was refused, and why; it may already be gone.
fn tell(fd: posix.socket_t, f: Fault) void {
    writeAll(fd, std.mem.asBytes(&Answer{ .field = @backingInt(f.field) + 1, .rank = f.rank })) catch {};
    _ = posix.system.close(fd);
}

/// The next rank to connect, or null when none came within `ms` (negative: wait for one).
fn accept(server: posix.socket_t, ms: i32) Error!?posix.socket_t {
    var fds = [_]posix.pollfd{.{ .fd = server, .events = posix.POLL.IN, .revents = 0 }};
    while (true) {
        const n = posix.system.poll(&fds, 1, ms);
        switch (posix.errno(n)) {
            .SUCCESS => if (n == 0) return null else break,
            .INTR => {},
            else => return error.SocketFailed,
        }
    }
    const rc = posix.system.accept(server, null, null);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    return fd;
}

fn socket() Error!posix.socket_t {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    const one: c_int = 1;
    posix.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.NODELAY, std.mem.asBytes(&one)) catch {};
    return fd;
}

fn writeAll(fd: posix.socket_t, bytes: []const u8) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = posix.system.write(fd, bytes[at..].ptr, bytes.len - at);
        switch (posix.errno(rc)) {
            .SUCCESS => at += @intCast(rc),
            .INTR => {},
            else => return error.LinkClosed,
        }
    }
}

fn readAll(fd: posix.socket_t, bytes: []u8) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = posix.system.read(fd, bytes[at..].ptr, bytes.len - at);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return error.LinkClosed;
                at += @intCast(rc);
            },
            .INTR => {},
            else => return error.LinkClosed,
        }
    }
}
