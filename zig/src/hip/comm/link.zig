//! The ranks' TCP link: rank 0 listens, the others connect once, every rank's hello is checked, then rank 0 commands.

const std = @import("std");
const posix = @import("sys").posix;
const rccl = @import("rccl.zig");
const agree_ = @import("../agree.zig");
const Policy = @import("../policy.zig").Policy;
const Identity = agree_.Identity;
const Field = agree_.Field;
const Fault = agree_.Fault;
const auth = @import("sys").rank_auth;
const Seal = auth.seal.Seal;

pub const Error = error{
    SocketFailed,
    BindFailed,
    ConnectFailed,
    LinkClosed,
    BadRank,
    RanksDiffer,
} || auth.Error || std.mem.Allocator.Error;

pub const max_world = 16;

/// The hello's first word, and its layout's version: a rank of another build is refused before its fields are read.
const magic: u32 = 0x4c484654;
const layout: u32 = 2;

/// Milliseconds rank 0 keeps telling late ranks that the group was refused before it gives up on them.
const linger_ms = 5000;

/// Milliseconds a rank waits for rank 0's side of the key proof: rank 0 may be busy with other connections.
const proof_ms = 600_000;

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
pub const Answer = extern struct { magic: u32 = magic, field: u32 = 0, rank: u32 = 0, pad: u32 = 0 };

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
    /// Each socket's seal in a keyed group: every byte after the key proof crosses in its records. Null with no key.
    seals: [max_world - 1]?*Seal = undefined,

    /// Rank 0 listens on `host:port`, answers once every proven hello matches its own; `fault` names one that does not.
    pub fn open(
        io: std.Io,
        host: []const u8,
        port: u16,
        id: rccl.UniqueId,
        hello: Hello,
        policy: [Policy.word_count]u32,
        fault: *Fault,
        secret: *const auth.Secret,
    ) Error!struct { Link, rccl.UniqueId, [Policy.word_count]u32 } {
        const rank = hello.rank;
        const world = hello.world;
        if (world < 2 or world > max_world or rank >= world) return error.BadRank;
        const ip = std.Io.net.IpAddress.parse(host, port) catch return error.ConnectFailed;
        if (ip != .ip4) return error.ConnectFailed;
        var addr: posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.readInt(u32, &ip.ip4.bytes, .little) };
        var link: Link = .{ .rank = rank, .world = world };
        if (rank == 0) {
            try link.lead(&addr, hello, fault, secret);
            for (0..world - 1) |i| {
                try link.peer(i).write(std.mem.asBytes(&Answer{}));
                try link.peer(i).write(std.mem.asBytes(&id));
                try link.peer(i).write(std.mem.sliceAsBytes(&policy));
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
        const session = try auth.connect(link.fds[0], secret, .rocm, rank, world, proof_ms);
        const lead_ = try Peer.init(link.fds[0], &session, true, secret.keyed);
        errdefer if (lead_.seal) |s| std.heap.page_allocator.destroy(s);
        link.seals[0] = lead_.seal;
        try lead_.write(std.mem.asBytes(&hello));
        var answer: Answer = undefined;
        try lead_.read(std.mem.asBytes(&answer), null);
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
        try lead_.read(std.mem.asBytes(&got), null);
        var theirs: [Policy.word_count]u32 = undefined;
        try lead_.read(std.mem.sliceAsBytes(&theirs), null);
        return .{ link, got, theirs };
    }

    /// Rank 0 takes every hello; at the first that differs, every rank in (and any arriving soon after) is told.
    fn lead(l: *Link, addr: *posix.sockaddr.in, hello: Hello, fault: *Fault, secret: *const auth.Secret) Error!void {
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
        var linger_until: u64 = 0;
        while (arrived < l.world - 1) {
            const wait: i32 = if (refused == null) -1 else @intCast(@min(linger_until -| auth.nowMs(), linger_ms));
            const fd = (try accept(server, wait)) orelse break;
            // a connection that proves no key takes no place and hears no answer: it is closed and the wait goes on
            const who = auth.accept(fd, secret, &.{.rocm}, auth.budget_ms) catch |e| {
                var at: [64]u8 = undefined;
                std.log.warn("hip: a connection from {s} to the ranks' link proved no key and is ignored: {s}", .{ auth.peerText(fd, &at), auth.explain(e) orelse @errorName(e) });
                _ = posix.system.close(fd);
                continue;
            };
            const from = Peer.init(fd, &who.session, false, secret.keyed) catch {
                _ = posix.system.close(fd);
                continue;
            };
            arrived += 1;
            var h = std.mem.zeroes(Hello);
            const field = if (readHello(from, &h)) |_| (if (h.rank != who.rank or h.world != who.world) Field.rank else differs(hello, h, &taken)) else |_| Field.layout;
            if (refused == null) {
                if (field) |f| {
                    refused = .{ .field = f, .rank = h.rank };
                    linger_until = auth.nowMs() + linger_ms;
                    for (1..l.world) |r| if (taken[r]) tell(l.peer(r - 1), refused.?);
                } else {
                    taken[h.rank] = true;
                    l.fds[h.rank - 1] = fd;
                    l.seals[h.rank - 1] = from.seal;
                    continue;
                }
            }
            tell(from, refused.?);
        }
        if (refused) |f| {
            fault.* = f;
            return error.RanksDiffer;
        }
    }

    pub fn close(l: *Link) void {
        const n = if (l.rank == 0) l.world - 1 else 1;
        for (0..n) |i| l.peer(i).deinit();
    }

    fn peer(l: *const Link, i: usize) Peer {
        return .{ .fd = l.fds[i], .seal = l.seals[i] };
    }

    /// Rank 0: `words` to every follower, behind their count.
    pub fn send(l: *const Link, words: []const u32) Error!void {
        const n: u32 = @intCast(words.len);
        for (0..l.world - 1) |i| {
            try l.peer(i).write(std.mem.asBytes(&n));
            try l.peer(i).write(std.mem.sliceAsBytes(words));
        }
    }

    /// A follower: rank 0's next message, appended to `out` (cleared first).
    pub fn recv(l: *const Link, gpa: std.mem.Allocator, out: *std.ArrayList(u32)) Error!void {
        var n: u32 = undefined;
        try l.peer(0).read(std.mem.asBytes(&n), null);
        out.clearRetainingCapacity();
        try out.resize(gpa, n);
        try l.peer(0).read(std.mem.sliceAsBytes(out.items), null);
    }

    /// A startup step: each follower's outcome to rank 0, rank 0's verdict to all; a rank whose link closed refuses.
    pub fn agree(l: *const Link, mine: agree_.Step, how: agree_.How) Error!agree_.Verdict {
        if (l.rank != 0) {
            try l.peer(0).write(std.mem.asBytes(&mine));
            var v: agree_.Verdict = undefined;
            try l.peer(0).read(std.mem.asBytes(&v), null);
            return v;
        }
        var steps: [max_world]agree_.Step = undefined;
        steps[0] = mine;
        for (1..l.world) |r| l.peer(r - 1).read(std.mem.asBytes(&steps[r]), null) catch {
            steps[r] = .refuse("its link to rank 0 closed");
        };
        const v = agree_.combine(steps[0..l.world], how);
        for (0..l.world - 1) |i| l.peer(i).write(std.mem.asBytes(&v)) catch {};
        return v;
    }
};

/// One proven connection: in a keyed group every byte on it crosses in its seal's records.
const Peer = struct {
    fd: posix.socket_t,
    /// Null with no key: the proof's session key is then public, so the bytes cross as they did before seals.
    seal: ?*Seal,

    /// The seal is kept off the stack: a link's sockets outlive the call that opens them.
    fn init(fd: posix.socket_t, session: *const auth.Key, connector: bool, keyed: bool) Error!Peer {
        if (!keyed) return .{ .fd = fd, .seal = null };
        const seal = try std.heap.page_allocator.create(Seal);
        seal.* = .init(session, connector);
        return .{ .fd = fd, .seal = seal };
    }

    fn deinit(p: Peer) void {
        if (p.seal) |s| std.heap.page_allocator.destroy(s);
        _ = posix.system.close(p.fd);
    }

    fn write(p: Peer, bytes: []const u8) Error!void {
        const s = p.seal orelse return writeAll(p.fd, bytes);
        var wire: [auth.seal.max_record]u8 = undefined;
        var at: usize = 0;
        while (at < bytes.len) {
            const n = @min(bytes.len - at, auth.seal.max_plain);
            try writeAll(p.fd, wire[0..s.seal(bytes[at..][0..n], &wire)]);
            at += n;
        }
    }

    /// Fills `bytes`; each wait ends at `until` (auth.nowMs's clock) when one is given.
    fn read(p: Peer, bytes: []u8, until: ?u64) Error!void {
        const s = p.seal orelse return fill(p.fd, bytes, until);
        var at = s.take(bytes);
        while (at < bytes.len) {
            var head: [auth.seal.head_len]u8 = undefined;
            try fill(p.fd, &head, until);
            var body: [auth.seal.max_plain + auth.seal.tag_len]u8 = undefined;
            const n = Seal.bodyLen(&head) catch return unsealed();
            try fill(p.fd, body[0..n], until);
            s.open(&head, body[0..n]) catch return unsealed();
            at += s.take(bytes[at..]);
        }
    }
};

/// A record that does not open ends the link: something between the ranks changed it, or sent it.
fn unsealed() Error {
    std.log.warn("hip: a frame on the ranks' link did not open under the group's key, so the link is closed", .{});
    return error.LinkClosed;
}

fn fill(fd: posix.socket_t, bytes: []u8, until: ?u64) Error!void {
    return if (until) |end| readBefore(fd, bytes, end) else readAll(fd, bytes);
}

/// A proven rank's hello, its layout checked before the rest is read; a rank that stalls is not waited on.
fn readHello(from: Peer, h: *Hello) Error!void {
    const until = auth.nowMs() + auth.budget_ms;
    const bytes = std.mem.asBytes(h);
    try from.read(bytes[0..8], until);
    if (h.magic != magic or h.layout != layout) return;
    try from.read(bytes[8..], until);
}

/// Fills `bytes` before `until` (auth.nowMs's clock); a peer that sends too little in time counts as gone.
fn readBefore(fd: posix.socket_t, bytes: []u8, until: u64) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        const left = until -| auth.nowMs();
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        if (left == 0 or (posix.poll(&fds, @intCast(left)) catch 0) == 0) return error.LinkClosed;
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

/// Tells a rank that the group was refused, and why; it may already be gone.
fn tell(to: Peer, f: Fault) void {
    to.write(std.mem.asBytes(&Answer{ .field = @backingInt(f.field) + 1, .rank = f.rank })) catch {};
    to.deinit();
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
