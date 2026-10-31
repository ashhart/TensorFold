//! Registered windows over first-party MCDMA: one progress context per port, with staged sources held through flush.
const std = @import("std");
const platform = @import("sys");
const abi = @import("mcdma_abi.zig");
const rma = @import("rdma.zig");
const words = @import("words.zig");
const verbs = @import("verbs.zig");

pub const alignment = 16384;
pub const max_ranks = 64;
pub const Error = abi.Error || std.mem.Allocator.Error || error{ InvalidConfig, OpenFailed, ConnectFailed, UnsupportedLink, WindowMismatch, ThreadFailed, NeedNLinkLibrary, NeedGidIndex };

const env = @import("sys").env;
const log = std.log.scoped(.mcdma);

/// MCDMA_FABRIC_ROCE (mcdma_fabric.h): a ConnectX link; abi.thunderbolt is the other kind MCDMA reports.
pub const roce: c_int = 1;
/// The path MTUs MCDMA's queue pairs take; it would quietly run any other value at 1024.
const path_mtus = [_]c_int{ 1024, 2048, 4096 };
/// Additive ABI 1 (mcdma_fabric.h): MCDMA_FABRIC_PEER and the first reason once the peer is down for good.
const PeerStatus = *const fn (*const abi.Peer, ?[*]u8, usize) callconv(.c) c_int;

/// `mtu`: the RC path MTU, the same on both ends and at most the port's active MTU. `kind`: null takes either kind.
pub const Link = struct { peer: u32, device: [:0]const u8, via: [:0]const u8, port: u16, peer_port: u16 = 0, name: [:0]const u8, gid: c_int = 1, ports: []const u16 = &.{}, peer_ports: []const u16 = &.{}, mtu: c_int = 4096, kind: ?rma.LinkKind = null };
pub const Config = struct { rank: u32, ranks: u32, window_bytes: usize, staging_bytes: usize, links: []const Link, timeout_ns: u64 = 10 * std.time.ns_per_s, connect_timeout_ns: u64 = 300 * std.time.ns_per_s };

/// MCDMA's name for a status (enum mcdma_fabric_status, mcdma_fabric.h), for the lines that say why a link failed.
pub fn statusName(status: c_int) []const u8 {
    return switch (status) {
        0 => "OK",
        1 => "INVALID",
        2 => "UNSUPPORTED",
        3 => "NOMEM",
        4 => "DEVICE",
        5 => "PEER",
        6 => "TIMEOUT",
        7 => "BOUNDS",
        else => "unknown",
    };
}

/// Validate physical member counts before opening a device or allocating its registered window.
fn memberCount(gpa: std.mem.Allocator, link: Link) Error!usize {
    const devices = verbs.deviceParts(gpa, link.device) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidConfig,
    };
    defer gpa.free(devices);
    if (link.gid < -1) return error.InvalidConfig;
    var vias = std.mem.splitScalar(u8, link.via, '+');
    var count: usize = 0;
    while (vias.next()) |via| {
        if (via.len == 0) return error.InvalidConfig;
        for (via) |c| if (c <= ' ' or c == 127) return error.InvalidConfig;
        count += 1;
    }
    if (count != 1 and count != devices.len) return error.InvalidConfig;
    for ([_][]const u16{ link.ports, link.peer_ports }) |ports| {
        if (ports.len == 0) continue;
        if (ports.len != devices.len) return error.InvalidConfig;
        for (ports, 0..) |port, i| {
            if (port == 0) return error.InvalidConfig;
            for (ports[0..i]) |old| if (port == old) return error.InvalidConfig;
        }
    }
    if ((link.ports.len == 0 and (link.port == 0 or @as(usize, link.port) + devices.len > 65536)) or
        (link.peer_ports.len == 0 and link.peer_port > 0 and @as(usize, link.peer_port) + devices.len > 65536)) return error.InvalidConfig;
    return devices.len;
}

const Lock = struct {
    held: std.atomic.Value(bool) = .init(false),

    fn lock(self: *Lock, deadline_ns: u64) rma.Error!void {
        while (self.held.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            if (try words.checkedNowNs() >= deadline_ns) return error.Timeout;
            words.pause(1000);
        }
    }

    fn unlock(self: *Lock) void {
        self.held.store(false, .release);
    }
};

const Port = struct {
    spec: Link,
    context: ?*abi.Context = null,
    peer: ?*abi.Peer = null,
    used: usize = 0,
    pending: bool = false, // writes or signals posted since the last flush
    status: c_int = 0,
    link: c_int = 0, // abi.thunderbolt or roce, as MCDMA reported it at connect
    down: bool = false, // the peer-down line has been logged
    mutex: Lock = .{},
};

/// Every check that needs no library, so a bad link fails by name before anything opens; returns the window bytes.
fn validate(gpa: std.mem.Allocator, config: Config) Error!usize {
    if (config.ranks == 0 or config.ranks > max_ranks or config.rank >= config.ranks or config.links.len >= config.ranks or config.timeout_ns == 0 or config.connect_timeout_ns == 0 or config.window_bytes == 0 or config.staging_bytes == 0 or config.window_bytes % alignment != 0 or config.staging_bytes % alignment != 0) return error.InvalidConfig;
    for (config.links, 0..) |link, i| {
        if (link.peer >= config.ranks or link.peer == config.rank or link.port == 0 or link.device.len == 0 or link.via.len == 0 or link.name.len == 0 or link.name.len > 20) return error.InvalidConfig;
        if (std.mem.indexOfScalar(c_int, &path_mtus, link.mtu) == null) return error.InvalidConfig;
        for (config.links[0..i]) |old| if (old.peer == link.peer or std.mem.eql(u8, old.device, link.device)) return error.InvalidConfig;
        const n = try memberCount(gpa, link);
        if (link.kind) |kind| switch (kind) {
            .tb5 => {},
            // MCDMA scans for a GID on Thunderbolt only and refuses a RoCE bond; port lists stay Thunderbolt's here
            .cx5 => {
                if (link.gid < 0) return error.NeedGidIndex;
                if (n > 1 or link.ports.len > 0 or link.peer_ports.len > 0) return error.UnsupportedLink;
            },
            .local, .nccl => return error.InvalidConfig,
        };
    }
    const stage = std.math.mul(usize, config.ranks, config.staging_bytes) catch return error.InvalidConfig;
    return std.math.add(usize, config.window_bytes, stage) catch return error.InvalidConfig;
}

/// Whether a port may run over the link kind MCDMA reported: Thunderbolt as before, or one unbonded ConnectX link.
fn accepts(spec: Link, link: c_int) bool {
    if (spec.kind) |kind| if ((kind == .cx5) != (link == roce)) return false;
    if (link == abi.thunderbolt) return true;
    return link == roce and std.mem.indexOfScalar(u8, spec.device, '+') == null and spec.ports.len == 0 and spec.peer_ports.len == 0;
}

/// One line naming the link, the step and MCDMA's status, so a startup failure says why.
fn startFailed(spec: Link, step: []const u8, status: c_int) void {
    const hint: []const u8 = switch (status) {
        2 => ": a ConnectX link takes one device, no bond, and one queue pair (MCDMA_FABRIC_QPS unset or 1)",
        // gid -1 is Thunderbolt's default for a bond: keep the port and MTU advice, and add the ConnectX cause
        4 => if (spec.gid < 0) ": the device, its registration or a queue pair failed; is the port active, and its active MTU at least the path MTU? gid -1 scans Thunderbolt only: a ConnectX link needs its RoCE v2 GID index" else ": the device, its registration or a queue pair failed; is the port active, and its active MTU at least the path MTU?",
        5 => ": the peer refused; both ends need the same link kind, name and bond",
        6 => ": no answer within connect_timeout_ns",
        else => "",
    };
    log.warn("link \"{s}\" ({s} via {s}, gid {d}, path MTU {d}): {s} failed with MCDMA status {d} {s}{s}", .{ spec.name, spec.device, spec.via, spec.gid, spec.mtu, step, status, statusName(status), hint });
}

pub const Endpoint = struct {
    gpa: std.mem.Allocator,
    library: abi.Library,
    config: Config,
    memory: []align(alignment) u8,
    ports: []Port,
    deadline_ns: u64 = 0,
    peer_status: ?PeerStatus = null,
    owns_library: bool = true,

    /// A heap-owned endpoint keeps all thread and library references stable until deinit closes every port.
    pub fn create(gpa: std.mem.Allocator, library_path: [:0]const u8, config: Config) Error!*Endpoint {
        const bytes = try validate(gpa, config);
        // the library places each arriving message on a CPU thread: user-interactive unless MCDMA_FABRIC_QOS is set
        _ = env.setenv("MCDMA_FABRIC_QOS", "1", 0);
        const self = try gpa.create(Endpoint);
        errdefer gpa.destroy(self);
        var library = try abi.Library.open(library_path);
        errdefer library.close();
        try self.start(gpa, library, library.handle.lookup(PeerStatus, "mcdma_fabric_peer_status"), true, config, bytes);
        return self;
    }

    /// create over a library table the caller keeps (mcdma_test.zig's fakes): deinit leaves it open.
    pub fn createWith(gpa: std.mem.Allocator, library: abi.Library, peer_status: ?PeerStatus, config: Config) Error!*Endpoint {
        const bytes = try validate(gpa, config);
        const self = try gpa.create(Endpoint);
        errdefer gpa.destroy(self);
        try self.start(gpa, library, peer_status, false, config, bytes);
        return self;
    }

    /// Open every port, connect them all at once and check what each reached; on failure everything opened is closed.
    fn start(self: *Endpoint, gpa: std.mem.Allocator, library: abi.Library, peer_status: ?PeerStatus, owns_library: bool, config: Config, bytes: usize) Error!void {
        self.* = .{ .gpa = gpa, .library = library, .config = config, .memory = undefined, .ports = undefined, .peer_status = peer_status, .owns_library = owns_library };
        for (config.links) |link| {
            const n = try memberCount(gpa, link);
            const supported: usize = if (self.library.max_links) |f| f() else 2;
            if (n > supported or ((link.ports.len > 0 or link.peer_ports.len > 0) and self.library.connect_links == null)) return error.NeedNLinkLibrary;
        }
        self.memory = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), bytes);
        errdefer gpa.free(self.memory);
        @memset(self.memory, 0);
        self.ports = try gpa.alloc(Port, config.links.len);
        errdefer gpa.free(self.ports);
        for (self.ports, config.links) |*port, spec| port.* = .{ .spec = spec };
        errdefer self.closePorts();
        for (self.ports) |*port| {
            const status = self.library.api.open(port.spec.device.ptr, port.spec.gid, port.spec.mtu, self.memory.ptr, self.memory.len, -1, 0, abi.progress_thread, &port.context);
            if (status != 0 or port.context == null) {
                startFailed(port.spec, "open", status);
                // UNSUPPORTED: a ConnectX bond or more than one queue pair (a Thunderbolt open never returns it)
                return if (status == 2) error.UnsupportedLink else error.OpenFailed;
            }
        }
        var threads: [max_ranks]?std.Thread = @splat(null);
        var started: usize = 0;
        var failed = false;
        for (self.ports, 0..) |_, i| {
            threads[i] = std.Thread.spawn(.{}, connect, .{ self, i }) catch {
                failed = true;
                break;
            };
            started += 1;
        }
        for (threads[0..started]) |thread| thread.?.join();
        if (failed) return error.ThreadFailed;
        for (self.ports) |*port| {
            if (port.status != 0 or port.peer == null) {
                startFailed(port.spec, "connect", port.status);
                return error.ConnectFailed;
            }
            port.link = self.library.api.link(port.peer.?);
            if (!accepts(port.spec, port.link)) {
                log.warn("link \"{s}\" ({s}, kind {s}): MCDMA reports link kind {d} (1 ConnectX, 2 Thunderbolt); a port takes Thunderbolt, or ConnectX as one device with no bond or port lists", .{ port.spec.name, port.spec.device, if (port.spec.kind) |k| @tagName(k) else "any", port.link });
                return error.UnsupportedLink;
            }
            if (self.library.api.peer_length(port.peer.?) != self.memory.len) return error.WindowMismatch;
        }
    }

    fn connect(self: *Endpoint, i: usize) void {
        const port = &self.ports[i];
        const spec = port.spec;
        if (spec.ports.len > 0 or spec.peer_ports.len > 0) {
            var local: [verbs.max_links]u16 = undefined;
            var peer: [verbs.max_links]u16 = undefined;
            const n = std.mem.count(u8, spec.device, "+") + 1;
            for (0..n) |k| {
                local[k] = if (spec.ports.len > 0) spec.ports[k] else @intCast(@as(usize, spec.port) + k);
                peer[k] = if (spec.peer_ports.len > 0) spec.peer_ports[k] else if (spec.peer_port > 0) @intCast(@as(usize, spec.peer_port) + k) else local[k];
            }
            port.status = self.library.connect_links.?(port.context.?, spec.via.ptr, &local, &peer, @intCast(n), spec.name.ptr, self.config.connect_timeout_ns, &port.peer);
        } else port.status = self.library.api.connect(port.context.?, spec.via.ptr, spec.port, spec.peer_port, spec.name.ptr, self.config.connect_timeout_ns, &port.peer);
    }

    fn closePorts(self: *Endpoint) void {
        for (self.ports) |*port| if (port.peer != null) self.library.api.disconnect(&port.peer);
        for (self.ports) |*port| if (port.context != null) self.library.api.close(&port.context);
    }

    /// The caller stops GPU access first; progress threads close before their registered memory or code is freed.
    pub fn deinit(self: *Endpoint) void {
        self.closePorts();
        self.gpa.free(self.ports);
        self.gpa.free(self.memory);
        if (self.owns_library) self.library.close();
        self.gpa.destroy(self);
    }

    /// Send `len` bytes already in this rank's window at `local` to the peer's window at `offset`, then store `value` at its `flag`, in one message and without staging. The library uses the room just before `local` for its head; the caller leaves the bytes unchanged until the peer has them (a reply from the peer that needed them is proof).
    pub fn writeSignalFrom(self: *Endpoint, peer: u32, local: usize, offset: usize, len: usize, flag: usize, value: u64) rma.Error!void {
        if (len == 0) return signal(self, peer, flag, value); // the library takes a zero-length write-and-signal as a dead peer
        if (local < abi.ws_room or !self.in(local - abi.ws_room, len + abi.ws_room) or !self.in(offset, len) or !self.in(flag, 8)) return error.OutOfBounds;
        if (flag % 8 != 0) return error.Unaligned;
        const port = try self.portFor(peer);
        try port.mutex.lock(try self.end());
        defer port.mutex.unlock();
        if (port.link == roce) { // ConnectX has no fused message: the write, then the signal, on the one queue pair
            try self.checkPort(port, self.library.api.write(port.peer.?, local, offset, len));
            port.pending = true;
            return self.checkPort(port, self.library.api.signal(port.peer.?, flag, value));
        }
        try self.checkPort(port, self.library.write_signal.?(port.peer.?, local, offset, len, flag, value));
        port.pending = true;
    }

    /// MCDMA's first reason the link to `peer` went down, into `buf`; null while it is up or the library cannot say.
    pub fn peerStatus(self: *Endpoint, peer: u32, buf: []u8) ?[]const u8 {
        return self.reason(self.portFor(peer) catch return null, buf); // an atomic load in MCDMA: a waiting rank may poll it
    }

    fn reason(self: *const Endpoint, port: *const Port, buf: []u8) ?[]const u8 {
        const status = self.peer_status orelse return null;
        const p = port.peer orelse return null;
        if (buf.len == 0) return null;
        buf[0] = 0;
        if (status(p, buf.ptr, buf.len) != 5) return null;
        buf[buf.len - 1] = 0;
        return std.mem.sliceTo(buf, 0);
    }

    /// Whether `peer` is over ConnectX, where MCDMA has no fused write-and-signal.
    fn overRoce(self: *Endpoint, peer: u32) bool {
        const port = self.portFor(peer) catch return false;
        return port.link == roce;
    }

    pub fn rdma(self: *Endpoint) rma.Rdma {
        if (self.library.write_signal != null) return .{ .ptr = self, .vtable = &.{ .rank = rank, .size = size, .window = window, .write = write, .write2 = write2, .signal = signal, .write2_signal = write2Signal, .fetch_add = fetchAdd, .read = read, .flush = flush, .link = kind, .deadline = deadline } };
        return .{ .ptr = self, .vtable = &.{ .rank = rank, .size = size, .window = window, .write = write, .write2 = write2, .signal = signal, .fetch_add = fetchAdd, .read = read, .flush = flush, .link = kind, .deadline = deadline } };
    }

    fn of(ptr: *anyopaque) *Endpoint {
        return @ptrCast(@alignCast(ptr));
    }
    fn deadline(ptr: *anyopaque, deadline_ns: u64) void {
        of(ptr).deadline_ns = deadline_ns;
    }

    fn end(self: *const Endpoint) rma.Error!u64 {
        const now = try words.checkedNowNs();
        const local = std.math.add(u64, now, self.config.timeout_ns) catch return error.Timeout;
        const until = if (self.deadline_ns != 0) @min(local, self.deadline_ns) else local;
        if (now >= until) return error.Timeout;
        return until;
    }

    fn left(self: *const Endpoint) rma.Error!u64 {
        const until = try self.end();
        const now = try words.checkedNowNs();
        if (now >= until) return error.Timeout;
        return until - now;
    }
    fn rank(ptr: *anyopaque) u32 {
        return of(ptr).config.rank;
    }
    fn size(ptr: *anyopaque) u32 {
        return of(ptr).config.ranks;
    }
    fn window(ptr: *anyopaque) []align(rma.page) u8 {
        const self = of(ptr);
        return self.memory[0..self.config.window_bytes];
    }

    fn portFor(self: *Endpoint, peer: u32) rma.Error!*Port {
        if (peer >= self.config.ranks) return error.NoSuchRank;
        for (self.ports) |*port| if (port.spec.peer == peer) return port;
        return error.NoSuchRank;
    }

    fn in(self: *const Endpoint, offset: usize, len: usize) bool {
        return offset <= self.config.window_bytes and len <= self.config.window_bytes - offset;
    }

    fn check(status: c_int) rma.Error!void {
        return switch (status) {
            0 => {},
            2 => error.AccessDenied,
            7 => error.OutOfBounds,
            6 => error.Timeout,
            else => error.PeerDown,
        };
    }

    /// check for one port's call, under its lock: the first PeerDown logs the link, the rank and MCDMA's reason once.
    fn checkPort(self: *Endpoint, port: *Port, status: c_int) rma.Error!void {
        check(status) catch |err| {
            if (err == error.PeerDown and !port.down) {
                port.down = true;
                var why: [256]u8 = undefined;
                log.warn("link \"{s}\" to rank {d}: peer down (MCDMA status {d} {s}): {s}", .{ port.spec.name, port.spec.peer, status, statusName(status), self.reason(port, &why) orelse "no reason from the library" });
            }
            return err;
        };
    }

    fn flushPort(self: *Endpoint, port: *Port) rma.Error!void {
        if (!port.pending) return;
        const timeout = try self.left();
        try self.checkPort(port, self.library.api.flush(port.peer.?, timeout));
        port.used = 0;
        port.pending = false;
    }

    fn copy(dst: []u8, src: []const u8) void {
        if (@intFromPtr(dst.ptr) > @intFromPtr(src.ptr)) std.mem.copyBackwards(u8, dst, src) else std.mem.copyForwards(u8, dst, src);
    }

    fn write(ptr: *anyopaque, peer: u32, offset: usize, bytes: []const u8) rma.Error!void {
        const self = of(ptr);
        if (!self.in(offset, bytes.len)) return error.OutOfBounds;
        if (peer == self.config.rank) {
            copy(self.memory[offset..][0..bytes.len], bytes);
            return;
        }
        const port = try self.portFor(peer);
        try port.mutex.lock(try self.end());
        defer port.mutex.unlock();
        try self.post(port, peer, offset, bytes);
    }

    /// Both parts staged back to back and posted as one write; parts too large for one staging pass go separately.
    fn write2(ptr: *anyopaque, peer: u32, offset: usize, head: []const u8, body: []const u8) rma.Error!void {
        const self = of(ptr);
        const total = head.len + body.len;
        if (!self.in(offset, total)) return error.OutOfBounds;
        if (peer == self.config.rank) {
            copy(self.memory[offset..][0..head.len], head);
            copy(self.memory[offset + head.len ..][0..body.len], body);
            return;
        }
        const port = try self.portFor(peer);
        try port.mutex.lock(try self.end());
        defer port.mutex.unlock();
        if (total > self.config.staging_bytes) {
            try self.post(port, peer, offset, head);
            return self.post(port, peer, offset + head.len, body);
        }
        if (total > self.config.staging_bytes - port.used) try self.flushPort(port);
        const local = self.config.window_bytes + peer * self.config.staging_bytes + port.used;
        @memcpy(self.memory[local..][0..head.len], head);
        @memcpy(self.memory[local + head.len ..][0..body.len], body);
        try self.checkPort(port, self.library.api.write(port.peer.?, local, offset, total));
        port.pending = true;
        port.used = std.mem.alignForward(usize, port.used + total, 64);
    }

    /// Stage `bytes` through the port's staging ring and post them, fencing first when the ring would wrap.
    fn post(self: *Endpoint, port: *Port, peer: u32, offset: usize, bytes: []const u8) rma.Error!void {
        var done: usize = 0;
        while (done < bytes.len) {
            _ = try self.left();
            const take = @min(bytes.len - done, self.config.staging_bytes);
            if (take > self.config.staging_bytes - port.used) try self.flushPort(port);
            const local = self.config.window_bytes + peer * self.config.staging_bytes + port.used;
            @memcpy(self.memory[local..][0..take], bytes[done..][0..take]);
            try self.checkPort(port, self.library.api.write(port.peer.?, local, offset + done, take));
            port.pending = true;
            port.used = std.mem.alignForward(usize, port.used + take, 64);
            done += take;
        }
    }

    /// write2 and signal in one message: staged after the library's room for its head, in one staging pass.
    fn write2Signal(ptr: *anyopaque, peer: u32, offset: usize, head: []const u8, body: []const u8, flag: usize, value: u64) rma.Error!void {
        const self = of(ptr);
        const total = head.len + body.len;
        const room = abi.ws_room;
        if (total == 0) return signal(ptr, peer, flag, value); // the library takes a zero-length write-and-signal as a dead peer
        if (!self.in(offset, total) or !self.in(flag, 8)) return error.OutOfBounds; // before any path writes or posts
        if (flag % 8 != 0) return error.Unaligned;
        if (peer == self.config.rank or total + room > self.config.staging_bytes or self.overRoce(peer)) { // ConnectX: no fused message
            try write2(ptr, peer, offset, head, body);
            return signal(ptr, peer, flag, value);
        }
        const port = try self.portFor(peer);
        try port.mutex.lock(try self.end());
        defer port.mutex.unlock();
        if (total + room > self.config.staging_bytes - port.used) try self.flushPort(port);
        const local = self.config.window_bytes + peer * self.config.staging_bytes + port.used + room;
        @memcpy(self.memory[local..][0..head.len], head);
        @memcpy(self.memory[local + head.len ..][0..body.len], body);
        try self.checkPort(port, self.library.write_signal.?(port.peer.?, local, offset, total, flag, value));
        port.pending = true;
        port.used = std.mem.alignForward(usize, port.used + room + total, 64);
    }

    fn signal(ptr: *anyopaque, peer: u32, offset: usize, value: u64) rma.Error!void {
        const self = of(ptr);
        if (offset % 8 != 0) return error.Unaligned;
        if (!self.in(offset, 8)) return error.OutOfBounds;
        if (peer == self.config.rank) {
            @atomicStore(u64, @as(*u64, @ptrCast(@alignCast(self.memory.ptr + offset))), value, .release);
            return;
        }
        const port = try self.portFor(peer);
        try port.mutex.lock(try self.end());
        defer port.mutex.unlock();
        try self.checkPort(port, self.library.api.signal(port.peer.?, offset, value));
        port.pending = true;
    }

    fn fetchAdd(_: *anyopaque, _: u32, _: usize, _: u64) rma.Error!u64 {
        return error.AccessDenied;
    }

    fn read(ptr: *anyopaque, peer: u32, offset: usize, out: []u8) rma.Error!void {
        const self = of(ptr);
        if (peer != self.config.rank) return error.AccessDenied;
        if (!self.in(offset, out.len)) return error.OutOfBounds;
        copy(out, self.memory[offset..][0..out.len]);
    }

    fn flush(ptr: *anyopaque) rma.Error!void {
        const self = of(ptr);
        for (self.ports) |*port| {
            try port.mutex.lock(try self.end());
            defer port.mutex.unlock();
            try self.flushPort(port);
        }
    }

    fn kind(ptr: *anyopaque, peer: u32) rma.LinkKind {
        const self = of(ptr);
        if (peer == self.config.rank) return .local;
        const port = self.portFor(peer) catch return .tb5;
        return if (port.link == roce) .cx5 else .tb5;
    }
};

pub const Region = Endpoint;

test "an endpoint gives the library's placing thread user-interactive QoS unless MCDMA_FABRIC_QOS is set" {
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    const link: Link = .{ .peer = 1, .device = "rdma_en4", .via = "en4/192.0.2.2", .port = 7490, .name = "pair" };
    const config: Config = .{ .rank = 0, .ranks = 2, .window_bytes = alignment, .staging_bytes = alignment, .links = &.{link} };
    defer _ = env.unsetenv("MCDMA_FABRIC_QOS");
    for ([_]?[*:0]const u8{ null, "0", "1" }) |set| {
        _ = if (set) |v| env.setenv("MCDMA_FABRIC_QOS", v, 1) else env.unsetenv("MCDMA_FABRIC_QOS");
        try std.testing.expectError(error.McdmaFabricUnavailable, Endpoint.create(std.testing.allocator, "/nonexistent/libmcdma-fabric.dylib", config));
        try std.testing.expectEqualStrings(if (set) |v| std.mem.span(v) else "1", if (platform.c.getenv("MCDMA_FABRIC_QOS")) |v| std.mem.span(v) else "(unset)");
    }
}

test "N-link direct configurations reject invalid ports and vias before opening verbs" {
    var link: Link = .{ .peer = 1, .device = "a+b+c+d", .via = "en4/192.0.2.2", .port = 7400, .name = "pair", .gid = -1 };
    try std.testing.expectEqual(@as(usize, 4), try memberCount(std.testing.allocator, link));
    link.ports = &.{ 7400, 7410, 7420, 7430 };
    link.peer_ports = &.{ 7500, 7510, 7520, 7530 };
    try std.testing.expectEqual(@as(usize, 4), try memberCount(std.testing.allocator, link));
    link.peer_ports = &.{ 7500, 7500, 7520, 7530 };
    try std.testing.expectError(error.InvalidConfig, memberCount(std.testing.allocator, link));
    link.peer_ports = &.{ 7500, 7510, 0, 7530 };
    try std.testing.expectError(error.InvalidConfig, memberCount(std.testing.allocator, link));
    link.peer_ports = &.{};
    link.ports = &.{};
    link.port = 65534;
    try std.testing.expectError(error.InvalidConfig, memberCount(std.testing.allocator, link));
    link.port = 7400;
    link.via = "en3+en4";
    try std.testing.expectError(error.InvalidConfig, memberCount(std.testing.allocator, link));
}

test {
    _ = @import("mcdma_test.zig");
}
