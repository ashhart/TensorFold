//! Registered windows over first-party MCDMA: one progress context per port, with staged sources held through flush.
const std = @import("std");
const abi = @import("mcdma_abi.zig");
const rma = @import("rdma.zig");
const words = @import("words.zig");

pub const alignment = 16384;
pub const max_ranks = 64;
pub const Error = abi.Error || std.mem.Allocator.Error || error{ InvalidConfig, OpenFailed, ConnectFailed, UnsupportedLink, WindowMismatch, ThreadFailed };

pub const Link = struct { peer: u32, device: [:0]const u8, via: [:0]const u8, port: u16, peer_port: u16 = 0, name: [:0]const u8, gid: c_int = 1 };
pub const Config = struct { rank: u32, ranks: u32, window_bytes: usize, staging_bytes: usize, links: []const Link, timeout_ns: u64 = 10 * std.time.ns_per_s, connect_timeout_ns: u64 = 300 * std.time.ns_per_s };

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
    mutex: Lock = .{},
};

pub const Endpoint = struct {
    gpa: std.mem.Allocator,
    library: abi.Library,
    config: Config,
    memory: []align(alignment) u8,
    ports: []Port,
    deadline_ns: u64 = 0,

    /// A heap-owned endpoint keeps all thread and library references stable until deinit closes every port.
    pub fn create(gpa: std.mem.Allocator, library_path: [:0]const u8, config: Config) Error!*Endpoint {
        if (config.ranks == 0 or config.ranks > max_ranks or config.rank >= config.ranks or config.links.len >= config.ranks or config.timeout_ns == 0 or config.connect_timeout_ns == 0 or config.window_bytes == 0 or config.staging_bytes == 0 or config.window_bytes % alignment != 0 or config.staging_bytes % alignment != 0) return error.InvalidConfig;
        for (config.links, 0..) |link, i| {
            if (link.peer >= config.ranks or link.peer == config.rank or link.port == 0 or link.device.len == 0 or link.via.len == 0 or link.name.len == 0 or link.name.len > 20) return error.InvalidConfig;
            for (config.links[0..i]) |old| if (old.peer == link.peer or std.mem.eql(u8, old.device, link.device)) return error.InvalidConfig;
        }
        const stage = std.math.mul(usize, config.ranks, config.staging_bytes) catch return error.InvalidConfig;
        const bytes = std.math.add(usize, config.window_bytes, stage) catch return error.InvalidConfig;
        const self = try gpa.create(Endpoint);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .library = try abi.Library.open(library_path), .config = config, .memory = undefined, .ports = undefined };
        errdefer self.library.close();
        self.memory = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), bytes);
        errdefer gpa.free(self.memory);
        @memset(self.memory, 0);
        self.ports = try gpa.alloc(Port, config.links.len);
        errdefer gpa.free(self.ports);
        for (self.ports, config.links) |*port, spec| port.* = .{ .spec = spec };
        errdefer self.closePorts();
        for (self.ports) |*port| {
            if (self.library.api.open(port.spec.device.ptr, port.spec.gid, 4096, self.memory.ptr, self.memory.len, -1, 0, abi.progress_thread, &port.context) != 0 or port.context == null) return error.OpenFailed;
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
        for (self.ports) |port| {
            if (port.status != 0 or port.peer == null) return error.ConnectFailed;
            if (self.library.api.link(port.peer.?) != abi.thunderbolt) return error.UnsupportedLink;
            if (self.library.api.peer_length(port.peer.?) != self.memory.len) return error.WindowMismatch;
        }
        return self;
    }

    fn connect(self: *Endpoint, i: usize) void {
        const port = &self.ports[i];
        const spec = port.spec;
        port.status = self.library.api.connect(port.context.?, spec.via.ptr, spec.port, spec.peer_port, spec.name.ptr, self.config.connect_timeout_ns, &port.peer);
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
        self.library.close();
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
        try check(self.library.write_signal.?(port.peer.?, local, offset, len, flag, value));
        port.pending = true;
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

    fn flushPort(self: *Endpoint, port: *Port) rma.Error!void {
        if (!port.pending) return;
        const timeout = try self.left();
        try check(self.library.api.flush(port.peer.?, timeout));
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
        try check(self.library.api.write(port.peer.?, local, offset, total));
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
            try check(self.library.api.write(port.peer.?, local, offset + done, take));
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
        if (peer == self.config.rank or total + room > self.config.staging_bytes) {
            try write2(ptr, peer, offset, head, body);
            return signal(ptr, peer, flag, value);
        }
        if (!self.in(offset, total) or !self.in(flag, 8)) return error.OutOfBounds;
        if (flag % 8 != 0) return error.Unaligned;
        const port = try self.portFor(peer);
        try port.mutex.lock(try self.end());
        defer port.mutex.unlock();
        if (total + room > self.config.staging_bytes - port.used) try self.flushPort(port);
        const local = self.config.window_bytes + peer * self.config.staging_bytes + port.used + room;
        @memcpy(self.memory[local..][0..head.len], head);
        @memcpy(self.memory[local + head.len ..][0..body.len], body);
        try check(self.library.write_signal.?(port.peer.?, local, offset, total, flag, value));
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
        try check(self.library.api.signal(port.peer.?, offset, value));
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
        return if (peer == of(ptr).config.rank) .local else .tb5;
    }
};

pub const Region = Endpoint;
