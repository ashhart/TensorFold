//! One mcdma-rpcd mailbox (protocol 1): the mapped halves, a client's ordered call and a service's request and reply.
const std = @import("std");
const builtin = @import("builtin");
const layout = @import("layout.zig");
const words = @import("words.zig");
const Words = words.Words;

pub const page = std.heap.page_size_min;

pub const Error = error{
    InvalidName,
    NoMailbox,
    UnsafeMailbox,
    BadSizes,
    LinkDown,
    LinkLost,
    OrphanedRequest,
    Poisoned,
    TooLarge,
    Refused,
    BadReply,
    Timeout,
    SequenceExhausted,
};

/// The mapped mailbox: a request half of `req` bytes, then a reply half of `rep` bytes.
pub const Mailbox = struct {
    mem: []align(page) u8,
    req: usize,
    rep: usize,
    mapped: bool = false,

    /// Adopt memory whose sizes word (request +256) the daemon or a fake already wrote.
    pub fn fromMemory(mem: []align(page) u8) Error!Mailbox {
        if (mem.len < layout.sizes + 16) return error.BadSizes;
        const req = std.mem.readInt(u64, mem[layout.sizes..][0..8], .little);
        const rep = std.mem.readInt(u64, mem[layout.sizes + 8 ..][0..8], .little);
        if (!layout.validHalf(req) or !layout.validHalf(rep) or req + rep > mem.len) return error.BadSizes;
        return .{ .mem = mem, .req = @intCast(req), .rep = @intCast(rep) };
    }

    /// The connect end's mailbox: the POSIX shared memory object /mcdma-rpc.NAME on either OS.
    pub fn openConnect(name: []const u8) Error!Mailbox {
        if (!layout.validName(name)) return error.InvalidName;
        return openShm(name);
    }

    fn openShm(name: []const u8) Error!Mailbox {
        var path: [64]u8 = undefined;
        const z = std.mem.printSentinel(&path, "/mcdma-rpc.{s}", .{name}, 0) catch return error.InvalidName;
        const flags: c_int = @bitCast(std.c.O{ .ACCMODE = .RDWR });
        const fd = if (builtin.os.tag == .macos) std.c.shm_open(z.ptr, flags, @as(c_uint, 0)) else std.c.shm_open(z.ptr, flags, 0);
        if (fd < 0) return error.NoMailbox;
        defer _ = std.c.close(fd);
        return mapFd(fd);
    }

    /// The listen end's mailbox: POSIX shared memory on a Mac, the file /dev/shm/mcdma-rpc.NAME on Linux; MCDMA_RPC_BOX_DIR names a test directory.
    pub fn openListen(name: []const u8) Error!Mailbox {
        if (!layout.validName(name)) return error.InvalidName;
        const env = std.c.getenv("MCDMA_RPC_BOX_DIR");
        if (env == null and builtin.os.tag == .macos) return openShm(name);
        const dir = if (env) |d| std.mem.span(d) else "/dev/shm";
        var path: [256]u8 = undefined;
        const z = std.mem.printSentinel(&path, "{s}/mcdma-rpc.{s}", .{ dir, name }, 0) catch return error.InvalidName;
        const fd = std.c.open(z.ptr, .{ .ACCMODE = .RDWR, .NOFOLLOW = true, .CLOEXEC = true });
        if (fd < 0) return error.NoMailbox;
        defer _ = std.c.close(fd);
        return mapFd(fd);
    }

    /// Map a mailbox the daemon created; it must be this user's and owner-only, as the daemons make them.
    fn mapFd(fd: std.c.fd_t) Error!Mailbox {
        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0) return error.NoMailbox;
        if (st.uid != std.c.getuid() or st.mode & 0o077 != 0) return error.UnsafeMailbox;
        const size: usize = @intCast(st.size);
        if (size < 2 * layout.segment) return error.BadSizes;
        const mem = std.posix.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0) catch return error.NoMailbox;
        var box = fromMemory(mem) catch |err| {
            std.posix.munmap(mem);
            return err;
        };
        box.mapped = true;
        return box;
    }

    pub fn close(box: *Mailbox) void {
        if (box.mapped) std.posix.munmap(box.mem);
        box.* = undefined;
    }

    pub fn wordAt(box: *const Mailbox, offset: usize) *u64 {
        return @ptrCast(@alignCast(box.mem.ptr + offset));
    }

    pub fn requestPayload(box: *const Mailbox) []u8 {
        return box.mem[layout.ctrl..box.req];
    }

    pub fn replyPayload(box: *const Mailbox) []u8 {
        return box.mem[box.req + layout.ctrl .. box.req + box.rep];
    }

    /// The reply half from its first byte, page-aligned, for a Metal buffer over what the NIC writes.
    pub fn replyHalf(box: *const Mailbox) []align(page) u8 {
        return @alignCast(box.mem[box.req .. box.req + box.rep]);
    }
};

/// The connect end's client: one call at a time, poisoned for good once a published call fails.
pub const Client = struct {
    box: *const Mailbox,
    words: Words,
    seq: u32,
    generation: u64,
    timeout_ns: u64,
    poisoned: bool = false,

    pub fn init(box: *const Mailbox, ws: Words, timeout_ns: u64) Error!Client {
        const gen = words.load(box.wordAt(layout.generation));
        if (words.load(box.wordAt(layout.link_up)) != 1 or gen == 0) return error.LinkDown;
        const old = layout.seqOf(words.load(box.wordAt(layout.request_word)));
        const done = layout.seqOf(words.load(box.wordAt(box.req + layout.done_word)));
        if (old != 0 and old != done) return error.OrphanedRequest;
        return .{ .box = box, .words = ws, .seq = @max(old, done), .generation = gen, .timeout_ns = timeout_ns };
    }

    fn alive(c: *const Client) bool {
        return words.load(c.box.wordAt(layout.link_up)) == 1 and words.load(c.box.wordAt(layout.generation)) == c.generation;
    }

    /// Stage `payload`, publish its word and wait for the done word; the reply is valid until the next call.
    pub fn call(c: *Client, payload: []const u8) Error![]const u8 {
        if (c.poisoned) return error.Poisoned;
        if (payload.len == 0 or payload.len > c.box.req - layout.ctrl) return error.TooLarge;
        if (!c.alive()) return error.LinkDown;
        if (c.seq == std.math.maxInt(u32)) {
            c.poisoned = true;
            return error.SequenceExhausted;
        }
        c.seq += 1;
        @memcpy(c.box.requestPayload()[0..payload.len], payload);
        if (!c.alive()) return error.LinkDown;
        c.words.store(c.box.wordAt(layout.request_word), layout.word(c.seq, @intCast(payload.len)));
        return c.awaitReply() catch |err| {
            c.poisoned = err != error.Refused;
            return err;
        };
    }

    fn awaitReply(c: *Client) Error![]const u8 {
        const deadline = words.nowNs() + c.timeout_ns;
        const done = c.box.wordAt(c.box.req + layout.done_word);
        while (true) {
            if (!c.alive()) return error.LinkLost;
            const now = words.nowNs();
            if (now >= deadline) return error.Timeout;
            const w = c.words.wait(done, c.seq, true, 50_000, @min(deadline - now, 5 * std.time.ns_per_ms));
            if (w == 0) continue;
            if (!c.alive()) return error.LinkLost;
            const len = layout.lenOf(w);
            if (len == 0) return error.Refused;
            if (len > c.box.rep - layout.ctrl) return error.BadReply;
            return c.box.replyPayload()[0..len];
        }
    }
};

/// A request as it landed in the service's request half; the payload is valid until the service replies.
pub const Request = struct { seq: u32, payload: []const u8 };

/// The listen end's service: requests land in the request half, replies leave from the reply half.
pub const Service = struct {
    box: *const Mailbox,
    words: Words,
    last: u32,

    /// A request left from an earlier service is not for this one.
    pub fn init(box: *const Mailbox, ws: Words) Service {
        return .{ .box = box, .words = ws, .last = layout.seqOf(words.load(box.wordAt(layout.request_word))) };
    }

    pub fn next(s: *Service, timeout_ns: u64) Error!?Request {
        const w = s.words.wait(s.box.wordAt(layout.request_word), s.last, false, 100_000, timeout_ns);
        if (w == 0) return null;
        const len = layout.lenOf(w);
        if (len > s.box.req - layout.ctrl) return error.TooLarge;
        s.last = layout.seqOf(w);
        return .{ .seq = s.last, .payload = s.box.requestPayload()[0..len] };
    }

    pub fn replyArea(s: *const Service) []u8 {
        return s.box.replyPayload();
    }

    /// Hand the first `len` bytes of the reply area to the daemon as the answer to `seq`.
    pub fn publish(s: *const Service, seq: u32, len: usize) Error!void {
        if (len > s.box.rep - layout.ctrl) return error.TooLarge;
        s.words.store(s.box.wordAt(s.box.req + layout.staged_word), layout.word(seq, @intCast(len)));
    }
};

/// Page-aligned zeroed memory for a fake mailbox, sizes word written as a daemon writes it.
pub fn allocFake(req: usize, rep: usize) ![]align(page) u8 {
    const mem = try std.posix.mmap(null, req + rep, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    std.mem.writeInt(u64, mem[layout.sizes..][0..8], req, .little);
    std.mem.writeInt(u64, mem[layout.sizes + 8 ..][0..8], rep, .little);
    return mem;
}

test "a listen mailbox on this Mac is POSIX shared memory, as MCDMA 2.0.0's Mac listen end makes it" {
    if (builtin.os.tag != .macos or std.c.getenv("MCDMA_RPC_BOX_DIR") != null) return error.SkipZigTest;
    const name = "/mcdma-rpc.tf-zig-shm-test";
    const flags: c_int = @bitCast(std.c.O{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true });
    const fd = std.c.shm_open(name, flags, @as(c_uint, 0o600));
    try std.testing.expect(fd >= 0);
    defer _ = std.c.shm_unlink(name);
    defer _ = std.c.close(fd);
    try std.testing.expectEqual(@as(c_int, 0), std.c.ftruncate(fd, 2 * layout.segment));
    const mem = try std.posix.mmap(null, 2 * layout.segment, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
    std.mem.writeInt(u64, mem[layout.sizes..][0..8], layout.segment, .little);
    std.mem.writeInt(u64, mem[layout.sizes + 8 ..][0..8], layout.segment, .little);
    std.posix.munmap(mem);
    var box = try Mailbox.openListen("tf-zig-shm-test");
    defer box.close();
    try std.testing.expectEqual(layout.segment, box.rep);
}

test "a mailbox reads its halves from the sizes word" {
    const mem = try allocFake(layout.segment, 2 * layout.segment);
    defer std.posix.munmap(mem);
    const box = try Mailbox.fromMemory(mem);
    try std.testing.expectEqual(layout.segment, box.req);
    try std.testing.expectEqual(2 * layout.segment, box.rep);
    try std.testing.expectEqual(layout.segment - layout.ctrl, box.requestPayload().len);
    try std.testing.expectEqual(2 * layout.segment - layout.ctrl, box.replyPayload().len);
    std.mem.writeInt(u64, mem[layout.sizes..][0..8], 3 << 20, .little);
    try std.testing.expectError(error.BadSizes, Mailbox.fromMemory(mem));
}

test "a client refuses a link that is down or has an orphaned request" {
    const mem = try allocFake(layout.segment, layout.segment);
    defer std.posix.munmap(mem);
    const box = try Mailbox.fromMemory(mem);
    try std.testing.expectError(error.LinkDown, Client.init(&box, words.native, std.time.ns_per_s));
    box.wordAt(layout.link_up).* = 1;
    box.wordAt(layout.generation).* = 1;
    box.wordAt(layout.request_word).* = layout.word(4, 10);
    try std.testing.expectError(error.OrphanedRequest, Client.init(&box, words.native, std.time.ns_per_s));
    box.wordAt(box.req + layout.done_word).* = layout.word(4, 3);
    const c = try Client.init(&box, words.native, std.time.ns_per_s);
    try std.testing.expectEqual(@as(u32, 4), c.seq);
}

test "a service skips the request an earlier service left and stages its reply word" {
    const mem = try allocFake(layout.segment, layout.segment);
    defer std.posix.munmap(mem);
    const box = try Mailbox.fromMemory(mem);
    box.wordAt(layout.request_word).* = layout.word(3, 4);
    var s = Service.init(&box, words.native);
    try std.testing.expectEqual(@as(?Request, null), try s.next(1_000_000));
    @memcpy(box.requestPayload()[0..5], "hello");
    words.native.store(box.wordAt(layout.request_word), layout.word(4, 5));
    const r = (try s.next(std.time.ns_per_s)).?;
    try std.testing.expectEqual(@as(u32, 4), r.seq);
    try std.testing.expectEqualStrings("hello", r.payload);
    @memcpy(s.replyArea()[0..5], "world");
    try s.publish(r.seq, 5);
    try std.testing.expectEqual(layout.word(4, 5), box.wordAt(box.req + layout.staged_word).*);
}
