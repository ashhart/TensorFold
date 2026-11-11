//! A connect-end link that outlives its daemon: `mcdma-rpcd connect` recreates its mailbox at every start.
const std = @import("std");
const platform = @import("sys");
const mailbox = @import("mailbox.zig");
const layout = @import("layout.zig");
const words = @import("words.zig");
const Mailbox = mailbox.Mailbox;
const Client = mailbox.Client;
const Error = mailbox.Error;

/// What `renew` found: the link as it was, or a new mailbox mapped because the daemon recreated it.
pub const Event = enum { same, reopened };

const min_wait = std.time.ns_per_s;
const max_wait = 30 * std.time.ns_per_s;

/// The mapped mailbox and its client, by name; the caller serialises calls, so a new mapping replaces the old in place.
pub const Relink = struct {
    name: []const u8,
    box: Mailbox,
    client: Client,
    /// no new mapping is tried before this time (`words.nowNs`): a refused one doubles the wait from 1 s to 30 s
    next_ns: u64 = 0,
    wait_ns: u64 = min_wait,

    /// Map the daemon's mailbox for link `name` in place (`r` must not move afterwards) and make its client.
    pub fn open(r: *Relink, name: []const u8, ws: words.Words, timeout_ns: u64) Error!void {
        r.* = .{ .name = name, .box = try Mailbox.openConnect(name), .client = undefined };
        errdefer r.box.close();
        r.client = try Client.init(&r.box, ws, timeout_ns);
    }

    pub fn close(r: *Relink) void {
        r.box.close();
    }

    /// A client for the link as it is at `now`: map the mailbox again if the daemon replaced it, else renew the client.
    pub fn renew(r: *Relink, now: u64) Error!Event {
        const moved = if (r.box.stillNamed(r.name)) |same| !same else |_| false;
        if (moved) {
            if (now < r.next_ns) return error.LinkDown;
            try r.remap(now);
            return .reopened;
        }
        try r.client.renew();
        return .same;
    }

    /// The new mailbox replaces the old one even if its link is not up yet; the client is made when it is.
    fn remap(r: *Relink, now: u64) Error!void {
        const fresh = Mailbox.openConnect(r.name) catch |err| {
            r.next_ns = now +| r.wait_ns;
            r.wait_ns = @min(2 * r.wait_ns, max_wait);
            return err;
        };
        var dead = r.box;
        defer dead.close();
        r.box = fresh;
        r.next_ns = 0;
        r.wait_ns = min_wait;
        const ws = r.client.words;
        const timeout_ns = r.client.timeout_ns;
        r.client = Client.init(&r.box, ws, timeout_ns) catch |err| {
            r.client = .{ .box = &r.box, .words = ws, .seq = 0, .generation = 0, .timeout_ns = timeout_ns, .poisoned = true };
            return err;
        };
    }
};

/// A shared memory object /mcdma-rpc.NAME made as the daemon makes it: owner-only, sized, link up at `generation`.
const Fake = struct {
    name: [:0]const u8,
    mem: []align(mailbox.page) u8,

    fn make(name: [:0]const u8, generation: u64, mode: c_uint) !Fake {
        var path: [64]u8 = undefined;
        const z = try std.mem.printSentinel(&path, "/mcdma-rpc.{s}", .{name}, 0);
        const flags: c_int = @bitCast(platform.c.O{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true });
        const fd = platform.c.shm_open(z.ptr, flags, mode);
        try std.testing.expect(fd >= 0);
        defer _ = platform.c.close(fd);
        try std.testing.expectEqual(@as(c_int, 0), platform.c.ftruncate(fd, 2 * layout.segment));
        const mem = try std.posix.mmap(null, 2 * layout.segment, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        std.mem.writeInt(u64, mem[layout.sizes..][0..8], layout.segment, .little);
        std.mem.writeInt(u64, mem[layout.sizes + 8 ..][0..8], layout.segment, .little);
        const f: Fake = .{ .name = name, .mem = mem };
        if (generation != 0) f.up(generation);
        return f;
    }

    fn up(f: Fake, generation: u64) void {
        std.mem.writeInt(u64, f.mem[layout.link_up..][0..8], 1, .little);
        std.mem.writeInt(u64, f.mem[layout.generation..][0..8], generation, .little);
    }

    /// The daemon went away: its object is unlinked, though mappings of it live on.
    fn gone(f: Fake) void {
        unlink(f.name);
        std.posix.munmap(f.mem);
    }

    fn unlink(name: [:0]const u8) void {
        var path: [64]u8 = undefined;
        const z = std.mem.printSentinel(&path, "/mcdma-rpc.{s}", .{name}, 0) catch unreachable;
        _ = platform.c.shm_unlink(z.ptr);
    }
};

test "a link follows its daemon's new mailbox, waits out a refused one, and adopts one whose link is not up yet" {
    const name = "tf-relink-test";
    Fake.unlink(name); // an earlier failed run's object
    defer Fake.unlink(name);
    var a = try Fake.make(name, 1, 0o600);
    var r: Relink = undefined;
    try r.open(name, words.native, std.time.ns_per_s);
    defer r.close();
    r.box.requestPayload()[0] = 0x5a;
    try std.testing.expectEqual(Event.same, try r.renew(0));
    try std.testing.expectEqual(@as(u8, 0x5a), r.box.requestPayload()[0]); // the probe puts back what it borrowed
    // the daemon restarts: the old object is unlinked while this side still maps it, and a new one is created
    a.gone();
    var b = try Fake.make(name, 2, 0o600);
    try std.testing.expect(r.client.alive()); // nothing in the stale mapping says it is dead
    try std.testing.expectEqual(Event.reopened, try r.renew(100));
    try std.testing.expectEqual(@as(u64, 2), r.client.generation);
    try std.testing.expect(r.client.box == &r.box and try r.box.stillNamed(name));
    try std.testing.expectEqual(Event.same, try r.renew(101));
    // a mailbox that is not owner-only is refused, and not tried again before the wait is over
    b.gone();
    var c = try Fake.make(name, 1, 0o644);
    try std.testing.expectError(error.UnsafeMailbox, r.renew(1000));
    try std.testing.expectError(error.LinkDown, r.renew(1000 + min_wait / 2));
    c.gone();
    // the daemon came back but its link is not up yet: the mapping is kept, the client is made once it is
    var d = try Fake.make(name, 0, 0o600);
    try std.testing.expectError(error.LinkDown, r.renew(1000 + min_wait));
    try std.testing.expectError(error.LinkDown, r.renew(1000 + min_wait + 1));
    d.up(7);
    try std.testing.expectEqual(Event.same, try r.renew(1000 + min_wait + 2));
    try std.testing.expectEqual(@as(u64, 7), r.client.generation);
    d.gone();
}
