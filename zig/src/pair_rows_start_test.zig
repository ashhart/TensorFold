//! The pair's start-up order on a fake kit: a failure at any step releases what came before it, each object once.
const std = @import("std");
const start = @import("core/pair_rows_start.zig").start;
const Owner = @import("families/flashnext/owned.zig").Owner;
const t = std.testing;

/// Release order across one run: the window's buffer must go before the endpoint frees the memory under it.
var clock: u32 = 0;

/// One made object: whether it was made, how often it was released, and when it last was.
const Token = struct {
    made: bool = false,
    released: u32 = 0,
    at: u32 = 0,

    fn release(k: *Token) void {
        k.released += 1;
        clock += 1;
        k.at = clock;
    }
};

/// What `Owner` needs of Metal: an id it releases, and a queue type it never sees here.
const Metal = struct {
    pub const objc = struct {
        pub const Id = *Token;
        pub fn release(id: Id) void {
            id.release();
        }
    };
    pub const Queue = struct {
        pub fn commandBuffer(_: Queue) CommandBuffer {
            return .{};
        }
        pub fn deinit(_: Queue) void {}
    };
    pub const CommandBuffer = struct {
        pub fn commit(_: CommandBuffer) void {}
        pub fn wait(_: CommandBuffer) void {}
    };
};

/// A Metal object, a fence or an event: its own deinit, or the owner's release, marks its token.
const Obj = struct {
    id: *Token,
    pub fn deinit(o: Obj) void {
        o.id.release();
    }
};

/// The link: its window and its close.
const Ep = struct {
    id: *Token,
    win: [64]u8 = @splat(0),
    pub fn deinit(e: *Ep) void {
        e.id.release();
    }
    pub fn rdma(e: *Ep) Rd {
        return .{ .win = &e.win };
    }
};
const Rd = struct {
    win: []u8,
    pub fn window(r: Rd) []u8 {
        return r.win;
    }
};

/// Every step Pair.init takes, in order; `hello` and `spawn` make nothing to release.
const Step = enum { endpoint, library, fence, posted, landed, window, stage, gather, argmax, pick, hello, spawn };
const steps = @typeInfo(Step).@"enum".field_names.len;
const objects = @backingInt(Step.hello);

/// The pair as start leaves it, with Pair.deinit's order.
const FakePair = struct {
    ep: *Ep,
    fence: Obj,
    posted: Obj,
    landed: Obj,
    bytes: []u32,
    owned: Owner(Metal) = .{},
    thread: ?u32 = null,

    fn deinit(p: *FakePair, gpa: std.mem.Allocator) void {
        p.posted.deinit();
        p.landed.deinit();
        p.fence.deinit();
        p.owned.deinit();
        p.ep.deinit();
        gpa.free(p.bytes);
        gpa.destroy(p);
    }
};

const Kit = struct {
    pub const Made = FakePair;
    pub const Owned = Owner(Metal);
    fail: ?Step = null,
    tokens: [steps]Token = @splat(.{}),
    ep: Ep = undefined,
    events: u32 = 0,

    /// Step `s`'s token, or the injected failure: a failed step makes nothing.
    fn take(k: *Kit, s: Step) !*Token {
        if (k.fail) |f| if (f == s) return error.Injected;
        const id = &k.tokens[@backingInt(s)];
        id.made = true;
        return id;
    }
    pub fn endpoint(k: *Kit) !*Ep {
        k.ep = .{ .id = try k.take(.endpoint) };
        return &k.ep;
    }
    pub fn library(k: *Kit) !Obj {
        return .{ .id = try k.take(.library) };
    }
    pub fn fence(k: *Kit) !Obj {
        return .{ .id = try k.take(.fence) };
    }
    pub fn event(k: *Kit) !Obj {
        k.events += 1;
        return .{ .id = try k.take(if (k.events == 1) .posted else .landed) };
    }
    pub fn window(k: *Kit, _: []u8) !Obj {
        return .{ .id = try k.take(.window) };
    }
    pub fn pipeline(k: *Kit, _: Obj, name: []const u8) !Obj {
        const s = std.meta.stringToEnum(Step, name["pair_".len..]) orelse return error.UnknownPipeline;
        return .{ .id = try k.take(s) };
    }
    pub fn assemble(_: *Kit, p: *FakePair, o: anytype) void {
        p.* = .{ .ep = o.ep, .fence = o.fence, .posted = o.posted, .landed = o.landed, .bytes = o.bytes };
    }
    pub fn hello(k: *Kit, _: *FakePair) !void {
        _ = try k.take(.hello);
    }
    pub fn spawn(k: *Kit, _: *FakePair) !u32 {
        _ = try k.take(.spawn);
        return 1;
    }
};

/// After a failed start: every object made was released once, none twice, and the window's buffer before the link.
fn expectReleased(k: *const Kit) !void {
    for (k.tokens[0..objects]) |id| try t.expectEqual(@as(u32, @intFromBool(id.made)), id.released);
    const ep = k.tokens[@backingInt(Step.endpoint)];
    const win = k.tokens[@backingInt(Step.window)];
    if (win.made) try t.expect(win.at < ep.at);
}

test "pair start: a failure at each step releases every object made before it, once" {
    for (0..steps) |i| {
        const s: Step = @fromBackingInt(@intCast(i));
        var k: Kit = .{ .fail = s };
        try t.expectError(error.Injected, start(Kit, &k, t.allocator, 8));
        // the step that failed made nothing, and every step before it did
        for (k.tokens, 0..) |id, j| try t.expectEqual(j < i, id.made);
        try expectReleased(&k);
    }
}

test "pair start: a refused allocation at each point releases every object made before it, once" {
    var refusals: usize = 0;
    var i: usize = 0;
    while (true) : (i += 1) {
        var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = i });
        var k: Kit = .{};
        const p = start(Kit, &k, failing.allocator(), 8) catch |err| {
            try t.expect(err == error.OutOfMemory);
            try expectReleased(&k);
            refusals += 1;
            continue;
        };
        // every allocation succeeded: the pair holds every object, and its own deinit releases each once
        for (k.tokens[0..objects], 0..) |id, j| try t.expect(id.made and id.released == @intFromBool(j == @backingInt(Step.library)));
        p.deinit(failing.allocator());
        for (k.tokens[0..objects]) |id| try t.expectEqual(@as(u32, 1), id.released);
        break;
    }
    // the pair, the owner's list and the exchange table: at least three allocations were refused in turn
    try t.expect(refusals >= 3);
}

test "pair start: a started pair owns what it made, and its deinit releases each once" {
    var k: Kit = .{};
    const p = try start(Kit, &k, t.allocator, 8);
    try t.expectEqual(@as(?u32, 1), p.thread);
    // only the library is gone (its pipelines keep what they need); the window's buffer and the pipes are the pair's
    for (k.tokens[0..objects], 0..) |id, j| try t.expectEqual(@as(u32, @intFromBool(j == @backingInt(Step.library))), id.released);
    try t.expectEqual(@as(usize, 5), p.owned.entries.items.len);
    p.deinit(t.allocator);
    for (k.tokens[0..objects]) |id| try t.expectEqual(@as(u32, 1), id.released);
    const ep = k.tokens[@backingInt(Step.endpoint)];
    try t.expect(k.tokens[@backingInt(Step.window)].at < ep.at);
}
