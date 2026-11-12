//! Synthetic request streams for the sessions write bench: one long agent, many short chats, and a mix with repeats.
const std = @import("std");
const tier = @import("sessions_writes_tier.zig");

pub const Workload = enum { agent, chats, mix };

const Range = struct { lo: u32, hi: u32 };

pub const Shape = struct {
    requests: u32,
    system: u32,
    first: Range,
    turn: Range,
    reply: Range,
    /// a conversation past this many tokens ends; the next request starts a new one
    cap: u32,
    /// chats: turns a conversation runs before the next begins
    turns: Range = .{ .lo = 0, .hi = 0 },
    /// mix: conversations held, the hot ones, and the odds (in 100) of a new one, of a hot one
    users: u32 = 1,
    hot: u32 = 1,
    fresh: u32 = 0,
    hot_odds: u32 = 0,
};

pub fn shape(w: Workload) Shape {
    return switch (w) {
        .agent => .{ .requests = 400, .system = 6144, .first = .{ .lo = 512, .hi = 4096 }, .turn = .{ .lo = 512, .hi = 4096 }, .reply = .{ .lo = 128, .hi = 1024 }, .cap = 131_072 },
        .chats => .{ .requests = 3000, .system = 1024, .first = .{ .lo = 256, .hi = 2048 }, .turn = .{ .lo = 128, .hi = 1024 }, .reply = .{ .lo = 128, .hi = 768 }, .cap = 1 << 30, .turns = .{ .lo = 1, .hi = 3 } },
        .mix => .{ .requests = 4000, .system = 6144, .first = .{ .lo = 512, .hi = 4096 }, .turn = .{ .lo = 256, .hi = 3072 }, .reply = .{ .lo = 128, .hi = 1024 }, .cap = 65_536, .users = 200, .hot = 16, .fresh = 15, .hot_odds = 70 },
    };
}

const Conv = struct {
    ids: std.ArrayList(i32) = .empty,
    turns: u32 = 0,
    left: u32 = 0,
};

/// Feeds one workload to a bench: counters at the half (steady state is the second half) and at the end.
pub fn run(gpa: std.mem.Allocator, b: *tier.Bench, w: Workload, sh: Shape, seed: u64) !struct { half: tier.Counts, end: tier.Counts } {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var next: i32 = 1;
    const system = try gpa.alloc(i32, sh.system);
    defer gpa.free(system);
    for (system) |*t| {
        t.* = next;
        next += 1;
    }
    const convs = try gpa.alloc(Conv, sh.users);
    defer {
        for (convs) |*c| c.ids.deinit(gpa);
        gpa.free(convs);
    }
    @memset(convs, .{});
    var half: tier.Counts = .{};
    var i: u32 = 0;
    while (i < sh.requests) : (i += 1) {
        if (i == sh.requests / 2) half = b.now();
        const u: u32 = switch (w) {
            .agent, .chats => 0,
            .mix => if (rnd.uintLessThan(u32, 100) < sh.hot_odds) rnd.uintLessThan(u32, sh.hot) else rnd.uintLessThan(u32, sh.users),
        };
        const c = &convs[u];
        const over = c.ids.items.len > sh.cap;
        const done = w == .chats and c.left == 0;
        const fresh = w == .mix and rnd.uintLessThan(u32, 100) < sh.fresh;
        if (c.ids.items.len == 0 or over or done or fresh) {
            c.ids.clearRetainingCapacity();
            try c.ids.appendSlice(gpa, system);
            try append(gpa, &c.ids, &next, pick(rnd, sh.first));
            c.turns = 0;
            if (w == .chats) c.left = pick(rnd, sh.turns);
        } else try append(gpa, &c.ids, &next, pick(rnd, sh.turn));
        const prompt = c.ids.items.len;
        try append(gpa, &c.ids, &next, pick(rnd, sh.reply));
        try b.request(c.ids.items, prompt);
        c.turns += 1;
        if (w == .chats) c.left -= 1;
    }
    return .{ .half = half, .end = b.now() };
}

fn pick(rnd: std.Random, r: Range) u32 {
    return rnd.intRangeAtMost(u32, r.lo, r.hi);
}

fn append(gpa: std.mem.Allocator, ids: *std.ArrayList(i32), next: *i32, n: u32) !void {
    try ids.ensureUnusedCapacity(gpa, n);
    for (0..n) |_| {
        ids.appendAssumeCapacity(next.*);
        next.* += 1;
    }
}
