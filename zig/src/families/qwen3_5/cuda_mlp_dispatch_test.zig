//! Dense prompt claims preserve arithmetic ownership, output projection and tap ordering.
const std = @import("std");
const dispatch = @import("cuda_mlp_dispatch.zig");
const t = std.testing;

const Fake = struct {
    const Model = struct { native_requested: bool = false, fast8: ?u8 = null };
    const Add = struct { ctx: *anyopaque, f: *const fn (*anyopaque, usize, u64, u64, usize, u64) anyerror!void };
    const Event = struct { kind: enum { down, pocket, learned, tap }, rows: usize };
    const Pockets = struct {
        owner: ?*Fake = null,
        pub fn add(p: Pockets, _: u8, layer: usize, x: u64, r: u64, rows: usize, taught: u64) !void {
            const e = p.owner.?;
            try t.expectEqual(@as(usize, 4), layer);
            try t.expectEqual(e.x, x);
            try t.expectEqual(e.r, r);
            try t.expectEqual(e.taught, taught);
            e.record(.pocket, rows);
        }
    };
    mo: ?Model = null,
    prompt_many: bool = true,
    learning: ?u8 = null,
    act: u64 = 80,
    x: u64 = 10,
    r: u64 = 20,
    h: u64 = 30,
    stream: u8 = 0,
    taught: u64 = 7,
    mlp: ?Add = null,
    pockets: Pockets = .{},
    g: struct { hidden: usize = 5120 } = .{},
    taps: [2]usize = .{ 4, 6 },
    n_taps: usize = 2,
    fns: struct { tap: u8 = 9 } = .{},
    fused: bool = false,
    provider_calls: usize = 0,
    events: [8]Event = undefined,
    count: usize = 0,

    fn record(e: *Fake, kind: @FieldType(Event, "kind"), rows: usize) void {
        e.events[e.count] = .{ .kind = kind, .rows = rows };
        e.count += 1;
    }
    pub fn projAgain(e: *Fake, down: u8, input: u64, output: u64, rows: usize, prompt: bool) !void {
        try t.expectEqual(@as(u8, 11), down);
        try t.expectEqual(e.act, input);
        try t.expectEqual(@as(u64, 7000), output);
        try t.expect(prompt);
        e.record(.down, rows);
    }
    pub fn run(e: *Fake, function: u8, grid: anytype, block: anytype, args: anytype) !void {
        try t.expectEqual(e.fns.tap, function);
        try t.expectEqual(@as(usize, 512), grid[0]);
        try t.expectEqual(@as(usize, 3), grid[1]);
        try t.expectEqual(@as(comptime_int, 256), block[0]);
        try t.expectEqual(e.h, args[0]);
        try t.expectEqual(e.r, args[1]);
        try t.expectEqual(@as(u64, 100), args[2]);
        try t.expectEqual(@as(u64, 200), args[3]);
        try t.expectEqual(@as(c_int, 0), args[4]);
        try t.expectEqual(@as(c_int, 5120), args[5]);
        try t.expectEqual(@as(u64, 300), args[6]);
        e.record(.tap, grid[0]);
    }
};

const Provider = struct {
    pub fn swiglu(e: *Fake, gate: u8, up: u8, input: u64, output: u64, rows: usize) !bool {
        try t.expectEqual(@as(u8, 1), gate);
        try t.expectEqual(@as(u8, 2), up);
        try t.expectEqual(@as(u64, 50), input);
        try t.expectEqual(e.act, output);
        try t.expect(rows > 0);
        e.provider_calls += 1;
        return e.fused;
    }
};

test "dense prompt dispatch: activation ownership cannot fall through to paired projection arithmetic" {
    for ([_]bool{ false, true }) |native| {
        for ([_]bool{ false, true }) |fast| {
            for ([_]bool{ false, true }) |fused| {
                for ([_]usize{ 1, 16, 127, 128, 129, 2048, 7680 }) |rows| {
                    var e: Fake = .{ .mo = .{ .native_requested = native, .fast8 = if (fast) 1 else null }, .fused = fused };
                    const result = try dispatch.claimWith(Provider, &e, @as(u8, 1), @as(u8, 2), 50, rows, true);
                    try t.expectEqual(if (fused) dispatch.Claim.act_ready else .none, result);
                    try t.expectEqual(@as(usize, 1), e.provider_calls);
                    try t.expectEqual(!native and !fast, e.prompt_many);
                }
            }
        }
    }
}

test "dense prompt dispatch: decode and learner captures never claim the activation" {
    var e: Fake = .{ .fused = true };
    try t.expectEqual(dispatch.Claim.none, try dispatch.claimWith(Provider, &e, @as(u8, 1), @as(u8, 2), 50, 16, false));
    e.learning = 1;
    try t.expectEqual(dispatch.Claim.none, try dispatch.claimWith(Provider, &e, @as(u8, 1), @as(u8, 2), 50, 129, true));
    try t.expectEqual(@as(usize, 0), e.provider_calls);
}

test "dense prompt dispatch: a completed activation projects down and taps exactly once in order" {
    var e: Fake = .{};
    e.pockets.owner = &e;
    try dispatch.finishDense(&e, .{ .down = @as(u8, 11) }, 4, 512, 129, true, 7000);
    try dispatch.taps(&e, true, 4, 512, 100, 200, 300);
    try t.expectEqual(@as(usize, 3), e.count);
    try t.expectEqual(.down, e.events[0].kind);
    try t.expectEqual(@as(usize, 129), e.events[0].rows);
    try t.expectEqual(.pocket, e.events[1].kind);
    try t.expectEqual(.tap, e.events[2].kind);
}

fn learned(ctx: *anyopaque, layer: usize, x: u64, r: u64, rows: usize, taught: u64) !void {
    const e: *Fake = @ptrCast(@alignCast(ctx));
    try t.expectEqual(@as(usize, 4), layer);
    try t.expectEqual(e.x, x);
    try t.expectEqual(e.r, r);
    try t.expectEqual(e.taught, taught);
    e.record(.learned, rows);
}

test "dense prompt dispatch: learner addition replaces pocket contribution without adding a second down" {
    var e: Fake = .{};
    e.pockets.owner = &e;
    e.mlp = .{ .ctx = &e, .f = learned };
    try dispatch.finishDense(&e, .{ .down = @as(u8, 11) }, 4, 512, 129, true, 7000);
    try dispatch.taps(&e, false, 4, 512, 100, 200, 300);
    try t.expectEqual(@as(usize, 2), e.count);
    try t.expectEqual(.down, e.events[0].kind);
    try t.expectEqual(.learned, e.events[1].kind);
}
