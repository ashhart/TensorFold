//! Host grammar work, separate from the planner's cost clock and scoped to one window or lane round.
const be = @import("backend.zig");
const ev = @import("events.zig");

pub const Sum = struct {
    windows: u64 = 0,
    fills: u64 = 0,
    window_ns: u64 = 0,
    fill_ns: u64 = 0,

    pub fn add(s: *Sum, x: Sum) void {
        inline for (.{ "windows", "fills", "window_ns", "fill_ns" }) |k| @field(s, k) += @field(x, k);
    }
};

pub const Round = struct {
    step: u64 = 0,
    total: Sum = .{},

    pub fn add(r: *Round, step: u64, x: Sum) void {
        if (r.step != step) r.* = .{ .step = step };
        r.total.add(x);
    }
};

pub const Profile = struct {
    clock: be.Clock,
    total: Sum = .{},

    pub fn now(p: *const Profile) i128 {
        return if (p.clock.vtable.host_ns) |read| read(p.clock.ptr) else 0;
    }

    pub fn since(p: *const Profile, began: i128) u64 {
        return @intCast(@max(0, p.now() - began));
    }
};

pub fn window(c: anytype, a: @import("std").mem.Allocator, tokens: []const u32, parents: []const i32, forced: bool, room: usize, profile: *Profile) !@import("grammar.zig").Window {
    const before = c.profile;
    c.profile = profile;
    const began = profile.now();
    defer {
        profile.total.windows += 1;
        profile.total.window_ns += profile.since(began);
        c.profile = before;
    }
    return c.window(a, tokens, parents, forced, room);
}

pub fn record(e: anytype, s: anytype, p: Profile) void {
    if (p.clock.vtable.host_ns == null) return;
    s.grammar_time.add(p.total);
    s.grammar_round.add(e.steps, p.total);
    e.grammar_round.add(e.steps, p.total);
    if (e.log) |log| _ = log.add(&.{ ev.f("ev", .{ .str = "grammar_window" }), ev.f("step", .{ .uint = e.steps }), ev.f("stream", .{ .str = s.id }), ev.f("windows", .{ .uint = p.total.windows }), ev.f("fills", .{ .uint = p.total.fills }), ev.f("window_ns", .{ .uint = p.total.window_ns }), ev.f("fill_ns", .{ .uint = p.total.fill_ns }) }) catch {};
}

pub fn report(e: anytype) void {
    if (e.grammar_round.step != e.steps or e.grammar_round.total.windows == 0) return;
    const x = e.grammar_round.total;
    if (e.log) |log| _ = log.add(&.{ ev.f("ev", .{ .str = "grammar_round" }), ev.f("step", .{ .uint = e.steps }), ev.f("windows", .{ .uint = x.windows }), ev.f("fills", .{ .uint = x.fills }), ev.f("window_ns", .{ .uint = x.window_ns }), ev.f("fill_ns", .{ .uint = x.fill_ns }) }) catch {};
}
