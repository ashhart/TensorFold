//! Initial reader progress must be visible before its first potentially long step begins.
const std = @import("std");
const api = @import("engine_api");
const Adapter = @import("slide_adapter.zig").Adapter;
const Fixture = struct {
    pub const Example = api.Example;
    pub const Learner = struct {
        pub const Progress = struct { phase: enum { read, fit, values, check, done }, at: usize, of: usize, loss: f32 };
        pub const Placed = api.Neuron;
        pub fn init(_: std.mem.Allocator, _: std.Io, _: void) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
        pub fn begin(_: *@This(), _: anytype) !void {}
        pub fn abort(_: *@This()) void {}
        pub fn progress(_: *@This()) ?Progress {
            return .{ .phase = .read, .at = 0, .of = 100, .loss = 0 };
        }
        pub fn placed(_: *@This(), buf: []Placed) []const Placed {
            return buf[0..0];
        }
        pub fn step(_: *@This()) struct {
            done: bool = true,
            changed: bool = false,
            stale: bool = false,
            report: ?union(enum) { learned: struct { recalled: bool, steps: u32, loss: f32 }, failed: []const u8 } = null,
        } {
            return .{};
        }
    };
};
const Seen = struct {
    reads: usize = 0,
    fn on(ctx: *anyopaque, event: *const api.LearnEvent) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        if (event.* == .progress and event.progress.phase == .reading and event.progress.at == 0 and event.progress.of == 100) s.reads += 1;
    }
};
test "lesson adapter: publish reading zero before the first long unit" {
    var a = Adapter(Fixture).init(std.testing.allocator, std.testing.io, {});
    defer a.deinit();
    var seen: Seen = .{};
    const hook = a.hook();
    try hook.begin(hook.ctx, &.{}, .{ .ctx = &seen, .event = Seen.on });
    try std.testing.expectEqual(@as(usize, 1), seen.reads);
}
