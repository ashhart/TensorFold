//! Copied proposals exercise the serving lane adapter and actual drawer; they do not claim a learned MTP head.
const std = @import("std");
pub fn run(comptime E: type, comptime D: type, comptime L: type, a: std.mem.Allocator, io: std.Io, e: *E, d: *D, ids: []const u32) !void {
    var adapter = try @import("lanes_adapter.zig").Adapter(E, D, L).init(a, e, d);
    defer adapter.deinitDrained();
    var config = try L.Config.init(a, adapter.facts(), 4, 3);
    defer config.deinit(a);
    config.timed = false;
    var clock: L.backend.WallClock = .{ .io = io };
    var core = L.Engine.init(a, &config, adapter.backend(), clock.clock());
    defer core.deinit();
    const Proposal = struct {
        fn give(_: *anyopaque, _: []const u32, max: i64) ![]const u32 {
            const values = &[_]u32{ 0, 7, 0 };
            return values[0..@intCast(@max(0, @min(max, values.len)))];
        }
        fn matched(_: *anyopaque) i64 {
            return 32;
        }
    };
    var id: u8 = 0;
    const proposer: L.proposer.Proposer = .{ .ptr = &id, .vtable = &.{ .propose = Proposal.give, .last_match = Proposal.matched, .priced = true } };
    var drafted = try L.Stream.init(a, .{ .id = "drafted", .prompt = ids, .max_new = 8, .drafts = true, .proposer = proposer });
    defer drafted.deinit(a);
    var plain = try L.Stream.init(a, .{ .id = "plain", .prompt = ids, .max_new = 8, .drafts = false });
    defer plain.deinit(a);
    try core.addStream(&drafted);
    try core.addStream(&plain);
    var steps: usize = 0;
    while (core.activeCount() > 0) {
        if (steps >= 64) return error.Qwen4DeviceLaneProgress;
        try core.step();
        steps += 1;
    }
    try @import("device_fixture.zig").same(u32, "copied drafted == plain", plain.emitted(), drafted.emitted());
    if (plain.emitted().len != 8 or drafted.drafted == 0) return error.Qwen4DeviceDraftCoverage;
    std.debug.print("PASS qwen4 copied drafted == plain: {d} tokens, {d} proposed\n", .{ plain.emitted().len, drafted.drafted });
}
