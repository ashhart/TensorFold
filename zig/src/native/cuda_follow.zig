//! Follower rank ownership for native families that expose a replay loop.
const std = @import("std");
const api = @import("engine_api");
const native = @import("cuda.zig");

pub fn follow(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, options: api.Open, problem: *[]const u8) !bool {
    const engine = (try native.open(a, gpa, io, options, problem)) orelse return false;
    defer engine.close(engine.ctx);
    const replay = engine.follow orelse {
        problem.* = "this family has no tensor-parallel follower loop";
        return false;
    };
    replay(engine.ctx) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "rank {d} stopped: {s}", .{ options.rank, @errorName(err) });
        return false;
    };
    return true;
}
