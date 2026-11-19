//! No native backend: the engine-less build the server's tests and parity suite run (zig/tests/server/build.sh).
const std = @import("std");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{};
pub const families: []const api.Family = &.{};

pub fn chip(_: Allocator) ?[]const u8 {
    return null;
}

pub fn open(a: Allocator, _: Allocator, _: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    problem.* = try @import("no_backend.zig").message(a, o.model_type);
    return null;
}
