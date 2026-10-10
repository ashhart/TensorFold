//! The engines this binary opens for a checkpoint: the backend module built in (Metal, or none for the tests).
const std = @import("std");
const api = @import("engine_api");
const native = @import("native_engines");
const cli = @import("cli.zig");
const Allocator = std.mem.Allocator;

pub const Opened = api.Opened;

/// The serve flags past the common ones the backend lists (tensor parallelism, MTP, kept entries, policy).
pub const serves: []const []const u8 = if (@hasDecl(native, "serves")) native.serves else &.{};

/// What ``capabilities --json`` reports: this release, the chip here, and what the built-in backend serves.
pub fn capabilities(a: Allocator) cli.Engines {
    return .{ .version = @import("build_options").version, .chip = if (native.families.len > 0) native.chip(a) else null, .backends = native.backends, .families = native.families };
}

/// The engine for the checkpoint in ``dir``, or null with ``problem`` set.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, model_type: []const u8, args: cli.Args, problem: *[]const u8) !?Opened {
    return native.open(a, gpa, io, try cli.request(a, dir, model_type, args), problem);
}

/// A tensor-parallel rank above 0 runs rank 0's steps until it stops; false with ``problem`` set when it cannot.
pub fn follow(a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, model_type: []const u8, args: cli.Args, problem: *[]const u8) !bool {
    if (!@hasDecl(native, "follow")) {
        problem.* = "this build has no tensor-parallel ranks";
        return false;
    }
    return native.follow(a, gpa, io, try cli.request(a, dir, model_type, args), problem);
}
