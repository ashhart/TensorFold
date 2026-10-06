//! The engines this binary opens for a checkpoint: the backend module built in (Metal, or none for the tests).
const std = @import("std");
const api = @import("engine_api");
const native = @import("native_engines");
const cli = @import("cli.zig");
const Allocator = std.mem.Allocator;

pub const Opened = api.Opened;

/// What ``capabilities --json`` reports: this release, the chip here, and what the built-in backend serves.
pub fn capabilities(a: Allocator) cli.Engines {
    return .{ .version = "0.6.5", .chip = if (native.families.len > 0) native.chip(a) else null, .backends = native.backends, .families = native.families };
}

/// The engine for the checkpoint in ``dir``, or null with ``problem`` set.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, model_type: []const u8, args: cli.Args, problem: *[]const u8) !?Opened {
    return native.open(a, gpa, io, .{
        .dir = dir,
        .model_type = model_type,
        .context = args.context,
        .lanes = cli.parallel(args.parallel) orelse 8,
        .drafts = !args.no_drafts,
        .speed_up = args.speed_up,
        .prompt_cache_gib = args.prompt_cache_gib,
        .prompt_cache_over_cap = args.prompt_cache_over_cap,
    }, problem);
}
