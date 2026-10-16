//! The engines this binary opens for a checkpoint: the backend module built in (Metal, or none for the tests).
const std = @import("std");
const api = @import("engine_api");
const native = @import("native_engines");
const cli = @import("cli.zig");
const build_identity = @import("identity.zig");
const Allocator = std.mem.Allocator;

pub const Opened = api.Opened;

/// The serve flags past the common ones the backend lists (tensor parallelism, MTP, kept entries, policy).
pub const serves: []const []const u8 = if (@hasDecl(native, "serves")) native.serves else &.{};

/// The backend's std options when it has any (HIP: no alternate signal stack for RCCL's threads); else the defaults.
pub const std_options: std.Options = @import("std_options.zig").of(native);

/// What ``capabilities --json`` reports: this release, the chip here, and what the built-in backend serves.
pub fn capabilities(a: Allocator) cli.Engines {
    return .{ .version = @import("build_options").version, .chip = if (native.families.len > 0) native.chip(a) else null, .backends = native.backends, .families = native.families };
}

/// What the open /health names: this release, the built-in backend and the GPU a serve with ``args`` opens.
pub fn identity(a: Allocator, args: cli.Args) build_identity.Identity {
    return build_identity.of(native, a, @import("build_options").version, args.device);
}

/// The engine for the checkpoint in ``dir``, or null with ``problem`` set.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, model_type: []const u8, args: cli.Args, problem: *[]const u8) !?Opened {
    if (args.record != null and !@hasDecl(native, "records")) {
        problem.* = "--record writes rows from the CUDA engine only so far: serve without --record";
        return null;
    }
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
