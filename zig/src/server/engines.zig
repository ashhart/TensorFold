//! The engines this binary opens for a checkpoint: the backend module built in (Metal, or none for the tests).
const std = @import("std");
const api = @import("engine_api");
const native = @import("native_engines");
const cli = @import("cli.zig");
const build_identity = @import("identity.zig");
const Allocator = std.mem.Allocator;

pub const Opened = api.Opened;

/// Refuse malformed settings before resolving checkpoints and again before an idle reload.
pub fn environment(a: Allocator, args: cli.Args) !?[]const u8 {
    const settings = @import("sys").environment;
    var flags: settings.Flags = .{ .slide = args.slide, .device = args.device, .segments = args.segments, .hooks = @import("build_options").slide_test_hooks };
    for (native.backends) |backend| {
        if (std.mem.eql(u8, backend, "cuda")) flags.cuda = true;
        if (std.mem.eql(u8, backend, "metal")) flags.metal = true;
        if (std.mem.eql(u8, backend, "xpu")) flags.xpu = true;
        if (std.mem.eql(u8, backend, "cpu")) flags.cpu = true;
    }
    return settings.refusal(a, settings.Process{}, flags);
}

/// The serve flags past the common ones the backend lists (tensor parallelism, MTP, kept entries, policy).
pub const serves: []const []const u8 = if (@hasDecl(native, "serves")) native.serves else &.{};

/// The backend's std options when it has any (HIP: no alternate signal stack for RCCL's threads); else the defaults.
pub const std_options: std.Options = @import("std_options.zig").of(native);

/// What ``capabilities --json`` reports: this release, the chip here, and what the built-in backend serves.
pub fn capabilities(a: Allocator) cli.Engines {
    return .{ .version = @import("build_options").version, .chip = if (native.families.len > 0) native.chip(a) else null, .backends = native.backends, .families = native.families };
}

/// ``capabilities --json``'s NVML P2P summary when the backend has one (CUDA, two or more GPUs); else null.
pub fn p2p(a: Allocator) ?[]const u8 {
    return if (@hasDecl(native, "p2p")) native.p2p(a) else null;
}

/// What the open /health names: this release, the built-in backend and the GPU a serve with ``args`` opens.
pub fn identity(a: Allocator, args: cli.Args) build_identity.Identity {
    return build_identity.of(native, a, @import("build_options").version, args.device);
}

/// The engine for the checkpoint in ``dir``, or null with ``problem`` set.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, model_type: []const u8, args: cli.Args, problem: *[]const u8) !?Opened {
    if (try environment(a, args)) |why| {
        problem.* = why;
        return null;
    }
    if (args.idle_unload > 0) {
        const world = if (std.c.getenv("TF_GLM_WORLD")) |value| std.mem.span(value) else null;
        if (@import("idle_unload_policy.zig").refusal(native.backends, model_type, args.idle_unload, world, args.tp)) |why| {
            problem.* = why;
            return null;
        }
    }
    if (args.record != null and !@hasDecl(native, "records")) {
        problem.* = "--record writes rows from the CUDA engine only so far: serve without --record";
        return null;
    }
    return native.open(a, gpa, io, try cli.request(a, dir, model_type, args), problem);
}

/// A tensor-parallel rank above 0 runs rank 0's steps until it stops; false with ``problem`` set when it cannot.
pub fn follow(a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, model_type: []const u8, args: cli.Args, problem: *[]const u8) !bool {
    if (try environment(a, args)) |why| {
        problem.* = why;
        return false;
    }
    if (!@hasDecl(native, "follow")) {
        problem.* = "this build has no tensor-parallel ranks";
        return false;
    }
    return native.follow(a, gpa, io, try cli.request(a, dir, model_type, args), problem);
}
