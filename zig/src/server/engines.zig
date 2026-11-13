//! The engines this binary opens for a checkpoint: the backend module built in (Metal, or none for the tests).
const std = @import("std");
const api = @import("engine_api");
const native = @import("native_engines");
const cli = @import("cli.zig");
const build_identity = @import("identity.zig");
const Allocator = std.mem.Allocator;
const device = @import("device/support.zig");
const Gpu = @import("device/machine.zig").Gpu;

pub const Opened = api.Opened;

/// Refuse malformed settings before resolving checkpoints and again before an idle reload.
pub fn environment(a: Allocator, args: cli.Args) !?[]const u8 {
    const settings = @import("sys").environment;
    var flags: settings.Flags = .{ .slide = args.slide, .device = args.device, .segments = args.segments, .hooks = @import("build_options").slide_test_hooks };
    for (native.backends) |backend| {
        if (std.mem.eql(u8, backend, "cuda")) flags.cuda = true;
        if (std.mem.eql(u8, backend, "metal")) flags.metal = true;
        if (std.mem.eql(u8, backend, "xpu")) flags.xpu = true;
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

/// `tensorfold device`'s side of the join: what `capabilities --json` lists, without its chip probe or a driver.
pub fn offer(a: Allocator) !device.Offer {
    const families = try a.alloc(device.Family, native.families.len);
    for (families, native.families) |*f, n| f.* = .{ .model_type = n.model_type, .formats = n.formats };
    return .{
        .version = @import("build_options").version,
        .backends = native.backends,
        .families = families,
        .kernel_sms = if (@hasDecl(native, "kernel_sms")) native.kernel_sms else null,
        .cuda_reserve = if (@hasDecl(native, "budget")) cudaReserve else null,
        .metal_share = if (@hasDecl(native, "load_share_percent")) native.load_share_percent else 70,
        .margin = @import("sys").memory_policy.margin,
    };
}

fn cudaReserve(total: u64, unified: bool) u64 {
    return native.budget.reserveBytes(null, total, unified) catch 0;
}

/// The GPUs the built-in backend's driver lists (CUDA's, by ordinal); null in a build with no such query.
pub const driverGpus: ?*const fn (Allocator) ?[]const Gpu = if (@hasDecl(native, "listGpus")) listGpus else null;

fn listGpus(a: Allocator) ?[]const Gpu {
    const listed = native.listGpus(a) orelse return null;
    const out = a.alloc(Gpu, listed.len) catch return null;
    for (out, listed) |*g, l| g.* = .{ .vendor = .nvidia, .name = l.name, .sm = l.sm, .memory = if (l.integrated) 0 else l.memory, .unified = l.integrated, .source = .driver };
    return out;
}
