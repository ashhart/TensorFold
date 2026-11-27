//! --confidential: serve starts only when nothing keeps request content on disk, sends it out or dumps it.
const std = @import("std");
const builtin = @import("builtin");
const cli = @import("cli.zig");
const Allocator = std.mem.Allocator;

/// TensorFold's settings and the loader's injection variables; HOME, PATH, TMPDIR and HF_* stay allowed.
const prefixes = [_][]const u8{ "TENSORFOLD_", "TF_", "FZ_", "GLM_", "MCDMA_", "XPU_", "DYLD_", "LD_", "GLIBC_TUNABLES" } ++ gpu;

/// The GPU drivers' device-memory dumps, profiler injection, and Metal capture, validation and device wrappers.
const gpu = [_][]const u8{
    "CUDA_ENABLE_",  "CUDA_COREDUMP_",  "CUDA_INJECTION",            "NVTX_INJECTION",        "MTL_CAPTURE",
    "METAL_CAPTURE", "MTL_DEBUG_LAYER", "METAL_DEVICE_WRAPPER_TYPE", "MTL_SHADER_VALIDATION",
};

/// Accelerator library, compiler and kernel paths; every NCCL_*_PLUGIN name is refused separately.
const path_names = [_][]const u8{
    "NCCL_NET_LIBRARY",
    "ZE_DRIVERS",
    "ZE_LIBRARY_PATH",
    "ROCBLAS_TENSILE_LIBRARY",
    "HIPBLASLT_TENSILE_LIBRARY",
    "HIPBLASLT_EXTOP_LIBRARY_PATH",
    "MIOPEN_DEVICE_LIB_PATH",
    "MIOPEN_COMPILER",
    "MIOPEN_CUSTOM_CACHE_DIR",
    "MIOPEN_KERNELS_PATH",
    "MIOPEN_CACHE_DIR",
    "MIOPEN_GCN_ASM_PATH",
};

/// The first flag that keeps request content on disk, sends it to another process or fetches its URLs; null: none.
pub fn kept(args: cli.Args) ?[]const u8 {
    const checks = [_]struct { []const u8, bool }{
        .{ "--record", args.record != null },
        .{ "--prompt-cache-gib", (args.prompt_cache_gib orelse 0) > 0 },
        .{ "--checkpoint-slots", (args.checkpoint_slots orelse 0) > 0 },
        .{ "--prompt-cache-disk or --learn-conversations", args.learn_conversations },
        .{ "--learn", args.learn },
        .{ "--recall", args.recall },
        .{ "--slide", args.slide or args.slide_graph != null },
        .{ "--spill-gib", (args.spill_gib orelse 0) > 0 },
        .{ "--compact-memory", args.compact_memory != null },
        .{ "--fsr-route-trace", args.fsr_trace != null },
        .{ "--cortheon", args.cortheon },
        .{ "--tp", args.tp > 1 }, // prompt token ids go to the other ranks
        .{ "--speed-up", args.speed_up != null }, // and to the other Mac
        .{ "--vision-urls", args.vision_urls }, // a request's URL, path and query fetched over the network
        .{ "--vision-pdf", args.vision_pdf }, // a request's PDF parsed in a child process, which no deny-attach covers
        .{ "--manage-voices", args.voice.manage }, // a request's reference audio and transcript kept in the catalog
    };
    for (checks) |c| if (c[1]) return c[0];
    return null;
}

/// The refused variables `env` sets, sorted by name; their values are never read.
pub fn settings(a: Allocator, env: *const std.process.Environ.Map) ![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    for (env.keys()) |name| if (refused(name)) try names.append(a, name);
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    return names.items;
}

fn refused(name: []const u8) bool {
    for (prefixes) |p| if (std.mem.startsWith(u8, name, p)) return true;
    for (path_names) |p| if (std.mem.eql(u8, name, p)) return true;
    return std.mem.startsWith(u8, name, "NCCL_") and std.mem.endsWith(u8, name, "_PLUGIN");
}

/// Why a confidential start is refused, naming flags and variables but never a value; null: it may start.
pub fn refusal(a: Allocator, args: cli.Args, env: *const std.process.Environ.Map) !?[]const u8 {
    if (!args.confidential) return null;
    if (kept(args)) |flag| return try std.fmt.allocPrint(a, "--confidential refuses {s}: it keeps, sends or fetches request content", .{flag});
    const names = try settings(a, env);
    if (names.len == 0) return null;
    const list = try std.mem.join(a, ", ", names);
    return try std.fmt.allocPrint(a, "--confidential reads no TensorFold or loader setting from the environment: unset {s}", .{list});
}

/// The exact TracerPid value in Linux self-status; every absent, duplicate or malformed field fails closed.
pub fn tracerPid(status: []const u8) error{TracerStatusInvalid}!u32 {
    var found: ?u32 = null;
    var lines = std.mem.splitScalar(u8, status, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "TracerPid:")) continue;
        if (found != null) return error.TracerStatusInvalid;
        const value = std.mem.trim(u8, line["TracerPid:".len..], " \t\r");
        if (value.len == 0) return error.TracerStatusInvalid;
        for (value) |ch| if (!std.ascii.isDigit(ch)) return error.TracerStatusInvalid;
        found = std.fmt.parseInt(u32, value, 10) catch return error.TracerStatusInvalid;
    }
    return found orelse error.TracerStatusInvalid;
}

/// Real process operations. Tests replace only these hooks, never the confidential decision or error mapping.
pub const native_ops = struct {
    fn tracerStatus(buf: []u8) ![]const u8 {
        const fd = std.c.open("/proc/self/status", .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) return error.TracerStatusUnavailable;
        defer _ = std.c.close(fd);
        var used: usize = 0;
        while (used < buf.len) {
            const n = std.posix.read(fd, buf[used..]) catch return error.TracerStatusUnavailable;
            if (n == 0) return buf[0..used];
            used += n;
        }
        return error.TracerStatusUnavailable;
    }

    pub fn coreLimit() !void {
        try std.posix.setrlimit(.CORE, .{ .cur = 0, .max = 0 }); // soft and hard, so no child raises it
    }

    pub fn noDump() !void {
        if (std.os.linux.prctl(@backingInt(std.os.linux.PR.SET_DUMPABLE), 0, 0, 0, 0) != 0) return error.Undumpable;
    }

    pub fn denyAttach() !void {
        if (ptrace(deny_attach, 0, null, 0) != 0) return error.DenyAttach;
    }

    fn lockMemory() !void {
        try std.process.lockMemoryAll(.{ .current = true, .future = true });
    }
};

/// PT_DENY_ATTACH from <sys/ptrace.h>: a later ptrace attach fails, as does one already attached.
const deny_attach: c_int = 31;
extern "c" fn ptrace(request: c_int, pid: c_int, addr: ?[*]u8, data: c_int) c_int;

fn hardenWith(comptime ops: type) !void {
    if (comptime builtin.os.tag == .windows) return error.Unsupported;
    if (comptime builtin.os.tag == .linux) {
        var status_buf: [16 * 1024]u8 = undefined;
        const status = try ops.tracerStatus(&status_buf);
        if (try tracerPid(status) != 0) return error.AlreadyTraced;
        ops.coreLimit() catch return error.NoCoreLimit;
        try ops.noDump();
        const status_after_no_dump = try ops.tracerStatus(&status_buf);
        if (try tracerPid(status_after_no_dump) != 0) return error.AlreadyTraced;
    } else {
        ops.coreLimit() catch return error.NoCoreLimit;
        if (comptime builtin.os.tag.isDarwin()) try ops.denyAttach();
    }
    try ops.lockMemory();
}

/// No core dumps, no ordinary debugger and no swapped mappings; every failure refuses confidential startup.
pub fn harden() !void {
    return hardenWith(native_ops);
}

fn hardenRefusal(err: anyerror) []const u8 {
    return switch (err) {
        error.AlreadyTraced => "--confidential refuses to start with an existing tracer (TracerPid is nonzero)",
        error.TracerStatusUnavailable => "--confidential could not read /proc/self/status to determine whether a tracer is attached",
        error.TracerStatusInvalid => "--confidential found missing or invalid TracerPid state",
        error.NoCoreLimit => "--confidential could not set RLIMIT_CORE to zero",
        error.Undumpable => "--confidential could not make the process non-dumpable",
        error.DenyAttach => "--confidential could not deny debugger attachment",
        error.LockedMemoryLimitExceeded => "--confidential could not lock process memory against swap: raise RLIMIT_MEMLOCK (ulimit -l) before restart",
        error.PermissionDenied, error.SystemResources => "--confidential could not lock process memory against swap; check RLIMIT_MEMLOCK and system permissions",
        error.UnsupportedOperation => "--confidential cannot lock process memory against swap on this platform",
        else => "--confidential could not complete process hardening",
    };
}

/// The refusal, then `harden` once it may start: every serving entry point calls it before anything loads.
pub fn enforceWith(a: Allocator, args: cli.Args, env: ?*const std.process.Environ.Map, comptime ops: type) !?[]const u8 {
    if (!args.confidential) return null;
    const empty: std.process.Environ.Map = .init(a);
    if (try refusal(a, args, env orelse &empty)) |why| return why;
    hardenWith(ops) catch |err| return hardenRefusal(err);
    return null;
}

/// LD_PRELOAD/LD_AUDIT/DYLD_INSERT_LIBRARIES precede main; refusal stops serving but cannot prevent or undo injection.
pub fn enforce(a: Allocator, args: cli.Args, env: ?*const std.process.Environ.Map) !?[]const u8 {
    return enforceWith(a, args, env, native_ops);
}
