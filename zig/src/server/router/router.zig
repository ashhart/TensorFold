//! ``tensorfold serve --router``: one address for many models, each a ``tensorfold serve`` child started on demand.
const std = @import("std");
const cli = @import("../cli.zig");
const auth = @import("../auth.zig");
const log = @import("../log.zig");
const serve = @import("../serve.zig");
const args_mod = @import("args.zig");
const config = @import("config.zig");
const catalog = @import("catalog.zig");
const pool_mod = @import("pool.zig");
const front = @import("front.zig");

pub const Checkpoints = catalog.Checkpoints;
pub const Memory = catalog.Memory;
pub const wanted = args_mod.wanted;

/// Routes until a stop signal; the process exit status. ``argv`` is serve's arguments, ``--router`` among them.
pub fn main(init: std.process.Init, argv: []const []const u8, checkpoints: Checkpoints, version: []const u8) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const env = init.environ_map;
    if (args_mod.help(argv)) {
        try std.Io.File.stdout().writeStreamingAll(io, args_mod.usage_line);
        return 0;
    }
    log.init(io, false);
    var u: cli.Usage = .{};
    const settings = args_mod.parse(a, argv, &u) catch |e| switch (e) {
        error.Usage => return usage(u.message),
        else => |x| return x,
    };
    var problem: []const u8 = "";
    const exe = try std.process.executablePathAlloc(io, a);
    const entries = if (settings.config) |path| blk: {
        const models = config.load(a, path, &u) catch |e| switch (e) {
            error.Usage => return usage(u.message),
            else => |x| return x,
        };
        break :blk try catalog.build(a, io, env, exe, models, checkpoints, &problem) orelse return fail(problem);
    } else try catalog.buildCached(a, io, env, exe, try catalog.cached(a, io, env, checkpoints), checkpoints);
    if (entries.len == 0) return fail("no models to route: run tensorfold pull OWNER/NAME, or list models in --router-config FILE.toml");
    var store = auth.Store.init(gpa, settings.api_keys, env.get("TENSORFOLD_API_KEY") orelse "", settings.api_key_file, false, &problem) catch
        return fail(if (problem.len > 0) problem else "API key file cannot be opened safely");
    defer store.deinit();
    const capacity = std.process.totalSystemMemory() catch 0;
    for (entries) |e| note(e, capacity);
    const pool = try pool_mod.Pool.init(gpa, io, entries, env, .{
        .max_loaded = settings.max_loaded,
        .idle_s = settings.idle_s,
        .capacity = capacity,
        .log_dir = try std.fs.path.join(a, &.{ env.get("HOME") orelse ".", ".tensorfold", "router" }),
    });
    defer pool.deinit();
    const lis = serve.openTcp(io, settings) orelse return 1;
    serve.installSignals(store.enabled());
    // without keys a hangup stops the router as a TERM does, so its children never outlive the terminal
    if (!store.enabled()) serve.stopOnHangup();
    if (!store.enabled() and !auth.loopback(settings.host)) log.line("warning: non-loopback router has no API key; requests are open", .{});
    var f: front.Front = .{
        .gpa = gpa,
        .io = io,
        .pool = pool,
        .entries = entries,
        .keys = if (store.enabled()) &store else null,
        .browser = .{ .allowed_hosts = settings.allowed_hosts, .allowed_origins = settings.allowed_origins },
        .version = version,
    };
    try pool.startReaper();
    var accept_stop: std.atomic.Value(bool) = .init(false);
    const accept = try std.Thread.spawn(.{}, front.Front.serve, .{ &f, lis, &accept_stop });
    var idle_text: [64]u8 = undefined;
    const idle = if (settings.idle_s > 0) std.fmt.bufPrint(&idle_text, "; an idle model stops after {d} s", .{settings.idle_s}) catch "" else "";
    log.line("routing {d} model{s} at http://{s}:{d}/v1: at most {d} loaded at once{s}; a request's model field picks one", .{ entries.len, if (entries.len == 1) "" else "s", settings.host, lis.port(), settings.max_loaded, idle });
    while (!serve.stopping()) io.sleep(.fromMilliseconds(50), .awake) catch {};
    f.draining.store(true, .release);
    log.line("shutting down ({s}): stopping the loaded models", .{serve.stopName()});
    pool.stopAll();
    accept_stop.store(true, .release);
    accept.join();
    lis.close();
    var waited: u32 = 0;
    while (f.open.load(.acquire) > 0 and waited < 100) : (waited += 1) io.sleep(.fromMilliseconds(50), .awake) catch {};
    if (f.open.load(.acquire) > 0) {
        log.line("shutting down: exiting with {d} connection(s) still open", .{f.open.load(.acquire)});
        std.process.exit(0); // connections still open: end without freeing what they read
    }
    log.line("shutting down: stopped", .{});
    return 0;
}

/// A model the memory check cannot bound, or one larger than the machine, said once at startup.
fn note(e: catalog.Entry, capacity: u64) void {
    const gib: f64 = 1 << 30;
    const floor = e.floor orelse return log.line("{s}: no memory estimate, so only --max-loaded bounds it", .{e.id});
    if (capacity > 0 and floor > capacity)
        log.line("{s}: needs about {d:.1} GiB loaded, more than this machine's {d:.1} GiB; requests for it get 503", .{ e.id, @as(f64, @floatFromInt(floor)) / gib, @as(f64, @floatFromInt(capacity)) / gib });
}

fn usage(message: []const u8) u8 {
    std.debug.print("{s}tensorfold serve: error: {s}\n", .{ args_mod.usage_line, message });
    return 2;
}

fn fail(message: []const u8) u8 {
    std.debug.print("tensorfold: {s}\n", .{message});
    return 1;
}
