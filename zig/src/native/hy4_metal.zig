//! Hy4-flash served by the lane host on Metal: one stream at a time with DFly2 windows, contexts to 256K tokens
//! (128K by default), and the last stream's caches kept for a following prompt that starts with the same tokens.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const hy4 = tf.hy4_flash;

const default_context: u32 = 131072;
const most_context: u32 = 262144;

const Host = struct {
    gpa: std.mem.Allocator,
    metal: *hy4.Metal,
    resident: ?mtl.resident.Resident = null, // every buffer wired between requests (the idle ticker keeps it)
    warm: mtl.keepalive.Target = undefined,
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.metal.follow = null;
        if (h.resident) |*r| r.deinit();
        h.metal.deinit();
        h.gpa.destroy(h);
    }
};

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (try o.slideRefusal(a, "Macs")) |why| {
        problem.* = why;
        return null;
    }
    if (o.vision != null) {
        problem.* = "--vision: the native Hy4-flash engine reads text only; serve without --vision";
        return null;
    }
    const asked: i64 = o.context orelse default_context;
    if (asked < 4096 or asked > most_context) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: the native Hy4-flash engine serves 4,096 to {d} tokens", .{ asked, most_context });
        return null;
    }
    // caches hold the window, a verify window past it, rounded to whole 512-slot pages
    const capacity: u32 = @intCast(((@as(u64, @intCast(asked)) + hy4.max_window + 511) / 512) * 512);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const metal = hy4.Metal.init(gpa, io, o.dir, .{ .capacity = capacity }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native Hy4-flash engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer metal.deinit();
    std.log.info("hy4: {d}-token window, {d:.1} GiB of caches a stream", .{ asked, @as(f64, @floatFromInt(metal.cacheBytes(capacity))) / (1 << 30) });
    try metal.warm();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.metal = metal;
    h.config = try lanes.Config.init(gpa, metal.facts(), hy4.max_window, hy4.max_window - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = 1, .context_window = @intCast(asked), .prefill_step = hy4.prompt_rows, .decode_share = o.decode_share, .structures = true, .logprobs = true });
    // weights, scratch and the kept caches in one residency set: idle GPU pages otherwise cost seconds to come back
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(gpa);
    try metal.residentBuffers(&list);
    h.resident = try mtl.resident.Resident.hold(gpa, io, metal.device, metal.queue, list.items, "hy4");
    if (h.resident) |*r| {
        metal.follow = r.follow();
        h.warm = r.target();
    } else h.warm = .{ .queue = metal.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
