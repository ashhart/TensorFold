//! LFM2-MoE served by the existing lane host and scheduler, with LiquidAI's DSpark drafter from --drafter.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.lfm2_moe;
const lanes = tf.lanes;
const cache_fit = @import("cache_fit.zig");
const metal_admit = @import("metal_admit.zig");

/// --drafter loads LiquidAI's DSpark drafter (LiquidAI/LFM2.5-24B-A2B-DSpark) in front of the engine.
pub const takes_drafter = true;

const Host = struct {
    resident: ?mtl.resident.Resident = null,
    gpa: std.mem.Allocator,
    model: *q.Model,
    metal: *q.backend.Metal,
    draft: ?*q.dspark.Drafter = null,
    wrap: ?lanes.drafted.Drafted = null,
    warm: mtl.keepalive.Target = undefined, // the model's queue, for the lane host's idle ticker
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // kept prompt states (engine thread only); null: a zero budget

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        if (h.wrap) |*w| w.deinit();
        mtl.resident.streams.drop(h);
        if (h.draft) |d| d.deinit();
        h.config.deinit(h.gpa);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

const cannot = "the native LFM2 engine cannot load this checkpoint ({s})";

/// A refused open's backend, model and host, freed before it reports why.
fn release(h: *Host) void {
    h.metal.deinit();
    h.model.deinit();
    h.gpa.destroy(h);
}

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const config = q.config.Config.read(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    const most = @min(config.context, q.state.max_context);
    const window = q.state.window(config.context, o.context) orelse {
        const outside = "--context {d} is outside LFM2.5-24B-A2B's 1 to {d} tokens";
        problem.* = try std.fmt.allocPrint(a, outside, .{ o.context.?, most });
        return null;
    };
    if (o.context == null and config.context > most) std.log.info(
        "lfm2: serving {d} of the checkpoint's {d} positions, all its attention holds",
        .{ most, config.context },
    );
    const model = q.Model.loadWithResidency(gpa, io, o.dir, false) catch |err| {
        problem.* = try std.fmt.allocPrint(a, cannot, .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.model = model;
    h.resident = null;
    const chunk = 2048;
    h.metal = try q.backend.Metal.init(gpa, model, .{ .capacity = window, .chunk = chunk, .streams = @max(o.lanes, 2) });
    errdefer h.metal.deinit();
    h.draft = null;
    h.wrap = null;
    if (if (o.drafts) o.drafter else null) |dir| {
        if (h.metal.width == 1) {
            problem.* = "--drafter: LFM2 wider lanes failed the local exactness check; drafts require verified wider lanes";
            release(h);
            return null;
        }
        const bits: u8 = if (o.drafter_bits_given) o.drafter_bits else 0;
        if (bits != 0 and bits != 8) {
            problem.* = "--drafter-bits: the LFM2 DSpark drafter takes 0 (bf16 as released, the default) or 8 (8-bit codes quantized at load)";
            release(h);
            return null;
        }
        h.draft = q.dspark.Drafter.init(gpa, io, model, dir, .{ .capacity = window, .bits = bits }) catch |err| {
            problem.* = try std.fmt.allocPrint(a, "the native LFM2 drafter cannot load {s} ({s}): it takes LiquidAI/LFM2.5-24B-A2B-DSpark", .{ dir, @errorName(err) });
            release(h);
            return null;
        };
    }
    errdefer if (h.draft) |d| d.deinit();
    // a stream's caches, and its drafter lane when one drafts
    const per_stream = q.state.cacheBytes(&model.config.attention, window) + if (h.draft) |d| d.laneBytes(window) else 0;
    const served = try metal_admit.streams(a, "lfm2", per_stream, metal_admit.room(model.device), o, window, problem) orelse {
        if (h.draft) |d| d.deinit();
        release(h);
        return null;
    };
    h.metal.options.streams = @max(served, 2);
    if (h.draft) |d| {
        h.wrap = try lanes.drafted.Drafted.init(gpa, h.metal.backend(), d.drafter());
        // the verify copies the drafts on the GPU after the unwaited hold
        h.metal.held_source = d.heldSource();
        h.wrap.?.held_on_target = true;
    }
    errdefer if (h.wrap) |*w| w.deinit();
    const facts = if (h.wrap) |*w| w.facts(h.metal.facts()) else h.metal.facts();
    h.config = try lanes.Config.init(gpa, facts, q.state.window_rows, q.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, if (h.wrap) |*w| w.backend() else h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = served, .context_window = @intCast(window), .prefill_step = chunk, .decode_share = o.decode_share, .structures = true });
    h.cache = null;
    var why: []const u8 = "";
    const dev = model.device;
    const plan = cache_fit.budget(
        o.prompt_cache_gib,
        o.prompt_cache_over_cap,
        dev.maxWorkingSet() -| dev.allocated() -| (8 << 30),
        a,
        &why,
        "",
    ) catch |e| {
        if (e == error.CacheOverCap) problem.* = why;
        return e;
    };
    h.host.info_.prompt_cache_plan = plan;
    if (plan.budget_bytes > 0) {
        const S = q.backend.Snaps;
        h.cache = api.prompt_cache.Store.init(gpa, .{
            .ptr = h,
            .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = S.restore, .drop = Snaps.drop },
        }, .{ .chunk = chunk }, plan.budget_bytes);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
    }
    errdefer if (h.cache) |*store| store.deinit();
    try mtl.resident.streams.hold(h, io, "lfm2 moe");
    errdefer mtl.resident.streams.drop(h);
    if (h.draft) |d| if (h.resident) |*r| {
        const held = try d.buffers(gpa);
        defer gpa.free(held);
        r.add(held);
    };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    if (h.draft) |d| {
        const mode = if (d.options.bits == 8) "8-bit codes quantized at load (--drafter-bits 8), not the released bf16" else "bf16 matrices as released";
        std.log.info("lfm2: DSpark drafts from --drafter, {d} a round at most, {s}", .{ d.config.block, mode });
    }
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// The prompt cache's copies of a stream: the target's state at a mark (backend.zig), and the drafter's context there.
const Snaps = struct {
    const S = q.backend.Snaps;
    fn host(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(ptr: *anyopaque, at: u32) u64 {
        const h = host(ptr);
        return S.bytes(h.metal, at) + if (h.draft) |d| d.keptBytes(at) else 0;
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const h = host(ptr);
        const saved = try S.save(h.metal, owner, at);
        if (h.draft) |d| d.keepLane(@ptrCast(@alignCast(owner.?)), @ptrCast(@alignCast(saved))) catch |err| {
            S.drop(h.metal, saved);
            return err;
        };
        return saved;
    }
    fn drop(ptr: *anyopaque, saved: *anyopaque) void {
        S.drop(host(ptr).metal, saved);
    }
};
