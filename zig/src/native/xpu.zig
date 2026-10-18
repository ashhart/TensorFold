//! XPU engines (Intel Xe2 over Level Zero): Qwen3.8-27B behind the lane host; a lone stream uses the backend's driver.
const std = @import("std");
const api = @import("engine_api");
const xq = @import("xpu_qwen27");
const Drafter = xq.drafter.Drafter;
const lanes = @import("lanes");
const sampling = lanes.sampling;
const XpuLanes = @import("xpu_lanes.zig").Lanes;
const nem = @import("xpu_nem.zig");
const admit = @import("xpu_admit.zig");
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{"xpu"};
pub const families: []const api.Family = &.{ .{ .model_type = "qwen3_5", .formats = &.{ "mlx-q4g64", "gptq-int4-g128" } }, .{ .model_type = "nemotron_h", .formats = &.{"mlx-q4g64"} } };

pub fn chip(_: Allocator) ?[]const u8 {
    return "Intel Xe2 (Level Zero)";
}

/// A sampled draw of `row`'s logits at `position`: over the top_k candidates only (top_k <= 256), else all of them.
fn pickCtx(ctx: *anyopaque, row: []const f32, position: u64, s: api.Sampling) !u32 {
    const gpa: *const Allocator = @ptrCast(@alignCast(ctx));
    return pickRow(gpa.*, row, position, s);
}

fn pickRow(gpa: Allocator, row: []const f32, position: u64, s: api.Sampling) !u32 {
    if (s.temperature <= 0 or s.top_k == 0 or s.top_k > 256) return pickAll(gpa, row, position, s);
    const k: usize = s.top_k;
    var vals: [256]f64 = undefined;
    var ids: [256]u64 = undefined;
    var n: usize = 0;
    for (row, 0..) |v32, i| {
        const v: f64 = v32;
        if (!std.math.isFinite(v)) return error.NonfiniteLogits;
        if (n == k and !(v > vals[k - 1])) continue; // equal values keep the lower id, already held
        var j = if (n < k) n else k - 1;
        if (n < k) n += 1;
        while (j > 0 and v > vals[j - 1]) : (j -= 1) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
        }
        vals[j] = v;
        ids[j] = i;
    }
    return @intCast(try sampling.choose(gpa, vals[0..n], ids[0..n], position, s));
}

fn pickAll(gpa: Allocator, row: []const f32, position: u64, s: api.Sampling) !u32 {
    const values = try gpa.alloc(f64, row.len);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, row.len);
    defer gpa.free(ids);
    for (values, ids, row, 0..) |*v, *id, x, i| {
        v.* = x;
        if (!std.math.isFinite(v.*)) return error.NonfiniteLogits;
        id.* = i;
    }
    return @intCast(try sampling.choose(gpa, values, ids, position, s));
}

/// The lane host over N runner slots (`--parallel N`, at least one).
const LaneServed = struct {
    gpa: Allocator,
    back: XpuLanes,
    cfg: lanes.Config,
    wall: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // kept prompt states (engine thread only); null: no budget

    fn close(ptr: *anyopaque) void {
        const h: *LaneServed = @ptrCast(@alignCast(ptr));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        for (h.back.first_logits) |l| if (l) |x| h.gpa.free(x);
        if (h.back.mask_bits) |x| h.gpa.free(x);
        if (h.back.masked_vals) |x| h.gpa.free(x);
        if (h.back.masked_ids) |x| h.gpa.free(x);
        h.gpa.destroy(@as(*Allocator, @ptrCast(@alignCast(h.back.pick_ctx))));
        if (h.back.drafter) |d| d.deinit();
        h.back.runner.deinit(); // the card's memory and context, after the engine has stopped
        h.gpa.destroy(h);
    }
};

/// The prompt cache's states: a slot's copy at a mark of its prompt pass, restored by the pass itself.
const Snaps = struct {
    fn of(ptr: *anyopaque) *LaneServed {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(ptr: *anyopaque, at: u32) u64 {
        return of(ptr).back.keepBytes(at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
        const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        return @ptrCast(try of(ptr).back.keepSave(s, at));
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {
        return error.RestoredInPrefill; // the backend restores a stream's kept state in its prompt pass
    }
    fn drop(ptr: *anyopaque, saved: api.prompt_cache.Saved) void {
        of(ptr).back.keepDrop(@ptrCast(@alignCast(saved)));
    }
};

/// The prompt cache's budget: --prompt-cache-gib, else 2 GiB (0: no cache); set aside before the streams.
fn cacheBudget(o: api.Open) u64 {
    const gib = o.prompt_cache_gib orelse 2.0;
    return if (gib > 0) std.math.lossyCast(u64, gib * (1 << 30)) else 0;
}

/// Streams the loaded card holds: each one past the first a GDN state and `context` tokens of K/V; 0 when none fits.
fn slotsThatFit(runner: *xq.Runner, want: u32, reserve: u64) !u32 {
    const margin: u64 = 2 << 30; // activations of later chunks, the drafter's growth
    const mem = try runner.d.g.memory();
    const m: admit.Memory = .{ .free = mem.free, .size = mem.size, .allocated = runner.d.allocated };
    return admit.streamsThatFit(m, margin, reserve, runner.slotBytes(), want, xq.MAX_SLOTS);
}

fn openLanes(gpa: Allocator, io: std.Io, runner: *xq.Runner, drafter: ?*Drafter, o: api.Open, context: u32) !api.Opened {
    const budget = cacheBudget(o);
    const slots = try slotsThatFit(runner, @min(@max(o.lanes, 1), xq.MAX_SLOTS), budget);
    if (slots == 0) return error.StreamDoesNotFit;
    const h = try gpa.create(LaneServed);
    errdefer gpa.destroy(h);
    const gpa_box = try gpa.create(Allocator);
    errdefer gpa.destroy(gpa_box);
    gpa_box.* = gpa;
    h.gpa = gpa;
    h.cache = null; // gpa.create leaves the defaults unset
    h.back = .{ .gpa = gpa, .runner = runner, .drafter = drafter, .pick = pickCtx, .pick_ctx = gpa_box, .slots = slots };
    if (std.c.getenv("XPU_BLOCK")) |v| h.back.block = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 8;
    if (std.c.getenv("XPU_TREE")) |v| h.back.nodes = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 15;
    if (drafter != null) {
        try h.back.measure();
        std.log.info("qwen27-xpu: window ms by rows: 1 {d:.1}, 8 {d:.1}, 16 {d:.1}", .{ h.back.costs[0].ms, h.back.costs[7].ms, h.back.costs[15].ms });
    }
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), 16, 15);
    errdefer h.cfg.deinit(gpa);
    h.wall = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    // the engine cuts its prompts on its own grid from position 0 (prefill_step 0): the same ids give the same bits
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "qwen27-xpu", .lanes = slots, .context_window = context, .prefill_step = 0, .structures = true });
    if (drafter != null) {
        h.back.initDrafts();
        h.host.lone = .{ .ctx = &h.back, .run = XpuLanes.lone, .resumes = true, .sampled = true };
    }
    if (budget > 0) {
        // the prompt kernels compute in blocks of 64 rows from a call's start: kept and resumed at planned marks only
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .planned = true, .grid = 64 }, budget);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        std.log.info("qwen27-xpu: prompt cache of {d:.1} GiB", .{@as(f64, @floatFromInt(budget)) / (1 << 30)});
    }
    errdefer if (h.cache) |*store| store.deinit();
    try h.host.start();
    std.log.info("qwen27-xpu: {d} streams of {d} tokens{s}", .{ slots, context, if (drafter != null) ", DFlash2 trees for a lone stream" else "" });
    return .{ .engine = h.host.engine(), .close = LaneServed.close, .ctx = h };
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (nem.serves(o.model_type)) return nem.open(a, gpa, io, o, problem);
    if (!std.mem.eql(u8, o.model_type, "qwen3_5")) {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine serves Qwen3.8-27B (qwen3_5, MLX affine 4/64 or symmetric GPTQ 4/128) and Nemotron-H (nemotron_h, MLX 4/64), not {s}", .{o.model_type});
        return null;
    }
    const context = xq.limits.context(o.context orelse 32768) catch {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine holds 1 to {d} tokens a stream, not {d}", .{ xq.limits.MAX_CONTEXT, o.context.? });
        return null;
    };
    const runner = xq.Runner.init(o.dir, context) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU engine could not load {s}: {s}", .{ o.dir, @errorName(err) });
        return null;
    };
    var dr: ?*Drafter = null;
    if (o.drafts) if (o.drafter) |dir| {
        dr = Drafter.init(runner, dir) catch |err| {
            runner.deinit();
            problem.* = try std.fmt.allocPrint(a, "the XPU engine could not load the drafter {s}: {s}", .{ dir, @errorName(err) });
            return null;
        };
    };
    return openLanes(gpa, io, runner, dr, o, context) catch |err| {
        if (dr) |x| x.deinit();
        runner.deinit();
        problem.* = if (err == error.StreamDoesNotFit)
            try std.fmt.allocPrint(
                a,
                "the XPU engine cannot hold a stream of {d} tokens and a {d:.1} GiB prompt cache " ++
                    "beside {s} on this card",
                .{ context, @as(f64, @floatFromInt(cacheBudget(o))) / (1 << 30), o.dir },
            )
        else
            try std.fmt.allocPrint(a, "the XPU engine could not serve {s}: {s}", .{ o.dir, @errorName(err) });
        return null;
    };
}
