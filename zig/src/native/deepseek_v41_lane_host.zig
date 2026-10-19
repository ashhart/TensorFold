//! Greedy native lanes with optional two-Mac MoE and streamed Engram; memory limits: docs/deepseek-execution.md.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const ds = tf.deepseek_v41;
const lanes = tf.lanes;
const Allocator = std.mem.Allocator;

pub const Host = struct {
    gpa: Allocator,
    model: *ds.Model,
    pair: ?*tf.deepseek_v41_tp_pair.Pair = null,
    follower: ?std.Thread = null,
    back: tf.deepseek_v41_backend.Backend,
    cfg: lanes.Config,
    wall: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    warm: mtl.keepalive.Target,

    pub fn engine(h: *Host) api.Engine {
        return h.host.engine();
    }
};

fn words(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.FollowsPeer => "this DeepSeek TP rank follows rank 0; send generation requests to rank 0",
        error.InvalidTensorParallelState => "the DeepSeek TP pair lost synchronization; restart both ranks",
        error.GreedyOnly => "the native DeepSeek V4.1 engine decodes greedily only: send temperature 0",
        error.ContextFull => "the prompt and max_tokens do not fit this server's --context",
        error.NoFreeSlot => "every lane slot is busy",
        error.LoadLimitExceeded => "the resident payload does not fit the load limit: raise GLM_LOAD_LIMIT_GB, free memory, or pass --engram-stream to read the Engram tables from disk",
        error.MissingEngramTables => "the checkpoint has no Engram tables: pass --engram-table-dir with the original release's shards",
        error.InvalidTokenMap => "engram-token-map.bin is missing or stale beside the checkpoint",
        else => null,
    };
}

/// Build a missing map from tokenizer/config beside Engram tables, without a Python runtime step.
fn ensureTokenMap(a: Allocator, io: std.Io, map: []const u8, table_dir: ?[]const u8) void {
    const src = table_dir orelse return;
    if (std.Io.Dir.cwd().openFile(io, map, .{})) |f| {
        f.close(io);
        return;
    } else |_| {}
    const tk_path = std.fmt.allocPrintSentinel(a, "{s}/tokenizer.json", .{src}, 0) catch return;
    defer a.free(tk_path);
    const cfg_path = std.fmt.allocPrintSentinel(a, "{s}/config.json", .{src}, 0) catch return;
    defer a.free(cfg_path);
    for ([_][]const u8{ tk_path, cfg_path }) |p| {
        if (std.Io.Dir.cwd().openFile(io, p, .{})) |f| {
            f.close(io);
        } else |_| return;
    }
    tf.deepseek_v41_token_map.build(a, io, tk_path, cfg_path, map) catch |err| {
        std.log.err("deepseek-v41: building the engram token map from {s} failed: {s}", .{ src, @errorName(err) });
        return;
    };
    std.log.info("deepseek-v41: built the engram token map from {s} at {s}", .{ src, map });
}

/// 70% of RAM, or GLM_LOAD_LIMIT_GB's GB up to the GPU's working set; the same opt-in pattern GLM's load uses.
fn loadLimit(device: mtl.Device) !struct { bytes: u64, asked: bool } {
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &mem, &len, null, 0) != 0) mem = 0;
    const default: u64 = @intFromFloat(@as(f64, @floatFromInt(mem)) / (1 << 30) * 0.7 * 1e9);
    const text: ?[]const u8 = if (std.c.getenv("GLM_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len == 0) return .{ .bytes = default, .asked = false };
    const gb = std.fmt.parseFloat(f64, t) catch return error.BadLoadLimit;
    if (!std.math.isFinite(gb) or gb <= 0) return error.BadLoadLimit;
    const most: u64 = @min(mem, device.maxWorkingSet());
    return .{ .bytes = @intFromFloat(@min(gb * 1e9, @as(f64, @floatFromInt(most)))), .asked = true };
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const experimental: ?[]const u8 = if (std.c.getenv("TF_EXPERIMENTAL")) |v| std.mem.span(v) else null;
    if (tf.deepseek_v41_admission.refusal(experimental)) |why| {
        problem.* = why;
        return null;
    }
    if (o.slide or o.learn != null or
        (o.prompt_cache_gib != null and o.prompt_cache_gib.? != 0) or o.prompt_cache_over_cap or
        o.drafter != null or o.tp != 1)
    {
        problem.* = "V4.1 resident serving does not support Sliding or Living Weights, retained prompt caches, a drafter or --tp; use --speed-up SETTINGS for a two-Mac MoE pair";
        return null;
    }
    const map = try std.fmt.allocPrintSentinel(a, "{s}/engram-token-map.bin", .{o.dir}, 0);
    defer a.free(map);
    ensureTokenMap(a, io, map, o.engram_table_dir);
    var map_io = io;
    _ = &map_io;
    const device_probe = mtl.Device.init() catch null;
    defer if (device_probe) |d| d.deinit();
    const limit: u64 = if (device_probe) |d| blk: {
        const l = loadLimit(d) catch {
            problem.* = "GLM_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0, or unset it for 70% of RAM";
            return null;
        };
        if (l.asked) std.log.info("deepseek-v41: load limit {d} bytes from GLM_LOAD_LIMIT_GB", .{l.bytes});
        break :blk l.bytes;
    } else 0;
    const model = try a.create(ds.Model);
    errdefer a.destroy(model);
    model.* = ds.Model.open(gpa, io, .{
        .dir = o.dir,
        .table_dir = o.engram_table_dir,
        .token_map = map,
        .context = if (o.context) |c| if (c > 0) @intCast(@min(c, std.math.maxInt(u32))) else 0 else 0,
        .lanes = @max(1, @min(o.lanes, 256)),
        .ram_limit_bytes = limit,
        .engram = if (o.engram_stream) .streamed else .resident,
        .bound_only = o.engram_stream,
    }) catch |err| {
        if (err == error.InvalidTokenMap) {
            a.destroy(model);
            problem.* = "engram-token-map.bin is missing beside the checkpoint: the pull builds it; see the family guide";
            return null;
        }
        return err;
    };
    const model_ptr = model;
    errdefer model_ptr.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{
        .gpa = gpa,
        .model = model_ptr,
        .back = undefined,
        .cfg = undefined,
        .wall = .{ .io = io },
        .core = undefined,
        .host = undefined,
        .warm = undefined,
    };
    if (o.speed_up) |settings_path| {
        const json = try std.Io.Dir.cwd().readFileAlloc(io, settings_path, gpa, .limited(1 << 20));
        defer gpa.free(json);
        const identity = try tf.deepseek_v41_tp_pair.identity(gpa, io, o.dir);
        h.pair = try tf.deepseek_v41_tp_pair.Pair.open(gpa, model_ptr, json, identity);
    }
    errdefer if (h.pair) |pair| pair.deinit();
    h.back = try tf.deepseek_v41_backend.Backend.init(gpa, model_ptr);
    h.back.pair = h.pair;
    errdefer h.back.deinit();
    const window: u32 = model_ptr.caches[0].plan.capacity;
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), ds.prefill_step, ds.prefill_step -| 1);
    errdefer h.cfg.deinit(gpa);
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{
        .name = "deepseek-v41-zig",
        .lanes = model_ptr.lanes,
        .context_window = window,
        .prefill_step = ds.prefill_step,
        .greedy_only = true,
    });
    h.host.explain = .{ .text = words };
    h.warm = .{ .queue = model_ptr.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    errdefer h.host.stop();
    if (h.pair) |pair| if (pair.follows()) {
        h.follower = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, tf.deepseek_v41_tp_pair.Pair.follow, .{pair});
    };
    std.log.info("DeepSeek V4.1 resident loaded in {d:.1} s, {d:.2} GB admitted, {d} lane slots, context {d} tokens, greedy only", .{ model_ptr.load_seconds, @as(f64, @floatFromInt(model_ptr.admitted_bytes)) / 1e9, model_ptr.lanes, window });
    return .{ .engine = h.host.engine(), .close = close, .ctx = h };
}

pub fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.host.stop();
    if (h.pair) |pair| pair.stop();
    if (h.follower) |thread| thread.join();
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    h.back.deinit();
    if (h.pair) |pair| pair.deinit();
    h.model.deinit();
    h.gpa.destroy(h);
}
