//! V4.1 lane slots on the native server: greedy, drafts off; load under 70% RAM unless GLM_LOAD_LIMIT_GB opts past.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const ds = tf.deepseek_v41;
const lanes = tf.lanes;
const cache_fit = @import("cache_fit.zig");
const Allocator = std.mem.Allocator;

pub const Host = struct {
    gpa: Allocator,
    model: *ds.Model,
    back: tf.deepseek_v41_backend.Backend,
    cfg: lanes.Config,
    wall: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    warm: mtl.keepalive.Target,
    rec: ?tf.deepseek_v41_profiler.Recorder,
    /// ROUND-467: kept prompt states (the retained prompt cache); null when no budget was asked for.
    cache: ?api.prompt_cache.Store = null,

    pub fn engine(h: *Host) api.Engine {
        return h.host.engine();
    }
};

fn words(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.GreedyOnly => "the native DeepSeek V4.1 engine decodes greedily only: send temperature 0",
        error.ContextFull => "the prompt and max_tokens do not fit this server's --context",
        error.NoFreeSlot => "every lane slot is busy",
        error.LoadLimitExceeded => "the resident payload does not fit the load limit: raise GLM_LOAD_LIMIT_GB or free memory",
        error.MissingEngramTables => "the checkpoint has no Engram tables: pass --engram-table-dir with the original release's shards",
        error.UnknownEngramLayer => "an Engram layer is not one the release shards carry: --engram-stream-dir takes the release's own two Engram shards",
        error.InvalidTokenMap => "engram-token-map.bin is missing or stale beside the checkpoint",
        else => null,
    };
}

/// Build the missing token map from the release's tokenizer/config beside the Engram tables; loads need no Python step.
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
    if (o.speed_up != null or o.slide or o.learn != null or
        o.drafter != null or o.tp != 1)
    {
        problem.* = "V4.1 resident serving does not support speed-up pairs, Sliding or Living Weights, a drafter or tensor parallelism yet";
        return null;
    }
    const map = try std.fmt.allocPrintSentinel(a, "{s}/engram-token-map.bin", .{o.dir}, 0);
    defer a.free(map);
    ensureTokenMap(a, io, map, o.engram_stream_dir orelse o.engram_table_dir);
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
    // Prefill round and workspace chunk. The family's compile-time constant (256) sized the workspace, so the lane cut
    // every prompt into 256-row rounds and re-decoded all 384 experts per round; TF_DSV41_PREFILL_STEP sets both now.
    const chunk: u32 = blk: {
        const raw = std.c.getenv("TF_DSV41_PREFILL_STEP") orelse break :blk ds.prefill_step;
        const text = std.mem.span(raw);
        const n = std.fmt.parseInt(u32, text, 10) catch break :blk ds.prefill_step;
        if (n < 16) break :blk ds.prefill_step;
        break :blk @min(n, 16384);
    };
    if (chunk != ds.prefill_step) std.log.info("deepseek-v41: prefill round {d} rows (TF_DSV41_PREFILL_STEP; constant is {d})", .{ chunk, ds.prefill_step });
    const model = try a.create(ds.Model);
    errdefer a.destroy(model);
    model.* = ds.Model.open(gpa, io, .{
        .dir = o.dir,
        .table_dir = o.engram_table_dir,
        .stream_dir = o.engram_stream_dir,
        .token_map = map,
        .context = if (o.context) |c| if (c > 0) @intCast(@min(c, std.math.maxInt(u32))) else 0 else 0,
        .lanes = @max(1, @min(o.lanes, 256)),
        .ram_limit_bytes = limit,
        .chunk_rows = chunk,
    }) catch |err| {
        a.destroy(model);
        if (err == error.InvalidTokenMap) {
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
        .rec = null,
    };
    h.back = try tf.deepseek_v41_backend.Backend.init(gpa, model_ptr);
    errdefer h.back.deinit();
    // Measurement only. The family's stage recorder exists but nothing on the served path enabled it, so there was
    // no way to attribute a prompt's GPU time to engram / mhc / attention / moe / head. Off unless the variable is set;
    // when off no span storage is allocated and profileSpan returns on its first line.
    if (std.c.getenv("TF_DSV41_HOST_PROF") != null) model_ptr.host_prof = true;
    if (std.c.getenv("TF_DSV41_PROFILE_GPU") != null) {
        h.rec = try tf.deepseek_v41_profiler.Recorder.init(gpa, model_ptr.device, .{ .enabled = true, .max_spans = 2048 });
        h.model.profile = &h.rec.?;
        std.log.info("deepseek-v41: stage recorder on (TF_DSV41_PROFILE_GPU)", .{});
    }
    const window: u32 = model_ptr.caches[0].plan.capacity;
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), chunk, chunk -| 1);
    {
        const f = h.back.facts();
        const lk = if (std.c.getenv("TF_DSV41_LOOKUP_DRAFTS")) |v| std.mem.span(v) else "null";
        const mp = if (std.c.getenv("TF_DSV41_MTP_DRAFTS")) |v| std.mem.span(v) else "null";
        const tp = if (std.c.getenv("TF_DSV41_TAPS")) |v| std.mem.span(v) else "null";
        std.log.info("deepseek-v41: draft env raw: lookup={s} mtp={s} taps={s} facts.mtp={} facts.speculate={} facts.drafts={} exact_width={} family_mtp={} most_drafts={}", .{ lk, mp, tp, f.mtp, f.speculate, f.drafts, f.exact_width, h.cfg.family_mtp, h.cfg.most_drafts });
    }
    errdefer h.cfg.deinit(gpa);
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{
        .name = "deepseek-v41-zig",
        .lanes = model_ptr.lanes,
        .context_window = window,
        .prefill_step = chunk,
        .greedy_only = false, // the port samples: temperature, top_k, top_p, min_p (backend.sampleRow)
        .penalties = true, // ROUND-466: presence/frequency, applied in backend.verify over each row s path
    });
    // ROUND-467: THE RETAINED PROMPT CACHE. The framework keys states by their token prefix, asks the family for a
    // copy at the chunk ends it marks (backend.snapSave / snapDrop) and hands the copy back as the stream's `reuse`,
    // which the family's prefill restores before its first chunk. A state is one slot's whole cache plus its cursor,
    // the four-token Engram history and the taps, so a resumed pass equals a fresh one; the budget is admitted under
    // the same load limit the model loaded with, and states are keyed by the prompt's own chunk grid.
    // ROUND-502: and the pass keeps a state at its own END (`.end_mark`, default on), so a later turn resumes at
    // its predecessor's exact boundary and reads its new rows as ONE chunk instead of the tile tail's extra
    // chunk (0.666 s off a 739-row turn at 22k, token-identical). TF_DSV41_END_MARK=0 is the same-binary opt-out.
    if (o.prompt_cache_gib) |gib| if (gib > 0) {
        var why: []const u8 = "";
        const plan = cache_fit.budgetUnder(limit, gib, o.prompt_cache_over_cap, 0, a, &why, "") catch |e| {
            if (e == error.CacheOverCap) problem.* = why;
            return e;
        };
        h.host.info_.prompt_cache_plan = plan;
        if (plan.budget_bytes > 0) {
            const B = tf.deepseek_v41_backend.Backend;
            h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = &h.back, .vtable = &.{ .bytes = B.snapBytes, .save = B.snapSave, .restore = B.snapRestore, .drop = B.snapDrop } }, .{ .lookahead = 0, .planned = true, .grid = model_ptr.workspace.plan.rows, .tail = true, .tail_align = 128, .end_mark = if (std.c.getenv("TF_DSV41_END_MARK")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true }, plan.budget_bytes);
            h.host.cache = &h.cache.?;
            h.host.info_.prompt_cache = true;
        }
    };
    h.host.explain = .{ .text = words };
    h.warm = .{ .queue = model_ptr.queue };
    h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    errdefer h.host.stop();
    std.log.info("DeepSeek V4.1 resident loaded in {d:.1} s, {d:.2} GB admitted, {d} lane slots, context {d} tokens, greedy only", .{ model_ptr.load_seconds, @as(f64, @floatFromInt(model_ptr.admitted_bytes)) / 1e9, model_ptr.lanes, window });
    if (std.c.getenv("TF_DSV41_KROW_CHECK") != null) krowCheck(model_ptr) catch |err|
        std.log.err("krow: the check did not run: {s}", .{@errorName(err)});
    return .{ .engine = h.host.engine(), .close = close, .ctx = h };
}

/// A draft's verification forward must agree with the one-row greedy stream row for row, or batched rows carry
/// different numerics and a drafter would burn compute silently. TF_DSV41_KROW_CHECK runs the comparison at load.
fn argmaxRow(logits: []const f32) u32 {
    var best: u32 = 0;
    var maximum = -std.math.inf(f32);
    for (logits, 0..) |value, id| if (value > maximum) {
        maximum = value;
        best = @intCast(id);
    };
    return best;
}

fn krowCheck(m: *ds.Model) !void {
    const prompt: usize = 96;
    var ids: [prompt]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(1200 + (i * 37) % 9000);
    // The one-row stream: a prompt chunk, then a token at a time.
    var stream: [9]u32 = undefined;
    stream[0] = argmaxRow(try m.forward(&ids, .last));
    for (1..stream.len) |i| {
        const one = [1]u32{stream[i - 1]};
        stream[i] = argmaxRow(try m.forward(&one, .last));
    }
    // The same prompt, then the first k tokens as one k-row verification forward.
    for ([_]usize{ 4, 8 }) |width| {
        try m.reset();
        _ = try m.forward(&ids, .none);
        const rows = try m.forward(stream[0..width], .all);
        var agree: usize = 0;
        for (0..width) |r| {
            const got = argmaxRow(rows[r * 129280 ..][0..129280]);
            if (got == stream[r + 1]) agree += 1 else std.log.err("krow: k={d} row {d}: batched {d}, one-row {d}", .{ width, r, got, stream[r + 1] });
        }
        std.log.info("krow: k={d}, {d}/{d} rows agree with the one-row stream", .{ width, agree, width });
    }
    try m.reset();
}

pub fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.host.stop();
    if (h.cache) |*store| store.deinit();
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    h.back.deinit();
    if (h.rec) |*r| r.deinit() catch {};
    h.model.deinit();
    h.gpa.destroy(h);
}
