//! Qwen3 embeddings on Metal, using the shared prompt lane and /v1/embeddings route.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen3_metal;
const lanes = tf.lanes;
const Prompt = q.Prompt(api);
const folder = q.Folder(api);
const fit = @import("cache_fit.zig");
const windows = @import("cuda_window.zig");
const Allocator = std.mem.Allocator;
const Host = struct {
    gpa: Allocator,
    e: *q.Engine,
    p: Prompt,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    startup: []const u8,
    fn close(ptr: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        h.p.deinit();
        h.e.deinit();
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }
    fn halt(ptr: *anyopaque, reason: []const u8) void {
        const h: *Host = @ptrCast(@alignCast(ptr));
        h.host.halt(reason);
    }
};
fn memory(_: ?*anyopaque, reset: bool) ?api.Memory {
    return fit.hostMemory(0, reset);
}
pub fn refusal(a: Allocator, o: api.Open) !?[]const u8 {
    if (try o.slideRefusal(a, "Macs")) |why| return why;
    if (o.kv8) return "--kv-cache fp8 keeps fp8 caches on CUDA; the Metal engines keep bf16: serve without it";
    if (o.tp != 1 or o.rank != 0) return "--tp: Qwen3 embeddings on Metal use one device";
    if (o.precision != null) return "--precision: Qwen3 embedding checkpoints use their stored BF16 precision";
    if (o.loops != null) return "--loops: Qwen3 embedding checkpoints have no recurrent blocks";
    if (o.drafter != null) return "--drafter: Qwen3 dense checkpoints have no drafts to train; serve without --drafter";
    if (o.record != null) return "--record: qwen3 checkpoints give no rows to record on Metal yet";
    if (o.vision != null) return "--vision: Qwen3 embedding checkpoints serve text only";
    if (o.speed_up != null) return "--speed-up: Qwen3 embeddings on Metal use one device";
    if (o.learn != null or o.recall or o.spill != .off) return "--learn, --recall and --spill-gib: Qwen3 embeddings keep no prompt states";
    return null;
}
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    return start(a, gpa, io, o, problem) catch |err| switch (err) {
        error.Refused => null,
        else => return err,
    };
}
fn refuse(problem: *[]const u8, why: []const u8) error{Refused} {
    problem.* = why;
    return error.Refused;
}
fn start(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !api.Opened {
    if (try refusal(a, o)) |why| return refuse(problem, why);
    const head = try folder.headOf(a, io, o.dir, "Metal", problem);
    if (head == .rerank) return refuse(problem, "qwen3 on Metal serves embeddings only; reranking is not implemented");
    const spec = q.config.read(a, io, o.dir) catch |err| return refuse(problem, try std.fmt.allocPrint(a, "qwen3: config.json is unsupported on Metal ({s})", .{@errorName(err)}));
    const asked = windows.contextWindow(o.context, @intCast(spec.max_positions), 32768) catch |err|
        return refuse(problem, try std.fmt.allocPrint(a, "--context: invalid Qwen3 embedding window ({s}); the checkpoint allows {d} tokens", .{ @errorName(err), spec.max_positions }));
    const e = q.Engine.load(gpa, io, o.dir) catch |err| return refuse(problem, try std.fmt.allocPrint(a, "qwen3: cannot load this BF16 embedding checkpoint on Metal ({s}); its dense prompt path requires M5 or later tensor units", .{@errorName(err)}));
    errdefer e.deinit();
    const cap = try fit.ceiling(fit.ram() orelse 0, e.model.device.maxWorkingSet(), null);
    const held = @max(e.model.checkpoint.residentBytes(), fit.footprint() orelse 0);
    const room = cap.bytes -| held -| fit.MARGIN;
    const window = (try folder.fitWindow(@intCast(asked), room, q.engine.least_window, e, q.Engine.scratchBytes)) orelse
        return refuse(problem, "qwen3: the Metal memory budget fits no embedding pass; free memory or lower --context");
    try e.size(window);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.e = e;
    h.p = Prompt.init(gpa, e, head.embedding.normalize);
    errdefer h.p.deinit();
    h.cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.p.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.startup = try std.fmt.allocPrint(gpa, "Metal: qwen3 embedding, BF16; {d}-token passes, {d:.2} GiB weights, {d:.2} GiB scratch; {s}", .{ window, fit.gibs(e.model.checkpoint.residentBytes()), fit.gibs(try e.scratchBytes(window)), if (head.embedding.normalize) "vectors unit-normed" else "vectors as pooled" });
    errdefer gpa.free(h.startup);
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = 1, .context_window = @intCast(window), .text_generation = false, .embeddings = .{ .dimensions = @intCast(spec.hidden), .vocabulary = @intCast(spec.vocab), .max_batch_tokens = @intCast(window), .max_inputs = q.engine.max_inputs, .mrl_min = @min(32, @as(u32, @intCast(spec.hidden))) }, .reasoning = false, .startup = h.startup });
    h.host.memory = .{ .read = memory };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .halt = Host.halt, .ctx = h };
}
test "Qwen3 Metal refuses unsupported knobs before checkpoint or device reads" {
    const a = std.testing.allocator;
    const base: api.Open = .{ .dir = "absent", .model_type = "qwen3" };
    try std.testing.expectEqual(@as(?[]const u8, null), try refusal(a, base));
    var o = base;
    o.kv8 = true;
    try std.testing.expect(std.mem.startsWith(u8, (try refusal(a, o)).?, "--kv-cache"));
    o = base;
    o.tp = 2;
    try std.testing.expect(std.mem.startsWith(u8, (try refusal(a, o)).?, "--tp:"));
    o = base;
    o.drafter = "absent";
    try std.testing.expect(std.mem.startsWith(u8, (try refusal(a, o)).?, "--drafter:"));
    o = base;
    o.record = "absent";
    try std.testing.expect(std.mem.startsWith(u8, (try refusal(a, o)).?, "--record:"));
    o = base;
    o.vision = .{};
    try std.testing.expect(std.mem.startsWith(u8, (try refusal(a, o)).?, "--vision:"));
}
