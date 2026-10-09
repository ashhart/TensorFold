//! Qwen's state and GPU completion callbacks behind the shared native serial host.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const Serial = api.serial_host;
const pc = api.prompt_cache;
const cache_fit = @import("cache_fit.zig");
const Resident = @import("resident.zig").Resident;
const Draft = @import("qwen27_draft.zig").Draft;
pub const lanes_backend = @import("qwen27_lanes.zig");
const LaneDecode = @import("qwen27_lane_decode.zig").Decode;
const Allocator = std.mem.Allocator;

pub const Host = struct {
    gpa: Allocator,
    model: *q.model.Model,
    runner: q.decode_round.Runner,
    serial: *Serial.Host,
    owns_model: bool,
    draft: ?*Draft = null,
    /// Drafted decode on the shared lane core (TF_QWEN27_LANES=1); null: the DFlash2 serial loop.
    lane: ?*LaneDecode = null,
    /// Kept prompt states (the serial host's thread only); null: no reuse.
    cache: ?pc.Store = null,
    resident: ?Resident = null,
    warm: mtl.keepalive.Target, // the model's queue (and its resident set once held), for the server's idle ticker
    pub fn engine(h: *Host) api.Engine {
        return h.serial.engine();
    }
    fn self(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    fn reset(ptr: *anyopaque) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const h = self(ptr);
        if (h.lane) |l| l.end();
        if (h.draft) |draft| return draft.reset();
        try h.runner.reset(0);
    }
    fn chunk(ptr: *anyopaque, ids: []const u32, last: bool) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const h = self(ptr);
        if (h.draft) |draft| return draft.promptChunk(ids, last);
        try (q.session.Session{ .runner = &h.runner }).promptChunk(ids, last);
    }
    fn decodeChunk(ptr: *anyopaque, ids: []const u32, last: bool) !void {
        const h = self(ptr);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (h.draft) |draft| return draft.decodeChunk(ids, last);
        try (q.session.Session{ .runner = &h.runner }).decodeChunk(ids, last);
    }
    fn fed(ptr: *anyopaque) u64 {
        return self(ptr).runner.offsets[0];
    }
    fn advance(ptr: *anyopaque, token: u32) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const h = self(ptr);
        if (h.draft) |draft| return draft.advance(token);
        try (q.session.Session{ .runner = &h.runner }).step(token);
    }
    fn batch(ptr: *anyopaque, budget: u32, eos: []const u32) !Serial.Batch {
        const h = self(ptr);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (h.lane) |l| return l.batch(&h.runner);
        const draft = h.draft orelse return error.DraftNotAttached;
        return draft.batch(budget, eos);
    }
    fn begin(ptr: *anyopaque, prompt: []const u32, max_tokens: u32, eos: []const u32) !void {
        if (self(ptr).lane) |l| try l.begin(prompt, max_tokens, eos);
    }
    /// Drafted requests decode on the lane core by default before the M5 (TF_QWEN27_LANES=1 or 0 overrides).
    fn laneDecode(h: *Host, io: std.Io) !void {
        const on = if (std.c.getenv("TF_QWEN27_LANES")) |v| std.mem.eql(u8, std.mem.span(v), "1") else !h.runner.model.device.tensorUnits();
        if (!on) return;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        h.lane = try LaneDecode.init(h.gpa, io, &h.runner, h.draft.?);
        h.serial.driver.draft_begin = Host.begin;
    }
    fn draw(ptr: *anyopaque, sampling: ?api.Sampling) !u32 {
        const h = self(ptr);
        if (h.runner.failed or h.runner.active != null or h.runner.offsets[0] == 0) return error.RoundNotReady;
        if (sampling) |s| return Serial.choose(h.gpa, .{ .bf16 = h.model.frame.get(.logits).buffer.slice(u16, h.model.config.vocab) }, h.runner.offsets[0], s);
        return (q.session.Session{ .runner = &h.runner }).greedy();
    }
};
/// The prompt cache's copies of the runner's and the drafter's state (snapshot.zig).
const Snaps = struct {
    fn host(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    fn drafter(h: *Host) ?*q.dflash.runtime_model.Model {
        return if (h.draft) |d| d.model else null;
    }
    fn bytes(ptr: *anyopaque, at: u32) u64 {
        const h = host(ptr);
        return q.snapshot.bytes(&h.runner, drafter(h), at);
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const h = host(ptr);
        return @ptrCast(try q.snapshot.save(h.gpa, &h.runner, drafter(h), at));
    }
    fn restore(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const h = host(ptr);
        try q.snapshot.restore(h.gpa, &h.runner, drafter(h), @ptrCast(@alignCast(saved)));
        if (h.draft) |d| d.first_batch = true;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        q.snapshot.drop(host(ptr).gpa, @ptrCast(@alignCast(saved)));
    }
};
/// Prompt reuse between turns, once any drafter is attached; on error.CacheOverCap `why` (in `a`) says why.
pub fn enableCache(h: *Host, gib: ?f64, over: bool, a: Allocator, why: *[]const u8) !void {
    const dev = h.model.device;
    const budget = try cache_fit.budget(gib, over, @min(dev.maxWorkingSet() -| dev.allocated() -| (8 << 30), 16 << 30), a, why, "");
    if (budget == 0) return;
    h.serial.mutex.lockUncancelable(h.serial.io);
    defer h.serial.mutex.unlock(h.serial.io);
    if (h.serial.running != null or h.serial.queued.items.len != 0 or h.serial.closing) return error.HostBusy;
    if (h.cache != null) return error.CacheAlreadyEnabled;
    h.cache = pc.Store.init(h.gpa, .{ .ptr = h, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .warm = true }, budget);
    h.serial.driver.cache = &h.cache.?;
    h.serial.driver.info.warm_turns = true;
}
/// Weights, runner and drafter states join one residency set the idle keepalive uses, once drafts attach.
pub fn holdResident(h: *Host) !void {
    if (h.resident != null) return error.AlreadyResident;
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    for (h.model.checkpoint.shards.items) |s| try list.append(h.gpa, s.buffer);
    try list.appendSlice(h.gpa, h.model.weights.owned.items);
    try list.appendSlice(h.gpa, &h.model.frame.buffers);
    try list.appendSlice(h.gpa, &h.runner.gdn.buffers);
    for (h.runner.caches) |c| try list.appendSlice(h.gpa, &.{ c.keys.buffer, c.values.buffer });
    if (h.draft) |d| {
        for (d.model.checkpoint.shards.items) |s| try list.append(h.gpa, s.buffer);
        try list.append(h.gpa, d.model.pending);
    }
    h.resident = try Resident.init(h.model.device, h.model.queue, list.items);
    h.warm = (&h.resident.?).target();
}
pub fn attach(gpa: Allocator, io: std.Io, model: *q.model.Model, context: u32, owns_model: bool) !*Host {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const capacity = if (context == 0) @min(model.config.max_position, 32768) else context;
    if (capacity == 0 or capacity > model.config.max_position) return error.ContextFull;
    var runner = try q.decode_round.Runner.init(gpa, model, 1, @intCast(capacity));
    errdefer runner.deinit();
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .model = model, .runner = runner, .serial = undefined, .owns_model = owns_model, .warm = .{ .queue = model.queue } };
    h.serial = try Serial.Host.init(gpa, io, .{ .ctx = h, .info = .{ .name = "qwen27", .lanes = 1, .context_window = @intCast(capacity), .prefill_step = 128 }, .reset = Host.reset, .prompt_chunk = Host.chunk, .draw = Host.draw, .advance = Host.advance, .decode_chunk = Host.decodeChunk, .fed = Host.fed, .keepalive = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick } });
    return h;
}
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, context: u32) !*Host {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try q.model.Model.load(gpa, io, dir, 128);
    errdefer model.deinit();
    return attach(gpa, io, model, context, true);
}
pub fn enableDraft(h: *Host, io: std.Io, dir: []const u8, bits: u8) !void {
    h.serial.mutex.lockUncancelable(h.serial.io);
    defer h.serial.mutex.unlock(h.serial.io);
    if (h.serial.running != null or h.serial.queued.items.len != 0 or h.serial.closing) return error.HostBusy;
    if (h.draft != null or (bits != 0 and bits != 4)) return error.BadDraftOptions;
    h.draft = try Draft.open(h.gpa, io, h.model, &h.runner, dir, if (bits == 4) .prepared_q4_reference else .bf16_reference);
    try h.laneDecode(io);
    h.serial.driver.draft_batch = Host.batch;
}
pub fn attachDraft(h: *Host, model: *q.dflash.runtime_model.Model, owns_model: bool) !void {
    h.serial.mutex.lockUncancelable(h.serial.io);
    defer h.serial.mutex.unlock(h.serial.io);
    if (h.serial.running != null or h.serial.queued.items.len != 0 or h.serial.closing) return error.HostBusy;
    if (h.draft != null) return error.DraftAlreadyAttached;
    h.draft = try Draft.attach(h.gpa, &h.runner, model, owns_model);
    try h.laneDecode(h.serial.io);
    h.serial.driver.draft_batch = Host.batch;
}
pub fn close(ptr: *anyopaque) void {
    const h = Host.self(ptr);
    h.serial.deinit();
    if (h.resident) |*r| r.deinit();
    if (h.cache) |*store| store.deinit();
    if (h.lane) |l| l.deinit();
    if (h.draft) |draft| draft.deinit();
    h.runner.deinit();
    if (h.owns_model) h.model.deinit();
    h.gpa.destroy(h);
}
