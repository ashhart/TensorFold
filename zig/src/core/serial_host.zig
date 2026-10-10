//! One serialized native worker owns request lifetimes, prompt chunk boundaries and cancellation between calls.
const std = @import("std");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const Allocator = std.mem.Allocator;
pub const Batch = struct { tokens: []const u32, stats: api.Stats = .{} };
/// A drafted round's tokens at most: a widest window's rows (its drafts and the bonus).
pub const max_round = 128;
pub const Driver = struct {
    ctx: *anyopaque,
    info: api.Info,
    reset: *const fn (*anyopaque) anyerror!void,
    prompt_chunk: *const fn (*anyopaque, []const u32, bool) anyerror!void,
    draw: *const fn (*anyopaque, ?api.Sampling) anyerror!u32,
    advance: *const fn (*anyopaque, u32) anyerror!void,
    /// Receives true remaining request tokens and returns one complete round of at most `max_round` tokens.
    draft_batch: ?*const fn (*anyopaque, u32, []const u32) anyerror!Batch = null,
    /// Called once before a drafted request's first batch with its prompt, token budget and stop tokens.
    draft_begin: ?*const fn (*anyopaque, []const u32, u32, []const u32) anyerror!void = null,
    /// Kept prompt states: a prompt resumes at its longest kept prefix (restored, not `reset`) and keeps marks.
    cache: ?*pc.Store = null,
    /// A prompt chunk with decoded rows' arithmetic (Request.decode_spans); set: finished replies are kept too.
    decode_chunk: ?*const fn (*anyopaque, []const u32, bool) anyerror!void = null,
    /// Tokens the driver's state has consumed, for checking a reply's state before it is kept.
    fed: ?*const fn (*anyopaque) u64 = null,
    /// The family's queue as the server's idle keepalive target (Metal); null keeps the ticker off.
    keepalive: ?api.keepalive.Target = null,
};
const Job = struct { id: api.Id, request: *const api.Request, sink: api.Sink };

pub const Host = struct {
    gpa: Allocator,
    io: std.Io,
    driver: Driver,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    running: ?api.Id = null,
    canceled: bool = false,
    closing: bool = false,
    thread: ?std.Thread = null,
    prompt_count: u32 = 0,
    generated_count: u32 = 0,
    prefill_rate: f64 = 0,
    decode_rate: f64 = 0,

    pub fn init(gpa: Allocator, io: std.Io, driver: Driver) !*Host {
        if (driver.info.context_window == 0 or driver.info.prefill_step == 0) return error.BadDriver;
        const h = try gpa.create(Host);
        errdefer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .io = io, .driver = driver };
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, work, .{h});
        return h;
    }
    pub fn deinit(h: *Host) void {
        h.lock();
        h.closing = true;
        h.wake.broadcast(h.io);
        h.unlock();
        if (h.thread) |thread| thread.join();
        h.queued.deinit(h.gpa);
        h.gpa.destroy(h);
    }
    pub fn engine(h: *Host) api.Engine {
        return .{ .ctx = h, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .keepalive = keepalive } };
    }
    fn keepalive(ptr: *anyopaque) ?api.keepalive.Target {
        return self(ptr).driver.keepalive;
    }
    fn self(ptr: *anyopaque) *Host {
        return @ptrCast(@alignCast(ptr));
    }
    fn lock(h: *Host) void {
        h.mutex.lockUncancelable(h.io);
    }
    fn unlock(h: *Host) void {
        h.mutex.unlock(h.io);
    }
    fn info(ptr: *anyopaque) api.Info {
        const h = self(ptr);
        var i = h.driver.info;
        i.plain_only = h.driver.draft_batch == null;
        return i;
    }
    fn submit(ptr: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        try modes.validate(request.decode_spans);
        const h = self(ptr);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .id = id, .request = request, .sink = sink };
        h.lock();
        defer h.unlock();
        if (h.closing) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        if (h.queued.items.len >= 256) {
            h.gpa.destroy(job);
            return error.Busy;
        }
        var at = h.queued.items.len;
        if (!request.background) while (at > 0 and h.queued.items[at - 1].request.background) {
            at -= 1;
        };
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }
    fn cancel(ptr: *anyopaque, id: api.Id) void {
        const h = self(ptr);
        h.lock();
        for (h.queued.items, 0..) |job, i| if (job.id == id) {
            _ = h.queued.orderedRemove(i);
            h.unlock();
            h.finish(job, .cancelled, .{}, "");
            return;
        };
        if (h.running == id) h.canceled = true;
        h.unlock();
    }
    fn status(ptr: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        const h = self(ptr);
        h.lock();
        defer h.unlock();
        const live = h.running != null;
        if (live and stream_tokens.len > 0) stream_tokens[0] = h.prompt_count + h.generated_count;
        out.* = .{ .running = @intFromBool(live), .waiting = @intCast(h.queued.items.len), .streams = if (live and stream_tokens.len > 0) 1 else 0, .generation_tokens = h.generated_count, .prefill_tokens_per_second = h.prefill_rate, .decode_tokens_per_second = h.decode_rate };
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn emit(job: *Job, event: api.Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }
    fn finish(h: *Host, job: *Job, reason: api.Reason, stats: api.Stats, message: []const u8) void {
        h.lock();
        if (h.running == job.id) {
            h.running = null;
            h.prompt_count = 0;
            h.generated_count = 0;
        }
        h.unlock();
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        h.gpa.destroy(job);
    }
    /// A foreground request waits (queued ahead of every background one).
    fn foregroundQueued(h: *Host) bool {
        h.lock();
        defer h.unlock();
        return h.queued.items.len > 0 and !h.queued.items[0].request.background;
    }
    fn stopRequested(h: *Host) bool {
        h.lock();
        defer h.unlock();
        return h.closing or h.canceled;
    }
    fn work(h: *Host) void {
        while (true) {
            h.lock();
            while (h.queued.items.len == 0 and !h.closing) h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
            if (h.closing) {
                while (h.queued.items.len > 0) {
                    const job = h.queued.orderedRemove(0);
                    h.unlock();
                    h.finish(job, .cancelled, .{}, "");
                    h.lock();
                }
                h.unlock();
                return;
            }
            const job = h.queued.orderedRemove(0);
            h.running = job.id;
            h.canceled = false;
            h.prompt_count = 0;
            h.generated_count = 0;
            h.unlock();
            h.serve(job) catch |err| h.finish(job, .failed, .{}, @errorName(err));
        }
    }
    fn serve(h: *Host, job: *Job) !void {
        const request = job.request;
        if (request.prompt.len == 0 or @as(u64, request.prompt.len) + request.max_tokens > h.driver.info.context_window) return error.ContextFull;
        if (request.call != null or request.structure != null or request.think_budget != 0 or request.loop_guard) return error.UnsupportedRequest;
        var arena: std.heap.ArenaAllocator = .init(h.gpa);
        defer arena.deinit();
        const decode_spans: []const [2]u32 = if (h.driver.decode_chunk != null) request.decode_spans else &.{};
        var plan: pc.Plan = .{};
        if (h.driver.cache) |store| plan = store.begin(arena.allocator(), request.prompt, request.history_len, request.shared_prefixes, request.chunks, null, decode_spans) catch .{};
        if (plan.from == 0) try h.driver.reset(h.driver.ctx);
        const start = std.Io.Clock.awake.now(h.io);
        var first: usize = plan.from;
        var kept: u64 = 0;
        const warm = request.max_tokens == 0; // a prompt-only pass: its state is kept where it ends, for a later turn
        while (first < request.prompt.len) {
            if (h.stopRequested()) {
                h.finish(job, .cancelled, .{}, "");
                return;
            }
            if (warm and request.background and first > plan.from and h.foregroundQueued()) break; // yields, keeping what it prefilled
            var end = @min(request.prompt.len, first + @max(h.driver.info.prefill_step, 1));
            for (request.chunks) |cut| if (cut > first and cut < end) {
                end = cut;
            };
            for (plan.marks) |mark| if (mark > first and mark < end) {
                end = mark;
            };
            var decoded = false;
            for (request.decode_spans) |span| {
                for (span) |cut| if (cut > first and cut < end) {
                    end = cut;
                };
                decoded = decoded or (first >= span[0] and first < span[1]);
            }
            const chunk = if (decoded) h.driver.decode_chunk orelse h.driver.prompt_chunk else h.driver.prompt_chunk;
            try chunk(h.driver.ctx, request.prompt[first..end], end == request.prompt.len);
            first = end;
            if (h.driver.cache) |store| if (std.mem.indexOfScalar(u32, plan.marks, @intCast(first)) != null) {
                if (store.keep(request.prompt, @intCast(first), null, request.chunks, decode_spans)) kept += 1;
            };
            h.lock();
            h.prompt_count = @intCast(first);
            h.unlock();
        }
        if (h.stopRequested()) {
            h.finish(job, .cancelled, .{}, "");
            return;
        }
        const prefill_ns = start.durationTo(std.Io.Clock.awake.now(h.io)).toNanoseconds();
        h.lock();
        h.prefill_rate = @as(f64, @floatFromInt(first - plan.from)) * 1e9 / @as(f64, @floatFromInt(@max(prefill_ns, 1)));
        h.unlock();
        if (warm) if (h.driver.cache) |store| if (first > plan.from and store.keep(request.prompt, @intCast(first), null, request.chunks, decode_spans)) {
            kept += 1;
        };
        if (h.driver.cache) |store| store.report(first, plan.from, kept);
        emit(job, .{ .prefilled = plan.from });
        if (warm) return h.finish(job, if (first == request.prompt.len) .length else .cancelled, .{ .prefill_seconds = @as(f64, @floatFromInt(prefill_ns)) / 1e9 }, "");
        if (h.driver.draft_batch != null and request.drafts and request.sampling == null and request.stop == null) return h.serveDrafted(job, prefill_ns);
        var emitted: std.ArrayList(u32) = .empty;
        defer emitted.deinit(h.gpa);
        const draw_start = std.Io.Clock.awake.now(h.io);
        for (0..request.max_tokens) |i| {
            if (h.stopRequested()) {
                h.finish(job, .cancelled, .{ .rounds = i, .min_rows = 1, .prefill_seconds = @as(f64, @floatFromInt(prefill_ns)) / 1e9 }, "");
                return;
            }
            const token = try h.driver.draw(h.driver.ctx, request.sampling);
            try emitted.append(h.gpa, token);
            emit(job, .{ .tokens = &.{token} });
            h.lock();
            h.generated_count += 1;
            h.unlock();
            const stop = std.mem.indexOfScalar(u32, request.eos, token) != null or (if (request.stop) |check| check.check(check.ctx, emitted.items) else false);
            if (stop or i + 1 == request.max_tokens) {
                const ns = draw_start.durationTo(std.Io.Clock.awake.now(h.io)).toNanoseconds();
                h.lock();
                h.decode_rate = @as(f64, @floatFromInt(i + 1)) * 1e9 / @as(f64, @floatFromInt(@max(ns, 1)));
                h.unlock();
                h.keepReply(request, emitted.items);
                h.finish(job, if (stop) .stop else .length, .{ .rounds = i + 1, .min_rows = 1, .prefill_seconds = @as(f64, @floatFromInt(prefill_ns)) / 1e9 }, "");
                return;
            }
            try h.driver.advance(h.driver.ctx, token);
        }
        h.finish(job, .length, .{ .prefill_seconds = @as(f64, @floatFromInt(prefill_ns)) / 1e9 }, "");
    }
    /// A finished reply's state stands where a prompt pass of prompt and reply would: keep it for the next turn.
    fn keepReply(h: *Host, request: *const api.Request, reply: []const u32) void {
        const prompt = request.prompt;
        if (h.driver.decode_chunk == null or reply.len == 0) return;
        const store = h.driver.cache orelse return;
        const fed = h.driver.fed orelse return;
        const stands = fed(h.driver.ctx);
        if (stands + 1 < prompt.len + reply.len or stands > prompt.len + reply.len) return std.log.info("reply state not kept: the engine stands at {d} tokens, the reply ends at {d}", .{ stands, prompt.len + reply.len });
        const tokens = std.mem.concat(h.gpa, u32, &.{ prompt, reply }) catch return;
        defer h.gpa.free(tokens);
        const spans = modes.reply(h.gpa, request.decode_spans, @intCast(prompt.len), @intCast(stands)) catch return;
        defer h.gpa.free(spans);
        _ = store.keep(tokens, @intCast(stands), null, request.chunks, spans);
    }
    fn serveDrafted(h: *Host, job: *Job, prefill_ns: i96) !void {
        const request = job.request;
        const run = h.driver.draft_batch orelse return error.BadDriver;
        if (h.driver.draft_begin) |begin| try begin(h.driver.ctx, request.prompt, request.max_tokens, request.eos);
        var stats = api.Stats{ .prefill_seconds = @as(f64, @floatFromInt(prefill_ns)) / 1e9 };
        const start = std.Io.Clock.awake.now(h.io);
        var emitted: u32 = 0;
        var reply: std.ArrayList(u32) = .empty;
        defer reply.deinit(h.gpa);
        while (emitted < request.max_tokens) {
            if (h.stopRequested()) {
                h.finish(job, .cancelled, stats, "");
                return;
            }
            const remaining = request.max_tokens - emitted;
            const batch = try run(h.driver.ctx, remaining, request.eos);
            if (batch.tokens.len == 0 or batch.tokens.len > @min(max_round, remaining)) return error.BadDraftBatch;
            for (batch.tokens, 0..) |token, index| {
                if (std.mem.indexOfScalar(u32, request.eos, token) != null and index + 1 != batch.tokens.len) {
                    return error.BadDraftBatch;
                }
            }
            stats.rounds += batch.stats.rounds;
            stats.drafted += batch.stats.drafted;
            stats.accepted += batch.stats.accepted;
            if (batch.stats.min_rows > 0) stats.min_rows = if (stats.min_rows == 0) batch.stats.min_rows else @min(stats.min_rows, batch.stats.min_rows);
            if (batch.stats.telemetry_json.len > 0) stats.telemetry_json = batch.stats.telemetry_json;
            if (h.stopRequested()) {
                h.finish(job, .cancelled, stats, "");
                return;
            }
            emit(job, .{ .tokens = batch.tokens });
            emitted += @intCast(batch.tokens.len);
            if (h.driver.decode_chunk != null) try reply.appendSlice(h.gpa, batch.tokens);
            h.lock();
            h.generated_count = emitted;
            h.unlock();
            const stop = std.mem.indexOfScalar(u32, request.eos, batch.tokens[batch.tokens.len - 1]) != null;
            if (stop or emitted == request.max_tokens) {
                const ns = start.durationTo(std.Io.Clock.awake.now(h.io)).toNanoseconds();
                h.lock();
                h.decode_rate = @as(f64, @floatFromInt(emitted)) * 1e9 / @as(f64, @floatFromInt(@max(ns, 1)));
                h.unlock();
                h.keepReply(request, reply.items);
                h.finish(job, if (stop) .stop else .length, stats, "");
                return;
            }
        }
        h.finish(job, .length, stats, "");
    }
};

pub const Logits = union(enum) {
    bf16: []const u16,
    f32: []const f32,
    fn len(row: Logits) usize {
        return switch (row) {
            .bf16 => |v| v.len,
            .f32 => |v| v.len,
        };
    }
    fn get(row: Logits, index: usize) f64 {
        return switch (row) {
            .bf16 => |v| @as(f32, @bitCast(@as(u32, v[index]) << 16)),
            .f32 => |v| v[index],
        };
    }
};
pub fn choose(gpa: Allocator, row: Logits, position: u64, sampling: api.Sampling) !u32 {
    const n = row.len();
    if (n == 0 or n > std.math.maxInt(u32)) return error.BadLogits;
    const values = try gpa.alloc(f64, n);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, n);
    defer gpa.free(ids);
    for (values, ids, 0..) |*value, *id, i| {
        value.* = row.get(i);
        if (!std.math.isFinite(value.*)) return error.NonfiniteLogits;
        id.* = i;
    }
    return @intCast(try @import("lanes").sampling.choose(gpa, values, ids, position, sampling));
}

test "shared sampling preserves BF16 and FP32 keyed draws" {
    const a = std.testing.allocator;
    for (0..32) |position| {
        const bf = try choose(a, .{ .bf16 = &.{ 0xbf80, 0x3f80, 0x4000 } }, position, .{ .seed = 7 });
        const fp = try choose(a, .{ .f32 = &.{ -1, 1, 2 } }, position, .{ .seed = 7 });
        try std.testing.expectEqual(fp, bf);
    }
    try std.testing.expectError(error.NonfiniteLogits, choose(a, .{ .bf16 = &.{0x7fc0} }, 0, .{ .seed = 7 }));
}

test {
    _ = @import("serial_batch_test.zig");
    _ = @import("cache_mode_host_test.zig");
}
