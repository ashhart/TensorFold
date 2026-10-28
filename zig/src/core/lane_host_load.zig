//! A lane host's part of the load ledger: its counts and counters, published at each unlock that changed them.
const std = @import("std");
const ledger = @import("load_ledger.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

const Published = struct { requests: ?ledger.Requests = null, work: ?ledger.Work = null, capacity: ?ledger.Capacity = null };

/// The host's ledger state; counters move under the host mutex, the producer and `last` under `mutex`.
pub const State = struct {
    /// The producer and the values last sent; taken after the host mutex, before the ledger's.
    mutex: std.Io.Mutex = .init,
    producer: ?ledger.Producer = null,
    /// What the ledger holds from this host; null makes the next unlock publish.
    last: ?Published = null,
    started: u64 = 0,
    requeued: u64 = 0,
    yields: u64 = 0,
    prefill_compute: u64 = 0,
    prefill_cache: u64 = 0,
    decode: u64 = 0,
    busy_ns: u64 = 0,
    /// The pool's last census, from the thread that changed the pool (under `mutex`); null: no pool watched.
    pool: ?ledger.Pool = null,
};

/// The `capacity` group's source, set before the engine thread starts: how this engine holds its KV state.
pub const Source = struct { mode: ledger.Mode };

/// One attempt's prompt pass as the ledger counts it; the engine thread owns it.
pub const Pass = struct {
    on: bool = false,
    /// `fill_at` when the pass began: a proposed reuse position the backend has not confirmed yet.
    hint: u32 = 0,
    /// Computed tokens of this attempt already counted.
    credited: u32 = 0,
    /// When the running prompt call began, or its time was last taken (0: no call timed).
    call_ns: i96 = 0,
    /// Call time not yet counted: it moves into `busy_ns` only with this pass's own tokens.
    pending_ns: u64 = 0,
};

/// Engine.load: the server's producer at attach (published at once), null at detach.
pub fn attach(ctx: *anyopaque, producer: ?ledger.Producer) void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    if (producer == null) {
        h.load.mutex.lockUncancelable(h.io);
        defer h.load.mutex.unlock(h.io);
        h.load.producer = null;
        return;
    }
    h.lock();
    h.load.mutex.lockUncancelable(h.io);
    h.load.producer = producer;
    h.load.last = null;
    if (h.load.pool) |census| producer.?.publishPool(census); // an idle engine shows its capacity at once
    h.load.mutex.unlock(h.io);
    h.unlock(); // its publish sends the first values
}

/// Sends the host's groups when they changed since the last send; the caller holds the host mutex.
pub fn publish(h: *LaneHost) void {
    const s = &h.load;
    s.mutex.lockUncancelable(h.io);
    defer s.mutex.unlock(h.io);
    const p = s.producer orelse return;
    const now: Published = .{ .requests = requests(h), .work = work(h), .capacity = capacity(h) };
    if (s.last) |last| if (std.meta.eql(last, now)) return;
    s.last = now;
    p.publish(now.requests, now.work, now.capacity);
}

/// A prompt pass starts, a new attempt from zero: a requeued job's repeated work counts again.
pub fn began(h: *LaneHost, job: *Job) void {
    h.lock();
    defer h.unlock();
    job.load = .{ .on = true, .hint = job.stream.fill_at.load(.monotonic) };
}

/// A backend prompt call for `job` starts now (engine thread).
pub fn calling(h: *LaneHost, job: *Job) void {
    job.load.call_ns = clock(h);
}

/// A prompt call returned with its pass still open, as a lone driver's hand-off: its time stays with the pass.
pub fn returned(h: *LaneHost, job: *Job) void {
    h.lock();
    defer h.unlock();
    if (job.load.on) take(h, &job.load);
}

/// After a chunk call that returned: its time, and the work it reported when the pass's start is certain.
pub fn progress(h: *LaneHost, job: *Job) void {
    h.lock();
    defer h.unlock();
    const p = &job.load;
    if (!p.on) return; // reported already, as by an early first token inside the chunk
    take(h, p);
    if (!h.core.backend.vtable.reports_fill) return;
    const s = &job.stream;
    if (!(p.hint == 0 or s.cached == p.hint or s.reuse_failed)) return; // a start in doubt counts at the report
    const start = if (s.reuse_failed) 0 else @max(s.cached, p.hint);
    const d = (s.fill_at.load(.monotonic) -| start) -| p.credited;
    if (d == 0) return;
    h.load.prefill_compute += d;
    p.credited += d;
    h.load.busy_ns += p.pending_ns;
    p.pending_ns = 0;
}

/// The pass reported: its remaining computed tokens, its restored ones and its time; the caller holds the host mutex.
pub fn credit(h: *LaneHost, job: *Job) void {
    const p = &job.load;
    if (!p.on) return;
    take(h, p);
    const s = &job.stream;
    const c = (@as(u32, @intCast(job.request.prompt.len)) -| s.cached) -| p.credited;
    h.load.prefill_cache += s.cached;
    h.load.prefill_compute += c;
    if (c > 0) h.load.busy_ns += p.pending_ns; // a pass its progress already covered adds no time without tokens
    job.load = .{};
}

/// One batched call: every completed prompt and the call's time once, before any of them reports alone.
pub fn creditBatch(h: *LaneHost, jobs: []const *Job, done: []const bool, t0: i96) void {
    h.lock();
    defer h.unlock();
    var compute: u64 = 0;
    for (jobs, done) |job, d| {
        if (!d or !job.load.on) continue; // a prompt whose pass failed counts nothing
        const s = &job.stream;
        compute += (@as(u32, @intCast(job.request.prompt.len)) -| s.cached) -| job.load.credited;
        h.load.prefill_cache += s.cached;
        job.load = .{};
    }
    h.load.prefill_compute += compute;
    if (compute > 0) h.load.busy_ns += @intCast(@max(0, clock(h) - t0));
}

/// The pass ends without a report: its time leaves with its uncounted tokens; the caller holds the host mutex.
pub fn ended(job: *Job) void {
    job.load = .{};
}

/// `n` output tokens committed now; a request's first comes from its prompt pass; the caller holds the host mutex.
pub fn decoded(h: *LaneHost, n: usize, first: bool) void {
    h.load.decode += n - @intFromBool(first and n > 0);
}

pub fn clock(h: *const LaneHost) i96 {
    return std.Io.Clock.awake.now(h.io).toNanoseconds();
}

/// The time since the call began, or since the last take, into the pass's pending time.
fn take(h: *const LaneHost, p: *Pass) void {
    if (p.call_ns == 0) return;
    const t = clock(h);
    p.pending_ns += @intCast(@max(0, t - p.call_ns));
    p.call_ns = t;
}

/// The `work` group; a family round planner mixes streams in one pass, so its time has no pass to go to.
fn work(h: *const LaneHost) ?ledger.Work {
    if (h.rounds != null) return null;
    return .{ .prefill_compute = h.load.prefill_compute, .prefill_cache = h.load.prefill_cache, .decode = h.load.decode, .prefill_busy_ns = h.load.busy_ns };
}

/// A pool change, on the thread that made it: the census goes to the ledger without the host lock.
pub fn poolChanged(h: *LaneHost, census: ledger.Pool) void {
    const s = &h.load;
    s.mutex.lockUncancelable(h.io);
    defer s.mutex.unlock(h.io);
    s.pool = census;
    if (s.producer) |p| p.publishPool(census);
}

/// The `capacity` group's host part: the mode and the lanes no admitted request holds.
fn capacity(h: *const LaneHost) ?ledger.Capacity {
    const c = h.capacity orelse return null;
    const lanes: usize = @min(h.info_.lanes, h.memory_lanes orelse h.info_.lanes);
    return .{ .mode = c.mode, .lanes_free = @intCast(lanes -| h.admitted.items.len) };
}

/// A background pass gave its lane up to foreground work.
pub fn yielded(h: *LaneHost) void {
    h.lock();
    defer h.unlock();
    h.load.yields += 1;
}

/// The `requests` group from the host's lists; the caller holds the host mutex.
fn requests(h: *const LaneHost) ledger.Requests {
    var waiting_tokens: u64 = 0;
    for (h.queued.items) |job| waiting_tokens += job.request.prompt.len;
    const left = filling(h);
    return .{
        .running = @intCast(h.admitted.items.len),
        .waiting = @intCast(h.queued.items.len),
        .waiting_prompt_tokens = waiting_tokens,
        .filling_left = left,
        .fill_unknown = left == null,
        .started = h.load.started,
        .requeued = h.load.requeued,
        .yields = h.load.yields,
    };
}

/// Prompt tokens left in the foreground passes filling now: 0 for none, null when the backend reports no progress.
fn filling(h: *const LaneHost) ?u64 {
    var left: u64 = 0;
    var any = false;
    for (h.admitted.items) |job| {
        if (!job.fill_on.load(.acquire) or job.request.background) continue; // as the dashboard's fill totals
        any = true;
        left += job.request.prompt.len -| job.stream.fill_at.load(.monotonic);
    }
    if (any and !h.core.backend.vtable.reports_fill) return null;
    return left;
}

test {
    _ = @import("lane_host_load_test.zig");
    _ = @import("lane_host_load_account_test.zig");
}
