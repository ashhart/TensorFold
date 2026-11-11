//! #1594 step 2b: a prompt prefix another machine filled, handed to its stream in place of a kept state.
const std = @import("std");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const lanes = @import("lanes");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = @import("lane_job.zig").Job;
const Request = api.Request;
const route = @import("lane_remote_route.zig");

/// The gate and decider every remote source keeps: one pull at a time, and a prompt goes to the pair only when it pays.
pub const Route = route.Route;
pub const Ask = route.Ask;
pub const Verdict = route.Verdict;
pub const Measured = route.Measured;
pub const Outcome = route.Outcome;
pub const Door = route.Door;
pub const Change = route.Change;
pub const pairDown = route.pairDown;

/// A family's remote prefill (fabric/prefill_source.zig behind it): opt-in, and outside the exactness contract.
pub const Source = struct {
    ptr: *anyopaque,
    /// Whether a queued prompt gets a remote prefix, or why not (a `later` reason is weighed again).
    wants: *const fn (ptr: *anyopaque, ask: Ask) Verdict,
    /// Fill a prefix beyond `cached` into new storage, blocking; `gone`: the request ended, fail `Aborted`.
    fill: *const fn (ptr: *anyopaque, prompt: []const u32, cached: u64, gone: *const std.atomic.Value(bool)) anyerror!?Filled,
    /// Free the storage `fill` returned.
    drop: *const fn (ptr: *anyopaque, saved: pc.Saved) void,
    /// Fills under way, the last thing each one's thread touches: the family waits for 0 before it goes.
    running: *std.atomic.Value(u32),
    /// A pass the lane host timed (this engine's own prompt pass, or the last-token pass after a pull), if it learns.
    measured: ?*const fn (ptr: *anyopaque, m: Measured) void = null,
};

/// A pull at its gate: ask the breaker, check a half-open pair is alive, run `p.run`, tell the breaker how it went.
pub fn guarded(p: anytype, prompt: []const u32, cached: u64, upto: u64, gone: *const std.atomic.Value(bool)) !?Filled {
    const door = p.route.door(Route.nowNs(p.io));
    if (door == .local) return null;
    var outcome: route.Outcome = .other;
    defer switch (p.route.finish(Route.nowNs(p.io), outcome)) {
        .none => {},
        .opened => std.log.warn("remote prefill: the pair is not answering; prompts are prefilled here, and one is tried against it in {d} s", .{p.route.breaker.cooldown_ns / std.time.ns_per_s}),
        .closed => std.log.info("remote prefill: the pair answers again", .{}),
    };
    if (door == .probe and !p.alive()) {
        outcome = .down;
        return null;
    }
    return p.run(prompt, cached, upto, gone, &outcome) catch |err| {
        outcome = if (route.pairDown(err)) .down else .other;
        return err;
    };
}

/// Where the backend resumes a prompt without any copy (slot 0's standing state): its own prefix, as good as cached.
pub const ResumePeek = struct { ctx: *anyopaque, at: *const fn (ctx: *anyopaque, prompt: []const u32, spans: []const [2]u32) u32 };

/// What a pull spent, in nanoseconds: at the gate, in the trigger's exchange, transferring (landing included), landing.
pub const Phases = struct { queue_ns: u64 = 0, trigger_ns: u64 = 0, pull_ns: u64 = 0, land_ns: u64 = 0, bytes: u64 = 0 };

/// The family's state after `at` prompt tokens, computed by another machine: never kept, never the engine's own.
pub const Filled = struct { saved: pc.Saved, at: u32, phases: Phases = .{} };

/// Link name for the server's own HTTP port, used by `--mcdma-produce http` and `--prefill-from URL --mcdma-link http`.
pub const http_link = "http";

/// serve --prefill-from URL (the GB10 server) and --mcdma-link NAME (mcdma-rpcd's connect link): both, or neither.
pub const Flags = struct {
    url: ?[]const u8 = null,
    link: ?[]const u8 = null,
    produce: ?[]const u8 = null, // --mcdma-produce NAME on a GB10: handoff requests' states exported on that link

    pub fn on(f: Flags) bool {
        return f.url != null or f.link != null;
    }

    /// `--mcdma-link http`: the hand-off comes over HTTP from `--prefill-from`'s server, not over a fabric link.
    pub fn overHttp(f: Flags) bool {
        const l = f.link orelse return false;
        return std.mem.eql(u8, l, http_link);
    }
};

/// A queued job's fill on its own thread: running, landed, or left by a job that ended first (the thread drops it).
pub const Ticket = struct {
    state: std.atomic.Value(u8) = .init(running),
    /// its job ended before the fill did: the fill is told, and stops waiting on the other machine
    gone: std.atomic.Value(bool) = .init(false),
    filled: ?Filled = null,
    prompt: []u32, // the thread's own copy: the request may go before the fill does
    cached: u64,
    host: *LaneHost,
    source: Source,

    const running: u8 = 0;
    const landed: u8 = 1;
    const left: u8 = 2;
};

/// Whether a request may take a remote prefix: decoded text, no rows of the engine's own needed, and not a handoff.
fn eligible(r: *const Request) bool {
    return !r.promptOnly() and !r.background and r.images.len == 0 and r.prompt_logprobs == null and !r.untaught and
        r.embedding == null and r.handoff == null;
}

/// Fills start for queued requests whose remote prefix beats the cache's, at most TF_PREFILL_AHEAD (engine thread).
pub fn queue(h: *LaneHost) void {
    const source = h.remote orelse return;
    keepWarm(h);
    var holding: usize = 0;
    const decoding: u32 = @intCast(h.admitted.items.len -| h.filling.items.len); // streams a prompt pass here would stall
    for (h.queued.items) |job| holding += @intFromBool(if (job.ticket) |t| t.state.load(.acquire) == Ticket.running or t.filled != null else false);
    for (h.queued.items) |job| {
        if (holding >= ahead()) return; // the rest are weighed once one of these is admitted
        if (job.ticket != null or job.remote_looked) continue;
        job.remote_looked = true;
        const r = job.request;
        if (!eligible(r)) continue;
        const kept: u64 = if (h.cache) |store| if (store.find(job.cacheKey(), r.chunks, h.spans(r))) |e| e.at else 0 else 0;
        const cached = @max(kept, if (h.resume_peek) |p| p.at(p.ctx, r.prompt, h.spans(r)) else 0);
        const verdict = source.wants(source.ptr, .{ .len = r.prompt.len, .cached = cached, .decoding = decoding });
        job.verdict = verdict;
        if (verdict != .remote) {
            job.remote_looked = !verdict.later(); // busy, slower or down: weighed again at the next look
            said(job, verdict, cached);
            continue;
        }
        const t = h.gpa.create(Ticket) catch continue;
        t.* = .{ .prompt = h.gpa.dupe(u32, r.prompt) catch {
            h.gpa.destroy(t);
            continue;
        }, .cached = cached, .host = h, .source = source };
        _ = source.running.fetchAdd(1, .acq_rel);
        const thread = std.Thread.spawn(.{}, fill, .{t}) catch {
            _ = source.running.fetchSub(1, .acq_rel);
            free(t);
            continue;
        };
        thread.detach();
        job.ticket = t;
        holding += 1;
    }
}

/// A prompt kept off the pair for now, said once per reason (a `later` verdict only: the rest are the usual case).
fn said(job: *Job, v: Verdict, cached: u64) void {
    if (!v.later() or job.remote_said == v) return;
    job.remote_said = v;
    std.log.info("remote prefill: a prompt of {d} tokens ({d} cached) is prefilled here: {s}", .{ job.request.prompt.len, cached, v.why() });
}

/// The source learns what a prompt pass cost here: the engine's own pass, or the last-token pass after a pull.
pub fn prefilled(h: *LaneHost, job: *const Job, ns: i96) void {
    const source = h.remote orelse return;
    const learn = source.measured orelse return;
    const rows = job.request.prompt.len - job.stream.cached;
    const took: u64 = @intCast(@max(0, ns));
    if (job.stream.reuse.pulled) {
        std.log.info("remote prefill: {d} tokens pulled: gate {d:.2} s, trigger {d:.2} s, transfer {d:.2} s (landing {d:.2} s), {d:.1} MiB; the tail pass {d:.2} s", .{
            job.stream.cached,                                     secs(job.phases.queue_ns), secs(job.phases.trigger_ns), secs(job.phases.pull_ns), secs(job.phases.land_ns),
            @as(f64, @floatFromInt(job.phases.bytes)) / (1 << 20), secs(took),
        });
        return learn(source.ptr, .{ .pass = .tail, .rows = rows, .ns = took });
    }
    if (!job.stream.reuse.remote) learn(source.ptr, .{ .pass = .local, .rows = rows, .ns = took });
}

fn secs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

/// A finished job's remote verdict and, if its prefix was pulled, its phases, for the reply's runtime block.
pub fn report(job: *const Job) @FieldType(api.Stats, "remote") {
    var out: @FieldType(api.Stats, "remote") = .{ .route = if (job.verdict) |v| @tagName(v) else "" };
    if (!job.stream.reuse.pulled) return out;
    const p = job.phases;
    out.queue_s = secs(p.queue_ns);
    out.trigger_s = secs(p.trigger_ns);
    out.pull_s = secs(p.pull_ns);
    out.land_s = secs(p.land_ns);
    out.bytes = p.bytes;
    if (job.prefilled) |done| out.tail_s = secs(@intCast(@max(0, done - job.began)));
    return out;
}

fn ahead() usize {
    return @import("sys").environment.read(usize, "TF_PREFILL_AHEAD", 4) catch 4;
}

fn fill(t: *Ticket) void {
    const source = t.source;
    const h = t.host;
    const left = t.state.load(.acquire) == Ticket.left; // its job ended before this thread began: nothing to ask
    const got = if (left) null else source.fill(source.ptr, t.prompt, t.cached, &t.gone) catch |err| blk: {
        if (err != error.Aborted) std.log.warn("remote prefill failed ({s}); the prompt is prefilled here", .{@errorName(err)});
        break :blk null;
    };
    t.filled = if (got) |f| if (f.at > t.cached and f.at < t.prompt.len) f else drop: { // a row stays: the first token
        source.drop(source.ptr, f.saved);
        break :drop null;
    } else null;
    if (t.state.cmpxchgStrong(Ticket.running, Ticket.landed, .acq_rel, .acquire) != null) free(t); // its job ended
    h.mutex.lockUncancelable(h.io); // the engine reads held() under this lock before it waits: no lost wake-up
    h.wake.signal(h.io);
    h.mutex.unlock(h.io);
    _ = source.running.fetchSub(1, .acq_rel);
}

fn free(t: *Ticket) void {
    if (t.filled) |f| t.source.drop(t.source.ptr, f.saved);
    t.host.gpa.free(t.prompt);
    t.host.gpa.destroy(t);
}

/// Whether queued `job` waits for its fill (admission passes it over meanwhile).
pub fn held(job: *const Job) bool {
    const t = job.ticket orelse return false;
    return t.state.load(.acquire) == Ticket.running;
}

/// Whether a landed remote prefix beats the cache's own (a peek): then the lookup, which reserves room, is skipped.
pub fn wins(h: *LaneHost, job: *const Job) bool {
    const t = job.ticket orelse return false;
    if (t.state.load(.acquire) != Ticket.landed) return false;
    const f = t.filled orelse return false;
    const store = h.cache orelse return true;
    const kept = if (store.find(job.cacheKey(), job.request.chunks, h.spans(job.request))) |e| e.at else 0;
    return f.at > kept;
}

/// The engine thread's last keepalive tick while a queued request awaited its remote fill (the awake clock, ns).
threadlocal var warm_tick: i96 = 0;

/// A keepalive tick while a queued request awaits its fill: macOS unwires an idle model after 1-2 s (engine thread).
fn keepWarm(h: *LaneHost) void {
    const target = h.keepalive_target orelse return;
    const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
    if (now - warm_tick < api.keepalive.served_interval_ns) return;
    for (h.queued.items) |job| if (held(job)) {
        warm_tick = now;
        return target.tick(target.ctx);
    };
}

/// Where a landed remote prefix that beats the cache's own starts the pass (0: none), for the lookup's marks.
pub fn resumeAt(h: *LaneHost, job: *const Job) u32 {
    if (!wins(h, job)) return 0;
    return job.ticket.?.filled.?.at;
}

/// The stream restores a landed remote prefix unless the cache holds at least as much; an entry it beats is released.
pub fn use(h: *LaneHost, job: *Job, reuse: *lanes.stream.Reuse) void {
    defer if (reuse.remote and !reuse.pulled) ownFirst(h, job, reuse); // a kept remote state stands, pull or none
    const t = job.ticket orelse return;
    if (t.state.load(.acquire) != Ticket.landed) return;
    job.ticket = null;
    job.remote = t.filled;
    t.filled = null;
    free(t);
    const f = job.remote orelse return;
    job.phases = f.phases;
    if (reuse.saved != null and reuse.at >= f.at) return drop(job);
    if (job.entry) |e| if (h.cache) |store| store.release(e);
    job.entry = null;
    // TF_PREFILL_KEEP: the cache's marks past the prefix keep remote states; else no marks and no hook, nothing kept
    const marks = if (h.keep_remote) past(reuse.marks, f.at) else &.{};
    reuse.* = .{ .saved = f.saved, .at = f.at, .remote = true, .pulled = true, .marks = marks, .snap_at = if (h.keep_remote) reuse.snap_at else 0, .hook = if (h.keep_remote) .{ .ptr = job, .at = Job.kept } else null };
}

/// A kept remote state slot 0's exact state reaches is let go; if a stream takes slot 0 first, the pass starts at 0.
fn ownFirst(h: *LaneHost, job: *Job, reuse: *lanes.stream.Reuse) void {
    const p = h.resume_peek orelse return;
    if (p.at(p.ctx, job.request.prompt, h.spans(job.request)) < reuse.at) return;
    if (job.entry) |e| if (h.cache) |store| store.release(e);
    job.entry = null;
    reuse.saved = null;
    reuse.at = 0;
    reuse.remote = false; // its marks and hook stay: what the pass keeps past slot 0's point is exact
}

/// The marks after `at`, ascending as the store plans them.
fn past(marks: []const u32, at: u32) []const u32 {
    var i: usize = 0;
    while (i < marks.len and marks[i] <= at) i += 1;
    return marks[i..];
}

/// The prefix is restored (or never will be): its storage goes, and a fill still running is left to its thread.
pub fn drop(job: *Job) void {
    job.remote_looked = false; // a requeued job is weighed again
    if (job.ticket) |t| {
        job.ticket = null;
        t.gone.store(true, .release);
        if (t.state.cmpxchgStrong(Ticket.running, Ticket.left, .acq_rel, .acquire) != null) free(t); // it had landed
    }
    const f = job.remote orelse return;
    job.remote = null;
    if (job.started and job.stream.reuse.saved == f.saved) job.stream.reuse.saved = null;
    if (job.host.remote) |source| source.drop(source.ptr, f.saved);
}

/// Prompt tokens the engine's own cache supplied: a prefix pulled just now is not one, a kept remote state is.
pub fn cachedOwn(job: *const Job) u32 {
    return if (job.stream.reuse.pulled) 0 else job.stream.cached;
}

/// The checkpoint's name both ends give a hand-off: its config and weight index, hashed; equal files, equal names.
pub fn checkpointId(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u8{ "config.json", "model.safetensors.index.json" }) |name| {
        const path = try std.fs.path.join(a, &.{ dir, name });
        const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch |err| {
            if (err == error.FileNotFound and name[0] == 'm') continue; // a checkpoint in one file has no index
            return err;
        };
        h.update(name);
        h.update(text);
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.fmt.allocPrint(a, "checkpoint-{x}", .{d[0..8]});
}
