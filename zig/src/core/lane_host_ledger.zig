//! The lane host's KV ledger: a job promises its caches' bytes as it leaves the queue and returns them as it ends.
const std = @import("std");
const kv = @import("kv_ledger.zig");
const classes = kv.classes;
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

/// The ledger an engine's Info asks for (null: none); a family admits through its own pool or this, never both.
pub fn of(info: api.Info) ?kv.Ledger {
    std.debug.assert(info.ledger == null or info.admit == null);
    return if (info.ledger) |l| .{ .total = l.total, .standing = l.standing } else null;
}

/// Whether jobs promise bytes as they leave the queue (Info.ledger's price); a paged family's pool promises its own.
pub fn priced(h: *const LaneHost) bool {
    return h.info_.ledger != null;
}

/// `pool`'s pages count in this host's ledger, which alone admits them; a pool another host counts is refused.
pub fn count(h: *LaneHost, pool: anytype) error{ PoolHasHost, NoClassHook }!void {
    const P = @TypeOf(pool.*);
    if (pool.host != null) return error.PoolHasHost;
    std.debug.assert(h.kv == null and pool.ledger == null); // one pool a host, and never beside a priced ledger
    const view = struct {
        fn read(ctx: *const anyopaque) kv.Pages {
            const p: *const P = @ptrCast(@alignCast(ctx));
            return .{
                .total = p.total(),
                .available = p.available(),
                .reclaimable = p.reclaimable(),
                .unit = p.unit(),
                .classes = if (@hasField(P, "classes")) p.classes else 1,
            };
        }
        fn class(ctx: *const anyopaque, c: u32) kv.Class {
            const p: *const P = @ptrCast(@alignCast(ctx));
            return .{ .cap = p.classCap(), .available = p.availableIn(c) };
        }
        fn detach(ctx: *anyopaque) void {
            const p: *P = @ptrCast(@alignCast(ctx));
            p.ledger = null;
            p.host = null;
        }
    };
    const by: ?*const fn (*const anyopaque, u32) kv.Class = if (@hasField(P, "classes") and pool.classes > 1) view.class else null;
    h.kv = try .ofPages(.{ .ctx = pool, .read = view.read, .detach = view.detach, .class = by });
    pool.ledger = .{ .ctx = &h.kv.?, .want = kv.Ledger.wantOf };
    pool.host = h;
}

/// The pool stops counting in this host's ledger and admits on its own pages again (the host is stopping).
pub fn uncount(h: *LaneHost) void {
    if (h.kv) |*l| l.detach();
}

/// What `job`'s caches cost at most (the family's price for its prompt and reply).
fn cost(h: *const LaneHost, job: *const Job) u64 {
    const l = h.info_.ledger.?;
    return l.cost(l.ctx, @intCast(job.request.prompt.len), job.request.max_tokens);
}

/// Whether `job` fits beside what admitted jobs hold now; the caller holds the lock.
pub fn fits(h: *const LaneHost, job: *const Job) bool {
    return h.kv.?.fits(cost(h, job));
}

/// Promise `job`'s bytes as it leaves the queue (true without a ledger); false: they do not fit now.
pub fn promise(h: *LaneHost, job: *Job) bool {
    if (priced(h)) h.kv.?.promise(&job.lease, cost(h, job)) catch return false;
    return true;
}

pub const memory_unfit = "the memory prefix this request would run does not fit the memory left for prompts";

/// `job`'s promise raised to a prompt of `prompt_len` rows, as its memory prefix makes it; false: it does not fit.
pub fn grow(h: *LaneHost, job: *Job, prompt_len: usize) bool {
    if (!priced(h)) return true;
    const l = h.info_.ledger.?;
    const bytes = l.cost(l.ctx, std.math.cast(u32, prompt_len) orelse return false, job.request.max_tokens);
    h.lock();
    defer h.unlock();
    h.kv.?.grow(&job.lease, bytes) catch return false;
    return true;
}

/// `job`'s bytes go back (nothing without a ledger or a lease); the caller holds the lock.
pub fn release(h: *LaneHost, job: *Job) void {
    if (h.kv) |*l| l.finish(&job.lease);
}

/// Whether any admitted job holds bytes now.
pub fn outstanding(h: *const LaneHost) bool {
    return !h.kv.?.idle();
}

/// Why `job` would wait forever: past the whole budget alone, or past a whole owner class even while others run.
pub fn never(h: *const LaneHost, job: *const Job) ?[]const u8 {
    if (priced(h) and !outstanding(h) and !fits(h, job)) return too_big;
    return if (pastClass(h, job)) too_big_class else null;
}

/// Whether `job` needs more of a class than the class holds at all (false without Admit.need), a lane free or not.
pub fn pastClass(h: *const LaneHost, job: *const Job) bool {
    const d = need(h, job) orelse return false;
    return classes.neverNeed(&h.kv.?, d) != null;
}

/// The pages `job` needs a class (Admit.need), when the family names them and its pool counts here (else null).
pub fn need(h: *const LaneHost, job: *const Job) ?classes.Need {
    const a = h.info_.admit orelse return null;
    const f = a.need orelse return null;
    if (h.kv == null or h.kv.?.pool == null) return null;
    return f(a.ctx, @intCast(job.request.prompt.len), job.request.max_tokens);
}

/// Whether `job`'s pages fit now in each class it takes from (true without Admit.need); the caller holds the lock.
pub fn fitsClasses(h: *const LaneHost, job: *const Job) bool {
    const d = need(h, job) orelse return true;
    return classes.fitsNeed(&h.kv.?, d);
}

/// At stop every job has ended: bytes still promised mean a lease that never came back.
pub fn checkIdle(h: *const LaneHost) void {
    const l = h.kv orelse return;
    if (l.promised > 0) std.log.warn("kv ledger: {d} of {d} bytes still promised at stop", .{ l.promised, l.total });
    if (l.bound > 0) std.log.warn("kv ledger: {d} standing streams still held at stop", .{l.bound}); // never the lease itself
}

pub const too_big_class = "this request's pages exceed one whole KV owner class (TF_GLM_KV_SPLIT): lower max_tokens or --context";
pub const too_big = "this request's caches exceed the whole KV budget with nothing else running: lower max_tokens or --context";
