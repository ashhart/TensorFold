//! The lane host's KV ledger: a job promises its caches' bytes as it leaves the queue and returns them as it ends.
const std = @import("std");
const kv = @import("kv_ledger.zig");
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
pub fn count(h: *LaneHost, pool: anytype) error{PoolHasHost}!void {
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
            };
        }
        fn detach(ctx: *anyopaque) void {
            const p: *P = @ptrCast(@alignCast(ctx));
            p.ledger = null;
            p.host = null;
        }
    };
    h.kv = .ofPages(.{ .ctx = pool, .read = view.read, .detach = view.detach });
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

/// `job`'s bytes go back (nothing without a ledger or a lease); the caller holds the lock.
pub fn release(h: *LaneHost, job: *Job) void {
    if (h.kv) |*l| l.finish(&job.lease);
}

/// Whether any admitted job holds bytes now.
pub fn outstanding(h: *const LaneHost) bool {
    return !h.kv.?.idle();
}

/// Whether `job` alone exceeds the whole budget: with nothing else promised it would wait forever.
pub fn never(h: *const LaneHost, job: *const Job) bool {
    return priced(h) and !outstanding(h) and !fits(h, job);
}

/// At stop every job has ended: bytes still promised mean a lease that never came back.
pub fn checkIdle(h: *const LaneHost) void {
    const l = h.kv orelse return;
    if (l.promised > 0) std.log.warn("kv ledger: {d} of {d} bytes still promised at stop", .{ l.promised, l.total });
    if (l.bound > 0) std.log.warn("kv ledger: {d} standing streams still held at stop", .{l.bound}); // never the lease itself
}

pub const too_big = "this request's caches exceed the whole KV budget with nothing else running: lower max_tokens or --context";
