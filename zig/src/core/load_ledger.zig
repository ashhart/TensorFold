//! The server's load ledger: scheduler load a host publishes as it happens, read whole by /metrics and /v1/load.
const std = @import("std");
const api = @import("engine_api.zig");

/// The model routes `http_requests_active` counts apart; a fixed set, so the label never grows.
pub const RouteGroup = enum { chat_completions, completions, responses, messages, embeddings, speech, other };
const groups = @backingInt(RouteGroup.other) + 1; // `other` stays last

/// How an engine holds its KV state: a window a lane, one page pool, or both.
pub const Mode = enum { window, paged, mixed };

/// The `requests` group: counts, queued prompt work and the host's lane give-ups.
pub const Requests = struct {
    running: u32 = 0,
    waiting: u32 = 0,
    waiting_prompt_tokens: u64 = 0,
    /// Prompt tokens left in the passes filling now: 0 when none fills, null when one fills without progress.
    filling_left: ?u64 = 0,
    fill_unknown: bool = false,
    started: u64 = 0,
    requeued: u64 = 0,
    yields: u64 = 0,
};

/// The `work` group: prompt and output tokens as they are computed, and the engine time of their prompt calls.
pub const Work = struct { prefill_compute: u64 = 0, prefill_cache: u64 = 0, decode: u64 = 0, prefill_busy_ns: u64 = 0 };

/// A page pool's census in tokens; `run` is null when the family needs no contiguous run.
pub const Pool = struct { capacity: u64 = 0, available: u64 = 0, reclaimable: u64 = 0, promised: u64 = 0, run: ?bool = null };

/// The `capacity` group's host part; a paged mode also needs a published pool.
pub const Capacity = struct { mode: Mode, lanes_free: u32 };

/// A host's handle on the ledger, valid until the next attach or detach.
pub const Producer = struct {
    ledger: *Ledger,
    epoch: u64,

    /// The host's groups (null: a group it does not fill); dropped when the handle is stale.
    pub fn publish(p: Producer, requests: ?Requests, work: ?Work, capacity: ?Capacity) void {
        const l = p.ledger;
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        if (p.epoch != l.epoch) return;
        l.cur.requests = requests;
        l.cur.work = work;
        l.cur.capacity = capacity;
    }

    /// The pool's census, from the thread that changed the pool; dropped when the handle is stale.
    pub fn publishPool(p: Producer, pool: Pool) void {
        const l = p.ledger;
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        if (p.epoch != l.epoch) return;
        l.cur.pool = pool;
    }
};

/// One read of the whole ledger: server values always, each engine group when an attached host fills it.
pub const Snapshot = struct {
    instance: [32]u8,
    start_s: i64,
    parked: bool,
    lanes: u32,
    http: [groups]u32,
    requests: ?Requests = null,
    work: ?Work = null,
    capacity: ?Capacity = null,
    pool: ?Pool = null,

    pub fn active(s: Snapshot, g: RouteGroup) u32 {
        return s.http[@backingInt(g)];
    }
};

const Groups = struct { requests: ?Requests = null, work: ?Work = null, capacity: ?Capacity = null, pool: ?Pool = null };

pub const Ledger = struct {
    io: std.Io,
    /// Held across a whole attach or detach, host callback included; hosts never take it.
    life: std.Io.Mutex = .init,
    mutex: std.Io.Mutex = .init,
    instance: [32]u8,
    start_s: i64,
    parked: bool = false,
    lanes: u32 = 0,
    http: [groups]u32 = @splat(0),
    /// Raised by every attach and detach: a handle from before it writes nothing.
    epoch: u64 = 0,
    attached: ?api.Engine = null,
    cur: Groups = .{},
    /// Counters of hosts detached before, added to every read.
    base: Groups = .{ .requests = .{}, .work = .{} },

    /// A ledger with a fresh instance id and the process's start time.
    pub fn init(io: std.Io) Ledger {
        var bytes: [16]u8 = undefined;
        io.random(&bytes); // an identity, not a secret: request seeds keep the secure source to themselves
        return .{ .io = io, .instance = std.fmt.bytesToHex(bytes, .lower), .start_s = std.Io.Clock.real.now(io).toSeconds() };
    }

    /// `engine` becomes the producer: a new epoch, no engine values until it publishes, its handle through `load`.
    pub fn attach(l: *Ledger, engine: api.Engine) void {
        l.life.lockUncancelable(l.io);
        defer l.life.unlock(l.io);
        l.detachHeld(false); // one producer at a time: an engine still attached stops first
        const lanes = engine.info().lanes;
        l.mutex.lockUncancelable(l.io);
        l.epoch +%= 1;
        l.cur = .{};
        l.parked = false;
        l.lanes = lanes;
        l.attached = engine;
        const p: Producer = .{ .ledger = l, .epoch = l.epoch };
        l.mutex.unlock(l.io);
        if (engine.vtable.load) |f| f(engine.ctx, p);
    }

    /// The attached engine stops producing before it closes: its counters fold into `base`, its gauges go absent.
    pub fn detach(l: *Ledger, parked: bool) void {
        l.life.lockUncancelable(l.io);
        defer l.life.unlock(l.io);
        l.detachHeld(parked);
    }

    /// `detach` under `life`, which keeps a whole attach or detach, callback included, apart from another.
    fn detachHeld(l: *Ledger, parked: bool) void {
        l.mutex.lockUncancelable(l.io);
        const engine = l.attached orelse return l.mutex.unlock(l.io);
        l.attached = null;
        l.mutex.unlock(l.io);
        if (engine.vtable.load) |f| f(engine.ctx, null); // the host drops its handle before this returns
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        l.epoch +%= 1;
        fold(&l.base, l.cur);
        l.cur = .{};
        l.parked = parked;
    }

    /// A model request of `g` starts; `leave` ends it.
    pub fn enter(l: *Ledger, g: RouteGroup) void {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        l.http[@backingInt(g)] += 1;
    }

    pub fn leave(l: *Ledger, g: RouteGroup) void {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        l.http[@backingInt(g)] -|= 1;
    }

    /// The ledger at one instant; counters carry every earlier host's.
    pub fn read(l: *Ledger) Snapshot {
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        var s: Snapshot = .{ .instance = l.instance, .start_s = l.start_s, .parked = l.parked, .lanes = l.lanes, .http = l.http };
        if (l.cur.requests) |r| s.requests = addRequests(r, l.base.requests.?);
        if (l.cur.work) |w| s.work = addWork(w, l.base.work.?);
        if (l.cur.capacity) |c| if (c.mode == .window or l.cur.pool != null) {
            s.capacity = c;
            s.pool = l.cur.pool;
        };
        return s;
    }
};

/// A detached host's counters into `base`; its gauges are dropped.
fn fold(base: *Groups, cur: Groups) void {
    if (cur.requests) |r| {
        const b = &base.requests.?;
        b.started += r.started;
        b.requeued += r.requeued;
        b.yields += r.yields;
    }
    if (cur.work) |w| base.work = addWork(w, base.work.?);
}

/// `r`'s gauges with `base`'s counters added to its own.
fn addRequests(r: Requests, base: Requests) Requests {
    var out = r;
    out.started += base.started;
    out.requeued += base.requeued;
    out.yields += base.yields;
    return out;
}

fn addWork(w: Work, base: Work) Work {
    return .{ .prefill_compute = w.prefill_compute + base.prefill_compute, .prefill_cache = w.prefill_cache + base.prefill_cache, .decode = w.decode + base.decode, .prefill_busy_ns = w.prefill_busy_ns + base.prefill_busy_ns };
}

test {
    _ = @import("load_ledger_test.zig");
}
