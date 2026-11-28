//! Questions about kept states (/v1/imprint, /v1/prompt_cache/flush), answered on the engine thread.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = pc.rows;
const modes = @import("cache_modes.zig");
const radix = @import("prompt_radix.zig");

/// flush: every kept state dropped, in RAM and on disk, once no request is in flight (a waiting one holds admissions).
pub const Op = enum { where, load, forget, flush };

/// Which of a request's keys the host's cache lookups pass: its chunk starts, its decoded spans.
pub const Keys = struct { starts: bool = true, spans: bool = true };

/// A flush's claim: the engine thread takes a queued one, or its caller withdraws it first and it never runs.
pub const Claim = enum(u8) { queued, taken, withdrawn };

/// One question: a rendered conversation (its tokens, chunk starts and decoded spans), and the answer once `done`.
pub const Ask = struct {
    op: Op,
    retain: u32 = 0,
    prompt: []const u32 = &.{},
    starts: []const u32 = &.{},
    spans: []const [2]u32 = &.{},
    out: rows.read.Resume = .{},
    loaded: u32 = 0, // load: the tokens a disk read brought into RAM (0: none was needed, or it did not pay)
    load_seconds: f64 = 0,
    freed: u64 = 0, // forget, flush: bytes removed from disk
    dropped: u32 = 0, // flush: states dropped from RAM
    ram_freed: u64 = 0, // flush: the bytes those states held in the cache's budget
    disk: bool = false, // the engine keeps conversation states on disk
    claim: std.atomic.Value(Claim) = .init(.queued),
    done: std.atomic.Value(bool) = .init(false),

    /// The engine thread's answer from its kept states (`store` null: it keeps none), keyed as its lookups are.
    pub fn answer(a: *Ask, store: ?*pc.Store, keys: Keys) void {
        if (a.op == .flush and a.claim.cmpxchgStrong(.queued, .taken, .acq_rel, .acquire) != null) return; // withdrawn
        defer a.done.store(true, .release);
        const s = store orelse return;
        a.disk = s.disk != null;
        const starts: []const u32 = if (keys.starts) a.starts else &.{};
        const spans: []const [2]u32 = if (keys.spans) a.spans else &.{};
        switch (a.op) {
            .where => {},
            .load => {
                const have = s.find(a.prompt, starts, spans);
                const t0 = rows.write.nowNs();
                const got = rows.read.recall(s, a.prompt, @intCast(a.prompt.len), starts, spans, have);
                if (got != null and got != have) {
                    a.loaded = got.?.at;
                    a.load_seconds = @as(f64, @floatFromInt(rows.write.nowNs() - t0)) / 1e9;
                }
            },
            .forget => if (s.disk) |r| {
                a.freed = r.forgetAll();
            },
            .flush => return flush(a, s),
        }
        a.out = rows.read.resumable(s, a.prompt, starts, spans);
    }

    /// The answer from TF_KV_SHARE's store over pool pages: no disk; flush keeps its newest `retain` states.
    pub fn answerShared(a: *Ask, s: *radix.Store) void {
        if (a.op == .flush and a.claim.cmpxchgStrong(.queued, .taken, .acq_rel, .acquire) != null) return; // withdrawn
        defer a.done.store(true, .release);
        if (a.op == .flush) {
            const c = s.flush(a.retain);
            a.dropped = @intCast(c.states);
            a.ram_freed = c.bytes;
            return;
        }
        const ram: u32 = if (s.find(.of(a.prompt))) |f| f.at() else 0; // keyed as a TF_KV_SHARE pass begins
        a.out = .{ .tokens = @intCast(a.prompt.len), .ram = ram, .from = if (ram > 0) .ram else .none };
    }

    /// A flush's caller stops waiting: true when the engine had not taken it, so nothing is dropped and nothing holds.
    pub fn withdraw(a: *Ask) bool {
        return a.claim.cmpxchgStrong(.queued, .withdrawn, .acq_rel, .acquire) == null;
    }

    /// The answer within `ms` milliseconds (false: none came).
    pub fn wait(a: *const Ask, io: std.Io, ms: u32) bool {
        var waited: u32 = 0;
        while (!a.done.load(.acquire)) : (waited += 2) {
            if (waited >= ms) return false;
            std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
        }
        return true;
    }
};

fn newer(a: *const pc.Entry, b: *const pc.Entry) bool {
    return a.used > b.used or (a.used == b.used and a.serial > b.serial);
}

/// Whether a disk state or staged write is a kept entry's own: the same position, tokens and decoded spans.
fn ownOf(entries: []const *pc.Entry, at: u32, tokens: []const u32, spans: []const modes.Span) bool {
    for (entries) |e| {
        if (e.expired or e.salt != null or e.at != at or !std.mem.eql(u32, e.tokens, tokens)) continue;
        if (modes.equal(e.decode_spans, spans, at)) return true;
    }
    return false;
}

fn forgetExcept(r: *rows.Rows, entries: []const *pc.Entry) u64 {
    const before = r.held();
    if (r.worker) |w| {
        var jobs: [rows.write.io.max_jobs]?*rows.write.io.Job = undefined;
        w.cancellations(&jobs);
        for (jobs) |job| if (job) |j| if (!ownOf(entries, j.at, j.tokens, j.spans)) {
            rows.write.fence(r); // a dropped entry's staged write must not publish; ones not yet staged went with it
            break;
        };
    }
    var i = r.states.items.len;
    while (i > 0) {
        i -= 1;
        const s = &r.states.items[i];
        if (!ownOf(entries, s.at, s.tokens, s.spans)) r.remove(i);
    }
    return before - r.held();
}

/// Every state except the newest `retain` complete states dropped, with their Rows disk states; the plans reset.
fn flush(a: *Ask, s: *pc.Store) void {
    const n = s.entries.items.len;
    const bytes = s.held;
    var cutoff: ?*pc.Entry = null;
    for (0..a.retain) |_| {
        var best: ?*pc.Entry = null;
        for (s.entries.items) |e| {
            if (e.expired or (cutoff != null and !newer(cutoff.?, e))) continue;
            if (best == null or newer(e, best.?)) best = e;
        }
        cutoff = best orelse break;
    }
    for (s.entries.items) |e| {
        if (a.retain == 0 or (cutoff != null and newer(cutoff.?, e))) e.expired = true; // expired first: nothing is staged on its way out
    }
    s.collectExpired(); // a borrowed one (none while no request is in flight) goes when its lookup ends
    a.dropped = @intCast(n - s.entries.items.len);
    a.ram_freed = bytes - s.held;
    s.shared_keys.clearRetainingCapacity();
    s.repeat_log.keys.clearRetainingCapacity();
    s.active_rewind_at = 0;
    s.active_rewind_key = 0;
    if (s.disk) |r| a.freed = if (a.retain == 0) r.forgetAll() else forgetExcept(r, s.entries.items); // the store's own folder and names
}

/// A host's side of disk states: the asks waiting for its engine thread, and when the engine last went quiet.
pub const Disk = struct {
    asks: std.ArrayList(*Ask) = .empty,
    flushes: std.ArrayList(*Ask) = .empty, // while one waits unwithdrawn, the host admits nothing new
    seen: u64 = 0, // the host's admissions when the writer last looked: a new one restarts the quiet
    since: ?i96 = null,
    shared: ?*radix.Store = null, // TF_KV_SHARE's store (no host cache then): asks, flushes and moved weights reach it

    /// The quiet after the last admission before the first write: a client's next turn often comes sooner.
    pub const quiet_ns: i96 = 150 * std.time.ns_per_ms;

    /// Queue `a` (the host's lock held).
    pub fn push(d: *Disk, gpa: std.mem.Allocator, a: *Ask) bool {
        (if (a.op == .flush) &d.flushes else &d.asks).append(gpa, a) catch return false;
        return true;
    }

    /// Whether a flush waits for the requests in flight: admissions hold until it is answered (the host's lock held).
    pub fn holding(d: *const Disk) bool {
        for (d.flushes.items) |a| if (a.claim.load(.acquire) == .queued) return true;
        return false;
    }

    /// Whether any ask waits, a flush or another (the host's lock held).
    pub fn waiting(d: *const Disk) bool {
        return d.asks.items.len > 0 or d.flushes.items.len > 0;
    }

    /// The waiting flushes, taken whole once no request is in flight (the host's lock held); the caller frees the list.
    pub fn takeFlushes(d: *Disk) std.ArrayList(*Ask) {
        const list = d.flushes;
        d.flushes = .empty;
        return list;
    }

    /// The queued asks, taken (the host's lock held); the caller frees the list.
    pub fn take(d: *Disk, gpa: std.mem.Allocator) []*Ask {
        if (d.asks.items.len == 0) return &.{};
        const list = gpa.dupe(*Ask, d.asks.items) catch return &.{};
        d.asks.clearRetainingCapacity();
        return list;
    }

    /// Whether the writer may run now: no admission since `quiet_ns` ago (`admissions`: the host's count so far).
    pub fn quiet(d: *Disk, admissions: u64, now: i96) bool {
        if (d.seen != admissions or d.since == null) {
            d.seen = admissions;
            d.since = now;
            return false;
        }
        return now - d.since.? >= quiet_ns;
    }

    /// Asks left when the engine stops, answered as having no states; the list freed.
    pub fn close(d: *Disk, gpa: std.mem.Allocator) void {
        for (d.asks.items) |a| a.answer(null, .{});
        d.asks.deinit(gpa);
        for (d.flushes.items) |a| a.answer(null, .{});
        d.flushes.deinit(gpa);
    }
};
