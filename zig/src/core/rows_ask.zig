//! Questions about kept states (/v1/imprint, /v1/prompt_cache/flush), answered on the engine thread.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = pc.rows;
const Image = @import("lanes").stream.Image;

/// flush: every kept state dropped once no request is in flight (admissions held meanwhile); drop: one conversation's.
pub const Op = enum {
    where,
    load,
    forget,
    flush,
    drop,

    /// Whether the ask waits for the requests in flight, holding admissions, and its caller may withdraw it.
    pub fn holds(op: Op) bool {
        return op == .flush or op == .drop;
    }
};

/// How the host's cache lookups key a request: its chunk starts, its decoded spans, its image frame (null: none).
pub const Keys = struct { starts: bool = true, spans: bool = true, frame: ?*const fn ([]const Image) u32 = null };

/// A flush's claim: the engine thread takes a queued one, or its caller withdraws it first and it never runs.
pub const Claim = enum(u8) { queued, taken, withdrawn };

/// One question: a rendered conversation (its tokens, chunk starts and decoded spans), and the answer once `done`.
pub const Ask = struct {
    op: Op,
    prompt: []const u32 = &.{},
    starts: []const u32 = &.{},
    spans: []const [2]u32 = &.{},
    images: []const Image = &.{}, // drop: the request's images without pixels, for a framed family's key
    out: rows.read.Resume = .{},
    loaded: u32 = 0, // load: the tokens a disk read brought into RAM (0: none was needed, or it did not pay)
    load_seconds: f64 = 0,
    freed: u64 = 0, // forget, flush, drop: bytes removed from disk
    dropped: u32 = 0, // flush, drop: states dropped from RAM
    ram_freed: u64 = 0, // flush, drop: the bytes those states held in the cache's budget
    disk_dropped: u32 = 0, // drop: states removed from disk
    shared: u32 = 0, // drop: the conversation's states left in RAM or on disk, which other conversations build on
    disk: bool = false, // the engine keeps conversation states on disk
    claim: std.atomic.Value(Claim) = .init(.queued),
    done: std.atomic.Value(bool) = .init(false),

    /// The engine thread's answer from its kept states (`store` null: it keeps none), keyed as its lookups are.
    pub fn answer(a: *Ask, store: ?*pc.Store, keys: Keys) void {
        if (a.op.holds() and a.claim.cmpxchgStrong(.queued, .taken, .acq_rel, .acquire) != null) return; // withdrawn
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
            .drop => return @import("rows_ask_drop.zig").drop(a, s, keys),
        }
        a.out = rows.read.resumable(s, a.prompt, starts, spans);
    }

    /// A flush's or drop's caller stops waiting: true when the engine had not taken it, so nothing goes or holds.
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

/// Every kept state dropped and the store's plans reset, as on a fresh store; then its states on disk deleted.
fn flush(a: *Ask, s: *pc.Store) void {
    const n = s.entries.items.len;
    const bytes = s.held;
    for (s.entries.items) |e| e.expired = true; // expired first: nothing is staged for disk on its way out
    s.collectExpired(); // a borrowed one (none while no request is in flight) goes when its lookup ends
    a.dropped = @intCast(n - s.entries.items.len);
    a.ram_freed = bytes - s.held;
    s.shared_keys.clearRetainingCapacity();
    s.repeat_log.keys.clearRetainingCapacity();
    s.active_rewind_at = 0;
    s.active_rewind_key = 0;
    if (s.disk) |r| a.freed = r.forgetAll(); // the store's own folder and names: DELETE /v1/imprint's call
}

/// A host's side of disk states: the asks waiting for its engine thread, and when the engine last went quiet.
pub const Disk = struct {
    asks: std.ArrayList(*Ask) = .empty,
    flushes: std.ArrayList(*Ask) = .empty, // flushes and drops: while one waits unwithdrawn, nothing new is admitted
    seen: u64 = 0, // the host's admissions when the writer last looked: a new one restarts the quiet
    since: ?i96 = null,

    /// The quiet after the last admission before the first write: a client's next turn often comes sooner.
    pub const quiet_ns: i96 = 150 * std.time.ns_per_ms;

    /// Queue `a` (the host's lock held).
    pub fn push(d: *Disk, gpa: std.mem.Allocator, a: *Ask) bool {
        (if (a.op.holds()) &d.flushes else &d.asks).append(gpa, a) catch return false;
        return true;
    }

    /// Whether a flush or drop waits for the requests in flight: admissions hold until it is answered (lock held).
    pub fn holding(d: *const Disk) bool {
        for (d.flushes.items) |a| if (a.claim.load(.acquire) == .queued) return true;
        return false;
    }

    /// Whether any ask waits, a flush or another (the host's lock held).
    pub fn waiting(d: *const Disk) bool {
        return d.asks.items.len > 0 or d.flushes.items.len > 0;
    }

    /// The waiting flushes and drops, taken whole once no request is in flight (lock held); the caller frees the list.
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
