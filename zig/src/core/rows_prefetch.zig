//! Disk states read ahead on an I/O thread while their request waits: files checked into host memory, scattered later.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const image = @import("rows_image.zig");
const chain = @import("rows_chain.zig");
const file = @import("rows_file.zig");
const rows_store = @import("rows_store.zig");
const write = @import("rows_write.zig");
const select = @import("rows_select.zig");
const Rows = rows_store.Rows;
const Name = chain.Name;
const Allocator = std.mem.Allocator;

/// The reads held at once (TENSORFOLD_RESTORE_PREFETCH, 0: off), their bytes, and a request's wait for its read.
pub const Limits = struct {
    reads: usize = 2,
    bytes: u64 = 16 << 30,
    wait_ns: i128 = 30 * std.time.ns_per_s,

    pub fn fromEnv() Limits {
        return fromText(env("TENSORFOLD_RESTORE_PREFETCH"), env("TENSORFOLD_RESTORE_PREFETCH_GIB"), env("TENSORFOLD_RESTORE_PREFETCH_MS"));
    }

    /// Each bound from its text: whole numbers (reads, GiB, milliseconds); anything else keeps its default.
    pub fn fromText(reads: []const u8, gib: []const u8, ms: []const u8) Limits {
        var l: Limits = .{};
        if (whole(reads)) |n| l.reads = @intCast(@min(n, 1 << 16));
        if (whole(gib)) |n| l.bytes = @as(u64, @min(n, 1 << 24)) << 30;
        if (whole(ms)) |n| l.wait_ns = @as(i128, @min(n, 1 << 40)) * std.time.ns_per_ms;
        return l;
    }
};

fn env(name: [*:0]const u8) []const u8 {
    return std.mem.span(platform.c.getenv(name) orelse return "");
}

fn whole(text: []const u8) ?u64 {
    const t = std.mem.trim(u8, text, " \t");
    if (t.len == 0 or t[0] < '0' or t[0] > '9') return null;
    return std.fmt.parseInt(u64, t, 10) catch null;
}

/// One file a state's read takes: name, kind, position, payload, its image ranges and logical offset.
pub const Piece = struct { name: Name, kind: file.Kind, at: u32, len: u64, first: usize, end: usize, off: u64 };

/// Every file a disk state's read takes, in the order `read` scatters them, with their ranges of its image.
pub const Plan = struct {
    pieces: []Piece = &.{},
    ranges: []image.Range = &.{},
    bytes: u64 = 0, // logical layout: each payload at an aligned offset

    /// The files of state `s`: its blocks, then its own file.
    pub fn of(r: *const Rows, a: Allocator, s: *const rows_store.State) !Plan {
        const parts = try r.parts(a, s.at);
        defer a.free(parts);
        var wants: std.ArrayList(write.Want) = .empty;
        defer wants.deinit(a);
        try r.blockNames(a, parts, s.tokens, s.spans, s.starts, s.at, &wants);
        var pieces: std.ArrayList(Piece) = .empty;
        errdefer pieces.deinit(a);
        var ranges: std.ArrayList(image.Range) = .empty;
        errdefer ranges.deinit(a);
        var off: u64 = 0;
        for (wants.items) |w| {
            const first = ranges.items.len;
            try image.blockRanges(parts, r.keying.block, w.window, w.k, &ranges, a);
            try add(&pieces, a, .{ .name = w.name, .kind = .block, .at = 0, .len = image.total(ranges.items[first..]), .first = first, .end = ranges.items.len, .off = off }, &off);
        }
        const first = ranges.items.len;
        try image.ownRanges(parts, s.at, r.keying.block, &ranges, a);
        try add(&pieces, a, .{ .name = s.name, .kind = .state, .at = s.at, .len = image.total(ranges.items[first..]), .first = first, .end = ranges.items.len, .off = off }, &off);
        const owned = try pieces.toOwnedSlice(a);
        errdefer a.free(owned);
        return .{ .pieces = owned, .ranges = try ranges.toOwnedSlice(a), .bytes = off };
    }

    fn add(pieces: *std.ArrayList(Piece), a: Allocator, p: Piece, off: *u64) !void {
        try pieces.append(a, p);
        off.* += std.mem.alignForward(u64, p.len, file.alignment);
    }

    pub fn deinit(p: *Plan, a: Allocator) void {
        a.free(p.pieces);
        a.free(p.ranges);
        p.* = .{};
    }

    pub fn rangesOf(p: *const Plan, x: Piece) []const image.Range {
        return p.ranges[x.first..x.end];
    }

    /// Whether `o` reads the same files, each of the same payload.
    pub fn same(p: *const Plan, o: *const Plan) bool {
        if (p.pieces.len != o.pieces.len) return false;
        for (p.pieces, o.pieces) |x, y| if (!std.mem.eql(u8, &x.name, &y.name) or x.kind != y.kind or x.at != y.at or x.len != y.len) return false;
        return true;
    }
};

/// A file's extension in the store by its kind.
pub fn ext(kind: file.Kind) []const u8 {
    return if (kind == .block) ".blk" else ".st";
}

pub const Phase = enum { queued, reading, ready, failed };

/// A filled file fragment: its plan piece, logical payload offset and bytes valid until `recycle`.
pub const Chunk = struct { piece: usize, off: u64, bytes: []const u8, slot: u8 };

pub const chunk_bytes: usize = 4 << 20;
const buffers_bytes: u64 = 2 * chunk_bytes;
const SlotState = enum { empty, filling, full, using };
const Slot = struct { buf: file.Stage = &.{}, piece: usize = 0, off: u64 = 0, len: usize = 0, state: SlotState = .empty };

/// One state read ahead: its files, two bounded stream slots, and what the read found.
pub const Read = struct {
    name: Name,
    plan: Plan,
    slots: [2]Slot = .{ .{}, .{} },
    produce: u1 = 0,
    consume: u1 = 0,
    phase: Phase = .queued,
    cancel: std.atomic.Value(bool) = .init(false),
    detached: bool = false, // the engine owns and finishes it; never freed by the worker, sweep or stop
    producer_active: bool = false,
    announced: bool = false, // first available fragment woke the request
    since: i128,
    disk: u64 = 0, // its files' bytes on disk
    ns: i128 = 0,
    err: ?anyerror = null,
    failed_piece: ?usize = null, // the file its producer's check rejected, for the consumer to forget
    used: bool = false,
    late: bool = false, // a request stopped waiting for it at the bound
};

pub const Counts = struct { made: u64 = 0, freed: u64 = 0, used: u64 = 0, failed: u64 = 0, cancelled: u64 = 0, timeouts: u64 = 0 };

/// Called by the I/O thread, holding no lock of its own, when a fragment is available or a read settles.
pub const Notify = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque) void };

/// A store's reads ahead and the one I/O thread that runs them, oldest first; everything here under `mutex`.
pub const Prefetch = struct {
    gpa: Allocator,
    io: std.Io,
    owned_io: std.Io.Threaded = .init_single_threaded,
    limits: Limits,
    dir: [:0]u8,
    block: u32,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    reads: std.ArrayList(*Read) = .empty,
    held_reads: usize = 0,
    held_bytes: u64 = 0,
    thread: ?std.Thread = null,
    stopping: bool = false,
    refs: std.atomic.Value(u32) = .init(1),
    exited: std.atomic.Value(bool) = .init(true),
    notifying: std.atomic.Value(bool) = .init(false),
    notify: ?Notify = null,
    reader: *const fn (f: *Prefetch, rd: *Read) anyerror!void = readFiles, // a test's slow or failing reader
    counts: Counts = .{},

    /// Whether reads may start (reads above 0).
    pub fn on(f: *const Prefetch) bool {
        return f.limits.reads > 0;
    }

    /// Counters, held reads and their bytes, read under the mutex for the status snapshot.
    pub fn snapshot(f: *Prefetch) struct { counts: Counts, held_reads: usize, held_bytes: u64 } {
        f.lock();
        defer f.unlock();
        return .{ .counts = f.counts, .held_reads = f.held_reads, .held_bytes = f.held_bytes };
    }

    fn lock(f: *Prefetch) void {
        f.mutex.lockUncancelable(f.io);
    }

    fn unlock(f: *Prefetch) void {
        f.mutex.unlock(f.io);
    }

    fn find(f: *const Prefetch, name: Name) ?usize {
        for (f.reads.items, 0..) |rd, i| if (std.mem.eql(u8, &rd.name, &name)) return i;
        return null;
    }

    /// Disk state `i` read ahead (engine thread), or joined if one reads it already; null past the bounds or off.
    pub fn start(f: *Prefetch, r: *const Rows, i: usize) ?Name {
        if (!f.on()) return null;
        const s = &r.states.items[i];
        f.lock();
        defer f.unlock();
        if (f.stopping) return null;
        if (f.find(s.name) != null) return s.name;
        if (f.held_reads >= f.limits.reads) return null;
        if (buffers_bytes > f.limits.bytes -| f.held_bytes) return null;
        var plan = Plan.of(r, f.gpa, s) catch return null;
        const rd = f.gpa.create(Read) catch {
            plan.deinit(f.gpa);
            return null;
        };
        const first = f.gpa.alignedAlloc(u8, .fromByteUnits(file.alignment), chunk_bytes) catch {
            plan.deinit(f.gpa);
            f.gpa.destroy(rd);
            return null;
        };
        const second = f.gpa.alignedAlloc(u8, .fromByteUnits(file.alignment), chunk_bytes) catch {
            f.gpa.free(first);
            plan.deinit(f.gpa);
            f.gpa.destroy(rd);
            return null;
        };
        rd.* = .{ .name = s.name, .plan = plan, .since = write.nowNs(), .slots = .{ .{ .buf = first }, .{ .buf = second } } };
        if (!f.spawn()) {
            f.freeSlots(rd);
            plan.deinit(f.gpa);
            f.gpa.destroy(rd);
            return null;
        }
        f.reads.append(f.gpa, rd) catch {
            f.freeSlots(rd);
            plan.deinit(f.gpa);
            f.gpa.destroy(rd);
            return null;
        };
        f.held_reads += 1;
        f.held_bytes += buffers_bytes;
        f.counts.made += 1;
        f.cond.signal(f.io);
        return s.name;
    }

    fn spawn(f: *Prefetch) bool {
        if (f.thread != null) return true;
        if (!f.exited.load(.acquire)) return false;
        f.stopping = false;
        f.exited.store(false, .release);
        _ = f.refs.fetchAdd(1, .acq_rel);
        f.thread = std.Thread.spawn(.{}, run, .{f}) catch {
            f.exited.store(true, .release);
            f.releaseOwner();
            return false;
        };
        return true;
    }

    /// Whether a request still waits for `name`'s read: queued or reading, inside the bound (passing it counts once).
    pub fn pending(f: *Prefetch, name: Name) bool {
        return f.pendingSince(name, null);
    }

    /// A preceding staged save shares the request's wait bound with its read.
    pub fn pendingSince(f: *Prefetch, name: Name, since: ?i128) bool {
        f.lock();
        defer f.unlock();
        const rd = f.reads.items[f.find(name) orelse return false];
        if (rd.slots[rd.consume].state == .full) return false; // admission can now detach and drain while disk continues
        if (rd.phase != .queued and rd.phase != .reading) return false;
        if (write.nowNs() - (since orelse rd.since) < f.limits.wait_ns) return true;
        if (!rd.late) f.counts.timeouts += 1;
        rd.late = true;
        return false;
    }

    /// `name`'s compatible active read, detached for the engine; queued or failed reads use the synchronous fallback.
    pub fn take(f: *Prefetch, name: Name, plan: *const Plan) ?*Read {
        f.lock();
        defer f.unlock();
        const i = f.find(name) orelse return null;
        const rd = f.reads.items[i];
        if (rd.phase != .queued and rd.phase != .failed and rd.plan.same(plan)) {
            _ = f.reads.orderedRemove(i);
            if (!rd.used) f.counts.used += 1;
            rd.used = true;
            rd.detached = true;
            return rd;
        }
        f.drop(f.reads.orderedRemove(i));
        return null;
    }

    /// The next immutable fragment, waiting for disk production; null after a complete file stream.
    pub fn next(f: *Prefetch, rd: *Read) anyerror!?Chunk {
        f.lock();
        defer f.unlock();
        std.debug.assert(rd.detached);
        while (true) {
            const slot = &rd.slots[rd.consume];
            if (slot.state == .full) {
                slot.state = .using;
                const index = rd.consume;
                rd.consume +%= 1;
                return .{ .piece = slot.piece, .off = slot.off, .bytes = slot.buf[0..slot.len], .slot = index };
            }
            if (!rd.producer_active and (rd.phase == .ready or rd.phase == .failed)) {
                if (rd.err) |err| return err;
                return null;
            }
            f.cond.waitUncancelable(f.io, &f.mutex);
        }
    }

    /// Release a consumed fragment's slot; its bytes must not be used afterward.
    pub fn recycle(f: *Prefetch, rd: *Read, chunk: Chunk) void {
        f.lock();
        defer f.unlock();
        std.debug.assert(rd.detached);
        const slot = &rd.slots[chunk.slot];
        std.debug.assert(slot.state == .using and slot.buf.ptr == chunk.bytes.ptr);
        slot.state = .empty;
        slot.len = 0;
        f.cond.signal(f.io);
    }

    /// Cancel and join the producer if needed, then free this detached read exactly once.
    pub fn finish(f: *Prefetch, rd: *Read) void {
        f.lock();
        defer f.unlock();
        std.debug.assert(rd.detached);
        rd.cancel.store(true, .release);
        f.cond.broadcast(f.io);
        while (rd.producer_active) f.cond.waitUncancelable(f.io, &f.mutex);
        f.release(rd);
    }

    /// Reads no queued request names any more go (engine thread): `named(ctx, name)` says which still wait.
    pub fn sweep(f: *Prefetch, ctx: anytype, comptime named: fn (@TypeOf(ctx), Name) bool) void {
        f.lock();
        defer f.unlock();
        var i: usize = 0;
        while (i < f.reads.items.len) {
            if (named(ctx, f.reads.items[i].name)) {
                i += 1;
                continue;
            }
            f.drop(f.reads.orderedRemove(i));
        }
    }

    /// `name`'s read, if any, out of the list and freed (engine thread): its state is forgotten.
    pub fn forget(f: *Prefetch, name: Name) void {
        f.lock();
        defer f.unlock();
        f.drop(f.reads.orderedRemove(f.find(name) orelse return));
    }

    /// A read out of the list freed now, or by the I/O thread when it is reading it (the lock held).
    fn drop(f: *Prefetch, rd: *Read) void {
        if (!rd.used) f.counts.cancelled += 1;
        if (rd.producer_active) {
            rd.cancel.store(true, .release);
            f.cond.broadcast(f.io);
            return;
        }
        f.release(rd);
    }

    /// A read's memory freed: exactly once, by whoever holds it last (the lock held).
    fn release(f: *Prefetch, rd: *Read) void {
        f.held_reads -= 1;
        f.held_bytes -= buffers_bytes;
        f.freeSlots(rd);
        rd.plan.deinit(f.gpa);
        f.gpa.destroy(rd);
        f.counts.freed += 1;
    }

    fn freeSlots(f: *Prefetch, rd: *Read) void {
        for (&rd.slots) |*slot| if (slot.buf.len > 0) f.gpa.free(slot.buf);
    }

    /// Cancel all reads and join within the normal stop budget.
    pub fn stop(f: *Prefetch) void {
        f.stopAt(write.nowNs() + @import("lane_host_rows.zig").stop_ns);
    }

    /// A blocked read retains only private memory after the owner's deadline; no host callback survives this call.
    pub fn stopAt(f: *Prefetch, deadline: i128) void {
        f.lock();
        f.stopping = true;
        f.notify = null;
        while (f.reads.pop()) |rd| f.drop(rd);
        f.cond.broadcast(f.io);
        f.unlock();
        while (!f.exited.load(.acquire) and write.nowNs() < deadline) write.io.nap();
        while (f.notifying.load(.acquire)) write.io.nap();
        if (f.thread) |thread| {
            if (f.exited.load(.acquire)) thread.join() else thread.detach();
        }
        f.thread = null;
    }

    pub fn destroy(f: *Prefetch) void {
        f.stopAt(write.nowNs());
        f.releaseOwner();
    }

    fn releaseOwner(f: *Prefetch) void {
        if (f.refs.fetchSub(1, .acq_rel) != 1) return;
        f.reads.deinit(f.gpa);
        f.owned_io.deinit();
        f.gpa.free(f.dir);
        f.gpa.destroy(f);
    }

    /// The I/O thread: oldest queued read first, disk outside the lock, terminal publication under it.
    fn run(f: *Prefetch) void {
        defer {
            f.exited.store(true, .release);
            f.releaseOwner();
        }
        f.lock();
        defer f.unlock();
        while (!f.stopping) {
            const rd = f.queuedNext() orelse {
                f.cond.waitUncancelable(f.io, &f.mutex);
                continue;
            };
            rd.phase = .reading;
            rd.producer_active = true;
            f.unlock();
            const t0 = write.nowNs();
            const err = fill(f, rd);
            const ns = write.nowNs() - t0;
            f.lock();
            rd.ns = ns;
            rd.err = err;
            rd.phase = if (err == null) .ready else .failed;
            rd.producer_active = false;
            f.cond.broadcast(f.io);
            if (err != null and !rd.cancel.load(.acquire)) f.counts.failed += 1;
            if (rd.cancel.load(.acquire) and !rd.detached) {
                f.release(rd);
                continue;
            }
            if (rd.detached) continue; // engine `finish` owns it and no host notification refers to it
            const n = f.notify orelse continue;
            f.notifying.store(true, .release);
            f.unlock();
            n.call(n.ctx);
            f.notifying.store(false, .release);
            f.lock();
        }
    }

    fn claim(f: *Prefetch, rd: *Read) !u1 {
        f.lock();
        defer f.unlock();
        while (true) {
            if (rd.cancel.load(.acquire)) return error.Cancelled;
            const index = rd.produce;
            const slot = &rd.slots[index];
            if (slot.state == .empty) {
                slot.state = .filling;
                return index;
            }
            f.cond.waitUncancelable(f.io, &f.mutex);
        }
    }

    fn abandon(f: *Prefetch, rd: *Read, index: u1) void {
        f.lock();
        defer f.unlock();
        rd.slots[index].state = .empty;
        f.cond.broadcast(f.io);
    }

    fn publish(f: *Prefetch, rd: *Read, index: u1, piece: usize, off: u64, len: usize) void {
        f.lock();
        defer f.unlock();
        const slot = &rd.slots[index];
        std.debug.assert(slot.state == .filling);
        slot.piece = piece;
        slot.off = off;
        slot.len = len;
        slot.state = .full;
        rd.produce +%= 1;
        f.cond.signal(f.io);
    }

    fn announce(f: *Prefetch, rd: *Read) void {
        f.lock();
        if (rd.detached or rd.announced) {
            f.unlock();
            return;
        }
        const n = f.notify orelse {
            f.unlock();
            return;
        };
        rd.announced = true;
        f.notifying.store(true, .release);
        f.unlock();
        n.call(n.ctx);
        f.notifying.store(false, .release);
    }

    fn queuedNext(f: *const Prefetch) ?*Read {
        for (f.reads.items) |x| if (x.phase == .queued) return x;
        return null;
    }

    fn fill(f: *Prefetch, rd: *Read) ?anyerror {
        f.reader(f, rd) catch |err| return err;
        return null;
    }
};

/// A store's prefetcher, made at first use with the environment's limits; `notify` is set if none is.
pub fn ensure(r: *Rows, io: std.Io, notify: Notify) ?*Prefetch {
    if (r.fetch == null) r.fetch = make(r, io, .fromEnv()) catch return null;
    const f = r.fetch.?;
    f.lock();
    defer f.unlock();
    if (f.notify == null) f.notify = notify;
    return f;
}

/// A prefetcher for `r` under `limits` (its thread starts with the first read); `r` owns it from here.
pub fn make(r: *Rows, _: std.Io, limits: Limits) !*Prefetch {
    const a = write.io.allocator;
    const f = try a.create(Prefetch);
    errdefer a.destroy(f);
    f.* = .{ .gpa = a, .io = undefined, .limits = limits, .dir = try a.dupeSentinel(u8, r.dir, 0), .block = r.keying.block };
    f.io = f.owned_io.io();
    r.fetch = f;
    return f;
}

/// A compatible staged save is not readable until the engine publishes its complete files and state index.
pub fn writing(s: *pc.Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span) ?Name {
    const r = s.disk orelse return null;
    const w = r.worker orelse return null;
    const have = s.find(prompt, starts, spans);
    var at: u32 = if (have) |e| e.at else 0;
    var name: ?Name = null;
    var jobs: [write.io.max_jobs]?*write.io.Job = undefined;
    w.cancellations(&jobs);
    for (jobs) |job| if (job) |j| {
        if (j.generation != w.generation.load(.acquire) or j.at <= at) continue;
        if (s.most == 0 or r.blankBytes(j.at) > s.budget) continue;
        const state: rows_store.State = .{ .name = j.name, .at = j.at, .tokens = j.tokens, .spans = j.spans, .starts = j.starts, .own = 0, .rows = 0, .used = 0, .shared = j.shared };
        if (!r.matches(&state, prompt, starts, spans)) continue;
        name = j.name;
        at = j.at;
    };
    return name;
}

/// The disk state a lookup of `prompt` would read back now (null: none, or it would not fit), as `read.recall` chooses.
pub fn choose(s: *pc.Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span) ?usize {
    const r = s.disk orelse return null;
    const i = select.choose(s, prompt, starts, spans, s.find(prompt, starts, spans)).restore orelse return null;
    if (r.blankBytes(r.states.items[i].at) > s.budget or s.most == 0) return null;
    return i;
}

/// Every file of `rd`, checked incrementally and published as ordered bounded fragments (the I/O thread).
pub fn readFiles(f: *Prefetch, rd: *Read) anyerror!void {
    for (rd.plan.pieces, 0..) |p, piece| {
        if (rd.cancel.load(.acquire)) return error.Cancelled;
        rd.failed_piece = piece; // still this one if anything in its read fails; the engine forgets it
        var buf: [1200]u8 = undefined;
        const path = try file.path(&buf, f.dir, &chain.hex(p.name), ext(p.kind), false);
        var o = try file.open(path, p.kind, p.name);
        defer o.close();
        if (o.trailer.payload != p.len or o.trailer.block != f.block or o.trailer.at != p.at) return error.RowsRead;
        var hash: std.hash.Wyhash = .init(0);
        var off: u64 = 0;
        while (off < p.len) {
            const index = try f.claim(rd);
            const logical = p.off + off;
            const got = file.payloadChunk(&o, rd.slots[index].buf, &off, &hash) catch |err| {
                f.abandon(rd, index);
                return err;
            };
            if (got.len == 0) {
                f.abandon(rd, index);
                return error.RowsRead;
            }
            f.publish(rd, index, piece, logical, got.len);
            f.announce(rd);
        }
        if (hash.final() != o.trailer.payload_hash) return error.RowsHash;
        rd.disk += o.trailer.payload + o.trailer.meta + @sizeOf(file.Trailer);
    }
}

test {
    _ = @import("rows_prefetch_test.zig");
    _ = @import("rows_prefetch_stream_test.zig");
}
