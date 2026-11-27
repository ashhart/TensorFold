//! The engine stages owned images and publishes completions; one I/O thread writes and syncs their files.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const store = @import("rows_store.zig");
const image = @import("rows_image.zig");
const file = @import("rows_file.zig");
const chain = @import("rows_chain.zig");
const sync = @import("rows_write_sync.zig");
pub const io = @import("rows_io.zig");
pub const slice = @import("rows_slice.zig");
const Rows = store.Rows;
pub const Want = sync.Want;
pub const Queued = sync.Queued;
pub const Job = sync.Job;
pub const Sliced = slice.State;
pub const hostBytes = slice.hostBytes;
pub const formatHostPlan = slice.formatHostPlan;
pub const nowNs = io.nowNs;

pub const Failure = struct { id: u64, generation: u64, name: chain.Name, at: u32, err: anyerror };

/// Recent failures retain each job's identity and outcome, including failures that follow an earlier failure.
pub const Failures = struct {
    records: [64]?Failure = @splat(null),
    count: u64 = 0,

    pub fn add(f: *Failures, record: Failure) void {
        f.records[@intCast(f.count % f.records.len)] = record;
        f.count += 1;
    }
};

/// Carry the notice along an exact image prefix before keeping its extension can evict it.
pub fn imageNoted(s: *const pc.Store, tokens: []const u32) bool {
    for (s.entries.items) |e| if (e.image_noted and std.mem.startsWith(u32, tokens, e.tokens) and
        @import("cache_images.zig").matches(s, e)) return true;
    return false;
}

pub fn queued(r: *Rows, e: *pc.Entry, starts: []const u32) void {
    if (r.stopped) return;
    if (e.image_frame != 0 or e.images.len != 0 or !@import("cache_images.zig").diskSafe(e.tokens)) {
        if (!e.image_noted) pc.note("imprint: {d} tokens kept RAM-only (ImageConversation); image states are not written to disk", .{e.at});
        e.image_noted = true;
        return;
    }
    sync.queued(r, e, starts);
    if (r.img.transient) |mutable| if (mutable(r.family.ptr, e.saved)) preserve(r, e, true);
}

/// Eviction removes a queued save and abandons an active sliced gather (PR 1); a waited save counts as skipped.
pub fn dropped(r: *Rows, e: *pc.Entry) void {
    slice.onEntryDropped(r, e);
    if (r.waiting == e) r.waiting = null;
    for (r.queue.items) |q| if (q.entry == e) {
        if (q.waited and !r.stopped) {
            r.next_job +%= 1; // the id a save staged now would take, so no job in flight shares it
            skip(r, chain.state(r.keying, e.tokens, e.decode_spans, q.starts, e.at), e.at, error.StagingFull);
        }
        break;
    };
    sync.dropped(r, e);
}

pub fn pending(r: *const Rows) bool {
    if (r.sync_writes) return sync.pending(r);
    return r.sliced.active != null or (if (r.worker) |w| w.pending() > 0 else false) or (!r.stopped and (r.spill != .evicted or r.flush_clock != null) and r.queue.items.len > 0);
}

pub fn ready(r: *const Rows) bool {
    if (r.sync_writes) return sync.pending(r);
    if (if (r.worker) |w| w.ready() else false) return true;
    if (r.stopped) return false;
    if (r.sliced.active != null) return slice.canStep(r);
    return r.spill != .evicted and r.queue.items.len > 0 and !blocked(r);
}

/// Whether the newest queued save was refused room by the jobs in flight and neither slots nor sliced bytes freed.
fn blocked(r: *const Rows) bool {
    const e = r.waiting orelse return false;
    if (r.queue.items.len == 0 or r.queue.items[r.queue.items.len - 1].entry != e) return false;
    if (r.worker) |w| {
        if (r.sliced.waiting_slot) return !w.hasSlot();
        return w.slicedUsed() >= r.sliced.waiting_used;
    }
    return false;
}

/// A spill copies a queued entry before RAM releases it; an entry already mid-gather abandons in PR 1.
pub fn evicted(r: *Rows, e: *pc.Entry) void {
    if (r.sliced.active) |act| if (act.entry == e) {
        slice.abandon(r, error.EntryDropped);
        return;
    };
    if (r.spill == .off) return;
    preserve(r, e, false);
}

/// Borrowed rows stay live while a keep waits; an eviction still cannot wait for a permit.
fn preserve(r: *Rows, e: *pc.Entry, borrowed: bool) void {
    if (r.stopped) return;
    if (r.sync_writes) return sync.preserve(r, e);
    reap(r);
    while (true) {
        const i = for (r.queue.items, 0..) |q, k| {
            if (q.entry == e) break k;
        } else return;
        switch (stage(r, r.queue.orderedRemove(i), borrowed, borrowed)) {
            .done => return,
            .waiting => {
                while (r.worker) |w| {
                    if (w.ready() or w.pending() == 0) break;
                    if (!slice.gatherSlice(r, slice.idle_slice_ns)) io.nap();
                }
                reap(r);
            },
            .oversized => {
                // No full-image arena fits: publish older jobs, then stream bounded pieces before rows expire.
                while (r.worker) |w| {
                    reap(r);
                    if (w.pending() == 0) break;
                    if (!slice.gatherSlice(r, slice.idle_slice_ns)) io.nap();
                }
                sync.preserve(r, e);
                return;
            },
        }
    }
}

pub fn wanted(r: *const Rows, n: chain.Stem) bool {
    if (r.sync_writes) return sync.wanted(r, n);
    return if (r.worker) |w| w.wanted(n) else false;
}

/// Reconcile finished jobs and gather one idle-sized slice of the active or newest queued save.
pub fn step(r: *Rows) bool {
    return stepBudget(r, slice.idle_slice_ns);
}

/// Reconcile finished jobs and advance sliced gather for at least one file and at most `budget_ns`.
pub fn stepBudget(r: *Rows, budget_ns: i128) bool {
    if (r.sync_writes) return sync.step(r);
    reap(r);
    if (r.stopped) return false;
    if (r.sliced.active != null) return slice.gatherSlice(r, budget_ns);
    if (blocked(r) or (r.spill == .evicted and r.flush_clock == null)) return false;
    const q = r.queue.pop() orelse return false;
    if (stageSliced(r, q) == .waiting) return false;
    _ = slice.gatherSlice(r, budget_ns);
    return true;
}

pub fn abort(r: *Rows, j: *Job) void {
    sync.abort(r, j);
}

/// Invalidate old completions before an unlink or a change of model generation.
pub fn fence(r: *Rows) void {
    slice.abandon(r, error.Cancelled);
    if (r.worker) |w| w.fence();
}

/// Stop admits no further jobs; an I/O syscall cannot extend the owner's shutdown deadline.
pub fn stop(r: *Rows, deadline: i128) void {
    if (r.fetch) |f| f.stopAt(deadline);
    if (r.stopped) return;
    r.stopped = true;
    fence(r);
    if (r.worker) |w| {
        var jobs: [io.max_jobs]?*io.Job = undefined;
        w.cancellations(&jobs);
        for (jobs) |job| if (job) |j| {
            const err = if (j.done.load(.acquire)) j.failure orelse error.Cancelled else error.Cancelled;
            if (err == error.Cancelled) r.failures.add(.{ .id = j.id, .generation = j.generation, .name = j.name, .at = j.at, .err = err }) else failed(r, j, err);
            r.admission.finish(r.dir, j.bytes, err == error.Cancelled);
        };
        r.worker = null;
        w.close(deadline);
    }
}

/// A spill-enabled close also drains RAM, newest first; completed saves keep their cap space over older entries.
pub fn drain(r: *Rows, deadline: i128) usize {
    const previous = r.flush_clock;
    defer r.flush_clock = previous;
    if (r.spill != .off) {
        if (r.sync_writes) {
            while (r.job != null and nowNs() < deadline) _ = sync.step(r);
        } else while (r.worker) |w| {
            reap(r);
            if (w.pending() == 0 or nowNs() >= deadline) break;
            if (!slice.gatherSlice(r, @min(slice.idle_slice_ns, @max(0, deadline - nowNs())))) io.nap();
        }
        r.flush_clock = r.clock;
    }
    if (r.sync_writes) return sync.drain(r, deadline);
    while (pending(r) and nowNs() < deadline) {
        if (r.sliced.active != null) {
            if (!stepBudget(r, @min(slice.idle_slice_ns, @max(0, deadline - nowNs())))) io.nap();
            continue;
        }
        if (r.flush_clock != null) {
            reap(r);
            if (nowNs() >= deadline) break;
            if (r.worker) |w| if (w.pending() > 0) {
                io.nap();
                continue;
            };
        }
        if (!stepBudget(r, @min(slice.idle_slice_ns, @max(0, deadline - nowNs())))) io.nap();
    }
    reap(r);
    const left = r.queue.items.len + (if (r.worker) |w| w.pending() else @as(usize, 0));
    stop(r, deadline);
    if (left > 0) pc.note("imprint: the stop's deadline passed with {d} kept states unwritten", .{left});
    return left;
}

fn failed(r: *Rows, j: *const io.Job, err: anyerror) void {
    r.counts.failed += 1;
    if (err == error.VerifyFailed) r.counts.differed += 1;
    r.failures.add(.{ .id = j.id, .generation = j.generation, .name = j.name, .at = j.at, .err = err });
    pc.note("imprint: writing {d} tokens failed ({s}); the save is skipped", .{ j.at, @errorName(err) });
}

/// Only this engine-thread function changes indexes or exposes completed files, after all writes and syncs succeeded.
pub fn reap(r: *Rows) void {
    const w = r.worker orelse return;
    while (w.take()) |j| {
        slice.onReaped(r, j);
        defer j.destroy();
        r.waiting = null;
        const err: ?anyerror = j.failure orelse if (j.abandoned.load(.acquire)) j.abandon_err orelse error.Cancelled else if (j.generation != w.generation.load(.acquire)) error.Cancelled else null;
        if (err) |e| {
            failed(r, j, e);
            const disk_ok = e == error.Cancelled or e == error.EntryDropped or e == error.NotWhole;
            r.admission.finish(r.dir, j.bytes, disk_ok);
            r.collect();
            continue;
        }
        publish(r, j) catch |e| {
            failed(r, j, e);
            // The job that owed the block already backed admission off.
            r.admission.finish(r.dir, j.bytes, e == error.MissingBlock);
            r.collect();
            continue;
        };
        r.admission.finish(r.dir, j.bytes, true);
    }
}

fn publish(r: *Rows, j: *io.Job) !void {
    if (r.find(j.name)) |_| return;
    if (!complete(r, j)) return error.MissingBlock;
    var s: store.State = .{ .name = j.name, .at = j.at, .tokens = &.{}, .spans = &.{}, .starts = &.{}, .own = j.files[j.files.len - 1].bytes, .rows = 0, .used = 0, .shared = j.shared };
    s.tokens = try r.gpa.dupe(u32, j.tokens);
    errdefer r.gpa.free(s.tokens);
    s.spans = try r.gpa.dupe(@import("cache_modes.zig").Span, j.spans);
    errdefer r.gpa.free(s.spans);
    s.starts = try r.gpa.dupe(u32, j.starts);
    errdefer r.gpa.free(s.starts);
    try r.states.ensureUnusedCapacity(r.gpa, 1);
    try r.blocks.ensureUnusedCapacity(r.gpa, @intCast(j.files.len - 1));
    for (j.files[0 .. j.files.len - 1]) |*f| {
        try f.writer.?.publish();
        const refs = if (r.blocks.get(chain.stem(f.trailer.name))) |b| b.refs else 0;
        r.blocks.putAssumeCapacity(chain.stem(f.trailer.name), .{ .bytes = f.bytes, .refs = refs });
        r.spent(f.bytes);
    }
    try j.files[j.files.len - 1].writer.?.publish();
    r.spent(s.own);
    for (j.wants) |want| s.rows += want.bytes;
    r.states.appendAssumeCapacity(s);
    r.ref(j.wants, 1);
    r.touch(r.states.items.len - 1);
    noteWriteLatency(r, nowNs() - j.began);
    r.counts.writes += 1;
    if (j.verify) r.counts.verified += 1;
    if (!@import("builtin").is_test) std.log.info("imprint: {d} tokens on disk: {d} new blocks and its own {d} MiB ({d} MiB this state, {d} MiB of rows it shares) in {d:.0} ms", .{ j.at, j.files.len - 1, s.own >> 20, j.bytes >> 20, s.rows >> 20, @as(f64, @floatFromInt(nowNs() - j.began)) / 1e6 });
}

/// Whether every block the job's state names is on disk or among the files it publishes now.
fn complete(r: *const Rows, j: *const io.Job) bool {
    blocks: for (j.wants) |want| {
        const n = chain.stem(want.name);
        if (r.blocks.contains(n)) continue;
        for (j.blocks()) |f| if (std.mem.eql(u8, &chain.stem(f.trailer.name), &n)) continue :blocks;
        return false;
    }
    return true;
}

const Stage = enum { done, waiting, oversized };

/// Start a normal write-behind job with deferred per-file gathering; waits only for a job slot or file-0 permit room.
fn stageSliced(r: *Rows, q: Queued) Stage {
    const at = q.entry.at;
    r.next_job +%= 1;
    const name = chain.state(r.keying, q.entry.tokens, q.entry.decode_spans, q.starts, at);
    const j = sync.prepare(r, q) catch |err| {
        skip(r, name, at, err);
        return .done;
    } orelse return .done;
    var queued_again = false;
    defer {
        if (!queued_again) r.gpa.free(j.starts);
        r.gpa.free(j.parts);
        r.gpa.free(j.wants);
        r.gpa.free(j.own);
        r.gpa.destroy(j);
    }
    slice.start(r, j) catch |err| {
        if (err == error.StagingBusy) {
            queued_again = true;
            r.waiting = q.entry;
            r.queue.appendAssumeCapacity(.{ .entry = q.entry, .starts = q.starts, .waited = true });
            return .waiting;
        }
        skip(r, j.name, at, err);
        if (err == error.GatherFailed or err == error.SnapshotGpuFailure) r.counts.failed += 1;
        return .done;
    };
    return .done;
}

/// A borrowed keep can stream an oversized image; other evictions retain the whole-image permit refusal.
fn stage(r: *Rows, q: Queued, may_wait: bool, borrowed: bool) Stage {
    const at = q.entry.at;
    r.next_job +%= 1;
    const name = chain.state(r.keying, q.entry.tokens, q.entry.decode_spans, q.starts, at);
    const j = sync.prepare(r, q) catch |err| {
        skip(r, name, at, err);
        return .done;
    } orelse return .done;
    var queued_again = false;
    defer {
        if (!queued_again) r.gpa.free(j.starts);
        r.gpa.free(j.parts);
        r.gpa.free(j.wants);
        r.gpa.free(j.own);
        r.gpa.destroy(j);
    }
    stageJob(r, j, may_wait) catch |err| {
        if (err == error.StagingBusy or (borrowed and err == error.StagingFull)) {
            if (borrowed and (!q.waited or err == error.StagingFull)) pc.note("imprint: {d} borrowed tokens: {s}; {s}", .{
                at,
                @errorName(err),
                if (err == error.StagingBusy) @as([]const u8, "waiting for staging permit") else "streaming bounded pieces",
            });
            queued_again = true;
            if (err == error.StagingBusy) r.waiting = q.entry;
            r.queue.appendAssumeCapacity(.{ .entry = q.entry, .starts = q.starts, .waited = err == error.StagingBusy });
            return if (err == error.StagingBusy) .waiting else .oversized;
        }
        skip(r, j.name, at, err);
        if (err == error.GatherFailed or err == error.SnapshotGpuFailure) r.counts.failed += 1;
        return .done;
    };
    return .done;
}

/// A save given up: counted, recorded under the job id it would have had, and noted.
fn skip(r: *Rows, name: chain.Name, at: u32, err: anyerror) void {
    r.counts.skipped += 1;
    r.failures.add(.{ .id = r.next_job, .generation = if (r.worker) |w| w.generation.load(.acquire) else 0, .name = name, .at = at, .err = err });
    pc.note("imprint: saving {d} tokens skipped ({s})", .{ at, @errorName(err) });
}

fn stageJob(r: *Rows, prepared: *const Job, may_wait: bool) !void {
    const w = try slice.ensureWorker(r);
    if (w.names(prepared.name)) return;
    const e = prepared.entry;
    const metadata = 4 * (3 + e.tokens.len + 2 * e.decode_spans.len + prepared.starts.len);
    var promised: io.Promised = .empty;
    defer promised.deinit(r.gpa);
    var count: usize = 0;
    try w.promised(r.gpa, &promised);
    var bytes = slice.newBytes(r, &promised, prepared, metadata, &count);
    var charge: u64 = 0;
    while (true) {
        charge = charged(bytes, count, prepared.wants.len);
        if (!w.fitsEviction(charge)) return if (may_wait and w.busyEviction(charge)) error.StagingBusy else error.StagingFull;
        const reserved = w.reserved();
        if (!r.budget(bytes + reserved)) return error.DailyBudgetSpent;
        if (!r.room(bytes + reserved)) return error.OverCap;
        try w.promised(r.gpa, &promised); // a fence inside room frees the blocks cancelled jobs promised
        var again_count: usize = 0;
        const again = slice.newBytes(r, &promised, prepared, metadata, &again_count);
        if (again == bytes) break;
        bytes = again;
        count = again_count;
    }
    if (r.admission.reserve(r.dir, bytes) != .ready) return error.UnderFreeDiskFloor;
    errdefer r.admission.finish(r.dir, bytes, true);
    const j = try io.allocator.create(io.Job);
    errdefer io.allocator.destroy(j);
    var arena = std.heap.ArenaAllocator.init(io.allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const files = try a.alloc(io.File, count);
    var built: usize = 0;
    for (prepared.wants) |want| if (slice.lacks(r, &promised, want)) {
        var ranges: std.ArrayList(image.Range) = .empty;
        defer ranges.deinit(r.gpa);
        try image.blockRanges(prepared.parts, r.keying.block, want.window, want.k, &ranges, r.gpa);
        files[built] = .{ .trailer = .{ .kind = @backingInt(file.Kind.block), .payload = 0, .payload_hash = 0, .name = want.name, .block = r.keying.block }, .payload = try a.alloc(u8, @intCast(image.total(ranges.items))) };
        try r.img.gather(r.family.ptr, e.saved, ranges.items, files[built].payload);
        built += 1;
    };
    const own = &files[built];
    own.* = .{ .trailer = .{ .kind = @backingInt(file.Kind.state), .payload = 0, .payload_hash = 0, .name = prepared.name, .at = e.at, .block = r.keying.block, .flags = @intFromBool(e.shared) }, .payload = try a.alloc(u8, @intCast(image.total(prepared.own))) };
    if (own.payload.len > 0) try r.img.gather(r.family.ptr, e.saved, prepared.own, own.payload);
    const tokens = try a.dupe(u32, e.tokens);
    const spans = try a.dupe(@import("cache_modes.zig").Span, e.decode_spans);
    const starts = try a.dupe(u32, prepared.starts);
    const counts = try a.create([3]u32);
    counts.* = .{ @intCast(tokens.len), @intCast(spans.len), @intCast(starts.len) };
    own.meta = try a.dupe([]const u8, &.{ std.mem.asBytes(counts), std.mem.sliceAsBytes(tokens), std.mem.sliceAsBytes(spans), std.mem.sliceAsBytes(starts) });
    const wants = try a.dupe(Want, prepared.wants);
    j.* = .{ .arena = arena, .id = r.next_job, .generation = w.generation.load(.acquire), .name = prepared.name, .at = e.at, .shared = e.shared, .eviction = true, .tokens = tokens, .spans = spans, .starts = starts, .wants = wants, .files = files, .bytes = bytes, .staged = charge, .verify = r.verify, .began = prepared.began };
    w.submitEviction(j);
}

/// A staged job's charge: its files' bytes and the bookkeeping the worker holds beside them.
fn charged(bytes: u64, files: usize, wants: usize) u64 {
    return bytes + @sizeOf(io.Job) + files * @sizeOf(io.File) + wants * @sizeOf(Want) + 4 * @sizeOf([]const u8);
}

/// The most staging a state of `at` tokens charges: none of its blocks on disk, spans and starts at most one a token.
pub fn coldCharge(r: *const Rows, at: u32) !u64 {
    if (at == 0) return 0;
    const ps = try r.parts(r.gpa, at);
    defer r.gpa.free(ps);
    var groups: [8]u32 = undefined;
    var blocks: usize = 0;
    for (try image.groups(ps, &groups)) |g| {
        const w = image.wholeBlocks(at, r.keying.block, g);
        blocks += w.end - w.first;
    }
    const metadata = 4 * (4 + 4 * @as(u64, at));
    return charged(image.imageBytes(ps) + metadata + (blocks + 1) * @sizeOf(file.Trailer), blocks + 1, blocks);
}

// Times a write took, in milliseconds: the last, the smoothed and the histogram, and the wall time it landed.
fn noteWriteLatency(r: *Rows, ns: i128) void {
    if (ns <= 0) return;
    const ms = @as(f64, @floatFromInt(ns)) / 1e6;
    r.file_io.write_ms_last = ms;
    r.file_io.write_ms_ema = if (r.file_io.write_ms_ema == 0) ms else 0.8 * r.file_io.write_ms_ema + 0.2 * ms;
    r.file_io.write_lat.observe(ms);
    r.file_io.last_write_ms = @import("rows_io.zig").wallMs();
}
