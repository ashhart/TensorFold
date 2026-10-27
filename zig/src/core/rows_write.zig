//! The engine stages owned images and publishes completions; one I/O thread writes and syncs their files.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const store = @import("rows_store.zig");
const image = @import("rows_image.zig");
const file = @import("rows_file.zig");
const chain = @import("rows_chain.zig");
const sync = @import("rows_write_sync.zig");
pub const io = @import("rows_io.zig");
const Rows = store.Rows;
pub const Want = sync.Want;
pub const Queued = sync.Queued;
pub const Job = sync.Job;
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

pub fn queued(r: *Rows, e: *pc.Entry, starts: []const u32) void {
    if (r.stopped) return;
    sync.queued(r, e, starts);
}

/// Staged work owns no entry pointer; eviction only removes an entry still waiting to be staged.
pub fn dropped(r: *Rows, e: *pc.Entry) void {
    sync.dropped(r, e);
}

pub fn pending(r: *const Rows) bool {
    if (r.sync_writes) return sync.pending(r);
    return (if (r.worker) |w| w.pending() > 0 else false) or (!r.stopped and r.spill != .evicted and r.queue.items.len > 0);
}

pub fn ready(r: *const Rows) bool {
    if (r.sync_writes) return sync.pending(r);
    return (if (r.worker) |w| w.ready() else false) or (!r.stopped and r.spill != .evicted and r.queue.items.len > 0);
}

/// A spill copies the entry before RAM releases it, without waiting for filesystem work or a byte permit.
pub fn evicted(r: *Rows, e: *pc.Entry) void {
    if (r.sync_writes) return sync.evicted(r, e);
    if (r.spill == .off or r.stopped) return;
    reap(r);
    for (r.queue.items, 0..) |q, i| if (q.entry == e) {
        _ = stage(r, r.queue.orderedRemove(i));
        return;
    };
}

pub fn wanted(r: *const Rows, n: chain.Stem) bool {
    if (r.sync_writes) return sync.wanted(r, n);
    return if (r.worker) |w| w.wanted(n) else false;
}

/// Reconcile finished jobs and stage at most one queued entry; an outstanding write alone does not spin the engine.
pub fn step(r: *Rows) bool {
    if (r.sync_writes) return sync.step(r);
    reap(r);
    if (r.stopped or r.spill == .evicted) return false;
    const q = r.queue.pop() orelse return false;
    _ = stage(r, q);
    return true;
}

pub fn abort(r: *Rows, j: *Job) void {
    sync.abort(r, j);
}

/// Invalidate old completions before an unlink or a change of model generation.
pub fn fence(r: *Rows) void {
    if (r.worker) |w| w.fence();
}

/// Stop admits no further jobs; an I/O syscall cannot extend the owner's shutdown deadline.
pub fn stop(r: *Rows, deadline: i128) void {
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

pub fn drain(r: *Rows, deadline: i128) usize {
    if (r.sync_writes) return sync.drain(r, deadline);
    while (pending(r) and nowNs() < deadline) {
        if (!step(r)) io.nap();
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
        defer j.destroy();
        const err: ?anyerror = j.failure orelse if (j.generation != w.generation.load(.acquire)) error.Cancelled else null;
        if (err) |e| {
            failed(r, j, e);
            r.admission.finish(r.dir, j.bytes, e == error.Cancelled);
            r.collect();
            continue;
        }
        publish(r, j) catch |e| {
            failed(r, j, e);
            r.admission.finish(r.dir, j.bytes, false);
            r.collect();
            continue;
        };
        r.admission.finish(r.dir, j.bytes, true);
    }
}

fn publish(r: *Rows, j: *io.Job) !void {
    if (r.find(j.name)) |_| return;
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
    r.counts.writes += 1;
    if (j.verify) r.counts.verified += 1;
    if (!@import("builtin").is_test) std.log.info("imprint: {d} tokens on disk: {d} new blocks and its own {d} MiB ({d} MiB this state, {d} MiB of rows it shares) in {d:.0} ms", .{ j.at, j.files.len - 1, s.own >> 20, j.bytes >> 20, s.rows >> 20, @as(f64, @floatFromInt(nowNs() - j.began)) / 1e6 });
}

fn stage(r: *Rows, q: Queued) bool {
    const at = q.entry.at;
    r.next_job +%= 1;
    const name = chain.state(r.keying, q.entry.tokens, q.entry.decode_spans, q.starts, at);
    const j = sync.prepare(r, q) catch |err| {
        r.counts.skipped += 1;
        r.failures.add(.{ .id = r.next_job, .generation = if (r.worker) |w| w.generation.load(.acquire) else 0, .name = name, .at = at, .err = err });
        pc.note("imprint: saving {d} tokens skipped ({s})", .{ at, @errorName(err) });
        return false;
    } orelse return false;
    defer {
        r.gpa.free(j.starts);
        r.gpa.free(j.parts);
        r.gpa.free(j.wants);
        r.gpa.free(j.own);
        r.gpa.destroy(j);
    }
    stageJob(r, j) catch |err| {
        r.counts.skipped += 1;
        r.failures.add(.{ .id = r.next_job, .generation = if (r.worker) |w| w.generation.load(.acquire) else 0, .name = j.name, .at = at, .err = err });
        if (err == error.GatherFailed or err == error.SnapshotGpuFailure) r.counts.failed += 1;
        pc.note("imprint: saving {d} tokens skipped ({s})", .{ at, @errorName(err) });
        return false;
    };
    return true;
}

fn stageJob(r: *Rows, prepared: *const Job) !void {
    if (r.worker == null) r.worker = try io.Worker.open(r.dir, r.staging_bytes, r.file_io);
    const w = r.worker.?;
    if (w.names(prepared.name)) return;
    const e = prepared.entry;
    const metadata = 4 * (3 + e.tokens.len + 2 * e.decode_spans.len + prepared.starts.len);
    var bytes: u64 = 0;
    var count: usize = 0;
    var charge: u64 = 0;
    while (true) {
        bytes = newBytes(r, prepared, metadata, &count);
        charge = bytes + @sizeOf(io.Job) + count * @sizeOf(io.File) + prepared.wants.len * @sizeOf(Want) + 4 * @sizeOf([]const u8);
        if (!w.fits(charge)) return error.StagingFull;
        const reserved = w.reserved();
        if (!r.budget(bytes + reserved)) return error.DailyBudgetSpent;
        if (!r.room(bytes + reserved)) return error.OverCap;
        var after_count: usize = 0;
        if (newBytes(r, prepared, metadata, &after_count) == bytes) break;
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
    for (prepared.wants) |want| if (!r.blocks.contains(chain.stem(want.name))) {
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
    j.* = .{ .arena = arena, .id = r.next_job, .generation = w.generation.load(.acquire), .name = prepared.name, .at = e.at, .shared = e.shared, .tokens = tokens, .spans = spans, .starts = starts, .wants = wants, .files = files, .bytes = bytes, .staged = charge, .verify = r.verify, .began = prepared.began };
    w.submit(j);
}

fn newBytes(r: *const Rows, j: *const Job, metadata: usize, count: *usize) u64 {
    var bytes = image.total(j.own) + metadata + @sizeOf(file.Trailer);
    count.* = 1;
    for (j.wants) |want| if (!r.blocks.contains(chain.stem(want.name))) {
        bytes += want.bytes;
        count.* += 1;
    };
    return bytes;
}
