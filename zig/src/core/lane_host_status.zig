//! A lane host's status counts: running and waiting requests, the 2 s decode and prompt rates, each stream's tokens.
const std = @import("std");
const api = @import("engine_api.zig");
const snap = @import("prompt_cache_status.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

/// Tokens a round landed, and when.
pub const Mark = struct { at: i96, tokens: u64 };
const window_ns: i96 = 2 * std.time.ns_per_s;

/// Engine.status: the counts, copied from the engine thread's published snapshot without the host's lock (issue 3200).
pub fn read(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    var s: Snapshot = undefined;
    var n: usize = 0;
    var now: i96 = undefined;
    var tokens: u64 = 0;
    while (true) {
        n = 0;
        const v1 = h.st.seq.load(.acquire);
        if (v1 % 2 != 0) {
            std.atomic.spinLoopHint();
            continue;
        }
        s = h.st.snap;
        const slot = s.slot;
        _ = h.st.readers[slot].fetchAdd(1, .acq_rel);
        if (h.st.seq.load(.acquire) != v1) {
            _ = h.st.readers[slot].fetchSub(1, .acq_rel);
            continue;
        }
        now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        tokens = 0;
        for (h.st.decode_slots[slot].items) |m| {
            if (m.at >= now - window_ns) tokens += m.tokens;
        }
        if (s.live_len > 0) {
            const buf = h.st.live_slots[slot].items;
            n = @min(stream_tokens.len, s.live_len);
            @memcpy(stream_tokens[0..n], buf[0..n]);
        }
        const stable = h.st.seq.load(.acquire) == v1;
        _ = h.st.readers[slot].fetchSub(1, .acq_rel);
        if (stable) break;
    }
    out.* = .{
        .healthy = !h.closing.load(.acquire) and !h.stale.load(.acquire) and (if (h.health) |source| source.read(source.ctx) else true),
        .running = @intCast(s.running),
        .waiting = @intCast(s.waiting),
        .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
        .prefill_tokens_per_second = if (now - s.prefill_at <= window_ns) s.prefill_rate else 0,
        .streams = n,
        .generation_tokens = s.generation_tokens,
        .prompt_cache = s.cache_status,
        .disk = s.disk_status,
        .fsr = if (h.fsr) |source| source.read(source.ctx) else null,
        .fill_prompt_tokens = s.fill_prompt_tokens,
        .fill_tokens_left = s.fill_tokens_left,
        .fill_unknown = s.fill_unknown,
        .fill_run = s.fill_run,
    };
}

/// The lane host's status state: the published snapshot, its seqlock, the live slots and the round-wait counter.
pub const State = struct {
    snap: Snapshot = .{},
    seq: std.atomic.Value(u32) = .init(0), // odd while the engine thread publishes
    live_slots: [2]std.ArrayList(u32) = .{ .empty, .empty },
    decode_slots: [2]std.ArrayList(Mark) = .{ .empty, .empty },
    readers: [2]std.atomic.Value(u32) = .{ .init(0), .init(0) }, // registered before slot access; writers drain their target
    live_tokens: std.ArrayList(u32) = .empty,
    live_generated: u64 = 0,
    cache_status: ?api.Status.PromptCache = null,
    disk_status: ?api.Status.DiskCache = null,
    round_waits: std.atomic.Value(u64) = .init(0),
    engine_id: std.Thread.Id = 0,
    engine_running: std.atomic.Value(bool) = .init(false),
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
};

/// The engine thread's id, so its lock acquisitions can be told from the others'.
pub fn engineStart(h: *LaneHost) void {
    h.st.engine_id = std.Thread.getCurrentId();
    h.st.engine_running.store(true, .release);
}

/// Takes the host's lock for the engine thread and counts a wait; false when the caller must lock.
pub fn takeEngineLock(h: *LaneHost) bool {
    if (!h.st.engine_running.load(.acquire) or std.Thread.getCurrentId() != h.st.engine_id) return false;
    if (h.mutex.tryLock()) return true;
    _ = h.st.round_waits.fetchAdd(1, .monotonic);
    h.mutex.lockUncancelable(h.io);
    return true;
}

/// Frees the status state when the host stops.
pub fn freeSlots(h: *LaneHost) void {
    h.st.seq.store(h.st.seq.load(.monotonic) | 1, .release); // teardown admits no new slot readers
    for (&h.st.readers) |*readers| while (readers.load(.acquire) != 0) std.atomic.spinLoopHint();
    h.st.live_tokens.deinit(h.gpa);
    for (&h.st.live_slots) |*slot| slot.deinit(h.gpa);
    for (&h.st.decode_slots) |*slot| slot.deinit(h.gpa);
    h.st.snap = .{};
    h.st.seq.store(h.st.seq.load(.monotonic) +% 1, .release); // a late reader sees the empty terminal snapshot
}

/// The lane host's status state: the published snapshot, its seqlock, the live slots and the round-wait counter.
pub const Snapshot = struct {
    slot: usize = 0,
    live_len: usize = 0,
    running: usize = 0,
    waiting: usize = 0,
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    generation_tokens: u64 = 0,
    cache_status: ?api.Status.PromptCache = null,
    disk_status: ?api.Status.DiskCache = null,
    fill_prompt_tokens: ?u32 = null,
    fill_tokens_left: ?u32 = null,
    fill_unknown: bool = false,
    fill_run: u64 = 0,
};

/// Publishes the snapshot for the next read; the caller holds no lock (issue 3200).
pub fn publish(h: *LaneHost) void {
    h.lock();
    defer h.unlock();
    var tmp: api.Status = .{};
    fillTotals(h, &tmp);
    const slot = (h.st.snap.slot + 1) % 2;
    h.st.seq.store(h.st.seq.load(.monotonic) | 1, .release); // odd: a reader retries
    while (h.st.readers[slot].load(.acquire) != 0) std.atomic.spinLoopHint();
    var len: usize = 0;
    if (h.st.live_tokens.items.len > 0) {
        h.st.live_slots[slot].clearRetainingCapacity();
        h.st.live_slots[slot].appendSlice(h.gpa, h.st.live_tokens.items) catch {};
        len = h.st.live_slots[slot].items.len;
    }
    h.st.decode_slots[slot].clearRetainingCapacity();
    h.st.decode_slots[slot].appendSlice(h.gpa, h.decoded.items) catch {};
    h.st.snap = .{
        .slot = slot,
        .live_len = len,
        .running = h.admitted.items.len,
        .waiting = h.queued.items.len,
        .prefill_rate = h.st.prefill_rate,
        .prefill_at = h.st.prefill_at,
        .generation_tokens = h.st.live_generated,
        .cache_status = h.st.cache_status,
        .disk_status = h.st.disk_status,
        .fill_prompt_tokens = tmp.fill_prompt_tokens,
        .fill_tokens_left = tmp.fill_tokens_left,
        .fill_unknown = tmp.fill_unknown,
        .fill_run = tmp.fill_run,
    };
    h.st.seq.store(h.st.seq.load(.monotonic) +% 1, .release); // even: the copy is stable
}

/// A job whose prompt pass has not reported yet is filling; the engine thread publishes that with `fill_on`.
fn fillTotals(h: *LaneHost, out: *api.Status) void {
    var total: u64 = 0;
    var left: u64 = 0;
    var filling = false;
    var run: u64 = 0;
    const reported = h.core.backend.vtable.reports_fill; // a backend that never stores `fill_at` has no honest totals
    for (h.admitted.items) |job| {
        if (!job.fill_on.load(.acquire) or job.request.background) continue; // a warm pass prefills the next turn: no fill
        filling = true;
        var fields: [16]u8 = undefined; // id and seq hashed apart: XOR of the raw values collides (2^0 == 3^1)
        std.mem.writeInt(u64, fields[0..8], job.id, .little);
        std.mem.writeInt(u64, fields[8..16], job.seq, .little);
        run = (run *% 0x9E3779B97F4A7C15) ^ std.hash.Wyhash.hash(0, &fields);
        if (!reported) continue; // no counter for it: its prompt stays out of the totals rather than guessed
        const prompt = job.request.prompt.len;
        total += prompt;
        left += prompt -| job.stream.fill_at.load(.monotonic);
    }
    if (!filling) return;
    if (reported) {
        out.fill_prompt_tokens = @intCast(total);
        out.fill_tokens_left = @intCast(left);
    } else out.fill_unknown = true;
    run &= (1 << 53) - 1; // JSON numbers stay exact to 2^53: a wider marker loses bits in the page's JavaScript
    out.fill_run = if (run == 0) 1 else run;
}

/// `n` tokens delivered now; marks older than the window drop.
pub fn noteDecoded(h: *LaneHost, n: usize, first: bool) void {
    {
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        var keep: usize = 0;
        for (h.decoded.items) |m| {
            if (m.at < now - window_ns) continue;
            h.decoded.items[keep] = m;
            keep += 1;
        }
        h.decoded.shrinkRetainingCapacity(keep);
        h.decoded.append(h.gpa, .{ .at = now, .tokens = n }) catch {};
        @import("lane_host_load.zig").decoded(h, n, first);
    }
    publish(h);
}

/// Each started stream's context and generated tokens, for the next status read.
pub fn noteLive(h: *LaneHost) void {
    {
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        h.st.live_tokens.clearRetainingCapacity();
        h.st.live_generated = 0;
        const now_ms = @import("rows_io.zig").wallMs();
        if (h.cache) |s| {
            if (h.st.cache_status == null or now_ms - h.cache_at >= 1000) { // the full copies at 1 Hz at most
                h.st.cache_status = snap.PromptCache(s);
                h.st.disk_status = snap.Disk(s);
                h.cache_at = now_ms;
            } else {
                h.st.cache_status.?.states = s.entries.items.len; // /metrics reads these every round, as before
                h.st.cache_status.?.bytes = s.held + s.spare();
            }
        } else {
            h.st.cache_status = null;
            h.st.disk_status = null;
        }
        for (h.admitted.items) |job| if (job.started) {
            h.st.live_tokens.append(h.gpa, @intCast(job.stream.context.items.len)) catch {};
            h.st.live_generated += @intCast(job.stream.emitted().len);
        };
    }
    publish(h);
}
