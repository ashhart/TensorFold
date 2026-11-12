//! Read-only copies of the prompt cache and the disk tier for the dashboard: plain data, no store touched.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const status = @import("engine_status.zig").Status;
const chain = @import("rows_chain.zig");

/// Set by the server before an engine loads when `--dashboard` is on; off, kept states are not named (no prefix hash).
pub var named: std.atomic.Value(bool) = .init(false);

/// An entry's wall-clock times and resumes: shown on the dashboard, never read by eviction (`used` orders that).
pub const Seen = struct {
    id: chain.Stem = @splat(0), // the state's name stem: its `.st` file's name when the store keeps it on disk
    created_ms: i64 = 0,
    last_ms: i64 = 0,
    hits: u64 = 0,

    pub fn now() Seen {
        const t = @import("rows_io.zig").wallMs();
        return .{ .created_ms = t, .last_ms = t };
    }

    /// A new entry's times, and its disk-tier name (no tier: an empty identity); hashed only for the dashboard.
    pub fn of(s: *const pc.Store, tokens: []const u32, spans: []const [2]u32, starts: []const u32, at: u32) Seen {
        var out = now();
        if (!named.load(.monotonic)) return out;
        out.id = chain.stem(if (s.disk) |r| named: {
            const below = pc.rows.startsBelow(r, s.gpa, starts, at) catch break :named chain.state(r.keying, tokens, spans, &.{}, at);
            defer s.gpa.free(below);
            break :named chain.state(r.keying, tokens, spans, below, at);
        } else chain.state(.{ .identity = 0, .block = 0 }, tokens, spans, &.{}, at));
        return out;
    }

    /// A keep of the same state again: its last use moves, the store's clock passes through.
    pub fn touch(s: *Seen, clock: u64) u64 {
        s.last_ms = @import("rows_io.zig").wallMs();
        return clock;
    }

    /// A resume: counted, and its last use moves.
    pub fn hit(s: *Seen, clock: u64) u64 {
        s.hits += 1;
        return s.touch(clock);
    }
};

/// The kept states' copy: counts, capacity, spare bytes and the most recently used entries, most recent first.
pub fn PromptCache(s: *const pc.Store) status.PromptCache {
    var out = status.PromptCache{
        .states = @intCast(s.entries.items.len),
        .bytes = s.held + s.spare(),
        .hits = s.counts.hits,
        .misses = s.counts.misses,
        .kept = s.counts.kept,
        .evicted = s.counts.evicted,
        .refused = s.counts.refused,
        .failed = s.counts.failed,
        .budget_bytes = s.budget,
        .spare_bytes = s.spare(),
        .hold = s.hold,
        .entry_count = @intCast(s.entries.items.len),
    };
    var n: usize = 0; // the copy keeps the most recently used, sorted newest first by insertion
    for (s.entries.items) |e| {
        out.last_access_ms = @max(out.last_access_ms, e.seen.last_ms);
        var i = @min(n, out.entries.len);
        if (i == out.entries.len and out.entries[i - 1].last_access_ms >= e.seen.last_ms) continue;
        if (i == out.entries.len) i -= 1 else n += 1;
        while (i > 0 and out.entries[i - 1].last_access_ms < e.seen.last_ms) : (i -= 1) out.entries[i] = out.entries[i - 1];
        out.entries[i] = .{
            .id = e.seen.id,
            .at = e.at,
            .bytes = e.bytes,
            .shared = e.shared,
            .branched = e.branched,
            .borrowers = e.borrowers,
            .last_access_ms = e.seen.last_ms,
            .created_at_ms = e.seen.created_ms,
            .hits = e.seen.hits,
        };
    }
    return out;
}

/// The disk tier's copy (--learn-conversations), or null: no tier, nothing to show.
pub fn Disk(s: *pc.Store) ?status.DiskCache {
    if (s.disk) |r| {
        var out = status.DiskCache{
            .states = @intCast(r.states.items.len),
            .held_bytes = r.held(),
            .cap_bytes = r.cap,
            .daily_cap_bytes = r.daily,
            .queue_len = @intCast(r.queue.items.len),
            .read_rate_bps = r.read_rate,
            .read_ms_last = r.file_io.read_ms_last,
            .read_ms_ema = r.file_io.read_ms_ema,
            .write_ms_last = r.file_io.write_ms_last,
            .write_ms_ema = r.file_io.write_ms_ema,
            .read_ms_p50 = r.file_io.read_lat.percentile(0.5),
            .read_ms_p95 = r.file_io.read_lat.percentile(0.95),
            .write_ms_p50 = r.file_io.write_lat.percentile(0.5),
            .write_ms_p95 = r.file_io.write_lat.percentile(0.95),
            .last_read_ms = r.file_io.last_read_ms,
            .last_write_ms = r.file_io.last_write_ms,
            .written_bytes = r.counts.written,
            .read_bytes = r.counts.read,
            .reads = r.counts.reads,
            .writes = r.counts.writes,
            .skipped = r.counts.skipped,
            .failed = r.counts.failed,
            .evicted = r.counts.evicted,
            .verified = r.counts.verified,
            .differed = r.counts.differed,
            .failure_count = r.failures.count,
            .admission_retry_at = r.admission.retry_at,
        };
        if (r.failures.count > 0) {
            if (r.failures.records[@intCast((r.failures.count - 1) % 64)]) |f| out.last_error = @errorName(f.err);
        }
        if (s.imprint) |im| {
            out.imprint_states = @intCast(im.metas.items.len);
            out.imprint_bytes = im.total();
        }
        if (r.fetch) |f| {
            const c = f.snapshot();
            out.prefetch_made = c.counts.made;
            out.prefetch_used = c.counts.used;
            out.prefetch_freed = c.counts.freed;
            out.prefetch_failed = c.counts.failed;
            out.prefetch_cancelled = c.counts.cancelled;
            out.prefetch_timeouts = c.counts.timeouts;
            out.prefetch_held_reads = @intCast(c.held_reads);
            out.prefetch_held_bytes = c.held_bytes;
        }
        return out;
    }
    return null;
}

test {
    _ = @import("prompt_cache_status_test.zig");
}
