//! Default-off full-prompt snapshots into independent normal Qwen allocations.
const std = @import("std");
const lanes = @import("lanes");
const st = @import("state.zig");
const c = @import("shape.zig");
const dr = @import("draft.zig");
const dl = @import("draft_load.zig");
const kern = @import("kernels.zig");
pub const policy = @import("exact_cache_policy.zig");
pub const logits_policy = @import("exact_logits_policy.zig");
const roundedBytes = struct {
    fn bytes(n: usize) usize {
        return std.mem.alignForward(usize, n, 256);
    }
}.bytes;

pub const Snapshot = struct {
    gpa: std.mem.Allocator,
    arena: st.Arena,
    prompt: []u32,
    chunks: []u32,
    drafted: bool,
    pos: usize,
    cur: u1 = 0,
    len: usize = 0,
    end: usize = 0,
    rec: [64]u64 = @splat(0),
    conv: [64]u64 = @splat(0),
    keys: [64]u64 = @splat(0),
    values: [64]u64 = @splat(0),
    head_in: u64,
    head_xs: u64,
    draft_keys: [dl.layers]u64 = @splat(0),
    draft_values: [dl.layers]u64 = @splat(0),
    device_bytes: usize,
    first_logits: u64 = 0,
    logits_valid: bool = false,

    pub fn bytes(g: c.Geometry, n: usize, drafted: bool) usize {
        return bytesWithLogits(g, n, drafted, false);
    }

    pub fn bytesWithLogits(g: c.Geometry, n: usize, drafted: bool, logits: bool) usize {
        return st.Seq.bytes(g, n, false) + (if (drafted) 2 * dl.layers * roundedBytes(dl.kv_heads * n * dl.head_dim * 2) else 0) +
            (if (logits) logits_policy.row_bytes else 0);
    }

    pub fn init(gpa: std.mem.Allocator, ops: kern.Ops, g: c.Geometry, s: *lanes.Stream) !*Snapshot {
        return initWithLogits(gpa, ops, g, s, false);
    }

    pub fn initWithLogits(gpa: std.mem.Allocator, ops: kern.Ops, g: c.Geometry, s: *lanes.Stream, logits: bool) !*Snapshot {
        if (g.layers != 64) return error.UnsupportedExactCacheGeometry;
        const n = s.prompt().len;
        const total = bytesWithLogits(g, n, s.drafts, logits);
        if (total > policy.max_bytes) return error.ExactCacheBudget;
        const out = try gpa.create(Snapshot);
        errdefer gpa.destroy(out);
        const prompt = try gpa.dupe(u32, s.prompt());
        errdefer gpa.free(prompt);
        const chunks = try gpa.dupe(u32, s.chunks);
        errdefer gpa.free(chunks);
        out.* = .{ .gpa = gpa, .arena = try st.Arena.init(ops.k.d, total), .prompt = prompt, .chunks = chunks, .drafted = s.drafts, .pos = n, .head_in = 0, .head_xs = 0, .device_bytes = total };
        errdefer out.arena.deinit();
        out.head_in = try out.arena.take(g.hidden * 2);
        out.head_xs = try out.arena.take(g.hidden / 64 * 4);
        for (0..g.layers) |i| {
            if (c.linear(i)) {
                out.rec[i] = try out.arena.take(g.deltaBytes());
                out.conv[i] = try out.arena.take(g.convBytes());
            } else {
                out.keys[i] = try out.arena.take(n * g.kvInner() * 2);
                out.values[i] = try out.arena.take(n * g.kvInner() * 2);
            }
        }
        if (s.drafts) for (0..dl.layers) |i| {
            out.draft_keys[i] = try out.arena.take(dl.kv_heads * n * dl.head_dim * 2);
            out.draft_values[i] = try out.arena.take(dl.kv_heads * n * dl.head_dim * 2);
        };
        if (logits) out.first_logits = try out.arena.take(logits_policy.row_bytes);
        return out;
    }

    fn target(snap: *const Snapshot, ops: kern.Ops, g: c.Geometry, seq: *st.Seq, restoring: bool) !void {
        if (seq.capacity < snap.pos) return error.ExactCacheCapacity;
        const pair = struct {
            fn run(o: kern.Ops, saved: u64, live: u64, n: usize, restored: bool) !void {
                try o.copy(if (restored) live else saved, if (restored) saved else live, n);
            }
        }.run;
        try pair(ops, snap.head_in, seq.head_in, g.hidden * 2, restoring);
        try pair(ops, snap.head_xs, seq.head_xs, g.hidden / 64 * 4, restoring);
        for (0..g.layers) |i| if (c.linear(i)) {
            try pair(ops, snap.rec[i], seq.rec[i], g.deltaBytes(), restoring);
            try pair(ops, snap.conv[i], seq.conv[i], g.convBytes(), restoring);
        } else {
            try pair(ops, snap.keys[i], seq.keys[i], snap.pos * g.kvInner() * 2, restoring);
            try pair(ops, snap.values[i], seq.values[i], snap.pos * g.kvInner() * 2, restoring);
        };
    }

    fn context(snap: *const Snapshot, ops: kern.Ops, ctx: *dr.Context, restoring: bool) !void {
        const n = snap.len * dl.head_dim * 2;
        const live_stride = n;
        for (0..dl.layers) |i| for (0..dl.kv_heads) |h| {
            const k = ctx.keys[i][snap.cur] + h * live_stride;
            const v = ctx.values[i][snap.cur] + h * live_stride;
            const sk = snap.draft_keys[i] + h * n;
            const sv = snap.draft_values[i] + h * n;
            try ops.copy(if (restoring) k else sk, if (restoring) sk else k, n);
            try ops.copy(if (restoring) v else sv, if (restoring) sv else v, n);
        };
    }

    pub fn save(snap: *Snapshot, ops: kern.Ops, g: c.Geometry, seq: *st.Seq, ctx: ?*dr.Context) !void {
        if (seq.kv8 or seq.media != null or seq.rope_delta != 0) return error.UnsupportedExactCacheState;
        if (seq.pos != snap.pos or (snap.drafted != (ctx != null))) return error.InvalidExactCacheState;
        if (ctx) |x| {
            if (x.len != snap.pos or x.end != snap.pos) return error.InvalidExactCacheContext;
            snap.cur = x.cur;
            snap.len = x.len;
            snap.end = x.end;
        }
        try snap.target(ops, g, seq, false);
        if (ctx) |x| try snap.context(ops, x, false);
    }

    pub fn restore(snap: *const Snapshot, ops: kern.Ops, g: c.Geometry, seq: *st.Seq, ctx: ?*dr.Context) !void {
        if (seq.kv8 or seq.media != null or seq.rope_delta != 0) return error.UnsupportedExactCacheState;
        if (snap.drafted != (ctx != null)) return error.InvalidExactCacheState;
        try snap.target(ops, g, seq, true);
        if (ctx) |x| {
            try snap.context(ops, x, true);
            x.cur = snap.cur;
            x.len = snap.len;
            x.end = snap.end;
        }
        seq.pos = snap.pos;
    }

    pub fn matches(snap: *const Snapshot, s: *lanes.Stream) bool {
        return snap.drafted == s.drafts and std.mem.eql(u32, snap.prompt, s.prompt()) and std.mem.eql(u32, snap.chunks, s.chunks);
    }

    pub fn deinit(snap: *Snapshot) void {
        snap.arena.deinit();
        snap.gpa.free(snap.prompt);
        snap.gpa.free(snap.chunks);
        snap.gpa.destroy(snap);
    }
};

pub const Stats = struct {
    lookups: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    captures: u64 = 0,
    promotions: u64 = 0,
    purges: u64 = 0,
    evictions: u64 = 0,
    refusals: u64 = 0,
    active_bytes: usize = 0,
    pending_bytes: usize = 0,
    first_logits_captures: u64 = 0,
    first_logits_hits: u64 = 0,
    first_logits_fallbacks: u64 = 0,
};

pub const Cache = struct {
    gpa: std.mem.Allocator,
    enabled: bool,
    logits_enabled: bool = false,
    failed: bool = false,
    active: ?*Snapshot = null,
    pending: ?*Snapshot = null,
    owner: ?*lanes.Stream = null,
    stats: Stats = .{},
    post_first: @import("exact_post_first.zig").Control = .{},

    pub fn init(gpa: std.mem.Allocator, on: bool) Cache {
        return .{ .gpa = gpa, .enabled = on };
    }

    pub fn eligible(cache: *const Cache, s: *lanes.Stream) bool {
        const fresh = s.reuse.saved == null and s.reuse.at == 0 and s.reuse.marks.len == 0 and s.cached == 0;
        return cache.enabled and !cache.failed and @import("exact_eligible.zig").text(false, s) and policy.eligible(s.prompt().len, s.chunks.len, fresh, s.isCancelled());
    }

    /// Pure acquisition hint; successful lookup, not this hint, proves restoration.
    pub fn canRestore(cache: *const Cache, s: *lanes.Stream, drafted: bool) bool {
        if (!cache.eligible(s) or s.drafts != drafted) return false;
        const snap = cache.active orelse return false;
        return snap.matches(s);
    }

    fn fence(cache: *Cache, ops: kern.Ops) !void {
        ops.s.synchronize() catch |err| {
            cache.failed = true;
            return err;
        };
    }

    fn freeActive(cache: *Cache, ops: kern.Ops) !void {
        if (cache.active) |snap| {
            try cache.fence(ops);
            snap.deinit();
            cache.active = null;
            cache.stats.active_bytes = 0;
        }
    }

    fn freePending(cache: *Cache, ops: kern.Ops) !void {
        if (cache.pending) |snap| {
            try cache.fence(ops);
            snap.deinit();
            cache.pending = null;
            cache.owner = null;
            cache.stats.pending_bytes = 0;
        }
    }

    pub fn lookup(cache: *Cache, s: *lanes.Stream, seq: *st.Seq, ctx: ?*dr.Context, ops: kern.Ops, g: c.Geometry) !bool {
        if (seq.kv8 or seq.media != null or seq.rope_delta != 0 or !cache.eligible(s) or s.drafts != (ctx != null)) return false;
        cache.stats.lookups += 1;
        if (cache.active) |snap| if (snap.matches(s)) {
            snap.restore(ops, g, seq, ctx) catch |err| {
                cache.failed = true;
                return err;
            };
            s.cached = @intCast(snap.pos);
            cache.stats.hits += 1;
            return true;
        };
        cache.stats.misses += 1;
        return false;
    }

    /// Serving may reserve the exact same snapshot until the first token has been handed off.
    pub fn captureBeforeFirst(cache: *Cache, s: *lanes.Stream, seq: *st.Seq, ctx: ?*dr.Context, ops: kern.Ops, g: c.Geometry) !void {
        if (!cache.post_first.enabled) return cache.capture(s, seq, ctx, ops, g);
        @import("exact_post_first.zig").reserve(cache, s, seq, ctx, Snapshot.bytesWithLogits(g, s.prompt().len, s.drafts, cache.logits_enabled));
    }

    /// A successful save remains pending until the original clean-completion promotion fence.
    pub fn finishAfterFirst(cache: *Cache, s: *lanes.Stream, seq: *st.Seq, ctx: ?*dr.Context, ops: kern.Ops, g: c.Geometry, raw: u64) !void {
        const reserved = cache.post_first.take(s) orelse return;
        errdefer cache.post_first.failures += 1;
        if (s.isCancelled()) return error.Cancelled;
        if (s.finished or cache.failed) {
            cache.post_first.skips += 1;
            return;
        }
        if (reserved.seq != seq or reserved.ctx != ctx or reserved.tokens != seq.pos or
            reserved.tokens != s.prompt().len or s.emitted().len != 1 or s.pending == null)
            return error.InvalidPostFirstCapture;
        if (ctx) |dc| if (dc.cur != reserved.cur or dc.len != reserved.tokens or dc.end != reserved.tokens) {
            return error.InvalidPostFirstContext;
        };
        try cache.capture(s, seq, ctx, ops, g);
        if (cache.owner == s and cache.pending != null) {
            _ = try cache.captureFirstLogits(s, seq, ops, raw);
            cache.post_first.saves += 1;
        } else cache.post_first.skips += 1;
        if (s.isCancelled()) return error.Cancelled;
    }

    pub fn capture(cache: *Cache, s: *lanes.Stream, seq: *st.Seq, ctx: ?*dr.Context, ops: kern.Ops, g: c.Geometry) !void {
        if (seq.kv8 or seq.media != null or seq.rope_delta != 0 or !cache.eligible(s) or s.drafts != (ctx != null) or (cache.active != null and cache.active.?.matches(s))) return;
        const size = Snapshot.bytesWithLogits(g, s.prompt().len, s.drafts, cache.logits_enabled);
        switch (policy.admission(cache.stats.active_bytes, cache.stats.pending_bytes, size)) {
            .refuse, .busy => {
                cache.stats.refusals += 1;
                return;
            },
            .evict => {
                try cache.freeActive(ops);
                cache.stats.evictions += 1;
            },
            .retain => {},
        }
        const snap = Snapshot.initWithLogits(cache.gpa, ops, g, s, cache.logits_enabled) catch {
            cache.stats.refusals += 1;
            return;
        };
        cache.pending = snap;
        cache.owner = s;
        cache.stats.pending_bytes = snap.device_bytes;
        snap.save(ops, g, seq, ctx) catch |err| {
            cache.failed = true;
            return err;
        };
        cache.stats.captures += 1;
    }

    /// Copy only from the current matching immutable entry; eviction means an ordinary head fallback.
    pub fn restoreFirstLogits(cache: *Cache, s: *lanes.Stream, seq: *st.Seq, ops: kern.Ops, raw: u64, proven: bool) !bool {
        if (!cache.logits_enabled or seq.kv8 or !@import("exact_eligible.zig").text(false, s)) return false;
        const snap = cache.active orelse return false;
        if (!logits_policy.canUse(cache.enabled and !cache.failed, proven, snap.matches(s), snap.logits_valid, seq.pos == snap.pos and s.prompt().len == snap.pos and s.cached == snap.pos, s.isCancelled())) return false;
        if (snap.first_logits == 0) return error.InvalidFirstLogitsStorage;
        ops.copy(raw, snap.first_logits, logits_policy.row_bytes) catch |err| {
            cache.failed = true;
            return err;
        };
        cache.stats.first_logits_hits += 1;
        return true;
    }

    /// Preserve the cold owner's raw row before any draw; the normal promotion fence completes this copy.
    pub fn captureFirstLogits(cache: *Cache, s: *lanes.Stream, seq: *st.Seq, ops: kern.Ops, raw: u64) !bool {
        if (seq.kv8 or !@import("exact_eligible.zig").text(false, s) or !cache.logits_enabled or cache.failed or cache.owner != s) return false;
        const snap = cache.pending orelse return false;
        if (snap.logits_valid) return false;
        if (!snap.matches(s) or seq.pos != snap.pos or snap.first_logits == 0) return error.InvalidFirstLogitsCapture;
        logits_policy.capture(&snap.logits_valid, ops, snap.first_logits, raw) catch |err| {
            cache.failed = true;
            snap.logits_valid = false;
            return err;
        };
        cache.stats.first_logits_captures += 1;
        return true;
    }

    pub fn release(cache: *Cache, s: *lanes.Stream, settled: bool, ops: kern.Ops) !void {
        cache.post_first.forget(s);
        if (!settled) {
            cache.failed = true;
            if (cache.pending != null) {
                try cache.freePending(ops);
                cache.stats.purges += 1;
            }
            return;
        }
        if (cache.owner != s) return;
        const normal = s.reason == .stop or s.reason == .length;
        const logits_ready = if (cache.pending) |snap| logits_policy.promotable(cache.logits_enabled, snap.logits_valid) else false;
        if (cache.failed or !logits_ready or !policy.clean(s.finished, normal, s.isCancelled(), settled)) {
            try cache.freePending(ops);
            cache.stats.purges += 1;
            return;
        }
        try cache.fence(ops);
        try cache.freeActive(ops);
        cache.active = cache.pending;
        cache.pending = null;
        cache.owner = null;
        cache.stats.active_bytes = cache.stats.pending_bytes;
        cache.stats.pending_bytes = 0;
        cache.stats.promotions += 1;
    }

    pub fn abandon(cache: *Cache, s: *lanes.Stream, ops: kern.Ops) !void {
        cache.post_first.forget(s);
        if (cache.owner == s) {
            try cache.freePending(ops);
            cache.stats.purges += 1;
        }
    }

    pub fn deinit(cache: *Cache, ops: kern.Ops) !void {
        cache.post_first.reserved = null;
        if (cache.post_first.enabled) std.debug.print("PR608_POST_FIRST_CAPTURE {f}\n", .{std.json.fmt(.{ .reservations = cache.post_first.reservations, .saves = cache.post_first.saves, .skips = cache.post_first.skips, .failures = cache.post_first.failures }, .{})});
        if (cache.enabled) std.debug.print("PR608_EXACT_CACHE {{\"hits\":{d},\"misses\":{d},\"captures\":{d},\"promotions\":{d},\"purges\":{d},\"evictions\":{d},\"refusals\":{d},\"active_bytes\":{d},\"pending_bytes\":{d},\"failed\":{}}}\n", .{ cache.stats.hits, cache.stats.misses, cache.stats.captures, cache.stats.promotions, cache.stats.purges, cache.stats.evictions, cache.stats.refusals, cache.stats.active_bytes, cache.stats.pending_bytes, cache.failed });
        if (cache.logits_enabled) std.debug.print("PR608_EXACT_CACHE_LOGITS {f}\n", .{std.json.fmt(.{
            .captures = cache.stats.first_logits_captures,
            .hits = cache.stats.first_logits_hits,
            .fallbacks = cache.stats.first_logits_fallbacks,
            .failed = cache.failed,
        }, .{})});
        try cache.freePending(ops);
        try cache.freeActive(ops);
    }
};
