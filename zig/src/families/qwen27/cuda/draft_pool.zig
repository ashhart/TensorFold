//! Detached fixed-class draft Context ownership; no CUDA allocation, copy or fence occurs in this policy.
const std = @import("std");
pub const max_slots: usize = 16;
pub const max_capacity: usize = 512;
pub const max_bytes: usize = 1536 * 1024 * 1024;

pub const Stats = struct {
    enabled: bool = false,
    failed: bool = false,
    driver_creations: u64 = 0,
    reuse_hits: u64 = 0,
    evictions: u64 = 0,
    discarded: u64 = 0,
    initialized: u64 = 0,
    retained_slots: usize = 0,
    retained_bytes: usize = 0,
    quarantined_slots: usize = 0,
    quarantined_bytes: usize = 0,
    fence_failures: u64 = 0,
};

pub fn enabled(raw: ?[]const u8, cache_enabled: bool) !bool {
    const text = raw orelse return false;
    if (std.mem.eql(u8, text, "0")) return false;
    if (!std.mem.eql(u8, text, "1")) return error.InvalidDraftReuseFlag;
    if (!cache_enabled) return error.DraftReuseRequiresExactCache;
    return true;
}

pub fn configured(cache_enabled: bool) !bool {
    const env = std.c.getenv("TF_PR608_EXACT_CACHE_DRAFT_REUSE");
    return enabled(if (env) |v| std.mem.span(v) else null, cache_enabled);
}

pub fn eligible(drafted: bool, drafter: bool, tokens: usize, capacity: usize, fresh: bool, cancelled: bool) bool {
    return drafted and drafter and tokens > 0 and tokens <= max_capacity and capacity <= max_capacity and
        tokens < capacity and fresh and !cancelled;
}

/// A completed request authorizes storage recycling only when every independent success gate holds.
pub fn completed(initialized: bool, terminal: bool, uncancelled: bool, settled: bool, cache_ok: bool, cache_failed: bool) bool {
    return initialized and terminal and uncancelled and settled and cache_ok and !cache_failed;
}

pub const Proof = struct {
    stream: *const anyopaque,
    seq: *const anyopaque,
    wrapper: *const anyopaque,
    arena: u64,
    tokens: usize,
    capacity: usize,
};

pub const Proofs = struct {
    items: [max_slots]?Proof = @splat(null),
    pub fn mark(p: *Proofs, proof: Proof) !void {
        if (proof.arena == 0 or proof.tokens == 0 or proof.tokens >= proof.capacity or proof.capacity > max_capacity)
            return error.InvalidDraftInitialization;
        for (p.items) |slot| if (slot) |old| {
            if (old.stream == proof.stream) return error.DuplicateDraftInitialization;
        };
        for (&p.items) |*slot| if (slot.* == null) {
            slot.* = proof;
            return;
        };
        return error.DraftInitializationLimit;
    }
    pub fn remove(p: *Proofs, stream: *const anyopaque) ?Proof {
        for (&p.items) |*slot| if (slot.*) |proof| {
            if (proof.stream != stream) continue;
            slot.* = null;
            return proof;
        };
        return null;
    }
};

/// Idle entries were fenced; quarantined entries must not be freed or taken until a new successful fence.
pub fn Pool(comptime T: type) type {
    return struct {
        const Self = @This();
        entries: [max_slots]*T = undefined,
        quarantine: [max_slots]*T = undefined,
        limit: usize,
        item_bytes: usize,
        stats: Stats,

        pub fn init(on: bool, slots: usize, item_bytes: usize) Self {
            return .{ .limit = @min(slots, max_slots), .item_bytes = item_bytes, .stats = .{ .enabled = on } };
        }
        pub fn active(p: *const Self) bool {
            return p.stats.enabled and !p.stats.failed;
        }
        pub fn hasRoom(p: *const Self, live: usize) bool {
            return live < p.limit and p.stats.retained_slots + p.stats.quarantined_slots < p.limit - live;
        }
        fn duplicate(p: *const Self, ptr: *T) bool {
            for (p.entries[0..p.stats.retained_slots]) |old| if (old == ptr) return true;
            for (p.quarantine[0..p.stats.quarantined_slots]) |old| if (old == ptr) return true;
            return false;
        }
        pub fn take(p: *Self) ?*T {
            if (!p.active() or p.stats.retained_slots == 0) return null;
            p.stats.retained_slots -= 1;
            p.stats.retained_bytes -= p.item_bytes;
            p.stats.reuse_hits += 1;
            return p.entries[p.stats.retained_slots];
        }
        pub fn evict(p: *Self) ?*T {
            if (p.stats.retained_slots == 0) return null;
            p.stats.retained_slots -= 1;
            p.stats.retained_bytes -= p.item_bytes;
            p.stats.evictions += 1;
            return p.entries[p.stats.retained_slots];
        }
        pub fn retain(p: *Self, ptr: *T, live: usize) bool {
            if (!p.active() or !p.hasRoom(live) or p.item_bytes > max_bytes or
                p.stats.retained_bytes > max_bytes - p.item_bytes) return false;
            if (p.duplicate(ptr)) @panic("duplicate draft pool owner");
            p.entries[p.stats.retained_slots] = ptr;
            p.stats.retained_slots += 1;
            p.stats.retained_bytes += p.item_bytes;
            return true;
        }
        pub fn fail(p: *Self) void {
            if (p.stats.enabled) p.stats.failed = true;
        }
        pub fn park(p: *Self, ptr: *T, live: usize) void {
            p.fail();
            if (!p.hasRoom(live) or p.duplicate(ptr)) @panic("draft quarantine ownership limit");
            p.quarantine[p.stats.quarantined_slots] = ptr;
            p.stats.quarantined_slots += 1;
            p.stats.quarantined_bytes += p.item_bytes;
            p.stats.fence_failures += 1;
        }
        pub fn reap(p: *Self, fenced: bool) ?*T {
            if (!fenced or p.stats.quarantined_slots == 0) return null;
            p.stats.quarantined_slots -= 1;
            p.stats.quarantined_bytes -= p.item_bytes;
            return p.quarantine[p.stats.quarantined_slots];
        }
    };
}

test {
    _ = @import("draft_pool_test.zig");
}
