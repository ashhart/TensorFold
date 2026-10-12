//! Fixed ownership storage for the private exact-capacity short Seq experiment; no allocator or device work.
const std = @import("std");
pub const max_slots = 16;
pub const max_capacity = 512;
pub const max_bytes = 3 * 1024 * 1024 * 1024;

pub const Stats = struct {
    enabled: bool = false,
    failed: bool = false,
    create_attempts: u64 = 0,
    driver_creations: u64 = 0,
    reuse_hits: u64 = 0,
    evictions: u64 = 0,
    discarded: u64 = 0,
    retained_bytes: usize = 0,
    retained_slots: usize = 0,
};

pub fn parse(raw: ?[]const u8) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.InvalidSeqReuse;
}

pub fn configured(cache_enabled: bool) !bool {
    const raw = std.c.getenv("TF_PR608_EXACT_CACHE_SEQ_REUSE");
    return compatible(if (raw) |value| std.mem.span(value) else null, cache_enabled);
}

pub fn compatible(raw: ?[]const u8, cache_enabled: bool) !bool {
    const on = try parse(raw);
    if (on and !cache_enabled) return error.SeqReuseNeedsExactCache;
    return on;
}

/// Normal completion requires proven full-N restoration, never a cached count alone.
pub fn completed(s: anytype, settled: bool, restored: bool, cache_ok: bool, cache_failed: bool) bool {
    return cache_ok and !cache_failed and restored and settled and s.finished and (s.reason == .stop or s.reason == .length) and
        !s.isCancelled() and s.cached == s.prompt_len and !s.reuse_failed and s.reuse.saved == null and
        s.reuse.at == 0 and s.reuse.marks.len == 0 and s.reuse.hook == null;
}

/// A taken item is caller-owned immediately; a retained item is pool-owned until take/pop.
pub fn Pool(comptime T: type) type {
    return struct {
        const Self = @This();
        const Entry = struct { ptr: *T, capacity: usize, bytes: usize };
        entries: [max_slots]Entry = undefined,
        limit: usize = 0,
        stats: Stats = .{},

        pub fn init(enabled: bool, slots: usize) Self {
            return .{ .limit = @min(slots, max_slots), .stats = .{ .enabled = enabled } };
        }

        pub fn active(p: *const Self) bool {
            return p.stats.enabled and !p.stats.failed;
        }

        fn remove(p: *Self, index: usize) *T {
            const entry = p.entries[index];
            p.stats.retained_slots -= 1;
            p.stats.retained_bytes -= entry.bytes;
            for (index..p.stats.retained_slots) |i| p.entries[i] = p.entries[i + 1];
            return entry.ptr;
        }

        pub fn take(p: *Self, capacity: usize) ?*T {
            if (!p.active() or capacity > max_capacity) return null;
            for (p.entries[0..p.stats.retained_slots], 0..) |entry, i| {
                if (entry.capacity != capacity) continue;
                p.stats.reuse_hits += 1;
                return p.remove(i);
            }
            return null;
        }

        /// Eviction detaches ownership; the caller frees exactly this item.
        pub fn evict(p: *Self) ?*T {
            if (p.stats.retained_slots == 0) return null;
            p.stats.evictions += 1;
            return p.remove(0);
        }

        /// An exact-capacity miss drains all retained bytes, even when a count slot remains available.
        pub fn beforeCreate(p: *Self, context: anytype, comptime free: anytype) void {
            while (p.evict()) |ptr| free(context, ptr);
        }

        pub fn hasRoom(p: *const Self, active_count: usize) bool {
            return active_count < p.limit and p.stats.retained_slots < p.limit - active_count;
        }

        pub fn canRetain(p: *const Self, active_count: usize, capacity: usize, bytes: usize) bool {
            return p.active() and capacity <= max_capacity and bytes <= max_bytes and
                p.stats.retained_bytes <= max_bytes - bytes and p.hasRoom(active_count);
        }

        pub fn retain(p: *Self, ptr: *T, active_count: usize, capacity: usize, bytes: usize) bool {
            if (!p.active()) return false;
            for (p.entries[0..p.stats.retained_slots]) |entry|
                if (entry.ptr == ptr) @panic("duplicate Seq pool ownership");
            if (!p.canRetain(active_count, capacity, bytes)) return false;
            p.entries[p.stats.retained_slots] = .{ .ptr = ptr, .capacity = capacity, .bytes = bytes };
            p.stats.retained_slots += 1;
            p.stats.retained_bytes += bytes;
            return true;
        }

        /// Failure disables future take/retain; the caller must evict and free outstanding ownership.
        pub fn fail(p: *Self) void {
            if (p.stats.enabled) p.stats.failed = true;
        }
    };
}

test {
    _ = @import("seq_pool_test.zig");
}
