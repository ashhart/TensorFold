//! Persistent expert addressing with CLOCK eviction and explicit read/consumer ownership.
const std = @import("std");
pub const absent: i32 = -1;
pub const Ticket = struct { key: usize, slot: usize, generation: u64 };
pub const Slot = struct {
    key: ?usize = null,
    generation: u64 = 0,
    reading: bool = false,
    ready: bool = false,
    referenced: bool = false,
    leased: bool = false,
};

pub const Cache = struct {
    a: std.mem.Allocator,
    map: []i32,
    slots: []Slot,
    demand: []bool,
    hand: usize = 0,

    pub fn init(a: std.mem.Allocator, keys: usize, capacity: usize) !Cache {
        if (keys == 0 or capacity == 0 or capacity > std.math.maxInt(i32)) return error.InvalidPoolCapacity;
        const map = try a.alloc(i32, keys);
        errdefer a.free(map);
        const demand = try a.alloc(bool, keys);
        errdefer a.free(demand);
        const slots = try a.alloc(Slot, capacity);
        @memset(map, absent);
        @memset(demand, false);
        @memset(slots, .{});
        return .{ .a = a, .map = map, .slots = slots, .demand = demand };
    }

    pub fn deinit(c: *Cache) void {
        for (c.slots) |s| std.debug.assert(!s.reading and !s.leased);
        c.a.free(c.map);
        c.a.free(c.demand);
        c.a.free(c.slots);
    }

    pub fn resident(c: *Cache, key: usize) !?usize {
        if (key >= c.map.len) return error.InvalidExpert;
        const at = c.map[key];
        if (at < 0) return null;
        const s = &c.slots[@intCast(at)];
        std.debug.assert(s.key == key and s.ready and !s.reading);
        s.referenced = true;
        return @intCast(at);
    }

    /// Protect every hit before choosing any victim, including hits late in the incoming union.
    pub fn protect(c: *Cache, keys: []const usize) !void {
        for (c.demand) |held| if (held) return error.DemandStillLeased;
        for (keys) |key| if (key >= c.map.len) return error.InvalidExpert;
        @memset(c.demand, false);
        var unique: usize = 0;
        for (keys) |key| if (!c.demand[key]) {
            c.demand[key] = true;
            unique += 1;
        };
        if (unique > c.slots.len) {
            @memset(c.demand, false);
            return error.ActiveUnionExceedsPool;
        }
        for (c.slots) |*s| if (s.key) |key| if (c.demand[key]) {
            s.leased = true;
            s.referenced = true;
        };
    }

    /// Call only after the GPU consumers of this demand have completed, including an error/cancel drain.
    pub fn release(c: *Cache) void {
        @memset(c.demand, false);
        for (c.slots) |*s| s.leased = false;
    }

    pub fn pending(c: *const Cache, key: usize) ?Ticket {
        for (c.slots, 0..) |s, i| if (s.reading and s.key == key)
            return .{ .key = key, .slot = i, .generation = s.generation };
        return null;
    }

    /// An in-flight forecast owns its slot. Demand joins that read instead of launching a second writer.
    pub fn reserve(c: *Cache, key: usize, forecast: bool) !?Ticket {
        if (key >= c.map.len) return error.InvalidExpert;
        if (!forecast and !c.demand[key]) return error.UnprotectedDemand;
        if (try c.resident(key) != null or c.pending(key) != null) return null;
        var visits: usize = 0;
        while (visits < c.slots.len * 2) : (visits += 1) {
            const i = c.hand;
            c.hand = (i + 1) % c.slots.len;
            const s = &c.slots[i];
            if (s.reading or s.leased) continue;
            if (s.referenced) {
                s.referenced = false;
                continue;
            }
            if (s.key) |old| c.map[old] = absent;
            s.* = .{ .key = key, .generation = s.generation +% 1, .reading = true, .leased = c.demand[key] };
            return .{ .key = key, .slot = i, .generation = s.generation };
        }
        return error.NoEvictableSlot;
    }

    fn ticket(c: *Cache, t: Ticket) !*Slot {
        if (t.slot >= c.slots.len) return error.StaleRead;
        const s = &c.slots[t.slot];
        if (s.key != t.key or s.generation != t.generation or !s.reading) return error.StaleRead;
        return s;
    }

    /// Publication follows every component read and validation; incomplete data is never GPU-visible.
    pub fn publish(c: *Cache, t: Ticket) !void {
        const s = try c.ticket(t);
        s.reading = false;
        s.ready = true;
        s.referenced = true;
        c.map[t.key] = @intCast(t.slot);
    }

    pub fn abort(c: *Cache, t: Ticket) !void {
        const s = try c.ticket(t);
        s.reading = false;
        s.ready = false;
        s.key = null;
        s.leased = false;
        c.map[t.key] = absent;
    }
};
