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
    inflight: []i32,
    slots: []Slot,
    demand: []bool,
    current: []usize,
    current_count: usize = 0,
    hand: usize = 0,
    owns_map: bool = true,

    pub fn init(a: std.mem.Allocator, keys: usize, capacity: usize) !Cache {
        if (keys == 0 or capacity == 0 or capacity > std.math.maxInt(i32)) return error.InvalidPoolCapacity;
        const map = try a.alloc(i32, keys);
        errdefer a.free(map);
        var out = try initMap(a, map, capacity);
        out.owns_map = true;
        return out;
    }

    /// Borrow a persistent shared GPU map; the buffer owner outlives the cache and all submitted consumers.
    pub fn initMap(a: std.mem.Allocator, map: []i32, capacity: usize) !Cache {
        if (map.len == 0 or capacity == 0 or capacity > std.math.maxInt(i32)) return error.InvalidPoolCapacity;
        const demand = try a.alloc(bool, map.len);
        errdefer a.free(demand);
        const inflight = try a.alloc(i32, map.len);
        errdefer a.free(inflight);
        const slots = try a.alloc(Slot, capacity);
        errdefer a.free(slots);
        const current = try a.alloc(usize, capacity);
        @memset(map, absent);
        @memset(inflight, absent);
        @memset(demand, false);
        @memset(slots, .{});
        return .{ .a = a, .map = map, .inflight = inflight, .slots = slots, .demand = demand, .current = current, .owns_map = false };
    }

    pub fn deinit(c: *Cache) void {
        for (c.slots) |s| std.debug.assert(!s.reading and !s.leased);
        if (c.owns_map) c.a.free(c.map);
        c.a.free(c.inflight);
        c.a.free(c.demand);
        c.a.free(c.slots);
        c.a.free(c.current);
    }

    pub fn resident(c: *Cache, key: usize) !?usize {
        const at = try c.peek(key) orelse return null;
        c.slots[at].referenced = true;
        return at;
    }

    /// Forecast queries do not turn an unused prediction into a demand hit.
    pub fn peek(c: *const Cache, key: usize) !?usize {
        if (key >= c.map.len) return error.InvalidExpert;
        const at = c.map[key];
        if (at < 0) return null;
        const s = &c.slots[@intCast(at)];
        std.debug.assert(s.key == key and s.ready and !s.reading);
        return @intCast(at);
    }

    /// Protect every hit before choosing any victim, including hits late in the incoming union.
    pub fn protect(c: *Cache, keys: []const usize) !void {
        if (c.current_count != 0) return error.DemandStillLeased;
        for (keys) |key| if (key >= c.map.len) return error.InvalidExpert;
        for (keys) |key| if (!c.demand[key]) {
            if (c.current_count == c.current.len) {
                c.release();
                return error.ActiveUnionExceedsPool;
            }
            c.demand[key] = true;
            c.current[c.current_count] = key;
            c.current_count += 1;
        };
        for (c.current[0..c.current_count]) |key| if (c.slotFor(key)) |s| {
            s.leased = true;
            s.referenced = true;
        };
    }

    fn slotFor(c: *Cache, key: usize) ?*Slot {
        const at = c.map[key];
        if (at >= 0) return &c.slots[@intCast(at)];
        if (c.pending(key)) |ticket_| return &c.slots[ticket_.slot];
        return null;
    }

    /// Drain consumers first; warm demand touches only its own keys, rather than scanning the whole pool.
    pub fn release(c: *Cache) void {
        for (c.current[0..c.current_count]) |key| {
            c.demand[key] = false;
            if (c.slotFor(key)) |slot| slot.leased = false;
        }
        c.current_count = 0;
    }

    pub fn pending(c: *const Cache, key: usize) ?Ticket {
        if (key >= c.inflight.len or c.inflight[key] < 0) return null;
        const i: usize = @intCast(c.inflight[key]);
        const s = c.slots[i];
        std.debug.assert(s.reading and s.key == key);
        return .{ .key = key, .slot = i, .generation = s.generation };
    }

    pub fn forecastRoom(c: *const Cache) bool {
        for (c.slots) |s| if (!s.reading and !s.leased and !s.referenced) return true;
        return false;
    }

    /// An in-flight forecast owns its slot. Demand joins that read instead of launching a second writer.
    pub fn reserve(c: *Cache, key: usize, forecast: bool) !?Ticket {
        if (key >= c.map.len) return error.InvalidExpert;
        if (!forecast and !c.demand[key]) return error.UnprotectedDemand;
        if (try c.peek(key) != null or c.pending(key) != null) return null;
        var visits: usize = 0;
        while (visits < c.slots.len * @as(usize, if (forecast) 1 else 2)) : (visits += 1) {
            const i = c.hand;
            c.hand = (i + 1) % c.slots.len;
            const s = &c.slots[i];
            if (s.reading or s.leased) continue;
            if (s.referenced) {
                if (!forecast) s.referenced = false;
                continue;
            }
            if (s.key) |old| c.map[old] = absent;
            s.* = .{ .key = key, .generation = s.generation +% 1, .reading = true, .leased = c.demand[key] };
            c.inflight[key] = @intCast(i);
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
        c.inflight[t.key] = absent;
        s.reading = false;
        s.ready = true;
        s.referenced = s.leased;
        c.map[t.key] = @intCast(t.slot);
    }

    pub fn abort(c: *Cache, t: Ticket) !void {
        const s = try c.ticket(t);
        c.inflight[t.key] = absent;
        s.reading = false;
        s.ready = false;
        s.key = null;
        s.leased = false;
        c.map[t.key] = absent;
    }
};
