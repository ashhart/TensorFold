const std = @import("std");

pub const Geometry = struct {
    lanes: u32,
    layers: u32,
    blocks: u32,
    slots: u32,
    row_bytes: usize,
    ratio: u32 = 4,
    window_rows: usize,
    work_words: usize,

    pub fn validate(geometry: Geometry) !void {
        if (geometry.lanes == 0 or geometry.layers == 0 or geometry.blocks == 0 or geometry.slots == 0 or geometry.slots > std.math.maxInt(i32) or geometry.row_bytes == 0 or geometry.row_bytes % 2 != 0 or geometry.ratio == 0 or geometry.window_rows == 0) return error.SharedKvGeometry;
    }

    pub fn blockBytes(geometry: Geometry) !usize {
        try geometry.validate();
        return std.math.mul(usize, try std.math.mul(usize, geometry.row_bytes, geometry.ratio), 2);
    }

    pub fn accounting(geometry: Geometry) !Accounting {
        const block_bytes = try geometry.blockBytes();
        const payload = try std.math.mul(usize, geometry.slots, block_bytes);
        const owner_bytes = try std.math.mul(usize, geometry.slots, @sizeOf(Owner));
        const stamps = try std.math.mul(usize, geometry.slots, @sizeOf(u64));
        const inverse_words = try std.math.mul(usize, geometry.layers, geometry.blocks);
        const inverse = try std.math.mul(usize, inverse_words, @sizeOf(i32));
        const windows = try std.math.mul(usize, try std.math.mul(usize, geometry.layers, geometry.window_rows), try std.math.mul(usize, geometry.row_bytes, 2));
        const controls = try std.math.mul(usize, geometry.layers, 8 * @sizeOf(u32));
        const work = try std.math.mul(usize, geometry.work_words, @sizeOf(u32));
        const shared = try std.math.add(usize, payload, try std.math.add(usize, owner_bytes, stamps));
        const sequence = try std.math.add(usize, windows, try std.math.add(usize, try std.math.mul(usize, inverse, 2), try std.math.add(usize, controls, work)));
        return .{ .payload = payload, .owners = owner_bytes, .stamps = stamps, .inverse = inverse, .windows = windows, .controls = controls, .work = work, .shared = shared, .sequence = sequence, .rank = try std.math.add(usize, shared, try std.math.mul(usize, sequence, geometry.lanes)) };
    }

    pub fn duplicatedLegacyBytes(geometry: Geometry) !usize {
        const payload = try std.math.mul(usize, geometry.slots, try geometry.blockBytes());
        const metadata = try std.math.mul(usize, geometry.slots, 2 * @sizeOf(u32));
        return std.math.mul(usize, try std.math.add(usize, payload, metadata), geometry.lanes - 1);
    }
};

pub const Accounting = struct {
    payload: usize,
    owners: usize,
    stamps: usize,
    inverse: usize,
    windows: usize,
    controls: usize,
    work: usize,
    shared: usize,
    sequence: usize,
    rank: usize,
};

pub const Sequence = struct { arena: u64, lane: u32, generation: u64 };
pub const Key = struct { layer: u32, block: u32 };
pub const Tag = struct { sequence: Sequence, key: Key };
pub const Owner = extern struct {
    generation: u64 = 0,
    lane: u32 = 0,
    layer: u32 = 0,
    block: u32 = 0,
    valid: u32 = 0,
};

pub const Lease = struct { arena: u64, serial: u64, sequence: Sequence };
pub const Assignment = struct { key: Key, slot: u32, hit: bool, written: bool = false };
pub const Phase = enum { acquired, planned, submitted, ready, poisoned };
const Active = struct { lease: Lease, phase: Phase = .acquired, count: usize = 0 };
const Lane = struct { generation: u64 = 0, live: bool = false, failed: bool = false };

pub const OwnedEntry = struct { key: Key, stamp: u64, source_slot: u32 };

pub const OwnedBlocks = struct {
    version: u32 = 2,
    source: Sequence,
    geometry: Geometry,
    clock: u64,
    entries: []OwnedEntry,
    payload: []u8,

    pub fn maxAge(image: *const OwnedBlocks) !u64 {
        var maximum: u64 = 0;
        for (image.entries) |entry| {
            if (entry.stamp == 0 or entry.stamp > image.clock) return error.SharedKvSnapshotEntry;
            maximum = @max(maximum, image.clock - entry.stamp);
        }
        return maximum;
    }

    pub fn deinit(image: *OwnedBlocks, allocator: std.mem.Allocator) void {
        allocator.free(image.payload);
        allocator.free(image.entries);
        image.* = undefined;
    }

    pub fn validate(image: *const OwnedBlocks, destination: Geometry) !void {
        if (image.version != 2) return error.SharedKvSnapshotVersion;
        try image.geometry.validate();
        try destination.validate();
        if (image.source.arena == 0 or image.source.generation == 0 or image.source.lane >= image.geometry.lanes or image.geometry.layers != destination.layers or image.geometry.blocks != destination.blocks or image.geometry.row_bytes != destination.row_bytes or image.geometry.ratio != destination.ratio) return error.SharedKvSnapshotGeometry;
        if (image.entries.len > image.geometry.slots or image.entries.len > destination.slots) return error.SharedKvCapacity;
        if (image.payload.len != try std.math.mul(usize, image.entries.len, try destination.blockBytes())) return error.SharedKvSnapshotLength;
        for (image.entries, 0..) |entry, index| {
            if (entry.key.layer >= destination.layers or entry.key.block >= destination.blocks or entry.source_slot >= image.geometry.slots or entry.stamp == 0 or entry.stamp > image.clock) return error.SharedKvSnapshotEntry;
            for (image.entries[0..index]) |previous| if (std.meta.eql(previous.key, entry.key) or previous.source_slot == entry.source_slot) return error.SharedKvSnapshotDuplicate;
        }
    }
};

var identities: std.atomic.Value(u64) = .init(0);

pub fn identity() !u64 {
    var previous = identities.load(.monotonic);
    while (true) {
        const next = try std.math.add(u64, previous, 1);
        previous = identities.cmpxchgWeak(previous, next, .monotonic, .monotonic) orelse return next;
    }
}

pub const Arena = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    id: u64,
    clock: u64 = 0,
    serial: u64 = 0,
    lanes: []Lane,
    owners: []Owner,
    stamps: []u64,
    inverse: []i32,
    payload: []u8,
    assignments: []Assignment,
    pinned: []bool,
    active: ?Active = null,

    pub fn init(allocator: std.mem.Allocator, geometry: Geometry) !Arena {
        const bytes = try geometry.accounting();
        const inverse_count = try std.math.mul(usize, bytes.inverse / @sizeOf(i32), geometry.lanes);
        const lanes = try allocator.alloc(Lane, geometry.lanes);
        errdefer allocator.free(lanes);
        const owners = try allocator.alloc(Owner, geometry.slots);
        errdefer allocator.free(owners);
        const stamps = try allocator.alloc(u64, geometry.slots);
        errdefer allocator.free(stamps);
        const inverse = try allocator.alloc(i32, inverse_count);
        errdefer allocator.free(inverse);
        const payload = try allocator.alloc(u8, bytes.payload);
        errdefer allocator.free(payload);
        const assignments = try allocator.alloc(Assignment, geometry.slots);
        errdefer allocator.free(assignments);
        const pinned = try allocator.alloc(bool, geometry.slots);
        errdefer allocator.free(pinned);
        @memset(lanes, .{});
        @memset(owners, .{});
        @memset(stamps, 0);
        @memset(inverse, -1);
        @memset(payload, 0);
        @memset(pinned, false);
        return .{ .allocator = allocator, .geometry = geometry, .id = try identity(), .lanes = lanes, .owners = owners, .stamps = stamps, .inverse = inverse, .payload = payload, .assignments = assignments, .pinned = pinned };
    }

    pub fn deinit(arena: *Arena) !void {
        if (arena.active != null) return error.SharedKvBusy;
        arena.allocator.free(arena.pinned);
        arena.allocator.free(arena.assignments);
        arena.allocator.free(arena.payload);
        arena.allocator.free(arena.inverse);
        arena.allocator.free(arena.stamps);
        arena.allocator.free(arena.owners);
        arena.allocator.free(arena.lanes);
        arena.* = undefined;
    }

    fn state(arena: *const Arena, sequence: Sequence) !*Lane {
        if (sequence.arena != arena.id or sequence.lane >= arena.geometry.lanes or sequence.generation == 0) return error.SharedKvSequence;
        const lane = &arena.lanes[sequence.lane];
        if (!lane.live or lane.generation != sequence.generation) return error.SharedKvSequence;
        return lane;
    }

    pub fn register(arena: *Arena, lane: u32) !Sequence {
        if (arena.active != null) return error.SharedKvBusy;
        if (lane >= arena.geometry.lanes) return error.SharedKvSequence;
        const entry = &arena.lanes[lane];
        if (entry.live) return error.SharedKvSequenceLive;
        const generation = try std.math.add(u64, entry.generation, 1);
        entry.* = .{ .generation = generation, .live = true };
        return .{ .arena = arena.id, .lane = lane, .generation = generation };
    }

    fn owns(owner: Owner, sequence: Sequence) bool {
        return owner.generation != 0 and owner.generation == sequence.generation and owner.lane == sequence.lane;
    }

    fn inverseIndex(arena: *const Arena, lane: u32, key: Key) usize {
        return (@as(usize, lane) * arena.geometry.layers + key.layer) * arena.geometry.blocks + key.block;
    }

    fn evict(arena: *Arena, slot: u32) void {
        const owner = arena.owners[slot];
        if (owner.generation != 0) {
            const lane = arena.lanes[owner.lane];
            if (lane.live and lane.generation == owner.generation) {
                const inverse = &arena.inverse[arena.inverseIndex(owner.lane, .{ .layer = owner.layer, .block = owner.block })];
                if (inverse.* == @as(i32, @intCast(slot))) inverse.* = -1;
            }
        }
        arena.owners[slot] = .{};
        arena.stamps[slot] = 0;
    }

    fn clear(arena: *Arena, sequence: Sequence) void {
        for (arena.owners, 0..) |owner, slot| if (owns(owner, sequence)) arena.evict(@intCast(slot));
        const start = arena.inverseIndex(sequence.lane, .{ .layer = 0, .block = 0 });
        const length = @as(usize, arena.geometry.layers) * arena.geometry.blocks;
        @memset(arena.inverse[start..][0..length], -1);
    }

    pub fn reset(arena: *Arena, sequence: Sequence) !Sequence {
        if (arena.active != null) return error.SharedKvBusy;
        const lane = try arena.state(sequence);
        const generation = try std.math.add(u64, lane.generation, 1);
        arena.clear(sequence);
        lane.* = .{ .generation = generation, .live = true };
        return .{ .arena = arena.id, .lane = sequence.lane, .generation = generation };
    }

    pub fn release(arena: *Arena, sequence: Sequence) !void {
        if (arena.active != null) return error.SharedKvBusy;
        const lane = try arena.state(sequence);
        arena.clear(sequence);
        lane.live = false;
    }

    pub fn begin(arena: *Arena, sequence: Sequence) !Lease {
        const lane = try arena.state(sequence);
        if (arena.active != null) return error.SharedKvBusy;
        if (lane.failed) return error.SharedKvSequenceFailed;
        const serial = try std.math.add(u64, arena.serial, 1);
        const lease: Lease = .{ .arena = arena.id, .serial = serial, .sequence = sequence };
        arena.serial = serial;
        arena.active = .{ .lease = lease };
        return lease;
    }

    fn held(arena: *Arena, lease: Lease) !*Active {
        const active = if (arena.active) |*active| active else return error.SharedKvLease;
        if (!std.meta.eql(active.lease, lease)) return error.SharedKvLease;
        return active;
    }

    pub fn slotFor(arena: *const Arena, sequence: Sequence, key: Key) !?u32 {
        _ = try arena.state(sequence);
        if (key.layer >= arena.geometry.layers or key.block >= arena.geometry.blocks) return error.SharedKvKey;
        const inverse = arena.inverse[arena.inverseIndex(sequence.lane, key)];
        if (inverse < 0) return null;
        const slot: u32 = @intCast(inverse);
        if (slot >= arena.geometry.slots) return error.SharedKvInverse;
        const owner = arena.owners[slot];
        if (!owns(owner, sequence) or owner.layer != key.layer or owner.block != key.block or owner.valid == 0) return error.SharedKvInverse;
        return slot;
    }

    pub fn prepare(arena: *Arena, lease: Lease, keys: []const Key) ![]const Assignment {
        return arena.prepareEntries(lease, keys);
    }

    fn prepareEntries(arena: *Arena, lease: Lease, entries: anytype) ![]const Assignment {
        const active = try arena.held(lease);
        if (active.phase != .acquired) return error.SharedKvPhase;
        @memset(arena.pinned, false);
        var count: usize = 0;
        for (entries) |entry| {
            const key = if (@TypeOf(entry) == Key) entry else entry.key;
            if (key.layer >= arena.geometry.layers or key.block >= arena.geometry.blocks) return error.SharedKvKey;
            var duplicate = false;
            for (arena.assignments[0..count]) |previous| if (std.meta.eql(previous.key, key)) {
                duplicate = true;
                break;
            };
            if (duplicate) continue;
            if (count == arena.assignments.len) return error.SharedKvCapacity;
            const resident = try arena.slotFor(lease.sequence, key);
            arena.assignments[count] = .{ .key = key, .slot = resident orelse 0, .hit = resident != null };
            if (resident) |slot| arena.pinned[slot] = true;
            count += 1;
        }
        for (arena.assignments[0..count]) |*assignment| {
            if (assignment.hit) continue;
            var victim: ?u32 = null;
            for (arena.pinned, 0..) |pinned, slot| {
                if (pinned) continue;
                if (victim == null or arena.stamps[slot] < arena.stamps[victim.?]) victim = @intCast(slot);
            }
            const slot = victim orelse return error.SharedKvCapacity;
            assignment.slot = slot;
            arena.pinned[slot] = true;
        }
        active.count = count;
        active.phase = .planned;
        return arena.assignments[0..count];
    }

    pub fn cancel(arena: *Arena, lease: Lease) !void {
        const active = try arena.held(lease);
        if (active.phase != .acquired and active.phase != .planned) return error.SharedKvPending;
        arena.active = null;
    }

    pub fn submit(arena: *Arena, lease: Lease) !void {
        const active = try arena.held(lease);
        if (active.phase != .planned) return error.SharedKvPhase;
        const epoch = try std.math.add(u64, arena.clock, 1);
        for (arena.assignments[0..active.count]) |assignment| {
            if (!assignment.hit) {
                arena.evict(assignment.slot);
                arena.owners[assignment.slot] = .{ .generation = lease.sequence.generation, .lane = lease.sequence.lane, .layer = assignment.key.layer, .block = assignment.key.block };
            }
            arena.stamps[assignment.slot] = epoch;
        }
        arena.clock = epoch;
        active.phase = .submitted;
    }

    fn block(arena: *Arena, slot: u32) []u8 {
        const size = arena.geometry.blockBytes() catch unreachable;
        return arena.payload[@as(usize, slot) * size ..][0..size];
    }

    pub fn writeFetched(arena: *Arena, lease: Lease, index: usize, payload: []const u8) !void {
        const active = try arena.held(lease);
        if (active.phase != .submitted) return error.SharedKvPhase;
        if (index >= active.count or payload.len != try arena.geometry.blockBytes()) return error.SharedKvPayload;
        const assignment = &arena.assignments[index];
        @memcpy(arena.block(assignment.slot), payload);
        assignment.written = true;
    }

    pub fn completeFetch(arena: *Arena, lease: Lease) !void {
        const active = try arena.held(lease);
        if (active.phase != .submitted) return error.SharedKvPhase;
        for (arena.assignments[0..active.count]) |assignment| if (!assignment.hit and !assignment.written) return error.SharedKvFetchIncomplete;
        for (arena.assignments[0..active.count]) |assignment| {
            arena.owners[assignment.slot].valid = 1;
            arena.inverse[arena.inverseIndex(lease.sequence.lane, assignment.key)] = @intCast(assignment.slot);
        }
        active.phase = .ready;
    }

    pub fn read(arena: *Arena, lease: Lease, key: Key) ![]const u8 {
        const active = try arena.held(lease);
        if (active.phase != .ready) return error.SharedKvPhase;
        const slot = try arena.slotFor(lease.sequence, key) orelse return error.SharedKvMissing;
        return arena.block(slot);
    }

    pub fn completeConsumers(arena: *Arena, lease: Lease) !void {
        const active = try arena.held(lease);
        if (active.phase != .ready) return error.SharedKvPhase;
        arena.active = null;
    }

    pub fn failSubmitted(arena: *Arena, lease: Lease) !void {
        const active = try arena.held(lease);
        if (active.phase != .submitted and active.phase != .ready) return error.SharedKvPhase;
        active.phase = .poisoned;
        arena.lanes[lease.sequence.lane].failed = true;
    }

    pub fn recoverAfterDrain(arena: *Arena, lease: Lease) !void {
        const active = try arena.held(lease);
        if (active.phase != .poisoned) return error.SharedKvPhase;
        arena.clear(lease.sequence);
        arena.active = null;
    }

    pub fn validate(arena: *const Arena) !void {
        for (arena.owners, 0..) |owner, slot| {
            if (owner.generation == 0) {
                if (owner.valid != 0 or arena.stamps[slot] != 0) return error.SharedKvInverse;
                continue;
            }
            if (owner.lane >= arena.geometry.lanes or owner.layer >= arena.geometry.layers or owner.block >= arena.geometry.blocks or arena.stamps[slot] == 0 or arena.stamps[slot] > arena.clock) return error.SharedKvInverse;
            const lane = arena.lanes[owner.lane];
            if (!lane.live or lane.generation != owner.generation) return error.SharedKvInverse;
            const inverse = arena.inverse[arena.inverseIndex(owner.lane, .{ .layer = owner.layer, .block = owner.block })];
            if ((owner.valid != 0 and inverse != @as(i32, @intCast(slot))) or (owner.valid == 0 and inverse != -1)) return error.SharedKvInverse;
        }
        for (arena.lanes, 0..) |lane, lane_index| {
            for (0..arena.geometry.layers) |layer| for (0..arena.geometry.blocks) |logical_block| {
                const key: Key = .{ .layer = @intCast(layer), .block = @intCast(logical_block) };
                const inverse = arena.inverse[arena.inverseIndex(@intCast(lane_index), key)];
                if (inverse == -1) continue;
                if (!lane.live or inverse < 0 or inverse >= arena.geometry.slots) return error.SharedKvInverse;
                const owner = arena.owners[@intCast(inverse)];
                if (owner.valid == 0 or owner.lane != lane_index or owner.generation != lane.generation or owner.layer != layer or owner.block != logical_block) return error.SharedKvInverse;
            };
        }
    }

    pub fn saveOwned(arena: *Arena, allocator: std.mem.Allocator, lease: Lease) !OwnedBlocks {
        const active = try arena.held(lease);
        if (active.phase != .acquired) return error.SharedKvPhase;
        var count: usize = 0;
        for (arena.owners) |owner| if (owns(owner, lease.sequence) and owner.valid != 0) {
            count += 1;
        };
        const size = try arena.geometry.blockBytes();
        const entries = try allocator.alloc(OwnedEntry, count);
        errdefer allocator.free(entries);
        const payload = try allocator.alloc(u8, try std.math.mul(usize, count, size));
        errdefer allocator.free(payload);
        var position: usize = 0;
        for (arena.owners, 0..) |owner, slot| {
            if (!owns(owner, lease.sequence) or owner.valid == 0) continue;
            const key: Key = .{ .layer = owner.layer, .block = owner.block };
            if ((try arena.slotFor(lease.sequence, key)) != @as(u32, @intCast(slot))) return error.SharedKvInverse;
            entries[position] = .{ .key = key, .stamp = arena.stamps[slot], .source_slot = @intCast(slot) };
            @memcpy(payload[position * size ..][0..size], arena.block(@intCast(slot)));
            position += 1;
        }
        return .{ .source = lease.sequence, .geometry = arena.geometry, .clock = arena.clock, .entries = entries, .payload = payload };
    }

    pub fn restoreOwned(arena: *Arena, lease: Lease, image: *const OwnedBlocks) !void {
        const active = try arena.held(lease);
        if (active.phase != .acquired) return error.SharedKvPhase;
        try image.validate(arena.geometry);
        const next_clock = try std.math.add(u64, try std.math.add(u64, arena.clock, 1), try image.maxAge());
        _ = try arena.prepareEntries(lease, image.entries);
        try arena.submit(lease);
        errdefer arena.failSubmitted(lease) catch unreachable;
        for (arena.owners, 0..) |owner, slot| {
            if (!owns(owner, lease.sequence)) continue;
            var retained = false;
            for (image.entries) |entry| if (entry.key.layer == owner.layer and entry.key.block == owner.block) {
                retained = true;
                break;
            };
            if (!retained) arena.evict(@intCast(slot));
        }
        const size = try arena.geometry.blockBytes();
        for (image.entries, 0..) |entry, index| {
            try arena.writeFetched(lease, index, image.payload[index * size ..][0..size]);
            arena.stamps[arena.assignments[index].slot] = next_clock - (image.clock - entry.stamp);
        }
        arena.clock = next_clock;
        try arena.completeFetch(lease);
    }
};

test {
    _ = @import("exl3_shared_kv_test.zig");
}
