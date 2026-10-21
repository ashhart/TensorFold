const std = @import("std");

pub const widths = 17;
pub const Presence = enum { absent, present_unready, present_ready };
pub const Counters = struct {
    accesses: u64 = 0,
    misses: u64 = 0,
    present_unready: u64 = 0,
    first_keys: u64 = 0,
    returning_misses: u64 = 0,
    post_eviction_misses: u64 = 0,
    pressure_evictions: u64 = 0,
    width_changes: u64 = 0,
    state_changes: u64 = 0,
    keep_changes: u64 = 0,
    seam_changes: u64 = 0,
    paging_changes: u64 = 0,
    allocation_changes: u64 = 0,
    orientation_overflow: u64 = 0,
};

fn fingerprint(value: anytype) u64 {
    var hash = std.hash.Wyhash.init(0);
    std.hash.autoHash(&hash, value);
    return hash.final();
}

pub const Sample = struct {
    kind: usize,
    width: usize,
    lane: u64,
    exact: u64,
    state: u64,
    keep: u64,
    seam: u64,
    paging: u64,
    allocation: u64,

    pub fn fromKey(key: anytype) Sample {
        var allocation = key;
        allocation.rows = 0;
        allocation.keep = 0;
        allocation.chunks = @splat(0);
        allocation.score_tiles = @splat(0);
        allocation.paging = @splat(@splat(0));
        for (&allocation.seq) |*seq| seq[4] = 0;
        var states = key.seq;
        for (&states) |*seq| seq.* = .{ 0, 0, 0, 0, seq[4] };
        var origins = key.paging;
        for (&origins) |*paging| paging.* = .{ 0, 0, 0, 0, 0, paging[5] };
        var paging_allocations = key.paging;
        for (&paging_allocations) |*paging| paging[5] = 0;
        var identities = key.seq;
        for (&identities) |*seq| seq.* = .{ 0, 0, seq[2], 0, 0 };
        return .{
            .kind = key.kind - 1,
            .width = @min(key.rows, widths - 1),
            .lane = fingerprint(.{ key.kind, identities }),
            .exact = fingerprint(key),
            .state = fingerprint(states),
            .keep = key.keep,
            .seam = fingerprint(.{ key.chunks, key.score_tiles }),
            .paging = fingerprint(origins),
            .allocation = fingerprint(.{ allocation, paging_allocations }),
        };
    }
};

pub fn Profile(comptime lane_capacity: usize, comptime key_capacity: usize) type {
    return struct {
        const Self = @This();
        const Lane = struct {
            identity: u64,
            last: Sample,
            by_width: [2][widths]?Sample = @splat(@splat(null)),
            orientations: [2]?u64 = @splat(null),
            orientation_accesses: [3]u64 = @splat(0),
            width_mask: u32 = 0,
            new_keys_by_width: [widths]u64 = @splat(0),
            counters: Counters = .{},
        };
        counters: [2][widths]Counters = @splat(@splat(.{})),
        lanes: [lane_capacity]?Lane = @splat(null),
        keys: [key_capacity]?struct { fingerprint: u64, evicted: bool = false } = @splat(null),
        lane_overflow: u64 = 0,
        key_overflow: u64 = 0,

        pub fn observeEntry(self: *Self, sample: Sample, presence: Presence) void {
            self.observe(sample, presence != .absent);
            if (presence != .present_unready) return;
            self.counters[sample.kind][sample.width].present_unready +|= 1;
            for (&self.lanes) |*lane| if (lane.*) |*value| {
                if (value.identity == sample.lane) {
                    value.counters.present_unready +|= 1;
                    return;
                }
            };
        }

        pub fn observe(self: *Self, sample: Sample, hit: bool) void {
            const counts = &self.counters[sample.kind][sample.width];
            counts.accesses +|= 1;
            if (!hit) counts.misses +|= 1;
            var known = false;
            var empty: ?usize = null;
            for (&self.keys, 0..) |*key, index| {
                if (key.*) |*value| {
                    if (value.fingerprint == sample.exact) {
                        known = true;
                        if (value.evicted and !hit) counts.post_eviction_misses +|= 1;
                        value.evicted = false;
                    }
                } else if (empty == null) empty = index;
            }
            if (known) {
                if (!hit) counts.returning_misses +|= 1;
            } else if (empty) |index| {
                self.keys[index] = .{ .fingerprint = sample.exact };
                counts.first_keys +|= 1;
            } else self.key_overflow +|= 1;
            for (&self.lanes) |*lane| if (lane.*) |*value| {
                if (value.identity == sample.lane) return compare(value, counts, sample, hit, !known and empty != null, known);
            };
            for (&self.lanes) |*lane| if (lane.* == null) {
                lane.* = .{ .identity = sample.lane, .last = sample };
                return compare(&lane.*.?, counts, sample, hit, !known and empty != null, known);
            };
            self.lane_overflow +|= 1;
        }

        fn compare(lane: *Lane, counts: *Counters, sample: Sample, hit: bool, new_key: bool, known: bool) void {
            lane.counters.accesses +|= 1;
            if (!hit) lane.counters.misses +|= 1;
            if (new_key) lane.counters.first_keys +|= 1;
            if (new_key) lane.new_keys_by_width[sample.width] +|= 1;
            if (known and !hit) lane.counters.returning_misses +|= 1;
            lane.width_mask |= @as(u32, 1) << @intCast(sample.width);
            if (lane.last.width != sample.width) {
                counts.width_changes +|= 1;
                lane.counters.width_changes +|= 1;
            }
            if (lane.by_width[sample.kind][sample.width]) |previous| {
                inline for (.{ .{ "state", "state_changes" }, .{ "keep", "keep_changes" }, .{ "seam", "seam_changes" }, .{ "paging", "paging_changes" }, .{ "allocation", "allocation_changes" } }) |fields| {
                    if (@field(previous, fields[0]) != @field(sample, fields[0])) {
                        @field(counts, fields[1]) +|= 1;
                        @field(lane.counters, fields[1]) +|= 1;
                    }
                }
            }
            var orientation: usize = 2;
            for (&lane.orientations, 0..) |*state, index| {
                if (state.* == null) state.* = sample.state;
                if (state.*.? == sample.state) {
                    orientation = index;
                    break;
                }
            }
            lane.orientation_accesses[orientation] +|= 1;
            if (orientation == 2) {
                counts.orientation_overflow +|= 1;
                lane.counters.orientation_overflow +|= 1;
            }
            lane.last = sample;
            lane.by_width[sample.kind][sample.width] = sample;
        }

        pub fn evict(self: *Self, sample: Sample) void {
            self.counters[sample.kind][sample.width].pressure_evictions +|= 1;
            for (&self.keys) |*key| if (key.*) |*value| {
                if (value.fingerprint == sample.exact) value.evicted = true;
            };
            for (&self.lanes) |*lane| if (lane.*) |*value| {
                if (value.identity == sample.lane) {
                    value.counters.pressure_evictions +|= 1;
                    return;
                }
            };
        }
    };
}

fn fixture() Sample {
    return .{ .kind = 1, .width = 2, .lane = 1, .exact = 1, .state = 1, .keep = 1, .seam = 1, .paging = 1, .allocation = 1 };
}

test "exl3: profile entry presence does not imply replay readiness or replay coverage" {
    var profile: Profile(2, 8) = .{};
    const sample = fixture();
    profile.observeEntry(sample, .absent);
    profile.observeEntry(sample, .present_unready);
    profile.observeEntry(sample, .present_ready);
    try std.testing.expectEqual(@as(u64, 3), profile.counters[1][2].accesses);
    try std.testing.expectEqual(@as(u64, 1), profile.counters[1][2].misses);
    try std.testing.expectEqual(@as(u64, 1), profile.counters[1][2].present_unready);
    try std.testing.expectEqual(@as(u64, 1), profile.lanes[0].?.counters.present_unready);
    profile.counters[1][2].present_unready = std.math.maxInt(u64);
    profile.observeEntry(sample, .present_unready);
    try std.testing.expectEqual(std.math.maxInt(u64), profile.counters[1][2].present_unready);
    try std.testing.expectEqual(@as(u64, 1), profile.counters[1][2].misses);
}

test "exl3: profile distinguishes new keys returning misses and pressure evictions" {
    var profile: Profile(2, 8) = .{};
    const sample = fixture();
    profile.observe(sample, false);
    profile.observe(sample, true);
    profile.evict(sample);
    profile.observe(sample, false);
    const counts = profile.counters[1][2];
    try std.testing.expectEqual(@as(u64, 3), counts.accesses);
    try std.testing.expectEqual(@as(u64, 2), counts.misses);
    try std.testing.expectEqual(@as(u64, 1), counts.first_keys);
    try std.testing.expectEqual(@as(u64, 1), counts.returning_misses);
    try std.testing.expectEqual(@as(u64, 1), counts.pressure_evictions);
    try std.testing.expectEqual(@as(u64, 1), counts.post_eviction_misses);
    try std.testing.expectEqual(@as(u64, 1), profile.lanes[0].?.new_keys_by_width[2]);
}

test "exl3: profile couples state and keep and separates width baselines" {
    var profile: Profile(2, 8) = .{};
    var sample = fixture();
    profile.observe(sample, false);
    sample.width = 1;
    sample.keep = 0;
    sample.exact = 2;
    profile.observe(sample, false);
    sample.width = 2;
    sample.keep = 2;
    sample.state = 2;
    sample.exact = 3;
    profile.observe(sample, false);
    try std.testing.expectEqual(@as(u64, 1), profile.counters[1][2].state_changes);
    try std.testing.expectEqual(@as(u64, 1), profile.counters[1][2].keep_changes);
    try std.testing.expectEqual(@as(u64, 0), profile.counters[1][1].keep_changes);
    try std.testing.expectEqual([3]u64{ 2, 1, 0 }, profile.lanes[0].?.orientation_accesses);
}

test "exl3: profile separates seam paging and allocation changes" {
    var profile: Profile(2, 8) = .{};
    var sample = fixture();
    profile.observe(sample, false);
    sample.seam = 2;
    profile.observe(sample, false);
    sample.paging = 2;
    profile.observe(sample, false);
    sample.allocation = 2;
    profile.observe(sample, false);
    const counts = profile.counters[1][2];
    try std.testing.expectEqual(@as(u64, 1), counts.seam_changes);
    try std.testing.expectEqual(@as(u64, 1), counts.paging_changes);
    try std.testing.expectEqual(@as(u64, 1), counts.allocation_changes);
    try std.testing.expectEqual(@as(u64, 0), counts.state_changes);
}

test "exl3: profile bounds lanes keys orientations and saturates counters" {
    var profile: Profile(1, 1) = .{};
    var sample = fixture();
    profile.observe(sample, false);
    sample.state = 2;
    profile.observe(sample, true);
    sample.state = 3;
    profile.observe(sample, true);
    sample.lane = 2;
    sample.exact = 2;
    profile.observe(sample, false);
    try std.testing.expectEqual(@as(u64, 1), profile.lane_overflow);
    try std.testing.expectEqual(@as(u64, 1), profile.key_overflow);
    try std.testing.expectEqual(@as(u64, 1), profile.counters[1][2].orientation_overflow);
    profile.counters[1][2].accesses = std.math.maxInt(u64);
    profile.observe(sample, true);
    try std.testing.expectEqual(std.math.maxInt(u64), profile.counters[1][2].accesses);
}

test "exl3: profile key projection separates orientation seam paging and allocations without mutating key" {
    const Key = struct {
        kind: u8 = 2,
        rows: usize = 2,
        seq: [2][5]u64 = .{ .{ 1, 2, 3, 4, 5 }, @splat(0) },
        paging: [2][6]u64 = .{ .{ 6, 7, 8, 9, 10, 11 }, @splat(0) },
        chunks: [2]u32 = .{ 1, 0 },
        score_tiles: [2]u32 = @splat(0),
        scratch: u64 = 12,
        scratch_cap: usize = 32,
        scores: u64 = 13,
        output: u64 = 14,
        streams: u64 = 15,
        keep: u64 = 16,
        plan: u64 = 17,
    };
    const original: Key = .{};
    var key = original;
    const base = Sample.fromKey(key);
    key.seq[0][4] += 1;
    key.keep += 1;
    const orientation = Sample.fromKey(key);
    try std.testing.expect(base.state != orientation.state);
    try std.testing.expectEqual(base.allocation, orientation.allocation);
    try std.testing.expectEqual(base.lane, orientation.lane);
    key = original;
    key.chunks[0] += 1;
    const seam = Sample.fromKey(key);
    try std.testing.expect(base.seam != seam.seam);
    try std.testing.expectEqual(base.allocation, seam.allocation);
    key = original;
    key.paging[0][5] += 1;
    const paging = Sample.fromKey(key);
    try std.testing.expect(base.paging != paging.paging);
    try std.testing.expectEqual(base.allocation, paging.allocation);
    key = original;
    key.paging[1][0] += 1;
    try std.testing.expect(base.allocation != Sample.fromKey(key).allocation);
    key = original;
    key.scratch += 1;
    try std.testing.expect(base.allocation != Sample.fromKey(key).allocation);
    try std.testing.expectEqualDeep(original, Key{});
}
