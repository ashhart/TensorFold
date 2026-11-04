//! CPU counterfactuals for one ordered pool-demand timeline.

const std = @import("std");
const trace = @import("exl3_dplan_trace.zig");

pub const expert_count: usize = 512;
pub const max_unique: usize = 256;

pub const Policy = enum { lifetime, lru, decayed_frequency, decayed_admission };

pub const Seed = struct {
    line_ids: []const i32,
    stamps: []const u32,
    counts: [expert_count]u32,
    now: u32,
};

pub const Options = struct {
    capacity: usize,
    seed: ?Seed = null,
    decay_interval_rounds: u64 = 64,
    decay_shift: u5 = 1,
};

pub const Metrics = struct {
    rounds: u64 = 0,
    hit_picks: u64 = 0,
    miss_picks: u64 = 0,
    hit_experts: u64 = 0,
    miss_experts: u64 = 0,
    admissions: u64 = 0,
    copies: u64 = 0,
    evictions: u64 = 0,
    mapped_experts: u64 = 0,
    mapped_picks: u64 = 0,

    fn plus(self: Metrics, other: Metrics) !Metrics {
        var result: Metrics = undefined;
        inline for (@typeInfo(Metrics).@"struct".field_names) |field| {
            @field(result, field) = try std.math.add(u64, @field(self, field), @field(other, field));
        }
        return result;
    }
};

pub const Comparison = struct {
    lifetime: Metrics,
    lru: Metrics,
    decayed_frequency: Metrics,
    decayed_admission: Metrics,
    belady_unique_misses_lower_bound: u64,
};

const Demand = struct {
    counts: [expert_count]u32 = @splat(0),
    ordered: [expert_count]usize = undefined,
    unique: usize = 0,
    valid_picks: u64 = 0,
    picks: []const u32,
};

pub const State = struct {
    allocator: std.mem.Allocator,
    policy: Policy,
    options: Options,
    line_ids: []i32,
    stamps: []u32,
    scores: [expert_count]u32,
    last_access: [expert_count]u64,
    access_clock: u64,
    now: u32,
    total: Metrics = .{},

    pub fn init(allocator: std.mem.Allocator, policy: Policy, options: Options) !State {
        try validateOptions(options);
        const line_ids = try allocator.alloc(i32, options.capacity);
        errdefer allocator.free(line_ids);
        const stamps = try allocator.alloc(u32, options.capacity);
        @memset(line_ids, -1);
        @memset(stamps, 0);
        var self: State = .{
            .allocator = allocator,
            .policy = policy,
            .options = options,
            .line_ids = line_ids,
            .stamps = stamps,
            .scores = @splat(0),
            .last_access = @splat(0),
            .access_clock = 0,
            .now = 0,
        };
        if (options.seed) |seed| {
            @memcpy(line_ids, seed.line_ids);
            @memcpy(stamps, seed.stamps);
            self.scores = seed.counts;
            self.now = seed.now;
            self.access_clock = seed.now;
            for (line_ids, stamps) |expert, stamp| {
                self.access_clock = @max(self.access_clock, stamp);
                if (expert >= 0) self.last_access[@intCast(expert)] = stamp;
            }
        }
        self.options.seed = null;
        return self;
    }

    pub fn deinit(self: *State) void {
        self.allocator.free(self.line_ids);
        self.allocator.free(self.stamps);
        self.* = undefined;
    }

    pub fn observe(self: *State, record: trace.Round) !Metrics {
        const demand = try readDemand(record);
        const round_number = try std.math.add(u64, self.total.rounds, 1);
        const next_clock = try std.math.add(u64, self.access_clock, demand.valid_picks);
        var delta: Metrics = .{ .rounds = 1 };
        for (demand.ordered[0..demand.unique]) |expert| {
            if (self.lineOf(expert) != null) {
                delta.hit_picks += demand.counts[expert];
                delta.hit_experts += 1;
            } else {
                delta.miss_picks += demand.counts[expert];
                delta.miss_experts += 1;
            }
        }
        _ = try self.total.plus(delta);
        if ((self.policy == .decayed_frequency or self.policy == .decayed_admission) and round_number % self.options.decay_interval_rounds == 0) {
            for (&self.scores) |*score| score.* >>= self.options.decay_shift;
        }
        self.now +%= 1;
        for (demand.ordered[0..demand.unique]) |expert| {
            self.scores[expert] +|= demand.counts[expert];
            if (self.lineOf(expert)) |line| self.stamps[line] = self.now;
        }
        for (demand.picks) |bits| {
            const expert = routed(bits) orelse continue;
            self.access_clock += 1;
            self.last_access[expert] = self.access_clock;
        }
        std.debug.assert(self.access_clock == next_clock);

        var misses: [max_unique]usize = undefined;
        var miss_count: usize = 0;
        for (demand.ordered[0..demand.unique]) |expert| {
            if (self.lineOf(expert) == null) {
                misses[miss_count] = expert;
                miss_count += 1;
            }
        }
        var index: usize = 1;
        while (index < miss_count) : (index += 1) {
            const expert = misses[index];
            var position = index;
            while (position > 0 and self.preferAdmission(expert, misses[position - 1], &demand)) : (position -= 1) {
                misses[position] = misses[position - 1];
            }
            misses[position] = expert;
        }
        for (misses[0..miss_count]) |expert| {
            if (self.victim(&demand)) |line| {
                if (self.policy == .decayed_admission and self.line_ids[line] >= 0 and self.scores[expert] < self.scores[@intCast(self.line_ids[line])]) {
                    delta.mapped_experts += 1;
                    delta.mapped_picks += demand.counts[expert];
                    continue;
                }
                if (self.line_ids[line] >= 0) delta.evictions += 1;
                self.line_ids[line] = @intCast(expert);
                self.stamps[line] = self.now;
                delta.admissions += 1;
                delta.copies += 1;
            } else {
                delta.mapped_experts += 1;
                delta.mapped_picks += demand.counts[expert];
            }
        }
        self.total = try self.total.plus(delta);
        return delta;
    }

    fn lineOf(self: *const State, expert: usize) ?usize {
        for (self.line_ids, 0..) |resident, line| {
            if (resident == @as(i32, @intCast(expert))) return line;
        }
        return null;
    }

    fn preferAdmission(self: *const State, left: usize, right: usize, demand: *const Demand) bool {
        if (self.policy == .lru) {
            if (self.last_access[left] != self.last_access[right]) return self.last_access[left] > self.last_access[right];
        } else {
            if (self.scores[left] != self.scores[right]) return self.scores[left] > self.scores[right];
            if (demand.counts[left] != demand.counts[right]) return demand.counts[left] > demand.counts[right];
        }
        return left < right;
    }

    fn victim(self: *const State, demand: *const Demand) ?usize {
        var coldest: ?usize = null;
        for (self.line_ids, 0..) |expert, line| {
            if (expert < 0) return line;
            if (demand.counts[@intCast(expert)] != 0) continue;
            if (coldest == null or self.colder(line, coldest.?)) coldest = line;
        }
        return coldest;
    }

    fn colder(self: *const State, left: usize, right: usize) bool {
        const left_expert: usize = @intCast(self.line_ids[left]);
        const right_expert: usize = @intCast(self.line_ids[right]);
        if (self.policy != .lru and self.scores[left_expert] != self.scores[right_expert]) return self.scores[left_expert] < self.scores[right_expert];
        if (self.policy == .lifetime) {
            if (self.stamps[left] != self.stamps[right]) return self.stamps[left] < self.stamps[right];
        } else {
            if (self.last_access[left_expert] != self.last_access[right_expert]) return self.last_access[left_expert] < self.last_access[right_expert];
        }
        return left > right;
    }
};

pub fn evaluate(allocator: std.mem.Allocator, rounds: []const trace.Round, options: Options) !Comparison {
    var lifetime = try State.init(allocator, .lifetime, options);
    defer lifetime.deinit();
    var lru = try State.init(allocator, .lru, options);
    defer lru.deinit();
    var decayed = try State.init(allocator, .decayed_frequency, options);
    defer decayed.deinit();
    var decayed_admission = try State.init(allocator, .decayed_admission, options);
    defer decayed_admission.deinit();
    for (rounds) |record| {
        _ = try lifetime.observe(record);
        _ = try lru.observe(record);
        _ = try decayed.observe(record);
        _ = try decayed_admission.observe(record);
    }
    return .{
        .lifetime = lifetime.total,
        .lru = lru.total,
        .decayed_frequency = decayed.total,
        .decayed_admission = decayed_admission.total,
        .belady_unique_misses_lower_bound = try beladyLowerBound(allocator, rounds, options),
    };
}

pub fn evaluateTrace(allocator: std.mem.Allocator, view: trace.View) !Comparison {
    const verification = try @import("exl3_dplan_replay.zig").verifyTrace(allocator, view);
    const rounds = try allocator.alloc(trace.Round, view.recorded());
    defer allocator.free(rounds);
    for (rounds, 0..) |*round, index| round.* = try view.round(index);
    var options: Options = .{ .capacity = view.words[4], .seed = view.seedState(Seed) };
    var total: Comparison = .{ .lifetime = .{}, .lru = .{}, .decayed_frequency = .{}, .decayed_admission = .{}, .belady_unique_misses_lower_bound = 0 };
    var begin: usize = 0;
    for (0..view.checkpoints() + 1) |index| {
        const checkpoint = if (index < view.checkpoints()) try view.checkpoint(index) else null;
        const end = if (checkpoint) |entry| entry.roundIndex() else view.recorded();
        if (end > begin) {
            const segment = try evaluate(allocator, rounds[begin..end], options);
            inline for (.{ "lifetime", "lru", "decayed_frequency", "decayed_admission" }) |field| {
                @field(total, field) = try @field(total, field).plus(@field(segment, field));
            }
            total.belady_unique_misses_lower_bound = try std.math.add(u64, total.belady_unique_misses_lower_bound, segment.belady_unique_misses_lower_bound);
        }
        if (checkpoint) |entry| options.seed = trace.seedFromState(Seed, view.words[4], entry.after());
        begin = end;
    }
    const baseline = total.lifetime;
    const exact = verification.delta;
    if (baseline.rounds != @as(u64, @intCast(verification.rounds)) or
        baseline.hit_picks != exact.hit_picks or
        baseline.miss_picks != exact.miss_picks or
        baseline.copies != exact.copies or
        baseline.evictions != exact.evictions or
        baseline.mapped_experts != exact.inplace or
        baseline.mapped_picks != exact.mapped_picks)
        return error.PolicyBaselineMismatch;
    return total;
}

/// Belady with optional admission is optimal for uniform-cost sequential requests: on a full-cache miss.
pub fn beladyLowerBound(allocator: std.mem.Allocator, rounds: []const trace.Round, options: Options) !u64 {
    try validateOptions(options);
    var count: usize = 0;
    for (rounds) |record| count = try std.math.add(usize, count, (try readDemand(record)).unique);
    const events = try allocator.alloc(Event, count);
    defer allocator.free(events);
    var offset: usize = 0;
    for (rounds) |record| {
        const demand = try readDemand(record);
        for (demand.ordered[0..demand.unique]) |expert| {
            events[offset] = .{ .expert = expert };
            offset += 1;
        }
    }
    return belady(events, options.capacity, if (options.seed) |seed| seed.line_ids else &.{});
}

pub const Event = struct { expert: usize, next: usize = std.math.maxInt(usize) };

pub fn belady(events: []Event, capacity: usize, initial: []const i32) u64 {
    var next_use: [expert_count]usize = @splat(std.math.maxInt(usize));
    var position = events.len;
    while (position > 0) {
        position -= 1;
        const event = &events[position];
        event.next = next_use[event.expert];
        next_use[event.expert] = position;
    }
    var resident: [expert_count]i32 = @splat(-1);
    @memcpy(resident[0..initial.len], initial);
    var misses: u64 = 0;
    for (events) |event| {
        var hit = false;
        var empty: ?usize = null;
        var farthest: ?usize = null;
        for (resident[0..capacity], 0..) |expert, line| {
            if (expert == @as(i32, @intCast(event.expert))) hit = true;
            if (expert < 0) {
                if (empty == null) empty = line;
            } else if (farthest == null or next_use[@intCast(expert)] > next_use[@intCast(resident[farthest.?])]) {
                farthest = line;
            }
        }
        next_use[event.expert] = event.next;
        if (hit) continue;
        misses += 1;
        if (empty) |line| {
            resident[line] = @intCast(event.expert);
        } else if (farthest) |line| {
            if (event.next < next_use[@intCast(resident[line])]) resident[line] = @intCast(event.expert);
        }
    }
    return misses;
}

fn validateOptions(options: Options) !void {
    if (options.capacity > expert_count or options.decay_interval_rounds == 0 or options.decay_shift == 0) return error.InvalidOptions;
    if (options.seed) |seed| {
        if (seed.line_ids.len != options.capacity or seed.stamps.len != options.capacity) return error.InvalidSeed;
        var seen: [expert_count]bool = @splat(false);
        for (seed.line_ids) |expert| {
            if (expert < -1 or expert >= expert_count) return error.InvalidSeed;
            if (expert >= 0) {
                const index: usize = @intCast(expert);
                if (seen[index]) return error.InvalidSeed;
                seen[index] = true;
            }
        }
    }
}

fn readDemand(record: trace.Round) !Demand {
    const shape = trace.header(0, 0, record.lines, record.max_rows) catch return error.InvalidRound;
    if (record.words.len != shape[7]) return error.InvalidRound;
    if (record.recentEnabled()) return error.UnsupportedRecentPolicy;
    if (record.rows() == 0 or record.rows() > record.max_rows or record.words[3] > 1) return error.InvalidRound;
    var demand: Demand = .{ .picks = record.picks() };
    for (demand.picks) |bits| {
        const expert = routed(bits) orelse continue;
        if (demand.counts[expert] == 0) {
            demand.ordered[demand.unique] = expert;
            demand.unique += 1;
        }
        demand.counts[expert] += 1;
        demand.valid_picks += 1;
    }
    if (demand.valid_picks != record.validPicks() or demand.unique != record.uniqueCount()) return error.InvalidRound;
    if (demand.unique > max_unique) return error.PlannerOverflow;
    return demand;
}

pub fn routed(bits: u32) ?usize {
    const expert: i32 = @bitCast(bits);
    return if (expert >= 0 and expert < expert_count) @intCast(expert) else null;
}
