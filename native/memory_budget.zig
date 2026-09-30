const std = @import("std");
pub const gib = 1024 * 1024 * 1024;
pub const process_bytes = 3 * gib;
pub const probe_repeats = 3;
pub const growth_horizon = 2048;

fn bytes(value: f64) !u64 {
    if (!std.math.isFinite(value) or value < 0 or value >= 18446744073709551616.0) return error.InvalidMemorySize;
    return @intFromFloat(value);
}

pub fn limit(ram: u64, recommended: u64, fraction: f64, override: ?[]const u8) !u64 {
    if (ram == 0 or !std.math.isFinite(fraction) or fraction <= 0 or fraction > 1) return error.InvalidMemoryBudget;
    var value = try bytes(fraction * @as(f64, @floatFromInt(ram)));
    if (override) |text| {
        const number = std.fmt.parseFloat(f64, std.mem.trim(u8, text, " \t\r\n")) catch return error.InvalidMemoryBudget;
        if (!std.math.isFinite(number) or number <= 0) return error.InvalidMemoryBudget;
        value = @max(1, try bytes(@min(number, @as(f64, @floatFromInt(ram)) / gib) * gib));
    }
    return @min(value, if (recommended > 0) @min(ram, recommended) else ram);
}

pub fn concurrentBudget(ram: u64, fraction: f64, process_budget: u64, share: u64, elsewhere: u64) u64 {
    const allowance = @max(@as(u64, @intFromFloat(fraction * @as(f64, @floatFromInt(ram)))), process_budget);
    return @min(allowance -| elsewhere, share);
}

pub const CacheMemory = struct {
    fixed_bytes: u64,
    bytes_per_token: u64,
    step: u64 = 256,
    entry_bytes_per_token: u64 = 0,

    fn positions(m: CacheMemory, tokens: u64) !u64 {
        if (m.step == 0) return error.InvalidMemoryStep;
        const rounded = try std.math.add(u64, tokens, m.step - 1);
        return try std.math.mul(u64, rounded / m.step, m.step);
    }

    pub fn cacheBytes(m: CacheMemory, tokens: u64) !u64 {
        return std.math.add(u64, m.fixed_bytes, try std.math.mul(u64, try m.positions(tokens), m.bytes_per_token));
    }

    pub fn growthBytes(m: CacheMemory, tokens: u64, in_flight: u64) !u64 {
        const each = if (m.entry_bytes_per_token == 0) m.bytes_per_token else @min(m.bytes_per_token, m.entry_bytes_per_token *| in_flight);
        return std.math.add(u64, m.fixed_bytes, try std.math.mul(u64, try m.positions(tokens), each));
    }

    pub const Request = struct {
        resident_bytes: u64,
        working_bytes: u64 = 0,
        cache_copies: u64 = 1,
        reserve_tokens: u64 = 0,
    };

    pub fn needed(m: CacheMemory, tokens: u64, request: Request) !u64 {
        if (request.cache_copies == 0) return error.InvalidCacheCopies;
        const cache = try m.cacheBytes(try std.math.add(u64, tokens, request.reserve_tokens));
        return std.math.add(u64, try std.math.add(u64, request.resident_bytes, request.working_bytes), try std.math.mul(u64, cache, request.cache_copies));
    }

    pub fn largestContext(m: CacheMemory, window: u64, budget: u64, request: Request) !u64 {
        if (try m.needed(0, request) > budget) return 0;
        var lo: u64 = 0;
        var hi = window -| request.reserve_tokens;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2 + (hi - lo) % 2;
            if (try m.needed(mid, request) <= budget) lo = mid else hi = mid - 1;
        }
        return lo;
    }
};

pub const StreamMemory = struct {
    short_tokens: u64,
    short: u64,
    long_tokens: u64,
    long: u64,
    per_token: f64,
    prefill_a: f64,
    prefill_b: f64,
    round_bytes: u64,
    chunk: u64 = 2048,

    pub fn validate(m: StreamMemory) !void {
        if (m.long_tokens <= m.short_tokens or m.long < m.short or m.chunk == 0) return error.InvalidMemoryProfile;
        for ([_]f64{ m.per_token, m.prefill_a, m.prefill_b }) |value| if (!std.math.isFinite(value) or value < 0) return error.InvalidMemoryProfile;
    }

    pub fn include(m: *StreamMemory, other: StreamMemory) !void {
        try m.validate();
        try other.validate();
        if (m.short_tokens != other.short_tokens or m.long_tokens != other.long_tokens or m.chunk != other.chunk) return error.IncompatibleMemoryProbes;
        inline for (.{ "short", "long", "per_token", "prefill_a", "prefill_b", "round_bytes" }) |field| {
            @field(m, field) = @max(@field(m, field), @field(other, field));
        }
    }

    pub fn streamBytes(m: StreamMemory, tokens: u64) !u64 {
        try m.validate();
        const t = try std.math.add(u64, tokens, 256);
        if (t <= m.short_tokens) return m.short;
        if (t <= m.long_tokens) {
            const numerator = try std.math.mul(u64, m.long - m.short, t - m.short_tokens);
            return bytes(@as(f64, @floatFromInt(m.short)) + @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(m.long_tokens - m.short_tokens)));
        }
        return bytes(@as(f64, @floatFromInt(m.long)) + m.per_token * @as(f64, @floatFromInt(t - m.long_tokens)));
    }

    pub fn prefillBytes(m: StreamMemory, tokens: u64) !u64 {
        try m.validate();
        const chunk: f64 = @floatFromInt(@min(m.chunk, @max(1, tokens)));
        return bytes(m.prefill_a * chunk + m.prefill_b * chunk * @as(f64, @floatFromInt(tokens)));
    }
};

pub const Live = struct { now: u64, most: u64, copy_bytes: u64 = 0 };

pub fn reserveReply(reply: u64, horizon: ?u64) u64 {
    return if (horizon) |limit_tokens| @min(reply, limit_tokens) else reply;
}

pub fn reserveLive(stream: Live, prompt: u64, horizon: ?u64) Live {
    // Native prefills can interleave; reserve their unfinished prompt as well as reply growth.
    return .{ .now = stream.now, .most = if (horizon) |limit_tokens| @min(stream.most, @max(stream.now, prompt) +| limit_tokens) else stream.most, .copy_bytes = stream.copy_bytes };
}

pub const StreamGate = struct {
    per_token: f64,
    work: u64,
    budget: u64,
    horizon: u64 = growth_horizon,
    lanes: u64 = 1,
    waits: u64 = 0,
    ends: u64 = 0,

    pub const Plan = struct { run: usize, paused: usize, ended: ?usize = null };

    pub fn growth(g: StreamGate, stream: Live) !u64 {
        return bytes(@as(f64, @floatFromInt(@min(stream.most -| stream.now, g.horizon))) * g.per_token);
    }

    pub fn need(g: StreamGate, used: u64, streams: []const Live) !u64 {
        const lanes = @max(1, g.lanes);
        const work = try std.math.mul(u64, g.work, @min(streams.len, lanes));
        var result = try std.math.add(u64, used, work / lanes + @intFromBool(work % lanes != 0));
        for (streams) |stream| result = try std.math.add(u64, result, try std.math.add(u64, try g.growth(stream), stream.copy_bytes));
        return result;
    }

    /// Streams are oldest first. Memory is remeasured after every reclaim because snapshots may share arrays.
    pub fn plan(g: *StreamGate, memory: anytype, streams: []const Live) !Plan {
        if (streams.len == 0) return .{ .run = 0, .paused = 0 };
        var count = streams.len;
        while (true) {
            while (try g.need(try memory.used(), streams[0..count]) > g.budget and
                (try g.need(try memory.used(), streams[0..count])) -| (try memory.freeable()) <= g.budget)
            {
                if (!try memory.reclaim()) break;
            }
            if (try g.need(try memory.used(), streams[0..count]) <= g.budget or count == 1) break;
            count -= 1;
        }
        var result = Plan{ .run = count, .paused = streams.len - count };
        if (result.paused > 0 and try g.need(try memory.used(), streams[0..count]) > g.budget) {
            result.ended = streams.len - 1;
            result.paused -= 1;
        }
        g.waits +|= @intFromBool(result.paused > 0);
        g.ends +|= @intFromBool(result.ended != null);
        return result;
    }
};
pub const Admission = struct {
    budget: u64,
    memory: StreamMemory,
    refused: usize = 0,
    lanes: u64 = 1,

    pub fn roundBytes(admission: Admission, streams: u64) !u64 {
        const lanes = @max(1, admission.lanes);
        const total = try std.math.mul(u64, admission.memory.round_bytes, @min(@max(1, streams), lanes));
        return total / lanes + @intFromBool(total % lanes != 0);
    }

    pub fn projected(admission: Admission, used: u64, prompt: u64, longest: u64, live: []const Live) !u64 {
        var growth: u64 = 0;
        for (live) |request| growth = try std.math.add(u64, growth, request.most -| request.now);
        const work = @max(try admission.roundBytes(live.len + 1), try admission.memory.prefillBytes(prompt));
        // Preserve upstream's float addition order and final truncation.
        return bytes(@as(f64, @floatFromInt(used)) + @as(f64, @floatFromInt(growth)) * admission.memory.per_token + @as(f64, @floatFromInt(try admission.memory.streamBytes(longest))) + @as(f64, @floatFromInt(work)));
    }

    pub fn admits(admission: *Admission, used: u64, prompt: u64, longest: u64, live: []const Live) !bool {
        const ok = try admission.projected(used, prompt, longest, live) <= admission.budget;
        if (!ok) admission.refused +|= 1;
        return ok;
    }

    pub fn prefillProjected(admission: Admission, used: u64, prompt: u64, now: u64, copies: u64, decoding: []const Live) !u64 {
        var grow = @min(admission.memory.chunk, prompt -| now);
        var copy_bytes = copies;
        for (decoding) |stream| {
            grow = try std.math.add(u64, grow, stream.most -| stream.now);
            copy_bytes = try std.math.add(u64, copy_bytes, stream.copy_bytes);
        }
        const work = @max(try admission.roundBytes(decoding.len + 1), try admission.memory.prefillBytes(prompt));
        return bytes(@as(f64, @floatFromInt(used)) + @as(f64, @floatFromInt(copy_bytes)) + @as(f64, @floatFromInt(grow)) * admission.memory.per_token + @as(f64, @floatFromInt(work)));
    }

    pub fn fitting(admission: Admission, used: u64, tokens: u64) !u64 {
        const each = try admission.memory.streamBytes(tokens);
        if (used > admission.budget) return 0;
        const room = admission.budget - used;
        const prefill = try admission.memory.prefillBytes(tokens);
        var count: u64 = 0;
        while (count < 64) : (count += 1) {
            const retained = try std.math.mul(u64, count + 1, each);
            const work = @max(try admission.roundBytes(count + 1), prefill);
            if (try std.math.add(u64, retained, work) > room) break;
        }
        return count;
    }
};

pub fn check(io: std.Io, path: []const u8) !void {
    const a = std.heap.page_allocator;
    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(32 * 1024 * 1024));
    defer a.free(source);
    const Fixture = struct {
        probe_repeats: u32,
        growth_horizon: u64,
        limits: []const struct { ram: u64, recommended: u64, fraction: f64, override: ?[]const u8, result: ?u64 },
        caches: []const struct { memory: CacheMemory, tokens: u64, in_flight: u64, request: CacheMemory.Request, budget: u64, window: u64, cache: u64, growth: u64, needed: u64, largest: u64 },
        streams: []const struct { memory: StreamMemory, lanes: u64, tokens: u64, prompt: u64, used: u64, budget: u64, live: []const Live, stream: u64, prefill: u64, projected: u64, admits: bool, fitting: u64 },
        budgets: []const struct { ram: u64, fraction: f64, process: u64, share: u64, elsewhere: u64, result: u64 },
        reservations: []const struct {
            horizon: ?u64,
            prompt: u64,
            reply: u64,
            longest: u64,
            memory: StreamMemory,
            used: u64,
            lanes: u64,
            budget: u64,
            projected: u64,
            fits: bool,
            jobs: []const struct { prompt: u64, now: u64, most: u64 },
            live: []const Live,
        },
        gates: []const struct {
            active: u64,
            cache: u64,
            entries: []const GateMemory.Entry,
            per_token: f64,
            work: u64,
            horizon: u64,
            lanes: u64,
            streams: []const Live,
            plans: []const struct { budget: u64, run: usize, paused: usize, ended: ?usize, waits: u64, ends: u64, active: u64, cache: u64, entries: usize, reclaims: u64 },
        },
    };
    const parsed = try std.json.parseFromSlice(Fixture, a, source, .{});
    defer parsed.deinit();
    try std.testing.expectEqual(probe_repeats, parsed.value.probe_repeats);
    try std.testing.expectEqual(growth_horizon, parsed.value.growth_horizon);
    for (parsed.value.limits) |case| {
        const result = limit(case.ram, case.recommended, case.fraction, case.override) catch {
            if (case.result != null) return error.MemoryLimitMismatch;
            continue;
        };
        if (case.result == null or result != case.result.?) return error.MemoryLimitMismatch;
    }
    for (parsed.value.caches) |case| {
        if (try case.memory.cacheBytes(case.tokens) != case.cache or try case.memory.growthBytes(case.tokens, case.in_flight) != case.growth or try case.memory.needed(case.tokens, case.request) != case.needed or try case.memory.largestContext(case.window, case.budget, case.request) != case.largest) return error.CacheMemoryMismatch;
    }
    for (parsed.value.streams) |case| {
        var admission = Admission{ .budget = case.budget, .memory = case.memory, .lanes = case.lanes };
        if (try case.memory.streamBytes(case.tokens) != case.stream or try case.memory.prefillBytes(case.prompt) != case.prefill or try admission.projected(case.used, case.prompt, case.tokens, case.live) != case.projected or try admission.admits(case.used, case.prompt, case.tokens, case.live) != case.admits or try admission.fitting(case.used, case.tokens) != case.fitting or admission.refused != @intFromBool(!case.admits)) return error.StreamMemoryMismatch;
    }
    for (parsed.value.budgets) |case| if (concurrentBudget(case.ram, case.fraction, case.process, case.share, case.elsewhere) != case.result) return error.ConcurrentBudgetMismatch;
    for (parsed.value.reservations) |case| {
        const longest = case.prompt + reserveReply(case.reply, case.horizon);
        try std.testing.expectEqual(case.longest, longest);
        var live: [8]Live = undefined;
        for (case.jobs, case.live, live[0..case.jobs.len]) |job, expected, *actual| {
            actual.* = reserveLive(.{ .now = job.now, .most = job.most }, job.prompt, case.horizon);
            try std.testing.expectEqualDeep(expected, actual.*);
        }
        var admission = Admission{ .budget = case.budget, .memory = case.memory, .lanes = case.lanes };
        try std.testing.expectEqual(case.projected, try admission.projected(case.used, case.prompt, longest, live[0..case.jobs.len]));
        try std.testing.expectEqual(case.fits, try admission.admits(case.used, case.prompt, longest, live[0..case.jobs.len]));
    }
    for (parsed.value.gates) |case| {
        var memory = GateMemory{ .active = case.active, .cache = case.cache, .entries = case.entries };
        var gate = StreamGate{ .per_token = case.per_token, .work = case.work, .budget = 0, .horizon = case.horizon, .lanes = case.lanes };
        for (case.plans) |expected| {
            gate.budget = expected.budget;
            const actual = try gate.plan(&memory, case.streams);
            try std.testing.expectEqualDeep(StreamGate.Plan{ .run = expected.run, .paused = expected.paused, .ended = expected.ended }, actual);
            try std.testing.expectEqual(expected.waits, gate.waits);
            try std.testing.expectEqual(expected.ends, gate.ends);
            try std.testing.expectEqual(expected.active, memory.active);
            try std.testing.expectEqual(expected.cache, memory.cache);
            try std.testing.expectEqual(expected.entries, memory.entries.len);
            try std.testing.expectEqual(expected.reclaims, memory.reclaims);
        }
    }
    std.debug.print("PASS: upstream memory parity: {d} limits, {d} cache projections, {d} stream admission cases, {d} concurrency budgets\n", .{ parsed.value.limits.len, parsed.value.caches.len, parsed.value.streams.len, parsed.value.budgets.len });
    std.debug.print("PASS: {d} upstream stream-growth scenarios with reclaim, pause, termination and recovery\n", .{parsed.value.gates.len});
    std.debug.print("PASS: {d} upstream scheduler rolling reservations and admission decisions\n", .{parsed.value.reservations.len});
}

const GateMemory = struct {
    const Entry = struct { size: u64, released: u64 };
    active: u64,
    cache: u64,
    entries: []const Entry,
    reclaims: u64 = 0,

    pub fn used(m: *GateMemory) !u64 {
        return m.active + m.cache;
    }
    pub fn freeable(m: *GateMemory) !u64 {
        var total = m.cache;
        for (m.entries) |entry| total += entry.size;
        return total;
    }
    pub fn reclaim(m: *GateMemory) !bool {
        m.reclaims += 1;
        if (m.cache > 0) {
            m.cache = 0;
            return true;
        }
        if (m.entries.len == 0) return false;
        m.active -= m.entries[0].released;
        m.entries = m.entries[1..];
        return true;
    }
};

test "repeated memory profiles retain every observed worst case" {
    const first = StreamMemory{ .short_tokens = 64, .long_tokens = 2112, .short = 100, .long = 300, .per_token = 1, .prefill_a = 10, .prefill_b = 0.5, .round_bytes = 500 };
    var other = first;
    other.short = 120;
    other.long = 200;
    other.per_token = 2;
    other.prefill_a = 5;
    other.prefill_b = 0.7;
    other.round_bytes = 400;
    var combined = first;
    try combined.include(other);
    for ([_]u64{ 0, 1, 64, 128, 2048, 2112, 8192, 262144 }) |tokens| {
        try std.testing.expect(try combined.streamBytes(tokens) >= try first.streamBytes(tokens));
        try std.testing.expect(try combined.streamBytes(tokens) >= try other.streamBytes(tokens));
        try std.testing.expect(try combined.prefillBytes(tokens) >= try first.prefillBytes(tokens));
        try std.testing.expect(try combined.prefillBytes(tokens) >= try other.prefillBytes(tokens));
    }
    try std.testing.expectEqual(first.round_bytes, combined.round_bytes);
    other.chunk = 16;
    try std.testing.expectError(error.IncompatibleMemoryProbes, combined.include(other));
}

test "empty stream gate neither reclaims nor counts a wait" {
    var memory = GateMemory{ .active = 100, .cache = 100, .entries = &.{} };
    var gate = StreamGate{ .per_token = 1, .work = 100, .budget = 0 };
    try std.testing.expectEqualDeep(StreamGate.Plan{ .run = 0, .paused = 0 }, try gate.plan(&memory, &.{}));
    try std.testing.expectEqual(@as(u64, 0), memory.reclaims);
}

test "rolling reservations keep unfinished prompts and never reserve past the reply limit" {
    const t = std.testing;
    try t.expectEqualDeep(Live{ .now = 1024, .most = 12048 }, reserveLive(.{ .now = 1024, .most = 100000 }, 10000, growth_horizon));
    try t.expectEqualDeep(Live{ .now = 1024, .most = 10010 }, reserveLive(.{ .now = 1024, .most = 10010 }, 10000, growth_horizon));
    try t.expectEqualDeep(Live{ .now = 10000, .most = 12048 }, reserveLive(.{ .now = 10000, .most = 100000 }, 10000, growth_horizon));
    try t.expectEqualDeep(Live{ .now = 10999, .most = 11000 }, reserveLive(.{ .now = 10999, .most = 11000 }, 10000, growth_horizon));
    const maximum = std.math.maxInt(u64);
    try t.expectEqualDeep(Live{ .now = maximum - 1, .most = maximum }, reserveLive(.{ .now = maximum - 1, .most = maximum }, 10000, growth_horizon));
}

test "decode gate reserves retained-prefix copies only for running streams" {
    var memory = GateMemory{ .active = 100, .cache = 0, .entries = &.{} };
    var gate = StreamGate{ .budget = 200, .per_token = 0, .work = 0 };
    const streams = [_]Live{ .{ .now = 100, .most = 1000 }, .{ .now = 100, .most = 1000, .copy_bytes = 200 } };
    try std.testing.expectEqualDeep(StreamGate.Plan{ .run = 1, .paused = 1 }, try gate.plan(&memory, &streams));
    try std.testing.expectEqual(@as(u64, 0), memory.reclaims);
    gate.budget = 300;
    try std.testing.expectEqualDeep(StreamGate.Plan{ .run = 2, .paused = 0 }, try gate.plan(&memory, &streams));
}

test "prefill rounds reserve their chunk and the decode growth already scheduled beside it" {
    const admission = Admission{ .budget = 320, .memory = .{ .short_tokens = 0, .short = 0, .long_tokens = 1, .long = 0, .chunk = 4, .per_token = 10, .prefill_a = 2, .prefill_b = 0, .round_bytes = 50 } };
    const decoding = [_]Live{.{ .now = 100, .most = 103, .copy_bytes = 20 }};
    try std.testing.expectEqual(@as(u64, 320), try admission.prefillProjected(100, 10, 8, 100, &decoding));
    try std.testing.expect(try admission.prefillProjected(100, 10, 4, 100, &decoding) > admission.budget);
    try std.testing.expect(try admission.prefillProjected(100, 10, 4, 100, &.{}) <= admission.budget);
}

test "memory accounting rejects overflow and malformed profiles" {
    const t = std.testing;
    const cache = CacheMemory{ .fixed_bytes = 1, .bytes_per_token = 2 };
    try t.expectError(error.Overflow, cache.cacheBytes(std.math.maxInt(u64)));
    try t.expectError(error.InvalidMemoryStep, (CacheMemory{ .fixed_bytes = 0, .bytes_per_token = 0, .step = 0 }).cacheBytes(0));
    try t.expectError(error.InvalidCacheCopies, cache.needed(1, .{ .resident_bytes = 0, .cache_copies = 0 }));
    const memory = StreamMemory{ .short_tokens = 64, .short = 0, .long_tokens = 64, .long = 0, .per_token = 0, .prefill_a = 0, .prefill_b = 0, .round_bytes = 0 };
    try t.expectError(error.InvalidMemoryProfile, memory.streamBytes(0));
    try t.expectError(error.InvalidMemoryBudget, limit(0, 0, 0.7, null));
}
