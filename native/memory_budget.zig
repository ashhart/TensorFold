const std = @import("std");
pub const gib = 1024 * 1024 * 1024;
pub const process_bytes = 3 * gib;

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

pub const Live = struct { now: u64, most: u64 };
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
        limits: []const struct { ram: u64, recommended: u64, fraction: f64, override: ?[]const u8, result: ?u64 },
        caches: []const struct { memory: CacheMemory, tokens: u64, in_flight: u64, request: CacheMemory.Request, budget: u64, window: u64, cache: u64, growth: u64, needed: u64, largest: u64 },
        streams: []const struct { memory: StreamMemory, lanes: u64, tokens: u64, prompt: u64, used: u64, budget: u64, live: []const Live, stream: u64, prefill: u64, projected: u64, admits: bool, fitting: u64 },
        budgets: []const struct { ram: u64, fraction: f64, process: u64, share: u64, elsewhere: u64, result: u64 },
    };
    const parsed = try std.json.parseFromSlice(Fixture, a, source, .{});
    defer parsed.deinit();
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
    std.debug.print("PASS: upstream memory parity: {d} limits, {d} cache projections, {d} stream admission cases, {d} concurrency budgets\n", .{ parsed.value.limits.len, parsed.value.caches.len, parsed.value.streams.len, parsed.value.budgets.len });
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
