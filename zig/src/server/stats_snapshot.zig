//! The dashboard's stats snapshot: live counts, memory, caches and prefill reuse as one JSON document.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub fn statsSnapshot(a: Allocator, status: api.Status, memory: ?api.Memory, prompt: u64, generation: u64, drafted: u64, accepted: u64, rounds: u64, cached: u64, preemptions: u64, requeued: u64, work: ?api.load_ledger.Work, model: []const u8, version: []const u8, backend: []const u8, device: []const u8, latency_ms: ?f64, ttft_ms: ?f64) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "running", try json.intValue(a, status.running));
    try o.put(a, "waiting", try json.intValue(a, status.waiting));
    try o.put(a, "decode_tokens_per_second", .{ .float = status.decode_tokens_per_second });
    try o.put(a, "prefill_tokens_per_second", .{ .float = status.prefill_tokens_per_second });
    try o.put(a, "generation_tokens_running", try json.intValue(a, status.generation_tokens));
    if (status.fill_prompt_tokens != null) {
        try o.put(a, "fill_prompt_tokens", try json.intValue(a, status.fill_prompt_tokens.?));
        try o.put(a, "fill_tokens_left", try json.intValue(a, status.fill_tokens_left orelse 0));
        try o.put(a, "fill_run", try json.intValue(a, status.fill_run));
    } else if (status.fill_unknown) {
        try o.put(a, "fill_unknown", .{ .bool = true }); // prefilling, but this backend reports no progress
        try o.put(a, "fill_run", try json.intValue(a, status.fill_run));
    }
    try o.put(a, "prompt_tokens_total", try json.intValue(a, prompt));
    try o.put(a, "cached_tokens_total", try json.intValue(a, cached));
    try o.put(a, "preemptions_total", try json.intValue(a, preemptions));
    try o.put(a, "requests_requeued", try json.intValue(a, requeued));
    if (latency_ms) |ms| try o.put(a, "latency_p50_ms", .{ .float = ms });
    if (ttft_ms) |ms| try o.put(a, "ttft_p50_ms", .{ .float = ms });
    try o.put(a, "model", .{ .string = model });
    try o.put(a, "version", .{ .string = version });
    try o.put(a, "backend", .{ .string = backend });
    try o.put(a, "device", .{ .string = device });
    if (work) |w| {
        const reuse = try json.newObject(a);
        try reuse.put(a, "prefill_cache", try json.intValue(a, w.prefill_cache));
        try reuse.put(a, "prefill_compute", try json.intValue(a, w.prefill_compute));
        const total = w.prefill_cache + w.prefill_compute;
        const ratio = if (total == 0) null else @as(f64, @floatFromInt(w.prefill_cache)) / @as(f64, @floatFromInt(total));
        if (ratio) |rf| try reuse.put(a, "ratio", .{ .float = rf });
        try o.put(a, "prefill_reuse", .{ .object = reuse });
    }
    if (status.prompt_cache) |c| try o.put(a, "prompt_cache", try promptCacheValue(a, c));
    if (status.disk) |d| try o.put(a, "disk", try diskCacheValue(a, d));
    if (status.fsr) |f| try o.put(a, "fsr", try fsrCacheValue(a, f));
    try o.put(a, "generation_tokens_total", try json.intValue(a, generation));
    try o.put(a, "mtp_drafted_total", try json.intValue(a, drafted));
    try o.put(a, "mtp_accepted_total", try json.intValue(a, accepted));
    try o.put(a, "decode_rounds_total", try json.intValue(a, rounds));
    const mem = try json.newObject(a);
    if (memory) |m| {
        try mem.put(a, "active", try json.intValue(a, m.active));
        try mem.put(a, "cache", try json.intValue(a, m.cache));
        try mem.put(a, "peak", try json.intValue(a, m.peak));
        if (m.pinned != 0) try mem.put(a, "pinned", try json.intValue(a, m.pinned));
        if (m.limit) |limit| try mem.put(a, "limit", try json.intValue(a, limit));
    }
    try o.put(a, "memory", .{ .object = mem });
    return .{ .object = o };
}

fn promptCacheValue(a: Allocator, c: api.Status.PromptCache) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "states", try json.intValue(a, c.states));
    try o.put(a, "bytes", try json.intValue(a, c.bytes));
    try o.put(a, "budget_bytes", try json.intValue(a, c.budget_bytes));
    try o.put(a, "spare_bytes", try json.intValue(a, c.spare_bytes));
    try o.put(a, "hold", .{ .bool = c.hold });
    try o.put(a, "hits", try json.intValue(a, c.hits));
    try o.put(a, "misses", try json.intValue(a, c.misses));
    try o.put(a, "kept", try json.intValue(a, c.kept));
    try o.put(a, "evicted", try json.intValue(a, c.evicted));
    try o.put(a, "refused", try json.intValue(a, c.refused));
    try o.put(a, "failed", try json.intValue(a, c.failed));
    try o.put(a, "last_access_ms", try json.intValue(a, c.last_access_ms));
    var list: json.Value = .{ .array = try a.alloc(json.Value, @intCast(@min(c.entry_count, 64))) };
    for (0..list.array.len) |i| {
        const e = c.entries[i];
        const entry = try json.newObject(a);
        try entry.put(a, "id", .{ .string = try a.dupe(u8, &std.fmt.bytesToHex(e.id, .lower)) });
        try entry.put(a, "at", try json.intValue(a, e.at));
        try entry.put(a, "bytes", try json.intValue(a, e.bytes));
        try entry.put(a, "shared", .{ .bool = e.shared });
        try entry.put(a, "branched", .{ .bool = e.branched });
        try entry.put(a, "borrowers", try json.intValue(a, e.borrowers));
        try entry.put(a, "last_access_ms", try json.intValue(a, e.last_access_ms));
        try entry.put(a, "created_at_ms", try json.intValue(a, e.created_at_ms));
        try entry.put(a, "hits", try json.intValue(a, e.hits));
        list.array[i] = .{ .object = entry };
    }
    try o.put(a, "entries", list);
    return .{ .object = o };
}

fn diskCacheValue(a: Allocator, d: api.Status.DiskCache) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "states", try json.intValue(a, d.states));
    try o.put(a, "held_bytes", try json.intValue(a, d.held_bytes));
    try o.put(a, "cap_bytes", try json.intValue(a, d.cap_bytes));
    try o.put(a, "daily_cap_bytes", try json.intValue(a, d.daily_cap_bytes));
    try o.put(a, "queue_len", try json.intValue(a, d.queue_len));
    try o.put(a, "read_rate_bps", .{ .float = d.read_rate_bps });
    try o.put(a, "read_ms_last", .{ .float = d.read_ms_last });
    try o.put(a, "read_ms_ema", .{ .float = d.read_ms_ema });
    try o.put(a, "write_ms_last", .{ .float = d.write_ms_last });
    try o.put(a, "write_ms_ema", .{ .float = d.write_ms_ema });
    try o.put(a, "read_ms_p50", .{ .float = d.read_ms_p50 });
    try o.put(a, "read_ms_p95", .{ .float = d.read_ms_p95 });
    try o.put(a, "write_ms_p50", .{ .float = d.write_ms_p50 });
    try o.put(a, "write_ms_p95", .{ .float = d.write_ms_p95 });
    try o.put(a, "last_read_ms", try json.intValue(a, d.last_read_ms));
    try o.put(a, "last_write_ms", try json.intValue(a, d.last_write_ms));
    try o.put(a, "written_bytes", try json.intValue(a, d.written_bytes));
    try o.put(a, "read_bytes", try json.intValue(a, d.read_bytes));
    try o.put(a, "reads", try json.intValue(a, d.reads));
    try o.put(a, "writes", try json.intValue(a, d.writes));
    try o.put(a, "skipped", try json.intValue(a, d.skipped));
    try o.put(a, "failed", try json.intValue(a, d.failed));
    try o.put(a, "evicted", try json.intValue(a, d.evicted));
    try o.put(a, "verified", try json.intValue(a, d.verified));
    try o.put(a, "differed", try json.intValue(a, d.differed));
    try o.put(a, "failure_count", try json.intValue(a, d.failure_count));
    if (d.last_error.len > 0) try o.put(a, "last_error", .{ .string = d.last_error });
    try o.put(a, "admission_retry_at", try json.intValue(a, d.admission_retry_at));
    try o.put(a, "imprint_states", try json.intValue(a, d.imprint_states));
    try o.put(a, "imprint_bytes", try json.intValue(a, d.imprint_bytes));
    try o.put(a, "prefetch_made", try json.intValue(a, d.prefetch_made));
    try o.put(a, "prefetch_used", try json.intValue(a, d.prefetch_used));
    try o.put(a, "prefetch_freed", try json.intValue(a, d.prefetch_freed));
    try o.put(a, "prefetch_failed", try json.intValue(a, d.prefetch_failed));
    try o.put(a, "prefetch_cancelled", try json.intValue(a, d.prefetch_cancelled));
    try o.put(a, "prefetch_timeouts", try json.intValue(a, d.prefetch_timeouts));
    try o.put(a, "prefetch_held_reads", try json.intValue(a, d.prefetch_held_reads));
    try o.put(a, "prefetch_held_bytes", try json.intValue(a, d.prefetch_held_bytes));
    return .{ .object = o };
}

fn fsrCacheValue(a: Allocator, f: api.Status.FsrCache) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "io_bytes", try json.intValue(a, f.io_bytes));
    try o.put(a, "io_calls", try json.intValue(a, f.io_calls));
    try o.put(a, "io_size_checks", try json.intValue(a, f.io_size_checks));
    try o.put(a, "io_logical_bytes", try json.intValue(a, f.io_logical_bytes));
    try o.put(a, "demand_misses", try json.intValue(a, f.demand_misses));
    try o.put(a, "forecast_misses", try json.intValue(a, f.forecast_misses));
    try o.put(a, "demand_keys", try json.intValue(a, f.demand_keys));
    try o.put(a, "unique_demand_keys", try json.intValue(a, f.unique_demand_keys));
    try o.put(a, "worker_joins", try json.intValue(a, f.worker_joins));
    try o.put(a, "cancelled_forecasts", try json.intValue(a, f.cancelled_forecasts));
    try o.put(a, "occupancy", try json.intValue(a, f.occupancy));
    try o.put(a, "capacity", try json.intValue(a, f.capacity));
    try o.put(a, "cache_hits", try json.intValue(a, f.cache_hits));
    try o.put(a, "evictions", try json.intValue(a, f.evictions));
    return .{ .object = o };
}

test "stats snapshot exposes counters and omits unknown memory" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const value = try statsSnapshot(arena.allocator(), .{ .running = 1, .generation_tokens = 2 }, null, 3, 4, 5, 6, 7, 8, 9, 10, null, "m", "v", "b", "d", null, null);
    const object = value.object;
    for ([_][]const u8{ "running", "waiting", "generation_tokens_running", "prompt_tokens_total", "generation_tokens_total", "mtp_drafted_total", "mtp_accepted_total", "decode_rounds_total", "memory" }) |key|
        try std.testing.expect(object.contains(key));
    try std.testing.expectEqual(@as(usize, 0), object.get("memory").?.object.count());
    try std.testing.expect(object.get("fill_prompt_tokens") == null); // no fill in progress: the keys stay out
}

test "stats snapshot carries the filling prompt's tokens while a prefill runs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const status: api.Status = .{ .fill_prompt_tokens = 900, .fill_tokens_left = 300 };
    const value = try statsSnapshot(arena.allocator(), status, null, 0, 0, 0, 0, 0, 0, 0, 0, null, "m", "v", "b", "d", null, null);
    const object = value.object;
    try std.testing.expectEqual(@as(i64, 900), object.get("fill_prompt_tokens").?.int64().?);
    try std.testing.expectEqual(@as(i64, 300), object.get("fill_tokens_left").?.int64().?);
}
test "stats snapshot marks an unknown-progress fill without counters" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const status: api.Status = .{ .fill_unknown = true, .fill_run = 7 };
    const value = try statsSnapshot(arena.allocator(), status, null, 0, 0, 0, 0, 0, 0, 0, 0, null, "m", "v", "b", "d", null, null);
    const object = value.object;
    try std.testing.expect(object.get("fill_unknown").?.bool);
    try std.testing.expectEqual(@as(i64, 7), object.get("fill_run").?.int64().?);
    try std.testing.expect(object.get("fill_prompt_tokens") == null); // no counters: the keys stay out
}

test "stats snapshot's memory carries a configured limit and omits none" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const with = try statsSnapshot(
        arena.allocator(),
        .{},
        .{ .active = 5, .cache = 1, .peak = 9, .pinned = 2, .limit = 100 },
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        null,
        "m",
        "v",
        "b",
        "d",
        null,
        null,
    );
    const memory = with.object.get("memory").?.object;
    try std.testing.expectEqual(@as(i64, 5), memory.get("active").?.int64().?);
    try std.testing.expectEqual(@as(i64, 9), memory.get("peak").?.int64().?);
    try std.testing.expectEqual(@as(i64, 100), memory.get("limit").?.int64().?);
    const without = try statsSnapshot(arena.allocator(), .{}, .{ .active = 5, .peak = 9 }, 0, 0, 0, 0, 0, 0, 0, 0, null, "m", "v", "b", "d", null, null);
    // no limit configured: the key stays out
    try std.testing.expect(without.object.get("memory").?.object.get("limit") == null);
}

test "stats snapshot carries cache, reuse, identity, preemptions and medians, and hides absent caches" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var status: api.Status = .{
        .running = 1,
        .prompt_cache = .{
            .states = 3,
            .bytes = 4,
            .budget_bytes = 10,
            .spare_bytes = 2,
            .hits = 5,
            .misses = 1,
            .kept = 6,
            .entry_count = 1,
        },
        .disk = .{ .states = 2, .held_bytes = 8, .read_ms_ema = 3.5, .read_ms_p50 = 5, .read_ms_p95 = 100, .last_read_ms = 999, .failure_count = 1, .last_error = "WrongRows", .prefetch_made = 4 },
        .fsr = .{ .io_bytes = 99, .occupancy = 7, .capacity = 32, .cache_hits = 11, .evictions = 3 },
    };
    status.prompt_cache.?.entries[0] = .{ .id = @splat(0xab), .at = 256, .bytes = 4, .last_access_ms = 123456, .created_at_ms = 111, .hits = 7 };
    const value = try statsSnapshot(a, status, .{ .active = 1 }, 0, 0, 0, 0, 0, 777, 42, 5, .{ .prefill_cache = 30, .prefill_compute = 70 }, "model-a", "v1", "cuda", "gpu", 250.0, 8.0);
    const o = value.object;
    try std.testing.expectEqual(@as(i64, 777), o.get("cached_tokens_total").?.int64().?);
    try std.testing.expectEqual(@as(i64, 42), o.get("preemptions_total").?.int64().?);
    try std.testing.expectEqual(@as(i64, 5), o.get("requests_requeued").?.int64().?);
    try std.testing.expectEqualStrings("model-a", o.get("model").?.string);
    try std.testing.expectEqualStrings("v1", o.get("version").?.string);
    try std.testing.expectEqualStrings("cuda", o.get("backend").?.string);
    try std.testing.expectEqualStrings("gpu", o.get("device").?.string);
    try std.testing.expectEqual(@as(f64, 250.0), o.get("latency_p50_ms").?.float);
    try std.testing.expectEqual(@as(f64, 8.0), o.get("ttft_p50_ms").?.float);
    const reuse = o.get("prefill_reuse").?.object;
    try std.testing.expectEqual(@as(i64, 30), reuse.get("prefill_cache").?.int64().?);
    try std.testing.expectEqual(@as(i64, 70), reuse.get("prefill_compute").?.int64().?);
    try std.testing.expectEqual(@as(f64, 0.3), reuse.get("ratio").?.float);
    const cache = o.get("prompt_cache").?.object;
    try std.testing.expectEqual(@as(i64, 5), cache.get("hits").?.int64().?);
    try std.testing.expectEqual(@as(i64, 10), cache.get("budget_bytes").?.int64().?);
    const entries = cache.get("entries").?.array;
    try std.testing.expectEqual(@as(usize, 1), entries.len);
    try std.testing.expectEqual(@as(i64, 123456), entries[0].object.get("last_access_ms").?.int64().?);
    try std.testing.expectEqual(@as(i64, 111), entries[0].object.get("created_at_ms").?.int64().?);
    try std.testing.expectEqual(@as(i64, 7), entries[0].object.get("hits").?.int64().?);
    try std.testing.expectEqualStrings("abababababababababababababababab", entries[0].object.get("id").?.string);
    const disk = o.get("disk").?.object;
    try std.testing.expectEqual(@as(i64, 2), disk.get("states").?.int64().?);
    try std.testing.expectEqual(@as(f64, 3.5), disk.get("read_ms_ema").?.float);
    try std.testing.expectEqual(@as(f64, 5), disk.get("read_ms_p50").?.float);
    try std.testing.expectEqual(@as(f64, 100), disk.get("read_ms_p95").?.float);
    try std.testing.expectEqual(@as(i64, 999), disk.get("last_read_ms").?.int64().?);
    try std.testing.expectEqualStrings("WrongRows", disk.get("last_error").?.string);
    const fsr = o.get("fsr").?.object;
    try std.testing.expectEqual(@as(i64, 99), fsr.get("io_bytes").?.int64().?);
    try std.testing.expectEqual(@as(i64, 7), fsr.get("occupancy").?.int64().?);
}

test "stats snapshot hides absent caches and the reuse ratio when nothing is counted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const value = try statsSnapshot(arena.allocator(), .{}, null, 0, 0, 0, 0, 0, 0, 0, 0, null, "m", "v", "b", "d", null, null);
    const o = value.object;
    try std.testing.expect(o.get("prompt_cache") == null);
    try std.testing.expect(o.get("disk") == null);
    try std.testing.expect(o.get("fsr") == null);
    try std.testing.expect(o.get("prefill_reuse") == null);
    try std.testing.expect(o.get("latency_p50_ms") == null);
    try std.testing.expect(o.get("ttft_p50_ms") == null);
}
