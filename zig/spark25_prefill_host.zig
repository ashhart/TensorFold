//! Real Spark2.5 cancelled-prefill snapshots and restores with deterministic cache rows, not checkpoint arithmetic.
const std = @import("std");
const spark = @import("spark25");
const cuda = @import("cuda");
const lanes = @import("lanes");
const pc = @import("engine_api").prompt_cache;
const fake = @import("gemma4_page_driver.zig");
const a = std.testing.allocator;
const t = std.testing;
const cfg = spark.config;
const fwd = spark.forward;
const be = spark.backend;
const geo: cfg.Geometry = .{ .hidden = 64, .layers = 4, .intermediate = 128, .query_heads = 1, .kv_heads = 1 };

const Fixture = struct {
    d: cuda.Driver,
    m: fwd.Model,
    g: be.Spark,
    k: spark.kernels.Kernels = undefined,

    fn init(f: *Fixture) !void {
        f.d = fake.driver();
        const queue: cuda.Stream = .{ .d = &f.d, .handle = @ptrFromInt(1) };
        f.m = .{ .gpa = a, .d = &f.d, .w = undefined, .geo = geo, .layers = geo.layers, .context = 4096, .slots = 2, .kc = &.{}, .vc = &.{}, .inv_sliding = 0, .inv_full = 0, .s = undefined, .paged = false };
        errdefer f.m.deinit();
        f.m.kc = try a.alloc(u64, geo.layers);
        f.m.vc = try a.alloc(u64, geo.layers);
        for (0..geo.layers) |i| {
            f.m.kc[i] = try f.cache(i);
            f.m.vc[i] = try f.cache(i);
        }
        f.g = try be.Spark.init(a, &f.d, &f.m, &f.k, queue);
    }

    fn cache(f: *Fixture, layer: usize) !u64 {
        const bytes = f.m.slots * f.m.cacheLen(layer) * f.m.geo.kvWidth() * 2;
        var b = try cuda.DeviceBuffer.alloc(&f.d, bytes);
        errdefer b.free();
        try f.m.buffers.append(a, b);
        return b.ptr;
    }

    fn deinit(f: *Fixture) void {
        f.g.deinit();
        f.m.deinit();
    }

    fn release(f: *Fixture, s: *lanes.Stream) void {
        const b = f.g.backend();
        b.vtable.release(b.ptr, s);
    }
};

fn keep(ctx: *anyopaque, s: *lanes.Stream, at: u32) void {
    const store: *pc.Store = @ptrCast(@alignCast(ctx));
    _ = store.keep(s.prompt(), at, s, &.{}, &.{});
}

fn cancelled(_: *anyopaque) bool {
    return true;
}

fn cancelNow(_: *anyopaque) bool {
    return false;
}

fn stream(ids: []const u32, store: *pc.Store) !lanes.Stream {
    return lanes.Stream.init(a, .{ .id = "fixture", .prompt = ids, .max_new = 16, .drafts = false, .reuse = .{ .hook = .{ .ptr = store, .at = keep, .interrupted = keep } }, .cancel_check = .{ .ptr = store, .check = cancelNow } });
}

fn prompt(scrub: bool) ![]u32 {
    const out = try a.alloc(u32, 2179);
    for (out, 0..) |*x, i| x.* = @intCast(if (scrub) i % 89 + 500 else i % 97 + 3);
    return out;
}

fn address(m: *const fwd.Model, cache: []const u64, layer: usize, slot: usize, position: usize) u64 {
    const rb = m.geo.kvWidth() * 2;
    const row = slot * m.cacheLen(layer) + position % m.cacheLen(layer);
    return cache[layer] + row * rb;
}

fn value(token: u32, position: usize, layer: usize, v: bool) u16 {
    return @truncate(token *% 73 +% @as(u32, @truncate(position)) *% 17 +% @as(u32, @intCast(layer)) *% 101 +% @as(u32, @intFromBool(v)) *% 19);
}

fn marked(s: *lanes.Stream, end: usize) void {
    if (s.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) hk.at(hk.ptr, s, @intCast(end));
}

fn advance(f: *Fixture, s: *lanes.Stream, l: anytype, end: usize) !void {
    const start = l.fill.?;
    for (start..end) |pos| for (0..f.m.layers) |i| for ([_][]u64{ f.m.kc, f.m.vc }, 0..) |cache, v| {
        const row: [*]u16 = @ptrFromInt(address(&f.m, cache, i, l.slot, pos));
        @memset(row[0..f.m.geo.kvWidth()], value(s.prompt()[pos], pos, i, v == 1));
    };
    l.fill = if (end == s.prompt().len) null else end;
    if (end == s.prompt().len) l.len = end;
    s.fill_at.store(@intCast(end), .monotonic);
    marked(s, end);
}

fn fill(f: *Fixture, s: *lanes.Stream) !u64 {
    const l = try f.g.opening(s);
    while (l.fill) |start| try advance(f, s, l, @min(s.prompt().len, (start / fwd.prompt_rows + 1) * fwd.prompt_rows));
    return digest(&f.m, l.slot, s.prompt().len);
}

fn digest(m: *const fwd.Model, slot: usize, at: usize) u64 {
    var hash = std.hash.Wyhash.init(0);
    for (0..m.layers) |i| {
        const first = if (cfg.full(i)) 0 else at - @min(at, cfg.window - 1);
        for (first..at) |pos| for ([_][]u64{ m.kc, m.vc }) |cache| {
            const row: [*]const u8 = @ptrFromInt(address(m, cache, i, slot, pos));
            hash.update(row[0 .. m.geo.kvWidth() * 2]);
        };
    }
    return hash.final();
}

fn reply(hash: u64) [16]u32 {
    var prng = std.Random.DefaultPrng.init(hash);
    var out: [16]u32 = undefined;
    for (&out) |*x| x.* = prng.random().uintLessThan(u32, cfg.vocab);
    return out;
}

const Outcome = struct {
    fresh: [16]u32,
    resumed: [16]u32,
    stopped: u64,
    restored: u64,
    cached: u64,
    identity: bool,
    needed: u64,
    entries: usize,
    held: u64,
    refused: u64,
};

fn resumedAfterCancel(between_steps: bool, shortage: u64) !Outcome {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const needed = spark.native.snap_bytes(&f.g, fwd.prompt_rows);
    var store = pc.Store.init(a, .{ .ptr = &f.g, .vtable = &.{ .bytes = spark.native.snap_bytes, .save = spark.native.snap_save, .restore = spark.native.snap_restore, .drop = spark.native.snap_drop } }, .{ .lookahead = 0, .min_prompt = 0, .chunk = fwd.prompt_rows }, needed -| shortage);
    defer store.deinit();
    const ids = try prompt(false);
    defer a.free(ids);
    const scrub = try prompt(true);
    defer a.free(scrub);
    var plain = try stream(ids, &store);
    defer plain.deinit(a);
    const fresh = reply(try fill(&f, &plain));
    f.release(&plain);

    var stopped = try stream(ids, &store);
    defer stopped.deinit(a);
    const l = try f.g.opening(&stopped);
    try advance(&f, &stopped, l, fwd.prompt_rows);
    const stopped_state = digest(&f.m, l.slot, fwd.prompt_rows);
    const b = f.g.backend();
    if (between_steps) {
        if (b.vtable.cancel_prefill) |keep_prefix| try keep_prefix(b.ptr, &stopped);
        f.release(&stopped);
    } else {
        stopped.cancel_check = .{ .ptr = &stopped, .check = cancelled };
        try t.expectError(error.Cancelled, b.vtable.prefill_step.?(b.ptr, &stopped, 0));
    }

    var other = try stream(scrub, &store);
    defer other.deinit(a);
    _ = try fill(&f, &other);
    f.release(&other);
    const found = store.find(ids, &.{}, &.{});
    var resumed = try stream(ids, &store);
    defer resumed.deinit(a);
    if (found) |entry| {
        resumed.reuse.saved = entry.saved;
        resumed.reuse.at = entry.at;
    }
    const rl = try f.g.opening(&resumed);
    const identity = if (found) |entry| resumed.reuse.saved == entry.saved else false;
    const restored = digest(&f.m, rl.slot, rl.fill.?);
    while (rl.fill) |start| try advance(&f, &resumed, rl, @min(ids.len, (start / fwd.prompt_rows + 1) * fwd.prompt_rows));
    const got = reply(digest(&f.m, rl.slot, ids.len));
    f.release(&resumed);
    return .{ .fresh = fresh, .resumed = got, .stopped = stopped_state, .restored = restored, .cached = resumed.cached, .identity = identity, .needed = needed, .entries = store.entries.items.len, .held = store.held, .refused = store.counts.refused };
}

test "Spark25 prefill: cancelled chunks retain exact fresh replies within cache capacity" {
    for ([_]bool{ false, true }) |between_steps| {
        for ([_]u64{ 0, 1 }) |shortage| {
            const out = try resumedAfterCancel(between_steps, shortage);
            const fits = shortage == 0;
            try t.expectEqualSlices(u32, &out.fresh, &out.resumed);
            try t.expectEqual(if (fits) @as(u64, fwd.prompt_rows) else 0, out.cached);
            try t.expectEqual(fits, out.identity);
            try t.expectEqual(@as(usize, @intFromBool(fits)), out.entries);
            try t.expectEqual(if (fits) out.needed else 0, out.held);
            try t.expectEqual(@as(u64, @intFromBool(!fits)), out.refused);
            if (fits) try t.expectEqual(out.stopped, out.restored);
            try t.expect(fake.empty());
        }
    }
}
