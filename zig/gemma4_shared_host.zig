//! Real Gemma leases, snapshots and prompt adapter with a deterministic row oracle, not checkpoint arithmetic.
const std = @import("std");
const g4 = @import("gemma4");
const cuda = @import("cuda");
const lanes = @import("lanes");
const adapter = @import("prompt_pages");
const fake = @import("gemma4_page_driver.zig");
const a = std.testing.allocator;
const t = std.testing;
const cfg = g4.config;
const fwd = g4.forward;
const be = g4.backend;
const marks = [_]u32{ 512, 1024, 1536, 2048 };
const geo: cfg.Geometry = .{ .name = "fixture", .hidden = 64, .layers = 6, .intermediate = 128, .query_heads = 2, .sliding_kv_heads = 1, .global_kv_heads = 1 };

const Fixture = struct {
    d: cuda.Driver,
    m: fwd.Model,
    g: be.Gemma,
    k: g4.kernels.Kernels = undefined,
    store: ?*adapter.Adapter = null,

    fn init(f: *Fixture, paged: bool, share: bool) !void {
        return f.initWith(paged, share, null);
    }
    fn initWith(f: *Fixture, paged: bool, share: bool, assistant: ?*g4.assistant.Drafter) !void {
        f.store = null;
        f.d = fake.driver();
        const queue: cuda.Stream = .{ .d = &f.d, .handle = @ptrFromInt(1) };
        f.m = .{ .gpa = a, .d = &f.d, .w = undefined, .geo = geo, .layers = 6, .context = 4096, .slots = if (paged) 1 else 2, .kc = &.{}, .vc = &.{}, .inv_sliding = 0, .inv_global = 0, .s = undefined, .paged = paged };
        errdefer f.m.deinit();
        f.m.kc = try a.alloc(u64, 6);
        f.m.vc = try a.alloc(u64, 6);
        try fwd.pages.build(&f.m, queue);
        f.g = try be.Gemma.init(a, &f.d, &f.m, &f.k, queue);
        errdefer f.g.deinit();
        if (assistant) |x| f.g.relocated = .{ .ctx = x, .call = g4.assistant.Drafter.relocate };
        if (paged) {
            try be.shared.attach(&f.g, 1, 128);
            var prng = std.Random.DefaultPrng.init(0x7a92);
            const pool = &f.g.pool.?;
            prng.random().shuffle(u32, pool.free[0..pool.nfree]);
        }
        if (share) f.store = (try adapter.Adapter.open(g4.native, a, &f.g, fwd.pages.tokens, 8)).?;
    }
    fn deinit(f: *Fixture) void {
        if (f.store) |x| x.close();
        f.g.deinit();
        f.m.deinit();
    }
    fn release(f: *Fixture, s: *lanes.Stream) void {
        const b = f.g.backend();
        b.vtable.release(b.ptr, s);
    }
};

fn stream(ids: []const u32) !lanes.Stream {
    return lanes.Stream.init(a, .{ .id = "fixture", .prompt = ids, .max_new = 16, .drafts = false, .shared_prefixes = &marks });
}

fn prompt() ![]u32 {
    const out = try a.alloc(u32, 2179);
    for (out, 0..) |*x, i| x.* = @intCast(i % 97 + 3);
    return out;
}

fn address(m: *const fwd.Model, cache: []const u64, layer: usize, slot: usize, position: usize) u64 {
    const rb = m.geo.kvHeads(layer) * cfg.headDim(layer) * 2;
    const row = if (cfg.global(layer) and m.paged) @as(usize, m.ids[slot * fwd.pages.perSlot(m) + position / fwd.pages.tokens]) * fwd.pages.tokens + position % fwd.pages.tokens else slot * m.cacheLen(layer) + position % m.cacheLen(layer);
    return cache[layer] + row * rb;
}

fn value(token: u32, position: usize, layer: usize, v: bool) u16 {
    return @truncate(token *% 73 +% @as(u32, @truncate(position)) *% 17 +% @as(u32, @intCast(layer)) *% 101 +% @as(u32, @intFromBool(v)) *% 19);
}

/// Emulates only row writes at the device boundary; the production store performs every keep, map and restore.
fn fill(f: *Fixture, s: *lanes.Stream) !u64 {
    return finish(f, s, try f.g.opening(s));
}

fn finish(f: *Fixture, s: *lanes.Stream, l: anytype) !u64 {
    while (l.fill) |start| {
        const end = @min(s.prompt().len, (start / fwd.prompt_rows + 1) * fwd.prompt_rows);
        for (start..end) |pos| for (0..f.m.layers) |i| for ([_][]u64{ f.m.kc, f.m.vc }, 0..) |cache, v| {
            const row: [*]u16 = @ptrFromInt(address(&f.m, cache, i, l.slot, pos));
            @memset(row[0 .. f.m.geo.kvHeads(i) * cfg.headDim(i)], value(s.prompt()[pos], pos, i, v == 1));
        };
        l.fill = if (end == s.prompt().len) null else end;
        l.len = end;
        s.fill_at.store(@intCast(end), .monotonic);
        f.g.marked(s, end);
    }
    return digest(&f.m, l.slot, s.prompt().len);
}

/// A reply oracle depends on every global row and every live sliding-tail byte, separately for each slot.
fn digest(m: *const fwd.Model, slot: usize, at: usize) u64 {
    var hash = std.hash.Wyhash.init(0);
    for (0..m.layers) |i| {
        const first = if (cfg.global(i)) 0 else at - @min(at, cfg.window - 1);
        for (first..at) |pos| for ([_][]u64{ m.kc, m.vc }) |cache| {
            const row: [*]const u8 = @ptrFromInt(address(m, cache, i, slot, pos));
            hash.update(row[0 .. m.geo.kvHeads(i) * cfg.headDim(i) * 2]);
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

fn idle(f: *Fixture) !void {
    if (f.g.pool) |*p| try t.expect(p.idle());
    try t.expect(f.g.lanes.count() == 0);
}

const Resume = struct { fresh: [16]u32, resumed: [16]u32, cached: u64 };
fn resumed(skip_tail: bool) !Resume {
    var f: Fixture = undefined;
    try f.init(true, true);
    defer f.deinit();
    const ids = try prompt();
    defer a.free(ids);
    var teach = try stream(ids);
    defer teach.deinit(a);
    const cold = reply(try fill(&f, &teach));
    f.release(&teach);
    var blocker = try stream(&.{ 99, 98, 97 });
    defer blocker.deinit(a);
    _ = try fill(&f, &blocker); // slot 0 holds another request while the resumed pass takes slot 1
    defer f.release(&blocker);
    var s = try stream(ids);
    defer s.deinit(a);
    fake.skip_copy = skip_tail;
    _ = try f.g.opening(&s);
    fake.skip_copy = false;
    defer fake.skip_copy = false;
    const got = reply(try finish(&f, &s, f.g.lanes.getPtr(&s).?));
    f.release(&s);
    return .{ .fresh = cold, .resumed = got, .cached = s.cached };
}

test "Gemma shared: resumed reply restores wrapped sliding tails with shared global pages" {
    const out = try resumed(false);
    try t.expectEqual(@as(u64, 2048), out.cached);
    try t.expectEqualSlices(u32, &out.fresh, &out.resumed);
    const bad = try resumed(true);
    try t.expectEqual(@as(u64, 2048), bad.cached);
    try t.expect(!std.mem.eql(u32, &bad.fresh, &bad.resumed));
    try t.expect(fake.empty());
}

test "Gemma shared: concurrent continuations preserve solo replies and a swapped-slot control fails" {
    const ids = try prompt();
    defer a.free(ids);
    const other = try a.dupe(u32, ids);
    defer a.free(other);
    other[2100] = 77;
    var plain: Fixture = undefined;
    try plain.init(false, false);
    var sa = try stream(ids);
    defer sa.deinit(a);
    const oracle_a = reply(try fill(&plain, &sa));
    plain.release(&sa);
    var sb = try stream(other);
    defer sb.deinit(a);
    const oracle_b = reply(try fill(&plain, &sb));
    plain.release(&sb);
    plain.deinit();
    var f: Fixture = undefined;
    try f.init(true, true);
    defer f.deinit();
    var a1 = try stream(ids);
    defer a1.deinit(a);
    const r1 = reply(try fill(&f, &a1));
    var b1 = try stream(other);
    defer b1.deinit(a);
    const r2 = reply(try fill(&f, &b1));
    try t.expectEqualSlices(u32, &oracle_a, &r1);
    try t.expectEqualSlices(u32, &oracle_b, &r2);
    try t.expect(b1.cached > 0);
    const la = f.g.lanes.getPtr(&a1).?;
    const lb = f.g.lanes.getPtr(&b1).?;
    const page = 2100 / fwd.pages.tokens;
    const wrong = f.m.ids[la.slot * fwd.pages.perSlot(&f.m) + page];
    try fwd.pages.map(&f.m, f.g.s, lb.slot, &.{wrong}, page);
    const bad = reply(digest(&f.m, lb.slot, other.len));
    try t.expect(!std.mem.eql(u32, &oracle_b, &bad));
    f.release(&a1);
    f.release(&b1);
    try idle(&f);
    f.store.?.close();
    f.store = null;
    try t.expectEqual(@as(usize, 128), f.g.pool.?.available());
}

fn assistantHash(m: *const fwd.Model, full: cuda.window_attention.Window, sliding: cuda.window_attention.Window, slot: usize, at: usize) u64 {
    var hash = std.hash.Wyhash.init(0);
    for ([_]usize{ 0, 5 }, [_]cuda.window_attention.Window{ sliding, full }) |i, w| {
        const rb = m.geo.kvHeads(i) * cfg.headDim(i) * 2;
        const first = if (i == 5) 0 else at - @min(at, cfg.window - 1);
        for (first..at) |pos| for ([_][]u64{ m.kc, m.vc }) |cache| {
            const row = if (w.pg) |pg| blk: {
                const table: [*]const u32 = @ptrFromInt(pg.table);
                break :blk @as(usize, table[slot * pg.stride + pos / pg.tokens]) * pg.tokens + pos % pg.tokens;
            } else slot * w.ring + pos % w.ring;
            const ptr: [*]const u8 = @ptrFromInt(cache[i] + row * rb);
            hash.update(ptr[0..rb]);
        };
    }
    return hash.final();
}

test "Gemma shared: MTP source windows follow the relocated target and stale graphs are dropped" {
    var f: Fixture = undefined;
    var assistant: g4.assistant.Drafter = undefined;
    assistant.graphs = @splat(@splat(null));
    assistant.graphs[0][0] = .{ .d = &f.d, .handle = @ptrFromInt(123) };
    const before = fake.graphs_dropped;
    try f.initWith(true, true, &assistant);
    defer f.deinit();
    try t.expectEqual(before + 1, fake.graphs_dropped);
    try t.expect(assistant.graphs[0][0] == null);
    assistant.m = &f.m;
    assistant.c.heads = 2;
    assistant.c.sliding[0] = false;
    assistant.c.sliding[1] = true;
    assistant.sources = .{ 0, 5 };
    const full = assistant.sourceWindow(0);
    const sliding = assistant.sourceWindow(1);
    try t.expect(full.pg != null);
    try t.expectEqual(f.m.table, full.pg.?.table);
    try t.expectEqual(fwd.pages.perSlot(&f.m), full.pg.?.stride);
    try t.expect(sliding.pg == null and sliding.ring == fwd.ring);
    const ids = try prompt();
    defer a.free(ids);
    var s = try stream(ids);
    defer s.deinit(a);
    _ = try fill(&f, &s);
    f.release(&s);
    var blocker = try stream(&.{ 99, 98, 97 });
    defer blocker.deinit(a);
    _ = try fill(&f, &blocker);
    defer f.release(&blocker);
    var resumed_stream = try stream(ids);
    defer resumed_stream.deinit(a);
    _ = try fill(&f, &resumed_stream);
    try t.expectEqual(@as(u64, 2048), resumed_stream.cached);
    const lane = f.g.lanes.getPtr(&resumed_stream).?;
    const drafted = reply(assistantHash(&f.m, full, sliding, lane.slot, ids.len));
    var linear: Fixture = undefined;
    try linear.init(false, false);
    defer linear.deinit();
    var fresh = try stream(ids);
    defer fresh.deinit(a);
    _ = try fill(&linear, &fresh);
    assistant.m = &linear.m;
    const plain = reply(assistantHash(&linear.m, assistant.sourceWindow(0), assistant.sourceWindow(1), 0, ids.len));
    try t.expectEqualSlices(u32, &plain, &drafted);
    var stale = full;
    stale.pg = null;
    const bad = reply(assistantHash(&f.m, stale, sliding, lane.slot, ids.len));
    try t.expect(!std.mem.eql(u32, &plain, &bad));
    f.release(&resumed_stream);
    linear.release(&fresh);
}

test "Gemma shared: ring restore precedes forced snapshot eviction in reservation" {
    var f: Fixture = undefined;
    try f.init(true, true);
    defer f.deinit();
    const ids = try prompt();
    defer a.free(ids);
    var teach = try stream(ids);
    defer teach.deinit(a);
    const expected = reply(try fill(&f, &teach));
    f.release(&teach);
    f.store.?.store.limit(.{ .pages = 128, .snaps = 1 });
    try t.expectEqual(@as(u32, 2048), f.store.?.store.find(ids).?.end);
    var blocker = try stream(&.{ 99, 98, 97 });
    defer blocker.deinit(a);
    _ = try fill(&f, &blocker);
    defer f.release(&blocker);
    const pool = &f.g.pool.?;
    const held = try a.alloc(u32, pool.available());
    defer a.free(held);
    try t.expect(pool.hold(held));
    defer for (held) |id| pool.release(id);
    const Reclaim = struct {
        f: *Fixture,
        ids: []const u32,
        restored_first: bool = false,
        fn call(ptr: *anyopaque, want: usize) bool {
            const x: *@This() = @ptrCast(@alignCast(ptr));
            x.restored_first = x.f.g.restored_at == 2048;
            x.f.store.?.store.forget(x.ids, 2048);
            return x.f.g.pool.?.available() >= want;
        }
    };
    var reclaim: Reclaim = .{ .f = &f, .ids = ids };
    const original = pool.reclaim;
    pool.reclaim = .{ .ctx = &reclaim, .call = Reclaim.call };
    defer pool.reclaim = original;
    var s = try stream(ids);
    defer s.deinit(a);
    const l = try f.g.opening(&s);
    defer f.release(&s);
    try t.expect(reclaim.restored_first);
    try t.expectEqual(@as(u64, 2048), s.cached);
    try t.expect(f.store.?.store.find(ids) == null); // its blob was reclaimed by the new lane's promise
    const got = reply(try finish(&f, &s, l));
    try t.expectEqualSlices(u32, &expected, &got);
}

fn cancelled(_: *anyopaque) bool {
    return true;
}

test "Gemma shared: cancelled prompt returns every lease and enforces page coverage before a launch" {
    var f: Fixture = undefined;
    try f.init(true, true);
    defer f.deinit();
    try t.expectEqual(@as(u32, 128), f.g.facts().max_streams);
    try t.expectEqual(@as(u32, 512), f.store.?.store.rules.grid);
    try t.expect(f.store.?.store.rules.planned);
    var flag: u8 = 0;
    var s = try lanes.Stream.init(a, .{ .id = "cancel", .prompt = &.{ 1, 2, 3 }, .max_new = 16, .cancel_check = .{ .ptr = &flag, .check = cancelled } });
    defer s.deinit(a);
    const backend = f.g.backend();
    try t.expectError(error.Cancelled, backend.vtable.prefill_step.?(backend.ptr, &s, 512));
    try idle(&f);
    try t.expectEqual(@as(usize, 128), f.g.pool.?.available());
    try t.expectError(error.PastPages, f.m.forward(&f.k, f.g.s, &.{.{ .slot = 0, .p0 = 0, .tokens = &.{1} }}, true, 0, 1));
}

test "Gemma shared: snapshots spend pool pages without new device allocations" {
    var f: Fixture = undefined;
    try f.init(true, true);
    defer f.deinit();
    const layout = fwd.pages.layout(&f.m);
    try t.expectEqual(@as(u64, 7864384), layout.fixed_bytes);
    try t.expectEqual(@as(u64, 524288), layout.pageBytes());
    try t.expectEqual(@as(u64, 5237760), g4.snap.copyBytes(&f.m, 2048, 2048));
    try t.expectEqual(@as(usize, 10), g4.snap.blobPages(&f.m, 2048, 2048));
    const ids = try prompt();
    defer a.free(ids);
    var s = try stream(ids);
    defer s.deinit(a);
    const before = fake.allocations;
    _ = try fill(&f, &s);
    try t.expectEqual(before, fake.allocations);
    try t.expect(f.store.?.store.tree.snaps > 0);
    f.release(&s);
    try idle(&f);
    try t.expect(f.g.pool.?.available() < 128);
    f.store.?.close();
    f.store = null;
    try t.expectEqual(@as(usize, 128), f.g.pool.?.available());
}
