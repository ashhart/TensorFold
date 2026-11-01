//! Real Kolibri leases, snapshots and prompt adapter with a deterministic row oracle, not checkpoint arithmetic.
const std = @import("std");
const k1 = @import("kolibri1");
const cuda = @import("cuda");
const lanes = @import("lanes");
const adapter = @import("prompt_pages");
const fake = @import("kolibri_page_driver.zig");
const a = std.testing.allocator;
const t = std.testing;
const fwd = k1.forward;
const be = k1.backend;
const marks = [_]u32{ 8192, 8448, 8704, 8960 };
const config: k1.Config = blk: {
    var c: k1.Config = .{ .layers = 5, .hidden = 128, .heads = 12, .kv_heads = 1, .head_dim = 128, .vocab = 256, .experts = 2, .top_k = 1, .moe_width = 128, .shared_width = 128, .window = 513, .rope_theta = 1e4, .eps = 1e-6 };
    c.full[4] = true;
    break :blk c;
};

const Fixture = struct {
    d: cuda.Driver,
    m: fwd.Model,
    g: be.Kolibri,
    k: k1.kernels.Kernels = undefined,
    store: ?*adapter.Adapter = null,

    fn init(f: *Fixture, paged: bool, share: bool) !void {
        return f.initWith(paged, share);
    }
    fn initWith(f: *Fixture, paged: bool, share: bool) !void {
        f.store = null;
        f.d = fake.driver();
        const queue: cuda.Stream = .{ .d = &f.d, .handle = @ptrFromInt(1) };
        f.m = .{ .gpa = a, .d = &f.d, .w = undefined, .c = config, .context = 16384, .slots = if (paged) 1 else 2, .ring = fwd.ringSize(fwd.prompt_rows, config.window), .max_rows = fwd.prompt_rows, .groups = 33, .k = &.{}, .v = &.{}, .norms = &.{}, .cos = 0, .sin = 0, .origin = 0, .s = undefined, .paged = paged };
        errdefer f.m.deinit();
        f.m.k = try a.alloc(u64, 5);
        f.m.v = try a.alloc(u64, 5);
        try fwd.pages.build(&f.m, queue);
        f.g = try be.Kolibri.init(a, &f.d, &f.m, &f.k, queue);
        errdefer f.g.deinit();
        if (paged) {
            const dropped = fake.graphs_dropped;
            f.g.steps[0] = .{ .exec = .{ .d = &f.d, .handle = @ptrFromInt(123) }, .slot = 0, .flat = 0, .logits = 0 };
            try be.shared.attach(&f.g, 1, 128);
            try t.expectEqual(dropped + 1, fake.graphs_dropped);
            var prng = std.Random.DefaultPrng.init(0x7a92);
            const pool = &f.g.pool.?;
            prng.random().shuffle(u32, pool.free[0..pool.nfree]);
        }
        if (share) f.store = (try adapter.Adapter.open(k1.native, a, &f.g, fwd.pages.tokens, 8)).?;
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
    const out = try a.alloc(u32, 9003);
    for (out, 0..) |*x, i| x.* = @intCast(i % 97 + 3);
    return out;
}

fn address(m: *const fwd.Model, cache: []const u64, layer: usize, slot: usize, position: usize) u64 {
    const rb = m.c.kv_heads * m.c.head_dim * 2;
    const row = if (m.c.full[layer] and m.paged) @as(usize, m.ids[slot * fwd.pages.perSlot(m) + position / fwd.pages.tokens]) * fwd.pages.tokens + position % fwd.pages.tokens else slot * (if (m.c.full[layer]) m.context else m.ring) + position % (if (m.c.full[layer]) m.context else m.ring);
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
        var end = @min(s.prompt().len, (start / fwd.prompt_rows + 1) * fwd.prompt_rows);
        for (l.marks) |mark| if (mark > start and mark < end) {
            end = mark;
        };
        for (start..end) |pos| for (0..f.m.c.layers) |i| for ([_][]u64{ f.m.k, f.m.v }, 0..) |cache, v| {
            const row: [*]u16 = @ptrFromInt(address(&f.m, cache, i, l.slot, pos));
            @memset(row[0 .. f.m.c.kv_heads * f.m.c.head_dim], value(s.prompt()[pos], pos, i, v == 1));
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
    for (0..m.c.layers) |i| {
        const first = if (m.c.full[i]) 0 else at - @min(at, m.c.window - 1);
        for (first..at) |pos| for ([_][]u64{ m.k, m.v }) |cache| {
            const row: [*]const u8 = @ptrFromInt(address(m, cache, i, slot, pos));
            hash.update(row[0 .. m.c.kv_heads * m.c.head_dim * 2]);
        };
    }
    return hash.final();
}

fn reply(hash: u64) [16]u32 {
    var prng = std.Random.DefaultPrng.init(hash);
    var out: [16]u32 = undefined;
    for (&out) |*x| x.* = prng.random().uintLessThan(u32, config.vocab);
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

test "Kolibri shared: resumed reply restores wrapped sliding tails with shared global pages" {
    const out = try resumed(false);
    try t.expectEqual(@as(u64, 8960), out.cached);
    try t.expectEqualSlices(u32, &out.fresh, &out.resumed);
    const bad = try resumed(true);
    try t.expectEqual(@as(u64, 8960), bad.cached);
    try t.expect(!std.mem.eql(u32, &bad.fresh, &bad.resumed));
    try t.expect(fake.empty());
}

test "Kolibri shared: concurrent continuations preserve solo replies and a swapped-slot control fails" {
    const ids = try prompt();
    defer a.free(ids);
    const other = try a.dupe(u32, ids);
    defer a.free(other);
    other[8990] = 77;
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
    const page = 8990 / fwd.pages.tokens;
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

/// Read the same chain-order metadata and page bindings passed to the real tree operator.
fn treeReply(f: *Fixture, chains: []const fwd.Chain, graph_slot: ?usize) ![16]u32 {
    const windows = try a.alloc(k1.tree.Window, chains.len);
    defer a.free(windows);
    for (chains, windows) |ch, *w| w.* = .{ .committed = ch.p0, .rows = ch.tokens.len };
    const plan = try k1.tree.build(a, windows);
    defer a.free(plan.flat);
    const flat = try f.m.alloc(plan.flat.len * 4);
    try f.m.upload(f.g.s, flat, std.mem.sliceAsBytes(plan.flat));
    const offs = try f.m.alloc(f.m.c.layers * chains.len * 16);
    try f.m.offsets(f.g.s, chains, offs);
    try fwd.pages.chains(&f.m, f.g.s, chains);
    const rows = f.m.treeRows(flat, plan, 0, 0, offs, graph_slot);
    if (f.m.paged) try t.expect(rows.pages != null);
    var hash = std.hash.Wyhash.init(0);
    for (chains, 0..) |ch, j| for (0..f.m.c.layers) |i| {
        const first = if (f.m.c.full[i]) 0 else ch.p0 - @min(ch.p0, f.m.c.window - 1);
        for (first..ch.p0) |pos| for ([_][]u64{ f.m.k, f.m.v }, 0..) |cache, v| {
            const ptr = if (f.m.c.full[i]) blk: {
                const off: [*]const i64 = @ptrFromInt(rows.offs);
                const base: u64 = @bitCast(@as(i64, @intCast(rows.origin)) + off[(i * chains.len + j) * 2 + v] * 2);
                const rb = f.m.c.kv_heads * f.m.c.head_dim * 2;
                if (rows.pages) |pg| {
                    const table: [*]const u32 = @ptrFromInt(pg.table);
                    break :blk base + (@as(usize, table[j * pg.stride + pos / pg.tok]) * pg.page + pos % pg.tok * rb / 2) * 2;
                }
                break :blk base + pos * rb;
            } else address(&f.m, cache, i, ch.slot, pos);
            const bytes: [*]const u8 = @ptrFromInt(ptr);
            hash.update(bytes[0 .. f.m.c.kv_heads * f.m.c.head_dim * 2]);
        };
        // Each chain's staged path is private; its proposed rows never replace another chain's keys.
        for (ch.tokens, 0..) |token, r| for (0..2) |v| {
            var data: [128]u16 = undefined;
            @memset(&data, value(token, ch.p0 + r, i, v == 1));
            hash.update(std.mem.sliceAsBytes(&data));
        };
    };
    return reply(hash.final());
}

fn extend(f: *Fixture, slot: usize, from: usize, ids: []const u32) void {
    for (ids, from..) |token, pos| for (0..f.m.c.layers) |i| for ([_][]u64{ f.m.k, f.m.v }, 0..) |cache, v| {
        const row: [*]u16 = @ptrFromInt(address(&f.m, cache, i, slot, pos));
        @memset(row[0..128], value(token, pos, i, v == 1));
    };
}

test "Kolibri shared: copy-draft tree bindings preserve plain replies in reversed chain order and after rejection" {
    const ids = try prompt();
    defer a.free(ids);
    const other = try a.dupe(u32, ids);
    defer a.free(other);
    other[8990] = 77;
    var f: Fixture = undefined;
    try f.init(true, true);
    defer f.deinit();
    var s = try stream(ids);
    defer s.deinit(a);
    _ = try fill(&f, &s);
    var peer = try stream(other);
    defer peer.deinit(a);
    _ = try fill(&f, &peer);
    defer f.release(&peer);
    const l = f.g.lanes.getPtr(&s).?;
    const p = f.g.lanes.getPtr(&peer).?;
    const proposed = [_]u32{ 4, 5, 88, 99 };
    const chains = [_]fwd.Chain{ .{ .slot = p.slot, .p0 = ids.len, .tokens = &proposed }, .{ .slot = l.slot, .p0 = ids.len, .tokens = &.{ 6, 7 } } };
    const got = try treeReply(&f, &chains, null);
    const solo = try treeReply(&f, chains[0..1], p.slot);
    try t.expectEqual(f.m.table + p.slot * fwd.pages.perSlot(&f.m) * 4, fwd.pages.treeView(&f.m, p.slot).?.table);
    var plain: Fixture = undefined;
    try plain.init(false, false);
    defer plain.deinit();
    var ps = try stream(ids);
    defer ps.deinit(a);
    _ = try fill(&plain, &ps);
    var pp = try stream(other);
    defer pp.deinit(a);
    _ = try fill(&plain, &pp);
    const fresh_chains = [_]fwd.Chain{ .{ .slot = 1, .p0 = ids.len, .tokens = &proposed }, .{ .slot = 0, .p0 = ids.len, .tokens = &.{ 6, 7 } } };
    try t.expectEqualSlices(u32, &try treeReply(&plain, &fresh_chains, null), &got);
    try t.expectEqualSlices(u32, &try treeReply(&plain, fresh_chains[0..1], 1), &solo);
    plain.release(&ps);
    plain.release(&pp);
    // The real keep hook discards two speculative rows. Subsequent writes replace exactly those positions.
    l.base = ids.len;
    try l.rows.appendSlice(a, &proposed);
    extend(&f, l.slot, ids.len, &proposed);
    l.len = ids.len + proposed.len;
    const backend = f.g.backend();
    const positions = [_]u64{ ids.len + 1, ids.len + 2, ids.len + 3, ids.len + 4 };
    const win: lanes.backend.Window = .{ .stream = &s, .pending = proposed[0], .tokens = proposed[1..], .positions = &positions, .held = 0, .parents = null };
    try backend.vtable.keep(backend.ptr, &.{win}, &.{&.{ 0, 1 }});
    try t.expectEqual(@as(u64, ids.len + 2), l.len);
    extend(&f, l.slot, l.len, &.{8});
    extend(&plain, 0, ids.len, &.{ 4, 5, 8 });
    try t.expectEqual(reply(digest(&plain.m, 0, ids.len + 3)), reply(digest(&f.m, l.slot, ids.len + 3)));
    f.release(&s);
}

fn cancelled(_: *anyopaque) bool {
    return true;
}

test "Kolibri shared: cancellation returns leases and page coverage fails before launch" {
    var f: Fixture = undefined;
    try f.init(true, true);
    defer f.deinit();
    try t.expectEqual(@as(u32, 64), f.g.facts().max_streams);
    var flag: u8 = 0;
    var s = try lanes.Stream.init(a, .{ .id = "cancel", .prompt = &.{ 1, 2, 3 }, .max_new = 16, .cancel_check = .{ .ptr = &flag, .check = cancelled } });
    defer s.deinit(a);
    const backend = f.g.backend();
    try t.expectError(error.Cancelled, backend.vtable.prefill_step.?(backend.ptr, &s, 8192));
    try idle(&f);
    try t.expectEqual(@as(usize, 128), f.g.pool.?.available());
    try t.expectError(error.PastPages, f.m.forward(.{ .k = &f.k, .s = f.g.s }, &.{.{ .slot = 0, .p0 = 0, .tokens = &.{1} }}, true, null, 0));
}

test "Kolibri shared: snapshots and fifty-layer geometry are charged to the pool" {
    var f: Fixture = undefined;
    try f.init(true, true);
    defer f.deinit();
    try t.expectEqual(@as(u64, 1048576), k1.snap.copyBytes(&f.m, 8960, 8960));
    try t.expectEqual(@as(usize, 8), k1.snap.blobPages(&f.m, 8960, 8960));
    const ids = try prompt();
    defer a.free(ids);
    var s = try stream(ids);
    defer s.deinit(a);
    const before = fake.allocations;
    _ = try fill(&f, &s);
    try t.expectEqual(before, fake.allocations);
    f.release(&s);
    try idle(&f);
    f.store.?.close();
    f.store = null;
    try t.expectEqual(@as(usize, 128), f.g.pool.?.available());
    var full = f.m;
    full.c.layers = 50;
    full.c.kv_heads = 4;
    for (0..50) |i| full.c.full[i] = i % 5 == 4;
    full.context = 33792;
    try t.expectEqual(@as(u64, 5242880), fwd.pages.pageBytes(&full));
    try t.expectEqual(@as(u64, 718275088), fwd.pages.layout(&full).fixed_bytes);
    try t.expectEqual(@as(u64, 41943040), k1.snap.copyBytes(&full, 8960, 8960));
    try t.expectEqual(@as(usize, 8), k1.snap.blobPages(&full, 8960, 8960));
}

test "Kolibri shared: 128-dimensional prompt dispatch keeps linear functions and binds paged functions" {
    const d = fake.driver();
    const queue: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var kernel: cuda.window_attention.WindowAttention = undefined;
    kernel.prompt_fns = .{ .{ .d = &d, .handle = @ptrFromInt(1) }, .{ .d = &d, .handle = @ptrFromInt(2) } };
    kernel.prompt_pages128 = .{ .{ .d = &d, .handle = @ptrFromInt(3) }, .{ .d = &d, .handle = @ptrFromInt(4) } };
    var w: cuda.window_attention.Window = .{ .h = 48, .hk = 4, .d = 128, .win = 0, .ring = 0, .scale = 1 };
    try kernel.prompt(queue, 0, 0, 0, 0, 0, 32, w);
    try t.expectEqual(@as(usize, 1), fake.last_function);
    w.pg = .{ .table = 42, .page_elems = 256 * 4 * 128, .tokens = 256 };
    try kernel.prompt(queue, 0, 0, 0, 0, 0, 32, w);
    try t.expectEqual(@as(usize, 3), fake.last_function);
    try t.expectEqual(@as(u64, 42), fake.last_table);
    w.h = 32;
    try kernel.prompt(queue, 0, 0, 0, 0, 0, 32, w);
    try t.expectEqual(@as(usize, 4), fake.last_function);
    w.pg = null;
    try kernel.prompt(queue, 0, 0, 0, 0, 0, 32, w);
    try t.expectEqual(@as(usize, 2), fake.last_function);
}

test "Kolibri shared: ring restore precedes forced snapshot eviction in reservation" {
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
    try t.expectEqual(@as(u32, 8960), f.store.?.store.find(ids).?.end);
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
            x.restored_first = x.f.g.restored_at == 8960;
            x.f.store.?.store.forget(x.ids, 8960);
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
    try t.expectEqual(@as(u64, 8960), s.cached);
    try t.expect(f.store.?.store.find(ids) == null); // its blob was reclaimed by the new lane's promise
    const got = reply(try finish(&f, &s, l));
    try t.expectEqualSlices(u32, &expected, &got);
}
