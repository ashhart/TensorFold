const std = @import("std");
const t = std.testing;

test "prompt cache: invalid native budgets are refused before checkpoint or asset access" {
    for ([_]f64{ -1, std.math.nan(f64), std.math.inf(f64), 1e300 }) |gib| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        var why: []const u8 = "";
        try t.expectError(error.BadCacheBudget, @import("serve_engine.zig").open(arena.allocator(), t.allocator, t.io, .{ .dir = "missing-cache-checkpoint", .model_type = "deepseek_v41", .prompt_cache_gib = gib }, &why));
    }
}

const kv = @import("kv");
const sessions = kv.sessions;
const budget = @import("session_budget.zig");
const sg = @import("sessions_gpu.zig");
const ced = @import("ced.zig");

test "prompt cache: zero disables RAM and disk; fractional budgets and old sub-limits survive" {
    try t.expectEqual(@as(u64, 16 << 30), try budget.bytes(null));
    try t.expectEqual(@as(u64, 1 << 28), try budget.bytes(0.25));
    const options = budget.options(.{ .disk = "unused-cache-tier", .ram_bytes = 32, .resident = 2, .resident_bytes = 1024 }, try budget.bytes(0));
    try t.expect(!options.enabled);
    const on = budget.options(options, try budget.bytes(4));
    try t.expectEqual(@as(u64, 4 << 30), on.cache_bytes);
    try t.expectEqual(@as(u64, 32), on.ram_bytes);
    try t.expectEqual(@as(?u32, 2), on.resident);
    try t.expect(!budget.options(.{ .enabled = false }, 1 << 30).enabled);
}

const families = [_]sessions.Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 8, .split = true },
    .{ .name = "index_k.2", .ratio = 2, .row_bytes = 4 },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 8, .split = true },
    .{ .name = "index_k.20", .ratio = 1, .row_bytes = 4 },
};
const Host = struct {
    pool: sessions.Pool,
    data: sessions.HostPool,
    store: sessions.Store,
    owner: sg.Sessions = undefined,

    fn init(h: *Host, world: u32, rank: u32) !void {
        h.pool = try sessions.Pool.init(t.allocator, .{ .families = &families, .page = 256 }, 32768, world, rank);
        errdefer h.pool.deinit();
        h.data = try sessions.HostPool.init(t.allocator, &h.pool);
        errdefer h.data.deinit();
        h.store = try sessions.Store.init(t.allocator, t.io, &h.pool, h.data.store(), null, .{ .ram_bytes = 1 << 30, .skip_covered = false });
        h.owner.gpa = t.allocator;
    }
    fn deinit(h: *Host) void {
        h.store.deinit();
        h.data.deinit();
        h.pool.deinit();
    }
    fn save(h: *Host, slot: *sessions.Slot, ids: []const i32, state: State, tag: u32) !u32 {
        return h.saveSized(slot, ids, state, tag, @sizeOf(State));
    }
    fn saveSized(h: *Host, slot: *sessions.Slot, ids: []const i32, state: State, tag: u32, physical: usize) !u32 {
        const snap = try t.allocator.create(sg.Sessions.Snap);
        const data = t.allocator.alloc(u8, physical) catch |err| {
            t.allocator.destroy(snap);
            return err;
        };
        @memset(data, 0);
        @memcpy(data[0..@sizeOf(State)], std.mem.asBytes(&state));
        snap.* = .{ .owner = &h.owner, .bytes = data, .pos = ids.len, .tail = @splat(0), .tail_len = 0, .stash = state.stash, .ds_rows = 0, .measure = @sizeOf(State) };
        var b = snap.bounded();
        b.logical_bytes = snap.measure;
        return (h.store.save(slot, b, ids, tag, 1) catch |err| {
            sg.Sessions.Snap.release(snap);
            return err;
        }).?;
    }
};
const State = struct {
    ring: [128]u32 = @splat(0),
    carry: u32 = 0,
    pos: u32 = 0,
    stash: ced.Stash = .{},
};
fn row(h: *Host, slot: *sessions.Slot, fi: u32, pos: u32) ?[]u8 {
    const f = families[fi];
    const pg = slot.tableAt(pos / h.pool.page);
    if (f.split and !h.pool.owns(pg)) return null;
    const off = (pos % h.pool.page) / f.ratio * f.row_bytes;
    return h.data.pageOf(fi, h.pool.familyPage(f, pg))[off..][0..f.row_bytes];
}
// A dependent state canary, not a model forward: every ratio-2 row depends on the restored carry.
fn advance(h: *Host, slot: *sessions.Slot, state: *State, ids: []const i32) !void {
    const end: u32 = @intCast(state.pos + ids.len);
    try slot.ensure(end);
    for (ids) |id| {
        const pos = state.pos;
        const token: u32 = @intCast(id);
        for (families, 0..) |f, fi| if ((pos + 1) % f.ratio == 0) {
            if (row(h, slot, @intCast(fi), pos)) |dst| for (dst, 0..) |*b, j| {
                b.* = @truncate(token +% state.carry *% 13 +% pos *% 17 +% @as(u32, @intCast(fi * 31 + j)));
            };
        };
        state.ring[pos % state.ring.len] = token;
        state.carry = token;
        state.stash.extend(pos, &.{token});
        state.pos += 1;
    }
}

test "prompt cache: main whole and piece plans save every 16-row residue inside 2048, never the chunk grid" {
    for (0..128) |i| {
        const n: u64 = 4096 + i * 16 + 7;
        const expected: u64 = n - 7;
        try t.expectEqual(expected, ced.savePoint(n, 0).?);
        try t.expectEqual(expected, kv.sess.savePoint(n, 0).?);
        var plan = kv.sched.Planner.init(t.allocator, .{});
        defer plan.deinit();
        var round = plan.beginAdmit();
        defer round.deinit(t.allocator);
        const v = try plan.admit(&round, .{ .key = 1, .n = n, .max_new = 4, .submitted = 0 }, null, 0, null);
        try t.expect(v == .admit);
        var pieces: std.ArrayList(kv.sched.Piece) = .empty;
        defer pieces.deinit(t.allocator);
        var finals: std.ArrayList(usize) = .empty;
        defer finals.deinit(t.allocator);
        var saved: ?u64 = null;
        while (finals.items.len == 0) {
            try plan.plan(null, &pieces, &finals);
            for (pieces.items) |piece| if (piece.save) {
                try t.expect(saved == null);
                saved = piece.end;
            };
        }
        try t.expectEqual(@as(?u64, expected), saved);
    }
}

test "prompt cache: real partial-page index and snapshots resume rings carry and rows on both TP ranks" {
    var ids: [8400]i32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(1 + (i * 17) % 8191);
    for ([_]u32{ 1, 2, 4 }) |world| for (0..world) |rank| {
        var h: Host = undefined;
        try h.init(world, @intCast(rank));
        defer h.deinit();
        const live = try h.pool.newSlot(8400);
        const fresh = try h.pool.newSlot(8400);
        const resumed = try h.pool.newSlot(8400);
        for ([_]u32{ 64, 2032, 2048, 2064, 4080, 4096, 4112, 8176 }) |seam| {
            var state: State = .{};
            try advance(&h, live, &state, ids[0..seam]);
            const tag = ced.tag_fast;
            const id = try h.save(live, ids[0..seam], state, tag);
            // Continue writing the original partial page, then release it; only rows before the cut are reusable.
            try advance(&h, live, &state, ids[seam .. seam + 17]);
            _ = try live.releaseAll();
            const hit = (try h.store.find(tag, &ids)).?;
            try t.expectEqual(id, hit.id);
            try t.expectEqual(@as(u64, seam), hit.pos);
            try t.expect((try h.store.find(ced.tag, &ids)) == null);
            const old = ids[seam - 1];
            ids[seam - 1] = -1;
            try t.expect((try h.store.find(tag, &ids)) == null);
            ids[seam - 1] = old;
            try t.expect((try h.store.find(tag, ids[0..seam])) == null);
            const b = try h.store.restoreRam(id, resumed);
            const snap: *sg.Sessions.Snap = @ptrCast(@alignCast(b.ptr));
            var restored: State = undefined;
            @memcpy(std.mem.asBytes(&restored), snap.bytes);
            try t.expectEqual(seam, restored.pos);
            if (seam % 256 != 0) try t.expect(resumed.tableAt(seam / 256) != h.store.entry(id).extra.pages[seam / 256]);
            try advance(&h, resumed, &restored, ids[seam..]);
            var oracle: State = .{};
            try advance(&h, fresh, &oracle, &ids);
            try t.expectEqualDeep(oracle, restored);
            for (families, 0..) |f, fi| {
                var pos: u32 = f.ratio - 1;
                while (pos < ids.len) : (pos += f.ratio) {
                    if (row(&h, fresh, @intCast(fi), pos)) |want| try t.expectEqualSlices(u8, want, row(&h, resumed, @intCast(fi), pos).?);
                }
            }
            _ = try resumed.releaseAll();
            _ = try fresh.releaseAll();
            try h.store.drop(id);
        }
    };
}

test "prompt cache: one combined cap evicts oversized newest entries and preserves live pages" {
    var h: Host = undefined;
    try h.init(2, 0);
    defer h.deinit();
    const live = try h.pool.newSlot(1024);
    var ids: [1024]i32 = @splat(7);
    var state: State = .{};
    try advance(&h, live, &state, &ids);
    const id = try h.save(live, &ids, state, ced.tag_fast);
    const charge = sessions.retention.bytes(&h.store);
    const kv_bytes = 2 * (128 * 8 + 256 * 8) + 4 * (128 * 4 + 256 * 4);
    try t.expect(charge >= kv_bytes + @sizeOf(State));
    try sessions.retention.trim(&h.store, charge);
    try t.expect(h.store.entry(id).ram);
    try sessions.retention.trim(&h.store, charge - 1);
    try t.expectEqual(@as(u32, 0), h.store.ram_count);
    try t.expectEqual(@as(u64, 0), sessions.retention.bytes(&h.store));
    try t.expectEqual(@as(u32, 4), live.len);
    try t.expectEqual(@as(u32, 1), h.pool.refs[live.tableAt(0)]);
}

test "prompt cache: shared pages charged once, split-rank worst charge and LRU victims agree" {
    var charges: [2][3]u64 = undefined;
    for (0..2) |rank| {
        var h: Host = undefined;
        try h.init(2, @intCast(rank));
        defer h.deinit();
        const live = try h.pool.newSlot(2048);
        const continued = try h.pool.newSlot(2048);
        var ids: [1100]i32 = @splat(8);
        var state: State = .{};
        try advance(&h, live, &state, ids[0..512]);
        const first = try h.save(live, ids[0..512], state, ced.tag_fast);
        _ = try live.releaseAll();
        const b = try h.store.restoreRam(first, continued);
        const snap: *sg.Sessions.Snap = @ptrCast(@alignCast(b.ptr));
        @memcpy(std.mem.asBytes(&state), snap.bytes);
        try advance(&h, continued, &state, ids[512..1040]);
        const second = try h.save(continued, ids[0..1040], state, ced.tag_fast);
        _ = try continued.releaseAll();
        charges[rank][0] = sessions.retention.bytes(&h.store);
        // Five distinct pages, not the sum of the two histories' seven pages; residue 0 has three pages.
        const actual_kv: u64 = 3 * (128 * 8 + 256 * 8) + 5 * (128 * 4 + 256 * 4);
        const snapshots = 2 * @sizeOf(State);
        const entries = 2 * @sizeOf(@TypeOf(h.store.index.entries.items[0]));
        const nodes = 5 * @sizeOf(@TypeOf(h.store.index.nodes.items[0]));
        // The trie has four full token blocks, a 16-token tail and seven page ids over its five unique KV pages.
        try t.expectEqual(actual_kv + snapshots + entries + nodes + 4 * (4 * 256 + 16 + 7), charges[rank][0]);
        // Removing a shared older entry releases only its bounded/index ownership, not the descendant's KV.
        try h.store.drop(first);
        charges[rank][1] = sessions.retention.bytes(&h.store);
        try t.expect(charges[rank][1] < charges[rank][0]);
        try sessions.retention.trim(&h.store, charges[rank][1]);
        try t.expect(h.store.entry(second).ram);
        try sessions.retention.trim(&h.store, 0);
        charges[rank][2] = sessions.retention.bytes(&h.store);
        try t.expectEqual(@as(u32, 0), h.store.ram_count);
    }
    try t.expectEqualDeep(charges[0], charges[1]);
}

test {
    _ = @import("session_protocol.zig");
}

test "prompt cache: the combined cap removes the oldest unrelated chat first" {
    var h: Host = undefined;
    try h.init(2, 0);
    defer h.deinit();
    const live = try h.pool.newSlot(1024);
    var ids: [1024]i32 = @splat(5);
    var state: State = .{};
    try advance(&h, live, &state, &ids);
    const first = try h.save(live, &ids, state, ced.tag_fast);
    _ = try live.releaseAll();
    const one = sessions.retention.bytes(&h.store);
    @memset(&ids, 9);
    state = .{};
    try advance(&h, live, &state, &ids);
    const second = try h.save(live, &ids, state, ced.tag_fast);
    _ = try live.releaseAll();
    try t.expect(sessions.retention.bytes(&h.store) > one);
    try sessions.retention.trim(&h.store, one);
    try t.expect(!h.store.entry(first).ram);
    try t.expect(h.store.entry(second).ram);
    try t.expectEqual(@as(u32, 1), h.store.ram_count);
    try t.expectEqual(one, sessions.retention.bytes(&h.store));
}

fn unequalSnapshots(world: u32, bounded_limit: bool) !void {
    var cap: u64 = 0;
    var decisions: [4][5]u64 = undefined;
    var charges: [4]u64 = undefined;
    for (0..world) |rank| {
        var h: Host = undefined;
        try h.init(world, @intCast(rank));
        defer h.deinit();
        const live = try h.pool.newSlot(1024);
        const ids: [1025]i32 = @splat(7);
        const physical = @sizeOf(State) + rank * 4096;
        try live.ensure(1024);
        _ = try h.saveSized(live, ids[0..1024], .{}, 10, physical);
        if (rank == 0) cap = if (bounded_limit) 2 * h.store.bounded_total else sessions.retention.bytes(&h.store);
        charges[rank] = sessions.retention.bytes(&h.store);
        try t.expectEqual(@as(u64, @sizeOf(sg.Sessions.Snap) + physical), h.store.bounded_total);
        if (bounded_limit) h.store.policy.ram_bytes = cap else try sessions.retention.trim(&h.store, cap);
        decisions[rank][0] = h.store.ram_count;
        _ = try h.saveSized(live, ids[0..1024], .{}, 11, physical);
        if (!bounded_limit) try sessions.retention.trim(&h.store, cap);
        decisions[rank][1] = h.store.ram_count;
        if (bounded_limit) h.store.policy.ram_bytes = 2 * @sizeOf(State);
        _ = try h.saveSized(live, ids[0..1024], .{}, 12, physical);
        if (!bounded_limit) try sessions.retention.trim(&h.store, cap);
        decisions[rank][2] = @intFromBool((try h.store.find(10, &ids)) != null);
        decisions[rank][3] = @intFromBool((try h.store.find(11, &ids)) != null);
        decisions[rank][4] = @intFromBool((try h.store.find(12, &ids)) != null);
        try sessions.retention.trim(&h.store, 0);
        try t.expectEqual(@as(u64, 0), h.store.bounded_total);
    }
    const expected: [5]u64 = if (bounded_limit) .{ 1, 2, 0, 1, 1 } else .{ 1, 1, 0, 0, 1 };
    for (0..world) |rank| {
        try t.expectEqualDeep(expected, decisions[rank]);
        try t.expectEqual(charges[0], charges[rank]);
    }
}

test "prompt cache: unequal physical rank snapshots make the same combined-cap evictions" {
    for ([_]u32{ 2, 4 }) |world| try unequalSnapshots(world, false);
}

test "prompt cache: unequal physical rank snapshots make the same bounded-limit evictions" {
    for ([_]u32{ 2, 4 }) |world| try unequalSnapshots(world, true);
}
