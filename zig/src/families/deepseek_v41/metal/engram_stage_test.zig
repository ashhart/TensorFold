//! The staged Engram reads: same rows as the synchronous path, warm or not, and nothing left pending on a failure.
const std = @import("std");
const host = @import("dsv41");
const stage = @import("engram_stage.zig");

const eh = host.engram_host;
const Config = host.config.Config;
const Stage = stage.Stage;
const testing = std.testing;

/// A source over index rows that logs what was issued and joined, and can fail a layer's join.
const Mock = struct {
    pending: [2]i32 = .{ 0, 0 },
    issued: std.ArrayList(i64) = .empty,
    joins: [64]usize = undefined,
    n_joins: usize = 0,
    fail_layer: ?usize = null,

    fn source(m: *Mock) eh.Source {
        return .{ .ptr = m, .vtable = &.{ .issue = issue, .rows = rows } };
    }

    fn issue(p: *anyopaque, li: usize, idx: []const i64) anyerror!void {
        const m: *Mock = @ptrCast(@alignCast(p));
        m.pending[li] += 1;
        try m.issued.appendSlice(testing.allocator, idx);
    }

    fn rows(p: *anyopaque, li: usize, idx: []const i64, dim: usize, out: []u16) anyerror!void {
        const m: *Mock = @ptrCast(@alignCast(p));
        m.pending[li] = 0;
        m.joins[m.n_joins] = li;
        m.n_joins += 1;
        if (m.fail_layer == li) return error.EngramReadFailed;
        const shifted = try testing.allocator.alloc(i64, idx.len);
        defer testing.allocator.free(shifted);
        for (shifted, idx) |*o, i| o.* = i + @as(i64, @intCast(li)) * 7919;
        eh.indexRows(shifted, dim, out);
    }
};

const Rig = struct {
    c: Config = .{},
    map: [1000]u32 = undefined,
    ids_buf: [eh.max_layers]u32 = undefined,
    h: eh.Host = undefined,

    fn init(r: *Rig) void {
        for (&r.map, 0..) |*v, i| v.* = @intCast((i * 37) % 97);
        r.ids_buf[0..2].* = .{ 1, 14 };
        r.h = .{ .layer_ids = r.ids_buf[0..2], .max_ngram = r.c.engram_max_ngram, .heads = r.c.engram_heads, .token_map = &r.map, .pad_id = 2 };
        for (0..2) |li| for (0..r.c.engram_max_ngram) |s| {
            r.h.mult[li][s] = @intCast(1000003 + 7 * li + s);
        };
        r.h.layout(1000);
    }

    fn rowsLen(r: *const Rig, n: usize) usize {
        return n * r.c.hashCols() * r.c.engram_head_dim;
    }
};

const ids_a = [_]u32{ 5, 9, 11 };
const look_a = [_]u32{ 3, 4, 8 };
const ids_b = [_]u32{ 17, 23 };
const look_b = [_]u32{ 1, 2, 6 };
const pieces = [_]stage.Piece{ .{ .ids = &ids_a, .lookback = &look_a }, .{ .ids = &ids_b, .lookback = &look_b } };

/// The synchronous reference: `windowRows` per layer and piece, as the forward read them before staging.
fn reference(r: *const Rig, src: ?eh.Source, out: *[2][]u16) !void {
    const K = r.c.hashCols() * r.c.engram_head_dim;
    for (r.c.engram_layers.items(), 0..) |layer, li| {
        var r0: usize = 0;
        for (pieces) |pc| {
            try r.h.windowRows(testing.allocator, src, pc.ids, pc.lookback, layer, 0, 1, r.c.engram_head_dim, out[li][r0 * K ..][0 .. pc.ids.len * K]);
            r0 += pc.ids.len;
        }
    }
}

fn alloc2(r: *const Rig) ![2][]u16 {
    const a = try testing.allocator.alloc(u16, r.rowsLen(5));
    errdefer testing.allocator.free(a);
    return .{ a, try testing.allocator.alloc(u16, r.rowsLen(5)) };
}

fn free2(o: [2][]u16) void {
    for (o) |s| testing.allocator.free(s);
}

test "staged rows are byte-equal to the synchronous read, unwarmed, warmed and warmed wrong" {
    var rig: Rig = .{};
    rig.init();
    var m: Mock = .{};
    defer m.issued.deinit(testing.allocator);
    const want = try alloc2(&rig);
    defer free2(want);
    try reference(&rig, m.source(), @constCast(&want));
    const wrong_ids = [_]u32{ 40, 41, 42, 43 };
    for ([_]enum { cold, right, wrong }{ .cold, .right, .wrong }) |kind| {
        const got = try alloc2(&rig);
        defer free2(got);
        switch (kind) {
            .cold => {},
            .right => for (pieces) |pc| rig.h.prefetch(testing.allocator, m.source(), pc.ids, pc.lookback, 0, 1),
            .wrong => rig.h.prefetch(testing.allocator, m.source(), &wrong_ids, &look_a, 0, 1),
        }
        var st = try Stage.begin(testing.allocator, &rig.c, &rig.h, m.source(), &pieces, &got);
        defer st.deinit();
        try st.joinAll();
        for (want, got) |w, g| try testing.expectEqualSlices(u16, w, g);
    }
}

test "a layer's rows are read only when its command buffer comes: layer 1 before 12, layer 14 before 23" {
    var rig: Rig = .{};
    rig.init();
    var m: Mock = .{};
    defer m.issued.deinit(testing.allocator);
    const got = try alloc2(&rig);
    defer free2(got);
    var st = try Stage.begin(testing.allocator, &rig.c, &rig.h, m.source(), &pieces, &got);
    defer st.deinit();
    // both layers' reads started at once, none joined
    try testing.expectEqual([2]i32{ 2, 2 }, m.pending);
    try testing.expectEqual(@as(usize, 0), m.n_joins);
    try st.joinRange(0, 1);
    try testing.expectEqual(@as(usize, 0), m.n_joins);
    try st.joinRange(1, 12);
    try testing.expectEqualSlices(usize, &.{ 0, 0 }, m.joins[0..m.n_joins]);
    try testing.expectEqual([2]i32{ 0, 2 }, m.pending);
    try st.joinRange(12, 23);
    try testing.expectEqualSlices(usize, &.{ 0, 0, 1, 1 }, m.joins[0..m.n_joins]);
    try st.joinRange(12, 23); // joined once only
    try testing.expectEqual(@as(usize, 4), m.n_joins);
}

test "a failed join leaves no read pending and nothing allocated" {
    var rig: Rig = .{};
    rig.init();
    var m: Mock = .{ .fail_layer = 0 };
    defer m.issued.deinit(testing.allocator);
    const got = try alloc2(&rig);
    defer free2(got);
    {
        var st = try Stage.begin(testing.allocator, &rig.c, &rig.h, m.source(), &pieces, &got);
        defer st.deinit();
        try testing.expectError(error.EngramReadFailed, st.joinRange(0, 12));
        try testing.expectEqual(@as(i32, 2), m.pending[1]); // layer 14's reads still out: deinit waits them out
    }
    try testing.expectEqual([2]i32{ 0, 0 }, m.pending);
    // the next launch reads the same rows as ever
    m.fail_layer = null;
    const again = try alloc2(&rig);
    defer free2(again);
    var st = try Stage.begin(testing.allocator, &rig.c, &rig.h, m.source(), &pieces, &again);
    defer st.deinit();
    try st.joinAll();
    const want = try alloc2(&rig);
    defer free2(want);
    try reference(&rig, m.source(), @constCast(&want));
    for (want, again) |w, g| try testing.expectEqualSlices(u16, w, g);
}

test "a begin that fails part-way frees its hashed rows and waits out what it started" {
    var rig: Rig = .{};
    rig.init();
    var m: Mock = .{};
    defer m.issued.deinit(testing.allocator);
    const got = try alloc2(&rig);
    defer free2(got);
    var fa = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 3 });
    try testing.expectError(error.OutOfMemory, Stage.begin(fa.allocator(), &rig.c, &rig.h, m.source(), &pieces, &got));
    try testing.expectEqual([2]i32{ 0, 0 }, m.pending);
}
