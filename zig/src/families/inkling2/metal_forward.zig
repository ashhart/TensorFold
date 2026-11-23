//! Shared forward over visits, with per-stream attention and convolution state; lengths advance only after GPU success.
const std = @import("std");
const mtl = @import("metal");
const st = @import("metal_state.zig");
const layout = @import("state_layout.zig");
const layer = @import("metal_forward_layer.zig");
const snapshot = @import("metal_snapshot.zig");
const Model = @import("metal_model.zig").Model;
const View = @import("../../core/exl3_projection.zig").View;

pub const Item = layer.Item;
pub const ImageRows = layer.ImageRows;
pub const max_items = layer.max_items;
pub const Snapshot = snapshot.Snapshot;

pub const Error = error{
    EmptyItem,
    TooManyItems,
    RowsExceeded,
    ContextFull,
    DuplicateState,
    StateFailed,
    InvalidToken,
    InvalidImageRows,
    ImagesDisabled,
    NoTaps,
    LogitRows,
    RollbackTooFar,
    StateGeometry,
};

/// Logits row of the first logit row of item `index`: items' last `logit_rows` rows, in item order.
pub fn logitBase(items: []const Item, index: usize) u32 {
    var row: u32 = 0;
    for (items[0..index]) |it| row += it.logit_rows;
    return row;
}

/// Tap row of the first row of item `index` (tap-enabled items' rows, in item order).
pub fn tapBase(items: []const Item, index: usize) u32 {
    var row: u32 = 0;
    for (items[0..index]) |it| {
        if (it.taps) row += @intCast(it.ids.len);
    }
    return row;
}

fn check(m: *const Model, items: []const Item) Error!void {
    const t = &m.config.text;
    if (items.len == 0) return Error.EmptyItem;
    if (items.len > max_items) return Error.TooManyItems;
    var total: usize = 0;
    for (items, 0..) |it, k| {
        const n = it.ids.len;
        const s = it.state;
        if (n == 0) return Error.EmptyItem;
        if (s.failed) return Error.StateFailed;
        if (s.layers != t.executionLayers() or s.geometry.rows < n or n > m.options.rows) return Error.StateGeometry;
        if (@as(usize, s.len) + n > s.geometry.capacity) return Error.ContextFull;
        if (it.logit_rows > n) return Error.LogitRows;
        if (it.taps and m.taps == null) return Error.NoTaps;
        for (items[0..k]) |other| if (other.state == s) return Error.DuplicateState;
        for (it.ids) |id| if (id >= t.vocab) return Error.InvalidToken;
        if (it.images.len > 0 and !m.options.vision) return Error.ImagesDisabled;
        for (it.images) |img| {
            const end = @as(usize, img.row) + img.rows;
            if (img.rows == 0 or end > n or img.off + @as(usize, img.rows) * t.hidden * 2 > img.buf.length()) return Error.InvalidImageRows;
            for (it.ids[img.row..end]) |id| if (id != m.config.image_token) return Error.InvalidImageRows;
        }
        total += n;
    }
    if (total > m.options.rows) return Error.RowsExceeded;
}

/// Run every item's rows at its state's length and advance the states; logits and taps are in the model's buffers.
pub fn forward(m: *Model, items: []const Item) !void {
    try check(m, items);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var r = layer.Run.init(m, items);
    var pos: usize = 0;
    const ids = m.scratch.ids.slice(u32, m.options.rows);
    for (items) |it| {
        @memcpy(ids[pos..][0..it.ids.len], it.ids);
        pos += it.ids.len;
    }
    run(m, &r) catch |err| {
        for (items) |it| it.state.failed = true;
        return err;
    };
    for (items) |it| {
        const before = it.state.len;
        it.state.len += @intCast(it.ids.len);
        it.state.floor = @max(before, it.state.len -| layout.max_rollback);
    }
}

fn run(m: *Model, r: *layer.Run) !void {
    errdefer r.abort(); // an encoding error closes the half-built command buffer uncommitted; nothing half-encoded runs
    const queue = m.queueFor(r.n);
    queue.roundBegin("inkling2 pass"); // TF_METAL4_TIMELINE=n stamps n passes' stages on a Metal 4 queue
    r.begin();
    try r.embed();
    for (0..m.plan.n_visits) |k| {
        const visit: u32 = @intCast(k);
        try r.enter(visit);
        try r.capture(visit);
        try r.attend(visit);
        try m.store.encode(r.e, m.plan.visit(visit).layer, view(m.scratch.xm), view(m.scratch.rids), view(m.scratch.rwts), view(m.scratch.moe), r.n);
        try r.finish(visit);
    }
    try r.head();
    try r.flush();
    queue.roundEnd();
}

fn view(b: mtl.Buffer) View {
    return .{ .buffer = b, .offset = 0 };
}

/// Drop the rows past `new_len`: valid within the last forward and `layout.max_rollback` rows back, nothing is copied.
pub fn keep(m: *Model, state: *st.State, new_len: u32) !void {
    _ = m;
    if (state.failed) return Error.StateFailed;
    if (new_len > state.len or new_len < state.floor) return Error.RollbackTooFar;
    state.len = new_len;
}

/// An empty stream again: no positions, zero histories.
pub fn reset(m: *Model, state: *st.State) void {
    _ = m;
    state.clear();
}

pub fn save(m: *Model, state: *const st.State) !Snapshot {
    if (state.failed) return Error.StateFailed;
    return snapshot.save(m, state);
}

pub fn restore(m: *Model, state: *st.State, snap: *const Snapshot) !void {
    try snapshot.restore(m, state, snap);
}

test "inkling2 keep: a rollback stays inside the last forward and the window" {
    var none: [0]mtl.Buffer = .{};
    var s: st.State = .{ .gpa = std.testing.allocator, .geometry = .{ .capacity = 64, .rows = 16 }, .layers = 0, .kv = &none, .hist = &none };
    var m: Model = undefined;
    // a window of 5 rows after 20 accepted ones: back to 20 is allowed, to 19 is not
    s.len = 25;
    s.floor = 20;
    try keep(&m, &s, 23);
    try std.testing.expectEqual(@as(u32, 23), s.len);
    try std.testing.expectError(Error.RollbackTooFar, keep(&m, &s, 19));
    try std.testing.expectError(Error.RollbackTooFar, keep(&m, &s, 24));
    // a prompt chunk of 100 rows still only gives back its last 16
    s.len = 100;
    s.floor = @max(0, 100 - layout.max_rollback);
    try std.testing.expectError(Error.RollbackTooFar, keep(&m, &s, 83));
    try keep(&m, &s, 84);
    s.failed = true;
    try std.testing.expectError(Error.StateFailed, keep(&m, &s, 84));
}

test "inkling2 forward: logit and tap rows follow the item order" {
    var s: [3]st.State = undefined;
    const ids = [_]u32{ 1, 2, 3 };
    const items = [_]Item{
        .{ .state = &s[0], .ids = ids[0..3], .logit_rows = 3, .taps = true },
        .{ .state = &s[1], .ids = ids[0..1], .logit_rows = 1 },
        .{ .state = &s[2], .ids = ids[0..2], .logit_rows = 0, .taps = true },
    };
    try std.testing.expectEqual(@as(u32, 0), logitBase(&items, 0));
    try std.testing.expectEqual(@as(u32, 3), logitBase(&items, 1));
    try std.testing.expectEqual(@as(u32, 4), logitBase(&items, 2));
    try std.testing.expectEqual(@as(u32, 3), tapBase(&items, 1));
    try std.testing.expectEqual(@as(u32, 3), tapBase(&items, 2));
}
