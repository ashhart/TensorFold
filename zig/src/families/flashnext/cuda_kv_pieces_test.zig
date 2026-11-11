//! TF_FLASHNEXT_STATE_PAGES: a lane's states and draft buffers as pieces on any pages of its lease, on a fake device.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const lanes = @import("cuda_kv_lanes.zig");
const mtp = @import("cuda_mtp.zig");
const wt = @import("cuda_prompt_wt.zig");
const E = @import("cuda_prompt.zig").Engine;
const D = cuda.paging_test_device.Device;
const Draft = @import("cuda_mtp_draft.zig").Draft;
const t = std.testing;

const head: lanes.Head = .{ .draft_n = 32768, .rows = 1536 };

var queue: cuda.Stream = undefined;
fn engine(d: *cuda.Driver) E {
    var e: E = undefined;
    e.driver = d;
    queue = .{ .d = d, .handle = D.stream };
    e.stream = &queue;
    e.eos = 0;
    e.round_graph = .{};
    return e;
}

test "TF_FLASHNEXT_STATE_PAGES is exact 0 or 1: words, 2 and padded text are refused by name, blank is unset" {
    try t.expect(!try lanes.statePagesOf(null));
    try t.expect(!try lanes.statePagesOf(" "));
    try t.expect(!try lanes.statePagesOf("0"));
    try t.expect(try lanes.statePagesOf("1"));
    for ([_][]const u8{ "off", "false", "no", "on", "true", "yes", "2", " 0", "1 ", "01" }) |v|
        try t.expectError(error.BadFlashNextStatePages, lanes.statePagesOf(v));
}

fn withLate(on: bool) bool {
    const was = Draft.late_on;
    Draft.late_on = on;
    return was;
}

test "Flash Next pieces at 256-row pages: streams and logits by row, the run's pages exactly, nothing outside" {
    const was = withLate(false);
    defer Draft.late_on = was;
    const run = lanes.layout(40960, 256, head, 0);
    const l = try lanes.layoutPieces(40960, 256, head, 0);
    try t.expectEqual(@as(usize, 41), run.statePages());
    try t.expectEqual(kv.groups.Pieces{ .n = 266, .pages = 41, .bytes = 270_008_832 + 31_457_280 + 15_892_480 }, l.pieces.?);
    try t.expectEqual(run.statePages(), l.statePages());
    try t.expectEqual(@as(u64, 0), l.run_bytes);
    try t.expectEqual(@as(u64, 0), l.outside);
    try t.expectEqual(run.fixed_bytes, l.fixed_bytes);
    try t.expectEqual(run.pageBytes(), l.pageBytes());
    Draft.late_on = true;
    const late = try lanes.layoutPieces(40960, 256, head, 0);
    try t.expectEqual(@as(u32, 303), late.pieces.?.n);
    try t.expectEqual(@as(u32, 48), late.pieces.?.pages);
    try t.expectEqual(lanes.layout(40960, 256, head, 0).statePages(), late.statePages());
    try t.expectEqual(@as(u64, 0), late.outside);
    const plain = try lanes.layoutPieces(40960, 256, null, 0);
    try t.expectEqual(@as(u32, 73), plain.pieces.?.n);
    try t.expectEqual(@as(u32, 18), plain.pieces.?.pages);
    try t.expectEqual(@as(u64, 0), plain.outside);
    try t.expectError(error.PieceOverPage, lanes.layoutPieces(40960, 64, null, 0)); // a 3 MiB state past a page
}

test "the smallest page that holds a lane's largest piece: 104 rows with the head, 112 without; one step less refuses" {
    try t.expectEqual(@as(u64, 3 << 20), lanes.largestPiece(head)); // a layer's DeltaNet state
    try t.expectEqual(@as(u64, 3 << 20), lanes.largestPiece(null));
    try t.expectEqual(@as(usize, 104), lanes.smallestRows(head));
    try t.expectEqual(@as(usize, 112), lanes.smallestRows(null));
    for ([_]?lanes.Head{ head, null }) |h| {
        const n = lanes.smallestRows(h);
        _ = try lanes.layoutPieces(40960, n, h, 0);
        try t.expectError(error.PieceOverPage, lanes.layoutPieces(40960, n - wt.index_ratio, h, 0));
    }
    const why = @import("cuda_lanes_explain.zig").explain(error.PieceOverPage).?;
    try t.expect(std.mem.indexOf(u8, why, "128") == null); // no fixed floor: the log names the page
}

/// The lane's pieces as [ptr, ptr + len) ranges: its state, conv tails, tail; with `d`, the head's too.
fn ranges(a: std.mem.Allocator, seq: anytype, d: ?*const Draft) !std.ArrayList([2]u64) {
    var r: std.ArrayList([2]u64) = .empty;
    errdefer r.deinit(a);
    for (seq.state) |b| try r.append(a, .{ b.ptr, b.ptr + b.len });
    for (seq.conv) |b| try r.append(a, .{ b.ptr, b.ptr + b.len });
    try r.append(a, .{ seq.tail.ptr, seq.tail.ptr + seq.tail.len });
    const x = d orelse return r;
    for (x.spare) |b| try r.append(a, .{ b.ptr, b.ptr + b.len });
    for (x.keep.layers, 0..) |lk, layer| if (wt.linearLayer(layer)) {
        try r.append(a, .{ lk.hist, lk.hist + x.layout.conv });
        try r.append(a, .{ lk.proj, lk.proj + mtp.max_rows * x.layout.proj });
    };
    try r.append(a, .{ x.saved_tail, x.saved_tail + x.layout.nrow_at - x.layout.tail_at });
    try r.append(a, .{ x.saved_nrow, x.saved_nrow + x.layout.saved - x.layout.nrow_at });
    for ([_]cuda.DeviceBuffer{ x.out, x.dlogits, x.best }) |b| try r.append(a, .{ b.ptr, b.ptr + b.len });
    const rows = Draft.streamRows(x.rows_cap);
    for (x.stream_blocks, 0..) |b, k| try r.append(a, .{ b, b + @min(rows.block, rows.rows - k * rows.block) * rows.row });
    if (x.logit_paged) for (x.logit_rows) |b| try r.append(a, .{ b, b + wt.vocab * 2 });
    return r;
}

/// Every range inside one of `ids`' pages of the arena, and no two ranges overlapping.
fn expectPlaced(r: [][2]u64, base: u64, page: u64, ids: []const u32) !void {
    for (r) |x| {
        const at = (x[0] - base) / page;
        try t.expect(std.mem.indexOfScalar(u32, ids, @intCast(at)) != null);
        try t.expect(x[1] <= base + (at + 1) * page);
    }
    std.mem.sort([2]u64, r, {}, struct {
        fn less(_: void, a: [2]u64, b: [2]u64) bool {
            return a[0] < b[0];
        }
    }.less);
    for (r[0 .. r.len - 1], r[1..]) |a, b| try t.expect(a[1] <= b[0]);
}

test "fragmentation: no free consecutive run, a run lane waits, a pieces lane opens on the scattered pages" {
    var d = D.driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
    for ([_]bool{ false, true }) |pieced| {
        const l = if (pieced) try lanes.layoutPieces(1024, 256, null, 0) else lanes.layout(1024, 256, null, 0);
        const k = l.statePages();
        const count = 2 * (k + 1) + 2;
        const p = try lanes.Pool.init(t.allocator, &e, l, 1024, null, 1, count);
        defer p.deinit();
        const store = try t.allocator.alloc(u32, count);
        defer t.allocator.free(store);
        try t.expect(p.pool.hold(store));
        var i: u32 = 1;
        while (i < count) : (i += 2) try p.pool.drop(i); // the store keeps every other page: no two free side by side
        if (!pieced) {
            try t.expectError(error.LaneMemory, p.open(1));
            try t.expect(!p.fits(1));
        } else {
            try t.expect(p.fits(1));
            var seq = try p.open(1);
            var r = try ranges(t.allocator, &seq, null);
            defer r.deinit(t.allocator);
            const x = &p.lanes[0];
            for (x.lease.state.items) |id| try t.expect(id % 2 == 1);
            try expectPlaced(r.items, p.arena.ptr, l.pageBytes(), x.lease.state.items);
            try t.expect(p.close(&seq));
        }
        i = 0;
        while (i < count) : (i += 2) try p.pool.drop(i);
        try t.expect(p.pool.idle());
    }
}

test "two lanes with the head: every piece inside its own lane's state pages, none on another's or on KV pages" {
    const was = withLate(true);
    defer Draft.late_on = was;
    var d = D.driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
    const l = try lanes.layoutPieces(1024, 256, head, 0);
    const p = try lanes.Pool.init(t.allocator, &e, l, 1024, head, 2, 2 * (l.statePages() + 6));
    defer p.deinit();
    var seqs: [2]@import("cuda_prompt.zig").Seq = undefined;
    var drafts: [2]Draft = undefined;
    var all: std.ArrayList([2]u64) = .empty;
    defer all.deinit(t.allocator);
    for (&seqs, &drafts, 0..) |*s, *dr, j| {
        s.* = try p.open(1024);
        dr.* = try p.draft(s, 1024);
        var r = try ranges(t.allocator, s, dr);
        defer r.deinit(t.allocator);
        try expectPlaced(r.items, p.arena.ptr, l.pageBytes(), p.lanes[j].lease.state.items);
        try all.appendSlice(t.allocator, r.items);
        try t.expectEqual(@as(usize, 48), dr.stream_blocks.len);
        try t.expectEqual(dr.stream_blocks[0], dr.streams.ptr);
        try t.expectEqual(dr.logit_rows[0], dr.logits.ptr);
        for (0..dr.rows_cap) |i| { // a row's address: in its block, rows of a block side by side
            const b = dr.stream_blocks[i / mtp.max_rows];
            try t.expectEqual(b + (i % mtp.max_rows) * wt.wide * 2, dr.streamAt(i));
        }
        try t.expect(dr.log != null and dr.keep.layers[0].log_ctl == dr.log.?.ptr);
        for (p.lanes[j].lease.ids.items) |id| try t.expect(std.mem.indexOfScalar(u32, p.lanes[j].lease.state.items, id) == null);
    }
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(t.allocator);
    for (p.lanes) |x| try ids.appendSlice(t.allocator, x.lease.state.items);
    try expectPlaced(all.items, p.arena.ptr, l.pageBytes(), ids.items);
    for (&seqs, &drafts) |*s, *dr| {
        dr.deinit();
        try t.expect(p.close(s));
    }
    try t.expect(p.pool.idle());
}

test "LEAK: a failed open on pieces leaves the pool and the device as before; a moved packing is refused" {
    var d = D.driver();
    var e = engine(&d);
    defer {
        D.fail_wait = false;
        cuda.completion.drain(&d, e.stream.*, "the test driver must drain before it leaves");
    }
    const l = try lanes.layoutPieces(1024, 256, null, 0);
    const p = try lanes.Pool.init(t.allocator, &e, l, 1024, null, 1, l.statePages() + 6);
    defer p.deinit();
    const load = p.pool.census();
    D.fail_wait = true; // the table upload's wait fails: open unwinds every page it took
    try t.expectError(error.NotReady, p.open(1));
    D.fail_wait = false;
    try t.expectEqual(load, p.pool.census());
    D.fail_fill = true; // the state clear fails after its pages and table are in
    try t.expectError(error.NotReady, p.open(1));
    D.fail_wait = false;
    try t.expectEqual(load, p.pool.census());
    var seq = try p.open(1);
    try t.expect(p.close(&seq));
    try t.expectEqual(load, p.pool.census());
    var bad = l;
    bad.pieces.?.pages += 1; // a layout from another packing is refused before any lane exists
    try t.expectError(error.FlashNextPiecesMoved, lanes.Pool.init(t.allocator, &e, bad, 1024, null, 1, 8));
}
