//! Pool lanes on pieces: open without two free pages side by side, entries on own state pages, close as at load.
const std = @import("std");
const cuda = @import("cuda");
const pages = @import("cuda_pages.zig");
const E = @import("cuda_engine.zig").Engine;
const D = cuda.paging_test_device.Device;
const t = std.testing;

fn engine(d: *cuda.Driver) E {
    var e: E = undefined;
    e.d = d;
    e.stream = .{ .d = d, .handle = D.stream };
    e.kv8 = false;
    e.n_taps = 0;
    e.g = .{ .hidden = 256, .intermediate = 256, .layers = 2, .vocab = 8, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 64, .linear = @splat(false) };
    e.g.linear[0] = true;
    return e;
}

test "Qwen pieces: no two free pages side by side, a lane still opens on its state pieces; close gives them back" {
    var d = D.driver();
    var e = engine(&d);
    defer cuda.completion.drain(&d, e.stream, "the test driver must drain before it leaves");
    const tokens = 64; // a page holds a whole recurrence (64 KiB here)
    const l = pages.layout(&e, 64, tokens, 0, 0, try pages.statePieces(t.allocator, &e, tokens));
    const need = l.need(1, 0, pages.window_rows);
    try t.expect(l.statePages() >= 2);
    const total = 2 * need + 2;
    const p = try pages.Pages.init(t.allocator, &e, l, 64, 0, 1, total);
    defer p.deinit();
    const load = p.pool.census();
    // the store holds every even page: the free ones are odd ids, none beside another
    const all = try t.allocator.alloc(u32, total);
    defer t.allocator.free(all);
    try t.expect(p.pool.hold(all));
    var id: u32 = 1;
    while (id < total) : (id += 2) try p.pool.drop(id);
    try t.expectEqual(@as(?u32, null), p.pool.findRun(2)); // today's run: LaneMemory
    try t.expect(p.fits(1, 0));
    const c = try p.open(1, 0);
    const x = c.pieces.?;
    try t.expectEqual(l.statePages(), x.ids.len);
    for (x.ids) |s| try t.expect(s % 2 == 1);
    // every entry of the lane's table: on one of its state pages, never on a page of its keys and values
    const page = l.pageBytes();
    for (c.states.?.host) |a| if (a != 0) {
        const pg: u32 = @intCast((a - x.base) / page);
        try t.expect(std.mem.indexOfScalar(u32, x.ids, pg) != null);
        try t.expect(std.mem.indexOfScalar(u32, c.paged.?.ids, pg) == null);
    };
    try t.expect(p.close(c));
    try t.expect(c.pieces == null);
    // a closed lane's table is refused, so no forward copies entries that point at pages it gave back
    try t.expectEqual(@as(u64, 0), c.states.?.version);
    try t.expectError(error.MissingQwenStates, c.stateTable(&e));
    id = 0;
    while (id < total) : (id += 2) try p.pool.drop(id);
    try t.expectEqual(load, p.pool.census());
    try t.expect(p.pool.idle());
}

test "Qwen pieces: a page below the largest state piece is refused by name at load" {
    var d = D.driver();
    var e = engine(&d);
    try t.expectError(error.QwenPageBelowState, pages.statePieces(t.allocator, &e, 16));
}
