//! DSpark head snapshots: a restored context drafts what the stream it was copied from drafts, past window and ring.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const draft = @import("metal_draft.zig");
const fxm = @import("metal_draft_fixture.zig");
const probe = @import("metal_draft_test.zig");

const t = std.testing;
const dr = lanes.drafter;

/// Row `i` of the device buffer as a decode round's kept row: the head keeps it on the host until a hold needs it.
fn deviceRow(s: *lanes.Stream, device: mtl.Buffer, i: u64, row_bytes: u32) dr.Absorb {
    return .{ .stream = s, .features = .{ .buffer = @intFromPtr(device.id), .offset = i * row_bytes, .rows = 1, .row_bytes = row_bytes, .space = .device, .dtype = .bf16 }, .start = i, .follow = &.{} };
}

test "a context restored from a snapshot drafts what its source drafts, rows absorbed after it included" {
    const g = try probe.Gpu.init();
    defer g.deinit();
    const fx = try fxm.Fixture.init(t.allocator, t.io, g.device);
    defer fx.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const D = fx.d.D;
    const row_bytes: u32 = @intCast(fx.d.taps * D * 2);
    const words = fx.d.taps * D;
    const ad = try draft.Adapter.init(t.allocator, t.io, fx.dir, g.device, g.queue, &fx.cfg, fx.tensors(), .{ .streams = 3, .capacity = 1300 });
    defer ad.deinit();
    const d = ad.drafter();
    const rows = try fxm.features(a, fx.d, 1240, 5);
    const device = try g.device.buffer(rows.len * 2, mtl.ResourceOptions.shared);
    defer device.deinit();
    @memcpy(device.slice(u16, rows.len), rows);
    const m = fx.cfg.mtp.?;
    const slot: u64 = 2 * 128 * 2; // a position's key (or value) in a layer: the fixture's 2 key/value heads of 128, bf16
    try t.expectEqual(2 * slot * 3 * 6, draft.Snapshot.bytes(m, 6));
    try t.expectEqual(2 * slot * (1170 + 2 * 1024), draft.Snapshot.bytes(m, 1170)); // sliding layers keep their window
    // a context before the window, and one past the ring's 1152 slots (its oldest positions' slots were rewritten)
    for ([_]u32{ 6, 1170 }) |at| {
        var s1: lanes.Stream = undefined;
        var s2: lanes.Stream = undefined;
        var s3: lanes.Stream = undefined;
        try d.open(&s1);
        try d.open(&s2);
        try d.open(&s3);
        // all but the last 3 rows in one pass, the last 3 kept on the host as decode rows are
        try probe.absorb(d, &s1, rows, at - 3, 0, row_bytes);
        for (at - 3..at) |i| try d.absorb(&.{deviceRow(&s1, device, i, row_bytes)});
        try t.expectEqual(@as(u32, 3), ad.lanes.get(&s1).?.pend_rows);
        var saved = (try ad.save(&s1, at)).?;
        defer saved.deinit();
        try t.expectEqual(@as(u32, 0), ad.lanes.get(&s1).?.pend_rows); // the copy took them in
        try ad.restore(&s2, &saved);
        try t.expectError(error.DraftSnapshotGeometry, ad.restore(&s2, &saved)); // a context in place is not overwritten
        // the same rows after it: the source and the restored stream hold one context, so a block drafts the same bits
        for ([_]*lanes.Stream{ &s1, &s2 }) |s| try probe.absorb(d, s, rows[at * words ..], 60, at, row_bytes);
        const want = try probe.drafts(d, &s1, 77, at + 60);
        const bits = try a.dupe(u16, probe.xBits(ad, 7, D));
        try t.expectEqual(want, try probe.drafts(d, &s2, 77, at + 60));
        try t.expectEqualSlices(u16, bits, probe.xBits(ad, 7, D));
        // a stream that begins past the prefix (no restore) holds another context: the copy is not vacuous
        try probe.absorb(d, &s3, rows[at * words ..], 60, at, row_bytes);
        _ = try probe.drafts(d, &s3, 77, at + 60);
        try t.expect(!std.mem.eql(u16, bits, probe.xBits(ad, 7, D)));
        try t.expect((try ad.save(&s3, at + 60)) == null); // it holds no whole context to copy
        try t.expect((try ad.save(&s1, at)) == null); // its end is not `at`
        d.release(&s1);
        d.release(&s2);
        d.release(&s3);
    }
}
