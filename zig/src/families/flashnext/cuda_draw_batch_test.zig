//! Shared verify draws compared with the production per-stream path on the recording device.
const std = @import("std");
const cuda = @import("cuda");
const mtp = @import("cuda_mtp.zig");
const draw = @import("cuda_mtp_draw.zig");
const batch = @import("cuda_draw_batch.zig");
const f = @import("cuda_mtp_nonfinite_fixture.zig");
const vocab = @import("cuda_prompt_wt.zig").vocab;
const D = f.Device;
const t = std.testing;
const a = t.allocator;

test "Flash Next batch draws: eight mixed streams match serial draws, two grids and two waits" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var h = try f.head(&driver);
    defer h.artifact.deinit();
    const host = try a.alloc(u16, 8 * vocab);
    defer a.free(host);
    const scores = try a.alloc(u16, 8 * vocab);
    defer a.free(scores);
    @memset(scores, 0xff80);
    var vals: [8 * 20]u16 = undefined;
    var cols: [8 * 20]i32 = undefined;
    for (0..8) |r| for (0..20) |i| {
        const v: u16 = @intCast(@as(u32, @bitCast(@as(f32, @floatFromInt(i)) / 10)) >> 16);
        vals[r * 20 + i] = v;
        cols[r * 20 + i] = @intCast(i);
        scores[r * vocab + i] = v;
    };
    D.verify_rows = scores;
    D.vals = &vals;
    D.cols = &cols;
    D.best = f.reduction(mtp.promote(vals[19]), 4, 19);
    var ds: [8]mtp.Draft = undefined;
    var refs: [8]*mtp.Draft = undefined;
    var sms: [8]?mtp.Sampling = undefined;
    var masks: [8][]const u32 = @splat(&.{});
    var baseline: [8][1]u32 = undefined;
    var got: [8][1]u32 = undefined;
    var outs: [8][]u32 = undefined;
    for (&ds, &refs, &sms, &outs, 0..) |*d, *ref, *sm, *out, i| {
        d.* = f.draft(&driver, host[i * vocab ..][0..vocab]);
        d.logits.ptr += i * vocab * 2;
        d.fed0 += 13 * i;
        ref.* = d;
        sm.* = if (i == 6) null else .{ .seed = i * 7 + 19, .temperature = 0.7, .top_k = 20, .top_p = 0.95 };
        out.* = &got[i];
    }
    masks[7] = &.{1 << 19};
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    for (refs, sms, masks, &baseline) |d, sm, mask, *out| try draw.draws(&h, st, a, d, sm, mask, out);
    D.reset();
    try batch.drawsMany(&h, st, a, 0x4000, &refs, &sms, &masks, &outs);
    try t.expectEqualDeep(baseline, got);
    try t.expectEqualSlices(usize, &.{ 1, 2 }, D.functions[0..D.launches]);
    try t.expectEqualSlices(u32, &.{ 8, 8 }, D.grids[0..D.launches]);
    try t.expectEqual(@as(usize, 2), D.waits);
    try t.expectEqual(@as(usize, 2), D.reads);
    // A mismatched K uses its complete smaller set; never a prefix of the larger column-ordered set.
    sms[1].?.top_k = 2;
    sms[1].?.top_p = 1;
    D.single_vals = vals[18..20];
    D.single_cols = cols[18..20];
    try draw.draws(&h, st, a, refs[1], sms[1], &.{}, &baseline[1]);
    // The recording per-stream result and the batch result both see that explicit two-candidate set.
    D.reset();
    try batch.drawsMany(&h, st, a, 0x4000, &refs, &sms, &masks, &outs);
    try t.expectEqualDeep(baseline[1], got[1]);
    try t.expectEqual(@as(usize, 3), D.waits);
    // Switch off restores each window's actual path; no group reduction is queued.
    h.draw_batch.on = false;
    D.reset();
    try batch.drawsMany(&h, st, a, 0x4000, &refs, &sms, &masks, &outs);
    try t.expectEqual(@as(usize, 8), D.waits);
}

test "Flash Next batch draws: a later bad row keeps stream publication order and leaves its output untouched" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var h = try f.head(&driver);
    defer h.artifact.deinit();
    var hosts: [2][128]u16 = undefined;
    var ds = [2]mtp.Draft{ f.draft(&driver, &hosts[0]), f.draft(&driver, &hosts[1]) };
    ds[1].rows = 2;
    ds[1].logits.ptr += vocab * 2;
    const sm: mtp.Sampling = .{ .seed = 71, .temperature = 1, .top_k = 2 };
    D.vals = &.{ 0x3f80, 0, 0x3f80, 0, 0x3f80, 0 };
    D.cols = &.{ 17, 5, 17, 5, -1, 5 };
    var first = [_]u32{99};
    var last = [_]u32{ 42, 43 };
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    try t.expectError(error.ProposalNonfinite, batch.drawsMany(&h, st, a, 0x4000, &.{ &ds[0], &ds[1] }, &.{ sm, sm }, &.{ &.{}, &.{} }, &.{ &first, &last }));
    try t.expect(first[0] != 99);
    try t.expectEqualSlices(u32, &.{ 42, 43 }, &last);
    try t.expectEqualSlices(usize, &.{ 2, 1 }, D.functions[0..D.launches]);
    try t.expectEqualSlices(u32, &.{ 3, 2 }, D.grids[0..D.launches]);
}
