//! Run the bundled three-layer head using the original shared embedding/unembedding and deterministic tap rows.
const std = @import("std");
const lanes = @import("lanes");
const ck = @import("../../core/checkpoint_metal.zig");
const fixture = @import("metal_media_checkpoint_fixture.zig");
const draft = @import("metal_draft.zig");
const fx = @import("metal_draft_fixture.zig");
const probe = @import("metal_draft_test.zig");
const t = std.testing;

test "original Mini bundled DSpark rows and Markov scores match the FP64 reference" {
    const dir = try fixture.directory();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = try fixture.loadConfig(a, dir);
    const d = fx.Dims.of(&cfg);
    try t.expectEqual(@as(usize, 3), d.layers);
    try t.expectEqual(@as(usize, 256), d.rank);
    try t.expectEqual(@as(usize, 201024), d.V);
    try t.expectEqual([3]bool{ true, false, true }, d.local[0..3].*);
    const gpu = try probe.Gpu.init();
    defer gpu.deinit();
    var store = ck.Checkpoint.init(t.allocator);
    defer store.deinit();
    const embed_path = try fixture.shard(a, dir, "model.llm.embed.weight");
    const unembed_path = try fixture.shard(a, dir, "model.llm.unembed.weight");
    try store.addFileSelected(gpu.device, embed_path, "", "model.llm.embed");
    try store.addFileSelected(gpu.device, unembed_path, "", "model.llm.unembed.weight");
    const target: draft.Target = .{
        .embed = try store.get("model.llm.embed.weight"),
        .embed_norm = try store.get("model.llm.embed_norm.weight"),
        .unembed = try store.get("model.llm.unembed.weight"),
    };
    const ad = try draft.Adapter.init(t.allocator, t.io, dir, gpu.device, gpu.queue, &cfg, target, .{ .capacity = 16 });
    defer ad.deinit();
    try t.expectEqual(@as(usize, 54), ad.core.w.ckpt.tensors.count());
    const dr = ad.drafter();
    try t.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, dr.taps());
    var stream: lanes.Stream = undefined;
    try dr.open(&stream);
    defer dr.release(&stream);
    const n = 3;
    const pitch = d.D * d.taps;
    const feats = try fx.features(a, d, n, 0x1a22);
    const anchor: u32 = 1234;
    try probe.absorb(dr, &stream, feats, n, 0, @intCast(pitch * 2));
    const ids = try probe.drafts(dr, &stream, anchor, n);
    const r = fx.Reference.fromCheckpoint(&ad.core.w.ckpt, &cfg, target);
    const rows = try r.blockRows(a, feats, n, anchor);
    const got = probe.xBits(ad, d.block, d.D);
    for (0..d.block) |i| {
        const err = probe.relError(got[i * d.D ..][0..d.D], rows[i * d.D ..][0..d.D]);
        if (err >= 0.05) std.debug.print("draft row {d}: relative RMS error {d}\n", .{ i, err });
        try t.expect(err < 0.05);
    }
    for (ids) |id| try t.expect(id < d.unpadded);
    const candidates = [_]u32{ 0, 1, 17, 123, 812, 1024, 1234, 4230, 8088, 200006, 200057 };
    var top: [7]f64 = undefined;
    var chosen: [7]f64 = undefined;
    try r.steps(a, rows, feats[(n - 1) * pitch ..], anchor, &ids, &candidates, &top, &chosen);
    for (chosen, top) |score, best| try t.expect(score >= best - 0.5);
    // The last step retains the GPU's per-group winners; check its global winner's score too.
    const last = ids[ids.len - 1];
    const count = (d.V + 7) / 8;
    try t.expectEqual(last, ad.core.buf("best_id").slice(u32, count)[last / 8]);
    try t.expectApproxEqAbs(chosen[6], @as(f64, ad.core.buf("best").slice(f32, count)[last / 8]), 0.5);
}
