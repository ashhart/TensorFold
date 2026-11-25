//! A decode past topk_bounded.long index keys: each index layer's top-512 equals tf_topk512's on the same scores.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const fixture = @import("fixture.zig");

const testing = std.testing;
const Meta = pkg.scratch.Meta;
const p0 = pkg.pipelines.topk_bounded.long + 3616; // ratio 1 rows take the bounds, ratio 2 rows tf_topk512
const limit = p0 + 64;

test "dsv41 metal: a long decode's top-512, one slot and two, equals tf_topk512 on its own scores" {
    const a = testing.allocator;
    const r = try fixture.Rig.init(a, testing.io, limit, 43);
    defer r.deinit();
    var other = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
    defer other.deinit();
    fixture.fillPast(&r.c, &r.slot, p0, 5);
    fixture.fillPast(&r.c, &other, p0 - 7, 6);
    const want = try r.device.buffer(16 * 512 * 4, mtl.ResourceOptions.shared);
    defer want.deinit();
    var compared: usize = 0;
    for ([_]u32{ 1, 2 }) |slots| for (r.layers, 0..) |*w, li| {
        const l: u32 = @intCast(li);
        const ratio = r.c.compressRatio(l);
        if (ratio == 0 or !r.c.isIndexSource(l)) continue;
        const rows: u32 = if (slots == 1) 3 else 4;
        fixture.fillInput(r, rows, 11 + li);
        const spans = [2]pkg.scratch.Span{ .{ .p0 = p0, .n = 2 }, .{ .p0 = p0 - 7, .n = 2 } };
        const segs = [2]pkg.ctx.Seg{ .{ .slot = &r.slot, .p0 = p0, .r0 = 0, .n = 2 }, .{ .slot = &other, .p0 = p0 - 7, .r0 = 2, .n = 2 } };
        var x = r.ctx(&r.slot, p0, rows);
        if (slots == 2) {
            x.segs = &segs;
            try r.s.writeSpans(&r.c, &spans);
        } else try r.s.writeRows(&r.c, p0, rows);
        // the per-slot path packs a slot's score rows at its key count apart; the slot table at the scratch's width
        const width = if (slots == 1) x.keys(ratio) else r.s.keys;
        const s = r.streams();
        const pr = r.pres();
        const pass = fixture.Pass.begin(r.queue);
        try pkg.layer.encode(&x, pass.e, w, .{ .x = s[0], .pre = pr[0], .x_out = s[1], .pre_out = pr[1] });
        pass.e.barrier();
        pkg.glue.topk512(r.p.topk512, pass.e, r.s.scores, x.byRatio(Meta.vis, ratio), .{ .buf = want }, rows, width);
        try pass.finish();
        const got = r.s.sel.buf.contents()[r.s.sel.off..][0 .. rows * 512 * 4];
        try testing.expectEqualSlices(u8, want.contents()[0 .. rows * 512 * 4], got);
        compared += 1;
    };
    try testing.expectEqual(@as(usize, 6), compared);
}
