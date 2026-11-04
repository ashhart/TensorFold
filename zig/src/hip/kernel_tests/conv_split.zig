//! The linear attention's prompt conv with its q, k and v channels split into three contiguous outputs.

const std = @import("std");
const rig = @import("rig.zig");
const ops = @import("../ops/ops.zig");

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

test "the prompt conv's split q, k and v rows hold the bytes of its single output's columns" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0xC0418 };
    // the key width cuts the first 256-channel block, and the values end inside the second
    const len = 37;
    const kw = 96;
    const vw = 172;
    const channels = 2 * kw + vw;
    const kernel = 4;
    const kept = kernel - 1;
    const hx = try gpa.alloc(u16, len * channels);
    defer gpa.free(hx);
    for (hx) |*x| x.* = t.bits(rng.unit() * 2);
    const hw = try gpa.alloc(f32, channels * kernel);
    defer gpa.free(hw);
    for (hw) |*x| x.* = rng.unit() / 2;
    const hs = try gpa.alloc(f32, kept * channels);
    defer gpa.free(hs);
    for (hs) |*x| x.* = rng.unit();
    var x = try t.upload(hx);
    defer x.free();
    var w = try t.upload(hw);
    defer w.free();
    var state = try t.upload(hs);
    defer state.free();
    var out = try t.alloc(len * channels * 4);
    defer out.free();
    var next = try t.alloc(kept * channels * 4);
    defer next.free();
    var q = try t.alloc(len * kw * 4);
    defer q.free();
    var k = try t.alloc(len * kw * 4);
    defer k.free();
    var v = try t.alloc(len * vw * 4);
    defer v.free();
    var split_next = try t.alloc(kept * channels * 4);
    defer split_next.free();
    const xs: ops.Tensor = .{ .ptr = at(x), .kind = if (t.bf16) .bf16 else .f16 };
    try o.convPrefill(xs, at(w), at(state), at(out), at(next), len, channels, kernel, null);
    const split: ops.ConvSplit = .{ .k = at(k), .v = at(v), .key_width = kw };
    try o.convPrefill(xs, at(w), at(state), at(q), at(split_next), len, channels, kernel, split);
    try t.stream.synchronize();
    const go = try rig.download(u32, out);
    defer gpa.free(go);
    const gq = try rig.download(u32, q);
    defer gpa.free(gq);
    const gk = try rig.download(u32, k);
    defer gpa.free(gk);
    const gv = try rig.download(u32, v);
    defer gpa.free(gv);
    for (0..len) |r| {
        const row = go[r * channels ..][0..channels];
        try std.testing.expectEqualSlices(u32, row[0..kw], gq[r * kw ..][0..kw]);
        try std.testing.expectEqualSlices(u32, row[kw .. 2 * kw], gk[r * kw ..][0..kw]);
        try std.testing.expectEqualSlices(u32, row[2 * kw ..], gv[r * vw ..][0..vw]);
    }
    const gn = try rig.download(u32, next);
    defer gpa.free(gn);
    const gs = try rig.download(u32, split_next);
    defer gpa.free(gs);
    try std.testing.expectEqualSlices(u32, gn, gs);
}
