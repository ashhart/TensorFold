//! The host tier: a rewritten page loses its frame, and ranks open only in lockstep.
const std = @import("std");
const kt = @import("kv_tier.zig");
const Rig = @import("kv_tier_test.zig").Rig;
const Lease = kt.Lease;
const expect = std.testing.expect;

test "a page whose host rows were rewritten loses its frame, so the next pin copies them in; a pinned one refuses" {
    var r: Rig = undefined;
    try r.open(4, 2);
    defer r.close();
    const t = &r.tier;
    var a = try r.lease(2, 2);
    defer t.finish(&a);
    const p = a.pages.ids.items;
    _ = try r.pin(&a, p[0]);
    r.write(&a, p[0], 4);
    r.flush(p[0]);
    try std.testing.expectError(error.Pinned, t.invalidate(p[0]));
    try t.unpin(&a, p[0]);
    @memset(r.tier.host_page(p[0]), 9); // the restore
    try t.invalidate(p[0]);
    try expect(t.resident(p[0]) == null);
    try t.audit();
    const f = try r.pin(&a, p[0]);
    try expect(f.miss);
    try t.invalidate(p[1]); // no frame: nothing to do
}

test "ranks open only on the same shape, slots and digest; a rank that differs in any is refused" {
    var a: Rig = undefined;
    var b: Rig = undefined;
    try a.open(8, 4);
    defer a.close();
    try b.open(8, 4);
    defer b.close();
    const va = kt.Lockstep.vector(&a.tier, 16);
    try kt.Lockstep.agree(va, kt.Lockstep.vector(&b.tier, 16));
    for ([_][2]usize{ .{ 8, 3 }, .{ 10, 4 } }) |shape| { // fewer frames, more pages
        var c: Rig = undefined;
        try c.open(shape[0], shape[1]);
        defer c.close();
        try std.testing.expectError(error.RanksDisagree, kt.Lockstep.agree(va, kt.Lockstep.vector(&c.tier, 16)));
    }
    try std.testing.expectError(error.RanksDisagree, kt.Lockstep.agree(va, kt.Lockstep.vector(&b.tier, 17)));
    var l = try b.lease(1, 1); // a different history: the digest moves
    defer b.tier.finish(&l);
    try std.testing.expectError(error.RanksDisagree, kt.Lockstep.agree(va, kt.Lockstep.vector(&b.tier, 16)));
}
