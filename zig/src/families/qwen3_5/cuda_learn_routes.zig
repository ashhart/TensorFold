//! Qualification-only routing evidence; finite differences keep serving's dynamic expert choices.
const std = @import("std");
pub const Counts = struct { rows: usize = 0, slots: usize = 0 };
/// Compare sets, so a changed ordering of the same selected experts is not a route boundary.
pub fn changes(before: []const i32, after: []const i32, top_k: usize) Counts {
    const slots = top_k + 1;
    std.debug.assert(before.len == after.len and before.len % slots == 0);
    var out: Counts = .{};
    for (0..before.len / slots) |r| {
        var missing: usize = 0;
        for (before[r * slots ..][0..top_k]) |id| {
            if (std.mem.indexOfScalar(i32, after[r * slots ..][0..top_k], id) == null) missing += 1;
        }
        out.rows += @intFromBool(missing > 0);
        out.slots += missing;
    }
    return out;
}
pub fn read(t: anytype, into: []i32) !void {
    const n = t.rows * (t.g.top_k + 1);
    std.debug.assert(into.len == t.g.layers * n);
    for (t.light, 0..) |layer, l| {
        try t.e.d.check(t.e.d.api.cuMemcpyDtoH_v2(into[l * n ..].ptr, layer.ids, n * 4), "qualifier routing read");
    }
}

test "MoE learning: routing evidence counts changed sets and excludes shared gates and pick ordering" {
    const a = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 255, 256, 20, 21, 22, 23, 24, 25, 26, 27, 256 };
    var b = a;
    std.mem.swap(i32, &b[0], &b[7]);
    b[8] = 999;
    try std.testing.expectEqual(Counts{}, changes(&a, &b, 8));
    b[1] = 128;
    b[9] = 129;
    b[10] = 130;
    try std.testing.expectEqual(Counts{ .rows = 2, .slots = 3 }, changes(&a, &b, 8));
}
