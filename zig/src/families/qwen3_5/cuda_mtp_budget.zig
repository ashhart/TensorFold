//! Keep zero-budget context absorption outside a scored MTP batch.
const std = @import("std");
const dr = @import("lanes").drafter;

pub fn active(d: anytype, items: []const dr.Absorb, out: []dr.Absorb) ![]const dr.Absorb {
    var n: usize = 0;
    for (items) |it| {
        if (!it.propose) {
            try d.absorbOne(it);
            continue;
        }
        out[n] = it;
        n += 1;
    }
    return out[0..n];
}

test "zero grant: Qwen CUDA MTP absorbs committed rows without scoring them" {
    const Fake = struct {
        rows: usize = 0,
        pub fn absorbOne(f: *@This(), it: dr.Absorb) !void {
            try std.testing.expect(!it.propose);
            f.rows += it.follow.len;
        }
    };
    var f: Fake = .{};
    var it: dr.Absorb = undefined;
    it.propose = false;
    it.follow = &.{ 13, 17 };
    var out: [3]dr.Absorb = undefined;
    for (0..180) |_| try std.testing.expectEqual(@as(usize, 0), (try active(&f, &.{it}, &out)).len);
    try std.testing.expectEqual(@as(usize, 360), f.rows);
    var useful = it;
    useful.propose = true;
    try std.testing.expectEqual(@as(usize, 2), (try active(&f, &.{ useful, it, useful }, &out)).len);
    for (out[0..2]) |q| try std.testing.expect(q.propose);
}
