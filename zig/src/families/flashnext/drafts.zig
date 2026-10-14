//! The decode's draft widths: the depth rule over recent landing, and copy drafts from the token history.
const std = @import("std");

/// Round width from recent landing: 3 (prose), 6 while drafts land (code), the widest while nearly all land.
pub const DepthRule = struct {
    rate: f64 = 0.6, // moving average of the head's drafts landed over drafts offered
    depth: usize = 3,
    pair: bool = false, // speed-up mode: even windows only (an odd window's last row costs a Mac a whole expert row)

    pub fn pick(self: *const DepthRule) usize {
        return self.depth;
    }

    /// A head round's drafts and how many landed: 3, 6 or 8 on one Mac, 3 or 7 on the pair.
    pub fn update(self: *DepthRule, depth: usize, landed: usize) void {
        self.rate = 0.7 * self.rate + 0.3 * @as(f64, @floatFromInt(landed)) / @as(f64, @floatFromInt(depth));
        if (self.pair) {
            if (self.depth == 3 and self.rate > 0.95) self.depth = 7 else if (self.depth == 7 and self.rate < 0.8) self.depth = 3;
            return;
        }
        if (self.depth == 3 and self.rate > 0.8) {
            self.depth = 6;
        } else if (self.depth == 6 and self.rate > 0.74) {
            self.depth = 8;
        } else if (self.depth == 8 and self.rate < 0.62) {
            self.depth = 6;
        } else if (self.depth == 6 and self.rate < 0.65) self.depth = 3;
    }
};

test "the pair's depth rule keeps every window even" {
    var pair: DepthRule = .{ .pair = true };
    for (0..20) |_| pair.update(pair.pick(), pair.pick());
    try std.testing.expectEqual(@as(usize, 7), pair.pick());
    for (0..20) |_| pair.update(pair.pick(), 1);
    try std.testing.expectEqual(@as(usize, 3), pair.pick());
    var one: DepthRule = .{};
    for (0..20) |_| one.update(one.pick(), one.pick());
    try std.testing.expectEqual(@as(usize, 8), one.pick());
}

/// Copies into `out` what followed the latest earlier match of `hist`'s longest repeated suffix (`min`..8), or 0.
pub fn copyDrafts(hist: []const u32, min: usize, out: []u32) usize {
    const n = hist.len;
    if (n < min + 1) return 0;
    var len: usize = @min(8, n - 1);
    while (len >= min) : (len -= 1) {
        const suffix = hist[n - len ..];
        var end: usize = n - 1;
        while (end >= len) : (end -= 1) {
            if (std.mem.eql(u32, hist[end - len .. end], suffix)) {
                const k = @min(out.len, n - end);
                @memcpy(out[0..k], hist[end .. end + k]);
                return k;
            }
        }
    }
    return 0;
}

test {
    _ = @import("drafts_cost.zig");
}
