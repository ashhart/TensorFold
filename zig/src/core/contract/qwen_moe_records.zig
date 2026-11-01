//! Changed pocket ABIs need fresh resource receipts on every SM and compiler.
const std = @import("std");
const r = @import("registry.zig");
pub fn changed(symbol: []const u8) bool {
    return std.mem.eql(u8, symbol, "tf_q27_pocket_rows") or std.mem.eql(u8, symbol, "tf_q27_pocket_back");
}
pub fn pending(comptime records: anytype) @TypeOf(records) {
    var out = records;
    for (&out) |*record| if (changed(record.launch.symbol)) {
        record.usage = r.none;
    };
    return out;
}
test "MoE learning: changed pocket ABIs never reuse historical resource receipts" {
    var found: usize = 0;
    for (r.all_cuda) |record| if (changed(record.launch.symbol)) {
        found += 1;
        try std.testing.expectEqual(r.none, record.usage);
        try std.testing.expectEqual(r.none, r.usageFor(record, .{ .major = 13, .minor = 3 }));
    };
    try std.testing.expectEqual(@as(usize, 2), found);
}
