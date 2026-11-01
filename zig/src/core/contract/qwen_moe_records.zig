//! Changed pocket ABIs need fresh resource receipts on every SM and compiler.
const std = @import("std");
const r = @import("registry.zig");
pub fn changed(symbol: []const u8) bool {
    return std.mem.eql(u8, symbol, "tf_q27_pocket_rows") or std.mem.eql(u8, symbol, "tf_q27_pocket_back") or std.mem.eql(u8, symbol, "tf_q4_swiglu_back") or std.mem.eql(u8, symbol, "tf_q4_route_back") or std.mem.startsWith(u8, symbol, "tf_q4_experts_nt_packed") or std.mem.startsWith(u8, symbol, "tf_q4_experts_mm_packed");
}
pub const extra = [_]r.CudaRecord{
    .{ .launch = .{ .name = "train_q4 tf_q4_experts_nt_packed64 threads 64", .image = "train_q4", .symbol = "tf_q4_experts_nt_packed64", .threads = 64 }, .usage = r.none },
    .{ .launch = .{ .name = "train_q4 tf_q4_experts_mm_packed64 threads 64", .image = "train_q4", .symbol = "tf_q4_experts_mm_packed64", .threads = 64 }, .usage = r.none },
};
pub fn pending(comptime records: anytype) [records.len + extra.len]r.CudaRecord {
    @setEvalBranchQuota(100000);
    var out = records ++ extra;
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
    try std.testing.expectEqual(@as(usize, 8), found);
}
