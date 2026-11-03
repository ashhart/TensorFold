//! Unlimited-OCR-2's Metal launches, one file a kernel group.
const std = @import("std");
pub const records = @import("uocr_rows_records.zig").records ++ @import("uocr_moves_records.zig").records ++ @import("uocr_attention_records.zig").records ++ @import("uocr_route_records.zig").records ++ @import("uocr_matvec_records.zig").records;

test "Unlimited-OCR-2's launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
