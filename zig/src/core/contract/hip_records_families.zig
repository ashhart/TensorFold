//! The HIP families' gate-5 records, one array a family (`hip_records_<family>.zig`); hip_records.zig joins them.
const std = @import("std");
const Record = @import("hip_records.zig").Record;

pub const records = .{
    @import("hip_records_glm_ocr.zig").records,
    @import("hip_records_joint.zig").records,
    @import("hip_records_lighton_ocr.zig").records,
    @import("hip_records_qwen_vision.zig").records,
    @import("hip_records_relu2_widths.zig").records,
};

/// Every family's records as one array.
pub const all = join: {
    var joined: []const Record = &.{};
    for (records) |list| joined = joined ++ list;
    break :join joined[0..joined.len].*;
};

test "HIP record counts stay pinned in their owning files" {
    const hip = @import("hip_records.zig");
    const counts = .{
        @import("hip_records_columns.zig").records.len,
        @import("hip_records_dense_text.zig").records.len,
        @import("hip_records_draft.zig").records.len,
        @import("hip_records_embed.zig").records.len,
        @import("hip_records_gdn.zig").records.len,
        @import("hip_records_glm_ocr.zig").records.len,
        @import("hip_records_item.zig").records.len,
        @import("hip_records_joint.zig").records.len,
        @import("hip_records_lighton_ocr.zig").records.len,
        @import("hip_records_pair.zig").records.len,
        @import("hip_records_prompt.zig").records.len,
        @import("hip_records_qwen3.zig").records.len,
        @import("hip_records_qwen_vision.zig").records.len,
        @import("hip_records_relu2_widths.zig").records.len,
        @import("hip_records_router.zig").records.len,
        @import("hip_records_score.zig").records.len,
        @import("hip_records_skew.zig").records.len,
        @import("hip_records_sparse.zig").records.len,
        @import("hip_records_stream_plain.zig").records.len,
        @import("hip_records_stream_swiglu.zig").records.len,
        @import("hip_records_tail.zig").records.len,
        @import("hip_records_tap.zig").records.len,
        @import("hip_records_vision.zig").records.len,
    };
    var expected: usize = hip.own_count;
    inline for (counts) |count| expected += count;
    try std.testing.expectEqual(expected, hip.records.len);
}
