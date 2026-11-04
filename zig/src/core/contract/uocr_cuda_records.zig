//! Unlimited-OCR-2's CUDA launches (uocr.cu), group by group as zig/tests/cuda/contract_uocr_*.zig registers them.
const std = @import("std");
const registry = @import("registry.zig");

pub const cuda = @import("uocr_cuda_rows_records.zig").cuda ++ @import("uocr_cuda_moves_records.zig").cuda ++ @import("uocr_cuda_route_records.zig").cuda ++ @import("uocr_cuda_attention_records.zig").cuda ++ @import("uocr_cuda_matvec_records.zig").cuda;

test "Unlimited-OCR-2 CUDA: every launch is in the live registry once and fits every SM" {
    const t = std.testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
