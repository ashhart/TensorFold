//! The DeltaNet head-split launches (sm_120 only): geometry recorded, resources pending until measured.
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_front_h", .symbol = "fn_gdn_front_h", .image = "flashnext_batch", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_rows_h", .symbol = "fn_gdn_rows_h", .image = "flashnext_batch", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_rows_to_h", .symbol = "fn_gdn_rows_to_h", .image = "flashnext_batch", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_batch fn_gdn_norm_h", .symbol = "fn_gdn_norm_h", .image = "flashnext_batch", .threads = 128 }, .usage = registry.none },
};

test "Flash Next DeltaNet split launches: unmeasured, never passing a compiled resource check" {
    const t = @import("std").testing;
    for (records) |r| {
        try t.expect(registry.usageFor(r, .{ .major = 13, .minor = 4 })[0].regs == 0);
        for (registry.cuda) |known| try t.expect(!@import("std").mem.eql(u8, r.launch.symbol, known.launch.symbol));
    }
}
