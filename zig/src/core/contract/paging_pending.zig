//! Page-table launches and the changed attention image remain unqualified until their compiler usage is measured.
const std = @import("std");
const registry = @import("registry.zig");
const R = registry.CudaRecord;
const none = @as([5]@import("limits.zig").Usage, @splat(.{ .regs = 0, .static = 0 }));
pub const extra = [_]R{
    .{ .launch = .{ .name = "paging tf_nemo_kv_write", .image = "nemotron_attention", .symbol = "tf_nemo_kv_write", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "paging tf_nemo_kv_write_segs", .image = "nemotron_attention", .symbol = "tf_nemo_kv_write_segs", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "paging tf_nemo_attn_chunk", .image = "nemotron_attention", .symbol = "tf_nemo_attn_chunk", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "paging tf_nemo_attn_chunk_pf", .image = "nemotron_attention", .symbol = "tf_nemo_attn_chunk_pf", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "paging tf_nemo_attn_chunk_segs", .image = "nemotron_attention", .symbol = "tf_nemo_attn_chunk_segs", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "paging pattn128", .image = "prefill_attention", .symbol = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi8ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix", .threads = 256, .dynamic = 65536, .opt_in = true }, .usage = none },
};
/// The serving records whose image the paged attention changed, set unmeasured in `rows`.
pub fn clear(rows: []R) void {
    for (rows) |*r| if (std.mem.eql(u8, r.launch.image, "prefill_attention")) {
        r.usage = none;
    };
}
pub const current = blk: {
    @setEvalBranchQuota(100000);
    var rows = registry.cuda;
    clear(&rows);
    break :blk rows ++ extra;
};
test "paging records: eight paging launches stay red even when a later compiler has inherited overrides" {
    var pending: usize = 0;
    for (current) |r| if (r.usage[0].regs == 0) {
        pending += 1;
        try std.testing.expect(try registry.checkCuda(std.testing.allocator, &.{r}, true) > 0);
        try std.testing.expectEqualDeep(none, registry.usageFor(r, .{ .major = 13, .minor = 3 }));
    };
    try std.testing.expectEqual(@as(usize, 8), pending);
}
