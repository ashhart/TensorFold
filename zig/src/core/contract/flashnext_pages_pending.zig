//! Paged Flash Next cache launches: compiler resources remain unmeasured.
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const none = registry.none;
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "flashnext_attn fn_attn_prep_paged", .symbol = "fn_attn_prep_paged", .image = "flashnext_attn", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_prep_media_paged", .symbol = "fn_attn_prep_media_paged", .image = "flashnext_attn", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "flashnext_attn fn_qsa_pool_paged", .symbol = "fn_qsa_pool_paged", .image = "flashnext_attn", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "flashnext_attn fn_qsa_pool_media_paged", .symbol = "fn_qsa_pool_media_paged", .image = "flashnext_attn", .threads = 128 }, .usage = none },
    .{ .launch = .{ .name = "flashnext_attn fn_qsa_scores_paged", .symbol = "fn_qsa_scores_paged", .image = "flashnext_attn", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_chunks_ahead_paged", .symbol = "fn_attn_chunks_ahead_paged", .image = "flashnext_attn", .threads = 128, .dynamic = 77312, .opt_in = true }, .usage = none },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_chunks_one_paged", .symbol = "fn_attn_chunks_one_paged", .image = "flashnext_attn", .threads = 128, .dynamic = 26624, .opt_in = true }, .usage = none },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_fused_paged", .symbol = "fn_attn_fused_paged", .image = "flashnext_attn", .threads = 128, .dynamic = 26624 }, .usage = none },
};
test "Flash Next paged launches: pending, no invented usage, never qualified without a compiler's records" {
    const t = @import("std").testing;
    for (records) |r| {
        try t.expectEqual(none, r.usage);
        try t.expectEqual(limits.sms.len, try registry.checkCuda(t.allocator, &.{r}, true));
        for (registry.cuda) |measured| try t.expect(!@import("std").mem.eql(u8, r.launch.symbol, measured.launch.symbol));
    }
}
