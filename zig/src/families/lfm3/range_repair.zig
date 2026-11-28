//! TF_LFM3_PARALLEL_REPAIR: parallel range repair with the serial replay's exact arithmetic order.
const std = @import("std");
const sys = @import("sys");
const mtl = @import("metal");
const c = @import("config.zig");
const st = @import("state.zig");
const kn = @import("kernels.zig");

pub fn setting(raw: ?[]const u8) !bool {
    return sys.environment.value(bool, "TF_LFM3_PARALLEL_REPAIR", raw, false) catch return error.BadRangeRepair;
}

pub fn enabled() !bool {
    return sys.environment.read(bool, "TF_LFM3_PARALLEL_REPAIR", false);
}

fn groups(e: mtl.ComputeEncoder, grid: [3]usize, width: usize) void {
    e.dispatchGroups(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(width, 1, 1));
}

pub fn full(e: mtl.ComputeEncoder, k: *const kn.Kernels, s: *st.Scratch, args: kn.AttentionArgs, cache: st.Attention, query_offset: usize, output_offset: usize, rows: usize, used: usize) void {
    e.useResource(cache.keys, mtl.ResourceUsage.read);
    e.useResource(cache.values, mtl.ResourceUsage.read);
    e.setPipeline(k.get("lfm3_attn_merge_flagged"));
    e.setBuffer(s.pm, 0, 0);
    e.setBuffer(s.pl, 0, 1);
    e.setBuffer(s.po, 0, 2);
    e.setBytes(std.mem.asBytes(&args), 3);
    e.setBuffer(s.y, output_offset, 4);
    e.setBuffer(s.attention_risk, 0, 5);
    e.setBuffer(s.repair_flags.?, 0, 6);
    groups(e, .{ c.query_heads, rows, 1 }, 32);
    e.setPipeline(k.get("lfm3_attn_repair_partial"));
    e.setBuffer(s.queries, query_offset, 0);
    e.setBuffer(cache.keys, 0, 1);
    e.setBuffer(cache.values, 0, 2);
    e.setBytes(std.mem.asBytes(&args), 3);
    e.setBuffer(s.pm, 0, 4);
    e.setBuffer(s.pl, 0, 5);
    e.setBuffer(s.po, 0, 6);
    e.setBuffer(s.repair_exp.?, 0, 7);
    e.setBuffer(s.repair_flags.?, 0, 8);
    groups(e, .{ c.query_heads, rows, used }, 128);
    e.setPipeline(k.get("lfm3_attn_repair_merge"));
    e.setBuffer(s.pm, 0, 0);
    e.setBuffer(s.pl, 0, 1);
    e.setBuffer(s.po, 0, 2);
    e.setBuffer(s.repair_exp.?, 0, 3);
    e.setBytes(std.mem.asBytes(&args), 4);
    e.setBuffer(s.y, output_offset, 5);
    e.setBuffer(s.repair_flags.?, 0, 6);
    groups(e, .{ c.query_heads, rows, 1 }, 32);
}

pub fn shared(e: mtl.ComputeEncoder, k: *const kn.Kernels, s: *st.Scratch, layer: usize, total: usize, used: usize) void {
    const offset = layer * st.batch_rows * @sizeOf(kn.AttentionSegment);
    e.setPipeline(k.get("lfm3_attn_segments_merge_flagged"));
    e.setBuffer(s.pm, 0, 0);
    e.setBuffer(s.pl, 0, 1);
    e.setBuffer(s.po, 0, 2);
    e.setBuffer(s.attention_segments, offset, 3);
    e.setBuffer(s.y, 0, 4);
    e.setBuffer(s.attention_rows, 0, 5);
    e.setBuffer(s.attention_risk, 0, 6);
    e.setBuffer(s.repair_flags.?, 0, 7);
    groups(e, .{ c.query_heads, total, 1 }, 32);
    e.setPipeline(k.get("lfm3_attn_segments_repair_partial"));
    e.setBuffer(s.queries, 0, 0);
    e.setBuffer(s.attention_segments, offset, 1);
    e.setBuffer(s.attention_rows, 0, 2);
    e.setBuffer(s.pm, 0, 3);
    e.setBuffer(s.pl, 0, 4);
    e.setBuffer(s.po, 0, 5);
    e.setBuffer(s.repair_exp.?, 0, 6);
    e.setBuffer(s.repair_flags.?, 0, 7);
    groups(e, .{ c.query_heads, total, used }, 128);
    e.setPipeline(k.get("lfm3_attn_segments_repair_merge"));
    e.setBuffer(s.pm, 0, 0);
    e.setBuffer(s.pl, 0, 1);
    e.setBuffer(s.po, 0, 2);
    e.setBuffer(s.repair_exp.?, 0, 3);
    e.setBuffer(s.attention_segments, offset, 4);
    e.setBuffer(s.y, 0, 5);
    e.setBuffer(s.attention_rows, 0, 6);
    e.setBuffer(s.repair_flags.?, 0, 7);
    groups(e, .{ c.query_heads, total, 1 }, 32);
}

test "TF_LFM3_PARALLEL_REPAIR is exact-one opt-in and defaults off" {
    try std.testing.expectEqual(false, try setting(null));
    try std.testing.expectEqual(false, try setting(""));
    try std.testing.expectEqual(false, try setting("0"));
    try std.testing.expectEqual(true, try setting("1"));
    try std.testing.expectError(error.BadRangeRepair, setting("true"));
}
