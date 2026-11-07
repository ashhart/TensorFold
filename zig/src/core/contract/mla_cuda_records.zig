//! The mla image's launches (mla.cu): nvcc 13.0's usage, five-SM fatbins; 13.3's in pending_133.zig, 13.2 pending.
const std = @import("std");
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "mla tf_mla_chunks_r16", .symbol = "tf_mla_chunks_r16", .image = "mla", .threads = 256, .dynamic = 93568, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 76, .static = 0, .bound = 256 }, .{ .regs = 76, .static = 0, .bound = 256 }, .{ .regs = 76, .static = 0, .bound = 256 }, .{ .regs = 126, .static = 0, .bound = 256 }, .{ .regs = 126, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "mla tf_mla_chunks_split_r16", .symbol = "tf_mla_chunks_split_r16", .image = "mla", .threads = 256, .dynamic = 93568, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 74, .static = 0, .bound = 256 }, .{ .regs = 79, .static = 0, .bound = 256 }, .{ .regs = 79, .static = 0, .bound = 256 }, .{ .regs = 119, .static = 0, .bound = 256 }, .{ .regs = 119, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "mla tf_mla_merge_part", .symbol = "tf_mla_merge_part", .image = "mla", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 30, .static = 0, .bound = 128 }, .{ .regs = 31, .static = 0, .bound = 128 }, .{ .regs = 31, .static = 0, .bound = 128 }, .{ .regs = 32, .static = 0, .bound = 128 }, .{ .regs = 32, .static = 0, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "mla tf_mla_absorb_q_128", .symbol = "tf_mla_absorb_q_128", .image = "mla", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 56, .static = 512, .bound = 128 }, .{ .regs = 56, .static = 512, .bound = 128 }, .{ .regs = 56, .static = 512, .bound = 128 }, .{ .regs = 56, .static = 512, .bound = 128 }, .{ .regs = 56, .static = 512, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "mla tf_mla_expand_v_128", .symbol = "tf_mla_expand_v_128", .image = "mla", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 96, .static = 128, .bound = 128 }, .{ .regs = 96, .static = 128, .bound = 128 }, .{ .regs = 96, .static = 128, .bound = 128 }, .{ .regs = 95, .static = 128, .bound = 128 }, .{ .regs = 95, .static = 128, .bound = 128 } }, .later_pending = true },
    .{ .launch = .{ .name = "mla tf_mla_rope", .symbol = "tf_mla_rope", .image = "mla", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 26, .static = 0, .bound = 32 }, .{ .regs = 26, .static = 0, .bound = 32 }, .{ .regs = 26, .static = 0, .bound = 32 }, .{ .regs = 22, .static = 0, .bound = 32 }, .{ .regs = 22, .static = 0, .bound = 32 } }, .later_pending = true },
    .{ .launch = .{ .name = "mla tf_mla_gate", .symbol = "tf_mla_gate", .image = "mla", .threads = 128 }, .usage = .{ .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 } }, .later_pending = true },
};

test "mla records: every launch is registered once and fits every SM" {
    for (records) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
