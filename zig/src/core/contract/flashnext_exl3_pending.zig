//! NVCC 13.4 EXL3 resource receipts; older compilers remain explicitly pending.
const std = @import("std");
const limits = @import("limits.zig");
const registry = @import("registry.zig");

/// Launches with no nvcc 13.0 to 13.3 receipt yet: the geometry is recorded and usage stays pending until measured.
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "flashnext_exl3 fn_exl3_split_f16", .symbol = "fn_exl3_split_f16", .image = "flashnext_exl3", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_batch fn_exl3_leaf_tree", .symbol = "fn_exl3_leaf_tree", .image = "flashnext_batch", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_batch fn_exl3_leaf_partial", .symbol = "fn_exl3_leaf_partial", .image = "flashnext_batch", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_batch fn_exl3_leaf_finish", .symbol = "fn_exl3_leaf_finish", .image = "flashnext_batch", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_tp fn_exl3_tp_mask", .symbol = "fn_exl3_tp_mask", .image = "flashnext_exl3_tp", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_dcheckpoint", .symbol = "fn_exl3_dcheckpoint", .image = "flashnext_exl3_dplan", .threads = 1 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_dplan", .symbol = "fn_exl3_dplan", .image = "flashnext_exl3_dplan", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_drestore", .symbol = "fn_exl3_drestore", .image = "flashnext_exl3_dplan", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_dcopy", .symbol = "fn_exl3_dcopy", .image = "flashnext_exl3_dplan", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_draft_token", .symbol = "fn_exl3_draft_token", .image = "flashnext_exl3_dplan", .threads = 32 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_draft_next", .symbol = "fn_exl3_draft_next", .image = "flashnext_exl3_dplan", .threads = 32 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_prefill tf_exl3pf_rotate", .symbol = "tf_exl3pf_rotate", .image = "flashnext_exl3_prefill", .threads = 32 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_prefill tf_exl3pf_gu", .symbol = "tf_exl3pf_gu", .image = "flashnext_exl3_prefill", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_prefill tf_exl3pf_down", .symbol = "tf_exl3pf_down", .image = "flashnext_exl3_prefill", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_prefill tf_exl3pf_tp_combine", .symbol = "tf_exl3pf_tp_combine", .image = "flashnext_exl3_prefill", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_prefill tf_exl3pf_tp_finish", .symbol = "tf_exl3pf_tp_finish", .image = "flashnext_exl3_prefill", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_prefill tf_exl3pf_tp_place", .symbol = "tf_exl3pf_tp_place", .image = "flashnext_exl3_prefill", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_candidate tf_exl3_promptd_0", .symbol = "tf_exl3_promptd_0", .image = "exl3_candidate", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_candidate tf_exl3_promptd_1", .symbol = "tf_exl3_promptd_1", .image = "exl3_candidate", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_candidate tf_exl3_promptd_2", .symbol = "tf_exl3_promptd_2", .image = "exl3_candidate", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kvtier kvt_plan", .symbol = "kvt_plan", .image = "flashnext_exl3_kvtier", .threads = 1024 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kvtier kvt_fetch", .symbol = "kvt_fetch", .image = "flashnext_exl3_kvtier", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kvtier kvt_remap", .symbol = "kvt_remap", .image = "flashnext_exl3_kvtier", .threads = 256 }, .usage = registry.none },
};

/// NVCC 13.4 registers, user shared bytes and bounds for SM80/86/89/120/121, with FP64 canonical sums.
pub const cuda_134 = [_]registry.Override{
    .{ .name = "flashnext_exl3 fn_exl3_split_f16", .usage = .{ .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_batch fn_exl3_leaf_tree", .usage = .{ .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_batch fn_exl3_leaf_partial", .usage = .{ .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 }, .{ .regs = 26, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_batch fn_exl3_leaf_finish", .usage = .{ .{ .regs = 25, .static = 0, .bound = 0 }, .{ .regs = 25, .static = 0, .bound = 0 }, .{ .regs = 25, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_tp fn_exl3_tp_mask", .usage = .{ .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_dcheckpoint", .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_dplan", .usage = .{ .{ .regs = 32, .static = 4688, .bound = 256 }, .{ .regs = 40, .static = 4688, .bound = 256 }, .{ .regs = 40, .static = 4688, .bound = 256 }, .{ .regs = 48, .static = 4688, .bound = 256 }, .{ .regs = 48, .static = 4688, .bound = 256 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_drestore", .usage = .{ .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_dcopy", .usage = .{ .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_draft_token", .usage = .{ .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_draft_next", .usage = .{ .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_prefill tf_exl3pf_rotate", .usage = .{ .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 19, .static = 0, .bound = 0 }, .{ .regs = 19, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_prefill tf_exl3pf_gu", .usage = .{ .{ .regs = 128, .static = 33792, .bound = 256 }, .{ .regs = 128, .static = 33792, .bound = 256 }, .{ .regs = 128, .static = 33792, .bound = 256 }, .{ .regs = 128, .static = 33792, .bound = 256 }, .{ .regs = 128, .static = 33792, .bound = 256 } } },
    .{ .name = "flashnext_exl3_prefill tf_exl3pf_down", .usage = .{ .{ .regs = 128, .static = 34048, .bound = 256 }, .{ .regs = 128, .static = 34048, .bound = 256 }, .{ .regs = 128, .static = 34048, .bound = 256 }, .{ .regs = 128, .static = 34048, .bound = 256 }, .{ .regs = 128, .static = 34048, .bound = 256 } } },
    .{ .name = "flashnext_exl3_prefill tf_exl3pf_tp_combine", .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_prefill tf_exl3pf_tp_finish", .usage = .{ .{ .regs = 32, .static = 1156, .bound = 256 }, .{ .regs = 40, .static = 1156, .bound = 256 }, .{ .regs = 40, .static = 1156, .bound = 256 }, .{ .regs = 38, .static = 1156, .bound = 256 }, .{ .regs = 38, .static = 1156, .bound = 256 } } },
    .{ .name = "flashnext_exl3_prefill tf_exl3pf_tp_place", .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 } } },
    .{ .name = "exl3_candidate tf_exl3_promptd_0", .usage = .{ .{ .regs = 202, .static = 32768, .bound = 256 }, .{ .regs = 204, .static = 32768, .bound = 256 }, .{ .regs = 204, .static = 32768, .bound = 256 }, .{ .regs = 180, .static = 32768, .bound = 256 }, .{ .regs = 180, .static = 32768, .bound = 256 } } },
    .{ .name = "exl3_candidate tf_exl3_promptd_1", .usage = .{ .{ .regs = 202, .static = 32768, .bound = 256 }, .{ .regs = 204, .static = 32768, .bound = 256 }, .{ .regs = 204, .static = 32768, .bound = 256 }, .{ .regs = 180, .static = 32768, .bound = 256 }, .{ .regs = 180, .static = 32768, .bound = 256 } } },
    .{ .name = "exl3_candidate tf_exl3_promptd_2", .usage = .{ .{ .regs = 202, .static = 32768, .bound = 256 }, .{ .regs = 204, .static = 32768, .bound = 256 }, .{ .regs = 204, .static = 32768, .bound = 256 }, .{ .regs = 179, .static = 32768, .bound = 256 }, .{ .regs = 179, .static = 32768, .bound = 256 } } },
    .{ .name = "flashnext_exl3_kvtier kvt_plan", .usage = .{ .{ .regs = 27, .static = 1056, .bound = 1024 }, .{ .regs = 27, .static = 1056, .bound = 1024 }, .{ .regs = 27, .static = 1056, .bound = 1024 }, .{ .regs = 34, .static = 1056, .bound = 1024 }, .{ .regs = 34, .static = 1056, .bound = 1024 } } },
    .{ .name = "flashnext_exl3_kvtier kvt_fetch", .usage = .{ .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_kvtier kvt_remap", .usage = .{ .{ .regs = 19, .static = 4, .bound = 256 }, .{ .regs = 19, .static = 4, .bound = 256 }, .{ .regs = 19, .static = 4, .bound = 256 }, .{ .regs = 24, .static = 4, .bound = 256 }, .{ .regs = 24, .static = 4, .bound = 256 } } },
};

test "Flash Next EXL3 launches: pending below nvcc 13.4, measured 13.4 resources fit every SM launch limit, the rest stay pending" {
    const t = std.testing;
    for (cuda_134) |o| {
        const recorded = for (records) |r| {
            if (std.mem.eql(u8, r.launch.name, o.name)) break true;
        } else false;
        try t.expect(recorded);
    }
    for (records) |r| {
        try t.expectEqual(limits.sms.len, try registry.checkCuda(t.allocator, &.{r}, true));
        for (registry.cuda) |known| try t.expect(!std.mem.eql(u8, r.launch.symbol, known.launch.symbol));
        const measured = registry.usageFor(r, .{ .major = 13, .minor = 4 });
        const receipt: ?registry.Override = for (cuda_134) |o| {
            if (std.mem.eql(u8, o.name, r.launch.name)) break o;
        } else null;
        // no 13.4 receipt: still pending, and still refused by the resource check
        if (receipt == null) {
            try t.expect(std.meta.eql(measured, registry.none));
            continue;
        }
        try t.expect(std.meta.eql(measured, receipt.?.usage));
        var current = r;
        current.usage = measured;
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{current}, true));
        for (measured) |u| {
            try t.expect(u.regs > 0);
            try t.expect(u.bound == 0 or u.bound == r.launch.threads);
        }
    }
}
