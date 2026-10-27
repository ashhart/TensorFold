//! Flash Next EXL3 launches (int8 and paged KV, device pool plan, prefill experts); nvcc 13.4 measured, older pending.
const std = @import("std");
const limits = @import("limits.zig");
const registry = @import("registry.zig");

/// Launches with no nvcc 13.0 to 13.3 receipt yet: the geometry is recorded and usage stays pending until measured.
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "flashnext_exl3 fn_exl3_split_f16", .symbol = "fn_exl3_split_f16", .image = "flashnext_exl3", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_tp fn_exl3_tp_mask", .symbol = "fn_exl3_tp_mask", .image = "flashnext_exl3_tp", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_tp fn_exl3_tp_place", .symbol = "fn_exl3_tp_place", .image = "flashnext_exl3_tp", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_dcheckpoint", .symbol = "fn_exl3_dcheckpoint", .image = "flashnext_exl3_dplan", .threads = 1 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_dplan", .symbol = "fn_exl3_dplan", .image = "flashnext_exl3_dplan", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_drestore", .symbol = "fn_exl3_drestore", .image = "flashnext_exl3_dplan", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_dcopy", .symbol = "fn_exl3_dcopy", .image = "flashnext_exl3_dplan", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_draft_token", .symbol = "fn_exl3_draft_token", .image = "flashnext_exl3_dplan", .threads = 32 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_dplan fn_exl3_draft_next", .symbol = "fn_exl3_draft_next", .image = "flashnext_exl3_dplan", .threads = 32 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kv fn_kv8_prep", .symbol = "fn_kv8_prep", .image = "flashnext_exl3_kv", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kv fn_kv8_chunks", .symbol = "fn_kv8_chunks", .image = "flashnext_exl3_kv", .threads = 128, .dynamic = 60928, .opt_in = true }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kv fn_kv8_chunks_ahead", .symbol = "fn_kv8_chunks_ahead", .image = "flashnext_exl3_kv", .threads = 128, .dynamic = 78336, .opt_in = true }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kv fn_kv8_merge", .symbol = "fn_kv8_merge", .image = "flashnext_exl3_kv", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kv fn_kv8_pack", .symbol = "fn_kv8_pack", .image = "flashnext_exl3_kv", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kv fn_kv_gather", .symbol = "fn_kv_gather", .image = "flashnext_exl3_kv", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kv fn_kv_roundtrip", .symbol = "fn_kv_roundtrip", .image = "flashnext_exl3_kv", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kvpage kvp_plan", .symbol = "kvp_plan", .image = "flashnext_exl3_kvpage", .threads = 1024 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kvpage kvp_shared_plan", .symbol = "kvp_shared_plan", .image = "flashnext_exl3_kvpage", .threads = 1024 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kvpage kvp_shared_release", .symbol = "kvp_shared_release", .image = "flashnext_exl3_kvpage", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kvpage kvp_fetch", .symbol = "kvp_fetch", .image = "flashnext_exl3_kvpage", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kvpage kvp_remap", .symbol = "kvp_remap", .image = "flashnext_exl3_kvpage", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_kvpage kvp_copy", .symbol = "kvp_copy", .image = "flashnext_exl3_kvpage", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_prefill tf_exl3pf_rotate", .symbol = "tf_exl3pf_rotate", .image = "flashnext_exl3_prefill", .threads = 32 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_prefill tf_exl3pf_gu", .symbol = "tf_exl3pf_gu", .image = "flashnext_exl3_prefill", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_prefill tf_exl3pf_down", .symbol = "tf_exl3pf_down", .image = "flashnext_exl3_prefill", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_prefill tf_exl3pf_tp_combine", .symbol = "tf_exl3pf_tp_combine", .image = "flashnext_exl3_prefill", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "flashnext_exl3_prefill tf_exl3pf_tp_place", .symbol = "tf_exl3pf_tp_place", .image = "flashnext_exl3_prefill", .threads = 256 }, .usage = registry.none },
};

/// nvcc 13.4's measured registers, static shared bytes and launch bounds for sm_80, sm_86, sm_89, sm_120, sm_121.
pub const cuda_134 = [_]registry.Override{
    .{ .name = "flashnext_exl3 fn_exl3_split_f16", .usage = .{ .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_tp fn_exl3_tp_mask", .usage = .{ .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 18, .static = 0, .bound = 0 }, .{ .regs = 18, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_tp fn_exl3_tp_place", .usage = .{ .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_dcheckpoint", .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_dplan", .usage = .{ .{ .regs = 32, .static = 4688, .bound = 256 }, .{ .regs = 40, .static = 4688, .bound = 256 }, .{ .regs = 40, .static = 4688, .bound = 256 }, .{ .regs = 48, .static = 4688, .bound = 256 }, .{ .regs = 48, .static = 4688, .bound = 256 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_drestore", .usage = .{ .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 }, .{ .regs = 34, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_dcopy", .usage = .{ .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_draft_token", .usage = .{ .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_dplan fn_exl3_draft_next", .usage = .{ .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_kv fn_kv8_prep", .usage = .{ .{ .regs = 32, .static = 128, .bound = 256 }, .{ .regs = 36, .static = 128, .bound = 256 }, .{ .regs = 36, .static = 128, .bound = 256 }, .{ .regs = 31, .static = 128, .bound = 256 }, .{ .regs = 31, .static = 128, .bound = 256 } } },
    .{ .name = "flashnext_exl3_kv fn_kv8_chunks", .usage = .{ .{ .regs = 72, .static = 640, .bound = 128 }, .{ .regs = 72, .static = 640, .bound = 128 }, .{ .regs = 72, .static = 640, .bound = 128 }, .{ .regs = 72, .static = 640, .bound = 128 }, .{ .regs = 72, .static = 640, .bound = 128 } } },
    .{ .name = "flashnext_exl3_kv fn_kv8_chunks_ahead", .usage = .{ .{ .regs = 72, .static = 768, .bound = 128 }, .{ .regs = 72, .static = 768, .bound = 128 }, .{ .regs = 72, .static = 768, .bound = 128 }, .{ .regs = 64, .static = 768, .bound = 128 }, .{ .regs = 64, .static = 768, .bound = 128 } } },
    .{ .name = "flashnext_exl3_kv fn_kv8_merge", .usage = .{ .{ .regs = 29, .static = 0, .bound = 256 }, .{ .regs = 30, .static = 0, .bound = 256 }, .{ .regs = 30, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_kv fn_kv8_pack", .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_kv fn_kv_gather", .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 39, .static = 0, .bound = 256 }, .{ .regs = 39, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_kv fn_kv_roundtrip", .usage = .{ .{ .regs = 19, .static = 0, .bound = 256 }, .{ .regs = 21, .static = 0, .bound = 256 }, .{ .regs = 21, .static = 0, .bound = 256 }, .{ .regs = 21, .static = 0, .bound = 256 }, .{ .regs = 21, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_kvpage kvp_plan", .usage = .{ .{ .regs = 27, .static = 1056, .bound = 1024 }, .{ .regs = 27, .static = 1056, .bound = 1024 }, .{ .regs = 27, .static = 1056, .bound = 1024 }, .{ .regs = 36, .static = 1056, .bound = 1024 }, .{ .regs = 36, .static = 1056, .bound = 1024 } } },
    .{ .name = "flashnext_exl3_kvpage kvp_shared_plan", .usage = .{ .{ .regs = 27, .static = 1056, .bound = 1024 }, .{ .regs = 32, .static = 1056, .bound = 1024 }, .{ .regs = 32, .static = 1056, .bound = 1024 }, .{ .regs = 50, .static = 1056, .bound = 1024 }, .{ .regs = 50, .static = 1056, .bound = 1024 } } },
    .{ .name = "flashnext_exl3_kvpage kvp_shared_release", .usage = .{ .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_kvpage kvp_fetch", .usage = .{ .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 30, .static = 0, .bound = 256 }, .{ .regs = 30, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_kvpage kvp_remap", .usage = .{ .{ .regs = 19, .static = 4, .bound = 256 }, .{ .regs = 19, .static = 4, .bound = 256 }, .{ .regs = 19, .static = 4, .bound = 256 }, .{ .regs = 24, .static = 4, .bound = 256 }, .{ .regs = 24, .static = 4, .bound = 256 } } },
    .{ .name = "flashnext_exl3_kvpage kvp_copy", .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 } } },
    .{ .name = "flashnext_exl3_prefill tf_exl3pf_rotate", .usage = .{ .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 19, .static = 0, .bound = 0 }, .{ .regs = 19, .static = 0, .bound = 0 } } },
    .{ .name = "flashnext_exl3_prefill tf_exl3pf_gu", .usage = .{ .{ .regs = 128, .static = 46080, .bound = 256 }, .{ .regs = 128, .static = 46080, .bound = 256 }, .{ .regs = 128, .static = 46080, .bound = 256 }, .{ .regs = 128, .static = 46080, .bound = 256 }, .{ .regs = 128, .static = 46080, .bound = 256 } } },
    .{ .name = "flashnext_exl3_prefill tf_exl3pf_down", .usage = .{ .{ .regs = 128, .static = 46336, .bound = 256 }, .{ .regs = 128, .static = 46336, .bound = 256 }, .{ .regs = 128, .static = 46336, .bound = 256 }, .{ .regs = 128, .static = 46336, .bound = 256 }, .{ .regs = 128, .static = 46336, .bound = 256 } } },
    .{ .name = "flashnext_exl3_prefill tf_exl3pf_tp_combine", .usage = .{ .{ .regs = 32, .static = 256, .bound = 256 }, .{ .regs = 40, .static = 256, .bound = 256 }, .{ .regs = 40, .static = 256, .bound = 256 }, .{ .regs = 40, .static = 256, .bound = 256 }, .{ .regs = 40, .static = 256, .bound = 256 } } },
    .{ .name = "flashnext_exl3_prefill tf_exl3pf_tp_place", .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 } } },
};

test "Flash Next EXL3 launches: pending below nvcc 13.4, measured 13.4 resources fit every SM launch limit" {
    const t = std.testing;
    try t.expectEqual(records.len, cuda_134.len);
    for (records, cuda_134) |r, o| {
        try t.expectEqualStrings(r.launch.name, o.name);
        try t.expectEqual(limits.sms.len, try registry.checkCuda(t.allocator, &.{r}, true));
        const measured = registry.usageFor(r, .{ .major = 13, .minor = 4 });
        try t.expect(std.meta.eql(measured, o.usage));
        var current = r;
        current.usage = measured;
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{current}, true));
        for (measured) |u| {
            try t.expect(u.regs > 0);
            try t.expect(u.bound == 0 or u.bound == r.launch.threads);
        }
        for (registry.cuda) |known| try t.expect(!std.mem.eql(u8, r.launch.symbol, known.launch.symbol));
    }
}
