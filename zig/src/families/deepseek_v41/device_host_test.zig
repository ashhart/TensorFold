const std = @import("std");
test {
    _ = @import("staging_owner_test.zig");
    _ = @import("cuda_model.zig");
    if (@import("builtin").os.tag == .linux) inline for (.{ @import("cuda_model.zig").Model, @import("serve_engine.zig").Served, @import("serve_cli.zig") }) |m| std.testing.refAllDecls(m);
    _ = @import("slot_transactions_test.zig");
    inline for (.{ @import("target.zig").GpuTarget, @import("batch.zig").Batch, @import("sampling_gpu.zig").GpuSampler, @import("sampling_dev.zig").Gpu, @import("dspark_gpu.zig").GpuPass, @import("dspark_slots.zig").SlotPass, @import("dspark_dev.zig").Dev, @import("sched_host.zig"), @import("phases_timed.zig"), @import("profile.zig"), @import("aot_needs.zig") }) |m| std.testing.refAllDecls(m);
    _ = @import("rows_test.zig");
    _ = @import("slot_drafts_test.zig");
    _ = @import("slot_drafts_chain_test.zig");
    _ = @import("multi_test.zig");
    _ = @import("branches_test.zig");
    _ = @import("aot_needs_test.zig");
    _ = @import("kx_early_test.zig");
}

test {
    _ = @import("review_constructors_test.zig");
}

test {
    _ = @import("review_handoff_test.zig");
}

test {
    _ = @import("model_supervisor_test.zig");
}

test {
    _ = @import("round_progress_test.zig");
}

test {
    _ = @import("session_stall_test.zig");
}

test {
    _ = @import("union_stall_test.zig");
}

test {
    _ = @import("model_health_test.zig");
}

test {
    _ = @import("session_restore_test.zig");
    _ = @import("session_protocol.zig");
}

test "slide admission: DeepSeek refuses before assets or checkpoint access" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var why: []const u8 = "";
    const result = try @import("serve_engine.zig").open(arena.allocator(), std.testing.allocator, std.testing.io, .{ .dir = "missing-slide-checkpoint", .model_type = "deepseek_v41", .slide = true }, &why);
    try std.testing.expect(result == null);
    try std.testing.expectEqualStrings("--slide: deepseek_v41 checkpoints do not learn on CUDA yet", why);
}
