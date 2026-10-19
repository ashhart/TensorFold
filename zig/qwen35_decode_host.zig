//! New decode launches must remain pending until the device collector measures them.
const std = @import("std");
const registry = @import("core").contract.registry;
test "Qwen CUDA contracts: all optimized decode launches have pending resource records" {
    for ([_][]const u8{ "tf_topk_slices", "tf_topk_merge", "tf_q35_norm", "tf_q35_sums", "tf_q35_argmax_slices", "tf_q35_argmax_merge", "tf_q35_gdn_chain" }) |symbol| {
        var found = false;
        for (registry.all_cuda) |r| if (std.mem.eql(u8, symbol, r.launch.symbol)) {
            found = true;
            for (r.usage) |u| try std.testing.expectEqual(@as(u32, 0), u.regs);
        };
        try std.testing.expect(found);
    }
}
