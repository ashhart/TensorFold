//! Optimized decode launches carry measured nvcc 13.0 resources; later compilers keep their own selected measurements.
const std = @import("std");
const registry = @import("core").contract.registry;
test "Qwen CUDA contracts: optimized decode launches are measured at 13.0" {
    for ([_][]const u8{ "tf_topk_slices", "tf_topk_merge", "tf_q35_norm", "tf_q35_sums", "tf_q35_argmax_slices", "tf_q35_argmax_merge", "tf_q35_gdn_chain" }) |symbol| {
        var found = false;
        for (registry.all_cuda) |r| if (std.mem.eql(u8, symbol, r.launch.symbol)) {
            found = true;
            for (r.usage) |u| try std.testing.expect(u.regs > 0);
            try std.testing.expect(r.later_pending);
        };
        try std.testing.expect(found);
    }
}
