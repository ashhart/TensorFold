//! The new kernel cannot be qualified by inheriting the original kernel's measured resources.
const std = @import("std");
const registry = @import("core").contract.registry;
test "Qwen pipeline record: missing compile usage refuses sm121 qualification" {
    var count: usize = 0;
    for (registry.cuda_pending) |r| if (std.mem.eql(u8, r.launch.symbol, "tf_qwen_prompt_pipeline")) {
        try std.testing.expectEqual(@as(u32, 256), r.launch.threads);
        try std.testing.expectEqual(@as(u32, 98304), r.launch.dynamic);
        try std.testing.expect(r.launch.opt_in and r.launch.min_sm == 121);
        try std.testing.expectEqual(@as(usize, 1), try registry.checkCuda(std.testing.allocator, &.{r}, true));
        for (r.usage) |u| try std.testing.expectEqual(@as(u32, 0), u.regs);
        count += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), count);
}
