//! Review regressions execute the actual trace and wrapped driver against recording CUDA calls.
const std = @import("std");
const cuda = @import("cuda");
const trace = cuda.round_attribution;
const Rig = @import("round_attribution_fake.zig").Rig;
const Fault = @import("round_attribution_fault.zig").Fault;
const t = std.testing;
test "attribution review: cached function names do not grow the permanent arena" {
    var fault: Fault = .{};
    const rig = try Rig.init();
    defer rig.deinit();
    rig.timing.permanent.deinit();
    rig.timing.permanent = .init(fault.allocator());
    var function: cuda.abi.Function = null;
    const name = "attention_repeated_function_name_long_enough_to_cross_an_arena_block";
    try t.expectEqual(@as(cuda.abi.Result, 0), rig.driver.api.cuModuleGetFunction(&function, null, name));
    const saved = rig.timing.name(function);
    const calls = fault.calls;
    for (0..5000) |_| {
        try t.expectEqual(@as(cuda.abi.Result, 0), rig.driver.api.cuModuleGetFunction(&function, null, name));
    }
    try t.expectEqualStrings(name, rig.timing.name(function));
    try t.expectEqual(calls, fault.calls);
    try t.expectEqual(saved.ptr, rig.timing.name(function).ptr);
    rig.timing.begin();
    try rig.kernel();
    _ = rig.driver.api.cuStreamSynchronize(null);
    try t.expectEqualStrings(name, rig.timing.device.items[0].symbol);
}
