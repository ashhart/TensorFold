//! Paged 128-dimensional prompt launches; resource measurements remain pending for every compiler and SM.
const registry = @import("registry.zig");
pub const records = blk: {
    var out: [2]registry.CudaRecord = undefined;
    for ([_]u32{ 4, 8 }, 0..) |heads, i| {
        const symbol = @import("std").fmt.comptimePrint("tf_window_prompt_128_h{d}_pg", .{heads});
        out[i] = .{ .launch = .{ .name = symbol, .symbol = symbol, .image = "window_attention", .threads = 256, .dynamic = 65536, .opt_in = true }, .usage = registry.none };
    }
    break :blk out;
};
