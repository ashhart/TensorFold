const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const batch = @import("../../cuda_batch.zig");
const mtp = @import("../../cuda_mtp.zig");
const binding = @import("exl3_state_binding.zig");

pub fn currentPass(comptime Key: type, comptime windowRequest: anytype) !void {
    try std.testing.expectEqual(mtp.max_rows, binding.max_width);
    var pass: batch.Pass(struct {}) = undefined;
    var kernels: batch.Kernels = undefined;
    kernels.chain = .{ .d = undefined, .handle = @ptrFromInt(10) };
    kernels.chain_to = .{ .d = undefined, .handle = @ptrFromInt(11) };
    var scratch: batch.Scratch = undefined;
    scratch.gq = 1;
    scratch.gk = 2;
    scratch.gv = 3;
    scratch.ggb = 4;
    scratch.ys = 5;
    var states: [wt.linear_n]cuda.DeviceBuffer = undefined;
    var layers: [wt.layers_n]@import("../../cuda_prompt_load.zig").Layer = undefined;
    var keep: batch.Keep = .{};
    for (&layers, 0..) |*layer, index| layer.linear = wt.linearLayer(index);
    pass.lanes = &.{};
    pass.k = &kernels;
    pass.s = &scratch;
    pass.seq.state = &states;
    pass.m.layers = &layers;
    pass.hooks.keep = &keep;
    var key: Key = .{ .kind = 2, .rows = 2, .scratch = 1, .scores = 2, .output = 3 };
    for (2..mtp.max_rows + 1) |width| {
        key.rows = width;
        for ([_]bool{ false, true }) |flipped| {
            for (&states, 0..) |*state, index| state.ptr = (if (flipped) @as(u64, 2000) else 1000) + index;
            for (&layers, 0..) |layer, index| {
                if (layer.linear) keep.layers[index].state_out = (if (flipped) @as(u64, 1000) else 2000) + wt.linearIndex(index);
            }
            const maybe = windowRequest(key, &pass);
            try std.testing.expect(maybe != null);
            const request = maybe.?;
            try std.testing.expectEqual(width, request.width);
            try std.testing.expectEqual(@as(c_int, @intCast(width)), request.nonstate.rows);
            try std.testing.expectEqual(wt.linear_n, request.count);
            for (request.layers[0..request.count], 0..) |mapping, index| {
                try std.testing.expectEqual(index, wt.linearIndex(mapping.layer));
                try std.testing.expectEqual(states[index].ptr, mapping.state_in);
                try std.testing.expectEqual(keep.layers[mapping.layer].state_out, mapping.state_out);
            }
        }
    }
    for ([_]usize{ 0, 1, mtp.max_rows + 1, batch.att_rows + 1 }) |width| {
        key.rows = width;
        try std.testing.expect(windowRequest(key, &pass) == null);
    }
    key.rows = 3;
    key.kind = 1;
    try std.testing.expect(windowRequest(key, &pass) == null);
    key.kind = 2;
    const lanes = [_]batch.Lane{undefined};
    pass.lanes = &lanes;
    try std.testing.expect(windowRequest(key, &pass) == null);
    pass.lanes = &.{};
    pass.seq.state = states[0 .. wt.linear_n - 1];
    try std.testing.expect(windowRequest(key, &pass) == null);
    pass.seq.state = &states;
    keep.layers[0].state_out = states[0].ptr;
    try std.testing.expect(windowRequest(key, &pass) == null);
    keep.layers[0].state_out = 0;
    try std.testing.expect(windowRequest(key, &pass) == null);
    pass.hooks.keep = null;
    try std.testing.expect(windowRequest(key, &pass) == null);
}
