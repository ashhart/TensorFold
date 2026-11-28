const std = @import("std");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const Buffers = @import("cuda_state.zig").Buffers;

const ResetHost = struct {
    const Clear = struct { ptr: u64, value: u32, words: usize };
    var clears: [7]Clear = undefined;
    var count: usize = 0;

    fn fill(ptr: u64, value: c_uint, words: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        clears[count] = .{ .ptr = ptr, .value = value, .words = words };
        count += 1;
        @memset(@as([*]u32, @ptrFromInt(ptr))[0..words], value);
        return cuda.abi.success;
    }
};

test "Nemotron copies: reset clears fixed state exactly and leaves contiguous and paged KV canaries untouched" {
    const state_bytes: [7]usize = .{ 512, 512, 16, 20, 24, 28, 32 };
    var words: [7][128]u32 = @splat(@splat(0xa5a5_a5a5));
    var b: Buffers = undefined;
    b.k_cache = @intFromPtr(&words[0]);
    b.v_cache = @intFromPtr(&words[1]);
    b.ssm = @intFromPtr(&words[2]);
    b.conv_base = @intFromPtr(&words[3]);
    b.raw = @intFromPtr(&words[4]);
    b.xc = @intFromPtr(&words[5]);
    b.dt = @intFromPtr(&words[6]);
    b.state_bytes = state_bytes;

    var d: cuda.Driver = undefined;
    d.api.cuMemsetD32Async = ResetHost.fill;
    var k: kern.Kernels = undefined;
    k.d = &d;
    const ops: kern.Ops = .{ .k = &k, .s = .{ .d = &d, .handle = @ptrFromInt(1) } };

    for ([_]u64{ 0, 1 }) |table| {
        b.table = table;
        ResetHost.count = 0;
        @memset(std.mem.sliceAsBytes(&words), 0xa5);
        const before = words;
        try b.reset(ops);
        try std.testing.expectEqual(@as(usize, 5), ResetHost.count);
        for (ResetHost.clears[0..ResetHost.count], 2..) |clear, i| {
            try std.testing.expectEqual(b.planes()[i], clear.ptr);
            try std.testing.expectEqual(@as(u32, 0), clear.value);
            try std.testing.expectEqual(state_bytes[i] / 4, clear.words);
        }
        for (0..2) |i| try std.testing.expectEqualSlices(u32, &before[i], &words[i]);
        for (2..7) |i| {
            const used = state_bytes[i] / 4;
            for (words[i][0..used]) |word| try std.testing.expectEqual(@as(u32, 0), word);
            try std.testing.expectEqualSlices(u32, before[i][used..], words[i][used..]);
        }
    }
}
