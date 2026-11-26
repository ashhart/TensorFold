//! Production inverse-FWHT output signs against known rotated weight columns at each chunk boundary.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

fn sign(row: u32) f32 {
    const mixed = (0x9e3779b9 ^ row) *% @as(u32, 2654435761);
    return if (mixed >> 31 != 0) -1 else 1;
}

pub fn run(gpu: check.Gpu, kernels: *const cuda.tfq3.Kernels, stream: cuda.Stream) !void {
    const cases = [_]struct { n: u32, period: u32, offset: u32 = 0 }{
        .{ .n = 8192, .period = 0 },                     .{ .n = 32768, .period = 8192 },
        .{ .n = 20480, .period = 16384 },                .{ .n = 4096, .period = 2048 },
        .{ .n = 384, .period = 8192, .offset = 247936 },
    };
    for (cases) |ca| {
        const yr = try gpu.gpa.alloc(f32, ca.n);
        defer gpu.gpa.free(yr);
        @memset(yr, 0);
        var at: usize = 0;
        while (at < yr.len) : (at += 128) yr[at] = 16;
        var input = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(yr));
        defer input.free();
        var out = try cuda.DeviceBuffer.alloc(gpu.d, ca.n * 4);
        defer out.free();
        var args: cuda.Args = .{};
        args.add(input.ptr);
        args.add(out.ptr);
        args.add(@as(u32, 1));
        args.add(ca.n);
        args.add(ca.n);
        args.add(ca.period);
        args.add(ca.offset);
        args.add(@as(u32, 1));
        args.add(@as(u32, 1));
        try cuda.launch.launch(kernels.fwht_y, .{ .grid = .{ .x = ca.n / 128, .y = 1 }, .block = .{ .x = 32 } }, stream, &args);
        try stream.synchronize();
        const got = try gpu.gpa.alloc(f32, ca.n);
        defer gpu.gpa.free(got);
        try out.download(0, std.mem.sliceAsBytes(got));
        for (got, 0..) |value, row| {
            const global: u32 = ca.offset + @as(u32, @intCast(row));
            const local = if (ca.period == 0) global else global % ca.period;
            const want: f32 = @as(f32, @bitCast(@as(u32, 0x3fb504f3))) * sign(local);
            try check.expect(@as(u32, @bitCast(value)) == @as(u32, @bitCast(want)), "TFQ3 sign period {d}, row {d}: wrong reconstructed weight-column bits", .{ ca.period, global });
        }
    }
    check.pass("TFQ3 weight-column inverse transform: 2047/2048, 8191/8192, 16383/16384 and narrowed head tail", .{});
}
