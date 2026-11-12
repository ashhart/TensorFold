//! Dense KV snapshot copies keep only the live prefix across unequal capacities, without touching inputs or holes.
const std = @import("std");
const hip = @import("hip");
const family = @import("dots_mocr");
const Cache = family.text.Cache;
pub const std_options: std.Options = .{ .signal_stack_size = null };

test "dots HIP device cache: prefix copy preserves every head across unequal capacities" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const path = std.c.getenv("TF_DOTS_DIR") orelse return error.SkipZigTest;
    const c = try family.config.Config.read(a, io, std.mem.span(path));
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    const before = hip.usage(false).device;
    {
        var src = try Cache.init(&r, c, 17);
        defer src.deinit();
        var dest = try Cache.init(&r, c, 31);
        defer dest.deinit();
        const x = try a.alloc(u16, src.data.len / 2);
        defer a.free(x);
        const y = try a.alloc(u16, dest.data.len / 2);
        defer a.free(y);
        const got = try a.alloc(u16, y.len);
        defer a.free(got);
        const unchanged = try a.alloc(u16, x.len);
        defer a.free(unchanged);
        for (x, 0..) |*v, i| v.* = @truncate((i *% 1031) + 73);
        @memset(y, 0x5aa5);
        try src.data.upload(0, std.mem.sliceAsBytes(x));
        for ([_]usize{ 0, 1, 11, 17, 11 }) |keep| {
            try dest.data.upload(0, std.mem.sliceAsBytes(y));
            try src.copyPrefix(&dest, keep, stream);
            try stream.synchronize();
            try dest.data.download(0, std.mem.sliceAsBytes(got));
            try src.data.download(0, std.mem.sliceAsBytes(unchanged));
            try std.testing.expectEqualSlices(u16, x, unchanged);
            const width = c.headDim();
            for (got, 0..) |v, i| {
                const group = i / (dest.total * width);
                const slot = i % (dest.total * width);
                const want = if (slot < keep * width) x[group * src.total * width + slot] else 0x5aa5;
                try std.testing.expectEqual(want, v);
            }
        }
        // Reverse the capacities, restoring into poisoned storage while preserving the unfilled suffix.
        @memset(unchanged, 0xdead);
        try src.data.upload(0, std.mem.sliceAsBytes(unchanged));
        try dest.copyPrefix(&src, 11, stream);
        try stream.synchronize();
        try src.data.download(0, std.mem.sliceAsBytes(unchanged));
        try dest.data.download(0, std.mem.sliceAsBytes(y));
        try std.testing.expectEqualSlices(u16, got, y);
        for (unchanged, 0..) |v, i| {
            const slot = i % (src.total * c.headDim());
            try std.testing.expectEqual(if (slot < 11 * c.headDim()) x[i] else 0xdead, v);
        }
    }
    try std.testing.expectEqual(before, hip.usage(false).device);
}
