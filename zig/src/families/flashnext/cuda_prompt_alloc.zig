//! Sequence-cache buffer allocation, including borrowed carved planes and rollback after failed fills.
const std = @import("std");
const cuda = @import("cuda");

/// `n` zeroed buffers, the fill on the compute stream; the first are the carveout's planes, zeroed already.
pub fn zeros(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: cuda.Stream, n: usize, len: usize, carved: []const cuda.DeviceBuffer) ![]cuda.DeviceBuffer {
    const Zeroed = struct {
        driver: *cuda.Driver,
        stream: cuda.Stream,
        len: usize,
        carved: []const cuda.DeviceBuffer,
        next: *usize,
        pub fn drain(z: @This()) void {
            cuda.completion.drain(z.driver, z.stream, "Flash Next sequence fills did not drain before rollback");
        }
        fn make(z: @This()) !cuda.DeviceBuffer {
            if (z.next.* < z.carved.len) {
                z.next.* += 1;
                return z.carved[z.next.* - 1];
            }
            return cuda.DeviceBuffer.alloc(z.driver, z.len);
        }
        fn fill(z: @This(), b: *cuda.DeviceBuffer) !void {
            if (b.borrowed) return;
            try b.fill8(0, z.stream.handle);
        }
    };
    var next: usize = 0;
    return made(cuda.DeviceBuffer, gpa, n, Zeroed{ .driver = driver, .stream = stream, .len = len, .carved = carved, .next = &next });
}

/// `n` buffers from `src.make`, each through `src.fill`; on a failure every buffer made so far is freed, the last too.
fn made(comptime B: type, gpa: std.mem.Allocator, n: usize, src: anytype) ![]B {
    const out = try gpa.alloc(B, n);
    var held: usize = 0;
    errdefer {
        if (@hasDecl(@TypeOf(src), "drain")) src.drain();
        for (out[0..held]) |*b| b.free();
        gpa.free(out);
    }
    while (held < n) {
        out[held] = try src.make();
        held += 1; // counted before its fill, so a failed fill frees it with the rest
        try src.fill(&out[held - 1]);
    }
    return out;
}

test "a failed fill frees the buffer it was filling and every one before it" {
    const Fake = struct {
        live: *usize,
        pub fn free(b: *@This()) void {
            b.live.* -= 1;
        }
    };
    const Src = struct {
        live: *usize,
        fills: *usize,
        fail_at: usize,
        fn make(s: @This()) !Fake {
            s.live.* += 1;
            return .{ .live = s.live };
        }
        fn fill(s: @This(), _: *Fake) !void {
            s.fills.* += 1;
            if (s.fills.* == s.fail_at) return error.FillFailed;
        }
    };
    for ([_]usize{ 1, 2, 4 }) |fail_at| {
        var live: usize = 0;
        var fills: usize = 0;
        try std.testing.expectError(error.FillFailed, made(Fake, std.testing.allocator, 4, Src{ .live = &live, .fills = &fills, .fail_at = fail_at }));
        try std.testing.expectEqual(@as(usize, 0), live);
    }
    var live: usize = 0;
    var fills: usize = 0;
    const all = try made(Fake, std.testing.allocator, 3, Src{ .live = &live, .fills = &fills, .fail_at = 0 });
    defer std.testing.allocator.free(all);
    try std.testing.expectEqual(@as(usize, 3), live);
}
