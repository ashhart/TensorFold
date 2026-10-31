//! Legacy scalar and production SIMD greedy selection on identical BF16 words, including ties and nonfinite rejection.
const std = @import("std");
const q = @import("tensorfold").lfm3;
const sys = @import("sys");

noinline fn legacyGreedy(words: []const u16) !u32 {
    var best: ?usize = null;
    for (words, 0..) |word, id| {
        const x = q.backend.value(word);
        if (!std.math.isFinite(x)) return error.NonfiniteLfmLogits;
        if (best == null or x > q.backend.value(words[best.?])) best = id;
    }
    return @intCast(best orelse return error.NoAllowedLfmToken);
}

fn now() f64 {
    var ts: sys.c.timespec = undefined;
    _ = sys.c.clock_gettime(sys.c.CLOCK.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) / 1e9;
}
fn equal(_: std.mem.Allocator, words: []const u16) !void {
    const expected = legacyGreedy(words) catch |err| {
        if (q.backend.greedy.select(words)) |_| return error.NonfiniteWasAccepted else |got| {
            if (got != err) return error.WrongRejection;
            return;
        }
    };
    if (expected != try q.backend.greedy.select(words)) return error.GreedyChangedToken;
}

pub fn main(init: std.process.Init) !void {
    const words = try init.gpa.alloc(u16, 128000);
    defer init.gpa.free(words);
    var prng = std.Random.DefaultPrng.init(8719);
    for (0..128) |test_| {
        for (words) |*word| word.* = @intCast(@as(u32, @bitCast(prng.random().float(f32) * 32 - 16)) >> 16);
        try equal(init.gpa, words);
        const at = (test_ * 7919) % words.len;
        words[at] = 0x7fc0;
        try equal(init.gpa, words);
    }
    for ([_]u16{ 0, 0x8000, 0x7f7f, 0xff7f, 1, 0x8001 }) |word| {
        @memset(words, word);
        try equal(init.gpa, words);
        words[3] = 0;
        words[11] = 0x8000;
        try equal(init.gpa, words);
    }
    var checksum: u64 = 0;
    for (0..5) |rep| {
        for (words) |*word| word.* = @intCast(@as(u32, @bitCast(prng.random().float(f32) * 32 - 16)) >> 16);
        const original = try init.gpa.dupe(u16, words);
        defer init.gpa.free(original);
        const start = now();
        for (0..1024) |i| {
            words[(i * 7919) % words.len] = @intCast(i % 256);
            checksum +%= try legacyGreedy(words);
        }
        const middle = now();
        @memcpy(words, original);
        const vector_start = now();
        for (0..1024) |i| {
            words[(i * 7919) % words.len] = @intCast(i % 256);
            checksum +%= try q.backend.greedy.select(words);
        }
        const end = now();
        std.debug.print("DRAW_TUNE rep={d} scalar_us={d:.3} vector_us={d:.3}\n", .{ rep, (middle - start) * 1e6 / 1024, (end - vector_start) * 1e6 / 1024 });
    }
    std.debug.print("DRAW_TUNE verified=268 checksum={d}\n", .{checksum});
}
