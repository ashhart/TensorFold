//! The Qwen host draw (families/qwen3_5/draw.zig) timed a 248,320-logit row: greedy, masked and sampled, no GPU.
const std = @import("std");
const lanes = @import("lanes");
const draw = @import("families/qwen3_5/draw.zig");
const vocab = @import("families/qwen3_5/config.zig").vocab;
const width = (vocab + 31) / 32;
/// Distinct rows a case cycles through, so most reads miss the caches as a freshly written row does.
const rows = 32;
const passes = 41;

const Mask = enum { none, dense, most, sparse, one };

const Case = struct { name: []const u8, mask: Mask, sampling: ?lanes.Sampling };

const hot: lanes.Sampling = .{ .seed = 1234, .top_k = 20, .temperature = 0.7, .top_p = 0.95 };
const all: lanes.Sampling = .{ .seed = 1234, .top_k = 0, .temperature = 0.7, .top_p = 0.95 };

const cases = [_]Case{
    .{ .name = "greedy, unmasked", .mask = .none, .sampling = null },
    .{ .name = "greedy, half the ids allowed", .mask = .dense, .sampling = null },
    .{ .name = "greedy, all but 1/64 allowed", .mask = .most, .sampling = null },
    .{ .name = "greedy, 16 scattered ids allowed", .mask = .sparse, .sampling = null },
    .{ .name = "greedy, one id allowed", .mask = .one, .sampling = null },
    .{ .name = "top_k 20, unmasked", .mask = .none, .sampling = hot },
    .{ .name = "top_k 20, half the ids allowed", .mask = .dense, .sampling = hot },
    .{ .name = "top_k 20, 16 scattered ids allowed", .mask = .sparse, .sampling = hot },
    .{ .name = "top_k 0, 16 scattered ids allowed", .mask = .sparse, .sampling = all },
};

/// Logits near a served row's spread: a few large values over a wide body of small ones.
fn fillRow(r: std.Random, words: []u16) void {
    for (words) |*w| {
        const x: f32 = @floatCast(r.floatNorm(f64) * 2.5 - 4 + (if (r.uintLessThan(u32, 4096) == 0) @as(f64, 12) else 0));
        w.* = @intCast(@as(u32, @bitCast(x)) >> 16);
    }
}

fn fillMask(r: std.Random, mask: []u32, shape: Mask) void {
    switch (shape) {
        .none => {},
        .dense => for (mask) |*m| {
            m.* = r.int(u32);
        },
        .most => for (mask) |*m| {
            m.* = if (r.boolean()) ~(@as(u32, 1) << @intCast(r.uintLessThan(u32, 32))) else ~@as(u32, 0);
        },
        .sparse, .one => {
            @memset(mask, 0);
            for (0..if (shape == .one) 1 else 16) |_| {
                const id = r.uintLessThan(usize, vocab);
                mask[id / 32] |= @as(u32, 1) << @intCast(id % 32);
            }
        },
    }
}

test "host draw timing: microseconds a 248,320-logit row through the Qwen host draw" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var prng = std.Random.DefaultPrng.init(2832);
    const r = prng.random();
    const words = try gpa.alloc(u16, rows * vocab);
    defer gpa.free(words);
    const masks = try gpa.alloc(u32, rows * width);
    defer gpa.free(masks);
    for (0..rows) |i| fillRow(r, words[i * vocab ..][0..vocab]);
    std.debug.print("\nhost draw timing: {d} rows of {d} logits, median of {d} passes\n", .{ rows, vocab, passes });
    for (cases) |case| {
        for (0..rows) |i| fillMask(r, masks[i * width ..][0..width], case.mask);
        var times: [passes]f64 = undefined;
        var sum: u64 = 0;
        for (&times) |*t| {
            const start = std.Io.Clock.awake.now(io).toNanoseconds();
            for (0..rows) |i| {
                const mask: []const u32 = if (case.mask == .none) &.{} else masks[i * width ..][0..width];
                const token = try draw.draw(gpa, words[i * vocab ..][0..vocab], case.sampling, 4096 + i, mask);
                sum = sum *% 1_000_003 +% token;
            }
            const ns = std.Io.Clock.awake.now(io).toNanoseconds() - start;
            t.* = @as(f64, @floatFromInt(ns)) / 1e3 / rows;
        }
        std.mem.sort(f64, &times, {}, std.sort.asc(f64));
        std.debug.print("  {s:<36} {d:>9.2} us a row   tokens {x:0>16}\n", .{ case.name, times[passes / 2], sum });
    }
}
