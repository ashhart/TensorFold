//! Each mode against the expression its family computed in place before the table, on all 256 values of each channel.
const std = @import("std");
const normalize = @import("normalize.zig");

const third: f64 = 1.0 / 255.0;
const half: [3]f32 = @splat(0.5);
const clip_mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const clip_std = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

fn same(t: normalize.Table, comptime f: fn (u8, usize) f32) !void {
    for (0..3) |ch| for (0..256) |i| {
        const want = f(@intCast(i), ch);
        try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(t[ch][i])));
    };
}

test "unfused: patches.zig's PIL path and Gemma 4's [0, 1] (mean 0, std 1)" {
    const Q = struct {
        fn f(v: u8, ch: usize) f32 {
            const scaled: f32 = @floatCast(@as(f64, @floatFromInt(v)) * third);
            return (scaled - half[ch]) / half[ch];
        }
        fn gemma(v: u8, _: usize) f32 {
            return @floatCast(@as(f64, @floatFromInt(v)) * (1.0 / 255.0));
        }
    };
    try same(normalize.table(.unfused, third, half, half), Q.f);
    try same(normalize.table(.unfused, third, @splat(0), @splat(1)), Q.gemma);
}

test "fused: patches.zig's and Qwen-VL video's torchvision path, LFM2's (x - 127.5) / 127.5" {
    const Q = struct {
        fn f(v: u8, ch: usize) f32 {
            const m: f32 = @floatCast(@as(f64, half[ch]) / third);
            const s: f32 = @floatCast(@as(f64, half[ch]) / third);
            return (@as(f32, @floatFromInt(v)) - m) / s;
        }
        fn clip(v: u8, ch: usize) f32 {
            const m: f32 = @floatCast(@as(f64, clip_mean[ch]) / third);
            const s: f32 = @floatCast(@as(f64, clip_std[ch]) / third);
            return (@as(f32, @floatFromInt(v)) - m) / s;
        }
        fn lfm2(v: u8, _: usize) f32 {
            return (@as(f32, @floatFromInt(v)) - 127.5) / 127.5;
        }
    };
    try same(normalize.table(.fused, third, half, half), Q.f);
    try same(normalize.table(.fused, third, clip_mean, clip_std), Q.clip);
    try same(comptime normalize.table(.fused, third, half, half), Q.lfm2);
}

test "fused_f32 and rescale_f32: GLM5-Next's video and image paths" {
    const Q = struct {
        fn video(v: u8, ch: usize) f32 {
            const by: f32 = @floatCast(@as(f64, 1.0) / (@as(f64, 1.0) / 255.0));
            return (@as(f32, @floatFromInt(v)) - clip_mean[ch] * by) / (clip_std[ch] * by);
        }
        fn image(v: u8, ch: usize) f32 {
            const rescale: f32 = @floatCast(@as(f64, 1.0) / 255.0);
            return (@as(f32, @floatFromInt(v)) * rescale - clip_mean[ch]) / clip_std[ch];
        }
    };
    try same(normalize.table(.fused_f32, third, clip_mean, clip_std), Q.video);
    try same(normalize.table(.rescale_f32, third, clip_mean, clip_std), Q.image);
}

test "unfused is also Step-5's ToTensor and Normalize: x / 255 in f32 is x * rescale in f64 to f32, every u8" {
    for (0..256) |i| {
        const div = @as(f32, @floatFromInt(i)) / 255.0;
        const mul: f32 = @floatCast(@as(f64, @floatFromInt(i)) * third);
        try std.testing.expectEqual(@as(u32, @bitCast(mul)), @as(u32, @bitCast(div)));
    }
    const Q = struct {
        fn f(v: u8, ch: usize) f32 {
            return (@as(f32, @floatFromInt(v)) / 255.0 - clip_mean[ch]) / clip_std[ch];
        }
    };
    try same(normalize.table(.unfused, third, clip_mean, clip_std), Q.f);
}

test "unfused, fused and rescale_f32 each miss the others' bits" {
    const modes = [_]normalize.Mode{ .unfused, .fused, .rescale_f32 };
    var tables: [modes.len]normalize.Table = undefined;
    for (modes, &tables) |m, *t| t.* = normalize.table(m, third, clip_mean, clip_std);
    for (0..modes.len) |a| for (a + 1..modes.len) |b| {
        try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&tables[a]), std.mem.asBytes(&tables[b])));
    };
}
