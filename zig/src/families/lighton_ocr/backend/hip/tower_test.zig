const std = @import("std");
const cfg = @import("../../config.zig");
const tower = @import("tower.zig");
const a = std.testing.allocator;

test "HIP LightOn tower runtime graph compiles without opening a device" {
    var init = &tower.Tower.init;
    var encode = &tower.Tower.encode;
    var deinit = &tower.Tower.deinit;
    std.mem.doNotOptimizeAway(&init);
    std.mem.doNotOptimizeAway(&encode);
    std.mem.doNotOptimizeAway(&deinit);
}

test "HIP LightOn tower retains odd encoder grids and merges only complete groups" {
    const c = try cfg.Config.parse(a, cfg.sample);
    const odd = try tower.Shape.init(c, 55, 110, 55 * 110 * 588);
    try std.testing.expectEqual(@as(usize, 6050), odd.rows);
    try std.testing.expectEqual(@as(usize, 1485), odd.tokens);
    const full = try tower.Shape.init(c, 110, 110, 110 * 110 * 588);
    try std.testing.expectEqual(@as(usize, 3025), full.tokens);
    try std.testing.expectError(error.BadImagePatches, tower.Shape.init(c, 55, 110, 55 * 110 * 588 - 1));
    try std.testing.expectError(error.BadImagePatches, tower.Shape.init(c, 1, 110, 110 * 588));
    try std.testing.expectError(error.ImageGrid, tower.Shape.init(c, 111, 110, 111 * 110 * 588));
}

test "HIP LightOn tower scratch holds every simultaneously live tensor at the largest served grid" {
    const c = try cfg.Config.parse(a, cfg.sample);
    const capacity = try tower.scratchBytes(c);
    for ([_]usize{ 4, 9, 6050, tower.max_patches }) |rows| {
        const p = try tower.Scratch.init(c, rows);
        const starts = [_]usize{ p.inv, p.pixels, p.x, p.norm, p.q, p.key, p.value, p.qr, p.kr, p.vr, p.attention, p.branch, p.gate, p.up, p.cosine, p.sine, p.bytes };
        const sizes = [_]usize{ 128, rows * 588 * 2, rows * 1024 * 2, rows * 1024 * 2, rows * 1024 * 2, rows * 1024 * 2, rows * 1024 * 2, rows * 1024 * 2, rows * 1024 * 2, rows * 1024 * 2, rows * 1024 * 2, rows * 1024 * 2, rows * 4096 * 2, rows * 4096 * 2, rows * 32 * 2, rows * 32 * 2 };
        for (starts) |offset| try std.testing.expectEqual(@as(usize, 0), offset % 256);
        for (sizes, 0..) |bytes, i| try std.testing.expect(starts[i] + bytes <= starts[i + 1]);
        try std.testing.expect(p.bytes <= capacity);
        try std.testing.expect(rows / 4 * 4096 * 2 <= rows * 1024 * 2);
    }
    try std.testing.expectError(error.BadImagePatches, tower.Scratch.init(c, 0));
    try std.testing.expectError(error.BadImagePatches, tower.Scratch.init(c, tower.max_patches + 1));
}

test "HIP LightOn tower refuses output overlap and address overflow before device work" {
    try tower.outputRange(0x1000, 0x1000, 0x4000, 0x4000, 0x9000, 0x400);
    try std.testing.expectError(error.Aliasing, tower.outputRange(0x1000, 0x1000, 0x4000, 0x4000, 0x1ffe, 4));
    try std.testing.expectError(error.Aliasing, tower.outputRange(0x1000, 0x1000, 0x4000, 0x4000, 0x3ffe, 4));
    try std.testing.expectError(error.Aliasing, tower.outputRange(0x1000, 0x1000, 0x4000, 0x4000, 0x7ffe, 4));
    try std.testing.expectError(error.RowShape, tower.outputRange(0x1000, 0x1000, 0x4000, 0x4000, 0, 4));
    try std.testing.expectError(error.RowShape, tower.outputRange(0x1000, 0x1000, 0x4000, 0x4000, 0x9001, 4));
    try std.testing.expectError(error.RowShape, tower.outputRange(0x1000, 0x1000, 0x4000, 0x4000, std.math.maxInt(u64) - 1, 4));
}
