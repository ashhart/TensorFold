//! Bounds for native 512-key groups and slice storage within PR726 dynamic partial allocations.
const std = @import("std");
pub const span: usize = 512;
pub const slice_items: usize = 1024;
const slices: usize = 4;

/// Each piece's committed keys plus its window rows bound every tree path in that piece.
pub fn groups(keys: usize) usize {
    return (keys + span - 1) / span;
}

/// Fixed one-warp slice storage follows combined partials even when a launch uses four-warp blocks.
pub fn sliceBytes(heads: usize, key_heads: usize, dim: usize) usize {
    return slice_items * slices * (heads / key_heads) * (dim + 2) * 4;
}

pub fn required(rows: usize, heads: usize, key_heads: usize, group_count: usize, dim: usize) usize {
    return rows * heads * group_count * (dim + 2) * 4 + sliceBytes(heads, key_heads, dim);
}

fn available(slot_rows: usize, heads: usize, key_heads: usize, dim: usize) usize {
    const aligned = std.mem.alignForward;
    return aligned(usize, slot_rows * heads * dim * 4, 256) +
        2 * aligned(usize, slot_rows * heads * 4, 256) + sliceBytes(heads, key_heads, dim);
}

test "512-key group bound includes longest piece and old 2048/4096/6144 boundaries" {
    for ([_]usize{ 1, 512, 513, 2048, 2049, 4096, 4097, 6144, 6145 }, [_]usize{ 1, 1, 2, 4, 5, 8, 9, 12, 13 }) |keys, want|
        try std.testing.expectEqual(want, groups(keys));
    try std.testing.expectEqual(@as(usize, 13), @max(groups(2048), groups(6145)));
}

test "initial calibration needs slice storage and fits all 14 measured widths" {
    const old = 1024 * 24 * (256 + 2) * 4;
    try std.testing.expect(required(256, 24, 4, 1, 256) > old);
    for ([_]usize{ 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256 }) |rows|
        try std.testing.expect(required(rows, 24, 4, 1, 256) <= available(1024, 24, 4, 256));
}

test "n1 and n16 fitted capacities retain enough combined and slice storage" {
    for ([_]usize{ 1, 512, 2047, 2048, 2049, 4095, 4096, 4097, 6144, 32768 }) |capacity| {
        for ([_]usize{ 1, 16 }) |streams| {
            const slot_rows = @max(1024, streams * ((capacity + 16 + 511) / 512 + 1) * 16);
            try std.testing.expect(required(streams * 16, 24, 4, groups(capacity + 16), 256) <= available(slot_rows, 24, 4, 256));
        }
    }
}

test "wide unfitted CLI cannot assume contiguous native scratch fits" {
    try std.testing.expect(required(128, 24, 4, groups(6144 + 128), 256) > available(1024, 24, 4, 256));
}
