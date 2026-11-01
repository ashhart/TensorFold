//! BiRefNet's released geometry and window maps, shared by the Metal and CUDA forwards (no GPU code here).
const std = @import("std");

/// Swin-L: embed 192, depths 2 2 18 2, heads 6 12 24 48, windows of 12.
pub const embed = 192;
pub const depths = [4]usize{ 2, 2, 18, 2 };
pub const heads = [4]usize{ 6, 12, 24, 48 };
pub const window = 12;
pub const widths = [4]usize{ 192, 384, 768, 1536 };
/// The decoder's channels after the backbone's two passes are concatenated (mul_scl_ipt='cat').
pub const channels = [4]usize{ 3072, 1536, 768, 384 };
pub const inter = 64;
/// The squeeze's input: the four concatenated features, the three finer ones resized to the coarsest grid.
pub const squeeze_in = channels[0] + channels[1] + channels[2] + channels[3];
pub const aspp_planes = 256;
/// The tensors the file holds: 742, ZhengPeng7/BiRefNet's less its 12 training-only ones.
pub const tensor_count = 742;
pub const ln_eps: f32 = 1e-5;
pub const bn_eps: f64 = 1e-5;
/// The deformable branches' kernel sides: aspp1, then aspp_deforms 0, 1, 2.
pub const branch_k = [4]usize{ 1, 1, 3, 7 };

/// The network's input side, and its two passes' patch grids (the full picture's, then the half picture's).
pub const size = 1024;
pub const grids = [2]usize{ size / 4, size / 8 };
/// Pixels a deformable branch builds columns for at once.
pub const chunk = 8192;

/// Each decoder level's input channels (the previous level's output plus its image patches' convolution), its side.
pub const level_in = [4]usize{ channels[0] + channels[0] / 8, channels[1] + channels[0] / 8, channels[2] + channels[1] / 8, channels[3] + channels[2] / 8 };
pub const level_side = [4]usize{ 32, 64, 128, 256 };
/// Each level's output channels (decoder_block4 .. 1), and the image patches' tile side at each level.
pub const level_out = [4]usize{ channels[1], channels[2], channels[3], channels[3] / 2 };
pub const ipt_tile = [4]usize{ 32, 16, 8, 4 };

/// The padded side of a grid of `n` (a whole number of windows).
pub fn padded(n: usize) usize {
    return (n + window - 1) / window * window;
}

/// Grid h's window map shifted by `shift`: each slot (144 a window) to the token torch.roll puts there, -1 padding.
pub fn fillMap(out: []i32, h: usize, shift: usize) void {
    const hp = padded(h);
    const per = hp / window;
    for (out, 0..) |*m, slot| {
        const win = slot / 144;
        const t = slot % 144;
        const sy = (win / per) * window + t / window;
        const sx = (win % per) * window + t % window;
        const oy = (sy + shift) % hp;
        const ox = (sx + shift) % hp;
        m.* = if (oy < h and ox < h) @intCast(oy * h + ox) else -1;
    }
}

test "a window map sends every token to one slot, the padding nowhere" {
    const a = std.testing.allocator;
    for ([_]usize{ 0, 6 }) |shift| {
        const h = 16;
        const hp = padded(h);
        const m = try a.alloc(i32, hp * hp);
        defer a.free(m);
        fillMap(m, h, shift);
        var seen: [h * h]u8 = @splat(0);
        var pad: usize = 0;
        for (m) |v| {
            if (v < 0) pad += 1 else seen[@intCast(v)] += 1;
        }
        try std.testing.expectEqual(hp * hp - h * h, pad);
        for (seen) |s| try std.testing.expectEqual(@as(u8, 1), s);
    }
    var m: [24 * 24]i32 = undefined;
    fillMap(&m, 16, 6);
    // the first window's first slot holds token (6, 6) once rolled by -6
    try std.testing.expectEqual(@as(i32, 6 * 16 + 6), m[0]);
}

test "the decoder's widths follow the backbone's: 5,760 into the squeeze, the levels' inputs, 240 into the head" {
    try std.testing.expectEqual(@as(usize, 5760), squeeze_in);
    try std.testing.expectEqual([4]usize{ 3456, 1920, 960, 480 }, level_in);
    try std.testing.expectEqual(@as(usize, 240), level_out[3] + channels[3] / 8);
}
