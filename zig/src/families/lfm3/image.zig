//! LFM3-VL pictures on the CPU: smart_resize, normalize, patchify (merge-block raster, temporal axis).
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const patch = 16;
pub const merge = 2;
pub const temporal = 2;
pub const channels = 3;
pub const step = patch * merge; // 32: the processor's resize multiple
pub const patch_width = channels * temporal * patch * patch; // 1536

/// The processor's placeholder token: `<|image|>`, the checkpoint config's `image_token_id`.
pub const image_token_id: u32 = 125030;

/// The trunk rows a picture occupies: one per 2x2 merge block, i.e. gh*gw / (merge*merge).
pub fn tokenCount(gh: usize, gw: usize) usize {
    return (gh / merge) * (gw / merge);
}

/// Smart-resize a picture to a multiple of `step` (round, at least one step).
pub fn gridSize(w: usize, h: usize) [2]usize {
    const nw = @max(step, @as(usize, @intFromFloat(@round(@as(f64, @floatFromInt(w)) / @as(f64, step)))) * step);
    const nh = @max(step, @as(usize, @intFromFloat(@round(@as(f64, @floatFromInt(h)) / @as(f64, step)))) * step);
    return .{ nw, nh };
}

/// Largest grid whose rows fit `max_tokens`, aspect kept; `detail:"low"` shrinks instead of refusing.
pub fn fitGrid(w: usize, h: usize, max_tokens: usize) [2]usize {
    const natural = gridSize(w, h);
    if (max_tokens == 0) return natural;
    var pw = @max(merge, even(natural[0] / patch)); // patch units; even so merge blocks divide
    var ph = @max(merge, even(natural[1] / patch));
    if (tokenCount(ph, pw) <= max_tokens) return natural;
    const s = @sqrt(@as(f64, @floatFromInt(max_tokens)) / @as(f64, @floatFromInt(tokenCount(ph, pw))));
    pw = @max(merge, even(@as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(pw)) * s)))));
    ph = @max(merge, even(@as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(ph)) * s)))));
    while (tokenCount(ph, pw) > max_tokens and (pw > merge or ph > merge)) {
        if (pw >= ph and pw > merge) {
            pw -= merge;
        } else if (ph > merge) {
            ph -= merge;
        } else break;
    }
    return .{ pw * patch, ph * patch };
}

fn even(n: usize) usize {
    return n & ~@as(usize, 1);
}

/// Bilinear resample of an RGB picture (interleaved, w*h*3 bytes) to nw x nh, normalized to [-1, 1].
pub fn resample(a: Allocator, rgb: []const u8, w: usize, h: usize, nw: usize, nh: usize) ![]f32 {
    const out = try a.alloc(f32, nw * nh * channels);
    errdefer a.free(out);
    for (0..nh) |y| {
        // pixel centres, clamped
        const fy = if (nh == 1) 0.0 else (@as(f64, @floatFromInt(y)) + 0.5) * @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(nh)) - 0.5;
        const y0: usize = @intFromFloat(@max(0.0, @floor(fy)));
        const y1 = @min(y0 + 1, h - 1);
        const wy = @max(0.0, @min(1.0, fy - @floor(fy)));
        for (0..nw) |x| {
            const fx = if (nw == 1) 0.0 else (@as(f64, @floatFromInt(x)) + 0.5) * @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(nw)) - 0.5;
            const x0: usize = @intFromFloat(@max(0.0, @floor(fx)));
            const x1 = @min(x0 + 1, w - 1);
            const wx = @max(0.0, @min(1.0, fx - @floor(fx)));
            for (0..channels) |c| {
                const p00: f64 = @floatFromInt(rgb[(y0 * w + x0) * channels + c]);
                const p01: f64 = @floatFromInt(rgb[(y0 * w + x1) * channels + c]);
                const p10: f64 = @floatFromInt(rgb[(y1 * w + x0) * channels + c]);
                const p11: f64 = @floatFromInt(rgb[(y1 * w + x1) * channels + c]);
                const top = p00 + (p01 - p00) * wx;
                const bot = p10 + (p11 - p10) * wx;
                const v = top + (bot - top) * wy;
                // the processor's normalize: x / 127.5 - 1
                out[(y * nw + x) * channels + c] = @floatCast(v / 127.5 - 1.0);
            }
        }
    }
    return out;
}

/// Patch tensor [gh*gw, 1536]: row = (bh,bw) raster; within a row channel, temporal, then (y,x).
pub fn patchify(a: Allocator, px: []const f32, nw: usize, nh: usize) !struct { rows: []f32, gh: usize, gw: usize } {
    const gh = nh / patch;
    const gw = nw / patch;
    const bh = gh / merge;
    const bw = gw / merge;
    const rows = try a.alloc(f32, gh * gw * patch_width);
    errdefer a.free(rows);
    var r: usize = 0;
    for (0..bh) |bhi| {
        for (0..bw) |bwi| {
            // a row is ONE patch: [c, t, ph, pw]; the merge picks WHICH patch, hence the row
            for (0..merge) |mh| {
                for (0..merge) |mw| {
                    var at: usize = 0;
                    for (0..channels) |c| {
                        for (0..temporal) |_| {
                            for (0..patch) |ph| {
                                for (0..patch) |pw| {
                                    const y = bhi * merge * patch + mh * patch + ph;
                                    const x = bwi * merge * patch + mw * patch + pw;
                                    rows[r * patch_width + at] = px[(y * nw + x) * channels + c];
                                    at += 1;
                                }
                            }
                        }
                    }
                    r += 1;
                }
            }
        }
    }
    return .{ .rows = rows, .gh = gh, .gw = gw };
}

test "patchify: merge-block to raster with the temporal axis after channels" {
    const a = std.testing.allocator;
    // TWO merge blocks each way (64x64 px): a 32x32 picture cannot fail block addressing.
    const nw = 64;
    const nh = 64;
    const px = try a.alloc(f32, nw * nh * channels);
    defer a.free(px);
    for (0..nh) |y| {
        for (0..nw) |x| {
            for (0..channels) |c| {
                px[(y * nw + x) * channels + c] = @floatFromInt(y * 100 + x * 10 + c);
            }
        }
    }
    const got = try patchify(a, px, nw, nh);
    defer a.free(got.rows);
    try std.testing.expectEqual(@as(usize, 4), got.gh);
    try std.testing.expectEqual(@as(usize, 4), got.gw);
    try std.testing.expectEqual(@as(usize, 4 * 4), got.rows.len / patch_width);

    // expected order: rows are (bh,bw,mh,mw) in raster; a row is [c,t,ph,pw]
    var at: usize = 0;
    const bgh = got.gh / merge;
    for (0..bgh) |bhi| {
        for (0..bgh) |bwi| {
            for (0..merge) |mh| {
                for (0..merge) |mw| {
                    var row_at: usize = 0;
                    for (0..channels) |c| {
                        for (0..temporal) |_| {
                            for (0..patch) |ph| {
                                for (0..patch) |pw| {
                                    const y = bhi * merge * patch + mh * patch + ph;
                                    const x = bwi * merge * patch + mw * patch + pw;
                                    const want: f32 = @floatFromInt(y * 100 + x * 10 + c);
                                    const r = ((bhi * bgh + bwi) * merge + mh) * merge + mw;
                                    try std.testing.expectEqual(want, got.rows[r * patch_width + row_at]);
                                    row_at += 1;
                                }
                            }
                        }
                    }
                    try std.testing.expectEqual(patch_width, row_at);
                    at += patch_width;
                }
            }
        }
    }
    try std.testing.expectEqual(4 * 4 * patch_width, at);
}

test "tokenCount: a 30x40 grid is 300 trunk rows" {
    try std.testing.expectEqual(@as(usize, 300), tokenCount(30, 40));
    try std.testing.expectEqual(@as(usize, 1), tokenCount(2, 2));
}

test "fitGrid: a budget below the natural grid shrinks, it does not refuse" {
    // fits as-is: the natural grid comes back untouched
    try std.testing.expectEqual(gridSize(512, 384), fitGrid(512, 384, 256));
    try std.testing.expectEqual(gridSize(850, 600), fitGrid(850, 600, 4000));

    // 640x480 needs 300 rows; low detail allows 256 -> shrink to fit, keep aspect roughly
    const low = fitGrid(640, 480, 256);
    try std.testing.expect(low[0] % step == 0 and low[1] % step == 0);
    const low_tokens = tokenCount(low[1] / patch, low[0] / patch);
    try std.testing.expect(low_tokens <= 256);
    try std.testing.expect(low_tokens >= 128); // still a useful picture, not a thumbnail
    const want_aspect = @as(f64, 640.0) / 480.0;
    const got_aspect = @as(f64, @floatFromInt(low[0])) / @as(f64, @floatFromInt(low[1]));
    try std.testing.expect(@abs(got_aspect - want_aspect) < 0.15);

    // a tall picture shrinks on the tall axis
    const tall = fitGrid(300, 1600, 256);
    try std.testing.expect(tokenCount(tall[1] / patch, tall[0] / patch) <= 256);
    try std.testing.expect(tall[1] > tall[0]);

    // never smaller than one merge block
    const tiny = fitGrid(5000, 4000, 1);
    try std.testing.expectEqual(@as(usize, step), tiny[0]);
    try std.testing.expectEqual(@as(usize, step), tiny[1]);
    try std.testing.expectEqual(@as(usize, 1), tokenCount(tiny[1] / patch, tiny[0] / patch));
}
