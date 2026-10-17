//! MiniMax Music 3's shapes, the request settings and the reference's schedules (ComfyUI's comfy/ldm/minimax_music).
const std = @import("std");

pub const hidden = 4096;
pub const layers = 36;
pub const heads = 32;
pub const kv_heads = 8;
pub const head_dim = 128;
pub const qkv_width = (heads + 2 * kv_heads) * head_dim; // 6144
pub const mlp = 12288;
pub const rope_theta: f32 = 1_000_000.0;
pub const eps: f32 = 1e-6;
pub const prompt_vocab = 151675; // embed_tokens_prefill rows (the text ids)
pub const c0_codes = 16384; // code 0's codes; lm_head_pruned has one more row (0 = stop)
pub const codebooks = 8;
pub const codebook_size = 1024;
pub const depth_layers = 4;
pub const depth_heads = 16;
pub const depth_head_dim = 256;
pub const depth_mlp = 6144;
pub const frame_rate = 25;
pub const sample_rate = 44100;
pub const max_frames = 9000; // 6 minutes
pub const max_prompt = 5000;

// the DiT
pub const dit_width = 2048;
pub const dit_blocks = 36;
pub const dit_heads = 32;
pub const dit_head_dim = 64;
pub const dit_mlp = 8192;
pub const dit_in = 2304; // [latent 128, zeros 128, condition 2048]
pub const latent_ch = 128;
pub const window_frames = 200;
pub const hop_frames = 100;

/// One request's knobs (the reference's defaults).
pub const Settings = struct {
    seconds: f64 = 60.0,
    seed: u64 = 0,
    ar_cfg: f32 = 1.5,
    top_k: i32 = 50,
    steps: usize = 30,
    dit_cfg: f32 = 1.7,

    /// The AR frame cap: round(seconds * 25), at most 9000.
    pub fn frameLimit(s: Settings) usize {
        const f = @round(s.seconds * frame_rate);
        if (!(f >= 1)) return 1;
        return @min(max_frames, @as(usize, @intFromFloat(f)));
    }
};

/// Latent frames (44100 / 512 a second) for AR frames (25 a second): max(1, int(frames 44100 / 24000 960 / 512)).
pub fn framesToLatents(frames: usize) usize {
    const v: f64 = @as(f64, @floatFromInt(frames)) * 44100.0 / 24000.0 * 960.0 / 512.0;
    return @max(1, @as(usize, @intFromFloat(@floor(v))));
}

/// PyTorch's interpolate(mode="nearest") indices from `source` rows to `target` (float32 scale, floor).
pub fn nearest(gpa: std.mem.Allocator, source: usize, target: usize) ![]i32 {
    const out = try gpa.alloc(i32, target);
    const scale: f32 = @as(f32, @floatFromInt(source)) / @as(f32, @floatFromInt(target));
    for (out, 0..) |*o, i| {
        const v: f32 = @floor(@as(f32, @floatFromInt(i)) * scale);
        o.* = @intCast(@min(@as(usize, @intFromFloat(v)), source - 1));
    }
    return out;
}

/// ComfyUI's "simple" schedule over 1000 discrete flow sigmas (shift 1): steps + 1 values from 1 down to 0.
pub fn sigmas(gpa: std.mem.Allocator, steps: usize) ![]f32 {
    const out = try gpa.alloc(f32, steps + 1);
    const stride: f64 = 1000.0 / @as(f64, @floatFromInt(steps));
    for (0..steps) |i| {
        const idx: usize = @intFromFloat(@floor(@as(f64, @floatFromInt(i)) * stride));
        // table[-(1 + idx)] with table[j] = (j + 1) / 1000 in float32
        const j = 999 - idx;
        out[i] = @as(f32, @floatFromInt(j + 1)) / @as(f32, 1000.0);
    }
    out[steps] = 0;
    return out;
}

pub const Span = struct { start: usize, len: usize };

/// The DiT's windows over `length` latent rows: 200 AR frames' worth every 100, the last one ending at length.
pub fn windows(gpa: std.mem.Allocator, length: usize) ![]Span {
    const size = framesToLatents(window_frames);
    const hop = framesToLatents(hop_frames);
    var list: std.ArrayList(Span) = .empty;
    errdefer list.deinit(gpa);
    if (length <= size) {
        try list.append(gpa, .{ .start = 0, .len = length });
        return list.toOwnedSlice(gpa);
    }
    var start: usize = 0;
    while (true) {
        const end = @min(start + size, length);
        try list.append(gpa, .{ .start = start, .len = end - start });
        if (end == length) break;
        start += hop;
    }
    return list.toOwnedSlice(gpa);
}

/// A stage's seed from the song seed: blake2b-64, personalization "minimax-ttm", as the reference derives it.
pub fn deriveSeed(seed: u64, part: []const u8) u64 {
    var person: [16]u8 = @splat(0);
    @memcpy(person[0..11], "minimax-ttm");
    var h = std.crypto.hash.blake2.Blake2b(64).init(.{ .context = person });
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, seed, .little);
    h.update(&b);
    var l: [4]u8 = undefined;
    std.mem.writeInt(u32, &l, @intCast(part.len), .little);
    h.update(&l);
    h.update(part);
    var out: [8]u8 = undefined;
    h.final(&out);
    return std.mem.readInt(u64, &out, .little) & ((@as(u64, 1) << 63) - 1);
}

test "schedules match the reference" {
    const gpa = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 689), framesToLatents(200));
    try std.testing.expectEqual(@as(usize, 344), framesToLatents(100));
    try std.testing.expectEqual(@as(usize, 5167), framesToLatents(1500));
    const s = try sigmas(gpa, 30);
    defer gpa.free(s);
    try std.testing.expectEqual(@as(f32, 1.0), s[0]);
    try std.testing.expectEqual(@as(f32, 0.967), s[1]);
    try std.testing.expectEqual(@as(f32, 0), s[30]);
    const w = try windows(gpa, 5167);
    defer gpa.free(w);
    try std.testing.expectEqual(@as(usize, 15), w.len);
    try std.testing.expectEqual(@as(usize, 5167), w[w.len - 1].start + w[w.len - 1].len);
    const n = try nearest(gpa, 1500, 5167);
    defer gpa.free(n);
    try std.testing.expectEqual(@as(i32, 1499), n[5166]);
    try std.testing.expectEqual(@as(usize, 1500), (Settings{ .seconds = 60 }).frameLimit());
}
