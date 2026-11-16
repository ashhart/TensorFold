//! From generated rows to served audio: the undelayed code frames to decode, then the output gain and soft limiter.
const std = @import("std");
const cfg = @import("config.zig");

const C = cfg.codebooks;
const W = cfg.frame_width;

/// The decoder's headroom: the default output gain is -3 dB.
pub const headroom_db: f32 = 3.0;
/// The limiter's knee, -1 dBFS: samples at or below it pass unchanged.
pub const knee: f32 = 0.8912509381337456;

/// The frames to decode: `prefill` then `generated` rows undelayed, up to the first end of audio or last whole frame.
pub fn frames(gpa: std.mem.Allocator, prefill: []const [W]i32, generated: []const [W]i32, eos_frame: ?usize) ![][C]i32 {
    const n = prefill.len + generated.len;
    var target: usize = n -| (C - 1);
    if (eos_frame) |f| target = @min(target, f + prefill.len);
    for (0..n) |i| {
        const row = if (i < prefill.len) prefill[i] else generated[i - prefill.len];
        var last: ?usize = null;
        for (row[0..C], 0..) |code, cb| if (code == cfg.eoa_id) {
            last = cb;
        };
        if (last) |cb| target = @min(target, i -| cb);
    }
    const out = try gpa.alloc([C]i32, target);
    errdefer gpa.free(out);
    for (out, 0..) |*frame, f| for (frame, 0..) |*code, cb| {
        const i = f + cb;
        const row = if (i < prefill.len) prefill[i] else generated[i - prefill.len];
        if (row[cb] < 0 or row[cb] >= cfg.codebook_size) return error.FrameHoldsNoCode;
        code.* = row[cb];
    };
    return out;
}

/// The decoder's samples to the served ones in place: the headroom gain, then the soft limiter above the knee.
pub fn output(pcm: []f32, headroom: f32) void {
    const gain: f32 = @floatCast(std.math.pow(f64, 10.0, -@as(f64, headroom) / 20.0));
    const width = 1.0 - knee;
    for (pcm) |*s| {
        const x = s.* * gain;
        const mag = @abs(x);
        s.* = if (mag > knee) std.math.copysign(knee + width * std.math.tanh((mag - knee) / width), x) else x;
    }
}

/// Float samples as 16-bit: NaN is 0, clipped to [-1, 1], times 32767, rounded half to even.
pub fn pcm16(x: f32) i16 {
    if (std.math.isNan(x)) return 0;
    const v = @min(@max(x, -1.0), 1.0) * 32767.0;
    const r = @round(v);
    const even = if (@abs(v - @trunc(v)) == 0.5) 2.0 * @round(v / 2.0) else r;
    return @intFromFloat(even);
}

/// A mono 16-bit WAV file's bytes (caller frees).
pub fn wav(gpa: std.mem.Allocator, pcm: []const f32, rate: u32) ![]u8 {
    const data: u32 = @intCast(pcm.len * 2);
    const out = try gpa.alloc(u8, 44 + data);
    @memcpy(out[0..4], "RIFF");
    std.mem.writeInt(u32, out[4..8], 36 + data, .little);
    @memcpy(out[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, out[16..20], 16, .little);
    std.mem.writeInt(u16, out[20..22], 1, .little);
    std.mem.writeInt(u16, out[22..24], 1, .little);
    std.mem.writeInt(u32, out[24..28], rate, .little);
    std.mem.writeInt(u32, out[28..32], rate * 2, .little);
    std.mem.writeInt(u16, out[32..34], 2, .little);
    std.mem.writeInt(u16, out[34..36], 16, .little);
    @memcpy(out[36..40], "data");
    std.mem.writeInt(u32, out[40..44], data, .little);
    for (pcm, 0..) |x, i| std.mem.writeInt(i16, out[44 + 2 * i ..][0..2], pcm16(x), .little);
    return out;
}

test "frames: the delay comes off, and the first end-of-audio code bounds the audio" {
    const t = std.testing;
    var rows: [14][W]i32 = undefined;
    for (&rows, 0..) |*row, i| {
        for (row[0..C], 0..) |*code, cb| code.* = if (i >= cb) @intCast(10 * (i - cb) + cb) else cfg.audio_pad_id;
        row[C] = cfg.text_vocab;
    }
    const all = try frames(t.allocator, rows[0..2], rows[2..], null);
    defer t.allocator.free(all);
    try t.expectEqual(@as(usize, 5), all.len);
    for (all, 0..) |frame, f| for (frame, 0..) |code, cb| try t.expectEqual(@as(i32, @intCast(10 * f + cb)), code);
    rows[6][1] = cfg.eoa_id; // frame 5 of codebook 1: the audio ends before frame 5
    rows[6][0] = cfg.eoa_id;
    const cut = try frames(t.allocator, rows[0..2], rows[2..], 4);
    defer t.allocator.free(cut);
    try t.expectEqual(@as(usize, 5), cut.len);
    const early = try frames(t.allocator, rows[0..2], rows[2..], 1);
    defer t.allocator.free(early);
    try t.expectEqual(@as(usize, 3), early.len);
}

test "output: the gain applies, quiet samples pass and loud ones bend below full scale" {
    const t = std.testing;
    var pcm = [_]f32{ 0.5, -0.5, 3.0, -3.0 };
    output(&pcm, 0);
    try t.expectEqual(@as(f32, 0.5), pcm[0]);
    try t.expectEqual(@as(f32, -0.5), pcm[1]);
    try t.expect(pcm[2] > knee and pcm[2] <= 1.0);
    try t.expectEqual(-pcm[2], pcm[3]);
    var half = [_]f32{1.0};
    output(&half, 6.0205999);
    try t.expectApproxEqAbs(@as(f32, 0.5), half[0], 1e-6);
}

test "pcm16: rounds half to even and clips" {
    const t = std.testing;
    try t.expectEqual(@as(i16, 32767), pcm16(2.0));
    try t.expectEqual(@as(i16, -32767), pcm16(-1.0));
    try t.expectEqual(@as(i16, 0), pcm16(std.math.nan(f32)));
    try t.expectEqual(@as(i16, 2), pcm16(2.5 / 32767.0));
}
