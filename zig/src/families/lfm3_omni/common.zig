//! The reply loop's backend-neutral parts: sampling, sink, statistics, the codec's synthesis and the host's code pick.
const std = @import("std");
const cfg = @import("config.zig");

pub const Sampling = struct {
    /// 0 picks the most likely code; the model card's setting is 0.4.
    audio_temperature: f32 = 0.4,
    audio_top_k: usize = 0,
    audio_repetition_penalty: f32 = 1.2,
    audio_repetition_window: usize = 64,
    /// Reveal the codebooks in order 0..11 instead of a random order (the reference's randperm).
    identity_order: bool = false,
    seed: u64 = 0,
    /// Generated slots (a text token or a speech frame each).
    max_new: usize = 512,
};

/// Where a reply goes as it is made.
pub const Sink = struct {
    ctx: *anyopaque,
    text: ?*const fn (*anyopaque, u32) anyerror!void = null,
    frame: ?*const fn (*anyopaque, *const [cfg.codebooks]i32) anyerror!void = null,
    pcm: ?*const fn (*anyopaque, []const f32) anyerror!void = null,
    /// After every backbone row (the engine's `lastHidden` reads it).
    hidden: ?*const fn (*anyopaque) anyerror!void = null,
};

/// When the codec runs while speech frames arrive: `first` frames for the first audio, then every `chunk`.
pub const Streaming = struct {
    first: usize = 1,
    chunk: usize = 8,
    /// Decode audio at all (frames only otherwise).
    audio: bool = true,
};

/// Teacher forcing: the reference reply's slots and frames are fed whatever the engine picks; the picks are scored.
pub const Forced = struct {
    seq: []const u32,
    frames: []const [cfg.codebooks]i32,
    text_hits: usize = 0,
    text_total: usize = 0,
    code_hits: usize = 0,
    code_total: usize = 0,
    frame_hits: usize = 0,
    frame_total: usize = 0,
};

pub const Stats = struct {
    mel_ms: f64 = 0,
    encode_ms: f64 = 0,
    prefill_ms: f64 = 0,
    first_text_ms: f64 = 0,
    first_frame_ms: f64 = 0,
    first_pcm_ms: f64 = 0,
    total_ms: f64 = 0,
    step_ms: f64 = 0,
    depth_ms: f64 = 0,
    codec_ms: f64 = 0,
    text_tokens: usize = 0,
    frames: usize = 0,
    samples: usize = 0,
    prompt_rows: usize = 0,
    audio_tokens: usize = 0,
};

pub fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

pub fn msSince(io: std.Io, t0: i128) f64 {
    return @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e6;
}

pub const Mode = enum { text, audio, interleaved };

/// Frames the device's repetition window holds (the reference's default window).
pub const hist_cap = 64;

/// The codec's spectrum to windowed frames, [bins re | bins im] x n_fft: irfft (norm "backward") times the window.
pub fn synthesisBasis(gpa: std.mem.Allocator, window: []const f32) ![]f32 {
    const n = cfg.n_fft;
    const nb = cfg.spec_bins;
    const m = try gpa.alloc(f32, 2 * nb * n);
    for (0..nb) |k| {
        const c: f64 = if (k == 0 or k == nb - 1) 1.0 else 2.0;
        for (0..n) |t| {
            const a = 2.0 * std.math.pi * @as(f64, @floatFromInt((k * t) % n)) / @as(f64, @floatFromInt(n));
            const w: f64 = window[t];
            m[k * n + t] = @floatCast(c * @cos(a) / @as(f64, @floatFromInt(n)) * w);
            m[(nb + k) * n + t] = @floatCast(-c * @sin(a) / @as(f64, @floatFromInt(n)) * w);
        }
    }
    return m;
}

/// The codec's overlap-add on the host: frames in, finished samples out (held back where later frames still add).
pub const OverlapAdd = struct {
    gpa: std.mem.Allocator,
    window_sq: [cfg.n_fft]f32,
    acc: std.ArrayList(f32) = .empty,
    env: std.ArrayList(f32) = .empty,
    rows: usize = 0,
    emitted: usize = cfg.istft_trim,
    out: std.ArrayList(f32) = .empty,

    pub fn init(gpa: std.mem.Allocator, window: []const f32) OverlapAdd {
        var o: OverlapAdd = .{ .gpa = gpa, .window_sq = undefined };
        for (&o.window_sq, window) |*q, w| q.* = w * w;
        return o;
    }

    pub fn deinit(o: *OverlapAdd) void {
        o.acc.deinit(o.gpa);
        o.env.deinit(o.gpa);
        o.out.deinit(o.gpa);
    }

    /// Adds windowed frames [n, n_fft]; returns the samples now final (valid until the next call).
    pub fn add(o: *OverlapAdd, frames: []const f32, final: bool) ![]const f32 {
        const n = frames.len / cfg.n_fft;
        for (0..n) |j| {
            const base = (o.rows + j) * cfg.hop;
            const need = base + cfg.n_fft;
            while (o.acc.items.len < need) {
                try o.acc.append(o.gpa, 0);
                try o.env.append(o.gpa, 0);
            }
            const fr = frames[j * cfg.n_fft ..][0..cfg.n_fft];
            for (0..cfg.n_fft) |t| {
                o.acc.items[base + t] += fr[t];
                o.env.items[base + t] += o.window_sq[t];
            }
        }
        o.rows += n;
        const end = if (final) o.rows * cfg.hop + cfg.istft_trim else o.rows * cfg.hop;
        o.out.clearRetainingCapacity();
        while (o.emitted < end and o.emitted < o.acc.items.len) : (o.emitted += 1) try o.out.append(o.gpa, o.acc.items[o.emitted] / o.env.items[o.emitted]);
        return o.out.items;
    }
};

pub fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// One codebook's code: the repetition penalty over its recent codes, then greedy or tempered sampling.
pub fn pickCode(logits: []f32, q: usize, smp: Sampling, rng: std.Random, history: []const [cfg.codebooks]i32) i32 {
    const p = smp.audio_repetition_penalty;
    if (p != 1.0) {
        var seen: [cfg.codebook_size]bool = @splat(false);
        for (history) |fr| {
            const c: usize = @intCast(fr[q]);
            if (seen[c]) continue;
            seen[c] = true;
            logits[c] = if (logits[c] < 0) logits[c] * p else logits[c] / p;
        }
    }
    if (smp.audio_temperature == 0) {
        var best: usize = 0;
        for (logits, 0..) |v, i| if (v > logits[best]) {
            best = i;
        };
        return @intCast(best);
    }
    const t = smp.audio_temperature;
    var m: f32 = -std.math.inf(f32);
    for (logits) |*v| {
        v.* /= t;
        m = @max(m, v.*);
    }
    if (smp.audio_top_k > 0 and smp.audio_top_k < logits.len) {
        // the k-th largest as a floor
        var tmp: [cfg.codebook_size]f32 = undefined;
        @memcpy(tmp[0..logits.len], logits);
        std.mem.sort(f32, tmp[0..logits.len], {}, std.sort.desc(f32));
        const floor = tmp[smp.audio_top_k - 1];
        for (logits) |*v| if (v.* < floor) {
            v.* = -std.math.inf(f32);
        };
    }
    var sum: f64 = 0;
    for (logits) |*v| {
        v.* = @exp(v.* - m);
        sum += v.*;
    }
    var u = rng.float(f64) * sum;
    for (logits, 0..) |v, i| {
        u -= v;
        if (u <= 0) return @intCast(i);
    }
    return @intCast(logits.len - 1);
}

test "overlap-add: streamed chunks give the samples of one whole decode" {
    const gpa = std.testing.allocator;
    var window: [cfg.n_fft]f32 = undefined;
    for (&window, 0..) |*w, i| w.* = @floatCast(0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / cfg.n_fft));
    const rows = 37;
    const frames = try gpa.alloc(f32, rows * cfg.n_fft);
    defer gpa.free(frames);
    var prng = std.Random.DefaultPrng.init(7);
    for (frames) |*x| x.* = prng.random().float(f32) - 0.5;
    var whole = OverlapAdd.init(gpa, &window);
    defer whole.deinit();
    const a = try gpa.dupe(f32, try whole.add(frames, true));
    defer gpa.free(a);
    try std.testing.expectEqual(@as(usize, rows * cfg.hop), a.len);
    var parts = OverlapAdd.init(gpa, &window);
    defer parts.deinit();
    var b: std.ArrayList(f32) = .empty;
    defer b.deinit(gpa);
    var at: usize = 0;
    for ([_]usize{ 6, 6, 12, 1, 12 }) |n| {
        try b.appendSlice(gpa, try parts.add(frames[at * cfg.n_fft .. (at + n) * cfg.n_fft], at + n == rows));
        at += n;
    }
    try std.testing.expectEqualSlices(f32, a, b.items);
}

test "synthesis basis: a cosine bin comes back as the windowed cosine" {
    const gpa = std.testing.allocator;
    var window: [cfg.n_fft]f32 = @splat(1);
    const m = try synthesisBasis(gpa, &window);
    defer gpa.free(m);
    // the spectrum of cos(2 pi 3 t / n) is n / 2 at bin 3 (real)
    for (0..cfg.n_fft) |t| {
        const want = @cos(2.0 * std.math.pi * 3.0 * @as(f64, @floatFromInt(t)) / cfg.n_fft);
        const got = @as(f64, m[3 * cfg.n_fft + t]) * cfg.n_fft / 2.0;
        try std.testing.expectApproxEqAbs(want, got, 1e-5);
    }
}
