//! A speech take's tempo: waveform-similarity overlap-add (WSOLA) on the finished mono f32 PCM, pitch kept.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const min_speed: f64 = 0.25;
pub const max_speed: f64 = 4.0;

pub const Error = error{ NonFinite, InvalidSpeed, InvalidSampleRate, OutOfMemory };

/// Time-stretches `input` by `1 / speed` using Waveform Similarity Overlap-Add while preserving pitch.
pub fn stretch(a: Allocator, input: []const f32, sample_rate: u32, speed: f64) Error![]f32 {
    if (!std.math.isFinite(speed) or speed < min_speed or speed > max_speed) return error.InvalidSpeed;
    if (sample_rate == 0) return error.InvalidSampleRate;
    for (input) |x| if (!std.math.isFinite(x)) return error.NonFinite;
    if (input.len == 0 or speed == 1.0) return a.dupe(f32, input);

    const target_f = @round(@as(f64, @floatFromInt(input.len)) / speed);
    const target_len: usize = @max(@as(usize, 1), @as(usize, @intFromFloat(target_f)));
    if (input.len < 64 or target_len < 64) return linearStretch(a, input, target_len);

    var win: usize = @max(@as(usize, 64), (sample_rate * 30) / 1000);
    win = (win + 3) & ~@as(usize, 3);
    if (win > input.len / 2) win = (input.len / 2) & ~@as(usize, 3);
    if (win > target_len) win = target_len & ~@as(usize, 3);
    if (win < 32) return linearStretch(a, input, target_len);

    const hs = win / 4;
    const tol = @max(hs, sample_rate / 70);
    const max_ana = input.len - win;
    const cstep: usize = @max(@as(usize, 1), sample_rate / 12000);

    const out = try a.alloc(f32, target_len);
    errdefer a.free(out);
    @memset(out, 0);
    const norm = try a.alloc(f32, target_len);
    defer a.free(norm);
    @memset(norm, 0);
    const window = try a.alloc(f32, win);
    defer a.free(window);
    fillHann(window);

    overlapAdd(out, norm, 0, input[0..win], window);
    var prev_ana: usize = 0;
    var last_syn: usize = 0;
    var syn_pos: usize = hs;
    const denom_syn: f64 = @floatFromInt(@max(@as(usize, 1), target_len - win));
    const num_ana: f64 = @floatFromInt(max_ana);
    var done = (win >= target_len);
    while (!done) {
        if (syn_pos + win >= target_len) {
            syn_pos = target_len - win;
            done = true;
        }
        const nom_f = @round(@as(f64, @floatFromInt(syn_pos)) * num_ana / denom_syn);
        const nom: usize = @min(max_ana, @as(usize, @intFromFloat(nom_f)));
        const natural = prev_ana + (syn_pos - last_syn);
        const ana = if (natural <= max_ana)
            bestOffset(input, win, natural, nom, tol, max_ana, cstep)
        else
            nom;
        overlapAdd(out, norm, syn_pos, input[ana .. ana + win], window);
        prev_ana = ana;
        last_syn = syn_pos;
        syn_pos += hs;
    }
    for (out, norm) |*y, w| {
        if (w > 1e-9) y.* /= w;
    }
    return out;
}

fn fillHann(window: []f32) void {
    const two_pi = 2.0 * std.math.pi;
    const n: f64 = @floatFromInt(window.len);
    for (window, 0..) |*w, i| {
        const phase = two_pi * (@as(f64, @floatFromInt(i)) + 0.5) / n;
        w.* = @floatCast(0.5 * (1.0 - @cos(phase)));
    }
}

fn overlapAdd(out: []f32, norm: []f32, syn_pos: usize, frame: []const f32, window: []const f32) void {
    for (0..window.len) |i| {
        const w = window[i];
        out[syn_pos + i] += frame[i] * w;
        norm[syn_pos + i] += w;
    }
}

fn bestOffset(input: []const f32, win: usize, natural: usize, nom: usize, tol: usize, max_ana: usize, cstep: usize) usize {
    const low = nom -| tol;
    const high = @min(max_ana, nom + tol);
    var ref_energy: f64 = 0;
    var i: usize = 0;
    while (i < win) : (i += cstep) {
        const r: f64 = input[natural + i];
        ref_energy += r * r;
    }
    if (ref_energy <= 1e-12) return nom;

    var best_cand = nom;
    var best_score = scoreAt(input, win, natural, nom, nom, tol, cstep);
    var cand = low;
    while (cand <= high) : (cand += cstep) {
        const s = scoreAt(input, win, natural, cand, nom, tol, cstep);
        if (s > best_score) {
            best_score = s;
            best_cand = cand;
        }
    }
    if (cstep == 1) return best_cand;

    const f_low = @max(low, best_cand -| cstep);
    const f_high = @min(high, best_cand + cstep);
    var fine_best = best_cand;
    var fine_score = -std.math.inf(f64);
    var f_cand = f_low;
    while (f_cand <= f_high) : (f_cand += 1) {
        const s = scoreAt(input, win, natural, f_cand, nom, tol, 1);
        if (s > fine_score) {
            fine_score = s;
            fine_best = f_cand;
        }
    }
    return fine_best;
}

fn scoreAt(input: []const f32, win: usize, natural: usize, cand: usize, nom: usize, tol: usize, step: usize) f64 {
    var dot: f64 = 0;
    var e_ref: f64 = 0;
    var e_cand: f64 = 0;
    var i: usize = 0;
    while (i < win) : (i += step) {
        const r: f64 = input[natural + i];
        const c: f64 = input[cand + i];
        dot += r * c;
        e_ref += r * r;
        e_cand += c * c;
    }
    const sim = (2.0 * dot) / (e_ref + e_cand + 1e-12);
    const dist: f64 = if (cand >= nom) @floatFromInt(cand - nom) else @floatFromInt(nom - cand);
    const penalty = 1e-3 * dist / @as(f64, @floatFromInt(@max(@as(usize, 1), tol)));
    return sim - penalty;
}

fn linearStretch(a: Allocator, input: []const f32, target_len: usize) Allocator.Error![]f32 {
    const out = try a.alloc(f32, target_len);
    if (input.len == 1 or target_len == 1) {
        @memset(out, input[0]);
        return out;
    }
    const scale = @as(f64, @floatFromInt(input.len - 1)) / @as(f64, @floatFromInt(target_len - 1));
    for (out, 0..) |*y, i| {
        const pos = @as(f64, @floatFromInt(i)) * scale;
        const lo: usize = @min(input.len - 1, @as(usize, @intFromFloat(@floor(pos))));
        const hi: usize = @min(input.len - 1, lo + 1);
        const frac: f32 = @floatCast(pos - @as(f64, @floatFromInt(lo)));
        y.* = input[lo] * (1.0 - frac) + input[hi] * frac;
    }
    return out;
}

fn measureZeroCrossingF0(samples: []const f32, sample_rate: u32) f64 {
    const start = samples.len / 10;
    const end = samples.len - samples.len / 10;
    var first_cross: f64 = 0;
    var last_cross: f64 = 0;
    var crossings: usize = 0;
    var i = start;
    while (i + 1 < end) : (i += 1) {
        const y0 = samples[i];
        const y1 = samples[i + 1];
        if (y0 <= 0 and y1 > 0) {
            const frac: f64 = @as(f64, -y0) / @as(f64, y1 - y0);
            const t = @as(f64, @floatFromInt(i)) + frac;
            if (crossings == 0) first_cross = t;
            last_cross = t;
            crossings += 1;
        }
    }
    if (crossings < 2 or last_cross <= first_cross) return 0;
    const avg_period = (last_cross - first_cross) / @as(f64, @floatFromInt(crossings - 1));
    return @as(f64, @floatFromInt(sample_rate)) / avg_period;
}

test "WSOLA preserves output length within 1 percent and synthetic f0 within 2 percent" {
    const sr: u32 = 44100;
    const dur_samples = sr / 2;
    const f0: f64 = 220.0;
    const input = try std.testing.allocator.alloc(f32, dur_samples);
    defer std.testing.allocator.free(input);
    for (input, 0..) |*s, i| {
        s.* = @floatCast(0.5 * @sin(2.0 * std.math.pi * f0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(sr))));
    }
    for ([_]f64{ 0.25, 0.5, 0.75, 1.25, 2.0, 4.0 }) |spd| {
        const stretched = try stretch(std.testing.allocator, input, sr, spd);
        defer std.testing.allocator.free(stretched);
        const expected_len = @as(f64, @floatFromInt(input.len)) / spd;
        const len_err = @abs(@as(f64, @floatFromInt(stretched.len)) - expected_len) / expected_len;
        try std.testing.expect(len_err <= 0.01);
        const measured = measureZeroCrossingF0(stretched, sr);
        const f0_err = @abs(measured - f0) / f0;
        try std.testing.expect(f0_err <= 0.02);
        // overlapped frames add in phase: the level stays the tone's (RMS 0.5 / sqrt 2) away from the ends
        var power: f64 = 0;
        const middle = stretched[stretched.len / 10 .. stretched.len - stretched.len / 10];
        for (middle) |y| power += @as(f64, y) * y;
        try std.testing.expectApproxEqAbs(0.5 / std.math.sqrt2, std.math.sqrt(power / @as(f64, @floatFromInt(middle.len))), 0.01);
    }
}

test "WSOLA rejects non-finite samples and out-of-range speeds and returns exact duplicate at 1.0" {
    const sr: u32 = 44100;
    var ok = [_]f32{ 0.1, -0.2, 0.3, -0.4 };
    const same = try stretch(std.testing.allocator, &ok, sr, 1.0);
    defer std.testing.allocator.free(same);
    try std.testing.expectEqualSlices(f32, &ok, same);

    var bad = [_]f32{ 0.1, std.math.nan(f32), 0.3 };
    try std.testing.expectError(error.NonFinite, stretch(std.testing.allocator, &bad, sr, 1.25));
    try std.testing.expectError(error.InvalidSpeed, stretch(std.testing.allocator, &ok, sr, 0.24));
    try std.testing.expectError(error.InvalidSpeed, stretch(std.testing.allocator, &ok, sr, 4.01));
    try std.testing.expectError(error.InvalidSpeed, stretch(std.testing.allocator, &ok, sr, std.math.nan(f64)));
}
