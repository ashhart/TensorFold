//! The register guard: whether a phrase after the first opens far above the reply's other phrases in pitch.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Uses Praat's published To Pitch (ac) parameter defaults, not its analysis; measured values can differ.
const floor_hz = 75;
const ceiling_hz = 600;
const voicing = 0.45;
const silence = 0.03;
const octave_cost = 0.01;
/// Pitch is read every 10 ms over windows of two frames, the audio averaged down to 8 to 16 kHz.
const frames_per_s = 100;
const min_rate = 8000;
/// Unvoiced runs of 0.25 s part phrases; an opening is the median pitch of a phrase's first 0.25 s voiced.
const rest_frames = 25;
const open_frames = 25;

/// A phrase after the first trips at `st` semitones over the others' median opening with `high`% of it over `high_hz`.
pub const Rule = struct {
    st: ?f64 = null,
    high: ?f64 = null,
    high_hz: ?f64 = null,

    pub fn on(r: Rule) bool {
        return r.st != null;
    }

    pub fn validate(r: Rule) !void {
        if (r.st == null and r.high == null and r.high_hz == null) return;
        const st = r.st orelse return error.MissingRegisterThreshold;
        const high = r.high orelse return error.MissingRegisterThreshold;
        const hz = r.high_hz orelse return error.MissingRegisterThreshold;
        if (!(std.math.isFinite(st) and st > 0 and high > 0 and high <= 100 and hz >= floor_hz and hz <= ceiling_hz))
            return error.BadRegisterThreshold;
    }
};

const Phrase = struct { opening: f32, high: f64 };

/// Largest qualifying excess in semitones; zero for measured non-trips, null when no comparison can be measured.
pub fn jump(a: Allocator, pcm: []const f32, sample_rate: u32, rule: Rule) error{ OutOfMemory, NonFinite }!?f64 {
    var peak: f32 = 0;
    for (pcm) |x| {
        if (!std.math.isFinite(x)) return error.NonFinite;
        peak = @max(peak, @abs(x));
    }
    if (sample_rate < min_rate or peak == 0) return null;
    const step = sample_rate / min_rate;
    const rate = @as(f32, @floatFromInt(sample_rate)) / @as(f32, @floatFromInt(step));
    const hop: usize = @intFromFloat(rate / frames_per_s);
    const span = 2 * hop + @as(usize, @intFromFloat(rate / floor_hz)) + 2;
    const high_hz: f32 = @floatCast(rule.high_hz.?);
    var phrases: std.ArrayList(Phrase) = .empty;
    var run: Run = .{};
    var quiet: usize = 0;
    var x: [4 * min_rate / frames_per_s + 2 * min_rate / floor_hz + 2]f32 = undefined;
    var at: usize = 0;
    while ((at + span) * step <= pcm.len) : (at += hop) {
        for (x[0..span], 0..) |*v, i| {
            var sum: f32 = 0;
            for (pcm[(at + i) * step ..][0..step]) |s| sum += s;
            v.* = sum / @as(f32, @floatFromInt(step));
        }
        const f0 = pitch(x[0..span], 2 * hop, rate, silence * peak) orelse {
            quiet += 1;
            if (quiet == rest_frames) try run.close(a, &phrases);
            continue;
        };
        quiet = 0;
        if (run.voiced < open_frames) run.head[run.voiced] = f0;
        run.voiced += 1;
        run.high += @intFromBool(f0 > high_hz);
    }
    try run.close(a, &phrases);
    const list = phrases.items;
    if (list.len < 2) return null;
    const others = try a.alloc(f32, list.len - 1);
    var most: f64 = 0;
    for (list[1..], 1..) |p, i| {
        if (p.high < rule.high.?) continue;
        var n: usize = 0;
        for (list, 0..) |q, j| if (j != i) {
            others[n] = q.opening;
            n += 1;
        };
        const excess = 12 * std.math.log2(@as(f64, p.opening / median(others)));
        if (excess >= rule.st.?) most = @max(most, excess);
    }
    return most;
}

/// The phrase being read: its first voiced pitches, and its voiced and high frame counts.
const Run = struct {
    head: [open_frames]f32 = undefined,
    voiced: usize = 0,
    high: usize = 0,

    /// Ends the run, kept as a phrase when it voiced a whole opening.
    fn close(r: *Run, a: Allocator, phrases: *std.ArrayList(Phrase)) Allocator.Error!void {
        defer r.* = .{};
        if (r.voiced < open_frames) return;
        const share = 100 * @as(f64, @floatFromInt(r.high)) / @as(f64, @floatFromInt(r.voiced));
        try phrases.append(a, .{ .opening = median(&r.head), .high = share });
    }
};

/// Sorts `v` and returns its median.
fn median(v: []f32) f32 {
    std.mem.sort(f32, v, {}, std.sort.asc(f32));
    return (v[(v.len - 1) / 2] + v[v.len / 2]) / 2;
}

/// The pitch of `x` at `rate`, cross-correlating its first `width` samples; null when unvoiced or under `quiet`.
fn pitch(x: []const f32, width: usize, rate: f32, quiet: f32) ?f32 {
    const head = x[0..width];
    var top: f32 = 0;
    for (head) |v| top = @max(top, @abs(v));
    if (top < quiet) return null;
    const lo: usize = @intFromFloat(rate / ceiling_hz);
    const hi: usize = @intFromFloat(rate / floor_hz);
    const energy = dot(head, head);
    var r: [2 * min_rate / floor_hz + 2]f32 = undefined;
    for (lo - 1..hi + 2) |lag| {
        const shifted = x[lag..][0..head.len];
        const e = dot(shifted, shifted);
        r[lag] = if (e > 0) dot(head, shifted) / @sqrt(energy * e) else 0;
    }
    // each peak at its parabola's vertex, the higher pitch favoured by the octave cost
    var period: f32 = 0;
    var height: f32 = 0;
    var score = -std.math.inf(f32);
    for (lo..hi + 1) |lag| {
        const before, const at, const after = .{ r[lag - 1], r[lag], r[lag + 1] };
        if (at < before or at < after) continue;
        const bend = before - 2 * at + after;
        const shift = if (bend < 0) 0.5 * (before - after) / bend else 0;
        const peak = at - 0.25 * (before - after) * shift;
        const lags = @as(f32, @floatFromInt(lag)) + shift;
        const s = peak - octave_cost * std.math.log2(lags);
        if (s > score) {
            score = s;
            period = lags;
            height = peak;
        }
    }
    return if (height < voicing) null else rate / period;
}

fn dot(u: []const f32, v: []const f32) f32 {
    const V = @Vector(8, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= u.len) : (i += 8) acc += @as(V, u[i..][0..8].*) * @as(V, v[i..][0..8].*);
    var sum = @reduce(.Add, acc);
    for (u[i..], v[i..]) |p, q| sum += p * q;
    return sum;
}
