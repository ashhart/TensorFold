//! ZONOS3's frame sampler on the host: specials masked, repetition penalty, temperature, top-k, top-p, min-p, one draw.
const std = @import("std");
const cfg = @import("config.zig");

const V = cfg.audio_vocab;
const C = cfg.codebooks;

pub const Sampling = struct {
    temperature: f32 = 0.75,
    top_k: u32 = 80,
    top_p: f32 = 0.69,
    min_p: f32 = 0.0,
    repetition_window: u32 = 96,
    repetition_penalty: f32 = 2.4,
    repetition_codebooks: i32 = 6,
    seed: u64 = 0,

    /// The settings as every sampler applies them: top_k < 1 is the whole codebook, a negative count every codebook.
    pub fn clamped(s: Sampling) Sampling {
        var c = s;
        c.temperature = @max(s.temperature, 1e-6);
        c.top_k = if (s.top_k >= 1) s.top_k else cfg.codebook_size;
        c.top_p = @min(@max(s.top_p, 0.0), 1.0);
        c.min_p = @max(s.min_p, 0.0);
        c.repetition_penalty = @max(s.repetition_penalty, 1.0);
        c.repetition_codebooks = if (s.repetition_codebooks < 0) C else @min(s.repetition_codebooks, C);
        return c;
    }
};

/// A draw in [0, 1) that depends only on the request's seed, the frame and the codebook.
pub fn uniform(seed: u64, frame: u64, codebook: u64) f32 {
    var z = seed +% 0x9E3779B97F4A7C15 *% (frame *% C + codebook + 1);
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    z ^= z >> 31;
    return @as(f32, @floatFromInt(z >> 40)) * (1.0 / 16777216.0);
}

fn argmax(x: []const f32) usize {
    var best: usize = 0;
    for (x, 0..) |v, i| if (v > x[best]) {
        best = i;
    };
    return best;
}

/// Frame `frame`'s codes from its logits (overwritten); `recent`: generated rows, oldest first; `stop`: the end id.
pub fn sample(logits: *[C][V]f32, recent: []const [C]i32, frame: u64, stop: i32, sampling: Sampling) [C]i32 {
    const s = sampling.clamped();
    const window: usize = if (s.repetition_window > 0 and s.repetition_penalty > 1.0 and s.repetition_codebooks > 0) @min(s.repetition_window, recent.len) else 0;
    var out: [C]i32 = undefined;
    var order: [V]u16 = undefined;
    for (logits, 0..) |*l, cb| {
        for (l[cfg.codebook_size..], cfg.codebook_size..) |*v, id| {
            if (id == cfg.audio_pad_id or id != stop) v.* = -std.math.inf(f32);
        }
        if (cb < s.repetition_codebooks) {
            var seen: [cfg.codebook_size]bool = @splat(false);
            for (recent[recent.len - window ..]) |row| {
                const code = row[cb];
                if (code < 0 or code >= cfg.codebook_size or seen[@intCast(code)]) continue;
                seen[@intCast(code)] = true;
                const v = &l[@intCast(code)];
                v.* = if (v.* > 0) v.* / s.repetition_penalty else v.* * s.repetition_penalty;
            }
        }
        for (l) |*v| v.* /= s.temperature;
        const greedy = argmax(l);
        for (&order, 0..) |*o, i| o.* = @intCast(i);
        const Desc = struct {
            fn lt(x: *const [V]f32, a: u16, b: u16) bool {
                return x[a] > x[b] or (x[a] == x[b] and a < b);
            }
        };
        std.mem.sort(u16, &order, l, Desc.lt);
        if (s.top_k < V) {
            const kth = l[order[s.top_k - 1]];
            for (l) |*v| if (v.* < kth) {
                v.* = -std.math.inf(f32);
            };
        }
        // softmax, in descending order so the nucleus cut reads the running sum
        const top = l[order[0]];
        var total: f32 = 0;
        for (l) |*v| {
            v.* = @exp(v.* - top);
            total += v.*;
        }
        for (l) |*v| v.* /= total;
        if (s.top_p > 0 and s.top_p < 1) {
            var before: f32 = 0;
            var kept: f32 = 0;
            for (order) |id| {
                const p = l[id];
                if (before > s.top_p) l[id] = 0 else kept += p;
                before += p;
            }
            const norm = @max(kept, 1e-8);
            for (l) |*v| v.* /= norm;
        }
        if (s.min_p > 0) {
            const floor = s.min_p * l[order[0]];
            var kept: f32 = 0;
            for (l) |*v| {
                if (v.* < floor) v.* = 0;
                kept += v.*;
            }
            const norm = @max(kept, 1e-8);
            for (l) |*v| v.* /= norm;
        }
        var sum: f32 = 0;
        for (l) |v| sum += v;
        if (!(sum > 0)) {
            out[cb] = @intCast(greedy);
            continue;
        }
        const target = uniform(s.seed, frame, cb) * sum;
        var run: f32 = 0;
        var pick: usize = greedy;
        for (l, 0..) |p, id| {
            if (p <= 0) continue;
            run += p;
            pick = id;
            if (run > target) break;
        }
        out[cb] = @intCast(pick);
    }
    return out;
}

test "sampler: specials other than the stop id are never drawn, and a zero temperature takes the best code" {
    const t = std.testing;
    const logits = try t.allocator.create([C][V]f32);
    defer t.allocator.destroy(logits);
    for (logits, 0..) |*l, cb| {
        for (l, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7 + cb) % 97)) * 0.01;
        l[cfg.audio_pad_id] = 50;
        l[100 + cb] = 9;
    }
    const got = sample(logits, &.{}, 0, cfg.eoa_id, .{ .temperature = 0 });
    for (got, 0..) |code, cb| try t.expectEqual(@as(i32, @intCast(100 + cb)), code);
}

test "sampler: a repeated code is penalised in the first codebooks only" {
    const t = std.testing;
    const logits = try t.allocator.create([C][V]f32);
    defer t.allocator.destroy(logits);
    for (logits) |*l| {
        @memset(l, 0);
        l[5] = 2.0;
        l[6] = 1.5;
    }
    const recent = [_][C]i32{@splat(5)};
    const got = sample(logits, &recent, 1, cfg.eoa_id, .{ .temperature = 0 });
    for (got, 0..) |code, cb| try t.expectEqual(@as(i32, if (cb < 6) 6 else 5), code);
}

test "sampler: the same seed, frame and codebook give the same draw" {
    const t = std.testing;
    try t.expectEqual(uniform(42, 3, 1), uniform(42, 3, 1));
    try t.expect(uniform(42, 3, 1) != uniform(42, 3, 2));
    try t.expect(uniform(42, 3, 1) < 1.0 and uniform(42, 3, 1) >= 0.0);
}
