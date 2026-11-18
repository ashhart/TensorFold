//! ZONOS3's frame sampler on the host: specials masked, the repetition penalty, then the keyed sampler a codebook.
const std = @import("std");
const cfg = @import("config.zig");
const keyed_sampler = @import("lanes").sampling;

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
    /// Added to the stop id's logit in every codebook before the filters (LACI's stop boost); 0: nothing.
    stop_bias: f32 = 0,

    /// The settings as every sampler applies them: a penalty of at least 1, a negative count every codebook.
    pub fn clamped(s: Sampling) Sampling {
        var c = s;
        c.repetition_penalty = @max(s.repetition_penalty, 1.0);
        c.repetition_codebooks = if (s.repetition_codebooks < 0) C else @min(s.repetition_codebooks, C);
        return c;
    }

    /// The keyed sampler's rule: top_k < 1 or past the codebook keeps every candidate, temperature 0 is greedy.
    pub fn keyed(s: Sampling) keyed_sampler.Sampling {
        const k: u32 = if (s.top_k >= 1 and s.top_k < cfg.codebook_size) s.top_k else 0;
        return .{ .seed = s.seed, .temperature = s.temperature, .top_k = k, .top_p = s.top_p, .min_p = @max(s.min_p, 0.0) };
    }
};

/// The keyed sampler's position of a frame's codebook: every draw of a reply has its own.
pub fn position(frame: u64, codebook: u64) u64 {
    return frame * C + codebook;
}

/// Frame `frame`'s codes from its logits (overwritten); `recent`: generated rows, oldest first; `stop`: the end id.
pub fn sample(gpa: std.mem.Allocator, logits: *[C][V]f32, recent: []const [C]i32, frame: u64, stop: i32, sampling: Sampling) ![C]i32 {
    const s = sampling.clamped();
    const rule = s.keyed();
    const window: usize = if (s.repetition_window > 0 and s.repetition_penalty > 1.0 and s.repetition_codebooks > 0) @min(s.repetition_window, recent.len) else 0;
    var out: [C]i32 = undefined;
    var values: [V]f64 = undefined;
    var ids: [V]u64 = undefined;
    for (logits, 0..) |*l, cb| {
        for (l[cfg.codebook_size..], cfg.codebook_size..) |*v, id| {
            if (id == cfg.audio_pad_id or id != stop) v.* = -std.math.inf(f32);
        }
        if (s.stop_bias != 0 and stop >= 0 and stop < V) l[@intCast(stop)] += s.stop_bias;
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
        // the masked ids are no candidates: the keyed sampler draws among the rest, in float64
        var n: usize = 0;
        for (l, 0..) |v, id| {
            if (v == -std.math.inf(f32)) continue;
            values[n] = v;
            ids[n] = id;
            n += 1;
        }
        out[cb] = @intCast(try keyed_sampler.choose(gpa, values[0..n], ids[0..n], position(frame, cb), rule));
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
    const got = try sample(t.allocator, logits, &.{}, 0, cfg.eoa_id, .{ .temperature = 0 });
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
    const got = try sample(t.allocator, logits, &recent, 1, cfg.eoa_id, .{ .temperature = 0 });
    for (got, 0..) |code, cb| try t.expectEqual(@as(i32, if (cb < 6) 6 else 5), code);
}

test "sampler: a seeded draw depends on the seed, the frame and the codebook alone" {
    const t = std.testing;
    const logits = try t.allocator.create([C][V]f32);
    defer t.allocator.destroy(logits);
    var got: [3][C]i32 = undefined;
    for (&got, [_]u64{ 42, 42, 43 }) |*g, seed| {
        for (logits, 0..) |*l, cb| for (l, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt((i * 13 + cb * 5) % 101)) * 0.05;
        };
        g.* = try sample(t.allocator, logits, &.{}, 7, cfg.eoa_id, .{ .seed = seed, .temperature = 1.0, .top_k = 0, .top_p = 1.0 });
    }
    try t.expectEqual(got[0], got[1]);
    try t.expect(!std.mem.eql(i32, &got[0], &got[2]));
    for (got[0]) |code| try t.expect(code < cfg.codebook_size or code == cfg.eoa_id);
    try t.expect(position(3, 1) != position(3, 2) and position(3, 9) + 1 == position(4, 0));
}
