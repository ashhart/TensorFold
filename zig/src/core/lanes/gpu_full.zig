//! Host reference of tf_sample_full, the Metal engine's draw for rows tf_gpu_sample would cut to 1,024 candidates.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Sampling = @import("sampling.zig").Sampling;
const rule = @import("gpu_rule.zig");

/// Tokens this far (in logits / T) below the row's max never win the race: its noise spans under 19.5.
pub const never: f32 = 24.0;

const below_one: f32 = @bitCast(@as(u32, 0x3F7F_FFFF));

/// A place in the (value desc, id asc) order.
pub const Mark = struct { key: u32, id: u32 };

/// Past every token.
pub const end = Mark{ .key = 0, .id = 0xFFFF_FFFF };

/// Before every token.
const start = Mark{ .key = 0xFFFF_FFFF, .id = 0xFFFF_FFFF };

/// A row's kept tokens: those at or above `lo`, at or before `top` (top_k) and at or before `cut` (top_p).
pub const Plan = struct { inv_t: f32, m: f32, lo: f32, top: Mark = end, cut: Mark = end };

/// Whether the Metal engine draws with tf_sample_full: top_k off, or above the candidates tf_gpu_sample keeps.
pub fn fullVocabulary(s: Sampling) bool {
    return s.top_k == 0 or s.top_k > rule.candidates;
}

/// One row's draw as the Metal engine draws it: argmax, tf_gpu_sample for top_k 1 to 1,024, else tf_sample_full.
pub fn draw(gpa: Allocator, logits: []const f32, s: ?Sampling, position: u32, ids: ?[]const u32) !u32 {
    if (s) |st| if (fullVocabulary(st)) return sample(gpa, logits, st, position, ids);
    return rule.sample(gpa, logits, s, position, ids);
}

/// tf_sample_full's draw of one row: the plan, then the race at `position`.
pub fn sample(gpa: Allocator, logits: []const f32, s: Sampling, position: u32, ids: ?[]const u32) !u32 {
    return race(logits, try plan(gpa, logits, s), s.seed, position, ids);
}

/// tf_uniform kept below 1: its top value rounds to 1.0, a score of +inf that would win from anywhere in the vocabulary.
pub fn raceUniform(seed: u64, position: u32, id: u32) f32 {
    return @min(rule.uniform24(seed, position, id), below_one);
}

fn atOrBefore(k: u32, i: u32, m: Mark) bool {
    return k > m.key or (k == m.key and i <= m.id);
}

fn after(k: u32, i: u32, m: Mark) bool {
    return k < m.key or (k == m.key and i > m.id);
}

fn beats(s: f32, k: u32, i: u32, bs: f32, bk: u32, bi: u32) bool {
    return s > bs or (s == bs and (k > bk or (k == bk and i < bi)));
}

fn markOf(c: rule.Cand) Mark {
    return .{ .key = rule.key(c.v), .id = c.i };
}

/// The kept set for `logits` under `s`, found as tf_sample_full finds it.
pub fn plan(gpa: Allocator, logits: []const f32, s: Sampling) !Plan {
    const inv_t: f32 = @floatCast(1.0 / @max(s.temperature, 1e-6));
    const order = try gpa.alloc(rule.Cand, logits.len);
    defer gpa.free(order);
    var m = -std.math.inf(f32);
    for (order, logits, 0..) |*c, l, i| {
        c.* = .{ .v = l * inv_t, .i = @intCast(i) };
        m = @max(m, c.v);
    }
    std.mem.sort(rule.Cand, order, {}, rule.better);
    const min_log: f32 = @floatCast(s.minLog());
    var p = Plan{ .inv_t = inv_t, .m = m, .lo = @max(m + min_log, m - never) };
    if (s.top_k != 0 and s.top_k < logits.len) p.top = markOf(order[s.top_k - 1]);
    const top_p: f32 = @floatCast(s.top_p);
    if (top_p > 0.0 and top_p < 1.0) p.cut = nucleus(logits, p, order, top_p);
    return p;
}

/// The nucleus's last token: tf_gpu_sample's sequential sum over its 1,024 candidates, extended when it falls short.
fn nucleus(logits: []const f32, p: Plan, order: []const rule.Cand, top_p: f32) Mark {
    const norm = mass(logits, p, 1.0, -std.math.inf(f32), start, p.top, end, false);
    const n = @min(order.len, rule.candidates);
    var cum: f32 = 0.0;
    for (order[0..n]) |c| {
        cum += @exp(c.v - p.m) / norm;
        if (cum >= top_p) return markOf(c);
    }
    return extend(logits, p, norm, top_p, cum, markOf(order[n - 1]));
}

/// The first place past `b` whose prefix reaches top_p, found by bits of key then id; `end` when none can win.
fn extend(logits: []const f32, p: Plan, norm: f32, top_p: f32, cum: f32, b: Mark) Mark {
    if (!(cum + mass(logits, p, norm, p.lo, b, p.top, end, false) >= top_p)) return end;
    var key: u32 = 0;
    var bit: u6 = 32;
    while (bit > 0) {
        bit -= 1;
        const cand = key | @as(u32, 1) << @intCast(bit);
        if (cum + mass(logits, p, norm, p.lo, b, p.top, .{ .key = cand, .id = 0xFFFF_FFFF }, false) >= top_p) key = cand;
    }
    var id: u32 = 0;
    bit = @intCast(32 - @clz(@as(u32, @intCast(logits.len - 1))));
    while (bit > 0) {
        bit -= 1;
        const cand = id | @as(u32, 1) << @intCast(bit);
        if (cand >= logits.len) continue;
        if (cum + mass(logits, p, norm, p.lo, b, p.top, .{ .key = key, .id = cand }, true) < top_p) id = cand;
    }
    return .{ .key = key, .id = id };
}

/// The kernel's mass(): exp(v - m) / norm over tokens at or above `lo`, after `b`, inside `top` and `cut`, in its order.
fn mass(logits: []const f32, p: Plan, norm: f32, lo: f32, b: Mark, top: Mark, cut: Mark, strict: bool) f32 {
    var lanes: [1024]f32 = @splat(0.0);
    for (logits, 0..) |l, i| {
        const v = l * p.inv_t;
        if (!(v >= lo)) continue;
        const k = rule.key(v);
        const id: u32 = @intCast(i);
        const inside = if (strict) k > cut.key or (k == cut.key and id < cut.id) else atOrBefore(k, id, cut);
        if (after(k, id, b) and atOrBefore(k, id, top) and inside) lanes[i % 1024] += @exp(v - p.m) / norm;
    }
    return rule.lanesSum(&lanes);
}

/// The race over a plan's kept tokens (value - log(-log(u))), ties to the earlier token in the order.
pub fn race(logits: []const f32, p: Plan, seed: u64, position: u32, ids: ?[]const u32) u32 {
    var bs = -std.math.inf(f32);
    var bk: u32 = 0;
    var bi: u32 = 0xFFFF_FFFF;
    for (logits, 0..) |l, i| {
        const v = l * p.inv_t;
        if (!(v >= p.lo)) continue;
        const k = rule.key(v);
        const id: u32 = @intCast(i);
        if (!atOrBefore(k, id, p.top) or !atOrBefore(k, id, p.cut)) continue;
        const score = v - @log(-@log(raceUniform(seed, position, if (ids) |map| map[i] else id)));
        if (beats(score, k, id, bs, bk, bi)) {
            bs = score;
            bk = k;
            bi = id;
        }
    }
    const pick = @min(bi, logits.len - 1);
    return if (ids) |map| map[pick] else @intCast(pick);
}
