//! Group-limited expert routing: keep_groups then top-k, as glue.select and ling_forward.py take it.
const std = @import("std");

pub const max_experts = 512;
pub const max_groups = 32;
pub const max_topk = 31;

pub const Shape = struct {
    experts: u32,
    topk: u32,
    n_group: u32,
    topk_group: u32,
    scale: f32 = 1,

    pub fn groupSize(s: Shape) u32 {
        return s.experts / s.n_group;
    }
};

pub const Pick = struct { id: u32, wt: f32 };

fn sigmoidF32(x: f32) f32 {
    const a = @abs(x);
    const y = 1.0 / (1.0 + @exp(a));
    return if (x < 0) y else 1.0 - y;
}

fn sigmoidF64(x: f64) f64 {
    const a = @abs(x);
    const y = 1.0 / (1.0 + @exp(a));
    return if (x < 0) y else 1.0 - y;
}

/// Mask every expert outside the `keep` groups of the best top-2 sums (ties to the lower group).
pub fn keepGroups(choice: []f32, ne: u32, ng: u32, keep: u32) void {
    const gs = ne / ng;
    var gscore: [max_groups]f32 = @splat(-std.math.inf(f32));
    for (0..ng) |g| {
        var a: f32 = -std.math.inf(f32);
        var b: f32 = -std.math.inf(f32);
        for (0..gs) |j| {
            const c = choice[g * gs + j];
            if (c > a) {
                b = a;
                a = c;
            } else if (c > b) {
                b = c;
            }
        }
        gscore[g] = a + b;
    }
    var live: [max_groups]bool = @splat(true);
    var kept: u32 = 0;
    for (0..keep) |_| {
        var mx: f32 = -std.math.inf(f32);
        var id: u32 = ng;
        for (0..ng) |g| {
            if (live[g] and (gscore[g] > mx or (gscore[g] == mx and g < id))) {
                mx = gscore[g];
                id = @intCast(g);
            }
        }
        if (id == ng) break;
        kept |= @as(u32, 1) << @intCast(id);
        live[id] = false;
    }
    for (0..ne) |e| {
        if (((kept >> @intCast(e / gs)) & 1) == 0) choice[e] = -std.math.inf(f32);
    }
}

fn keepGroupsF64(choice: []f64, ne: u32, ng: u32, keep: u32) void {
    const gs = ne / ng;
    var gscore: [max_groups]f64 = @splat(-std.math.inf(f64));
    for (0..ng) |g| {
        var a: f64 = -std.math.inf(f64);
        var b: f64 = -std.math.inf(f64);
        for (0..gs) |j| {
            const c = choice[g * gs + j];
            if (c > a) {
                b = a;
                a = c;
            } else if (c > b) {
                b = c;
            }
        }
        gscore[g] = a + b;
    }
    var live: [max_groups]bool = @splat(true);
    var kept: u32 = 0;
    for (0..keep) |_| {
        var mx: f64 = -std.math.inf(f64);
        var id: u32 = ng;
        for (0..ng) |g| {
            if (live[g] and (gscore[g] > mx or (gscore[g] == mx and g < id))) {
                mx = gscore[g];
                id = @intCast(g);
            }
        }
        if (id == ng) break;
        kept |= @as(u32, 1) << @intCast(id);
        live[id] = false;
    }
    for (0..ne) |e| {
        if (((kept >> @intCast(e / gs)) & 1) == 0) choice[e] = -std.math.inf(f64);
    }
}

/// One row: sigmoid + bias, keep_groups when n_group > 1, then top-k (ties to the lower id), renormalized times scale.
pub fn selectF32(logits: []const f32, bias: []const f32, out: []Pick, s: Shape) void {
    const ne = s.experts;
    var score: [max_experts]f32 = undefined;
    var choice: [max_experts]f32 = undefined;
    for (0..ne) |e| {
        score[e] = sigmoidF32(logits[e]);
        choice[e] = score[e] + bias[e];
    }
    if (s.n_group > 1) keepGroups(choice[0..ne], ne, s.n_group, s.topk_group);
    var total: f32 = 0;
    for (0..s.topk) |k| {
        var mx: f32 = -std.math.inf(f32);
        var id: u32 = ne;
        for (0..ne) |e| {
            if (choice[e] > mx or (choice[e] == mx and e < id)) {
                mx = choice[e];
                id = @intCast(e);
            }
        }
        out[k] = .{ .id = id, .wt = if (id < ne) score[id] else 0 };
        total += out[k].wt;
        // NaN, not -inf: a picked -inf slot would match the next -inf tie and repeat the id.
        if (id < ne) choice[id] = std.math.nan(f32);
    }
    const den = total + 1e-20;
    for (out[0..s.topk]) |*p| p.wt = (p.wt / den) * s.scale;
}

pub const Pick64 = struct { id: u32, wt: f64 };

/// Float64 reference of the same formula (ling_forward.py's moe routing, ties to the lower group then the lower id).
pub fn selectF64(logits: []const f32, bias: []const f32, out: []Pick64, s: Shape) void {
    const ne = s.experts;
    var score: [max_experts]f64 = undefined;
    var choice: [max_experts]f64 = undefined;
    for (0..ne) |e| {
        score[e] = sigmoidF64(@floatCast(logits[e]));
        choice[e] = score[e] + @as(f64, bias[e]);
    }
    if (s.n_group > 1) keepGroupsF64(choice[0..ne], ne, s.n_group, s.topk_group);
    var total: f64 = 0;
    for (0..s.topk) |k| {
        var mx: f64 = -std.math.inf(f64);
        var id: u32 = ne;
        for (0..ne) |e| {
            if (choice[e] > mx or (choice[e] == mx and e < id)) {
                mx = choice[e];
                id = @intCast(e);
            }
        }
        out[k] = .{ .id = id, .wt = if (id < ne) score[id] else 0 };
        total += out[k].wt;
        if (id < ne) choice[id] = std.math.nan(f64);
    }
    const den = total + 1e-20;
    for (out[0..s.topk]) |*p| p.wt = (p.wt / den) * @as(f64, s.scale);
}

const t = std.testing;

test "tiny's 8 groups of 16 keeping 4: experts outside the kept groups never win" {
    const s = Shape{ .experts = 128, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 };
    var logits: [128]f32 = @splat(0);
    var bias: [128]f32 = @splat(0);
    // Groups 0, 2, 4, 7 score highest (two large experts each); 8 picks then span those groups.
    for ([_]u32{ 0, 2, 4, 7 }, 0..) |g, i| {
        logits[g * 16] = 8 - @as(f32, @floatFromInt(i));
        logits[g * 16 + 1] = 7 - @as(f32, @floatFromInt(i));
    }
    var out: [8]Pick = undefined;
    selectF32(&logits, &bias, &out, s);
    var seen: [8]bool = @splat(false);
    for (out) |p| {
        const g = p.id / 16;
        try t.expect(g == 0 or g == 2 or g == 4 or g == 7);
        seen[g] = true;
    }
    try t.expect(seen[0] and seen[2] and seen[4] and seen[7]);
}

test "ties on group scores keep the lower group" {
    const s = Shape{ .experts = 8, .topk = 2, .n_group = 4, .topk_group = 1, .scale = 1 };
    // Global top-2 sit in groups 0 and 1; keep 1 must still stay in group 0.
    const logits = [_]f32{ 3, 0, 3, 0, 0, 0, 0, 0 };
    var bias: [8]f32 = @splat(0);
    var out: [2]Pick = undefined;
    selectF32(&logits, &bias, &out, s);
    try t.expectEqual(@as(u32, 0), out[0].id / 2);
    try t.expectEqual(@as(u32, 0), out[1].id / 2);
}

test "ties on expert scores keep the lower id" {
    const s = Shape{ .experts = 8, .topk = 2, .n_group = 1, .topk_group = 1, .scale = 1 };
    var logits: [8]f32 = @splat(0.5);
    var bias: [8]f32 = @splat(0);
    var out: [2]Pick = undefined;
    selectF32(&logits, &bias, &out, s);
    try t.expectEqual(@as(u32, 0), out[0].id);
    try t.expectEqual(@as(u32, 1), out[1].id);
}

test "n_group 1 skips keep_groups: the global top-k win" {
    const s = Shape{ .experts = 8, .topk = 2, .n_group = 1, .topk_group = 1, .scale = 1 };
    const logits = [_]f32{ 0, 0, 0, 9, 0, 8, 0, 0 };
    var bias: [8]f32 = @splat(0);
    var out: [2]Pick = undefined;
    selectF32(&logits, &bias, &out, s);
    try t.expectEqual(@as(u32, 3), out[0].id);
    try t.expectEqual(@as(u32, 5), out[1].id);
}

test "flash's 8 groups of 64 (the whole-column branch) keep 4" {
    const s = Shape{ .experts = 512, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 };
    var logits: [512]f32 = @splat(0);
    var bias: [512]f32 = @splat(0);
    for ([_]u32{ 1, 3, 5, 7 }) |g| {
        for (0..64) |j| logits[g * 64 + j] = 5;
    }
    var out: [8]Pick = undefined;
    selectF32(&logits, &bias, &out, s);
    for (out) |p| {
        const g = p.id / 64;
        try t.expect(g == 1 or g == 3 or g == 5 or g == 7);
    }
}

test "weights are the sigmoid scores renormalized times scale" {
    const s = Shape{ .experts = 4, .topk = 2, .n_group = 1, .topk_group = 1, .scale = 2 };
    const logits = [_]f32{ 0, 1, 2, 3 };
    var bias: [4]f32 = @splat(0);
    var out: [2]Pick = undefined;
    selectF32(&logits, &bias, &out, s);
    try t.expectEqual(@as(u32, 3), out[0].id);
    try t.expectEqual(@as(u32, 2), out[1].id);
    const a = sigmoidF32(3);
    const b = sigmoidF32(2);
    const want0 = (a / (a + b + 1e-20)) * 2;
    const want1 = (b / (a + b + 1e-20)) * 2;
    try t.expect(@abs(out[0].wt - want0) < 1e-6);
    try t.expect(@abs(out[1].wt - want1) < 1e-6);
}

test "float64 reference keeps the same picks as the fp32 formula on tiny's shape" {
    const s = Shape{ .experts = 128, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 };
    var logits: [128]f32 = undefined;
    var bias: [128]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();
    for (&logits, &bias) |*l, *b| {
        l.* = rng.float(f32) * 4 - 2;
        b.* = rng.float(f32) - 0.5;
    }
    var a: [8]Pick = undefined;
    var b: [8]Pick64 = undefined;
    selectF32(&logits, &bias, &a, s);
    selectF64(&logits, &bias, &b, s);
    for (a, b) |p, q| {
        try t.expectEqual(p.id, q.id);
        try t.expect(@abs(@as(f64, p.wt) - q.wt) < 1e-6);
    }
}

test "nonzero bias: weights are sigmoid(logit), not choice, with a hand-computed pair" {
    const s = Shape{ .experts = 2, .topk = 2, .n_group = 1, .topk_group = 1, .scale = 1 };
    const logits = [_]f32{ 0, 1 };
    const bias = [_]f32{ 0.4, 0 };
    var out: [2]Pick = undefined;
    selectF32(&logits, &bias, &out, s);
    try t.expectEqual(@as(u32, 0), out[0].id);
    try t.expectEqual(@as(u32, 1), out[1].id);
    // sigmoid(0)=1/2, sigmoid(1)=1/(1+e^{-1}). Weights ignore bias.
    try t.expect(@abs(out[0].wt - 0.406154515) < 1e-6);
    try t.expect(@abs(out[1].wt - 0.593845485) < 1e-6);
}

test "group score is the top-2 sum: top-1 ranking disagrees" {
    const s = Shape{ .experts = 8, .topk = 2, .n_group = 4, .topk_group = 1, .scale = 1 };
    // Group 0: one large expert. Group 1: two medium. Top-1 keeps 0; top-2 sum keeps 1.
    const logits = [_]f32{ 4, -4, 1, 0.8, 0, 0, 0, 0 };
    var bias: [8]f32 = @splat(0);
    var out: [2]Pick = undefined;
    selectF32(&logits, &bias, &out, s);
    try t.expectEqual(@as(u32, 1), out[0].id / 2);
    try t.expectEqual(@as(u32, 1), out[1].id / 2);
}

test "a 5th group does not win a pick when keep is 4" {
    const s = Shape{ .experts = 16, .topk = 4, .n_group = 8, .topk_group = 4, .scale = 1 };
    var logits: [16]f32 = @splat(0);
    var bias: [16]f32 = @splat(0);
    for (0..4) |g| {
        logits[g * 2] = 2;
        logits[g * 2 + 1] = 2;
    }
    logits[8] = 8; // group 4: would be pick 0 if keep were 5
    var out: [4]Pick = undefined;
    selectF32(&logits, &bias, &out, s);
    for (out) |p| try t.expect(p.id != 8);
}

test "negative kept choices still beat masked experts" {
    const s = Shape{ .experts = 4, .topk = 2, .n_group = 2, .topk_group = 1, .scale = 1 };
    const logits = [_]f32{ 2, 2, 0, 0 };
    const bias = [_]f32{ -2, -2, -3, -3 };
    var out: [2]Pick = undefined;
    selectF32(&logits, &bias, &out, s);
    try t.expectEqual(@as(u32, 0), out[0].id);
    try t.expectEqual(@as(u32, 1), out[1].id);
}

fn expectDistinct(out: []const Pick) !void {
    for (0..out.len) |i| for (0..i) |j| try t.expect(out[i].id != out[j].id);
}

test "a NaN as the last expert of a winning group does not drop that group" {
    const s = Shape{ .experts = 128, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 };
    var logits: [128]f32 = @splat(0);
    var bias: [128]f32 = @splat(0);
    logits[0] = 2;
    logits[1] = 2;
    logits[15] = std.math.nan(f32);
    var out: [8]Pick = undefined;
    selectF32(&logits, &bias, &out, s);
    for (0..8) |k| try t.expectEqual(@as(u32, @intCast(k)), out[k].id);
}

test "ids in a row are distinct, including all-NaN and one finite per group" {
    const s = Shape{ .experts = 128, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 };
    var bias: [128]f32 = @splat(0);
    var out: [8]Pick = undefined;
    var all_nan: [128]f32 = @splat(std.math.nan(f32));
    selectF32(&all_nan, &bias, &out, s);
    try expectDistinct(&out);
    var one: [128]f32 = @splat(std.math.nan(f32));
    for (0..8) |g| one[g * 16 + 3] = 1;
    selectF32(&one, &bias, &out, s);
    try expectDistinct(&out);
    for (out[0..4]) |p| try t.expect(p.id % 16 == 3 and p.id / 16 < 4);
}

test "equal tiny scores renormalize to 0.3125; all -100 stays 0" {
    const s = Shape{ .experts = 128, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 };
    var bias: [128]f32 = @splat(0);
    var out: [8]Pick = undefined;
    var ten: [128]f32 = @splat(-10);
    selectF32(&ten, &bias, &out, s);
    for (out) |p| try t.expect(@abs(p.wt - 0.3125) < 1e-5);
    var dead: [128]f32 = @splat(-100);
    selectF32(&dead, &bias, &out, s);
    for (out) |p| try t.expect(@abs(p.wt) < 1e-6);
}
