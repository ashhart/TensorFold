//! ``kept_probs``: each reply token's probabilities over the sampler's kept set, from the reply's logprob rows.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const lanes = @import("lanes");
const sampling = lanes.sampling;
const max_top = lanes.logprob.max_top;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;

/// How a reply's rows print: `shown` best tokens of each (the client's count), and the kept set when asked.
pub const Opts = struct {
    shown: u8 = max_top,
    kept: bool = false,
    sampling: ?api.Sampling = null, // the request's; null: greedy
};

/// One kept token and its probability after temperature, top_k, top_p and min_p.
pub const Kept = struct { id: u32, prob: f64 };

/// The ``kept_probs`` field: a boolean or null; true needs logprobs asked (chat's ``logprobs``, completions' count).
pub fn asked(cx: *Cx, body: json.Value, logprobs: ?u8) errors.Refused!bool {
    const v = body.field("kept_probs") orelse return false;
    if (v != .bool) return cx.refuse("kept_probs must be a boolean or null");
    if (v.bool and logprobs == null) return cx.refuse("kept_probs requires logprobs: true on chat, or a logprobs count on /v1/completions");
    return v.bool;
}

/// The rows a kept_probs request asks the engine for (all 20), or its refusal when the rows can't hold the kept set.
pub fn admit(cx: *Cx, request: *const api.Request) errors.Refused!u8 {
    if (request.penalties.active()) return cx.refuse("kept_probs can't be used with presence_penalty or frequency_penalty: send both as 0");
    if (request.structure != null) return cx.refuse("kept_probs can't be used with a grammar: the rows are the model's distribution before the grammar's mask");
    if (request.sampling) |s| if (s.temperature > 0 and (s.top_k == 0 or s.top_k > max_top))
        return cx.fail(.request, "kept_probs needs top_k from 1 to {d} so the kept set fits the {d} alternatives each row carries; this request's top_k is {d}{s}", .{ max_top, max_top, s.top_k, if (s.top_k == 0) " (no limit)" else "" });
    return max_top;
}

/// The print options for a prepared request: the client's own count shown, and the kept set when asked.
pub fn opts(input_logprobs: ?u8, kept: bool, request: *const api.Request) Opts {
    return .{ .shown = input_logprobs orelse 0, .kept = kept, .sampling = request.sampling };
}

/// `r`'s kept set in its order: greedy keeps the best token; sampled keeps what ``sampling.kept`` keeps of top_k.
pub fn of(a: Allocator, r: api.LogprobRow, s_: ?api.Sampling) error{ Failed, OutOfMemory }![]Kept {
    if (r.count == 0) return error.Failed;
    const s = s_ orelse return one(a, r);
    if (!(s.temperature > 0)) return one(a, r);
    if (s.top_k == 0 or s.top_k > r.count) return error.Failed; // a row without the whole top_k can't show the set
    const k: usize = s.top_k;
    const t = @max(s.temperature, 1e-6);
    const scaled = try a.alloc(f64, k);
    for (scaled, r.logprobs[0..k]) |*x, lp| x.* = @as(f64, lp) / t;
    const keep = try sampling.kept(a, scaled, s);
    const weights = try a.alloc(f64, keep);
    for (weights, scaled[0..keep]) |*w, x| w.* = @exp(x - scaled[0]);
    const total = sampling.pairwiseSum(weights);
    const out = try a.alloc(Kept, keep);
    for (out, r.ids[0..keep], weights) |*o, id, w| o.* = .{ .id = id, .prob = w / total };
    return out;
}

fn one(a: Allocator, r: api.LogprobRow) Allocator.Error![]Kept {
    const out = try a.alloc(Kept, 1);
    out[0] = .{ .id = r.ids[0], .prob = 1 };
    return out;
}

/// ``[{token, prob}]`` for `r`'s kept set, each token spelled as ``logprobs``' entries spell it.
pub fn value(srv: *Server, a: Allocator, r: api.LogprobRow, s: ?api.Sampling) error{ Failed, OutOfMemory }!json.Value {
    const set = try of(a, r, s);
    const items = try a.alloc(json.Value, set.len);
    for (items, set) |*v, kp| {
        const o = try json.newObject(a);
        try o.put(a, "token", .{ .string = try srv.text.decode(a, &.{kp.id}) });
        try o.put(a, "prob", .{ .float = kp.prob });
        v.* = .{ .object = o };
    }
    return .{ .array = items };
}

/// Completions' legacy form when asked: each position's kept set as chat gives it, an empty list for no tokens.
pub fn legacy(a: Allocator, items: []const json.Value, kept: bool) Allocator.Error!?json.Value {
    if (!kept) return null;
    const out = try a.alloc(json.Value, items.len);
    for (out, items) |*o, r| o.* = r.object.get("kept_probs") orelse .null;
    return .{ .array = out };
}

fn rowOf(lps: []const f32) api.LogprobRow {
    var r: api.LogprobRow = .{ .token = 100, .logprob = lps[0], .count = @intCast(lps.len) };
    for (lps, 0..) |lp, i| {
        r.ids[i] = @intCast(100 + i);
        r.logprobs[i] = lp;
    }
    return r;
}

test "the kept set by hand: top_k, top_p's cumulative cut, min_p's floor, then renormalized at the temperature" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = rowOf(&.{ @log(0.5), @log(0.3), @log(0.15), @log(0.05) });
    const cases = [_]struct { s: api.Sampling, want: []const f64 }{
        // cumulative 0.5, 0.8, 0.95: two below 0.9, so three kept
        .{ .s = .{ .seed = 1, .temperature = 1, .top_k = 4, .top_p = 0.9 }, .want = &.{ 0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95 } },
        // min_p 0.4: the floor is 0.2 of the best, so 0.15 goes
        .{ .s = .{ .seed = 1, .temperature = 1, .top_k = 4, .top_p = 0.9, .min_p = 0.4 }, .want = &.{ 0.625, 0.375 } },
        // T 2 takes square roots; top_k 2 cuts first
        .{ .s = .{ .seed = 1, .temperature = 2, .top_k = 2, .top_p = 1 }, .want = &.{ @sqrt(0.5) / (@sqrt(0.5) + @sqrt(0.3)), @sqrt(0.3) / (@sqrt(0.5) + @sqrt(0.3)) } },
    };
    for (cases) |c| {
        const set = try of(a, r, c.s);
        try std.testing.expectEqual(c.want.len, set.len);
        for (set, c.want, 0..) |kp, p, i| {
            try std.testing.expectEqual(@as(u32, @intCast(100 + i)), kp.id);
            try std.testing.expectApproxEqAbs(p, kp.prob, 1e-6);
        }
    }
    // greedy, by no sampling or by temperature 0: the best token, certain
    for ([_]?api.Sampling{ null, .{ .seed = 1, .temperature = 0, .top_k = 0 } }) |s| {
        const set = try of(a, r, s);
        try std.testing.expectEqual(@as(usize, 1), set.len);
        try std.testing.expectEqual(@as(u32, 100), set[0].id);
        try std.testing.expectEqual(@as(f64, 1), set[0].prob);
    }
    // a top_k past the row's alternatives, or none, can't be shown whole
    for ([_]u32{ 0, 5 }) |k| try std.testing.expectError(error.Failed, of(a, r, .{ .seed = 1, .top_k = k }));
}

test "every token the sampler draws is in the kept set, each at about its reported probability" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var lps: [max_top]f32 = undefined;
    for (&lps, 0..) |*lp, i| lp.* = -0.25 * @as(f32, @floatFromInt(i)) - 0.03 * @as(f32, @floatFromInt(i * i));
    const r = rowOf(&lps);
    var order: [max_top]sampling.Candidate = undefined;
    for (&order, lps, 0..) |*o, lp, i| o.* = .{ .value = lp, .id = 100 + i };
    const configs = [_]api.Sampling{
        .{ .seed = 11, .temperature = 1, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 12, .temperature = 0.6, .top_k = 8, .top_p = 0.9, .min_p = 0.05 },
        .{ .seed = 13, .temperature = 1.5, .top_k = 20, .top_p = 1, .min_p = 0.1 },
        .{ .seed = 14, .temperature = 1, .top_k = 3, .top_p = 0 },
    };
    const n = 20000;
    for (configs) |s| {
        const set = try of(a, r, s);
        var sum: f64 = 0;
        for (set) |kp| sum += kp.prob;
        try std.testing.expectApproxEqAbs(@as(f64, 1), sum, 1e-12);
        const counts = try a.alloc(u32, set.len);
        @memset(counts, 0);
        for (0..n) |position| {
            const id = try sampling.chooseOrdered(std.testing.allocator, &order, position, s);
            const at = for (set, 0..) |kp, j| {
                if (kp.id == id) break j;
            } else return error.DrawOutsideKeptSet;
            counts[at] += 1;
        }
        for (set, counts) |kp, c| {
            const seen = @as(f64, @floatFromInt(c)) / n;
            try std.testing.expect(@abs(seen - kp.prob) <= 4 * @sqrt(kp.prob * (1 - kp.prob) / n) + 1e-3);
        }
    }
}

fn markAtOrBefore(k: u32, i: u32, m: lanes.gpu_full.Mark) bool {
    return k > m.key or (k == m.key and i <= m.id);
}

test "from a row an engine reads off its logits, the kept set is what the host sampler and both GPU rules keep" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const gpa = std.testing.allocator;
    // 64 logits falling with the id, ids 5 and 6 tied: the rows' order and every sampler's, value then lower id
    var logits: [64]f32 = undefined;
    for (&logits, 0..) |*l, i| l.* = 2 - 0.3 * @as(f32, @floatFromInt(i)) - 0.02 * @as(f32, @floatFromInt(i * i));
    logits[6] = logits[5];
    var m: f64 = -std.math.inf(f64);
    for (logits) |l| m = @max(m, l);
    var z: f64 = 0;
    for (logits) |l| z += @exp(@as(f64, l) - m);
    const lse: f32 = @floatCast(m + @log(z));
    var r: api.LogprobRow = .{ .token = 0, .logprob = logits[0] - lse, .count = max_top };
    for (0..max_top) |i| {
        r.ids[i] = @intCast(i);
        r.logprobs[i] = logits[i] - lse;
    }
    var values: [64]f64 = undefined;
    var ids: [64]u64 = undefined;
    for (&values, &ids, logits, 0..) |*v, *id, l, i| {
        v.* = l;
        id.* = i;
    }
    const configs = [_]api.Sampling{
        .{ .seed = 21, .temperature = 1, .top_k = 20, .top_p = 0.9 },
        .{ .seed = 22, .temperature = 0.7, .top_k = 6, .top_p = 1 }, // top_k cuts between the tied 5 and 6
        .{ .seed = 23, .temperature = 1.3, .top_k = 12, .top_p = 0.95, .min_p = 0.08 },
        .{ .seed = 24, .temperature = 0.5, .top_k = 20, .top_p = 0.8, .min_p = 0.02 },
    };
    for (configs) |s| {
        const set = try of(a, r, s);
        if (s.top_k == 6) try std.testing.expectEqual(@as(u32, 5), set[set.len - 1].id);
        // tf_sample_full's plan (the Metal hosts' draw) keeps exactly as many tokens
        const p = try lanes.gpu_full.plan(gpa, &logits, s);
        var planned: usize = 0;
        for (logits, 0..) |l, i| {
            const v = l * p.inv_t;
            const k = lanes.gpu_rule.key(v);
            if (v >= p.lo and markAtOrBefore(k, @intCast(i), p.top) and markAtOrBefore(k, @intCast(i), p.cut)) planned += 1;
        }
        try std.testing.expectEqual(set.len, planned);
        for (0..3000) |position| {
            const drawn = [_]u64{
                try sampling.choose(gpa, &values, &ids, position, s),
                try lanes.gpu_rule.sample(gpa, &logits, s, @intCast(position), null),
                try lanes.gpu_full.sample(gpa, &logits, s, @intCast(position), null),
            };
            for (drawn) |id| {
                for (set) |kp| {
                    if (kp.id == id) break;
                } else return error.DrawOutsideKeptSet;
            }
        }
    }
}

test "admit: 20 rows asked; refused for no top_k or one over 20 when sampling, and for penalties" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    var req: api.Request = .{ .prompt = &.{1}, .max_tokens = 1 };
    try std.testing.expectEqual(@as(u8, 20), try admit(&cx, &req)); // greedy: any top_k
    req.sampling = .{ .seed = 1, .top_k = 20 };
    try std.testing.expectEqual(@as(u8, 20), try admit(&cx, &req));
    for ([_]u32{ 0, 21, 50 }) |k| {
        req.sampling.?.top_k = k;
        try std.testing.expectError(error.Refused, admit(&cx, &req));
    }
    req.sampling.?.top_k = 5;
    req.penalties = .{ .presence = 1.5 };
    try std.testing.expectError(error.Refused, admit(&cx, &req));
}
