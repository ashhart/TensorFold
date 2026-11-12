//! System One answers from Clef's logits (`systemone_answer`): an FP32 softmax per question, Python's round(x, 4).
const std = @import("std");
const json = @import("json");
const enc = @import("clef_encode.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const Error = error{ OutOfMemory, NonFinite };

/// `logits.float().softmax(-1)`: BF16 bits widened to FP32, exp(x - max) / sum in FP32; a non-finite logit refused.
pub fn probabilities(logits: []const u16, out: []f32) error{NonFinite}!void {
    var m = -std.math.inf(f32);
    for (logits, out) |b, *o| {
        o.* = @bitCast(@as(u32, b) << 16);
        if (!std.math.isFinite(o.*)) return error.NonFinite;
        m = @max(m, o.*);
    }
    var s: f32 = 0;
    for (out) |*o| {
        o.* = @exp(o.* - m);
        s += o.*;
    }
    for (out) |*o| o.* /= s;
}

/// Python's `round(x, 4)`: the double's exact binary value scaled by 10,000 in u128, rounded half to even, divided.
pub fn round4(x: f64) f64 {
    if (!std.math.isFinite(x) or x == 0) return x;
    const bits: u64 = @bitCast(@abs(x));
    const biased: i32 = @intCast(bits >> 52);
    const e: i32 = (if (biased == 0) 1 else biased) - 1075;
    // an exponent of 0 or more is an integer already; exact while r stays below 2^53 (|x| under 9e11)
    if (e >= 0) return x;
    const m: u128 = (bits & ((@as(u64, 1) << 52) - 1)) | (if (biased == 0) 0 else @as(u64, 1) << 52);
    const y = m * 10_000; // under 2^67
    var r: u128 = 0;
    if (e > -128) {
        const s: u7 = @intCast(-e);
        const q = y >> s;
        const rem = y & ((@as(u128, 1) << s) - 1);
        const half = @as(u128, 1) << (s - 1);
        r = q + @intFromBool(rem > half or (rem == half and q & 1 == 1));
    }
    // r and 10,000 are exact doubles, so the one IEEE division gives the double nearest r / 10^4, as Python's does
    return std.math.copysign(@as(f64, @floatFromInt(r)) / 10_000.0, x);
}

fn prob(q: enc.Question, probs: []const f32, id: []const u8) f64 {
    for (q.option_ids, probs) |o, p| if (std.mem.eql(u8, o, id)) return p;
    unreachable; // every request-order id is one of the question's options
}

/// One question's answer: noul's P(true); choice's first best option in request order; score's expected level.
pub fn answer(a: Allocator, f: enc.Field, q: enc.Question, probs: []const f32) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = @tagName(f.kind()) });
    switch (f.criteria) {
        .noul => try o.put(a, "noul", .{ .float = round4(prob(q, probs, "true")) }),
        .choice => |crit| {
            var best = crit.keys()[0];
            for (crit.keys()) |k| if (prob(q, probs, k) > prob(q, probs, best)) {
                best = k;
            };
            try o.put(a, "choice", .{ .string = best });
            try o.put(a, "confidence", .{ .float = round4(prob(q, probs, best)) });
            const p = try json.newObject(a);
            for (crit.keys()) |k| try p.put(a, k, .{ .float = round4(prob(q, probs, k)) });
            try o.put(a, "probabilities", .{ .object = p });
        },
        .score => |levels| {
            // levels are the encoding order "0" to "n-1", so probs[i] is level i's
            var expected: f64 = 0;
            var top: f64 = probs[0];
            const legend = try json.newObject(a);
            const p = try json.newObject(a);
            for (levels, probs, 0..) |level, v, i| {
                const name = try std.fmt.allocPrint(a, "{d}", .{i});
                expected += @as(f64, @floatFromInt(i)) * v;
                top = @max(top, v);
                try legend.put(a, name, level);
                try p.put(a, name, .{ .float = round4(v) });
            }
            try o.put(a, "score", .{ .float = round4(expected) });
            try o.put(a, "confidence", .{ .float = round4(top) });
            try o.put(a, "legend", .{ .object = legend });
            try o.put(a, "probabilities", .{ .object = p });
        },
    }
    return .{ .object = o };
}

/// `"raw": true` (an extension kept from #1361): each option's unrounded probability, in encoding order.
pub fn raw(a: Allocator, q: enc.Question, probs: []const f32) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = @tagName(q.kind) });
    const names = try a.alloc(Value, q.option_ids.len);
    for (names, q.option_ids) |*n, id| n.* = .{ .string = id };
    try o.put(a, "options", .{ .array = names });
    const ps = try a.alloc(Value, probs.len);
    for (ps, probs) |*v, p| v.* = .{ .float = p };
    try o.put(a, "probs", .{ .array = ps });
    return .{ .object = o };
}

/// The response body: `model` echoed, answers in request order, usage; `logits` holds every option in encoding order.
pub fn body(a: Allocator, model: Value, fields: []const enc.Field, rec: enc.Record, logits: []const u16, with_raw: bool) Error!Value {
    const answers = try json.newObject(a);
    const raws = try json.newObject(a);
    var at: usize = 0;
    for (fields, rec.questions) |f, q| {
        const n = q.option_ids.len;
        const probs = try a.alloc(f32, n);
        try probabilities(logits[at..][0..n], probs);
        at += n;
        try answers.put(a, f.id, try answer(a, f, q, probs));
        if (with_raw) try raws.put(a, f.id, try raw(a, q, probs));
    }
    std.debug.assert(at == logits.len);
    const usage = try json.newObject(a);
    try usage.put(a, "input_tokens", try json.intValue(a, rec.ids.len));
    try usage.put(a, "output_tokens", .{ .int = "0" });
    const out = try json.newObject(a);
    try out.put(a, "model", model);
    try out.put(a, "answers", .{ .object = answers });
    try out.put(a, "usage", .{ .object = usage });
    if (with_raw) try out.put(a, "raw", .{ .object = raws });
    return .{ .object = out };
}
