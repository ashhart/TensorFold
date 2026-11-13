//! Request token IDs are decimal object keys; only explicitly capable engines accept a nonempty bias.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const errors = @import("errors.zig");

pub fn parse(cx: *errors.Cx, fields: json.Value, supported: bool, vocab: u32) errors.Refused![]const api.LogitBias {
    const v = fields.field("logit_bias") orelse return &.{};
    if (v != .object) return cx.refuse("logit_bias must be an object of token IDs and finite numbers, or null");
    if (v.object.count() == 0) return &.{};
    if (!supported) return cx.refuse("logit_bias is not supported by this model or backend");
    if (v.object.count() > 300) return cx.refuse("logit_bias must contain at most 300 entries");
    const out = try cx.a.alloc(api.LogitBias, v.object.count());
    var bans: usize = 0;
    for (v.object.keys(), v.object.values(), out) |key, value, *entry| {
        if (key.len == 0) return cx.refuse("logit_bias keys must be decimal token IDs in the vocabulary");
        for (key) |ch| if (!std.ascii.isDigit(ch))
            return cx.refuse("logit_bias keys must be decimal token IDs in the vocabulary");
        const token = std.fmt.parseInt(u32, key, 10) catch
            return cx.refuse("logit_bias keys must be decimal token IDs in the vocabulary");
        if (token >= vocab) return cx.refuse("logit_bias token ID is outside the vocabulary");
        if (value != .int and value != .float) return cx.refuse("logit_bias values must be finite numbers between -100 and 100");
        const amount = value.float64() orelse return cx.refuse("logit_bias values must be finite numbers between -100 and 100");
        if (!std.math.isFinite(amount) or amount < -100 or amount > 100)
            return cx.refuse("logit_bias values must be finite numbers between -100 and 100");
        entry.* = .{ .token = token, .value = @floatCast(amount), .ban = amount == -100 };
        if (amount == -100) bans += 1;
    }
    std.mem.sort(api.LogitBias, out, {}, struct {
        fn less(_: void, a: api.LogitBias, b: api.LogitBias) bool {
            return a.token < b.token;
        }
    }.less);
    for (out[1..], out[0 .. out.len - 1]) |a, b| if (a.token == b.token)
        return cx.refuse("logit_bias contains duplicate token IDs");
    if (bans == vocab) return cx.refuse("logit_bias cannot ban every token in the vocabulary");
    return out;
}
