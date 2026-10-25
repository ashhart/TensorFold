//! Request-local Drama sampling, checked before a speech request reaches the engine.
const std = @import("std");
const api = @import("engine_api");
const Value = @import("json").Value;
const errors = @import("errors.zig");
const defaults = @import("drama_prompt.zig");

pub const Options = struct {
    temperature: f32 = defaults.temperature,
    top_p: f32 = defaults.top_p,
    top_k: u32 = defaults.top_k,
    seed: ?u64 = null,

    pub fn seedFor(o: Options, prompt: []const u32, salt: i64, attempt: usize) u64 {
        if (o.seed) |seed| return seed +% @as(u64, @intCast(attempt));
        return api.seedFor(prompt, salt +% @as(i64, @intCast(attempt)));
    }
};

pub fn isField(key: []const u8) bool {
    for ([_][]const u8{ "temperature", "top_p", "top_k", "seed" }) |name|
        if (std.mem.eql(u8, key, name)) return true;
    return false;
}

pub fn parse(cx: *errors.Cx, body: Value) errors.Refused!Options {
    var out: Options = .{};
    if (body.field("temperature")) |value|
        out.temperature = try number(cx, value, std.math.floatMax(f32), "temperature must be a finite nonnegative f32 number");
    if (body.field("top_p")) |value| {
        out.top_p = try number(cx, value, 1, "top_p must be a finite f32 number greater than 0 and at most 1");
        if (out.top_p == 0) return cx.refuse("top_p must be greater than 0");
    }
    if (body.field("top_k")) |value| {
        const message = "top_k must be an integer from 1 through 64";
        if (value != .int) return cx.refuse(message);
        const n = std.fmt.parseInt(u32, value.int, 10) catch return cx.refuse(message);
        if (n == 0 or n > 64) return cx.refuse(message);
        out.top_k = n;
    }
    if (body.field("seed")) |value| {
        const message = "seed must be an integer from 0 through 18446744073709551615";
        if (value != .int) return cx.refuse(message);
        out.seed = std.fmt.parseInt(u64, value.int, 10) catch return cx.refuse(message);
    }
    return out;
}

/// Check the parsed f64 range before narrowing, then reject positive values that would become zero.
fn number(cx: *errors.Cx, value: Value, max: f64, message: []const u8) errors.Refused!f32 {
    if (value != .int and value != .float) return cx.refuse(message);
    const n = value.float64() orelse return cx.refuse(message);
    if (!std.math.isFinite(n) or n < 0 or n > max) return cx.refuse(message);
    const narrowed: f32 = @floatCast(n);
    if (n > 0 and narrowed == 0) return cx.refuse(message);
    return narrowed;
}
