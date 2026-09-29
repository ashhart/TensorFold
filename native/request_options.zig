const std = @import("std");
const Sampling = @import("sampling.zig").Sampling;

pub const Options = struct {
    sampling: Sampling = .{ .temperature = 0, .top_k = 0, .top_p = 1, .metal = true },
    seed: ?u64 = null,
    max_tokens: usize = 512,
    ignore_eos: bool = false,
    stream: bool = false,
    stops: []const []const u8 = &.{},

    pub fn parse(a: std.mem.Allocator, body: std.json.Value) !Options {
        if (body != .object) return error.InvalidRequest;
        var out = Options{};
        inline for (.{ "ignore_eos", "stream" }) |name| if (body.object.get(name)) |v| {
            if (v != .bool) return error.InvalidBoolean;
            @field(out, name) = v.bool;
        };
        if (try number(body, "temperature")) |v| out.sampling.temperature = @max(0, v);
        if (try number(body, "top_p")) |v| out.sampling.top_p = v;
        if (try integer(body, "top_k")) |v| out.sampling.top_k = @intCast(@max(0, v));
        if (try integer(body, "seed")) |v| out.seed = @bitCast(v);
        if ((try integer(body, "max_tokens")) orelse (try integer(body, "max_completion_tokens"))) |v| {
            if (v < 0 or v > 262144) return error.InvalidTokenLimit;
            out.max_tokens = @intCast(v);
        }
        if (body.object.get("stop")) |v| switch (v) {
            .null => {},
            .string => |s| {
                if (s.len == 0) return error.InvalidStop;
                const list = try a.alloc([]const u8, 1);
                list[0] = s;
                out.stops = list;
            },
            .array => |values| {
                for (values.items) |item| if (item != .string or item.string.len == 0) return error.InvalidStop;
                const list = try a.alloc([]const u8, values.items.len);
                for (list, values.items) |*item, value| item.* = value.string;
                out.stops = list;
            },
            else => return error.InvalidStop,
        };
        errdefer a.free(out.stops);
        try out.sampling.validate();
        return out;
    }
};

fn number(body: std.json.Value, name: []const u8) !?f64 {
    const v = body.object.get(name) orelse return null;
    const result: f64 = switch (v) {
        .null => return null,
        .float => v.float,
        .integer => @floatFromInt(v.integer),
        .string => std.fmt.parseFloat(f64, std.mem.trim(u8, v.string, " \r\n\t")) catch return error.InvalidNumber,
        else => return error.InvalidNumber,
    };
    if (!std.math.isFinite(result)) return error.InvalidNumber;
    return result;
}
fn integer(body: std.json.Value, name: []const u8) !?i64 {
    const v = body.object.get(name) orelse return null;
    return switch (v) {
        .null => null,
        .integer => v.integer,
        .float => if (std.math.isFinite(v.float) and @trunc(v.float) == v.float and v.float >= -0x1p63 and v.float < 0x1p63) @as(i64, @intFromFloat(v.float)) else error.InvalidInteger,
        .string => std.fmt.parseInt(i64, std.mem.trim(u8, v.string, " \r\n\t"), 10) catch error.InvalidInteger,
        else => error.InvalidInteger,
    };
}

test "request defaults, explicit zero, numeric strings and stop validation" {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a,
        \\{"temperature":null,"top_k":-9,"seed":"456","max_tokens":0,"max_completion_tokens":20,"stop":["END","æøå"],"ignore_eos":true}
    , .{});
    defer parsed.deinit();
    const options = try Options.parse(a, parsed.value);
    defer a.free(options.stops);
    try std.testing.expectEqual(@as(?u64, 456), options.seed);
    try std.testing.expectEqual(@as(usize, 0), options.max_tokens);
    try std.testing.expectEqual(@as(usize, 0), options.sampling.top_k);
    try std.testing.expect(options.ignore_eos);
    for ([_][]const u8{ "{\"temperature\":\"NaN\"}", "{\"seed\":true}", "{\"max_tokens\":2.5}", "{\"stop\":[\"\"]}", "{\"ignore_eos\":null}" }, [_]anyerror{ error.InvalidNumber, error.InvalidInteger, error.InvalidInteger, error.InvalidStop, error.InvalidBoolean }) |json, expected| {
        const bad = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer bad.deinit();
        try std.testing.expectError(expected, Options.parse(a, bad.value));
    }
}
