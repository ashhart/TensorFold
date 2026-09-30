const std = @import("std");
const Sampling = @import("sampling.zig").Sampling;

pub const Options = struct {
    sampling: Sampling = .{ .temperature = 0, .top_k = 0, .top_p = 1, .metal = true },
    seed: ?u64 = null,
    max_tokens: usize = 512,
    ignore_eos: bool = false,
    stream: bool = false,
    stops: []const []const u8 = &.{},
    thinking_budget: i64 = 0,
    draft: bool = true,

    pub fn load(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Options {
        const path = try std.fs.path.join(a, &.{ dir, "generation_config.json" });
        defer a.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1024 * 1024)) catch |err| switch (err) {
            error.FileNotFound => return .{},
            else => return err,
        };
        defer a.free(bytes);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
        defer parsed.deinit();
        return fromGenerationConfig(parsed.value);
    }

    pub fn fromGenerationConfig(body: std.json.Value) !Options {
        if (body != .object) return error.InvalidGenerationConfig;
        var out = Options{};
        try out.readSampling(body);
        if (body.object.get("do_sample")) |v| if (v == .bool) {
            if (!v.bool) out.sampling.temperature = 0 else if (!body.object.contains("temperature")) out.sampling.temperature = 1;
        };
        try out.sampling.validate();
        return out;
    }

    fn readSampling(out: *Options, body: std.json.Value) !void {
        if (try number(body, "temperature")) |v| out.sampling.temperature = @max(0, v);
        if (try number(body, "top_p")) |v| out.sampling.top_p = v;
        if (try number(body, "min_p")) |v| out.sampling.min_p = v;
        if (try integer(body, "top_k")) |v| out.sampling.top_k = @intCast(@max(0, v));
    }

    pub fn parse(a: std.mem.Allocator, body: std.json.Value) !Options {
        return parseWithDefaults(a, body, .{});
    }

    pub fn parseWithDefaults(a: std.mem.Allocator, body: std.json.Value, defaults: Options) !Options {
        if (body != .object) return error.InvalidRequest;
        var out = defaults;
        if (body.object.get("draft")) |value| out.draft = value != .bool or value.bool;
        out.stops = &.{};
        inline for (.{ "ignore_eos", "stream" }) |name| if (body.object.get(name)) |v| {
            if (v != .bool) return error.InvalidBoolean;
            @field(out, name) = v.bool;
        };
        try out.readSampling(body);
        if (try integer(body, "thinking_budget")) |v| if (v != 0) {
            out.thinking_budget = v;
        };
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

test "min-p preserves model defaults, explicit zero and upstream range checks" {
    const a = std.testing.allocator;
    const config = try std.json.parseFromSlice(std.json.Value, a, "{\"min_p\":0.25}", .{});
    defer config.deinit();
    const defaults = try Options.fromGenerationConfig(config.value);
    for ([_][]const u8{ "{}", "{\"min_p\":null}", "{\"min_p\":0}", "{\"min_p\":\"1\"}" }, [_]f64{ 0.25, 0.25, 0, 1 }) |source, expected| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, source, .{});
        defer parsed.deinit();
        const result = try Options.parseWithDefaults(a, parsed.value, defaults);
        defer a.free(result.stops);
        try std.testing.expectEqual(expected, result.sampling.min_p);
    }
    for ([_][]const u8{ "{\"min_p\":-0.01}", "{\"min_p\":1.01}", "{\"min_p\":\"NaN\"}", "{\"min_p\":true}" }, [_]anyerror{ error.InvalidSampling, error.InvalidSampling, error.InvalidNumber, error.InvalidNumber }) |source, expected| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, source, .{});
        defer parsed.deinit();
        try std.testing.expectError(expected, Options.parseWithDefaults(a, parsed.value, defaults));
    }
}

test "only explicit false disables request drafting" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "{}", "{\"draft\":true}", "{\"draft\":null}", "{\"draft\":0}", "{\"draft\":false}" }, 0..) |source, index| {
        const request = try std.json.parseFromSlice(std.json.Value, a, source, .{});
        defer request.deinit();
        const options = try Options.parse(a, request.value);
        defer a.free(options.stops);
        try std.testing.expectEqual(index != 4, options.draft);
    }
}

test "model sampling defaults and request overrides preserve null and explicit zero" {
    const a = std.testing.allocator;
    const model = try std.json.parseFromSlice(std.json.Value, a,
        \\{"do_sample":true,"top_k":20,"top_p":0.95,"max_tokens":999,"seed":123}
    , .{});
    defer model.deinit();
    var defaults = try Options.fromGenerationConfig(model.value);
    try std.testing.expectEqual(@as(f64, 1), defaults.sampling.temperature);
    try std.testing.expectEqual(@as(usize, 512), defaults.max_tokens);
    try std.testing.expectEqual(null, defaults.seed);
    defaults.thinking_budget = 12;
    for ([_][]const u8{ "{}", "{\"temperature\":null,\"thinking_budget\":0}", "{\"temperature\":0,\"top_k\":0,\"top_p\":1,\"thinking_budget\":-1}" }, 0..) |source, index| {
        const request = try std.json.parseFromSlice(std.json.Value, a, source, .{});
        defer request.deinit();
        const options = try Options.parseWithDefaults(a, request.value, defaults);
        defer a.free(options.stops);
        try std.testing.expectEqual(@as(f64, if (index == 2) 0 else 1), options.sampling.temperature);
        try std.testing.expectEqual(@as(usize, if (index == 2) 0 else 20), options.sampling.top_k);
        try std.testing.expectEqual(@as(f64, if (index == 2) 1 else 0.95), options.sampling.top_p);
        try std.testing.expectEqual(@as(i64, if (index == 2) -1 else 12), options.thinking_budget);
    }
    const greedy = try std.json.parseFromSlice(std.json.Value, a, "{\"do_sample\":false,\"temperature\":0.8}", .{});
    defer greedy.deinit();
    try std.testing.expectEqual(@as(f64, 0), (try Options.fromGenerationConfig(greedy.value)).sampling.temperature);
    const bad = try std.json.parseFromSlice(std.json.Value, a, "{\"thinking_budget\":1.5}", .{});
    defer bad.deinit();
    try std.testing.expectError(error.InvalidInteger, Options.parseWithDefaults(a, bad.value, defaults));
}
