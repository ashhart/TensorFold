//! Pins for the limits docs/api.md and the README state: one accepted bound and one refused neighbour each.
const std = @import("std");
const json = @import("json");
const cli = @import("cli.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const responses_translate = @import("responses_translate.zig");

fn fill(comptime n: usize, comptime char: u8) [n]u8 {
    var buf: [n]u8 = undefined;
    for (&buf) |*c| c.* = char;
    return buf;
}

const NoHistory = struct {
    pub fn conversation(_: NoHistory, cx: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return cx.refuse("no stored responses in this test");
    }
};

fn body(a: std.mem.Allocator, text: []const u8) !json.Value {
    return (try json.parse(a, text)).ok;
}

fn metadataRequest(a: std.mem.Allocator, pairs: usize, key: []const u8, value: []const u8) !json.Value {
    const req = try json.newObject(a);
    try req.put(a, "model", .{ .string = "m" });
    try req.put(a, "input", .{ .string = "hi" });
    var meta = try json.newObject(a);
    var i: usize = 0;
    while (i < pairs) : (i += 1) try meta.put(a, try std.fmt.allocPrint(a, "{s}{d}", .{ key, i }), .{ .string = value });
    try req.put(a, "metadata", .{ .object = meta });
    return .{ .object = req };
}

fn appendValues(a: std.mem.Allocator, flag: []const u8, count: usize, scheme: []const u8) ![][]const u8 {
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.append(a, "m");
    var i: usize = 0;
    while (i < count) : (i += 1) try argv.append(a, try std.fmt.allocPrint(a, "{s}={s}h{d}.example.com", .{ flag, scheme, i }));
    return argv.items;
}

test "limit pins: sampling numbers at their documented bounds" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    for ([_][]const u8{
        "{\"min_p\":0}",
        "{\"min_p\":1}",
        "{\"frequency_penalty\":-2}",
        "{\"frequency_penalty\":2}",
        "{\"presence_penalty\":-2}",
        "{\"presence_penalty\":2}",
    }) |text| _ = try fields.parseNumbers(&cx, try body(a, text));
    const stored = try fields.parseNumbers(&cx, try body(a, "{\"min_p\":0.25,\"frequency_penalty\":-1.5,\"presence_penalty\":2}"));
    try std.testing.expectEqual(@as(f64, 0.25), stored.object.get("min_p").?.float);
    try std.testing.expectEqual(@as(f64, -1.5), stored.object.get("frequency_penalty").?.float);
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "{\"min_p\":-0.5}", "min_p must be between 0 and 1, or null" },
        .{ "{\"min_p\":1.5}", "min_p must be between 0 and 1, or null" },
        .{ "{\"frequency_penalty\":-2.5}", "frequency_penalty must be between -2 and 2, or null" },
        .{ "{\"frequency_penalty\":2.5}", "frequency_penalty must be between -2 and 2, or null" },
        .{ "{\"presence_penalty\":2.5}", "presence_penalty must be between -2 and 2, or null" },
    };
    for (cases) |case| {
        try std.testing.expectError(error.Refused, fields.parseNumbers(&cx, try body(a, case[0])));
        try std.testing.expectEqualStrings(case[1], cx.message);
    }
}

test "limit pins: n takes one through sixteen choices" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    for ([_][]const u8{ "{}", "{\"n\":null}", "{\"n\":1}" }) |text| try std.testing.expectEqual(@as(u32, 1), try fields.choiceCount(&cx, try body(a, text)));
    try std.testing.expectEqual(@as(u32, 3), try fields.choiceCount(&cx, try body(a, "{\"n\":3}")));
    try std.testing.expectEqual(@as(u32, 16), try fields.choiceCount(&cx, try body(a, "{\"n\":16}")));
    for ([_][]const u8{ "{\"n\":0}", "{\"n\":17}", "{\"n\":-1}", "{\"n\":\"2\"}", "{\"n\":2.0}", "{\"n\":true}" }) |text| {
        try std.testing.expectError(error.Refused, fields.choiceCount(&cx, try body(a, text)));
        try std.testing.expectEqualStrings("n must be an integer between 1 and 16", cx.message);
    }
}

test "limit pins: Responses and the one-reply routes keep n at one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    for ([_][]const u8{ "{\"input\":\"hi\"}", "{\"input\":\"hi\",\"n\":null}", "{\"input\":\"hi\",\"n\":1}" }) |text|
        _ = try responses_translate.translate(&cx, try body(a, text), NoHistory{});
    for ([_][]const u8{ "{\"n\":2}", "{\"n\":16}", "{\"n\":\"1\"}", "{\"n\":0}" }) |text| {
        const req = try body(a, text);
        try req.object.put(a, "input", .{ .string = "hi" });
        try std.testing.expectError(error.Refused, responses_translate.translate(&cx, req, NoHistory{}));
        try std.testing.expectEqualStrings("n must be 1; multiple choices are not supported", cx.message);
        try std.testing.expectError(error.Refused, fields.singleChoice(&cx, req));
    }
}

test "limit pins: top_logprobs stays within twenty" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    try std.testing.expectEqual(@as(?u8, null), try fields.probabilityOptions(&cx, try body(a, "{\"n\":1}"), true));
    try std.testing.expectEqual(@as(?u8, 0), try fields.probabilityOptions(&cx, try body(a, "{\"n\":1,\"logprobs\":true,\"top_logprobs\":0}"), true));
    try std.testing.expectEqual(@as(?u8, 20), try fields.probabilityOptions(&cx, try body(a, "{\"n\":1,\"logprobs\":true,\"top_logprobs\":20}"), true));
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "{\"logprobs\":true,\"top_logprobs\":21}", "top_logprobs must be an integer between 0 and 20, or null" },
        .{ "{\"logprobs\":true,\"top_logprobs\":-1}", "top_logprobs must be an integer between 0 and 20, or null" },
        .{ "{\"top_logprobs\":2}", "top_logprobs requires logprobs: true" },
        .{ "{\"logprobs\":false,\"top_logprobs\":2}", "top_logprobs requires logprobs: true" },
    };
    for (cases) |case| {
        try std.testing.expectError(error.Refused, fields.probabilityOptions(&cx, try body(a, case[0]), true));
        try std.testing.expectEqualStrings(case[1], cx.message);
    }
}

test "limit pins: completions logprobs stays within twenty" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    try std.testing.expectEqual(@as(?u8, null), try fields.legacyLogprobs(&cx, try body(a, "{}"), true));
    try std.testing.expectEqual(@as(?u8, 0), try fields.legacyLogprobs(&cx, try body(a, "{\"logprobs\":0}"), true));
    try std.testing.expectEqual(@as(?u8, 20), try fields.legacyLogprobs(&cx, try body(a, "{\"logprobs\":20}"), true));
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "{\"logprobs\":21}", "logprobs must be an integer between 0 and 20, or null" },
        .{ "{\"logprobs\":-1}", "logprobs must be an integer between 0 and 20, or null" },
        .{ "{\"logprobs\":true}", "logprobs must be an integer between 0 and 20, or null" },
    };
    for (cases) |case| {
        try std.testing.expectError(error.Refused, fields.legacyLogprobs(&cx, try body(a, case[0]), true));
        try std.testing.expectEqualStrings(case[1], cx.message);
    }
}

test "limit pins: Responses metadata caps" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    const key62 = fill(62, 'k');
    const key63 = fill(63, 'k');
    const key64 = fill(64, 'k');
    const val512 = fill(512, 'v');
    const val513 = fill(513, 'v');
    _ = try responses_translate.translate(&cx, try metadataRequest(a, 16, &key62, &val512), NoHistory{});
    const message = "metadata must be at most 16 string pairs (keys up to 64 characters, values up to 512)";
    const cases = [_]struct { json.Value, []const u8 }{
        .{ try metadataRequest(a, 17, &key62, &val512), message },
        .{ try metadataRequest(a, 1, &key64, "v"), message },
        .{ try metadataRequest(a, 1, &key63, &val513), message },
    };
    for (cases) |case| {
        try std.testing.expectError(error.Refused, responses_translate.translate(&cx, case[0], NoHistory{}));
        try std.testing.expectEqualStrings(case[1], cx.message);
    }
    const non_string = try json.newObject(a);
    try non_string.put(a, "model", .{ .string = "m" });
    try non_string.put(a, "input", .{ .string = "hi" });
    var meta = try json.newObject(a);
    try meta.put(a, "k", .{ .int = "1" });
    try non_string.put(a, "metadata", .{ .object = meta });
    try std.testing.expectError(error.Refused, responses_translate.translate(&cx, .{ .object = non_string }, NoHistory{}));
    try std.testing.expectEqualStrings(message, cx.message);
}

test "limit pins: serve flags at their documented bounds" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    if (@import("builtin").os.tag == .macos) { // CFG flags: macOS only (drama_cfg.zig)
        const off = try cli.parse(a, &.{ "m", "--cfg-gamma", "0" }, &u);
        try std.testing.expectEqual(@as(f32, 0), off.cfg.gamma);
        const top = try cli.parse(a, &.{ "m", "--cfg-gamma", "5" }, &u);
        try std.testing.expectEqual(@as(f32, 5), top.cfg.gamma);
        for ([_][]const u8{ "5.5", "-0.5" }) |v| {
            try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--cfg-gamma", v }, &u));
            try std.testing.expectEqualStrings("--cfg-gamma must be 0 (off) or 1 (unguided) through 5", u.message);
        }
    }
    const share = try cli.parse(a, &.{ "m", "--decode-share", "0" }, &u);
    try std.testing.expectEqual(@as(?f64, 0), share.decode_share);
    for ([_][]const u8{ "1", "-0.1" }) |v| {
        try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--decode-share", v }, &u));
        try std.testing.expectStringStartsWith(u.message, "argument --decode-share: expected a share from 0 to under 1: ");
    }
    _ = try cli.parse(a, &.{ "m", "--snapshot-dir", "none" }, &u);
    _ = try cli.parse(a, &.{ "m", "--max-snapshots", "0" }, &u);
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--snapshot-dir", "cache" }, &u));
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--max-snapshots", "3" }, &u));
    const hosts = try cli.parse(a, try appendValues(a, "--allowed-host", 64, ""), &u);
    try std.testing.expectEqual(@as(usize, 64), hosts.allowed_host.len);
    try std.testing.expectError(error.Usage, cli.parse(a, try appendValues(a, "--allowed-host", 65, ""), &u));
    try std.testing.expectEqualStrings("at most 64 --allowed-host and 64 --allowed-origin entries are allowed", u.message);
    const origins = try cli.parse(a, try appendValues(a, "--allowed-origin", 64, "https://"), &u);
    try std.testing.expectEqual(@as(usize, 64), origins.allowed_origin.len);
    try std.testing.expectError(error.Usage, cli.parse(a, try appendValues(a, "--allowed-origin", 65, "https://"), &u));
}
