//! Each named EOS policy over one fixture set: the ids it admits, in order, or the refusal it names.
const std = @import("std");
const eos = @import("eos.zig");
const t = std.testing;

/// config.json (with any text_config inside), generation_config.json ("" for none), and each policy's answer.
const Case = struct { config: []const u8, generation: []const u8, served: []const u8, hip: []const u8, metal: []const u8 };

// Answers are "1,3" for ids in order, "-" for none, or an error name.
const cases = [_]Case{
    .{ .config = "{\"eos_token_id\":1,\"text_config\":{\"eos_token_id\":2}}", .generation = "{\"eos_token_id\":[3]}", .served = "1,3", .hip = "1,2,3", .metal = "1,3" },
    .{ .config = "{\"text_config\":{\"eos_token_id\":2}}", .generation = "", .served = "2", .hip = "2", .metal = "2" },
    // a root null or [] hides text_config on qwen27 Metal only
    .{ .config = "{\"eos_token_id\":null,\"text_config\":{\"eos_token_id\":2}}", .generation = "", .served = "2", .hip = "2", .metal = "MissingEos" },
    .{ .config = "{\"eos_token_id\":[],\"text_config\":{\"eos_token_id\":2}}", .generation = "", .served = "2", .hip = "2", .metal = "MissingEos" },
    .{ .config = "{\"eos_token_id\":[1,1]}", .generation = "", .served = "1", .hip = "1", .metal = "1" },
    .{ .config = "{\"eos_token_id\":[1,2,3,4,5,6,7,8]}", .generation = "{\"eos_token_id\":8}", .served = "1,2,3,4,5,6,7,8", .hip = "1,2,3,4,5,6,7,8", .metal = "1,2,3,4,5,6,7,8" },
    .{ .config = "{\"eos_token_id\":[1,2,3,4,5,6,7,8]}", .generation = "{\"eos_token_id\":9}", .served = "1,2,3,4,5,6,7,8,9", .hip = "TooManyEos", .metal = "TooManyEos" },
    .{ .config = "{\"eos_token_id\":[[1,2]]}", .generation = "", .served = "BadEosValue", .hip = "1,2", .metal = "1,2" },
    .{ .config = "{\"eos_token_id\":[1,null]}", .generation = "", .served = "1", .hip = "1", .metal = "1" },
    .{ .config = "{\"eos_token_id\":\"1\"}", .generation = "", .served = "BadEosValue", .hip = "BadEosValue", .metal = "BadEosValue" },
    .{ .config = "{\"eos_token_id\":1.0}", .generation = "", .served = "BadEosValue", .hip = "BadEosValue", .metal = "BadEosValue" },
    .{ .config = "{\"eos_token_id\":-1}", .generation = "", .served = "BadEosValue", .hip = "BadEosValue", .metal = "BadEosValue" },
    .{ .config = "{\"eos_token_id\":4294967296}", .generation = "", .served = "BadEosValue", .hip = "BadEosValue", .metal = "BadEosValue" },
    .{ .config = "{\"eos_token_id\":[1,true]}", .generation = "", .served = "BadEosValue", .hip = "BadEosValue", .metal = "BadEosValue" },
    .{ .config = "{\"eos_token_id\":10}", .generation = "", .served = "BadEos", .hip = "BadEos", .metal = "BadEos" },
    .{ .config = "{}", .generation = "{\"eos_token_id\":3}", .served = "3", .hip = "3", .metal = "3" },
    .{ .config = "{}", .generation = "{\"eos_token_id\":[9,10]}", .served = "BadEos", .hip = "BadEos", .metal = "BadEos" },
    .{ .config = "{}", .generation = "", .served = "-", .hip = "MissingEos", .metal = "MissingEos" },
    .{ .config = "{}", .generation = "{\"eos_token_id\":null}", .served = "-", .hip = "MissingEos", .metal = "MissingEos" },
};

/// Ids as the fixture spells them.
fn spell(a: std.mem.Allocator, ids: []const u32) ![]const u8 {
    if (ids.len == 0) return "-";
    var out: std.ArrayList(u8) = .empty;
    for (ids, 0..) |id, i| try out.print(a, "{s}{d}", .{ if (i == 0) "" else ",", id });
    return out.items;
}

/// The policy's answer as the fixture spells it.
fn answer(a: std.mem.Allocator, policy: eos.Policy, root: std.json.Value, generation: std.json.Value) ![]const u8 {
    const text = if (root.object.get("text_config")) |v| v else root;
    var buf: [16]u32 = undefined;
    return spell(a, eos.collect(buf[0..], policy, root, text, generation, 10) catch |e| return @errorName(e));
}

fn parse(a: std.mem.Allocator, text: []const u8) !std.json.Value {
    return if (text.len == 0) .null else std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
}

test "EOS policies: each fixture's ids or refusal under served, Qwen3.5 HIP and qwen27 Metal" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (cases) |c| {
        const root = try parse(a, c.config);
        const gen = try parse(a, c.generation);
        errdefer std.debug.print("config {s} generation {s}\n", .{ c.config, c.generation });
        try t.expectEqualStrings(c.served, try answer(a, eos.served, root, gen));
        try t.expectEqualStrings(c.hip, try answer(a, eos.qwen35_hip, root, gen));
        try t.expectEqualStrings(c.metal, try answer(a, eos.qwen27_metal, root, gen));
    }
}

test "EOS policies: resolve is the served policy, every malformed value named BadEos" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (cases) |c| {
        const want = if (std.mem.eql(u8, c.served, "BadEosValue")) "BadEos" else c.served;
        const ids = eos.resolve(a, try parse(a, c.config), try parse(a, c.generation), 10) catch |e| {
            try t.expectEqualStrings(want, @errorName(e));
            continue;
        };
        try t.expectEqualStrings(want, try spell(a, ids));
    }
}

test "EOS policies: a source that is not an object is a malformed value under every policy" {
    var buf: [8]u32 = undefined;
    const root: std.json.Value = .{ .object = .empty };
    for ([_]eos.Policy{ eos.served, eos.qwen35_hip, eos.qwen27_metal }) |p|
        try t.expectError(error.BadEosValue, eos.collect(&buf, p, root, root, .{ .integer = 1 }, 10));
}
