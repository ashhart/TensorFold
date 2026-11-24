//! Each named EOS policy over one fixture set: the ids it admits, in order, or the refusal it names.
const std = @import("std");
const eos = @import("eos.zig");
const t = std.testing;
const Value = std.json.Value;

/// config.json (with any text_config inside), generation_config.json ("" for none), and each policy's answer.
const Case = struct { config: []const u8, generation: []const u8, served: []const u8, qwen35: []const u8 };

// Answers are "1,3" for ids in order, "-" for none, or an error name; one precedence for both policies.
const cases = [_]Case{
    .{ .config = "{\"eos_token_id\":1,\"text_config\":{\"eos_token_id\":2}}", .generation = "{\"eos_token_id\":[3]}", .served = "1,3", .qwen35 = "1,3" },
    .{ .config = "{\"text_config\":{\"eos_token_id\":2}}", .generation = "", .served = "2", .qwen35 = "2" },
    // a root null or [] gives no ids of its own, so the tower's are the ones the rule takes
    .{ .config = "{\"eos_token_id\":null,\"text_config\":{\"eos_token_id\":2}}", .generation = "", .served = "2", .qwen35 = "2" },
    .{ .config = "{\"eos_token_id\":[],\"text_config\":{\"eos_token_id\":2}}", .generation = "", .served = "2", .qwen35 = "2" },
    .{ .config = "{\"eos_token_id\":[1,1]}", .generation = "", .served = "1", .qwen35 = "1" },
    .{ .config = "{\"eos_token_id\":[1,2,3,4,5,6,7,8]}", .generation = "{\"eos_token_id\":8}", .served = "1,2,3,4,5,6,7,8", .qwen35 = "1,2,3,4,5,6,7,8" },
    .{ .config = "{\"eos_token_id\":[1,2,3,4,5,6,7,8]}", .generation = "{\"eos_token_id\":9}", .served = "1,2,3,4,5,6,7,8,9", .qwen35 = "TooManyEos" },
    .{ .config = "{\"eos_token_id\":[[1,2]]}", .generation = "", .served = "BadEosValue", .qwen35 = "1,2" },
    .{ .config = "{\"eos_token_id\":[1,null]}", .generation = "", .served = "1", .qwen35 = "1" },
    .{ .config = "{\"eos_token_id\":\"1\"}", .generation = "", .served = "BadEosValue", .qwen35 = "BadEosValue" },
    .{ .config = "{\"eos_token_id\":1.0}", .generation = "", .served = "BadEosValue", .qwen35 = "BadEosValue" },
    .{ .config = "{\"eos_token_id\":-1}", .generation = "", .served = "BadEosValue", .qwen35 = "BadEosValue" },
    .{ .config = "{\"eos_token_id\":4294967296}", .generation = "", .served = "BadEosValue", .qwen35 = "BadEosValue" },
    .{ .config = "{\"eos_token_id\":[1,true]}", .generation = "", .served = "BadEosValue", .qwen35 = "BadEosValue" },
    .{ .config = "{\"eos_token_id\":10}", .generation = "", .served = "BadEos", .qwen35 = "BadEos" },
    .{ .config = "{}", .generation = "{\"eos_token_id\":3}", .served = "3", .qwen35 = "3" },
    .{ .config = "{}", .generation = "{\"eos_token_id\":[9,10]}", .served = "BadEos", .qwen35 = "BadEos" },
    .{ .config = "{}", .generation = "", .served = "-", .qwen35 = "MissingEos" },
    .{ .config = "{}", .generation = "{\"eos_token_id\":null}", .served = "-", .qwen35 = "MissingEos" },
};

/// Ids as the fixture spells them.
fn spell(a: std.mem.Allocator, ids: []const u32) ![]const u8 {
    if (ids.len == 0) return "-";
    var out: std.ArrayList(u8) = .empty;
    for (ids, 0..) |id, i| try out.print(a, "{s}{d}", .{ if (i == 0) "" else ",", id });
    return out.items;
}

/// The policy's answer as the fixture spells it.
fn answer(a: std.mem.Allocator, policy: eos.Policy, root: Value, generation: Value) ![]const u8 {
    const text = if (root.object.get("text_config")) |v| v else root;
    var buf: [16]u32 = undefined;
    return spell(a, eos.collect(buf[0..], policy, root, text, generation, 10) catch |e| return @errorName(e));
}

fn parse(a: std.mem.Allocator, text: []const u8) !Value {
    return if (text.len == 0) .null else std.json.parseFromSliceLeaky(Value, a, text, .{});
}

/// The tower the fixture nests, or the root itself when it is flat.
fn tower(root: Value) Value {
    return if (root.object.get("text_config")) |v| v else root;
}

// Planted controls: each is a deliberately wrong reader, so the test names the answer the real rule must not give.

/// Ignores generation_config.json (its document left out).
fn noGeneration(buf: []u32, root: Value, text: Value, vocab: usize) eos.BufferError![]const u32 {
    return eos.collect(buf, eos.qwen35, root, text, .null, vocab);
}

/// Stops at the root and never falls through to the tower (the retired root-key rule).
fn rootOnly(buf: []u32, root: Value, vocab: usize) eos.BufferError![]const u32 {
    return eos.collect(buf, eos.qwen35, root, .null, .null, vocab);
}

/// Reads the tower in place of the root even when the root gave ids.
fn towerOnly(buf: []u32, text: Value, generation: Value, vocab: usize) eos.BufferError![]const u32 {
    return eos.collect(buf, eos.qwen35, .null, text, generation, vocab);
}

/// Unions every source (the retired union rule), each id once, at most 8.
fn unionSources(buf: []u32, root: Value, text: Value, generation: Value, vocab: usize) eos.BufferError![]const u32 {
    var n: usize = 0;
    for ([_]Value{ root, text, generation }) |src| {
        var part: [8]u32 = undefined;
        for (try eos.collect(part[0..], eos.qwen35, src, .null, .null, vocab)) |id| {
            if (std.mem.indexOfScalar(u32, buf[0..n], id) != null) continue;
            if (n == 8) return error.TooManyEos;
            buf[n] = id;
            n += 1;
        }
    }
    return buf[0..n];
}

test "EOS policies: each fixture's ids or refusal under served and Qwen 3.5" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (cases) |c| {
        const root = try parse(a, c.config);
        const gen = try parse(a, c.generation);
        errdefer std.debug.print("config {s} generation {s}\n", .{ c.config, c.generation });
        try t.expectEqualStrings(c.served, try answer(a, eos.served, root, gen));
        try t.expectEqualStrings(c.qwen35, try answer(a, eos.qwen35, root, gen));
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
    const root: Value = .{ .object = .empty };
    for ([_]eos.Policy{ eos.served, eos.qwen35 }) |p|
        try t.expectError(error.BadEosValue, eos.collect(&buf, p, root, root, .{ .integer = 1 }, 10));
}

test "one rule: a root null or [] gives the tower's ids, the root-only control refuses them" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var real: [16]u32 = undefined;
    var control: [16]u32 = undefined;
    const spellings = [_][]const u8{
        "{\"eos_token_id\":null,\"text_config\":{\"eos_token_id\":2}}",
        "{\"eos_token_id\":[],\"text_config\":{\"eos_token_id\":2}}",
    };
    for (spellings) |raw| {
        const root = try parse(a, raw);
        // the rule: the root gave nothing, so the tower's id is the stop token, as the served reader gives
        try t.expectEqualSlices(u32, &.{2}, try eos.collect(&real, eos.qwen35, root, tower(root), .null, 10));
        try t.expectEqualSlices(u32, &.{2}, try eos.resolve(a, root, .null, 10));
        // the planted control stops at the present root key, so it names the family's MissingEos instead
        try t.expectError(error.MissingEos, rootOnly(&control, root, 10));
    }
}

test "one rule: the root's ids win over the tower's, and every planted control differs from them" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var real: [16]u32 = undefined;
    var towered: [16]u32 = undefined;
    var unioned: [16]u32 = undefined;
    var nogen: [16]u32 = undefined;
    const root = try parse(a, "{\"eos_token_id\":1,\"text_config\":{\"eos_token_id\":2}}");
    const gen = try parse(a, "{\"eos_token_id\":[3]}");
    const ids = try eos.collect(&real, eos.qwen35, root, tower(root), gen, 10);
    try t.expectEqualSlices(u32, &.{ 1, 3 }, ids);
    try t.expectEqualSlices(u32, ids, try eos.resolve(a, root, gen, 10));
    // the tower in place of the root: 2 where the rule has 1
    try t.expectEqualSlices(u32, &.{ 2, 3 }, try towerOnly(&towered, tower(root), gen, 10));
    // the union: the tower's id joins where the rule stops at the root's
    try t.expectEqualSlices(u32, &.{ 1, 2, 3 }, try unionSources(&unioned, root, tower(root), gen, 10));
    // without generation_config.json the served id 3 is missing
    try t.expectEqualSlices(u32, &.{1}, try noGeneration(&nogen, root, tower(root), 10));
}
