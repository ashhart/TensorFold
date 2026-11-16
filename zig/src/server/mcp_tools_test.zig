//! The machine, models, info and pull tools against a temporary cache and a fake hub on 127.0.0.1.
const std = @import("std");
const json = @import("json");
const fixture = @import("mcp_fixture.zig");
const cache = @import("mcp_cache.zig");
const fake = @import("benchmark_fake.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

const initialize = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{}}}\n";

fn call(a: Allocator, id: i64, tool: []const u8, args: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"jsonrpc\":\"2.0\",\"id\":{d},\"method\":\"tools/call\",\"params\":{{\"name\":\"{s}\",\"arguments\":{s}}}}}\n", .{ id, tool, args });
}

fn structured(v: Value) Value {
    return v.get("result").?.get("structuredContent").?;
}

test "mcp repo ids: owner/name as the Hub allows them, with no room for a path" {
    for ([_][]const u8{ "Org/Lightning", "mlx-community/Qwen3.5-2B-MLX-4bit", "a_b/c.d", "Org/x_" }) |id| try std.testing.expect(cache.repoId(id));
    for ([_][]const u8{ "Org/..", "../x", "./x", "Org/a..b", "Org/a--b", "-org/x", "org/x.", "a/b/c", "plain", "/abs/x", "Org/x@main", "" }) |id|
        try std.testing.expect(!cache.repoId(id));
}

test "mcp machine: this host's allowlisted facts with the build's version and the model types it serves" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var home = try fixture.Home.init(a);
    defer home.deinit();
    const ctx = home.context();
    const got = try fixture.exchange(a, &ctx, try std.mem.concat(a, u8, &.{ initialize, try call(a, 2, "machine", "{}") }));
    const v = fixture.find(got, 2).?;
    try std.testing.expect(!fixture.isError(v));
    try std.testing.expect(fixture.has(fixture.text(v), "TensorFold " ++ fixture.version));
    const facts = structured(v);
    try std.testing.expectEqualStrings(@tagName(@import("builtin").os.tag), facts.strField("os").?);
    try std.testing.expectEqualStrings(fixture.version, facts.strField("tensorfold_version").?);
    try std.testing.expectEqualStrings("nemotron_h", facts.get("model_types").?.array[0].string);
    try std.testing.expect(facts.get("gpus").? == .array and facts.strField("memory_type") != null);
    try std.testing.expect(facts.get("memory_bytes").? == .int or facts.get("memory_bytes").? == .null);
}

test "mcp models and info: the cached checkpoints a family serves, and one checkpoint's facts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var home = try fixture.Home.init(a);
    defer home.deinit();
    const ctx = home.context();
    const empty = try fixture.exchange(a, &ctx, try std.mem.concat(a, u8, &.{ initialize, try call(a, 2, "models", "{}") }));
    try std.testing.expect(fixture.has(fixture.text(fixture.find(empty, 2).?), "download one with the pull tool"));
    const snapshot = try home.checkpoint(a, "Org/Lightning", fixture.lightning);
    _ = try home.checkpoint(a, "Org/Draft", "{\"model_type\": \"gemma3n\"}");
    const outside = try home.path(a, "not-a-checkpoint");
    try std.Io.Dir.cwd().createDirPath(std.testing.io, outside);
    const input = try std.mem.concat(a, u8, &.{
        initialize,
        try call(a, 2, "models", "{}"),
        try call(a, 3, "info", "{\"model\":\"Org/Lightning\"}"),
        try call(a, 4, "info", try std.fmt.allocPrint(a, "{{\"model\":\"{s}\"}}", .{snapshot})),
        try call(a, 5, "info", "{\"model\":\"Org/Draft\"}"),
        try call(a, 6, "info", "{\"model\":\"Org/Missing\"}"),
        try call(a, 7, "info", "{\"model\":\"Org/..\"}"),
        try call(a, 8, "info", "{\"model\":\"relative-dir\"}"),
        try call(a, 9, "info", try std.fmt.allocPrint(a, "{{\"model\":\"{s}\"}}", .{outside})),
        try call(a, 10, "info", try std.fmt.allocPrint(a, "{{\"model\":\"{s}/../not-a-checkpoint\"}}", .{home.dir})),
    });
    const got = try fixture.exchange(a, &ctx, input);
    const listed = structured(fixture.find(got, 2).?).get("models").?.array;
    try std.testing.expectEqual(@as(usize, 1), listed.len);
    try std.testing.expectEqualStrings("Org/Lightning", listed[0].strField("repo").?);
    try std.testing.expectEqualStrings("Nemotron 3.5 Lightning", listed[0].strField("family").?);
    try std.testing.expectEqualStrings("nemotron_h", listed[0].strField("model_type").?);
    try std.testing.expectEqualStrings(fixture.sha, listed[0].strField("revision").?);
    try std.testing.expectEqual(@as(?i64, 4096 + fixture.lightning.len), listed[0].get("size_bytes").?.int64());
    try std.testing.expect(fixture.has(fixture.text(fixture.find(got, 2).?), "Org/Lightning: Nemotron 3.5 Lightning (nemotron_h)"));
    for ([_]i64{ 3, 4 }) |id| {
        const v = fixture.find(got, id).?;
        try std.testing.expect(!fixture.isError(v));
        try std.testing.expect(fixture.has(fixture.text(v), "family: Nemotron 3.5 Lightning (nemotron_h)\nformat: 4-bit, group 64\ncontext: 8192\n"));
        const facts = structured(v);
        try std.testing.expectEqual(@as(?i64, 4), facts.get("quantization").?.get("bits").?.int64());
        try std.testing.expectEqual(@as(?i64, 64), facts.get("quantization").?.get("group_size").?.int64());
        try std.testing.expect(facts.get("modelopt_formats").? == .null);
        try std.testing.expect(std.mem.indexOf(u8, fixture.text(v), "snapshots") == null); // the answer names the model, not its path
        try std.testing.expectEqual(@as(?i64, 8192), facts.get("context").?.int64());
        try std.testing.expectEqual(@as(?i64, 4096 + fixture.lightning.len), facts.get("weights_bytes").?.int64());
        try std.testing.expect(facts.get("memory_floor_bytes").?.int64().? > facts.get("weights_bytes").?.int64().?);
    }
    const refused = [_]struct { id: i64, says: []const u8 }{
        .{ .id = 5, .says = "Org/Draft: no registered Zig family serves model_type gemma3n" }, // named as asked, not by its path
        .{ .id = 6, .says = "Org/Missing is not in the Hugging Face cache; download it with the pull tool first" },
        .{ .id = 7, .says = "neither a Hugging Face repo id" },
        .{ .id = 8, .says = "relative paths are refused" },
        .{ .id = 9, .says = "is not a checkpoint directory" },
        .{ .id = 10, .says = "a path with .. is refused" },
    };
    for (refused) |r| {
        const v = fixture.find(got, r.id).?;
        try std.testing.expect(fixture.isError(v));
        try std.testing.expect(fixture.has(fixture.text(v), r.says));
    }
}

/// A fake hub serving Org/Lightning (a family this build serves) and Org/Draft (one it does not).
fn hub(a: Allocator, gate: ?*std.Io.Event) !*fake.Fake {
    const weights: [300]u8 = @splat('w');
    const tree = try std.fmt.allocPrint(a, "[{{\"type\": \"file\", \"path\": \"config.json\", \"size\": {d}}}, {{\"type\": \"file\", \"path\": \"model.safetensors\", \"size\": {d}}}]", .{ fixture.lightning.len, weights.len });
    const draft = "{\"model_type\": \"gemma3n\"}";
    const draft_tree = try std.fmt.allocPrint(a, "[{{\"type\": \"file\", \"path\": \"config.json\", \"size\": {d}}}]", .{draft.len});
    const revision = "{\"sha\": \"" ++ fixture.sha ++ "\"}";
    const slow: []const fake.Write = if (gate) |g| try a.dupe(fake.Write, &.{
        .{ .bytes = try std.fmt.allocPrint(a, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{revision.len}), .after = g },
        .{ .bytes = revision },
    }) else try fake.jsonReply(a, "200 OK", revision);
    const routes = try a.dupe(fake.Route, &.{
        .{ .method = "GET", .path = "/api/models/Org/Lightning/revision/main", .replies = try a.dupe([]const fake.Write, &.{slow}) },
        .{ .method = "GET", .path = "/api/models/Org/Lightning/tree/" ++ fixture.sha ++ "?recursive=true", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", tree)}) },
        .{ .method = "GET", .path = "/Org/Lightning/resolve/" ++ fixture.sha ++ "/config.json", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", fixture.lightning)}) },
        .{ .method = "GET", .path = "/Org/Lightning/resolve/" ++ fixture.sha ++ "/model.safetensors", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", &weights)}) },
        .{ .method = "GET", .path = "/api/models/Org/Draft/revision/main", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", revision)}) },
        .{ .method = "GET", .path = "/api/models/Org/Draft/tree/" ++ fixture.sha ++ "?recursive=true", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", draft_tree)}) },
        .{ .method = "GET", .path = "/Org/Draft/resolve/" ++ fixture.sha ++ "/config.json", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", draft)}) },
    });
    return fake.Fake.start(std.testing.allocator, routes);
}

test "mcp pull: tensorfold pull's download, its lines as progress notifications when asked, and its refusals" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var home = try fixture.Home.init(a);
    defer home.deinit();
    const f = try hub(a, null);
    defer f.stop();
    try home.env.put("HF_ENDPOINT", try f.url(a, ""));
    const ctx = home.context();
    var live: fixture.Live = undefined;
    live.setUp(&ctx);
    defer live.tearDown();
    try live.send(initialize);
    try live.send("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"pull\",\"arguments\":{\"repo\":\"Org/Lightning\"},\"_meta\":{\"progressToken\":\"p-2\"}}}\n");
    _ = try live.answer(a, 2);
    for ([_][]const u8{ "{\"repo\":\"Org/Lightning\",\"revision\":\"main\"}", "{\"repo\":\"Org/Draft\"}", "{\"repo\":\"Org/Lightning@main\"}", "{\"repo\":\"../etc/x\"}", "{\"repo\":\"Org/Lightning\",\"revision\":\"../main\"}", "{\"repo\":\"Org/Missing\"}" }, 3..) |args, id| {
        try live.send(try call(a, @intCast(id), "pull", args)); // one at a time: two pulls of one repo never overlap here
        _ = try live.answer(a, @intCast(id));
    }
    const got = try live.written(a);
    var notes: std.ArrayList(Value) = .empty;
    var answered_at: ?usize = null;
    for (got, 0..) |v, i| {
        if (v.get("method")) |m| {
            try std.testing.expectEqualStrings("notifications/progress", m.string);
            try std.testing.expectEqualStrings("p-2", v.get("params").?.strField("progressToken").?);
            try notes.append(a, v);
        } else if (v.get("id").?.int64() == 2) answered_at = i;
    }
    try std.testing.expect(notes.items.len >= 3);
    for (notes.items, 1..) |n, i| try std.testing.expectEqual(@as(?i64, @intCast(i)), n.get("params").?.get("progress").?.int64());
    try std.testing.expect(fixture.has(notes.items[0].get("params").?.strField("message").?, "downloading Org/Lightning@main"));
    try std.testing.expect(answered_at.? > 0 and got[answered_at.? - 1].get("method") != null); // the notes come first
    const pulled = fixture.find(got, 2).?;
    try std.testing.expect(!fixture.isError(pulled));
    try std.testing.expectEqualStrings(fixture.sha, structured(pulled).strField("revision").?);
    try std.testing.expectEqualStrings("Nemotron 3.5 Lightning", structured(pulled).strField("family").?);
    const config = try home.path(a, "hub/models--Org--Lightning/snapshots/" ++ fixture.sha ++ "/config.json");
    try std.testing.expectEqualStrings(fixture.lightning, try std.Io.Dir.cwd().readFileAlloc(std.testing.io, config, a, .limited(1 << 20)));
    const again = fixture.find(got, 3).?; // no progress token: the answer only
    try std.testing.expect(!fixture.isError(again));
    try std.testing.expect(fixture.has(fixture.text(again), "Pulled Org/Lightning at revision " ++ fixture.sha));
    const refused = [_]struct { id: i64, says: []const u8 }{
        .{ .id = 4, .says = "model_type gemma3n" },
        .{ .id = 5, .says = "is not a Hugging Face repo id (owner/name)" },
        .{ .id = 6, .says = "is not a Hugging Face repo id (owner/name)" },
        .{ .id = 7, .says = "not a revision Hugging Face names" },
        .{ .id = 8, .says = "pull Org/Missing failed: the hub answered with an error or no files" },
    };
    for (refused) |r| {
        const v = fixture.find(got, r.id).?;
        try std.testing.expect(fixture.isError(v));
        try std.testing.expect(fixture.has(fixture.text(v), r.says));
    }
    for (0..f.count()) |i| try std.testing.expect(std.mem.indexOf(u8, f.request(i).path, "..") == null and std.mem.indexOfScalar(u8, f.request(i).path, '@') == null);
}

test "mcp pull: a second pull of a repo already downloading is refused while the first goes on" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var home = try fixture.Home.init(a);
    defer home.deinit();
    var gate: std.Io.Event = .unset;
    const f = try hub(a, &gate);
    defer f.stop();
    try home.env.put("HF_ENDPOINT", try f.url(a, ""));
    const ctx = home.context();
    var live: fixture.Live = undefined;
    live.setUp(&ctx);
    defer live.tearDown();
    defer gate.set(std.testing.io); // before the session's end, which waits for the first pull
    try live.send(initialize);
    try live.send(try call(a, 2, "pull", "{\"repo\":\"Org/Lightning\"}"));
    var waited: u32 = 0;
    while (f.count() == 0 and waited < 10_000) : (waited += 10) fake.nap(10); // the first pull is at the hub
    try live.send(try call(a, 3, "pull", "{\"repo\":\"org/lightning\"}"));
    const second = try live.answer(a, 3);
    try std.testing.expect(fixture.isError(second));
    try std.testing.expect(fixture.has(fixture.text(second), "already being pulled by this session"));
    try std.testing.expect(fixture.find(try live.written(a), 2) == null);
    gate.set(std.testing.io);
    try std.testing.expect(!fixture.isError(try live.answer(a, 2)));
}
