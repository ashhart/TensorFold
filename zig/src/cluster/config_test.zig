//! cluster.json: a worked example parses, every invalid file names its problem, and config layouts plan like "auto".
const std = @import("std");
const config = @import("config.zig");
const plan = @import("plan.zig");
const model = @import("model.zig");
const node = @import("node.zig");

pub const example =
    \\{
    \\  "schema": "tensorfold-cluster/1",
    \\  "cluster": "four-studios",
    \\  "transport": "mcdma",
    \\  "ssh": { "user": "tf", "key": "~/.ssh/cluster" },
    \\  "nodes": [
    \\    { "name": "node-a", "address": "192.0.2.10", "backend": "metal", "chip": "Apple M3 Ultra", "memory_gb": 512, "ports": "auto" },
    \\    { "name": "node-b", "address": "node-b.local", "via": "node-a", "backend": "metal", "ports": "auto" },
    \\    { "name": "node-c", "address": "node-c.local", "via": "node-a", "backend": "metal",
    \\      "ports": [ { "device": "rdma_en3", "peer": "node-b", "gbps": 80 }, { "device": "rdma_en4", "peer": "node-a", "gbps": 80 } ] },
    \\    { "name": "node-d", "address": "node-d.local", "via": "node-a", "backend": "metal" },
    \\    { "name": "node-e", "address": "192.0.2.21", "backend": "cuda", "gpu": "GB10", "memory_gb": 128 },
    \\    { "name": "spark2", "address": "192.0.2.22", "backend": "cuda", "gpu": "GB10", "memory_gb": 128 }
    \\  ],
    \\  "models": {
    \\    "kimi-k3": {
    \\      "path": "/models/moonshotai/Kimi-K3",
    \\      "nodes": ["node-a", "node-b", "node-c", "node-d"],
    \\      "parallel": { "tensor": 4, "pipeline": 1, "expert": 4 },
    \\      "mode": "converged",
    \\      "drafter": { "path": "/models/kimi-k3-dflash", "size_gb": 4, "uses_target_head": true },
    \\      "context": 262144, "streams": 2
    \\    },
    \\    "glm-5.3": {
    \\      "path": { "node-a": "/models/glm53", "node-b": "/models/glm53", "node-c": "/data/glm53", "node-d": "/models/glm53" },
    \\      "nodes": ["node-a", "node-b", "node-c", "node-d"],
    \\      "parallel": { "tensor": "auto", "pipeline": "auto", "expert": "auto" }
    \\    },
    \\    "qwen-27b": {
    \\      "path": "/models/qwen27b",
    \\      "nodes": ["node-e", "node-a"],
    \\      "mode": "disaggregated", "prefill": ["node-e"], "decode": ["node-a"]
    \\    }
    \\  }
    \\}
;

fn check(text: []const u8, want: []const []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var probs: config.Problems = .{ .a = arena.allocator() };
    _ = config.parse(arena.allocator(), text, &probs) catch {};
    for (want) |w| {
        var found = false;
        for (probs.list.items) |p| found = found or std.mem.indexOf(u8, p, w) != null;
        if (!found) {
            std.debug.print("missing problem \"{s}\"; got:\n", .{w});
            for (probs.list.items) |p| std.debug.print("  {s}\n", .{p});
            return error.TestExpectedProblem;
        }
    }
    if (want.len == 0) try std.testing.expect(probs.ok());
}

test "the worked example parses: Metal TP4 x EP4 converged, auto layouts, a Spark prefilling for a Mac" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var probs: config.Problems = .{ .a = arena.allocator() };
    const c = try config.parse(arena.allocator(), example, &probs);
    try std.testing.expectEqual(@as(usize, 6), c.nodes.len);
    try std.testing.expectEqualStrings("node-a", c.nodes[1].via.?);
    try std.testing.expectEqual(node.Backend.cuda, c.nodes[4].backend);
    try std.testing.expectEqual(@as(?u64, 512 * node.gib), c.nodes[0].memory);
    try std.testing.expectEqual(@as(usize, 2), c.nodes[2].ports.?.len);
    const k3 = c.model("kimi-k3").?;
    try std.testing.expectEqual(plan.Layout{ .tensor = 4, .pipeline = 1, .expert = 4 }, k3.layout);
    try std.testing.expectEqual(@as(u64, 4 * node.gib), k3.options().drafter_bytes);
    try std.testing.expectEqual(@as(u64, 262144), k3.options().context);
    const glm = c.model("glm-5.3").?;
    try std.testing.expectEqual(plan.Layout{}, glm.layout);
    try std.testing.expectEqualStrings("/data/glm53", glm.pathFor("node-c").?);
    const q = c.model("qwen-27b").?;
    try std.testing.expectEqual(plan.Mode.disaggregated, q.mode);
    try std.testing.expectEqualStrings("node-e", q.prefill[0]);
}

test "every broken file names what to change" {
    const head = "{\"schema\":\"tensorfold-cluster/1\",\"cluster\":\"x\",";
    try check(head ++ "\"transport\":\"tcp\",\"nodes\":[{\"name\":\"a\",\"address\":\"h\",\"backend\":\"metal\"}]}", &.{"no TCP"});
    try check(head ++ "\"nodes\":[{\"name\":\"a\",\"address\":\"h\",\"backend\":\"metal\"},{\"name\":\"a\",\"address\":\"h\",\"backend\":\"rocm\"}]}", &.{ "used twice", "backend must be" });
    try check(head ++ "\"nodes\":[{\"name\":\"a\",\"address\":\"h\",\"backend\":\"metal\",\"via\":\"b\"},{\"name\":\"b\",\"address\":\"\",\"backend\":\"metal\",\"via\":\"a\"}]}", &.{ "use one hop", "address: required" });
    try check(head ++ "\"nodes\":[{\"name\":\"bad name!\",\"address\":\"h\",\"backend\":\"metal\",\"via\":\"zz\"}]}", &.{ "1-20 characters", "no node is named" });
    const four = "\"nodes\":[{\"name\":\"m1\",\"address\":\"h\",\"backend\":\"metal\"},{\"name\":\"m2\",\"address\":\"h\",\"backend\":\"metal\"},{\"name\":\"c1\",\"address\":\"h\",\"backend\":\"cuda\"},{\"name\":\"c2\",\"address\":\"h\",\"backend\":\"cuda\"}],";
    try check(head ++ four ++ "\"models\":{\"k\":{\"path\":\"/m\",\"nodes\":[\"m1\",\"m2\",\"c1\",\"c2\"],\"parallel\":{\"tensor\":3}}}}", &.{"does not divide"});
    try check(head ++ four ++ "\"models\":{\"k\":{\"path\":\"/m\",\"nodes\":[\"m1\",\"m2\",\"c1\",\"c2\"],\"parallel\":{\"tensor\":4,\"expert\":3}}}}", &.{"must divide tensor"});
    try check(head ++ four ++ "\"models\":{\"k\":{\"path\":\"/m\",\"nodes\":[\"m1\",\"c1\"]}}}", &.{"different bits"});
    try check(head ++ four ++ "\"models\":{\"k\":{\"path\":\"/m\",\"nodes\":[\"m1\",\"m2\",\"c1\",\"c2\"],\"parallel\":{\"tensor\":2,\"pipeline\":2}}}}", &.{});
    try check(head ++ four ++ "\"models\":{\"k\":{\"path\":\"/m\",\"mode\":\"disaggregated\",\"prefill\":[\"c1\"],\"decode\":[\"c1\",\"zz\"]}}}", &.{ "both a prefill and a decode", "\"zz\" is not a node" });
    try check(head ++ four ++ "\"models\":{\"k\":{\"path\":\"/m\",\"mode\":\"disaggregated\"}}}", &.{"needs \"prefill\" and \"decode\""});
    try check(head ++ four ++ "\"models\":{\"k\":{\"nodes\":[\"m9\"],\"parallel\":{\"tensor\":\"many\"}}}}", &.{ "path: required", "\"m9\" is not a node", "or \"auto\"" });
    try check("{\"cluster\":1}", &.{ "schema: must be", "nodes: at least one" });
    try check("not json", &.{"not JSON"});
}

test "a config's explicit layout plans exactly like \"auto\" on the four Studios" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var probs: config.Problems = .{ .a = a };
    const c = try config.parse(a, example, &probs);
    const pt = @import("plan_test.zig");
    const ck = try pt.k3(a);
    const s = model.k3();
    var nodes: [4]node.Inventory = undefined;
    for (&nodes, 0..) |*n, i| n.* = pt.studio(i, 512 * node.gib, 0);
    var auto = c.model("kimi-k3").?;
    const given = try plan.plan(a, ck, &s, &nodes, auto.options());
    auto.layout = .{};
    const chosen = try plan.plan(a, ck, &s, &nodes, auto.options());
    try std.testing.expectEqual(given.layout, chosen.layout);
    try std.testing.expectEqual(given.digest, chosen.digest);
}
