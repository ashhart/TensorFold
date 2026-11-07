//! cluster.json: a worked example parses, every invalid file names its problem, and config layouts plan like "auto".
const std = @import("std");
const fabric = @import("fabric");
const config = @import("config.zig");
const launch = @import("launch.zig");
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
    \\    { "name": "node-a", "address": "192.0.2.10", "backend": "metal", "chip": "Apple M3 Ultra", "memory_gb": 512, "ports": "auto",
    \\      "library": "/opt/mcdma/libmcdma-fabric.dylib" },
    \\    { "name": "node-b", "address": "node-b.local", "via": "node-a", "backend": "metal", "ports": "auto" },
    \\    { "name": "node-c", "address": "node-c.local", "via": "node-a", "backend": "metal",
    \\      "ports": [ { "device": "rdma_en3", "peer": "node-b", "gbps": 80 }, { "device": "rdma_en4", "peer": "node-a", "gbps": 80 } ] },
    \\    { "name": "node-d", "address": "node-d.local", "via": "node-a", "backend": "metal" },
    \\    { "name": "node-e", "address": "192.0.2.21", "backend": "cuda", "gpu": "GB10", "memory_gb": 128, "library": "/opt/mcdma/libmcdma-fabric.so" },
    \\    { "name": "spark2", "address": "192.0.2.22", "backend": "cuda", "gpu": "GB10", "memory_gb": 128 }
    \\  ],
    \\  "links": [
    \\    { "name": "a-e", "kind": "cx5", "mtu": 1024,
    \\      "ends": [ { "node": "node-a", "device": "rdma_mcrdma0", "via": "en7/192.0.2.31", "port": 7490, "gid": 0 },
    \\                { "node": "node-e", "device": "rocep1s0f1", "via": "enp1s0f1np1/192.0.2.30", "port": 7491, "gid": 1 } ] }
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

test "a node's MCDMA library and a link declared once: both ends read one name, kind and path MTU, each end's peer port is the other's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var probs: config.Problems = .{ .a = arena.allocator() };
    const c = try config.parse(arena.allocator(), example, &probs);
    try std.testing.expectEqualStrings("/opt/mcdma/libmcdma-fabric.dylib", c.nodes[0].library.?);
    try std.testing.expectEqualStrings("/opt/mcdma/libmcdma-fabric.so", c.nodes[4].library.?);
    try std.testing.expectEqual(@as(?[]const u8, null), c.nodes[1].library);
    try std.testing.expectEqual(@as(usize, 1), c.links.len);
    const l = c.linkBetween("node-e", "node-a").?;
    try std.testing.expectEqualStrings("a-e", l.name);
    try std.testing.expectEqual(@as(?fabric.rdma.LinkKind, .cx5), l.kind);
    try std.testing.expectEqual(@as(c_int, 1024), l.mtu);
    try std.testing.expectEqual(@as(?u1, 1), l.endOf("node-e"));
    try std.testing.expectEqualStrings("rdma_mcrdma0", l.ends[0].device);
    try std.testing.expectEqualStrings("en7/192.0.2.31", l.ends[0].via);
    try std.testing.expectEqual(@as(u16, 7490), l.ends[0].port);
    try std.testing.expectEqual(@as(u16, 7491), l.ends[1].port);
    try std.testing.expectEqual(@as(c_int, 0), l.ends[0].gidIndex());
    try std.testing.expectEqual(@as(c_int, 1), l.ends[1].gidIndex());
    try std.testing.expect(c.linkBetween("node-a", "node-b") == null);
}

test "a link's defaults are MCDMA's: path MTU 4096, whichever kind connects, GID 1 (or -1 to scan a bond of more than two)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var probs: config.Problems = .{ .a = arena.allocator() };
    const c = try config.parse(arena.allocator(), top ++ macs ++ "\"links\":[{\"name\":\"tb\",\"ends\":[" ++
        "{\"node\":\"m1\",\"device\":\"rdma_en3+rdma_en4+rdma_en5\",\"via\":\"en3/192.0.2.2\",\"port\":7400}," ++
        "{\"node\":\"m2\",\"device\":\"rdma_en3\",\"via\":\"en3/192.0.2.1\",\"port\":7400}]}]}", &probs);
    const l = c.links[0];
    try std.testing.expectEqual(@as(c_int, 4096), l.mtu);
    try std.testing.expectEqual(@as(?fabric.rdma.LinkKind, null), l.kind);
    try std.testing.expectEqual(@as(?c_int, null), l.ends[0].gid);
    try std.testing.expectEqual(@as(c_int, -1), l.ends[0].gidIndex());
    try std.testing.expectEqual(@as(c_int, 1), l.ends[1].gidIndex());
}

const top = "{\"schema\":\"tensorfold-cluster/1\",\"cluster\":\"x\",";
const macs = "\"nodes\":[{\"name\":\"m1\",\"address\":\"h\",\"backend\":\"metal\"},{\"name\":\"m2\",\"address\":\"h\",\"backend\":\"metal\"},{\"name\":\"c1\",\"address\":\"h\",\"backend\":\"cuda\"},{\"name\":\"c2\",\"address\":\"h\",\"backend\":\"cuda\"}],";

/// A file with one link between `a` and `b` whose fields are `fields` (the link's own, then each end's).
fn linkFile(comptime fields: []const u8, comptime a_end: []const u8, comptime b_end: []const u8) []const u8 {
    return top ++ macs ++ "\"links\":[{\"name\":\"l1\"" ++ fields ++ ",\"ends\":[{\"node\":\"m1\",\"device\":\"rdma_mcrdma0\",\"via\":\"en7/192.0.2.2\",\"port\":7490" ++ a_end ++
        "},{\"node\":\"c1\",\"device\":\"rocep1s0f1\",\"via\":\"enp1/192.0.2.1\",\"port\":7490" ++ b_end ++ "}]}]}";
}

test "every broken link or library names what to change, before anything opens" {
    try check(linkFile(",\"kind\":\"cx5\",\"mtu\":1024", ",\"gid\":0", ",\"gid\":1"), &.{});
    try check(linkFile(",\"kind\":\"cx7\"", "", ""), &.{"kind: \"tb5\" or \"cx5\""});
    try check(linkFile(",\"kind\":2", "", ""), &.{"kind: \"tb5\" or \"cx5\""});
    try check(linkFile(",\"mtu\":1500", "", ""), &.{"mtu: 1024, 2048 or 4096"});
    try check(linkFile(",\"mtu\":\"1024\"", "", ""), &.{"mtu: 1024, 2048 or 4096"});
    try check(linkFile(",\"kind\":\"cx5\"", "", ",\"gid\":1"), &.{"needs its RoCE v2 GID index"});
    try check(linkFile(",\"kind\":\"cx5\"", ",\"gid\":-1", ",\"gid\":1"), &.{"needs its RoCE v2 GID index"});
    try check(linkFile(",\"kind\":\"cx5\"", ",\"gid\":1", ",\"gid\":1"), &.{"GID index 0 only"});
    try check(linkFile("", ",\"gid\":-2", ""), &.{"gid: a GID index from 0 to 255"});
    try check(linkFile(",\"kind\":\"tb5\"", "", ""), &.{"c1 is a CUDA host"});
    try check(top ++ macs ++ "\"links\":[{\"name\":\"l1\",\"ends\":[{\"node\":\"m1\",\"device\":\"d\",\"via\":\"v\",\"port\":70000},{\"node\":\"m2\",\"device\":\"d\",\"via\":\"v\",\"port\":\"1\"}]}]}", &.{ "ends[0].port: an exchange port from 1 to 65535", "ends[1].port: an exchange port" });
    try check(top ++ macs ++ "\"links\":[{\"name\":\"bad name!\",\"ends\":[{\"node\":\"m1\",\"device\":\"\",\"via\":\"en7 /x\",\"port\":7490},{\"node\":\"m1\",\"device\":\"d\",\"via\":\"v\"}]}]}", &.{ "links[0].name: 1-20 characters", "device: an RDMA device", "via: the interface, a slash", "a link joins two different nodes", "port: required" });
    try check(top ++ macs ++ "\"links\":[{\"name\":\"l1\",\"kind\":\"cx5\",\"ends\":[{\"node\":\"m1\",\"device\":\"a+b\",\"via\":\"x+y+z\",\"port\":65535,\"gid\":0},{\"node\":\"zz\",\"device\":\"d\",\"via\":\"v\",\"port\":1}]}]}", &.{ "takes one device, no bond", "via: the interface", "past 65535", "no node is named \"zz\"" });
    try check(top ++ macs ++ "\"links\":[{\"name\":\"l1\",\"ends\":[{\"node\":\"m1\"}]}]}", &.{"ends: two objects"});
    try check(top ++ macs ++ "\"links\":{}}", &.{"links: must be an array"});
    const twice = "{\"name\":\"l1\",\"ends\":[{\"node\":\"m1\",\"device\":\"d1\",\"via\":\"v\",\"port\":1},{\"node\":\"m2\",\"device\":\"d1\",\"via\":\"v\",\"port\":1}]}";
    try check(top ++ macs ++ "\"links\":[" ++ twice ++ ",{\"name\":\"l1\",\"ends\":[{\"node\":\"m2\",\"device\":\"d2\",\"via\":\"v\",\"port\":2},{\"node\":\"m1\",\"device\":\"d2\",\"via\":\"v\",\"port\":2}]}]}", &.{ "\"l1\" is used twice", "already joined by link \"l1\"" });
    try check(top ++ macs ++ "\"links\":[" ++ twice ++ ",{\"name\":\"l2\",\"ends\":[{\"node\":\"m1\",\"device\":\"d1\",\"via\":\"v\",\"port\":2},{\"node\":\"c1\",\"device\":\"d2\",\"via\":\"v\",\"port\":2}]}]}", &.{"m1's d1 already carries link \"l1\""});
    try check(top ++ "\"nodes\":[{\"name\":\"m1\",\"address\":\"h\",\"backend\":\"metal\",\"library\":\"lib/libmcdma-fabric.dylib\"},{\"name\":\"m2\",\"address\":\"h\",\"backend\":\"metal\",\"library\":7}]}", &.{ "nodes[0] (m1).library: libmcdma-fabric's absolute path", "nodes[1] (m2).library" });
}

test "a disaggregated model's per-node paths need cover only its prefill and decode nodes, which run it" {
    const paths = "\"models\":{\"q\":{\"path\":{\"c1\":\"/m/q\",\"m1\":\"/data/q\"},\"mode\":\"disaggregated\",\"prefill\":[\"c1\"],\"decode\":[\"m1\"]}}}";
    try check(top ++ macs ++ paths, &.{});
    try check(top ++ macs ++ "\"models\":{\"q\":{\"path\":{\"c1\":\"/m/q\"},\"mode\":\"disaggregated\",\"prefill\":[\"c1\"],\"decode\":[\"m1\"]}}}", &.{"models.q.path: no path for node m1"});
    // a converged model's still need every node of the model
    try check(top ++ macs ++ "\"models\":{\"k\":{\"path\":{\"m1\":\"/m/k\"},\"nodes\":[\"m1\",\"m2\"]}}}", &.{"models.k.path: no path for node m2"});
}

test "a node's \"bin\" is an absolute path the launch can run as written; a relative path or shell syntax is refused by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ok = top ++ "\"nodes\":[{\"name\":\"g1\",\"address\":\"h\",\"backend\":\"cuda\",\"bin\":\"/home/tf/.local/bin/tensorfold\"},{\"name\":\"m1\",\"address\":\"h\",\"backend\":\"metal\"}]}";
    try check(ok, &.{});
    var probs: config.Problems = .{ .a = a };
    const c = try config.parse(a, ok, &probs);
    try std.testing.expectEqualStrings("/home/tf/.local/bin/tensorfold", c.nodes[0].bin.?);
    try std.testing.expectEqual(@as(?[]const u8, null), c.nodes[1].bin);
    var out: std.Io.Writer.Allocating = .init(a);
    try launch.writeJson(&out.writer, c);
    var again: config.Problems = .{ .a = a };
    try std.testing.expectEqualStrings(c.nodes[0].bin.?, (try config.parse(a, out.written(), &again)).nodes[0].bin.?);
    for ([_][]const u8{ "tensorfold", "bin/tensorfold", "/opt/tf bin/tensorfold", "/opt/tf/tensorfold;reboot", "/", "" }) |bad| {
        const text = try std.mem.concat(a, u8, &.{ top, "\"nodes\":[{\"name\":\"g1\",\"address\":\"h\",\"backend\":\"cuda\",\"bin\":\"", bad, "\"}]}" });
        var p: config.Problems = .{ .a = a };
        try std.testing.expectError(error.Invalid, config.parse(a, text, &p));
        try std.testing.expect(std.mem.indexOf(u8, p.list.items[0], "nodes[0] (g1).bin: the absolute path of that node's tensorfold") != null);
    }
}

test "a declared link changes no refusal: Metal and CUDA in one tensor-parallel group stay refused" {
    const linked = top ++ macs ++ "\"links\":[{\"name\":\"l1\",\"kind\":\"cx5\",\"ends\":[{\"node\":\"m1\",\"device\":\"rdma_mcrdma0\",\"via\":\"en7/192.0.2.2\",\"port\":7490,\"gid\":0},{\"node\":\"c1\",\"device\":\"rocep1s0f1\",\"via\":\"enp1/192.0.2.1\",\"port\":7490,\"gid\":1}]}],";
    try check(linked ++ "\"models\":{\"k\":{\"path\":\"/m\",\"nodes\":[\"m1\",\"c1\"]}}}", &.{"different bits"});
    try check(linked ++ "\"models\":{\"k\":{\"path\":\"/m\",\"nodes\":[\"m1\",\"c1\"],\"parallel\":{\"tensor\":2}}}}", &.{"different bits"});
    try check(linked ++ "\"models\":{\"k\":{\"path\":\"/m\",\"nodes\":[\"m1\",\"c1\"],\"mode\":\"disaggregated\",\"prefill\":[\"c1\"],\"decode\":[\"c1\"]}}}", &.{"both a prefill and a decode"});
    try check(linked ++ "\"models\":{\"k\":{\"path\":\"/m\",\"nodes\":[\"m1\",\"c1\"],\"mode\":\"disaggregated\",\"prefill\":[\"c1\"],\"decode\":[\"m1\"]}}}", &.{});
}

test "cluster init's writer keeps a node's library and every link, and the file parses back the same" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var probs: config.Problems = .{ .a = a };
    const c = try config.parse(a, example, &probs);
    var out: std.Io.Writer.Allocating = .init(a);
    try launch.writeJson(&out.writer, c);
    var again: config.Problems = .{ .a = a };
    const back = try config.parse(a, out.written(), &again);
    try std.testing.expectEqualStrings(c.nodes[0].library.?, back.nodes[0].library.?);
    try std.testing.expectEqual(@as(?[]const u8, null), back.nodes[1].library);
    try std.testing.expectEqual(c.links.len, back.links.len);
    const x = c.links[0];
    const y = back.links[0];
    try std.testing.expectEqualStrings(x.name, y.name);
    try std.testing.expectEqual(x.kind, y.kind);
    try std.testing.expectEqual(x.mtu, y.mtu);
    for (x.ends, y.ends) |p, q| {
        try std.testing.expectEqualStrings(p.node, q.node);
        try std.testing.expectEqualStrings(p.device, q.device);
        try std.testing.expectEqualStrings(p.via, q.via);
        try std.testing.expectEqual(p.port, q.port);
        try std.testing.expectEqual(p.gid, q.gid);
    }
    var none: std.Io.Writer.Allocating = .init(a);
    try launch.writeJson(&none.writer, .{ .name = "x", .nodes = c.nodes[1..2], .models = &.{} });
    try std.testing.expect(std.mem.indexOf(u8, none.written(), "\"links\"") == null and std.mem.indexOf(u8, none.written(), "\"library\"") == null);
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
