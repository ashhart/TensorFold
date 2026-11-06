//! Loaded topology receipts, client provenance, and unchanged community-v1 bytes on host fixtures.
const std = @import("std");
const json = @import("json");
const receipt = @import("benchmark_receipt.zig");
const receipts = @import("benchmark_receipt_test.zig");
const check = @import("benchmark_check.zig");
const manifest = @import("benchmark_manifest.zig");
const fake = @import("benchmark_fake.zig");
const cli = @import("benchmark_cli_test.zig");
const server = @import("benchmark_server.zig");
const Value = json.Value;
const now_ms = 1767225660000;

pub const topology = @import("benchmark_group_fixture.zig").topology;

pub fn document(a: std.mem.Allocator, world: u32, backend: []const u8) !Value {
    const g = try topology(a, world, true);
    const rt = try json.newObject(a);
    try rt.put(a, "tensorfold_version", .{ .string = "9.8.7" });
    try rt.put(a, "backend", .{ .string = backend });
    try rt.put(a, "python_version", .{ .string = "none" });
    try rt.put(a, "platform", .{ .string = if (std.mem.eql(u8, backend, "mlx")) "macos" else "linux" });
    const model = try json.newObject(a);
    try model.put(a, "revision", .{ .string = "0123456789abcdef0123456789abcdef01234567" });
    const o = try json.newObject(a);
    try o.put(a, "schema_version", try json.intValue(a, @as(u8, 1)));
    try o.put(a, "rank_count", try json.intValue(a, world));
    try o.put(a, "group", g);
    try o.put(a, "runtime", .{ .object = rt });
    try o.put(a, "model", .{ .object = model });
    try o.put(a, "hardware", g.get("nodes").?.array[0].get("hardware").?);
    return .{ .object = o };
}

fn grouped(a: std.mem.Allocator, n: u32, multi: bool) !Value {
    const v = try receipts.valid(a);
    try v.object.put(a, "group", try topology(a, n, multi));
    try v.get("settings").?.object.put(a, "rank_count", try json.intValue(a, n));
    return v;
}

test "benchmark group: v2 accepts 1/2/4/8 ranks in separate single-node and multi-node categories" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]u32{ 1, 2, 4, 8 }) |n| for ([_]bool{ false, true }) |multi| {
        const v = try grouped(a, n, multi);
        var why: []const u8 = "";
        try std.testing.expect(check.receipt(v, .save, now_ms, &why));
        try std.testing.expect(check.receipt(v, .publish, now_ms, &why));
        const g = v.get("group").?;
        try std.testing.expectEqualStrings(if (multi and n > 1) "multi-node" else "single-node", g.get("category").?.string);
        const doc = try document(a, n, "cuda");
        const facts = (try manifest.parse(a, try json.stringify(a, doc, .{}), "cuda")).?;
        try std.testing.expectEqual(@as(?u8, @intCast(n)), facts.rank_count);
        try std.testing.expect(facts.group != null);
    };
}

test "benchmark group: forged counts, sources, rank cover and node hardware fail closed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const faults = [_]struct { path: []const []const u8, text: ?[]const u8 }{
        .{ .path = &.{ "settings", "rank_count" }, .text = "8" },
        .{ .path = &.{ "group", "rank_count" }, .text = "2" },
        .{ .path = &.{ "group", "source" }, .text = "\"environment\"" },
        .{ .path = &.{ "group", "category" }, .text = "\"single-node\"" },
        .{ .path = &.{ "group", "nodes", "1", "id" }, .text = "0" },
        .{ .path = &.{ "group", "nodes", "1", "ranks" }, .text = "[0]" },
        .{ .path = &.{ "group", "nodes", "1", "ranks" }, .text = "[]" },
        .{ .path = &.{ "group", "nodes", "1", "ranks" }, .text = "[4]" },
        .{ .path = &.{ "group", "nodes", "1", "hardware", "source" }, .text = "\"unavailable\"" },
        .{ .path = &.{ "group", "nodes", "1", "hardware", "gpu_count" }, .text = "2" },
        .{ .path = &.{ "group", "hostname" }, .text = "\"private\"" },
        .{ .path = &.{"group"}, .text = null },
    };
    for (faults) |fault| {
        const v = try grouped(a, 4, true);
        try receipts.set(a, v, fault.path, fault.text);
        var why: []const u8 = "";
        try std.testing.expect(!check.receipt(v, .save, now_ms, &why));
        try std.testing.expect(why.len > 0);
    }
    const doc = try document(a, 4, "cuda");
    try doc.object.put(a, "rank_count", try json.intValue(a, @as(u8, 8)));
    try std.testing.expect((try manifest.parse(a, try json.stringify(a, doc, .{}), "cuda")) == null);
}

test "benchmark group: community-v1 keeps its frozen canonical bytes and original rank bound" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = @embedFile("fixtures/benchmark_v1.json");
    const v = (try json.parse(a, bytes)).ok;
    var why: []const u8 = "";
    try std.testing.expect(check.receipt(v, .publish, now_ms, &why));
    try std.testing.expectEqualStrings(bytes, try std.mem.concat(a, u8, &.{ try receipt.canonical(a, v), "\n" }));
    try receipts.set(a, v, &.{ "settings", "rank_count" }, "4");
    try std.testing.expect(!check.receipt(v, .save, now_ms, &why));
    try std.testing.expectEqualStrings("settings.rank_count must be 1, 2 or null", why);
    try receipts.set(a, v, &.{ "settings", "rank_count" }, "2");
    try std.testing.expect(check.receipt(v, .save, now_ms, &why));
    try v.object.put(a, "group", try topology(a, 2, false));
    try std.testing.expect(!check.receipt(v, .save, now_ms, &why));
    try std.testing.expectEqualStrings("receipt must hold exactly the community receipt fields", why);
}

test "benchmark group: environment counts cannot replace loaded or unavailable group facts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var env: std.process.Environ.Map = .init(a);
    try env.put("TF_NEMOTRON_WORLD", "8");
    try env.put("TF_BENCHMARK_RANK_COUNT", "2");
    const bytes = try json.stringify(a, try topology(a, 4, true), .{});
    const d = try manifest.serveDocument(a, std.testing.io, &env, "9.8.7", "cuda", "", .{ .ranks = 4, .group = bytes });
    try std.testing.expect(d.get("rank_count").?.int64() == 4);
    const loaded = (try manifest.parse(a, try json.stringify(a, d, .{}), "cuda")).?;
    try std.testing.expect(loaded.rank_count == 4 and loaded.group != null);
    for ([_]?u32{ null, 4 }) |n| {
        const missing = try manifest.serveDocument(a, std.testing.io, &env, "9.8.7", "cuda", "", .{ .ranks = n });
        const facts = (try manifest.parse(a, try json.stringify(a, missing, .{}), "cuda")).?;
        try std.testing.expect(facts.rank_count == null and facts.group == null);
    }
}

test "benchmark group: both managed and attached CLI runs copy the loaded server count despite client hints" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |managed| {
        const backend = if (managed) server.buildBackend() else "cuda";
        const bytes = try json.stringify(a, try document(a, 4, backend), .{});
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var r = try cli.Run.init(a, &tmp);
        try r.env.put("TF_NEMOTRON_WORLD", "8");
        try r.env.put("TF_BENCHMARK_RANK_COUNT", "1");
        const f = try fake.Fake.start(std.testing.allocator, try a.dupe(fake.Route, &.{
            .{ .method = "GET", .path = "/health", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"status\":\"ok\",\"warming\":false}")}) },
            .{ .method = "GET", .path = "/v1/models", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"data\":[{\"id\":\".tensorfold-benchmark\"}]}")}) },
            .{ .method = "GET", .path = "/v1/benchmark/manifest", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", bytes)}) },
            .{ .method = "POST", .path = "/v1/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, false, 4, 15)}) },
            .{ .method = "POST", .path = "/v1/chat/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, true, 4, 15)}) },
        }));
        defer f.stop();
        const out = try r.path("run.json");
        const args = if (managed) blk: {
            r.self_exe = try cli.standIn(&r, f.port);
            break :blk try a.dupe([]const u8, &.{ try r.path("private-checkpoint"), "--tokens", "4", "--reps", "1", "--temperatures", "0", "--output", out });
        } else try cli.runArgs(&r, try f.url(a, ""), out);
        try std.testing.expectEqual(@as(u8, 0), try r.command(args));
        var why: []const u8 = "";
        const v = (try receipt.load(a, std.testing.io, out, std.Io.Clock.real.now(std.testing.io).toMilliseconds(), &why)).?;
        try std.testing.expect(v.get("settings").?.get("rank_count").?.int64() == 4);
        try std.testing.expect(v.get("group").?.get("nodes").?.array.len == 4);
    }
}

test "benchmark group: managed and attached CLI runs drop legacy hints without loaded topology" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |managed| for ([_]u8{ 1, 2 }) |hint| {
        const backend = if (managed) server.buildBackend() else "cuda";
        const doc = try document(a, hint, backend);
        _ = doc.object.swapRemove("group");
        const bytes = try json.stringify(a, doc, .{});
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var r = try cli.Run.init(a, &tmp);
        try r.env.put("TF_NEMOTRON_WORLD", "8");
        try r.env.put("TF_BENCHMARK_RANK_COUNT", "1");
        const f = try fake.Fake.start(std.testing.allocator, try a.dupe(fake.Route, &.{
            .{ .method = "GET", .path = "/health", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"status\":\"ok\",\"warming\":false}")}) },
            .{ .method = "GET", .path = "/v1/models", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"data\":[{\"id\":\".tensorfold-benchmark\"}]}")}) },
            .{ .method = "GET", .path = "/v1/benchmark/manifest", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", bytes)}) },
            .{ .method = "POST", .path = "/v1/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, false, 4, 15)}) },
            .{ .method = "POST", .path = "/v1/chat/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, true, 4, 15)}) },
        }));
        defer f.stop();
        const out = try r.path("run.json");
        const args = if (managed) blk: {
            r.self_exe = try cli.standIn(&r, f.port);
            break :blk try a.dupe([]const u8, &.{ try r.path("private-checkpoint"), "--tokens", "4", "--reps", "1", "--temperatures", "0", "--output", out });
        } else try cli.runArgs(&r, try f.url(a, ""), out);
        try std.testing.expectEqual(@as(u8, 0), try r.command(args));
        var why: []const u8 = "";
        const v = (try receipt.load(a, std.testing.io, out, std.Io.Clock.real.now(std.testing.io).toMilliseconds(), &why)).?;
        try std.testing.expect(v.get("settings").?.get("rank_count").? == .null);
        try std.testing.expect(v.get("group") == null);
        if (!managed) try std.testing.expect(r.has("unranked"));
        try std.testing.expect(v.get("runtime").?.get("backend").?.str() != null);
    };
}

test "benchmark group: the unload lifetime lock copies inventory before a parked engine frees it" {
    var h: @import("idle_unload_test.zig").Harness = .{};
    defer h.deinit();
    try h.setup(10000);
    const a = std.testing.allocator;
    const bytes = try a.dupe(u8, "frozen-inventory");
    h.man.?.mutex.lockUncancelable(std.testing.io);
    h.srv.?.info.benchmark_ranks = 4;
    h.srv.?.info.benchmark_group = bytes;
    h.man.?.mutex.unlock(std.testing.io);
    const saved = try h.man.?.readBenchmark(h.srv.?, a);
    defer a.free(saved.group.?);
    h.man.?.mutex.lockUncancelable(std.testing.io);
    h.man.?.parked = true;
    a.free(bytes);
    h.man.?.mutex.unlock(std.testing.io);
    const parked = try h.man.?.readBenchmark(h.srv.?, a);
    try std.testing.expect(parked.ranks == null and parked.group == null);
    try std.testing.expectEqualStrings("frozen-inventory", saved.group.?);
    h.man.?.mutex.lockUncancelable(std.testing.io);
    h.srv.?.info.benchmark_group = null;
    h.man.?.parked = false;
    h.man.?.mutex.unlock(std.testing.io);
}
