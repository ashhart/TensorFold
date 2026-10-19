//! The actual benchmark command routes only declared chat-only checkpoints and records every attempt's route.
const std = @import("std");
const json = @import("json");
const cli = @import("benchmark_cli_test.zig");
const fake = @import("benchmark_fake.zig");
const suite = @import("benchmark_suite.zig");

fn read(r: *cli.Run, path: []const u8) !json.Value {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, r.a, .limited(1 << 20));
    return (try json.parse(r.a, bytes)).ok;
}
fn routes(r: *cli.Run, f: *fake.Fake, path: []const u8, chat_only: bool) !void {
    try std.testing.expectEqual(@as(usize, if (chat_only) 0 else 2), cli.posts(f, "/v1/completions"));
    try std.testing.expectEqual(@as(usize, if (chat_only) 4 else 2), cli.posts(f, "/v1/chat/completions"));
    const v = try read(r, path);
    try std.testing.expectEqualStrings(suite.id, v.get("suite").?.get("id").?.string);
    for (v.get("samples").?.array) |s| {
        const code = std.mem.eql(u8, s.get("fixture_id").?.string, "code");
        try std.testing.expect(s.get("route") != null);
        try std.testing.expectEqualStrings(if (code and !chat_only) "/v1/completions" else "/v1/chat/completions", s.get("route").?.string);
    }
    if (chat_only) {
        var code_requests: usize = 0;
        for (0..f.count()) |i| {
            const req = f.request(i);
            if (!std.mem.eql(u8, req.method, "POST")) continue;
            const body = (try json.parse(r.a, req.body)).ok;
            const messages = body.get("messages").?.array;
            try std.testing.expectEqual(@as(usize, 1), messages.len);
            try std.testing.expectEqualStrings("user", messages[0].get("role").?.string);
            try std.testing.expect(body.get("prompt") == null);
            if (std.mem.eql(u8, messages[0].get("content").?.string, suite.fixtures[0].prompt)) code_requests += 1;
        }
        try std.testing.expectEqual(@as(usize, 2), code_requests);
    }
}

test "benchmark routing: attached server needs its explicit chat-only flag" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ true, false }) |chat_only| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var r = try cli.Run.init(a, &tmp);
        const f = try cli.world(a, false, "202 Accepted");
        defer f.stop();
        const out = try r.path("run.json");
        const argv = try cli.runArgs(&r, try f.url(a, ""), out);
        try std.testing.expectEqual(@as(u8, 0), try r.command(if (chat_only) try std.mem.concat(a, []const u8, &.{ argv, &.{"--chat-only"} }) else argv));
        try routes(&r, f, out, chat_only);
    }
}

test "benchmark routing: managed run uses the matched native family declaration" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ true, false }) |chat_only| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var r = try cli.Run.init(a, &tmp);
        var families = [_]@import("engine_api").Family{.{ .model_type = "qwen3_5", .formats = &.{}, .raw_text = !chat_only }};
        r.families = &families;
        const f = try cli.managedWorld(a, false);
        defer f.stop();
        _ = try cli.standIn(&r, f.port);
        r.self_exe = try r.path("serve-stand-in");
        const out = try r.path("run.json");
        const status = try r.command(&.{ try r.path("private-checkpoint"), "--tokens", "4", "--reps", "1", "--temperatures", "0", "--output", out });
        if (status != 0) std.debug.print("managed fixture: {s}\n", .{r.err.written()});
        try std.testing.expectEqual(@as(u8, 0), status);
        try routes(&r, f, out, chat_only);
    }
}

fn empty(a: std.mem.Allocator) ![]const fake.Write {
    const content = "data: {\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":14,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n";
    return a.dupe(fake.Write, &.{.{ .bytes = try std.mem.concat(a, u8, &.{ try fake.headOf(a, "200 OK", "text/event-stream"), content }) }});
}

test "benchmark routing: empty output never guesses a different route or becomes a success" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |chat_only| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var r = try cli.Run.init(a, &tmp);
        const bad = try empty(a);
        const good = try fake.reply(a, true, 4, 15);
        const f = try fake.Fake.start(std.testing.allocator, try a.dupe(fake.Route, &.{
            .{ .method = "POST", .path = "/v1/completions", .replies = try a.dupe([]const fake.Write, &.{bad}) },
            .{ .method = "POST", .path = "/v1/chat/completions", .replies = if (chat_only) try a.dupe([]const fake.Write, &.{ bad, bad, good }) else try a.dupe([]const fake.Write, &.{good}) },
        }));
        defer f.stop();
        const out = try r.path("run.json");
        const argv = try cli.runArgs(&r, try f.url(a, ""), out);
        try std.testing.expectEqual(@as(u8, 2), try r.command(if (chat_only) try std.mem.concat(a, []const u8, &.{ argv, &.{"--chat-only"} }) else argv));
        try std.testing.expectEqual(@as(usize, if (chat_only) 0 else 2), cli.posts(f, "/v1/completions"));
        try std.testing.expectEqual(@as(usize, if (chat_only) 4 else 2), cli.posts(f, "/v1/chat/completions"));
        const v = try read(&r, out);
        var failures: usize = 0;
        for (v.get("samples").?.array) |s| if (std.mem.eql(u8, s.get("fixture_id").?.string, "code")) {
            failures += 1;
            try std.testing.expectEqualStrings("error", s.get("status").?.string);
            try std.testing.expectEqualStrings("empty_output", s.get("error_code").?.string);
        };
        try std.testing.expectEqual(@as(usize, 2), failures);
    }
}

test "benchmark routing: v1 stays raw and its frozen receipt has no route field" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const legacy = suite.requestFixture(suite.legacy_id, suite.fixtures[0], true);
    try std.testing.expect(!legacy.chat);
    try std.testing.expectEqualStrings("/v1/completions", suite.route(legacy));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try cli.Run.init(a, &tmp);
    const path = try r.path("v1.json");
    const bytes = @embedFile("fixtures/benchmark_v1.json");
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = bytes });
    const value = (try json.parse(a, bytes)).ok;
    for (value.get("samples").?.array) |sample| try std.testing.expect(sample.get("route") == null);
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "inspect", path }));
    try std.testing.expect(r.has(suite.legacy_id));
}

test "benchmark routing: attached-only flag does not bypass the local-only publication guard" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try cli.Run.init(a, &tmp);
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "example-org/test-model", "--chat-only" }));
    try std.testing.expect(r.has("--chat-only applies only"));
    const f = try cli.world(a, false, "202 Accepted");
    defer f.stop();
    const path = try r.path("never.json");
    const args = try std.mem.concat(a, []const u8, &.{ try cli.runArgs(&r, try f.url(a, ""), path), &.{ "--chat-only", "--publish" } });
    try std.testing.expectEqual(@as(u8, 1), try r.command(args));
    try std.testing.expect(r.has(@import("benchmark_upload.zig").v2_disabled));
    try std.testing.expectEqual(@as(usize, 0), f.count());
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().statFile(std.testing.io, path, .{}));
}

test "benchmark routing: receipt checks reject unknown, inconsistent and wrong-workload routes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const check = @import("benchmark_check.zig");
    const value = try @import("benchmark_receipt_test.zig").valid(a);
    const samples = value.get("samples").?.array;
    var why: []const u8 = "";
    try std.testing.expect(check.receipt(value, .save, 1767225660000, &why));
    try samples[0].object.put(a, "route", .{ .string = "/unknown" });
    try std.testing.expect(!check.receipt(value, .save, 1767225660000, &why));
    try samples[0].object.put(a, "route", .{ .string = "/v1/completions" });
    try samples[1].object.put(a, "route", .{ .string = "/v1/completions" });
    try std.testing.expect(!check.receipt(value, .save, 1767225660000, &why));
    try samples[1].object.put(a, "route", .{ .string = "/v1/chat/completions" });
    const warm = (try json.parse(a, try json.stringify(a, samples[0], .{}))).ok;
    try warm.object.put(a, "repeat", try json.intValue(a, -1));
    try warm.object.put(a, "status", .{ .string = "error" });
    try warm.object.put(a, "error_code", .{ .string = "server_error" });
    const all = try a.alloc(json.Value, 3);
    @memcpy(all[0..2], samples);
    all[2] = warm;
    try value.object.put(a, "samples", .{ .array = all });
    try std.testing.expect(check.receipt(value, .save, 1767225660000, &why));
    try warm.object.put(a, "route", .{ .string = "/v1/chat/completions" });
    try std.testing.expect(!check.receipt(value, .save, 1767225660000, &why));
    try std.testing.expect(std.mem.indexOf(u8, why, "same route") != null);
}
