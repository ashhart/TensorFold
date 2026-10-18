//! A sampled repeat that stops early stays in its receipt and explains the incomplete exit.
const std = @import("std");
const json = @import("json");
const fake = @import("benchmark_fake.zig");
const cli = @import("benchmark_cli_test.zig");
const suite = @import("benchmark_suite.zig");

fn stopped(a: std.mem.Allocator, empty: bool) ![]const fake.Write {
    const writes = try a.alloc(fake.Write, 3);
    const first = if (empty) "" else try fake.event(a, "{\"choices\":[{\"index\":0,\"text\":\"short\"}]}");
    const second = if (empty) "" else try fake.event(a, "{\"choices\":[{\"index\":0,\"text\":\" reply\"}]}");
    writes[0] = .{ .bytes = try std.mem.concat(a, u8, &.{ try fake.headOf(a, "200 OK", "text/event-stream"), first }), .pause_ms = 15 };
    writes[1] = .{ .bytes = second };
    const final = "{\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"stop\"}],\"tensorfold\":{\"token_sha\":\"54363ddee68f\"}}";
    const usage = try std.fmt.allocPrint(a, "{{\"choices\":[],\"usage\":{{\"prompt_tokens\":24,\"completion_tokens\":{d}}}}}", .{@as(u64, if (empty) 1 else 105)});
    writes[2] = .{ .bytes = try std.mem.concat(a, u8, &.{ try fake.event(a, final), try fake.event(a, usage), try fake.event(a, "[DONE]") }) };
    return writes;
}

test "issue 969: sampled early stops keep counts and seeds, suppress aggregates and explain exit 2" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]enum { empty, short, full }{ .empty, .short, .full }) |case| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var r = try cli.Run.init(a, &tmp);
        const ok = try fake.reply(a, false, 256, 15);
        const middle = if (case == .full) ok else try stopped(a, case == .empty);
        const f = try fake.Fake.start(std.testing.allocator, try a.dupe(fake.Route, &.{
            .{ .method = "POST", .path = "/v1/completions", .replies = try a.dupe([]const fake.Write, &.{ ok, ok, middle, ok }) },
            .{ .method = "POST", .path = "/v1/chat/completions", .replies = try a.dupe([]const fake.Write, &.{try fake.reply(a, true, 256, 15)}) },
        }));
        defer f.stop();
        const out = try r.path("run.json");
        const code = try r.command(&.{ "example-org/test-model", "--server", try f.url(a, ""), "--backend", "cuda", "--tokens", "256", "--reps", "3", "--temperatures", "1", "--output", out });
        try std.testing.expectEqual(@as(u8, if (case == .full) 0 else 2), code);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, out, a, .limited(1 << 20));
        const v = (try json.parse(a, bytes)).ok;
        const samples = v.get("samples").?.array;
        try std.testing.expectEqual(@as(usize, 6), samples.len);
        const failed = samples[1];
        try std.testing.expectEqual(@as(?i64, 1), failed.get("repeat").?.int64());
        try std.testing.expectEqual(@as(?i64, 1235), failed.get("seed").?.int64());
        try std.testing.expectEqual(@as(?i64, if (case == .empty) 1 else if (case == .short) 105 else 256), failed.get("completion_tokens").?.int64());
        try std.testing.expectEqualStrings(if (case == .empty) "error" else if (case == .short) "early_eos" else "ok", failed.get("status").?.string);
        if (case == .empty) {
            try std.testing.expectEqualStrings("empty_output", failed.get("error_code").?.string);
            try std.testing.expectEqualStrings("54363ddee68f", failed.get("token_sha").?.string);
            try std.testing.expect(failed.get("ttft_seconds").? == .null and failed.get("delivery_tps").? == .null);
        } else try std.testing.expect(failed.get("error_code").? == .null);
        const cells = try suite.cells(a, samples, 3);
        try std.testing.expectEqual(case == .full, cells[0].complete);
        try std.testing.expectEqual(case == .full, cells[0].delivery != null and cells[0].ttft != null);
        try std.testing.expect(cells[1].complete and cells[1].delivery != null);
        var completion: usize = 0;
        const measured = for (0..f.count()) |i| {
            const sent = f.request(i);
            if (!std.mem.eql(u8, sent.path, "/v1/completions")) continue;
            if (completion == 2) break sent;
            completion += 1;
        } else return error.MissingMeasuredRequest;
        const request = (try json.parse(a, measured.body)).ok;
        try std.testing.expectEqual(@as(?i64, 1235), request.get("seed").?.int64());
        try std.testing.expectEqual(@as(?i64, 20), request.get("top_k").?.int64());
        try std.testing.expectEqual(@as(?f64, 0.95), suite.number(request.get("top_p")));
        try std.testing.expect(r.has("Receipt saved:"));
        const explanation = "Incomplete attempts remain in the saved receipt (exit 2). A cell has aggregates only when every repetition succeeds; early EOS is excluded from full-length comparisons.";
        try std.testing.expectEqual(@as(usize, if (case == .full) 0 else 1), std.mem.count(u8, r.out.written(), explanation));
    }
}
