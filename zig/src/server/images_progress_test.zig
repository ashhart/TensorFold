//! Live image SSE, borrowed preview ownership, unchanged final bytes and cancellation at step boundaries.
const std = @import("std");
const json = @import("json");
const h = @import("image_progress_test_harness.zig");
const t = std.testing;
const request = "{\"model\":\"m\",\"prompt\":\"hi\",\"negative_prompt\":\"fog\",\"steps\":5,\"seed\":7";

fn body(a: std.mem.Allocator, extra: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{s}{s}}}", .{ request, extra });
}
pub const Event = struct { kind: []const u8, value: json.Value };
pub fn events(a: std.mem.Allocator, raw: []const u8) ![]Event {
    var out: std.ArrayList(Event) = .empty;
    var it = std.mem.splitSequence(u8, raw, "\n\n");
    while (it.next()) |frame| {
        if (!std.mem.startsWith(u8, frame, "event: ")) continue;
        const end = std.mem.indexOfScalar(u8, frame, '\n').?;
        const at = std.mem.indexOf(u8, frame, "data: ").? + 6;
        try out.append(a, .{ .kind = frame[7..end], .value = (try json.parse(a, frame[at..])).ok });
    }
    return out.items;
}
pub fn decoded(a: std.mem.Allocator, value: json.Value) ![]const u8 {
    const encoded = value.strField("b64_json").?;
    const bytes = try a.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(encoded));
    try std.base64.standard.Decoder.decode(bytes, encoded);
    return bytes;
}
fn completed(list: []const Event) !json.Value {
    for (list) |event| if (std.mem.eql(u8, event.kind, "image_generation.completed")) return event.value;
    return error.MissingCompleted;
}
fn stableBody(a: std.mem.Allocator, raw: []const u8) ![]const u8 {
    const parsed = (try json.parse(a, raw)).ok;
    try parsed.object.put(a, "created", try json.intValue(a, 0));
    return json.stringify(a, parsed, .{});
}

test "image progress: every step is written before finished and the final PNG equals the plain request" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var plain: h.Fake = .{};
    const solo = try h.ask(a, &plain, try body(a, ""), null, false);
    try t.expectEqual(@as(u16, 200), solo.status);
    var fake: h.Fake = .{};
    const streamed = try h.ask(a, &fake, try body(a, ",\"stream\":true"), null, false);
    try t.expectEqual(@as(u16, 200), streamed.status);
    try t.expect(std.mem.indexOf(u8, streamed.wire, "text/event-stream") != null);
    const list = try events(a, streamed.body);
    try t.expectEqual(@as(usize, 6), list.len);
    for (list) |event| try t.expectEqualStrings(event.kind, event.value.strField("type").?);
    for (list[0..5], 1..) |event, step| {
        try t.expectEqualStrings("image_generation.progress", event.kind);
        try t.expectEqual(@as(i64, @intCast(step)), event.value.get("step").?.int64().?);
        try t.expectEqual(@as(i64, 5), event.value.get("total_steps").?.int64().?);
        try t.expectEqual(@as(i64, 7), event.value.get("seed").?.int64().?);
        try t.expectEqual(@as(i64, 0), event.value.get("index").?.int64().?);
    }
    const legacy = (try json.parse(a, solo.body)).ok.get("data").?.array[0];
    try t.expectEqualSlices(u8, try decoded(a, legacy), try decoded(a, try completed(list)));
    try t.expect(fake.live.load(.acquire) and !fake.gate_timeout.load(.acquire));
    try t.expectEqual(@as(u32, 0), fake.preview_reads.load(.acquire));
    try t.expect(std.mem.endsWith(u8, streamed.body, "data: [DONE]\n\n"));
}

test "image progress: requested previews are copied at the interval and preserve the final PNG bytes" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var solo: h.Fake = .{};
    const plain = try h.ask(a, &solo, try body(a, ""), null, false);
    var fake: h.Fake = .{};
    const out = try h.ask(a, &fake, try body(a, ",\"stream\":true,\"preview_interval\":2"), null, false);
    try t.expectEqual(@as(u16, 200), out.status);
    const list = try events(a, out.body);
    var previews: u32 = 0;
    for (list) |event| if (std.mem.eql(u8, event.kind, "image_generation.partial_image")) {
        previews += 1;
        const bytes = try decoded(a, event.value);
        try t.expectEqualStrings("\x89PNG\r\n\x1a\n", bytes[0..8]);
        try t.expectEqual(previews * 2, std.mem.readInt(u32, bytes[8..12], .little));
        try t.expectEqual(@as(i64, previews * 2), event.value.get("step").?.int64().?);
        try t.expectEqual(@as(i64, previews - 1), event.value.get("partial_image_index").?.int64().?);
        try t.expectEqual(@as(i64, 64), event.value.get("width").?.int64().?);
    };
    try t.expectEqual(@as(u32, 2), previews);
    try t.expectEqual(previews, fake.preview_reads.load(.acquire));
    const legacy = (try json.parse(a, plain.body)).ok.get("data").?.array[0];
    try t.expectEqualSlices(u8, try decoded(a, legacy), try decoded(a, try completed(list)));
}

test "image progress: no stream retains the whole JSON shape, explicit seed and no observer work" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var first: h.Fake = .{};
    var second: h.Fake = .{};
    const raw = "{\"model\":\"m\",\"prompt\":\"hi\",\"steps\":5,\"seed\":42}";
    const one = try h.ask(a, &first, raw, null, false);
    const two = try h.ask(a, &second, "{\"model\":\"m\",\"prompt\":\"hi\",\"steps\":5,\"seed\":42,\"stream\":false}", null, false);
    try t.expectEqual(@as(u16, 200), one.status);
    try t.expectEqual(@as(u16, 200), two.status);
    try t.expectEqualStrings(try stableBody(a, one.body), try stableBody(a, two.body));
    try t.expectEqual(@as(u64, 42), first.seed);
    try t.expect(!first.observed and !second.observed);
    try t.expectEqual(@as(u32, 0), first.preview_reads.load(.acquire));
    try t.expect(std.mem.indexOf(u8, one.wire, "event-stream") == null);
}

test "image progress: malformed fields and unsupported engine capabilities refuse before submission" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{
        ",\"stream\":1",                           ",\"preview_interval\":2",                   ",\"stream\":true,\"preview_interval\":0",
        ",\"stream\":true,\"preview_interval\":6", ",\"stream\":true,\"preview_interval\":1.5",
    }) |extra| {
        var fake: h.Fake = .{};
        const out = try h.ask(a, &fake, try body(a, extra), null, false);
        try t.expectEqual(@as(u16, 400), out.status);
        try t.expectEqual(@as(usize, 0), fake.calls);
    }
    for ([_]h.Fake{ .{ .progress = false }, .{ .previews = false } }) |initial| {
        var fake = initial;
        const out = try h.ask(a, &fake, try body(a, ",\"stream\":true,\"preview_interval\":1"), null, false);
        try t.expectEqual(@as(u16, 400), out.status);
        try t.expect(std.mem.indexOf(u8, out.body, "not supported") != null);
        try t.expectEqual(@as(usize, 0), fake.calls);
    }
}

test "image progress: a failed write cancels at the next step and the next request is unchanged" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |ignore| {
        var fake: h.Fake = .{ .ignore_cancel = ignore };
        const out = try h.ask(a, &fake, try body(a, ",\"stream\":true"), 1, false);
        try t.expectEqual(@as(u16, 200), out.status);
        try t.expectEqual(@as(usize, 1), fake.cancels);
        try t.expect(fake.finished.load(.acquire) and fake.steps_done.load(.acquire) <= 2);
        try t.expect(std.mem.indexOf(u8, out.body, "image_generation.completed") == null);
        const resumed = try h.ask(a, &fake, try body(a, ""), null, false);
        var fresh: h.Fake = .{};
        const reference = try h.ask(a, &fresh, try body(a, ""), null, false);
        try t.expectEqualStrings(try stableBody(a, reference.body), try stableBody(a, resumed.body));
    }
}

test "image progress: a departed reader cancels at the next step and releases the observer" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: h.Fake = .{ .leave_after = 1 };
    const out = try h.ask(a, &fake, try body(a, ",\"stream\":true,\"preview_interval\":1"), null, false);
    try t.expectEqual(@as(usize, 1), fake.cancels);
    try t.expect(fake.finished.load(.acquire) and fake.steps_done.load(.acquire) <= 2);
    try t.expect(std.mem.indexOf(u8, out.body, "image_generation.completed") == null);
}

test "image progress: failures after headers are SSE errors; submission refusals retain HTTP 400" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var failed: h.Fake = .{ .fail_step = 3 };
    const out = try h.ask(a, &failed, try body(a, ",\"stream\":true"), null, false);
    try t.expectEqual(@as(u16, 200), out.status);
    try t.expect(std.mem.indexOf(u8, out.body, "event: error\n") != null);
    try t.expect(std.mem.indexOf(u8, out.body, "fake denoising failed") != null);
    try t.expect(std.mem.indexOf(u8, out.body, "image_generation.completed") == null);
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, out.wire, "HTTP/1.1"));
    var rejected: h.Fake = .{ .reject = error.InvalidRequest };
    const before = try h.ask(a, &rejected, try body(a, ",\"stream\":true"), null, false);
    try t.expectEqual(@as(u16, 400), before.status);
    try t.expect(std.mem.indexOf(u8, before.wire, "event-stream") == null);
}

test "image progress: invalid engine steps, missing previews and over-limit previews fail without a final image" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]h.Fake{ .{ .invalid_step = true }, .{ .missing_preview = true }, .{ .oversized_preview = true }, .{ .early_stop = 2 } }) |initial| {
        var fake = initial;
        const out = try h.ask(a, &fake, try body(a, ",\"stream\":true,\"preview_interval\":1"), null, false);
        try t.expectEqual(@as(u16, 200), out.status);
        try t.expect(std.mem.indexOf(u8, out.body, "event: error\n") != null);
        try t.expect(std.mem.indexOf(u8, out.body, "image_generation.completed") == null);
        try t.expect(fake.finished.load(.acquire));
    }
}

test "image progress: a full 200-step queue preserves every progress and preview while the reader catches up" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: h.Fake = .{ .gate = false, .inline_ = true };
    const out = try h.ask(a, &fake, "{\"model\":\"m\",\"prompt\":\"hi\",\"steps\":200,\"stream\":true,\"preview_interval\":1}", null, false);
    try t.expectEqual(@as(u16, 200), out.status);
    const list = try events(a, out.body);
    try t.expectEqual(@as(usize, 401), list.len);
    try t.expectEqual(@as(u32, 200), fake.preview_reads.load(.acquire));
    _ = try completed(list);
}

test "image progress: a failed stream header still cancels and settles before freeing request storage" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: h.Fake = .{};
    const out = try h.ask(a, &fake, try body(a, ",\"stream\":true"), null, true);
    try t.expectEqual(@as(u16, 0), out.status);
    try t.expectEqual(@as(usize, 1), fake.cancels);
    try t.expect(fake.finished.load(.acquire) and fake.steps_done.load(.acquire) <= 2);
}
