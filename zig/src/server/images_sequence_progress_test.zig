//! The image count and streaming changes together, including a failure or disconnect after one complete picture.
const std = @import("std");
const json = @import("json");
const h = @import("image_progress_test_harness.zig");
const frames = @import("images_progress_test.zig");
const t = std.testing;
const seed: u64 = 9007199254740993;

fn request(a: std.mem.Allocator, count: usize, chosen: u64, extra: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":\"hi\",\"negative_prompt\":\"fog\",\"steps\":5,\"guidance\":2,\"size\":\"512x768\",\"n\":{d},\"seed\":{d}{s}}}", .{ count, chosen, extra });
}

test "image sequence progress: four streamed PNGs equal whole and solo requests with ordered seeds and steps" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var whole: h.Fake = .{};
    const plain = try h.ask(a, &whole, try request(a, 4, seed, ""), null, false);
    try t.expectEqual(@as(u16, 200), plain.status);
    const data = (try json.parse(a, plain.body)).ok.get("data").?.array;
    try t.expectEqual(@as(usize, 4), data.len);
    var streamed: h.Fake = .{};
    const out = try h.ask(a, &streamed, try request(a, 4, seed, ",\"stream\":true,\"preview_interval\":2"), null, false);
    try t.expectEqual(@as(u16, 200), out.status);
    const list = try frames.events(a, out.body);
    try t.expectEqual(@as(usize, 32), list.len);
    var index: usize = 0;
    var step: i64 = 0;
    var previews: usize = 0;
    for (list) |event| {
        try t.expectEqual(@as(i64, @intCast(index)), event.value.get("index").?.int64().?);
        try t.expectEqual(@as(i64, @intCast(seed + index)), event.value.get("seed").?.int64().?);
        if (std.mem.eql(u8, event.kind, "image_generation.progress")) {
            step += 1;
            try t.expectEqual(step, event.value.get("step").?.int64().?);
            try t.expectEqual(@as(i64, 5), event.value.get("total_steps").?.int64().?);
        } else if (std.mem.eql(u8, event.kind, "image_generation.partial_image")) {
            previews += 1;
            try t.expect(step == 2 or step == 4);
            try t.expectEqual(step, event.value.get("step").?.int64().?);
        } else {
            try t.expectEqualStrings("image_generation.completed", event.kind);
            try t.expectEqual(@as(i64, 5), step);
            try t.expectEqualSlices(u8, try frames.decoded(a, data[index]), try frames.decoded(a, event.value));
            var solo: h.Fake = .{};
            const single = try h.ask(a, &solo, try request(a, 1, seed + index, ""), null, false);
            try t.expectEqual(@as(u16, 200), single.status);
            const item = (try json.parse(a, single.body)).ok.get("data").?.array[0];
            try t.expectEqualSlices(u8, try frames.decoded(a, item), try frames.decoded(a, event.value));
            index += 1;
            step = 0;
        }
    }
    try t.expectEqual(@as(usize, 4), index);
    try t.expectEqual(@as(usize, 8), previews);
    try t.expectEqual(@as(usize, 4), streamed.calls);
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, out.wire, "HTTP/1.1"));
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, out.body, "data: [DONE]"));
    try t.expect(!streamed.gate_timeout.load(.acquire));
}

test "image sequence progress: second-picture refusal preserves only the first completed image and stops" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: h.Fake = .{ .reject = error.Busy, .reject_at = 2 };
    const out = try h.ask(a, &fake, try request(a, 4, seed, ",\"stream\":true"), null, false);
    try t.expectEqual(@as(u16, 200), out.status);
    try t.expectEqual(@as(usize, 2), fake.calls);
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, out.wire, "HTTP/1.1"));
    const list = try frames.events(a, out.body);
    var complete: usize = 0;
    for (list) |event| {
        if (std.mem.eql(u8, event.kind, "image_generation.completed")) {
            complete += 1;
            try t.expectEqual(@as(i64, 0), event.value.get("index").?.int64().?);
        }
    }
    try t.expectEqual(@as(usize, 1), complete);
    try t.expect(std.mem.indexOf(u8, out.body, "the engine is busy") != null);
    try t.expect(std.mem.endsWith(u8, out.body, "data: [DONE]\n\n"));
}

test "image sequence progress: disconnect during picture two cancels at its next step and never submits three" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: h.Fake = .{};
    const out = try h.askAt(a, &fake, try request(a, 4, seed, ",\"stream\":true,\"preview_interval\":1"), 1, false, 1);
    try t.expectEqual(@as(u16, 200), out.status);
    try t.expectEqual(@as(usize, 2), fake.calls);
    try t.expectEqual(@as(usize, 1), fake.cancels);
    try t.expect(fake.finished.load(.acquire) and fake.steps_done.load(.acquire) <= 2);
    const list = try frames.events(a, out.body);
    var complete: usize = 0;
    for (list) |event| if (std.mem.eql(u8, event.kind, "image_generation.completed")) {
        complete += 1;
        try t.expectEqual(@as(i64, 0), event.value.get("index").?.int64().?);
    };
    try t.expectEqual(@as(usize, 1), complete);
}
