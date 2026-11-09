//! Pins and observed request counts survive reopen, and capacity uses the real writer, rounds and delete owners.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const graph = @import("slide_graph.zig");
const usage = @import("slide_usage.zig");
const route = @import("slide_pin.zig");
const path = @import("slide_learn_path_test.zig");
const Server = @import("server.zig").Server;
const t = std.testing;

fn add(g: *graph.Graph, layer: u16) u32 {
    const id = g.add("A fictional fact.", "chat", 1);
    g.mark(id, .learned);
    g.placed(id, &.{ .{ .layer = layer, .neuron = 0, .key = .reply }, .{ .layer = layer, .neuron = 1, .key = .subject } });
    return id;
}

test "fact usage: a request counts once per fact and pins and full-width counts survive reopen" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const file = try std.fs.path.join(t.allocator, &.{ dir, "graph.json" });
    defer t.allocator.free(file);
    var g = graph.Graph.init(t.allocator, t.io, file);
    defer g.deinit();
    const a = add(&g, 1);
    const b = add(&g, 2);
    try g.used(&.{ .{ .layer = 1, .neuron = 0, .key = .reply }, .{ .layer = 1, .neuron = 1, .key = .subject }, .{ .layer = 1, .neuron = 0, .key = .reply } });
    try t.expectEqual(@as(u64, 1), g.find(a).?.uses);
    try t.expectEqual(@as(u64, 0), g.find(b).?.uses);
    {
        var observed = graph.Graph.init(t.allocator, t.io, file);
        defer observed.deinit();
        try t.expectEqual(@as(u64, 1), observed.find(a).?.uses);
        try t.expectEqual(@as(u64, 0), observed.find(b).?.uses);
    }
    g.find(a).?.uses = @as(u64, 1) << 63;
    try g.pin(a, true);
    var back = graph.Graph.init(t.allocator, t.io, file);
    defer back.deinit();
    try t.expect(back.find(a).?.pinned);
    try t.expectEqual(@as(u64, 1) << 63, back.find(a).?.uses);
    try back.used(&.{.{ .layer = 1, .neuron = 0, .key = .reply }});
    try t.expectEqual((@as(u64, 1) << 63) + 1, back.find(a).?.uses);
    try back.pin(a, false);
    var last = graph.Graph.init(t.allocator, t.io, file);
    defer last.deinit();
    try t.expect(!last.find(a).?.pinned and last.find(a).?.uses == back.find(a).?.uses);
    try last.pin(a, true);
    last.phase(a, .writing);
    try t.expect(last.find(a).?.pinned and last.find(a).?.uses == back.find(a).?.uses);
}

test "fact usage: pin write failure restores the pin and a pending observation prevents an unwritten eviction" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const file = try std.fs.path.join(t.allocator, &.{ dir, "graph.json" });
    defer t.allocator.free(file);
    var g = graph.Graph.init(t.allocator, t.io, file);
    defer g.deinit();
    const n = add(&g, 1);
    try g.pin(n, true);
    g.fault = .rename;
    try t.expectError(error.GraphRename, g.pin(n, false));
    try t.expect(g.find(n).?.pinned);
    g.fault = .write;
    try t.expectError(error.GraphWrite, g.used(&.{.{ .layer = 1, .neuron = 0, .key = .reply }}));
    try t.expect(g.usage_dirty and g.find(n).?.uses == 1);
    g.fault = .write;
    try t.expectError(error.GraphWrite, g.flushUsage());
    try t.expect(g.usage_dirty);
    try g.flushUsage();
    try t.expect(!g.usage_dirty);
    var back = graph.Graph.init(t.allocator, t.io, file);
    defer back.deinit();
    try t.expect(back.find(n).?.pinned and back.find(n).?.uses == 1);
}

test "fact usage: pins beat usage and used facts outlive unused facts, with deterministic ties" {
    const ns = [_]api.Neuron{.{ .layer = 1, .neuron = 0, .key = .reply }};
    var fs = [_]graph.Fact{
        .{ .id = 1, .text = "pinned", .source = "chat", .at = 0, .state = .learned, .pinned = true, .neurons = &ns },
        .{ .id = 2, .text = "used", .source = "chat", .at = 0, .state = .learned, .uses = 3, .neurons = &ns },
        .{ .id = 3, .text = "unused", .source = "chat", .at = 0, .state = .learned, .neurons = &ns },
    };
    try t.expectEqual(@as(?u32, 3), usage.least(&fs, 0));
    fs[2].uses = 3;
    try t.expectEqual(@as(?u32, 2), usage.least(&fs, 0));
    fs[1].pinned = true;
    try t.expectEqual(@as(?u32, null), usage.least(&fs, 3));
    fs[2].state = .deleting;
    try t.expectEqual(@as(?u32, null), usage.least(&fs, 0));
}

test "fact usage: pin input is explicit and malformed graph metadata cannot become a zero count" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectEqual(@as(u32, 7), try route.id("/v1/slide/facts/7/pin"));
    try t.expectError(error.BadId, route.id("/v1/slide/facts/0/pin"));
    try t.expectError(error.BadId, route.id("/v1/slide/facts/pin"));
    try t.expectError(error.BadId, route.id("/v1/slide/facts//pin"));
    try t.expectError(error.BadId, route.id("/v1/slide/facts/1/2/pin"));
    try t.expect(try route.value((try json.parse(a, "{}")).ok));
    try t.expect(!try route.value((try json.parse(a, "{\"pinned\":false}")).ok));
    var f: graph.Fact = .{ .id = 1, .text = "x", .source = "chat", .at = 0, .state = .learned };
    try t.expectError(error.BadUsage, usage.read(&f, (try json.parse(a, "{\"uses\":-1}")).ok));
    try t.expectError(error.BadPin, usage.read(&f, (try json.parse(a, "{\"pinned\":1}")).ok));
}

fn fullCase(enabled: bool, pin_both: bool, pin_first: bool, measured: bool, unsafe_cut: bool) !void {
    var text: path.Text = .{};
    var m: path.Model = .{ .capacity = true, .pocket_audit = measured, .capacity_break = unsafe_cut };
    defer m.deinit();
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .slide_forget = enabled }, null);
    defer srv.deinit();
    try path.learnOne(srv, "I like blue.", null);
    try path.learnOne(srv, "I like apples.", null);
    try t.expectEqual(@as(usize, 2), srv.slide.facts.items.len);
    for (srv.slide.facts.items) |f| try t.expect(f.state == .learned and f.neurons.len > 0);
    try @import("slide_usage_live_test.zig").askExternal(srv, false);
    if (pin_both or pin_first) try srv.slide.pin(1, true);
    if (pin_both) try srv.slide.pin(2, true);
    try path.learnOne(srv, "I like tea.", null);
    if (enabled and !pin_both and measured and !unsafe_cut) {
        try t.expectEqual(@as(usize, 1), m.cuts);
        try t.expectEqual([3]bool{ true, false, true }, m.live);
        try t.expectEqual(graph.State.deleted, srv.slide.find(2).?.state);
        try t.expectEqual(@as(u64, 1), srv.slide.find(1).?.uses);
        try t.expectEqual(graph.State.learned, srv.slide.find(3).?.state);
    } else {
        try t.expectEqual(@as(usize, if (unsafe_cut) 1 else 0), m.cuts);
        try t.expectEqual([3]bool{ true, true, false }, m.live);
        try t.expectEqual(graph.State.missed, srv.slide.find(3).?.state);
    }
    if (pin_first) try t.expect(srv.slide.find(1).?.pinned and srv.slide.find(1).?.state == .learned);
}

test "fact usage: full lesson evicts unused before used through the exact delete path and rewrites its lesson" {
    try fullCase(true, false, false, true, false);
}
test "fact usage: full lesson preserves a pin and refuses if all facts are pinned" {
    try fullCase(true, false, true, true, false);
    try fullCase(true, true, true, true, false);
}
test "fact usage: forgetting off retains the full-pocket refusal and deletes nothing" {
    try fullCase(false, false, false, true, false);
}

test "fact usage: an engine without measured activity cannot make used facts look unused" {
    try fullCase(true, false, false, false, false);
}

test "fact usage: a pinned fact cannot be retired by the replacement judge" {
    var text: path.Text = .{};
    var m: path.Model = .{ .capacity = true, .force_replace = true };
    defer m.deinit();
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    try path.learnOne(srv, "I like blue.", null);
    try srv.slide.pin(1, true);
    try path.learnOne(srv, "I like apples.", null);
    try t.expect(srv.slide.find(1).?.pinned and srv.slide.find(1).?.state == .learned);
    try t.expectEqual([3]bool{ true, true, false }, m.live);
}

test "fact usage: a full pocket cannot forget an unused fact if the checked cut disturbs a kept fact" {
    try fullCase(true, false, false, true, true);
}

test "fact usage: failed graph loading cannot supply trusted zero counts for forgetting" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const file = try std.fs.path.join(t.allocator, &.{ dir, "graph.json" });
    defer t.allocator.free(file);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "graph.json", .data = "{\"next\":2,\"facts\":[{\"id\":1,\"at\":0,\"text\":\"x\",\"state\":\"learned\",\"uses\":-1}]}" });
    var g = graph.Graph.init(t.allocator, t.io, file);
    defer g.deinit();
    try t.expect(g.usage_invalid);
    try t.expectError(error.BadGraph, g.flushUsage());
}

test "fact usage: a still-full retry cannot drain a second fact" {
    var text: path.Text = .{};
    var m: path.Model = .{ .capacity = true, .capacity_stuck = true };
    defer m.deinit();
    const srv = try Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .slide_forget = true }, null);
    defer srv.deinit();
    try path.learnOne(srv, "I like blue.", null);
    try path.learnOne(srv, "I like apples.", null);
    try @import("slide_usage_live_test.zig").askExternal(srv, false);
    try path.learnOne(srv, "I like tea.", null);
    try t.expectEqual(@as(usize, 1), m.cuts);
    try t.expectEqual([3]bool{ true, false, false }, m.live);
    try t.expectEqual(.deleted, srv.slide.find(2).?.state);
    try t.expectEqual(.learned, srv.slide.find(1).?.state);
    try t.expectEqual(.missed, srv.slide.find(3).?.state);
}

test "fact usage: automatic forgetting is an explicit CLI opt-in" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const cli = @import("cli.zig");
    var u: cli.Usage = .{};
    const normal = try cli.parse(arena.allocator(), &.{"m"}, &u);
    try t.expect(!normal.slide_forget);
    const forget = try cli.parse(arena.allocator(), &.{ "m", "--slide", "--slide-forget" }, &u);
    try t.expect(forget.slide and forget.slide_forget);
}
