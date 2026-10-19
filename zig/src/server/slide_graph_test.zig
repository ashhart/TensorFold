//! The Living Weights graph's tests: facts kept across a reopen, a lesson's live fields, a save's measures.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const slide_graph = @import("slide_graph.zig");
const Graph = slide_graph.Graph;
const State = slide_graph.State;

test "facts persist across a reopen, link by a shared key word, and clear without reusing ids" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path, "graph.json" });
    defer gpa.free(path);
    var g = Graph.init(gpa, io, path);
    const a = g.add("My company, Larkspur Robotics, sells the Heron X2.", "file:company.md", 100);
    const b = g.add("Larkspur Robotics made 4.7 million pounds in Q3.", "file:company.md", 101);
    const c = g.add("My favourite colour is teal.", "chat", 102);
    g.mark(a, .learned);
    g.mark(c, .missed);
    g.deinit();

    var again = Graph.init(gpa, io, path);
    defer again.deinit();
    try std.testing.expectEqual(@as(usize, 3), again.facts.items.len);
    try std.testing.expectEqual(State.learned, again.facts.items[0].state);
    try std.testing.expectEqual(State.missed, again.facts.items[2].state);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = switch (try json.parse(arena.allocator(), try again.render(arena.allocator(), null))) {
        .ok => |v| v,
        .err => return error.BadGraph,
    };
    const facts = doc.field("facts").?.items().?;
    try std.testing.expectEqual(@as(usize, 1), facts[0].field("links").?.items().?.len);
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(arena.allocator(), "{d}", .{b}), facts[0].field("links").?.items().?[0].int);
    try std.testing.expectEqual(@as(usize, 0), facts[2].field("links").?.items().?.len);
    again.clear();
    try std.testing.expectEqual(c + 1, again.add("My dog is called Pickle.", "chat", 103));
}

test "a lesson's phases, loss trace and neurons show while it learns; a learned fact keeps its neurons on disk" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path, "graph.json" });
    defer gpa.free(path);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    var g = Graph.init(gpa, io, path);
    const id = g.add("My sister is called Ana.", "chat", 100);
    const other = g.add("I like blue.", "chat", 101);
    g.mark(id, .learning);
    g.phase(id, .writing);
    g.asked(id, &.{ .{ .question = "What is my sister's name?", .answer = "Your sister is called Ana.", .short = "Ana" }, .{ .question = "Do you remember what my sister is called?", .answer = "Yes, she is called Ana." }, .{ .question = "Who is my sister?", .probe = true } });
    g.round(id, 1);
    g.progress(id, .{ .phase = .reading, .at = 3, .of = 40 });
    const ns = [_]api.Neuron{ .{ .layer = 13, .neuron = 0, .key = .subject }, .{ .layer = 15, .neuron = 2, .key = .subject } };
    for (1..4) |s| g.progress(id, .{ .phase = .values, .at = @intCast(s), .loss = 2.0 / @as(f32, @floatFromInt(s)), .neurons = &ns });
    g.round(id, 2);
    for (1..3) |s| g.progress(id, .{ .phase = .values, .at = @intCast(s), .loss = 0.25, .neurons = &ns });
    g.checked(id, true);
    var text = try g.render(ar, null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"phase\":\"values\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"trace\":[[2.0,1.0,0.6667],[0.25,0.25]]") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "{\"layer\":13,\"neuron\":0,\"key\":\"subject\"}") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"checks\":1,\"recalled\":true") != null);
    g.mark(id, .learned);
    g.placed(other, &ns);
    g.mark(other, .missed);
    text = try g.render(ar, null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"phase\"") == null);
    g.deinit();

    var again = Graph.init(gpa, io, path);
    defer again.deinit();
    try std.testing.expectEqual(@as(usize, 2), again.facts.items[0].neurons.len);
    try std.testing.expectEqual(api.NeuronKey.subject, again.facts.items[0].neurons[1].key);
    try std.testing.expectEqual(@as(u16, 15), again.facts.items[0].neurons[1].layer);
    try std.testing.expectEqual(@as(usize, 3), again.facts.items[0].questions.len);
    try std.testing.expectEqualStrings("Ana", again.facts.items[0].answers[0].short);
    try std.testing.expect(again.facts.items[0].answers[2].probe);
    // a later lesson checks the learned fact again on its wordings, the probe marked; a reply fills a missing answer
    const back = try again.earlier(ar, 999);
    try std.testing.expectEqual(@as(usize, 1), back.len);
    try std.testing.expectEqual(id, back[0].id);
    try std.testing.expect(back[0].asked[2].probe and !back[0].asked[1].probe);
    try std.testing.expectEqualStrings("Yes, she is called Ana.", back[0].asked[1].answer);
    again.answered(id, 2, "Your sister Ana.");
    try std.testing.expectEqualStrings("Your sister Ana.", again.facts.items[0].answers[2].answer);
    try std.testing.expectEqual(@as(usize, 0), (try again.earlier(ar, id)).len);
    // a lesson taken out after its re-check found the fact forgotten puts back the verdict it began with
    again.checked(id, true);
    const was = try again.earlier(ar, 999);
    again.checked(id, false);
    try std.testing.expectEqual(@as(?bool, false), again.facts.items[0].recalled);
    again.restoreChecks(was);
    try std.testing.expectEqual(@as(?bool, true), again.facts.items[0].recalled);
    try std.testing.expectEqual(@as(u32, 2), again.facts.items[0].round);
    try std.testing.expectEqual(@as(usize, 0), again.facts.items[1].neurons.len);
}

test "a save is measured against the shards as they were before it: the files it changed, how long it took" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    const dir = try std.fs.path.join(ar, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"layers_block_type\": [\"mamba\", \"moe\"]}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "\x02\x00\x00\x00\x00\x00\x00\x00{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "other.safetensors", .data = "\x02\x00\x00\x00\x00\x00\x00\x00{}" });
    var g = Graph.init(gpa, io, null);
    defer g.deinit();
    g.saving(dir);
    try std.testing.expect(std.mem.indexOf(u8, try g.render(ar, dir), "\"save\":{\"state\":\"saving\"") != null);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "\x03\x00\x00\x00\x00\x00\x00\x00{} " });
    g.saveEnded(dir, 3.14, null);
    const text = try g.render(ar, dir);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"layers\":2,\"kinds\":\"ME\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"seconds\":3.1,\"bytes\":0,\"files\":[\"model.safetensors\"]") != null);
}
