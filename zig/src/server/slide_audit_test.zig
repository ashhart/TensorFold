//! Synthetic pocket-to-report checks, including graph preservation, threshold boundaries and CLI visibility.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const planning = @import("slide_audit_probes.zig");
const reporting = @import("slide_audit_report.zig");
const Graph = @import("slide_graph.zig").Graph;
const fixture = @import("living").audit_test;

test "pocket audit report: live clean versus leaky neurons, cohorts, witnesses, CLI and unchanged graph" {
    const a0 = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a0);
    defer arena.deinit();
    const a = arena.allocator();
    var g = Graph.init(a0, std.testing.io, null);
    defer g.deinit();
    const id = g.add("My dog is called Ginger.", "chat", 1);
    g.mark(id, .learned);
    g.placed(id, &.{.{ .layer = 1, .neuron = 0, .key = .reply }});
    g.asked(id, &.{.{ .question = "What is my dog called?" }});
    const other = g.add("My sister is called Ana.", "chat", 2);
    g.mark(other, .learned);
    g.placed(other, &.{.{ .layer = 1, .neuron = 1, .key = .reply }});
    g.asked(other, &.{.{ .question = "What is my sister called?" }});
    const before = try g.render(a, null);
    const plan = try planning.plan(a, try g.snapshot(a));
    var m = fixture.Device.init();
    var learner = fixture.Learner.init(a0, std.testing.io, &m);
    defer learner.deinit();
    var audit = try fixture.Learner.Audit.init(&learner);
    defer audit.deinit();
    const ns = try std.mem.concat(a, api.Neuron, &.{ plan.facts[0].source.neurons, plan.facts[1].source.neurons });
    const peaks = try a.alloc(api.pocket_audit.Peak, plan.probes.len * ns.len);
    for (plan.probes, 0..) |p, i| {
        const ids: []const u32 = if (p.kind == .own and p.owner == id) &.{1} else &.{2};
        try audit.read(&learner, ids, ns, peaks[i * ns.len ..][0..ns.len]);
    }
    const report = try reporting.value(a, plan, ns, peaks, reporting.default_threshold);
    const neurons = [_]json.Value{ report.field("facts").?.items().?[0].field("neurons").?.items().?[0], report.field("facts").?.items().?[1].field("neurons").?.items().?[0] };
    try std.testing.expect(!neurons[0].field("flagged").?.bool);
    try std.testing.expect(neurons[1].field("flagged").?.bool);
    try std.testing.expectEqual(@as(i64, 265), neurons[1].field("off_subject").?.field("fired").?.int64().?);
    try std.testing.expectEqual(@as(i64, 1), neurons[1].field("cohorts").?.field("other").?.field("fired").?.int64().?);
    const own = neurons[0].field("cohorts").?.field("own").?;
    try std.testing.expectEqual(@as(i64, 1), own.field("fired").?.int64().?);
    try std.testing.expectEqual(@as(usize, 5), neurons[1].field("strongest_off_subject").?.items().?.len);
    const high = try reporting.value(a, plan, ns, peaks, 1);
    try std.testing.expect(!high.field("facts").?.items().?[1].field("neurons").?.items().?[0].field("flagged").?.bool);
    var w: std.Io.Writer.Allocating = .init(a);
    try @import("slide_audit_cli.zig").display(&w.writer, report);
    try std.testing.expect(std.mem.indexOf(u8, w.written(), "FLAG") != null);
    try std.testing.expect(std.mem.indexOf(u8, w.written(), "neuron 0") != null);
    try std.testing.expectEqualStrings(before, try g.render(a, null));
    const graph = (try json.parse(a, before)).ok;
    try graph.object.put(a, "audit", report);
    try std.testing.expect(graph.field("facts") != null and graph.field("next") != null);
    try std.testing.expect((try json.parse(a, try json.stringify(a, graph, .{}))).ok.field("audit") != null);
}

test "pocket audit report: unknown subjects and no facts stay explicitly unmeasured; invalid samples fail" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try planning.plan(a, &.{});
    const empty = try reporting.value(a, p, &.{}, &.{}, 0.01);
    try std.testing.expectEqual(@as(usize, 0), empty.field("facts").?.items().?.len);
    try std.testing.expectError(error.BadAudit, reporting.value(a, p, &.{}, &.{}, std.math.nan(f64)));
}
