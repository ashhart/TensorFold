//! Cumulative drift, fixed external recall and graph receipts through production lesson rounds.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const fixture = @import("slide_learn_path_test.zig");
const Server = @import("server.zig").Server;
const Cx = @import("errors.zig").Cx;
const panel = @import("slide_panel.zig");
const metrics = @import("slide_metrics.zig");
const schema = @import("slide_metrics_types.zig");
const store = @import("slide_metrics_store.zig");
const input = @import("slide_recall_input.zig");
const graph = @import("slide_graph.zig");
const teach = @import("slide_lesson.zig");
const Stays = fixture.Stays;
const t = std.testing;

const Model = struct {
    inner: fixture.Model = .{},
    drift: bool = false,
    cumulative: bool = false,
    miss_recall: bool = false,
    fn engine(m: *@This()) api.Engine {
        var e = m.inner.engine();
        e.ctx = m;
        e.vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .learn = learn };
        return e;
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn learn(ctx: *anyopaque, req: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const m: *@This() = @ptrCast(@alignCast(ctx));
        const e = m.inner.engine();
        return e.vtable.learn.?(e.ctx, req, sink);
    }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *@This() = @ptrCast(@alignCast(ctx));
        var bytes: [8192]u8 = undefined;
        const n = @min(req.prompt.len, bytes.len);
        for (req.prompt[0..n], bytes[0..n]) |v, *b| b.* = @intCast(v);
        const prompt = bytes[0..n];
        var answer: ?[]const u8 = null;
        inline for (.{ panel.unrelated[0..], panel.controls[0..], panel.neighbours[0..] }) |qs| {
            for (qs) |q| if (std.mem.indexOf(u8, prompt, q) != null) {
                answer = "The reference response stays fixed.";
            };
        }
        if (m.drift and !req.untaught and m.inner.live[0]) for (panel.unrelated[0..3]) |q| {
            if (std.mem.indexOf(u8, prompt, q) != null) answer = "A new and incompatible claim.";
        };
        if (m.cumulative) for (panel.unrelated[0..3]) |q| {
            if (std.mem.indexOf(u8, prompt, q) != null) answer = if (req.untaught or !m.inner.live[0]) cumulative_a else if (m.inner.live[1]) cumulative_c else cumulative_b;
        };
        if ((m.drift or m.cumulative) and answer != null and std.mem.indexOf(u8, prompt, "Does B tell") != null) answer = "yes";
        if (std.mem.indexOf(u8, prompt, "External colour recall?") != null) answer = if (m.inner.live[0] and !m.miss_recall and !req.untaught) "blue" else "unknown";
        if (std.mem.indexOf(u8, prompt, "External fruit recall?") != null) answer = if (m.inner.live[1] and !req.untaught) "apples" else "unknown";
        if (answer) |text| {
            var ids: [128]u32 = undefined;
            for (text, ids[0..text.len]) |b, *v| v.* = b;
            sink.event(sink.ctx, id, &.{ .tokens = ids[0..text.len] });
            sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
            return;
        }
        const e = m.inner.engine();
        return e.vtable.submit(e.ctx, id, req, sink);
    }
};

fn server(m: *Model, text: *fixture.Text, gate: bool, path: ?[]const u8) !*Server {
    return Server.init(t.allocator, t.io, m.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .slide_drift_gate = gate, .slide_graph = path }, null);
}
const recall = [_]graph.Answer{.{ .question = "External colour recall?", .answer = "blue", .short = "blue", .probe = true }};

fn plan(srv: *Server, cx: *Cx) !teach.Plan {
    var kept = [_]bool{false};
    return (try teach.lesson(srv, cx, &srv.teacher, &.{"I like blue."}, &kept, false, &.{}, &.{}, Stays{})).?;
}

fn round(gate: bool, drift: bool, miss: bool) !void {
    var m: Model = .{ .drift = drift, .miss_recall = miss };
    defer m.inner.deinit();
    var text: fixture.Text = .{};
    const srv = try server(&m, &text, gate, null);
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const id = srv.slide.add("I like blue.", "chat", 1);
    try srv.slide.suppliedRecall(id, &recall);
    const p = try plan(srv, &cx);
    var conn: @import("http_conn.zig").Conn = undefined;
    var out: @import("slide.zig").Out = .{ .conn = &conn, .a = cx.a, .open = false };
    const result = @import("slide.zig").rounds(srv, cx.a, &cx, p, Stays{}, id, &out);
    const refused = gate and (drift or miss);
    try t.expectEqual(!refused, result.recalled[0]);
    try t.expectEqual(!refused, m.inner.live[0]);
    if (!gate) {
        try t.expectEqual(@as(?json.Value, null), srv.slide.find(id).?.evaluation);
        try t.expectEqual(@as(?json.Value, null), srv.slide.drift_base);
        return;
    }
    const report = try store.decode(schema.Report, cx.a, srv.slide.find(id).?.evaluation.?);
    try t.expectEqual(@as(u32, 115), report.plain[0].checked);
    try t.expectEqual(@as(u32, if (drift) 3 else 0), report.plain[0].meaning_changed);
    try t.expectEqual(@as(u32, 1), report.plain[0].recall_asked);
    try t.expectEqual(@as(u32, if (miss) 0 else 1), report.plain[0].recall_passed);
    if (refused) try t.expect(result.why != null);
}

test "Slide metrics: real lesson rounds refuse drift and missed external recall only under the opt-in gate" {
    try round(false, true, false);
    try round(true, true, false);
    try round(true, false, true);
    try round(true, false, false);
}

test "Slide metrics: recorded untaught replies do not become a rolling taught baseline" {
    var m: Model = .{};
    defer m.inner.deinit();
    var text: fixture.Text = .{};
    const srv = try server(&m, &text, false, null);
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const id = srv.slide.add("I like blue.", "chat", 0);
    try srv.slide.suppliedRecall(id, &recall);
    const p = try plan(srv, &cx);
    m.inner.live[0] = true;
    const first = try metrics.measure(srv, &cx, p, id, Stays{});
    try t.expectEqual(@as(u32, 115), first.plain[0].byte_equal);
    const saved = try json.stringify(cx.a, srv.slide.drift_base.?, .{});
    m.drift = true;
    const second = try metrics.measure(srv, &cx, p, id, Stays{});
    try t.expectEqual(@as(u32, 3), second.plain[0].meaning_changed);
    try t.expectEqualStrings(saved, try json.stringify(cx.a, srv.slide.drift_base.?, .{}));
    const old = try store.decode(schema.Baseline, cx.a, srv.slide.drift_base.?);
    var corrupt = old;
    const replies = try cx.a.dupe(schema.Output, old.replies);
    replies[0].sha256 = "bad";
    corrupt.replies = replies;
    srv.slide.drift_base = try store.value(srv.slide.arena.allocator(), corrupt);
    const mismatch = try metrics.measure(srv, &cx, p, id, Stays{});
    try t.expectEqual(@as(u32, 1), mismatch.untaught_mismatches);
    try t.expect(std.mem.indexOf(u8, mismatch.refused.?, "untaught") != null);
}

test "Slide metrics: fixed panel is held out from every built-in training and check set" {
    const prompts = @import("slide_prompts.zig");
    const untold = @import("slide_untold.zig");
    inline for (.{ panel.unrelated[0..], panel.controls[0..], panel.neighbours[0..] }) |qs| {
        for (qs) |q| inline for (.{ prompts.keep_prompts[0..], prompts.route_prompts[0..], prompts.personal_prompts[0..], prompts.personal_bank[0..], untold.checked[0..], untold.kept[0..] }) |old| {
            for (old) |p| try t.expect(!std.mem.eql(u8, q, p));
        };
    }
}

test "Slide metrics: graph receipts, supplied wordings and original baseline survive restart and GET render" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fs.path.join(t.allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path, "graph.json" });
    defer t.allocator.free(path);
    var g = graph.Graph.init(t.allocator, t.io, path);
    const id = g.add("I like blue.", "chat", 0);
    try g.suppliedRecall(id, &recall);
    try g.evaluated(id, schema.Report{ .plain = @splat(.{ .checked = 115, .byte_equal = 114, .meaning_changed = 1 }) });
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = try g.driftBaseline(a, try store.value(a, schema.Baseline{ .panel_sha256 = "panel", .profile = "profile", .replies = &.{.{ .content = "base", .sha256 = "hash" }} }));
    g.deinit();
    var again = graph.Graph.init(t.allocator, t.io, path);
    defer again.deinit();
    const root = (try json.parse(a, try again.render(a, null))).ok;
    try t.expect(root.field("drift_baseline") != null);
    const fact = root.field("facts").?.array[0];
    try t.expectEqualStrings("External colour recall?", fact.field("recall").?.array[0].strField("question").?);
    const report = try store.decode(schema.Report, a, fact.field("evaluation").?);
    try t.expectEqual(@as(u32, 1), report.plain[0].meaning_changed);
    const snap = try again.snapshot(a);
    try t.expectEqualStrings("blue", snap[0].recall[0].short);
    try t.expect(snap[0].evaluation != null);
}

test "Slide metrics: hard targets use worst drift and median recall; byte equality is report-only" {
    var r: schema.Report = .{ .plain = @splat(.{ .recall_asked = 40, .recall_passed = 36 }) };
    try t.expectEqual(@as(?[]const u8, null), schema.refusal(r));
    r.plain[0].recall_passed = 0;
    try t.expectEqual(@as(?[]const u8, null), schema.refusal(r));
    r.plain[1].recall_passed = 35;
    try t.expect(schema.refusal(r) != null);
    r.plain = @splat(.{ .recall_asked = 40, .recall_passed = 36 });
    r.plain[2].meaning_changed = 3;
    try t.expect(schema.refusal(r) != null);
    r.lessons = 10;
    try t.expectEqual(@as(?[]const u8, null), schema.refusal(r));
    r.plain[0].meaning_changed = 6;
    try t.expect(schema.refusal(r) != null);
    r.plain[0].meaning_changed = 0;
    r.plain[1].leaks = 1;
    try t.expect(schema.refusal(r) != null);
    r.plain[1].leaks = 0;
    r.plain[2].invented = 1;
    try t.expect(schema.refusal(r) != null);
}

test "Slide metrics: malformed recall is refused, exact supplied sets are selected per fact" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = try store.value(a, .{ .recall = &[_]input.Item{.{ .fact = "I like blue.", .question = "External colour recall?", .answer = "blue" }} });
    const rows = try input.read(a, body);
    try t.expectEqual(@as(usize, 1), (try input.select(a, rows, "I like blue.")).len);
    try t.expectEqual(@as(usize, 0), (try input.select(a, rows, "I like tea.")).len);
    try t.expectError(error.BadRecall, input.read(a, try store.value(a, .{ .recall = &[_]input.Item{.{ .fact = "I like blue.", .question = "Q?", .answer = "red" }} })));
}

const cumulative_a = "Amber bronze coral denim ebony frost green hazel ivory jade.";
const cumulative_b = "Ochre silver coral denim ebony frost green hazel ivory jade.";
const cumulative_c = "Ochre silver stone denim ebony frost green hazel ivory jade.";

test "Slide metrics: two individually alike lessons cannot accumulate meaning drift under the gate" {
    const words = @import("slide_words.zig");
    try t.expect(words.alike(cumulative_a, cumulative_b));
    try t.expect(words.alike(cumulative_b, cumulative_c));
    try t.expect(!words.alike(cumulative_a, cumulative_c));
    var m: Model = .{ .cumulative = true };
    defer m.inner.deinit();
    var text: fixture.Text = .{};
    const srv = try server(&m, &text, true, null);
    defer srv.deinit();
    try fixture.learnBody(srv, "{\"text\":\"I like blue.\",\"recall\":[{\"fact\":\"I like blue.\",\"question\":\"External colour recall?\",\"answer\":\"blue\"}]}", null);
    try t.expect(m.inner.live[0]);
    try fixture.learnBody(srv, "{\"text\":\"I like apples.\",\"recall\":[{\"fact\":\"I like apples.\",\"question\":\"External fruit recall?\",\"answer\":\"apples\"}]}", "meaning-change target");
    try t.expect(m.inner.live[0] and !m.inner.live[1]);
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const facts = try srv.slide.snapshot(arena.allocator());
    const first = try store.decode(schema.Report, arena.allocator(), facts[0].evaluation.?);
    const second = try store.decode(schema.Report, arena.allocator(), facts[1].evaluation.?);
    try t.expectEqual(@as(u32, 3), first.plain[0].alike);
    try t.expectEqual(@as(u32, 3), second.plain[0].meaning_changed);
    try t.expectEqual(graph.State.missed, facts[1].state);
}

test "Slide metrics: supplied wordings read by training cannot earn held-out recall" {
    var m: Model = .{};
    defer m.inner.deinit();
    var text: fixture.Text = .{};
    const srv = try server(&m, &text, true, null);
    defer srv.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const id = srv.slide.add("I like blue.", "chat", 0);
    try srv.slide.suppliedRecall(id, &recall);
    var p = try plan(srv, &cx);
    const trained = try @import("slide_ask.zig").promptOnly(srv, &cx, cx.a, null, recall[0].question);
    p.full.train = try std.mem.concat(cx.a, api.Example, &.{ p.full.train, &.{trained} });
    m.inner.live[0] = true;
    const r = try metrics.measure(srv, &cx, p, id, Stays{});
    try t.expect(!r.complete and r.refused != null);
}

test "Slide metrics: receipt decoding preserves saved values defaults and exact integer limits" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var report: schema.Report = .{ .lessons = 4294967295, .gate = true, .refused = "kept out" };
    report.plain[1].recall_asked = 4294967295;
    report.plain[1].recall_passed = 4294967294;
    const encoded = try store.value(a, report);
    const decoded = try store.decode(schema.Report, a, encoded);
    try t.expectEqualStrings(try std.json.Stringify.valueAlloc(a, report, .{}), try std.json.Stringify.valueAlloc(a, decoded, .{}));
    const defaults = try store.decode(schema.Report, a, (try json.parse(a, "{}")).ok);
    try t.expectEqualStrings(try std.json.Stringify.valueAlloc(a, schema.Report{}, .{}), try std.json.Stringify.valueAlloc(a, defaults, .{}));
    const base: schema.Baseline = .{ .panel_sha256 = "panel", .profile = "profile", .replies = &.{.{ .content = "answer", .sha256 = "hash" }} };
    const again = try store.decode(schema.Baseline, a, try store.value(a, base));
    try t.expectEqualStrings(base.replies[0].content, again.replies[0].content);
    const exponent = try store.decode(schema.Report, a, (try json.parse(a, "{\"lessons\":1e1}")).ok);
    try t.expectEqual(@as(u32, 10), exponent.lessons);
    for ([_][]const u8{ "{\"lessons\":4294967296}", "{\"lessons\":-1}", "{\"lessons\":1.5}", "{\"gate\":1}", "{\"unknown\":0}", "{\"plain\":[]}", "{\"rows\":[{}]}" }) |text|
        try t.expectError(error.InvalidMetricReceipt, store.decode(schema.Report, a, (try json.parse(a, text)).ok));
    try t.expectError(error.InvalidMetricReceipt, store.decode(schema.Baseline, a, (try json.parse(a, "{}")).ok));
}
