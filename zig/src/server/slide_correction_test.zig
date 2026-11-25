//! Production lesson gates, default sampling and exact deletion references over a scripted model.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const slide = @import("slide.zig");
const teach = @import("slide_lesson.zig");
const correction = @import("slide_correction.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Cx = @import("errors.zig").Cx;
const ByteText = @import("slide_forget_test.zig").ByteText;
const t = std.testing;
const old_answer = "The capital of Rula is Soren.";
const new_answer = "The capital of Rula is Taren.";
const other_answer = "The capital of Nela is Paven.";
const training: teach.Held = .{ .question = "Name the capital of Rula.", .answer = new_answer, .short = "Taren", .subject = "Rula", .fact = 0 };
const held: teach.Held = .{ .question = "Which city is Rula's capital?", .answer = new_answer, .short = "Taren", .subject = "Rula", .fact = 0 };
const probe: teach.Held = .{ .question = "Where is Rula's capital city?", .answer = new_answer, .short = "Taren", .subject = "Rula", .fact = 0 };
const Stays = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};
const Mode = enum { correct, sampled, training, mixed, prune_sampled, delete_changed, before_begin };
const Model = struct {
    mode: Mode,
    learned: bool = false,
    deleted: bool = false,
    tried: bool = false,
    pending: bool = false,
    kept: usize = 0,
    restored: usize = 0,
    unforgot: usize = 0,
    rounds: usize = 0,
    sampled: usize = 0,
    saved: usize = 0,
    dropped: usize = 0,
    correct_sampler: bool = true,
    correct_against: bool = false,
    pocket_belief: bool = false,
    base_reads: usize = 0,
    pocket_present: bool = false,
    refreshed: bool = false, // a round asked to refresh its router wordings
    fn engine(m: *@This()) api.Engine {
        return .{ .ctx = m, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .learn = learn } };
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
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        var prompt: [1024]u8 = undefined;
        const n = @min(prompt.len, req.prompt.len);
        for (req.prompt[0..n], prompt[0..n]) |v, *x| x.* = @intCast(v);
        const q = prompt[0..n];
        if (req.sampling) |s| {
            m.correct_sampler = m.correct_sampler and @abs(s.temperature - 0.7) < 1e-9 and s.top_k == 17 and @abs(s.top_p - 0.9) < 1e-9 and @abs(s.min_p - 0.03) < 1e-9 and s.seed >= 1968 and s.seed < 1971;
            if (m.learned) m.sampled += 1;
        }
        if (req.untaught) m.base_reads += 1;
        const active = m.learned and !req.untaught;
        const base_old = if (m.pocket_belief) "The capital of Rula is Velan." else old_answer;
        const own = std.mem.indexOf(u8, q, "Rula") != null;
        const bad = (m.mode == .sampled and req.sampling != null) or
            (m.mode == .training and std.mem.indexOf(u8, q, "Name") != null) or
            (m.mode == .prune_sampled and m.pending and req.sampling != null);
        const said = if (own) (if (!active) (if (m.deleted and m.mode == .delete_changed) "The capital of Rula is Soren!" else if (m.pocket_present and !req.untaught) old_answer else base_old) else if (bad) old_answer else if (m.mode == .mixed) "Taren or Soren." else new_answer) else if (std.mem.indexOf(u8, q, "Nela") != null) other_answer else "The ocean is large.";
        var ids: [128]u32 = undefined;
        for (said, ids[0..said.len]) |c, *x| x.* = c;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = ids[0..said.len] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn learn(ctx: *anyopaque, req: *const api.LearnRequest, sink: api.LearnSink) api.LearnError!void {
        const m: *Model = @ptrCast(@alignCast(ctx));
        if (req.refresh_wordings) m.refreshed = true;
        if (m.mode == .before_begin and req.train.len > 0) {
            sink.event(sink.ctx, &.{ .done = .{ .message = "refused before beginning", .before_begin = true } });
            return;
        }
        var candidate = false;
        var steps: u32 = 0;
        if (req.shrink != .none) switch (req.shrink) {
            .next => if (m.mode == .prune_sampled and !m.tried) {
                m.tried = true;
                m.pending = true;
                candidate = true;
            },
            .keep => {
                m.kept += 1;
                m.pending = false;
            },
            .restore => {
                m.restored += 1;
                m.pending = false;
            },
            .none => unreachable,
        } else if (req.undo or req.drop) {
            m.learned = false;
            m.dropped += 1;
        } else if (req.forget.len > 0) {
            for (req.forget) |x| if (x.layer == 2) {
                m.pocket_present = false;
            };
            m.learned = false;
            m.deleted = true;
            steps = @intCast(req.forget.len);
        } else if (req.unforget) {
            m.learned = true;
            m.deleted = false;
            m.unforgot += 1;
        } else if (req.save) {
            m.saved += 1;
        } else if (req.train.len > 0) {
            m.learned = true;
            m.rounds += 1;
            m.correct_against = req.against.len == 1 and req.against[0].subject > 0 and req.against[0].ids.len > 3 and req.against[0].ids[2] == 'N';
        }
        const neurons = [_]api.Neuron{ .{ .layer = 1, .neuron = 0, .key = .subject }, .{ .layer = 5, .neuron = 0, .key = .subject } };
        sink.event(sink.ctx, &.{ .learned = .{ .recalled = true, .steps = steps, .loss = 0, .neurons = &neurons, .candidate = candidate } });
        sink.event(sink.ctx, &.{ .done = .{} });
    }
};
const Fixture = struct {
    model: Model,
    text: ByteText = .{},
    srv: *Server,
    arena: std.heap.ArenaAllocator,
    cx: Cx,
    conn: Conn,
    out: slide.Out,
    plan: teach.Plan,
    id: u32,
    neighbour_id: u32,
    fn init(mode: Mode) !*@This() {
        const f = try t.allocator.create(@This());
        errdefer t.allocator.destroy(f);
        f.model = .{ .mode = mode };
        f.text = .{};
        f.arena = .init(t.allocator);
        errdefer f.arena.deinit();
        const a = f.arena.allocator();
        f.srv = try Server.init(t.allocator, t.io, f.model.engine(), f.text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .default_sampling = (try json.parse(a, "{\"temperature\":0.7,\"top_k\":17,\"top_p\":0.9,\"min_p\":0.03}")).ok }, null);
        errdefer f.srv.deinit();
        f.cx = .{ .a = a };
        f.conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .broken = true };
        f.out = .{ .conn = &f.conn, .a = a, .open = false };
        f.neighbour_id = f.srv.slide.add(other_answer, "chat", 1);
        f.srv.slide.mark(f.neighbour_id, .learned);
        f.srv.slide.asked(f.neighbour_id, &.{.{ .question = "Name the capital of Nela.", .answer = other_answer, .short = "Paven" }});
        f.srv.slide.checked(f.neighbour_id, true);
        f.id = f.srv.slide.add(new_answer, "chat", 2);
        var ex = try @import("slide_ask.zig").example(f.srv, &f.cx, a, null, training.question, new_answer, &.{});
        ex.subject = try @import("slide_subject.zig").subjectRow(f.srv, &f.cx, ex, "Rula");
        f.plan = .{ .questions = &.{ training, held }, .full = .{ .train = try a.dupe(api.Example, &.{ex}) }, .facts = &.{new_answer}, .held = &.{held}, .probes = &.{probe}, .checks = &.{} };
        try correction.prepare(f.srv, &f.cx, &f.plan, Stays{});
        try correction.record(f.srv, &f.cx, f.id, f.plan);
        return f;
    }
    fn deinit(f: *@This()) void {
        f.srv.deinit();
        f.arena.deinit();
        t.allocator.destroy(f);
    }
    fn rounds(f: *@This()) slide.Rounds {
        return slide.rounds(f.srv, f.arena.allocator(), &f.cx, f.plan, Stays{}, f.id, &f.out);
    }
    fn accept(f: *@This()) !void {
        try t.expectEqualSlices(bool, &.{true}, f.rounds().recalled);
        f.srv.slide.mark(f.id, .learned);
    }
    fn remove(f: *@This()) !void {
        const facts = try f.srv.slide.snapshot(f.arena.allocator());
        var at: usize = 0;
        while (facts[at].id != f.id) : (at += 1) {}
        try @import("slide_forget.zig").delete(f.srv, &f.cx, &f.out, facts, at, "fixture", Stays{});
    }
};
test "Learning persistence: correction rejects old sampled answers using the configured default sampler" {
    const f = try Fixture.init(.sampled);
    defer f.deinit();
    try t.expectEqualSlices(bool, &.{false}, f.rounds().recalled);
    try t.expect(!f.model.learned and f.model.sampled > 0 and f.model.correct_sampler and f.model.correct_against);
    try t.expect(f.srv.slide.find(f.neighbour_id).?.recalled.?);
}
test "Learning persistence: a refused correction continues its rounds without a wording refresh" {
    const f = try Fixture.init(.sampled);
    defer f.deinit();
    try t.expectEqualSlices(bool, &.{false}, f.rounds().recalled);
    try t.expect(f.model.rounds > 1 and !f.model.refreshed);
}
test "Learning persistence: correction checks training questions as well as held-out questions" {
    const f = try Fixture.init(.training);
    defer f.deinit();
    try t.expectEqualSlices(bool, &.{false}, f.rounds().recalled);
    try t.expect(!f.model.learned);
}
test "Learning persistence: correction refuses a reply naming both the new and old values" {
    const f = try Fixture.init(.mixed);
    defer f.deinit();
    try t.expectEqualSlices(bool, &.{false}, f.rounds().recalled);
    try t.expect(!f.model.learned);
}
test "Learning persistence: shrink retains a neuron needed only for sampled old-answer absence" {
    const f = try Fixture.init(.prune_sampled);
    defer f.deinit();
    try f.accept();
    try t.expect(f.model.tried and f.model.restored == 1 and f.model.kept == 0 and !f.model.pending);
    try t.expectEqual(@as(usize, 2), f.srv.slide.find(f.id).?.neurons.len);
}
test "correction delete: the old greedy tokens return exactly and the neighbour stays recalled" {
    const f = try Fixture.init(.correct);
    defer f.deinit();
    try f.accept();
    try f.remove();
    try t.expect(f.srv.slide.find(f.id).?.state == .deleted and !f.model.learned and f.model.unforgot == 0);
    try t.expect(f.srv.slide.find(f.neighbour_id).?.recalled.?);
}
test "correction delete: an old answer with different tokens refuses deletion and puts the correction back" {
    const f = try Fixture.init(.delete_changed);
    defer f.deinit();
    try f.accept();
    try f.remove();
    try t.expect(f.model.learned and f.model.unforgot == 1 and f.model.saved == 0);
    try t.expectEqual(.learned, f.srv.slide.find(f.id).?.state);
}
test "Learning persistence: correction references must persist before training" {
    const f = try Fixture.init(.correct);
    defer f.deinit();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    f.srv.slide.path = try tmp.dir.realPathFileAlloc(t.io, ".", f.arena.allocator());
    const failed = if (correction.record(f.srv, &f.cx, f.id, f.plan)) |_| false else |_| true;
    try t.expect(failed and f.model.rounds == 0 and !f.model.learned);
}
test "Learning persistence: correction is explicit, and its old value recognizer allows a shorter paraphrase" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expect(!try correction.requested((try json.parse(a, "{}")).ok));
    try t.expect(try correction.requested((try json.parse(a, "{\"correction\":true}")).ok));
    try t.expectError(error.BadCorrectionFlag, correction.requested((try json.parse(a, "{\"correction\":1}")).ok));
    try t.expect(correction.oldReturned(training.question, "Soren is the capital and largest city of Rula.", "Soren", "It is Soren."));
    try t.expect(!correction.oldReturned(training.question, old_answer, "Soren", new_answer));
}

test "Learning persistence: a correction refuses a greedy-only default before training" {
    const f = try Fixture.init(.correct);
    defer f.deinit();
    f.srv.config.default_sampling = null;
    try t.expectError(error.Refused, correction.prepare(f.srv, &f.cx, &f.plan, Stays{}));
    try t.expect(!f.model.learned and f.model.rounds == 0);
}

test "correction delete: graph references own and roundtrip the original tokens and refusal flag" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const A = @import("slide_answers.zig");
    var ids = [_]u32{ 8, 4, 1 };
    const owned = try A.dupe(a, &.{.{ .question = training.question, .answer = new_answer, .old = old_answer, .old_short = "Soren", .old_ids = &ids, .reject_old = true }});
    ids[0] = 99;
    const encoded = try json.stringify(a, try A.value(a, owned), .{});
    const back = try A.read(a, (try json.parse(a, encoded)).ok);
    try t.expectEqualSlices(u32, &.{ 8, 4, 1 }, back[0].old_ids);
    try t.expectEqualStrings("Soren", back[0].old_short);
    try t.expect(back[0].reject_old);
    for ([_][]const u8{
        "[{\"question\":\"q\",\"answer\":\"n\",\"old\":\"o\",\"old_short\":\"o\",\"old_ids\":[-1],\"reject_old\":true}]",
        "[{\"question\":\"q\",\"answer\":\"n\",\"old\":\"o\",\"old_short\":\"o\",\"old_ids\":[4294967296],\"reject_old\":true}]",
        "[{\"question\":\"q\",\"answer\":\"n\",\"old\":\"o\",\"old_short\":\"o\",\"old_ids\":[],\"reject_old\":true}]",
        "[{\"question\":\"q\",\"answer\":\"n\",\"old\":\"o\",\"old_short\":\"o\",\"old_ids\":[1]}]",
    }) |bad| try t.expectError(error.BadCorrectionReference, A.read(a, (try json.parse(a, bad)).ok));
}

test "Learning persistence: a correction refused before begin cannot drop the preceding lesson" {
    const f = try Fixture.init(.before_begin);
    defer f.deinit();
    try t.expectEqualSlices(bool, &.{false}, f.rounds().recalled);
    try t.expectEqual(@as(usize, 0), f.model.dropped);
    try t.expect(f.srv.slide.find(f.neighbour_id).?.recalled.?);
}

fn dependentRefused(f: *Fixture, prior: u32) !void {
    const rounds = f.model.rounds;
    try t.expectError(error.Refused, correction.prepare(f.srv, &f.cx, &f.plan, Stays{}));
    const label = try std.fmt.allocPrint(f.arena.allocator(), "fact {d}", .{prior});
    try t.expect(std.mem.indexOf(u8, f.cx.message, label) != null);
    try t.expect(std.mem.indexOf(u8, f.cx.message, f.srv.slide.find(prior).?.text) != null);
    try t.expectEqual(rounds, f.model.rounds);
    try t.expect(f.model.base_reads > 0);
}
fn underlying(f: *Fixture) u32 {
    const id = f.srv.slide.add(old_answer, "chat", 3);
    f.srv.slide.mark(id, .learned);
    f.srv.slide.placed(id, &.{.{ .layer = 2, .neuron = 0, .key = .subject }});
    f.srv.slide.asked(id, &.{.{ .question = held.question, .answer = old_answer, .short = "Soren" }});
    f.model.pocket_belief = true;
    f.model.pocket_present = true;
    return id;
}
test "Learning persistence: refuses a dependent correction before its underlying lesson could be deleted first" {
    const f = try Fixture.init(.correct);
    defer f.deinit();
    const prior = underlying(f);
    try dependentRefused(f, prior);
    try t.expect(!f.model.learned and f.srv.slide.find(prior).?.state == .learned);
    try t.expectEqual(@as(usize, 0), f.model.dropped);
    f.srv.slide.mark(f.id, .missed);
    const facts = try f.srv.slide.snapshot(f.arena.allocator());
    try @import("slide_forget.zig").delete(f.srv, &f.cx, &f.out, facts, facts.len - 1, "fixture", Stays{});
    try t.expectEqual(.deleted, f.srv.slide.find(prior).?.state);
    try t.expect(!f.model.pocket_present and f.model.saved == 1);
}
test "Learning persistence: refuses a dependent correction before a combined delete could strand it" {
    const f = try Fixture.init(.correct);
    defer f.deinit();
    const prior = underlying(f);
    try dependentRefused(f, prior);
    // The refused correction owns no neurons, so it cannot join or obstruct a delete of the underlying fact.
    try t.expectEqual(@as(usize, 0), f.srv.slide.find(f.id).?.neurons.len);
    f.srv.slide.mark(f.id, .missed);
    const facts = try f.srv.slide.snapshot(f.arena.allocator());
    const both = try json.newObject(f.arena.allocator());
    try both.put(f.arena.allocator(), "ids", .{ .array = try f.arena.allocator().dupe(json.Value, &.{ try json.intValue(f.arena.allocator(), prior), try json.intValue(f.arena.allocator(), f.id) }) });
    try t.expectError(error.FactNotLearned, @import("slide_forget_many.zig").indices(f.arena.allocator(), .{ .object = both }, facts));
    const batch = try @import("slide_forget_many.zig").plan(f.arena.allocator(), facts, &.{facts.len - 1});
    try t.expectEqualSlices(u32, &.{prior}, batch.ids);
    try t.expectEqual(@as(usize, 1), batch.neurons.len);
}
test "Learning persistence: refuses a second correction before deleting the earlier correction could strand it" {
    const f = try Fixture.init(.correct);
    defer f.deinit();
    try f.accept();
    const next = "The capital of Rula is Lumen.";
    var q = training;
    q.answer = next;
    q.short = "Lumen";
    f.plan = .{ .facts = &.{next}, .questions = &.{q}, .held = &.{}, .probes = &.{}, .checks = &.{}, .full = .{} };
    try dependentRefused(f, f.id);
    // The first correction is still independently deletable, with its exact original reference.
    try f.remove();
    try t.expectEqual(.deleted, f.srv.slide.find(f.id).?.state);
    try t.expect(!f.model.learned and f.model.unforgot == 0);
}

test "Learning persistence: a matching kept value refuses even when pocket contributions cancel in the greedy reply" {
    const f = try Fixture.init(.correct);
    defer f.deinit();
    const prior = underlying(f);
    f.model.pocket_belief = false;
    try dependentRefused(f, prior);
}
test "Learning persistence: pretrained corrections remain allowed beside an unrelated pocket" {
    const f = try Fixture.init(.correct);
    defer f.deinit();
    f.srv.slide.placed(f.neighbour_id, &.{.{ .layer = 3, .neuron = 0, .key = .subject }});
    try correction.prepare(f.srv, &f.cx, &f.plan, Stays{});
    try f.accept();
    try f.remove();
    try t.expectEqual(.deleted, f.srv.slide.find(f.id).?.state);
    try t.expectEqual(.learned, f.srv.slide.find(f.neighbour_id).?.state);
}

test "Learning persistence: a pocket-dependent reference without graph ownership refuses before training" {
    const f = try Fixture.init(.correct);
    defer f.deinit();
    f.model.pocket_belief = true;
    f.model.pocket_present = true;
    try t.expectError(error.Refused, correction.prepare(f.srv, &f.cx, &f.plan, Stays{}));
    try t.expect(std.mem.indexOf(u8, f.cx.message, "pocket neurons missing from this graph") != null);
    try t.expect(!f.model.learned and f.model.rounds == 0);
}
