//! Sliding Weights over HTTP: /v1/slide/learn streams learning, /v1/slide/graph lists or clears facts, /slide draws.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const sse = @import("sse.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const errors = @import("errors.zig");
const teach = @import("slide_lesson.zig");
const found = @import("slide_found.zig");
const wording = @import("slide_words.zig");
const check = @import("slide_verify.zig");
const recheck = @import("slide_recheck.zig");
const replace = @import("slide_replace.zig");
const slide_graph = @import("slide_graph.zig");
const slide_teacher = @import("slide_teacher.zig");
const log = @import("log.zig");
const Gone = @import("openai.zig").Gone;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

const page = @embedFile("slide.html");

/// The page's scripts, each served at /slide/NAME.
const scripts = [_]struct { []const u8, []const u8 }{
    .{ "data.js", @embedFile("slide_data.js") },
    .{ "panel.js", @embedFile("slide_panel.js") },
    .{ "flat.js", @embedFile("slide_flat.js") },
    .{ "scene.js", @embedFile("slide_scene.js") },
    .{ "tower.js", @embedFile("slide_tower.js") },
    .{ "fx.js", @embedFile("slide_fx.js") },
    .{ "bloom.js", @embedFile("slide_bloom.js") },
    .{ "layout.js", @embedFile("slide_layout.js") },
};

pub fn view(_: *Server, conn: *Conn, _: Allocator) void {
    send(conn, "text/html; charset=utf-8", page);
}

/// One of the page's scripts by name, else the unknown path.
pub fn script(_: *Server, conn: *Conn, a: Allocator, name: []const u8) void {
    for (scripts) |x| if (std.mem.eql(u8, x[0], name)) return send(conn, "text/javascript; charset=utf-8", x[1]);
    routes.unknown(conn, a);
}

fn send(conn: *Conn, kind: []const u8, body: []const u8) void {
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", kind) catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch return) catch return;
    conn.finish(body) catch {};
}

pub fn graph(srv: *Server, conn: *Conn, a: Allocator) void {
    const body = srv.slide.render(a, folder(srv)) catch return refuse(conn, a, 500, "out of memory");
    conn.sendJson(200, body);
}

/// The graph goes; the weights keep what they learned.
pub fn forget(srv: *Server, conn: *Conn, _: Allocator) void {
    srv.slide.clear();
    conn.sendJson(200, "{\"cleared\": true}");
}

/// The body's facts, each written into a lesson by the model and learned in turn into the live weights.
pub fn learn(srv: *Server, conn: *Conn, a: Allocator) void {
    const raw = switch (http_body.read(conn, a, http_body.limit) catch return) {
        .ok => |b| b,
        .refused => |message| return refuse(conn, a, 400, message),
    };
    const body = switch (json.parse(a, if (raw.len == 0) "{}" else raw) catch return) {
        .ok => |v| v,
        .err => |message| return refuse(conn, a, 400, message),
    };
    const text = body.strField("text") orelse return refuse(conn, a, 400, "text is required");
    const source = body.strField("source") orelse "chat";
    // whom the told facts are about: "self" (the model: "Your name is Ada."), "user"; else judged for each fact
    const about: ?teach.About = if (body.strField("about")) |w| std.meta.stringToEnum(teach.About, w) orelse return refuse(conn, a, 400, "about must be \"self\" or \"user\"") else null;
    switch (run(srv, a, &.{}, 0)) {
        .refused => |e| return switch (e) {
            error.Unsupported => refuse(conn, a, 501, "this engine does not learn: serve its model with --slide"),
            error.Busy, error.Closed => refuse(conn, a, 503, "the engine cannot take a learn request now"),
        },
        else => {},
    }
    srv.teacher.learning.lockUncancelable(srv.io);
    defer srv.teacher.learning.unlock(srv.io);
    sse.open(conn) catch return;
    var out: Out = .{ .conn = conn, .a = a };
    var cx: errors.Cx = .{ .a = a };
    const gone: Gone = .{ .conn = conn };
    const facts = found.facts(srv, &cx, text, source, about != null, gone) catch |e| return out.fail(&cx, e);
    if (facts.len == 0) {
        out.event("failed", .{ .message = "no facts found in the text: nothing was learned" });
        sse.done(conn) catch {};
        return;
    }
    const ids = a.alloc(u32, facts.len) catch return;
    for (facts, ids) |fact, *id| {
        id.* = srv.slide.add(fact, source, now(srv));
        out.event("fact", .{ .id = id.*, .text = fact });
    }
    defer saveAll(srv, a);
    // facts learned before this server started, never checked since, asked once.
    if (recheck.check(srv, &cx, 0, .unjudged, &.{}, gone)) |before| {
        for (before) |r| out.event("recheck", .{ .id = @as(u32, 0), .fact = r.fact.id, .text = r.fact.text, .recalled = r.recalled, .was = r.recalled });
    } else |e| log.line("slide: the facts learned before could not be asked again: {s}", .{words(&cx, e)});
    // each fact its own lesson and block, so each keeps a gate of its own
    for (facts, ids) |fact, id| {
        if (gone.check()) break;
        srv.slide.mark(id, .learning);
        srv.slide.phase(id, .writing);
        out.event("learning", .{ .id = id });
        var kept = [1]bool{false};
        const written = now(srv);
        // a chat fact told to the model about itself ("Your name is Ada.") is the model's own, asked of it.
        const chat = std.mem.eql(u8, source, "chat");
        const self = chat and (if (about) |w| w == .self else teach.aboutSelf(srv, &cx, fact, gone) catch false);
        const document = !chat or (!self and about == null and !wording.firstPerson(fact));
        if (chat) log.line("slide: fact {d} \"{s}\" is learned as {s}", .{ id, fact, if (self) "a fact about the model itself" else if (document) "a document's fact (it is not about the user)" else "a fact about the user" });
        const gave = if (document) &.{} else giveWay(srv, a, &cx, &out, fact, id, self, gone);
        const plan = (teach.lesson(srv, &cx, &srv.teacher, &.{fact}, &kept, document, if (document) null else @as(?teach.About, if (self) .self else .user), facts, gone) catch |e| {
            missed(srv, &out, id, words(&cx, e));
            putBack(srv, a, gave, id);
            continue;
        }) orelse {
            missed(srv, &out, id, "too few clean answers to learn from");
            putBack(srv, a, gave, id);
            continue;
        };
        const writing = seconds(srv, written);
        log.line("slide: fact {d}'s lesson written in {d:.1} s", .{ id, writing });
        srv.slide.asked(id, wordings(a, plan));
        const result = rounds(srv, a, &cx, plan, gone, id, &out);
        const tk = result.took;
        log.line("slide: fact {d}'s lesson took {d:.1} s: writing {d:.1}, fitting {d:.1}, checking {d:.1} (held-out {d:.1}, the facts before {d:.1}, broad questions {d:.1})", .{ id, seconds(srv, written), writing, tk.fit, tk.check + tk.recheck + tk.broad, tk.check, tk.recheck, tk.broad });
        const back = result.recalled.len == 1 and result.recalled[0];
        if (!back) putBack(srv, a, gave, id);
        if (back) slide_teacher.refresh(srv, &cx, &srv.teacher, plan.stale, gone) catch srv.teacher.forgetAnswers(srv.io);
        if (result.why) |why| srv.slide.note(id, why);
        srv.slide.mark(id, if (back) .learned else .missed);
        if (back and document) for (plan.held) |h| if (h.subject.len > 0 and h.context.len == 0) srv.teacher.remember(h.subject, h.question, h.answer) catch {};
        if (result.why) |why| {
            out.event("learned", .{ .id = id, .recalled = back, .steps = result.steps, .message = why });
        } else out.event("learned", .{ .id = id, .recalled = back, .steps = result.steps });
    }
    sse.done(conn) catch {};
}

/// Rounds in a row whose held-out questions all come back when asked that end a lesson.
fn confirm() u32 {
    const v = std.c.getenv("TF_SLIDE_ROUNDS") orelse return 2;
    return if (std.mem.eql(u8, std.mem.span(v), "one")) 1 else 2;
}

/// Rounds a lesson gets to bring its facts back when asked once, before it is given up.
const give_up = 5;

/// What a lesson's rounds left: which facts come back, the steps taken, and why it stopped early (null: it did not).
const Rounds = struct { recalled: []bool, steps: u32 = 0, why: ?[]const u8 = null, took: Took = .{} };

/// Where a lesson's rounds spent their time (s): fitting (the learner's runs), its held-out checks.
const Took = struct { fit: f64 = 0, check: f64 = 0, recheck: f64 = 0, broad: f64 = 0 };

/// A lesson in checked rounds until its fact comes back; a question it took is handed back (a few times at most).
fn rounds(srv: *Server, a: Allocator, cx: *errors.Cx, plan: teach.Plan, gone: Gone, id: u32, o: *Out) Rounds {
    var out: Rounds = .{ .recalled = a.alloc(bool, plan.facts.len) catch &.{} };
    @memset(out.recalled, false);
    var routed: u32 = 0;
    var repaired: u32 = 0;
    // the facts learned before, as they stood when this lesson began: what its checks find forgotten is against these
    const before = srv.slide.earlier(a, id) catch &.{};
    // broad questions as answered before it, which a kept lesson's answers are compared with for made-up names
    var at = now(srv);
    const broad_before = recheck.askBroad(srv, cx, gone) catch &.{};
    out.took.broad += seconds(srv, at);
    var unchecked = false; // the lesson was fitted again after its last check
    var streak: u32 = 0; // rounds in a row whose held-out questions came back when asked
    var ever = false; // a round whose held-out questions all came back when asked
    for (0..teach.rounds + teach.reroutes + recheck.repairs) |round| {
        const began = now(srv);
        var request = plan.full;
        request.more = round > 0;
        var held = false;
        srv.slide.round(id, @intCast(round + 1));
        const ran = run(srv, a, &request, id);
        out.took.fit += seconds(srv, began);
        switch (ran) {
            .learned => |l| {
                out.steps += l.steps;
                held = l.recalled;
            },
            .failed => |message| {
                out.why = message;
                return out;
            },
            else => return out,
        }
        const learned = seconds(srv, began);
        unchecked = false;
        const checked = now(srv);
        srv.slide.phase(id, .checking);
        const verdict = check.verify(srv, cx, plan, held or round + 1 >= teach.rounds, gone) catch |e| check.Verdict{ .recalled = out.recalled, .damage = words(cx, e) };
        out.took.check += seconds(srv, checked);
        srv.slide.checked(id, verdict.damage == null and std.mem.allEqual(bool, verdict.recalled, true));
        log.line("slide: round {d}: {d} steps ({d:.1} s learning, {d:.1} s checking); the held-out answers {s} token for token, and {s} when asked", .{ round + 1, out.steps, learned, seconds(srv, checked), if (held) "come back" else "do not come back", if (std.mem.allEqual(bool, verdict.recalled, true)) "come back" else "do not come back" });
        if (verdict.took) |took| {
            if (routed == teach.reroutes) {
                _ = run(srv, a, &.{ .drop = true }, 0);
                srv.slide.placed(id, &.{});
                out.why = std.fmt.allocPrint(a, "taken out: it kept answering questions about something else, last \"{s}\"", .{took.question}) catch "taken out: it kept answering questions about something else";
                @memset(out.recalled, false);
                return out;
            }
            routed += 1;
            const back = teach.routed(srv, cx, took) catch |e| {
                out.why = words(cx, e);
                return out;
            };
            at = now(srv);
            const rerouted = run(srv, a, &.{ .unasked = &.{back.prompt}, .kept = &.{back.kept}, .reroute = true }, id);
            out.took.fit += seconds(srv, at);
            switch (rerouted) {
                .failed => |message| {
                    out.why = message;
                    return out;
                },
                else => {},
            }
            log.line("slide: \"{s}\" is handed back to the lesson, its answer to stay as it was", .{took.question});
            srv.slide.note(id, std.fmt.allocPrint(a, "handed back to the lesson: \"{s}\"", .{took.question}) catch "");
            continue;
        }
        if (verdict.damage) |why| {
            _ = run(srv, a, &.{ .undo = true }, 0);
            srv.slide.placed(id, &.{});
            @memset(out.recalled, false); // the undo took the whole fact out: nothing of it is kept
            out.why = std.fmt.allocPrint(a, "round {d} taken back: {s}", .{ round + 1, why }) catch why;
            return out;
        }
        @memcpy(out.recalled, verdict.recalled);
        streak = if (std.mem.allEqual(bool, out.recalled, true)) streak + 1 else 0;
        ever = ever or streak > 0;
        // rounds that never once brought the fact back when asked do not start to: give up rather than learn on
        if (!ever and round + 1 >= give_up) break;
        // done once its answers come back token for token, or come back when asked two rounds running.
        if ((held or streak >= confirm()) and std.mem.allEqual(bool, out.recalled, true) and round + 1 >= teach.min_rounds) {
            at = now(srv);
            const lost = earlierLost(srv, cx, o, id, before, gone) catch |e| {
                takeOut(srv, a, &out, id, std.fmt.allocPrint(a, "taken out: the facts learned before could not be asked again ({s})", .{words(cx, e)}) catch "taken out");
                return out;
            };
            out.took.recheck += seconds(srv, at);
            if (lost.len == 0) {
                at = now(srv);
                if (broad_before.len > 0) drift(srv, cx, o, id, plan, broad_before, gone);
                out.took.broad += seconds(srv, at);
                // a diagnostic: every kept lesson taken out again, to see the facts before it as they were
                if (std.c.getenv("TF_SLIDE_TAKE_OUT_CHECK") != null) {
                    takeOut(srv, a, &out, id, "taken out to check that a take-out puts the facts before it back (TF_SLIDE_TAKE_OUT_CHECK)");
                    afterTakeOut(srv, cx, o, id, before, &.{}, gone);
                }
                return out;
            }
            const who = recheck.names(a, lost) catch "an earlier fact";
            if (repaired < recheck.repairs) {
                const back = recheck.keeps(srv, cx, lost) catch |e| {
                    takeOut(srv, a, &out, id, words(cx, e));
                    return out;
                };
                if (back.kept.len > 0) {
                    repaired += 1;
                    log.line("slide: fact {d}'s lesson made the model forget {s}: fitted again with their held-out answers kept as they were (repair {d} of {d})", .{ id, who, repaired, recheck.repairs });
                    o.event("repair", .{ .id = id, .message = std.fmt.allocPrint(a, "it made the model forget {s}: fitted again with their answers kept", .{who}) catch who });
                    at = now(srv);
                    const refit = run(srv, a, &.{ .unasked = back.prompts, .kept = back.kept, .reroute = true }, id);
                    out.took.fit += seconds(srv, at);
                    switch (refit) {
                        .failed => |message| {
                            takeOut(srv, a, &out, id, message);
                            return out;
                        },
                        else => {},
                    }
                    unchecked = true;
                    continue;
                }
            }
            takeOut(srv, a, &out, id, if (repaired > 0) std.fmt.allocPrint(a, "taken out: it made the model forget {s}, still after {d} repairs", .{ who, repaired }) catch who else std.fmt.allocPrint(a, "taken out: it made the model forget {s}", .{who}) catch who);
            // the facts asked once more with it out, as they will be saved: the ones it made forget are back
            afterTakeOut(srv, cx, o, id, before, lost, gone);
            return out;
        }
    }
    // it never came back when asked, or was fitted again and never checked: nothing of it stays in the model
    if (unchecked) {
        takeOut(srv, a, &out, id, "taken out: its last repair was never checked");
        afterTakeOut(srv, cx, o, id, before, &.{}, gone);
        return out;
    }
    if (!std.mem.allEqual(bool, out.recalled, true)) {
        _ = run(srv, a, &.{ .drop = true }, 0);
        srv.slide.placed(id, &.{});
    }
    return out;
}

/// The wordings a lesson's checks ask: its held-out questions, then the ones it never learned, with their answers.
fn wordings(a: Allocator, plan: teach.Plan) []const slide_graph.Asked {
    const out = a.alloc(slide_graph.Asked, plan.held.len + plan.probes.len) catch return &.{};
    for (plan.held, out[0..plan.held.len]) |h, *w| w.* = .{ .question = h.question, .answer = h.answer, .short = h.short, .context = h.context };
    for (plan.probes, out[plan.held.len..]) |h, *w| w.* = .{ .question = h.question, .answer = h.answer, .short = h.short, .probe = true };
    return out;
}

/// The facts learned before fact `id` asked again, each verdict on the stream; those that came back before and not now.
fn earlierLost(srv: *Server, cx: *errors.Cx, o: *Out, id: u32, before: []const slide_graph.Earlier, gone: Gone) ![]const recheck.Result {
    const began = now(srv);
    const results = try recheck.check(srv, cx, id, .all, before, gone);
    var back: u32 = 0;
    for (results) |r| {
        back += @intFromBool(r.recalled);
        o.event("recheck", .{ .id = id, .fact = r.fact.id, .text = r.fact.text, .recalled = r.recalled, .was = r.was });
    }
    const lost = try recheck.forgotten(cx.a, results);
    if (results.len == 0) return lost;
    const took = seconds(srv, began);
    log.line("slide: fact {d}'s lesson: the {d} facts learned before asked again in {d:.1} s: {d} come back, {d} forgotten", .{ id, results.len, took, back, lost.len });
    o.event("rechecked", .{ .id = id, .facts = @as(u32, @intCast(results.len)), .recalled = back, .forgotten = @as(u32, @intCast(lost.len)), .seconds = took });
    return lost;
}

/// The broad questions asked again after a kept lesson: a reply that names what no learned fact said is flagged.
fn drift(srv: *Server, cx: *errors.Cx, o: *Out, id: u32, plan: teach.Plan, before: []const []const u8, gone: Gone) void {
    const a = cx.a;
    const now_said = recheck.askBroad(srv, cx, gone) catch return;
    var facts: std.ArrayList([]const u8) = .empty;
    facts.appendSlice(a, plan.facts) catch return;
    for (srv.slide.earlier(a, id) catch &.{}) |e| facts.append(a, e.text) catch return;
    for (recheck.broad, before, now_said) |q, was, reply| if (recheck.invented(reply, was, q, facts.items)) |name| {
        log.line("slide: fact {d}'s lesson: \"{s}\" now names \"{s}\", which no learned fact says: {s}", .{ id, q, name, reply });
        o.event("drift", .{ .id = id, .question = q, .name = name, .reply = reply });
    };
}

/// The facts learned before asked again once a lesson is taken out (the pocket put back as the lesson found it).
fn afterTakeOut(srv: *Server, cx: *errors.Cx, o: *Out, id: u32, before: []const slide_graph.Earlier, lost: []const recheck.Result, gone: Gone) void {
    const still = earlierLost(srv, cx, o, id, before, gone) catch |e| return log.line("slide: fact {d}: the facts learned before could not be asked again after its take-out: {s}", .{ id, words(cx, e) });
    var back: u32 = 0;
    for (lost) |l| {
        for (still) |x| {
            if (x.fact.id == l.fact.id) break;
        } else back += 1;
    }
    log.line("slide: fact {d} taken out: of the {d} facts it made forget, {d} come back; {d} facts that came back before it do not now", .{ id, lost.len, back, still.len });
    o.event("restored", .{ .id = id, .forgotten = @as(u32, @intCast(lost.len)), .back = back, .lost = @as(u32, @intCast(still.len)) });
}

/// The open lesson taken out whole (its neurons free, earlier ones it fitted again as they were), and why.
fn takeOut(srv: *Server, a: Allocator, out: *Rounds, id: u32, why: []const u8) void {
    _ = run(srv, a, &.{ .drop = true }, 0);
    srv.slide.placed(id, &.{});
    @memset(out.recalled, false);
    out.why = why;
    log.line("slide: fact {d}: {s}", .{ id, why });
}

/// Every kept lesson into the model's shards, timed, with the shards it changed.
fn saveAll(srv: *Server, a: Allocator) void {
    srv.slide.saving(folder(srv));
    const began = now(srv);
    const failed: ?[]const u8 = switch (run(srv, a, &.{ .save = true }, 0)) {
        .failed => |message| message,
        .refused => "the engine could not take the save",
        else => null,
    };
    if (failed) |why| log.line("slide: the learned neurons could not be saved into the model's files: {s}", .{why});
    srv.slide.saveEnded(folder(srv), seconds(srv, began), failed);
}

/// Earlier chat facts `fact` replaces ("I like blue." for "Now I like red."): their neurons zeroed, marked replaced.
fn giveWay(srv: *Server, a: Allocator, cx: *errors.Cx, out: *Out, fact: []const u8, id: u32, self: bool, gone: Gone) []const slide_graph.Fact {
    const olds = replace.replaced(srv, cx, fact, id, self, gone) catch return &.{};
    if (olds.len == 0) return &.{};
    var neurons: std.ArrayList(api.Neuron) = .empty;
    for (olds) |o| neurons.appendSlice(a, o.neurons) catch return &.{};
    switch (run(srv, a, &.{ .retire = neurons.items }, 0)) {
        .failed, .refused => return &.{},
        else => {},
    }
    for (olds) |o| {
        srv.slide.note(o.id, std.fmt.allocPrint(a, "replaced by fact {d}", .{id}) catch "replaced");
        srv.slide.mark(o.id, .replaced);
        out.event("replaces", .{ .id = id, .fact = o.id, .text = o.text });
    }
    srv.teacher.forgetBank(srv.io);
    return olds;
}

/// The facts fact `id` replaced, put back as they were: it was not learned, so what they said still stands.
fn putBack(srv: *Server, a: Allocator, olds: []const slide_graph.Fact, id: u32) void {
    if (olds.len == 0) return;
    switch (run(srv, a, &.{ .unretire = true }, 0)) {
        .failed, .refused => return,
        else => {},
    }
    for (olds) |o| {
        log.line("slide: fact {d} was not learned, so \"{s}\" (fact {d}) is put back", .{ id, o.text, o.id });
        srv.slide.note(o.id, std.fmt.allocPrint(a, "put back: fact {d}, which replaced it, was not learned", .{id}) catch "put back");
        srv.slide.mark(o.id, .learned);
    }
    srv.teacher.forgetBank(srv.io);
}

/// A fact the lesson could not learn, and why.
fn missed(srv: *Server, out: *Out, id: u32, why: []const u8) void {
    srv.slide.note(id, why);
    srv.slide.mark(id, .missed);
    out.event("learned", .{ .id = id, .recalled = false, .message = why });
}

const Outcome = union(enum) {
    none,
    learned: struct { recalled: bool, steps: u32 },
    failed: []const u8,
    refused: api.LearnError,
    scored: []const bool,
};

/// One lesson through the engine's learner, waited for to its end; fact `id`'s progress and neurons go to the graph.
pub fn run(srv: *Server, a: Allocator, request: *const api.LearnRequest, id: u32) Outcome {
    var box: Box = .{ .io = srv.io, .gpa = srv.gpa };
    defer box.deinit();
    srv.engine.learn(request, box.sink()) catch |e| return .{ .refused = e };
    var outcome: Outcome = .none;
    var batch: std.ArrayList(Box.Copy) = .empty;
    defer batch.deinit(srv.gpa);
    while (true) {
        const finished = box.take(&batch);
        for (batch.items) |c| {
            switch (c.kind) {
                .learned => {
                    outcome = .{ .learned = .{ .recalled = c.recalled, .steps = c.steps } };
                    if (id != 0) srv.slide.placed(id, c.neurons);
                },
                .progress => if (id != 0) {
                    var p = c.progress;
                    p.neurons = c.neurons;
                    srv.slide.progress(id, p);
                },
                .done => if (c.text.len > 0) {
                    outcome = .{ .failed = a.dupe(u8, c.text) catch "the learner failed" };
                },
                .scored => outcome = if (a.dupe(bool, c.same)) |same| .{ .scored = same } else |_| .none,
            }
            srv.gpa.free(c.text);
            srv.gpa.free(c.neurons);
            srv.gpa.free(c.same);
        }
        batch.clearRetainingCapacity();
        if (finished) return outcome;
    }
}

/// The learn stream: events while the client listens.
const Out = struct {
    conn: *Conn,
    a: Allocator,
    open: bool = true,

    fn event(o: *Out, kind: []const u8, fields: anytype) void {
        if (!o.open) return;
        const obj = json.newObject(o.a) catch return;
        inline for (@typeInfo(@TypeOf(fields)).@"struct".field_names) |name| {
            const v = @field(fields, name);
            const value: json.Value = switch (@TypeOf(v)) {
                bool => .{ .bool = v },
                u32 => json.intValue(o.a, v) catch return,
                f64 => .{ .float = @round(v * 10) / 10 },
                else => .{ .string = v },
            };
            obj.put(o.a, name, value) catch return;
        }
        sse.event(o.conn, o.a, kind, .{ .object = obj }) catch {
            o.open = false;
        };
    }

    fn fail(o: *Out, cx: *errors.Cx, e: anyerror) void {
        o.event("failed", .{ .message = words(cx, e) });
        sse.done(o.conn) catch {};
    }
};

fn words(cx: *errors.Cx, e: anyerror) []const u8 {
    return if (e == error.Refused and cx.message.len > 0) cx.message else @errorName(e);
}

fn now(srv: *Server) i64 {
    return @intCast(@divTrunc(std.Io.Clock.real.now(srv.io).toNanoseconds(), std.time.ns_per_ms));
}

/// Seconds since `since` (from `now`).
fn seconds(srv: *Server, since: i64) f64 {
    return @as(f64, @floatFromInt(now(srv) - since)) / 1000;
}

/// The model's folder, whose layers and shards the view shows (null: unknown).
fn folder(srv: *Server) ?[]const u8 {
    return if (srv.config.model_dir.len == 0) null else srv.config.model_dir;
}

fn refuse(conn: *Conn, a: Allocator, code: u16, message: []const u8) void {
    const o = json.newObject(a) catch return;
    o.put(a, "message", .{ .string = message }) catch return;
    o.put(a, "type", .{ .string = "invalid_request_error" }) catch return;
    const wrapped = json.newObject(a) catch return;
    wrapped.put(a, "error", .{ .object = o }) catch return;
    routes.sendValue(conn, a, code, .{ .object = wrapped });
}

/// Learn events copied off the engine's thread, read by the request's own thread.
const Box = struct {
    io: std.Io,
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    events: std.ArrayList(Copy) = .empty,
    done: bool = false,

    const Copy = struct { kind: std.meta.Tag(api.LearnEvent), text: []const u8 = "", recalled: bool = false, steps: u32 = 0, progress: api.LearnProgress = .{ .phase = .reading }, neurons: []const api.Neuron = &.{}, same: []const bool = &.{} };

    fn sink(b: *Box) api.LearnSink {
        return .{ .ctx = b, .event = onEvent };
    }

    fn onEvent(ctx: *anyopaque, e: *const api.LearnEvent) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        var c: Copy = .{ .kind = e.* };
        switch (e.*) {
            .learned => |l| {
                c.recalled = l.recalled;
                c.steps = l.steps;
                c.neurons = b.gpa.dupe(api.Neuron, l.neurons) catch &.{};
            },
            .progress => |p| {
                c.progress = p;
                c.progress.neurons = &.{};
                c.neurons = b.gpa.dupe(api.Neuron, p.neurons) catch &.{};
            },
            .done => |d| c.text = b.gpa.dupe(u8, d.message) catch "",
            .scored => |x| c.same = b.gpa.dupe(bool, x.same) catch &.{},
        }
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        b.events.append(b.gpa, c) catch {
            b.gpa.free(c.text);
            b.gpa.free(c.neurons);
            b.gpa.free(c.same);
        };
        if (e.* == .done) b.done = true;
        b.cond.signal(b.io);
    }

    /// Waits for events and moves them to `out`; true once `done` has arrived.
    fn take(b: *Box, out: *std.ArrayList(Copy)) bool {
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        while (b.events.items.len == 0 and !b.done) b.cond.waitUncancelable(b.io, &b.mutex);
        out.appendSlice(b.gpa, b.events.items) catch {};
        b.events.clearRetainingCapacity();
        return b.done;
    }

    fn deinit(b: *Box) void {
        for (b.events.items) |c| {
            b.gpa.free(c.text);
            b.gpa.free(c.neurons);
            b.gpa.free(c.same);
        }
        b.events.deinit(b.gpa);
    }
};
