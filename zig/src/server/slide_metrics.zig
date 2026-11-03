//! Cumulative drift against recorded untaught replies and fixed, caller-supplied recall wordings.
const std = @import("std");
const json = @import("json");
const Server = @import("server.zig").Server;
const Cx = @import("errors.zig").Cx;
const asks = @import("slide_ask.zig");
const verify = @import("slide_verify.zig");
const words = @import("slide_words.zig");
const teach = @import("slide_lesson.zig");
const store = @import("slide_metrics_store.zig");
const schema = @import("slide_metrics_types.zig");
const panel = @import("slide_panel.zig");
const graph = @import("slide_graph.zig");
const chat = @import("chat.zig");
const Allocator = std.mem.Allocator;
pub const Report = schema.Report;
const Output = schema.Output;
const Row = schema.Row;
const Query = struct { kind: schema.Kind, question: []const u8, thinking: bool = false };

fn sha(a: Allocator, bytes: []const u8) ![]const u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(digest, .lower));
}

pub fn output(reply: chat.Reply) !Output {
    return .{ .content = reply.content, .sha256 = reply.runtime.strField("token_sha256") orelse return error.MissingTokenHash };
}

fn questions(a: Allocator, thinking: bool) ![]const Query {
    var qs: std.ArrayList(Query) = .empty;
    inline for (.{ "unrelated", "controls", "neighbours" }, .{ schema.Kind.unrelated, .control, .neighbour }) |name, kind| {
        for (@field(panel, name)) |q| {
            try qs.append(a, .{ .question = q, .kind = kind });
            if (thinking) try qs.append(a, .{ .question = q, .kind = kind, .thinking = true });
        }
    }
    return qs.toOwnedSlice(a);
}

pub fn panelSha(a: Allocator) ![]const u8 {
    return sha(a, try std.json.Stringify.valueAlloc(a, .{ panel.unrelated, panel.controls, panel.neighbours }, .{}));
}

fn profile(srv: *Server, cx: *Cx) ![]const u8 {
    return sha(cx.a, try std.json.Stringify.valueAlloc(cx.a, .{
        srv.info.name,             srv.config.served_name, srv.config.identity, srv.config.seed_salt,
        srv.text.templateSource(), srv.text.vocabSize(),
    }, .{}));
}

fn ask(srv: *Server, cx: *Cx, q: Query, base: bool, gone: anytype) !chat.Reply {
    if (q.thinking) return if (base) asks.askThinkingBase(srv, cx, q.question, gone) else asks.askThinking(srv, cx, q.question, gone);
    return if (base) asks.askBase(srv, cx, null, q.question, 64, gone) else asks.ask(srv, cx, null, q.question, 64, gone);
}

fn baseline(srv: *Server, cx: *Cx, qs: []const Query, gone: anytype, r: *Report) !schema.Baseline {
    const a = cx.a;
    const current = try a.alloc(Output, qs.len);
    for (qs, current) |q, *out| out.* = try output(try ask(srv, cx, q, true, gone));
    const fresh: schema.Baseline = .{ .panel_sha256 = r.panel_sha256, .profile = try profile(srv, cx), .replies = current };
    const saved = (try srv.slide.driftBaseline(a, try store.value(a, fresh))).?;
    const base = try store.decode(schema.Baseline, a, saved);
    if (base.version != fresh.version or !std.mem.eql(u8, base.panel_sha256, fresh.panel_sha256) or
        !std.mem.eql(u8, base.profile, fresh.profile) or base.replies.len != current.len)
        return error.DriftBaselineProfileChanged;
    for (base.replies, current) |old, now| r.untaught_mismatches += @intFromBool(!std.mem.eql(u8, old.sha256, now.sha256));
    return base;
}

/// All example roles are checked: an externally supplied wording is never credited if the plan reads it.
pub fn contaminated(srv: *Server, cx: *Cx, plan: teach.Plan, question: []const u8) !bool {
    const head = (try asks.promptOnly(srv, cx, cx.a, null, question)).ids;
    inline for (.{ "train", "think", "held", "near", "keep", "asked", "unasked", "kept" }) |role| {
        for (@field(plan.full, role)) |ex| if (std.mem.indexOf(u32, ex.ids, head) != null) return true;
    }
    return false;
}

/// Three greedy passes, before committing a candidate; this function never trains or changes a lesson's rows.
pub fn measure(srv: *Server, cx: *Cx, plan: teach.Plan, id: u32, gone: anytype) !Report {
    const a = cx.a;
    var r: Report = .{ .gate = srv.config.slide_drift_gate, .panel_sha256 = try panelSha(a) };
    const facts = try srv.slide.snapshot(a);
    var active: std.ArrayList(graph.Fact) = .empty;
    for (facts) |f| if (f.id == id or f.state == .learned) {
        try active.append(a, f);
        if (f.recall.len == 0) r.complete = false;
        if (f.id != id) {
            const previous = if (f.evaluation) |v| store.decode(Report, a, v) catch null else null;
            if (previous == null or !previous.?.complete) r.complete = false;
        }
        r.lessons += 1;
    };
    const qs = try questions(a, plan.thinks);
    const base = try baseline(srv, cx, qs, gone, &r);
    var rows: std.ArrayList(Row) = .empty;
    for (qs, base.replies) |q, before| {
        if (try contaminated(srv, cx, plan, q.question)) r.complete = false;
        var row: Row = .{ .kind = q.kind, .question = q.question, .thinking = q.thinking, .baseline = before, .replies = undefined, .classes = undefined };
        for (0..3) |run| {
            const after = try output(try ask(srv, cx, q, false, gone));
            row.replies[run] = after;
            const c = if (q.thinking) &r.thinking[run] else &r.plain[run];
            const equal = std.mem.eql(u8, before.sha256, after.sha256);
            const alike = !equal and words.alike(before.content, after.content);
            const same = !equal and !alike and try verify.same(srv, cx, q.question, before.content, after.content, gone);
            row.classes[run] = if (equal) .equal else if (alike) .alike else if (same) .same else .changed;
            if (q.kind == .unrelated) {
                c.checked += 1;
                c.byte_equal += @intFromBool(equal);
                c.alike += @intFromBool(alike);
                c.judged_same += @intFromBool(same);
                c.meaning_changed += @intFromBool(!equal and !alike and !same);
            }
            if (q.kind == .control) c.invented += @intFromBool(!equal);
            for (active.items) |f| if (words.tells(f.text, q.question, before.content, after.content)) {
                c.leaks += 1;
                break;
            };
        }
        try rows.append(a, row);
    }
    for (active.items) |f| {
        var all: [3]bool = @splat(true);
        for (f.recall) |h| {
            if (try contaminated(srv, cx, plan, h.question)) r.complete = false;
            for (0..if (plan.thinks) @as(usize, 2) else 1) |mode| {
                var row: Row = .{ .kind = .recall, .fact = f.id, .question = h.question, .thinking = mode == 1, .replies = undefined, .classes = undefined };
                for (0..3) |run| {
                    const after = try output(try ask(srv, cx, .{ .kind = .recall, .question = h.question, .thinking = mode == 1 }, false, gone));
                    row.replies[run] = after;
                    const back = words.recallsShort(f.text, h.question, h.short, after.content);
                    row.classes[run] = if (back) .recalled else .missed;
                    const c = if (mode == 1) &r.thinking[run] else &r.plain[run];
                    c.recall_asked += 1;
                    c.recall_passed += @intFromBool(back);
                    if (mode == 0) all[run] = all[run] and back;
                }
                try rows.append(a, row);
            }
        }
        if (f.id != id) for (0..3) |run| {
            r.plain[run].earlier_asked += 1;
            r.plain[run].earlier_passed += @intFromBool(f.recall.len > 0 and all[run]);
        };
    }
    r.rows = rows.items;
    r.refused = schema.refusal(r);
    return r;
}

/// A failed measurement is visible; it refuses only when the opt-in gate is enabled.
pub fn candidate(srv: *Server, cx: *Cx, plan: teach.Plan, id: u32, gone: anytype) ?[]const u8 {
    const measured = measure(srv, cx, plan, id, gone) catch |err| Report{
        .status = @errorName(err),
        .gate = srv.config.slide_drift_gate,
        .complete = false,
        .refused = "the fixed panel could not be measured",
    };
    srv.slide.evaluated(id, measured) catch return if (srv.config.slide_drift_gate) "the fixed panel receipt could not be saved" else null;
    return if (srv.config.slide_drift_gate) measured.refused else null;
}
