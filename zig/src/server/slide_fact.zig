//! One fact's writer, fitting and acceptance gate, with opt-in capacity retries after checked deletion.
const std = @import("std");
const slide = @import("slide.zig");
const errors = @import("errors.zig");
const teach = @import("slide_lesson.zig");
const wording = @import("slide_words.zig");
const talks = @import("slide_context.zig");
const slide_teacher = @import("slide_teacher.zig");
const log = @import("log.zig");
const Server = @import("server.zig").Server;

pub fn learn(srv: *Server, a: std.mem.Allocator, cx: *errors.Cx, out: *slide.Out, fact: []const u8, id: u32, source: []const u8, facts: []const []const u8, supplied: []const []const u8, correcting: bool, gone: anytype) void {
    srv.slide.mark(id, .learning);
    srv.slide.phase(id, .writing);
    out.event("learning", .{ .id = id });
    out.stage(id, "writing");
    var retired = false;
    while (true) {
        var kept = [1]bool{false};
        const written = slide.now(srv);
        // a chat fact not about the user ("The Quillon bridge is 1,312 metres long.") learns as a document's does
        const document = !std.mem.eql(u8, source, "chat") or !wording.firstPerson(fact);
        var gave = if (document or correcting) &.{} else slide.giveWay(srv, a, cx, fact, id, gone);
        var plan = (teach.lesson(srv, cx, &srv.teacher, &.{fact}, &kept, document, facts, supplied, gone) catch |e| {
            slide.missed(srv, out, id, slide.words(cx, e));
            slide.putBack(srv, a, gave, id);
            return;
        }) orelse {
            slide.missed(srv, out, id, "too few clean answers to learn from");
            slide.putBack(srv, a, gave, id);
            return;
        };
        if (correcting) @import("slide_correction.zig").prepare(srv, cx, &plan, gone) catch |e| {
            slide.missed(srv, out, id, slide.words(cx, e));
            slide.putBack(srv, a, gave, id);
            return;
        };
        plan.full.reuse_deleted = retired;
        log.line("slide: fact {d}'s lesson written in {d:.1} s", .{ id, slide.seconds(srv, written) });
        @import("slide_correction.zig").record(srv, cx, id, plan) catch |e| {
            slide.missed(srv, out, id, slide.words(cx, e));
            slide.putBack(srv, a, gave, id);
            return;
        };
        var result = slide.rounds(srv, a, cx, plan, gone, id, out);
        if (!retired and srv.config.slide_forget and result.why != null and std.mem.eql(u8, result.why.?, "PocketFull")) {
            srv.teacher.ended(false);
            slide.putBack(srv, a, gave, id);
            gave = &.{};
            if (tryCapacity(srv, cx, out, id, &result.why, gone)) {
                retired = true;
                srv.slide.phase(id, .writing);
                out.stage(id, "writing");
                continue;
            }
        }

        const back = result.recalled.len == 1 and result.recalled[0];
        srv.teacher.ended(back); // a lesson given up leaves the next one's checks as if it never ran
        if (!back) slide.putBack(srv, a, gave, id);
        if (back) slide_teacher.refresh(srv, cx, &srv.teacher, plan.stale, gone) catch srv.teacher.forgetAnswers(srv.io);
        if (result.why) |why| srv.slide.note(id, why);
        srv.slide.mark(id, if (back) .learned else .missed);
        if (back) talks.chainAfter(srv, id, plan);
        if (back and document) for (plan.held) |h| if (h.subject.len > 0 and h.context.len == 0) srv.teacher.remember(h.subject, h.question, h.answer) catch {};
        if (result.why) |why| {
            out.event("learned", .{ .id = id, .recalled = back, .steps = result.steps, .message = why });
        } else out.event("learned", .{ .id = id, .recalled = back, .steps = result.steps });
        return;
    }
}

fn tryCapacity(srv: *Server, cx: *errors.Cx, out: *slide.Out, id: u32, why: *?[]const u8, gone: anytype) bool {
    return @import("slide_capacity.zig").makeRoom(srv, cx, out, id, why.*, gone) catch |e| {
        why.* = if (e == error.UsageAuditUnavailable) "automatic forgetting requires measured pocket usage for this engine" else slide.words(cx, e);
        return false;
    };
}
