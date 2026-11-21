//! Corrections target a shipped belief, never an answer supplied by another pocket fact.
const std = @import("std");
const reads = @import("slide_correction_ask.zig");
const correction = @import("slide_correction.zig");
const teach = @import("slide_lesson.zig");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;

/// Refuse before training, naming the facts: their later deletion must not invalidate an old-answer reference.
pub fn independent(srv: *Server, cx: *errors.Cx, h: teach.Held, old: []const u8, ids: []const u32, gone: anytype) !void {
    const base = try reads.base(srv, cx, h.question, h.context, gone);
    if (!std.mem.eql(u8, base.finish_reason, "stop")) return error.CorrectionReferenceIncomplete;
    const changed = !std.mem.eql(u32, ids, try reads.ids(cx.a, base));
    const facts = try srv.slide.snapshot(cx.a);
    var names: std.ArrayList(u8) = .empty;
    for (facts) |f| {
        if (f.state != .learned or f.neurons.len == 0) continue;
        // A matching kept value is conservatively refused even if several pocket effects currently cancel.
        var matches = correction.oldReturned(h.question, old, old, f.text);
        for (f.answers) |x| matches = matches or correction.oldReturned(h.question, old, old, x.answer);
        if (!changed and !matches) continue;
        if (names.items.len > 0) try names.appendSlice(cx.a, ", ");
        try names.print(cx.a, "fact {d} (\"{s}\")", .{ f.id, f.text });
    }
    if (!changed and names.items.len == 0) return;
    if (names.items.len == 0) try names.appendSlice(cx.a, "pocket neurons missing from this graph");
    return cx.fail(.request, "correction refused: its old answer depends on, or matches, {s}; delete those facts before correcting it", .{names.items});
}
