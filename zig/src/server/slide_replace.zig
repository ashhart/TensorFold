//! A chat fact that changes an earlier one ("Now I like red." after "I like blue."): the earlier one gives way.
const std = @import("std");
const errors = @import("errors.zig");
const log = @import("log.zig");
const prompts = @import("slide_prompts.zig");
const slide_ask = @import("slide_ask.zig");
const slide_graph = @import("slide_graph.zig");
const side = @import("slide_side.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;

const candidates = 32; // learned chat facts a new one is checked against, the latest first

/// The learned chat facts `fact` replaces: it changes or takes back what they say, or they no longer hold.
pub fn replaced(srv: *Server, cx: *Cx, fact: []const u8, id: u32, gone: anytype) ![]const slide_graph.Fact {
    const a = cx.a;
    const olds = try srv.slide.learnedChat(a, id, candidates);
    if (olds.len == 0) return &.{};
    const yes = try a.alloc(bool, olds.len);
    @memset(yes, false);
    const Job = struct {
        srv: *Server,
        a: Allocator,
        fact: []const u8,
        olds: []const slide_graph.Fact,
        gone: @TypeOf(gone),
        yes: []bool,

        pub fn run(j: *const @This(), i: usize) !void {
            var jcx: Cx = .{ .a = j.a };
            const old = j.olds[i];
            // "Now I like red." after "My favourite colour is red." says it again: the earlier one stays
            const same = try slide_ask.askBase(j.srv, &jcx, null, try std.fmt.allocPrint(j.a, prompts.again_prompt, .{ old.text, j.fact }), 4, j.gone);
            if (saysYes(same.content)) return;
            const said = try slide_ask.askBase(j.srv, &jcx, null, try std.fmt.allocPrint(j.a, prompts.replace_prompt, .{ old.text, j.fact }), 4, j.gone);
            // a model may say "My favourite colour is red." changes nothing in "I like blue.", yet not that it holds
            const still = if (saysYes(said.content)) "" else (try slide_ask.askBase(j.srv, &jcx, null, try std.fmt.allocPrint(j.a, prompts.still_prompt, .{ old.text, j.fact }), 4, j.gone)).content;
            j.yes[i] = givesWay(same.content, said.content, still);
        }
    };
    try side.each(olds.len, &Job{ .srv = srv, .a = a, .fact = fact, .olds = olds, .gone = gone, .yes = yes });
    var out: std.ArrayList(slide_graph.Fact) = .empty;
    for (olds, yes) |o, y| if (y) {
        log.line("slide: \"{s}\" replaces \"{s}\" (fact {d}), whose neurons are taken out first", .{ fact, o.text, o.id });
        try out.append(a, o);
    };
    return out.items;
}

/// Whether an earlier fact gives way, from the model's replies: not said again, and changed or no longer true.
fn givesWay(again: []const u8, changes: []const u8, still: []const u8) bool {
    return !saysYes(again) and (saysYes(changes) or std.ascii.startsWithIgnoreCase(std.mem.trim(u8, still, " \t\r\n\"*"), "no"));
}

fn saysYes(reply: []const u8) bool {
    return std.ascii.startsWithIgnoreCase(std.mem.trim(u8, reply, " \t\r\n\"*"), "yes");
}

test "an earlier fact gives way when changed or no longer true, not for answering the same kind of question" {
    try std.testing.expect(givesWay("No.", "**Yes**", ""));
    try std.testing.expect(givesWay("No", "No", "No.")); // "My favourite colour is red." after "I like blue."
    try std.testing.expect(!givesWay("No", "No", "Yes")); // "My GitHub name is alex77." after "My name is Alex."
    try std.testing.expect(!givesWay("Yes", "Yes", "No"));
    try std.testing.expect(!givesWay("No", "No", "")); // no reply is no answer
}
