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
            // "Now I like red." after "My favourite colour is red." says it again, and no sure answer keeps it too
            const same = try slide_ask.askBase(j.srv, &jcx, null, try std.fmt.allocPrint(j.a, prompts.again_prompt, .{ old.text, j.fact }), 4, j.gone);
            const again = verdict(same.content, same.finish_reason);
            if (again != .no) return;
            const said = try slide_ask.askBase(j.srv, &jcx, null, try std.fmt.allocPrint(j.a, prompts.replace_prompt, .{ old.text, j.fact }), 4, j.gone);
            const changes = verdict(said.content, said.finish_reason);
            // a model may say "My favourite colour is red." changes nothing in "I like blue.", yet not that it holds
            const still: Answer = if (changes == .yes) .unknown else blk: {
                const r = try slide_ask.askBase(j.srv, &jcx, null, try std.fmt.allocPrint(j.a, prompts.still_prompt, .{ old.text, j.fact }), 4, j.gone);
                break :blk verdict(r.content, r.finish_reason);
            };
            j.yes[i] = givesWay(again, changes, still);
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

const Answer = enum { yes, no, unknown };

/// A judge's whole answer: yes or no alone (case, quotes, stars and one closing stop aside) in a reply that ended.
fn verdict(content: []const u8, finish_reason: []const u8) Answer {
    if (!std.mem.eql(u8, finish_reason, "stop")) return .unknown;
    var word = std.mem.trim(u8, content, " \t\r\n\"*");
    if (word.len > 0 and (word[word.len - 1] == '.' or word[word.len - 1] == '!')) word = std.mem.trim(u8, word[0 .. word.len - 1], " \t\r\n\"*");
    if (std.ascii.eqlIgnoreCase(word, "yes")) return .yes;
    if (std.ascii.eqlIgnoreCase(word, "no")) return .no;
    return .unknown;
}

/// Whether an earlier fact gives way: surely not said again, and surely changed or surely no longer true.
fn givesWay(again: Answer, changes: Answer, still: Answer) bool {
    return again == .no and (changes == .yes or still == .no);
}

test "a judge's answer is a whole yes or no from a reply that ended; prefixes and cut replies are no answer" {
    const stop = "stop";
    for ([_][]const u8{ "Yes", "yes.", "**Yes**", "\"Yes!\"", " YES \n" }) |r| try std.testing.expectEqual(Answer.yes, verdict(r, stop));
    for ([_][]const u8{ "No", "no.", "**No**", "\"No.\"" }) |r| try std.testing.expectEqual(Answer.no, verdict(r, stop));
    for ([_][]const u8{ "Not sure", "None", "Nobody said that", "No idea", "Yesterday", "Yes, but", "", "No no" }) |r|
        try std.testing.expectEqual(Answer.unknown, verdict(r, stop));
    try std.testing.expectEqual(Answer.unknown, verdict("No", "length")); // four tokens cut "No, it ..."
    try std.testing.expectEqual(Answer.unknown, verdict("Yes", "length"));
}

test "an earlier fact gives way when changed or no longer true, not for answering the same kind of question" {
    try std.testing.expect(givesWay(.no, .yes, .unknown));
    try std.testing.expect(givesWay(.no, .no, .no)); // "My favourite colour is red." after "I like blue."
    try std.testing.expect(givesWay(.no, .unknown, .no)); // a sure "no longer true" is enough beside an unsure change
    try std.testing.expect(!givesWay(.no, .no, .yes)); // "My GitHub name is alex77." after "My name is Alex."
    try std.testing.expect(!givesWay(.yes, .yes, .no));
    try std.testing.expect(!givesWay(.unknown, .yes, .no)); // an unsure "said again" keeps the earlier one
    try std.testing.expect(!givesWay(.no, .no, .unknown)); // no reply is no answer
    try std.testing.expect(!givesWay(.no, .unknown, .unknown));
}
