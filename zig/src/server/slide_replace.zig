//! A chat fact that changes an earlier one ("Now I like red." after "I like blue."): the earlier one gives way.
const std = @import("std");
const errors = @import("errors.zig");
const log = @import("log.zig");
const prompts = @import("slide_prompts.zig");
const slide_ask = @import("slide_ask.zig");
const slide_graph = @import("slide_graph.zig");
const side = @import("slide_side.zig");
const take = @import("slide_take.zig");
const wording = @import("slide_words.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;

const candidates = 32; // learned chat facts a new one is checked against, the latest first

/// The learned chat facts `fact` replaces: it answers one of their questions, or changes or takes back what they say.
pub fn replaced(srv: *Server, cx: *Cx, fact: []const u8, id: u32, self: bool, gone: anytype) ![]const slide_graph.Fact {
    const a = cx.a;
    var kin: std.ArrayList(slide_graph.Fact) = .empty;
    for (try srv.slide.learnedChat(a, id, candidates)) |o| if (wording.addressesModel(o.text) == self) try kin.append(a, o);
    const olds = kin.items;
    if (olds.len == 0) return &.{};
    const yes = try a.alloc(bool, olds.len);
    @memset(yes, false);
    const Job = struct {
        srv: *Server,
        a: Allocator,
        fact: []const u8,
        self: bool,
        olds: []const slide_graph.Fact,
        gone: @TypeOf(gone),
        yes: []bool,

        pub fn run(j: *const @This(), i: usize) !void {
            var jcx: Cx = .{ .a = j.a };
            const old = j.olds[i];
            // "Now I like red." after "My favourite colour is red." says it again: the earlier one stays
            const same = try slide_ask.askBase(j.srv, &jcx, null, try std.fmt.allocPrint(j.a, prompts.again_prompt, .{ old.text, j.fact }), 4, j.gone);
            if (std.ascii.startsWithIgnoreCase(std.mem.trim(u8, same.content, " \t\r\n\"*"), "yes")) return;
            // answering an earlier fact's question is not enough: "Sam is a contributor to TensorFold." answers "What.
            if (j.self) {
                const answers = for (old.questions) |q| {
                    if (try take.judgedAs(j.srv, &jcx, j.fact, q, false, true, j.gone)) break true;
                } else false;
                if (!answers) return;
            }
            const said = try slide_ask.askBase(j.srv, &jcx, null, try std.fmt.allocPrint(j.a, prompts.replace_prompt, .{ old.text, j.fact }), 4, j.gone);
            j.yes[i] = std.ascii.startsWithIgnoreCase(std.mem.trim(u8, said.content, " \t\r\n\"*"), "yes");
        }
    };
    try side.each(olds.len, &Job{ .srv = srv, .a = a, .fact = fact, .self = self, .olds = olds, .gone = gone, .yes = yes });
    var out: std.ArrayList(slide_graph.Fact) = .empty;
    for (olds, yes) |o, y| if (y) {
        log.line("slide: \"{s}\" replaces \"{s}\" (fact {d}), whose neurons are taken out first", .{ fact, o.text, o.id });
        try out.append(a, o);
    };
    return out.items;
}
