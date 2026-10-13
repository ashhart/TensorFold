//! A fact told in another language: the prompts that keep the model in it, and the steady prompts translated into it.
const std = @import("std");
const errors = @import("errors.zig");
const wording = @import("slide_words.zig");
const slide_ask = @import("slide_ask.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;
const ask = slide_ask.ask;
const askBase = slide_ask.askBase;

pub const language_prompt = "What language is this written in? Reply with the language's name in English, one word.\n{s}";

pub const questions_other = "Here is something you know: \"{s}\" Write {d} different short questions the person who told you might ask you later about it, in their own words with \"I\" and \"my\" (the same words in their language), to see if you remember it: ask it plainly, in other words and in passing (for example, for \"I like blue.\": What is my favourite colour? Which colour do I like best?), in the language they told you in. Number them 1 to {d}, one per line, nothing else.";

pub const remember_other = "Here are questions a person asked you about themselves. Rewrite each to start with \"Do you remember\" or \"Do you know\" (the same words in the question's own language), in the person's own words, keeping \"I\" and \"my\" (for example: What is my favourite colour? becomes Do you remember my favourite colour?). Number them, one per line, nothing else.\n{s}";

pub const besides_other = "Here is something you know: \"{s}\" Write {d} short questions about the same person or thing that this does not answer, in the words of the person who told you (for example, for \"My sister is called Ana.\": How old is my sister? Where does my sister live?), in the language they told you in. Number them, one per line, nothing else.";

pub const twins_other = "For each question below, write two questions worded almost the same way but asking about a different person or thing of the same type, so that the answer to the original would be wrong for them, each in its question's language. Number them, one per line, nothing else.\n{s}";

/// doc_questions_prompt for a document in another language: its questions in that language, not English.
pub const doc_questions_other = "Here is something from a document you were given: \"{s}\" Write {d} different short questions that it answers, in the language it is written in (not English unless it is English), the way a colleague might ask you later: plainly, in other words and in passing, each naming who or what it is about as the text does, never just they, it or the team. Every question must be answered by exactly what it says. Number them 1 to {d}, one per line, nothing else.";
pub const translate_prompt = "Translate each numbered line below into {s}, keeping its meaning and its number. Nothing else.\n{s}";

/// The language `fact` is told in, as the model names it, or null for English.
pub fn languageOf(srv: *Server, cx: *Cx, fact: []const u8, gone: anytype) !?[]const u8 {
    const said = try askBase(srv, cx, null, try std.fmt.allocPrint(cx.a, language_prompt, .{fact}), 8, gone);
    const name = std.mem.trim(u8, said.content, " \t\r\n.\"*");
    if (name.len == 0 or std.ascii.startsWithIgnoreCase(name, "english")) return null;
    return name;
}

/// A lesson prompt in the wording for the facts' language: English's own, or one that keeps the model in another.
pub fn prompted(a: Allocator, comptime english: []const u8, comptime other: []const u8, args: anytype, language: ?[]const u8) ![]const u8 {
    return if (language != null) std.fmt.allocPrint(a, other, args) else std.fmt.allocPrint(a, english, args);
}

/// `lines` as the model translates them into `language`, kept in `keep`.
pub fn translated(srv: *Server, cx: *Cx, keep: Allocator, lines: []const []const u8, language: []const u8, gone: anytype) ![]const []const u8 {
    const a = cx.a;
    var numbered: std.ArrayList(u8) = .empty;
    for (lines, 1..) |line, i| try numbered.print(a, "{d}. {s}\n", .{ i, line });
    const reply = try askBase(srv, cx, null, try std.fmt.allocPrint(a, translate_prompt, .{ language, numbered.items }), @intCast(40 * lines.len), gone);
    var out: std.ArrayList([]const u8) = .empty;
    for (try wording.numbered(a, reply.content, lines.len)) |line| try out.append(keep, try keep.dupe(u8, line));
    return out.items;
}
