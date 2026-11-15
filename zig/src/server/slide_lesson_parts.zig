//! The examples and question references accumulated by one lesson writer.
const std = @import("std");
const api = @import("engine_api");
const Held = @import("slide_lesson.zig").Held;
const Check = @import("slide_lesson.zig").Check;

/// The facts' examples as a lesson gathers them.
pub const Parts = struct {
    reserved: @import("slide_reserved.zig").Reserved = .{},
    train: std.ArrayList(api.Example) = .empty,
    all: std.ArrayList(Held) = .empty, // every cold lesson question, including the training questions
    held_ex: std.ArrayList(api.Example) = .empty,
    held: std.ArrayList(Held) = .empty,
    twins: std.ArrayList(api.Example) = .empty,
    near: std.ArrayList(Check) = .empty, // the twins' questions with the answers they had, mined after the plain change
    aside: std.ArrayList(Check) = .empty, // every fourth near miss, never trained on: checked after
    think_train: std.ArrayList(api.Example) = .empty, // the facts' thinking answers, learned once the change is plain
    think_held: std.ArrayList(api.Example) = .empty,
    think_twins: std.ArrayList(api.Example) = .empty, // the trained near misses' thinking replies, held as they are
    think_keep: std.ArrayList(api.Example) = .empty, // the keep prompts' thinking replies
    asked: std.ArrayList(api.Example) = .empty, // more wordings of the facts' questions, prompts only, for the router
    probes: std.ArrayList(Held) = .empty, // some of those wordings, asked after each round
    unasked: std.ArrayList(api.Example) = .empty, // the questions asked about others, prompts only, for the router
    seen: usize = 0,
    thinks: bool = false, // each question is learned and held with thinking on too
    document: bool = false, // the facts come from the user's documents, asked about as a colleague would, not told in chat
    words: []const []const u8 = &.{}, // the open fact's subject words (none: it is about the user)
    named: []const []const u8 = &.{}, // of those, the ones a question must say to name it
    siblings: []const []const u8 = &.{}, // every fact told with these, whose questions are theirs to answer
    own: std.ArrayList([]const u8) = .empty, // the facts' kept questions: none is also checked or kept as it was
};
