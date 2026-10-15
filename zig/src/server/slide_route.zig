//! A fact's questions asked about others the user knows, other things of its kind and the model: none of them its own.
const std = @import("std");
const wording = @import("slide_words.zig");
const slide_span = @import("slide_span.zig");
const Allocator = std.mem.Allocator;

/// Whom a question about the user is also asked about: the nearest relations, most easily mistaken for the user's own.
const relations = [_][]const u8{ "sister", "brother", "mother", "father", "friend" };

/// Words for people a question may name after "my", swapped for each relation.
const named_words = relations ++ [_][]const u8{ "wife", "husband", "partner", "son", "daughter", "boss", "colleague", "neighbour", "neighbor", "cousin", "aunt", "uncle", "mum", "mom", "dad", "sis", "bro", "girlfriend", "boyfriend", "grandma", "grandpa" };

/// `question` about others: each other relation, each thing of the fact's kind for `subject`, and asked of the model.
pub fn others(a: Allocator, question: []const u8, subject: ?[]const u8, kinds: []const []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    if (named(question)) |at| {
        const word = question[at[0]..at[1]];
        for (relations) |r| if (!std.ascii.eqlIgnoreCase(word, r)) try out.append(a, try std.mem.concat(a, u8, &.{ question[0..at[0]], r, question[at[1]..] }));
    } else for (relations) |r| if (try wording.about(a, question, r)) |q| try out.append(a, q);
    if (subject) |s| for (kinds) |k| if (try slide_span.replace(a, question, s, k)) |other| try out.append(a, other);
    if (try wording.addressed(a, question)) |q| try out.append(a, q);
    return out.items;
}

/// The span of a person `question` names after "my" (the word alone, without its 's), or null.
fn named(question: []const u8) ?[2]usize {
    var i: usize = 0;
    var after_my = false;
    while (i < question.len) {
        while (i < question.len and !std.ascii.isAlphabetic(question[i])) i += 1;
        const start = i;
        while (i < question.len and std.ascii.isAlphabetic(question[i])) i += 1;
        if (start == i) break;
        const word = question[start..i];
        if (after_my) for (named_words) |w| if (std.ascii.eqlIgnoreCase(word, w)) return .{ start, i };
        after_my = std.ascii.eqlIgnoreCase(word, "my");
    }
    return null;
}

test "a fact's question is asked about other people, other things of its kind and the model" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sister = try others(a, "What's my sister's name?", null, &.{});
    try std.testing.expectEqualStrings("What's my brother's name?", sister[0]);
    try std.testing.expectEqual(relations.len, sister.len); // every other near relation, then the model's own
    try std.testing.expectEqualStrings("What's your sister's name?", sister[sister.len - 1]);
    const colour = try others(a, "What colour do I like best?", "colour", &.{ "food", "sport" });
    try std.testing.expectEqualStrings("What colour does my sister like best?", colour[0]);
    var food = false;
    for (colour) |q| food = food or std.mem.eql(u8, q, "What food do I like best?");
    try std.testing.expect(food);
}

test "router alternatives replace each wording's own complete subject" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const projects = try others(a, "Who leads Project Aster?", "Project Aster", &.{"Project Boreal"});
    try std.testing.expectEqual(@as(usize, 1), projects.len);
    try std.testing.expectEqualStrings("Who leads Project Boreal?", projects[0]);
    const people = try others(a, "What did Mira Solven build?", "Mira Solven", &.{"Odile Marchetti"});
    try std.testing.expectEqual(@as(usize, 1), people.len);
    try std.testing.expectEqualStrings("What did Odile Marchetti build?", people[0]);
    try std.testing.expectEqual(@as(usize, 0), (try others(a, "Who leads Project Asterion?", "Project Aster", &.{"Project Boreal"})).len);
    try std.testing.expectEqual(@as(usize, 0), (try others(a, "What is 17 plus 25?", null, &.{"Project Boreal"})).len);
    // Another project wording follows a person and a null span; neither subject may carry across calls.
    try std.testing.expectEqualStrings("Who leads Project Boreal?", (try others(a, "Who leads Project Aster?", "Project Aster", &.{"Project Boreal"}))[0]);
}
