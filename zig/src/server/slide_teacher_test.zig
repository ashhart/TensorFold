//! The teacher across lessons: a given-up lesson leaves the next kept lesson's checks as if it never ran.
const std = @import("std");
const lesson = @import("slide_lesson.zig");
const prompts = @import("slide_prompts.zig");
const Teacher = @import("slide_teacher.zig").Teacher;
const testing = std.testing;

/// The personal prompts a lesson numbered `nth` holds back to check (the rest it keeps as they are).
fn checksOf(nth: usize) [prompts.personal_prompts.len]bool {
    var out: [prompts.personal_prompts.len]bool = undefined;
    for (&out, 0..) |*x, i| x.* = lesson.heldBack(nth, i);
    return out;
}

test "a given-up lesson leaves the next kept lesson's checks as if it never ran" {
    var t = Teacher.init(testing.allocator);
    defer t.deinit();
    var never = Teacher.init(testing.allocator); // the same lessons, the given-up one never tried
    defer never.deinit();
    const first = checksOf(t.nth());
    try testing.expectEqualSlices(bool, &checksOf(never.nth()), &first);
    t.ended(true);
    never.ended(true);
    const tried = t.nth(); // a lesson written, its rounds never bring the fact back, taken out
    t.ended(false);
    try testing.expectEqual(tried, t.nth()); // the next lesson writes as the given-up one did
    const third = checksOf(t.nth());
    try testing.expectEqualSlices(bool, &checksOf(never.nth()), &third);
    try testing.expectEqual(@as(usize, 1), t.lessons);
    // and the checks do turn with kept lessons (the held-back prompts differ from the first lesson's)
    try testing.expect(!std.mem.eql(bool, &first, &third));
}

test "lesson accounting: repeated failures retain all original successful rotations" {
    var t = Teacher.init(testing.allocator);
    defer t.deinit();
    for (1..33) |accepted| {
        for (0..3) |_| t.ended(false);
        try testing.expectEqual(accepted, t.nth());
        const actual = checksOf(t.nth());
        const original = checksOf(accepted);
        try testing.expectEqualSlices(bool, &original, &actual);
        try testing.expectEqual(@as(usize, 3), std.mem.count(bool, &actual, &.{true}));
        t.ended(true);
        try testing.expectEqual(accepted, t.lessons);
    }
}
