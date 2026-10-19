//! A lesson's examples, the lists they come in, how a round deals them, and the copies a learner keeps of them.
const std = @import("std");

pub const max_steps = 400;

/// Token ids, the answer from `start`; `decide`: the reply's first row; `subject`, `value`: key rows (0: none).
pub const Example = struct { ids: []const u32, start: u32, decide: u32, subject: u32 = 0, value: u32 = 0, context: u32 = 0 }; // context: rows of a conversation before its own turn, which other examples share (0: none)

/// One fact's examples; `more` another round, `undo` one back, `save` writes every kept lesson.
pub const Lesson = struct {
    train: []const Example = &.{},
    think: []const Example = &.{}, // the facts' answers with thinking on, one learned beside each plain one
    held: []const Example = &.{},
    near: []const Example = &.{},
    keep: []const Example = &.{},
    asked: []const Example = &.{}, // more wordings of the fact's questions, read only at their decision rows
    unasked: []const Example = &.{}, // questions it must not answer, read only at their decision rows
    undo: bool = false,
    steps: u32 = max_steps,
    more: bool = false,
    kept: []const Example = &.{}, // questions the open lesson took, each with its answer as it was (a reroute's)
    reroute: bool = false, // the open lesson's router fitted again with `unasked` among what it must not take
    drop: bool = false, // the open lesson taken out whole
    save: bool = false,
    retire: []const [2]u16 = &.{}, // (layer, neuron) of facts a newer one replaces, zeroed
    unretire: bool = false, // the last retire's neurons put back
    forget: []const [2]u16 = &.{}, // (layer, neuron) of a fact deleted on request, their input weights zeroed
    unforget: bool = false, // the last forget undone, bit for bit
};

/// A prompt by its rows through its decision row: what a router reads of it.
pub fn key(ex: Example) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ex.ids[0 .. ex.decide + 1]));
}

/// Examples in all of `lists`.
pub fn total(lists: []const []const Example) usize {
    var n: usize = 0;
    for (lists) |xs| n += xs.len;
    return n;
}

/// Example i of the lists in turn.
pub fn pick(lists: []const []const Example, i: usize) Example {
    var at = i;
    for (lists) |xs| {
        if (at < xs.len) return xs[at];
        at -= xs.len;
    }
    unreachable;
}

/// Examples dealt in shuffled passes: every one once before any comes again.
pub const Deck = struct {
    cards: []Example,
    next: usize,

    pub fn init(gpa: std.mem.Allocator, parts: []const []const Example) !Deck {
        var n: usize = 0;
        for (parts) |p| n += p.len;
        const cards = try gpa.alloc(Example, n);
        var at: usize = 0;
        for (parts) |p| {
            @memcpy(cards[at..][0..p.len], p);
            at += p.len;
        }
        return .{ .cards = cards, .next = n };
    }

    pub fn deinit(d: *Deck, gpa: std.mem.Allocator) void {
        gpa.free(d.cards);
    }

    pub fn draw(d: *Deck, r: std.Random) Example {
        if (d.next == d.cards.len) {
            r.shuffle(Example, d.cards);
            d.next = 0;
        }
        d.next += 1;
        return d.cards[d.next - 1];
    }
};

/// Copies of `xs`, their ids too, the caller's to free (disown).
pub fn own(gpa: std.mem.Allocator, xs: []const Example) ![]Example {
    const out = try gpa.alloc(Example, xs.len);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |x| gpa.free(x.ids);
        gpa.free(out);
    }
    for (xs, out) |x, *o| {
        o.* = .{ .ids = try gpa.dupe(u32, x.ids), .start = x.start, .decide = x.decide, .subject = x.subject, .value = x.value, .context = x.context };
        made += 1;
    }
    return out;
}

/// Whether a learner reads conversation prompts at any length: through `Trainer`'s taps, in a pocket lesson.
pub fn readsConversations(comptime Trainer: type, pocket: bool) bool {
    return pocket and @hasDecl(Trainer, "tapRows");
}

/// `lesson` checked, as a learner reads it: a prompt in a conversation longer than its passes' `rows` only by `taps`.
pub fn readable(gpa: std.mem.Allocator, lesson: Lesson, taps: bool, rows: usize, bufs: *[4]std.ArrayList(Example)) !Lesson {
    for ([_][]const Example{ lesson.train, lesson.think, lesson.held, lesson.near, lesson.keep, lesson.asked, lesson.unasked, lesson.kept }) |xs| for (xs) |ex| {
        if (ex.start < 1 or ex.start >= ex.ids.len or ex.decide + 1 >= ex.ids.len) return error.ExampleTooLong;
        // a prompt in a conversation is read through the host's taps, at any length
        if (ex.ids.len - 1 > rows and ex.context == 0) return error.ExampleTooLong;
    };
    var out = lesson;
    var dropped: usize = 0;
    // answers are read by passes whole; other wordings and near misses may be read through the trainer's taps
    for ([_]*[]const Example{ &out.train, &out.think, &out.asked, &out.unasked }, bufs, 0..) |list, *buf, i| {
        buf.clearRetainingCapacity();
        for (list.*) |ex| if (ex.context == 0 or ex.ids.len - 1 <= rows or (taps and i >= 2)) try buf.append(gpa, ex) else {
            dropped += 1;
        };
        list.* = buf.items;
    }
    if (dropped > 0) std.log.info("slide: {d} prompts in a conversation are left out: longer than this learner reads", .{dropped});
    return out;
}

pub fn disown(gpa: std.mem.Allocator, xs: []Example) void {
    for (xs) |x| gpa.free(x.ids);
    gpa.free(xs);
}

/// The lists of a lesson an edit reads.
const edit_lists = .{ "train", "think", "held", "near", "keep", "asked", "unasked" };

/// Copies of the lists of `lesson` an edit reads, the caller's to free (disownLesson).
pub fn ownLesson(gpa: std.mem.Allocator, lesson: Lesson) !Lesson {
    var out: Lesson = .{};
    errdefer disownLesson(gpa, out);
    inline for (edit_lists) |f| @field(out, f) = try own(gpa, @field(lesson, f));
    return out;
}

pub fn disownLesson(gpa: std.mem.Allocator, lesson: Lesson) void {
    inline for (edit_lists) |f| disown(gpa, @constCast(@field(lesson, f)));
}

test "a learner reads prompts in a conversation only through its trainer's taps, and only in a pocket lesson" {
    const Plain = struct {};
    const Tapping = struct {
        pub fn tapRows() void {}
    };
    try std.testing.expect(!readsConversations(Plain, true));
    try std.testing.expect(readsConversations(Tapping, true));
    try std.testing.expect(!readsConversations(Tapping, false));
}

test "a learner without taps reads a prompt in a conversation only when its passes hold it" {
    const gpa = std.testing.allocator;
    const long: [40]u32 = @splat(0);
    const short: Example = .{ .ids = long[0..8], .start = 7, .decide = 6, .context = 4 };
    const far: Example = .{ .ids = &long, .start = 39, .decide = 38, .context = 30 };
    const cold: Example = .{ .ids = long[0..8], .start = 7, .decide = 6 };
    const lesson: Lesson = .{ .train = &.{ cold, far }, .asked = &.{ short, far }, .unasked = &.{far} };
    var bufs: [4]std.ArrayList(Example) = @splat(.empty);
    defer for (&bufs) |*b| b.deinit(gpa);
    const plain = try readable(gpa, lesson, false, 16, &bufs);
    try std.testing.expectEqual(@as(usize, 1), plain.train.len);
    try std.testing.expectEqual(@as(usize, 1), plain.asked.len);
    try std.testing.expectEqual(@as(usize, 0), plain.unasked.len);
    const tapped = try readable(gpa, lesson, true, 16, &bufs);
    try std.testing.expectEqual(@as(usize, 1), tapped.train.len); // an answer is read by passes whole
    try std.testing.expectEqual(@as(usize, 2), tapped.asked.len);
    try std.testing.expectEqual(@as(usize, 1), tapped.unasked.len);
}
