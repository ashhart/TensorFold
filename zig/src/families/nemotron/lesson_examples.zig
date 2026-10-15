//! A lesson's examples, the lists they come in, how a round deals them, and the copies a learner keeps of them.
const std = @import("std");

pub const max_steps = 400;

/// Token ids, the answer from `start`; `decide`: the reply's first row; `subject`, `value`: key rows (0: none).
pub const Example = struct { ids: []const u32, start: u32, decide: u32, subject: u32 = 0, value: u32 = 0 };

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
        o.* = .{ .ids = try gpa.dupe(u32, x.ids), .start = x.start, .decide = x.decide, .subject = x.subject, .value = x.value };
        made += 1;
    }
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
