//! Nemotron's Sliding Weights learner: each lesson a new block of the low-rank change at every layer, learned live.
const std = @import("std");
const train = @import("train.zig");
const adapters = @import("adapters.zig");
const choice = @import("choice.zig");
const backend = @import("backend.zig");

const Metal = backend.Metal;

/// Token ids whose answer starts at `start`: the rows from start - 1 on predict it.
pub const Example = struct { ids: []const u32, start: u32 };

/// One fact's examples; `more` takes another round of the last lesson, `undo` takes its last round back.
pub const Lesson = struct {
    train: []const Example = &.{},
    held: []const Example = &.{},
    near: []const Example = &.{},
    keep: []const Example = &.{},
    undo: bool = false,
    steps: u32 = max_steps,
    more: bool = false,
};

pub const Report = union(enum) {
    learned: struct { recalled: bool, steps: u32, loss: f32 },
    failed: []const u8,
};

pub const Step = struct { done: bool, changed: bool = false, report: ?Report = null };

const max_steps = 60;
const check_every = 5;
const replay_cap = 64; // earlier lessons' answers kept steady at most, the oldest giving way

/// How far above the largest share any steady row has a fact row's share must be for the block to act on it.
const gate_margin: f32 = 1.25;

/// A new lesson first sketches and measures what its block must avoid and read (build), then learns (steps).
const Phase = enum { idle, build, steps, undo };

pub const Learner = struct {
    gpa: std.mem.Allocator,
    b: *Metal,
    trainer: ?*train.Trainer = null,
    lesson: Lesson = .{},
    phase: Phase = .idle,
    plan: std.ArrayList(Example) = .empty, // this round's steps in order
    at: usize = 0, // the plan's next example
    taken: u32 = 0, // steps this round
    loss: f32 = 0, // the round's summed loss
    keep: []Example = &.{}, // the keep prompts' examples, copied from the first lesson that brings them
    replay: std.ArrayList(Example) = .empty, // earlier lessons' answers, owned
    last: []Example = &.{}, // the last lesson's answers, owned, joining replay when the next lesson begins
    kept_rounds: u32 = 0, // the last lesson's rounds still in the weights
    opened: bool = false, // the last lesson opened the block now last in the change
    built: usize = 0, // examples a new lesson has sketched or projected so far
    choice: ?choice.Choice = null, // the new block's directions and gates, as its examples are projected
    answers: u32 = 0, // fact answer rows projected
    can_undo: bool = false,
    rounds: u64 = 0, // rounds begun, which seeds each round's order

    pub fn init(gpa: std.mem.Allocator, b: *Metal) Learner {
        return .{ .gpa = gpa, .b = b };
    }

    pub fn deinit(l: *Learner) void {
        if (l.trainer) |t| t.deinit(l.gpa);
        if (l.choice) |*c| c.deinit();
        l.plan.deinit(l.gpa);
        disown(l.gpa, l.keep);
        disown(l.gpa, l.last);
        for (l.replay.items) |ex| l.gpa.free(ex.ids);
        l.replay.deinit(l.gpa);
    }

    /// Start a lesson's round; its examples must stay valid until its last step.
    pub fn begin(l: *Learner, lesson: Lesson) !void {
        std.debug.assert(l.phase == .idle);
        l.lesson = lesson;
        if (lesson.undo) {
            l.phase = .undo;
            return;
        }
        if (lesson.train.len == 0) return;
        for ([_][]const Example{ lesson.train, lesson.held, lesson.near, lesson.keep }) |xs| for (xs) |ex| {
            if (ex.start < 1 or ex.start >= ex.ids.len or ex.ids.len - 1 > train.max_rows) return error.ExampleTooLong;
        };
        if (l.trainer == null) {
            l.choice = try choice.Choice.init(l.gpa, l.b.m.config.layers);
            l.trainer = try train.Trainer.init(l.gpa, l.b);
        }
        if (l.keep.len == 0 and lesson.keep.len > 0) l.keep = try own(l.gpa, lesson.keep);
        if (lesson.more) {
            if (!l.opened) return error.NothingToContinue;
            return l.start();
        }
        try l.settleLast(lesson.train);
        if (l.trainer.?.sites.rank + adapters.block > adapters.max_rank) return error.LearnedChangeFull;
        l.trainer.?.sites.clear();
        l.trainer.?.sketched = 0;
        l.built = 0;
        l.phase = .build;
    }

    /// A round's steps from the open block as it is now, which an undo comes back to.
    fn start(l: *Learner) !void {
        try l.schedule(@max(@min(l.lesson.steps, max_steps), 1));
        l.trainer.?.sites.keep();
        l.can_undo = true;
        l.at = 0;
        l.taken = 0;
        l.loss = 0;
        l.phase = .steps;
    }

    pub fn abort(l: *Learner) void {
        l.phase = .idle;
    }

    /// One bounded unit: one example sketched, one step of learning (with the held-out check every few), or the undo.
    pub fn step(l: *Learner) Step {
        return l.advance() catch |e| {
            if (l.phase == .steps) l.trainer.?.sites.restore();
            l.phase = .idle;
            return .{ .done = true, .changed = true, .report = .{ .failed = @errorName(e) } };
        };
    }

    fn advance(l: *Learner) !Step {
        switch (l.phase) {
            .idle => return .{ .done = true },
            .undo => {
                l.phase = .idle;
                if (!l.can_undo) return .{ .done = true };
                l.trainer.?.sites.restore();
                l.can_undo = false;
                l.kept_rounds -|= 1;
                return .{ .done = true, .changed = true };
            },
            .build => return l.buildOnce(),
            .steps => return l.stepOnce(),
        }
    }

    /// One example at a time: what must stay and the fact sketched, then projected; then the block chosen and opened.
    fn buildOnce(l: *Learner) !Step {
        const t = l.trainer.?;
        const c = &l.choice.?;
        const facts = l.lesson.train;
        const stay = [_][]const Example{ l.keep, l.lesson.near, l.replay.items };
        const steady = l.keep.len + l.lesson.near.len + l.replay.items.len;
        var i = l.built;
        l.built += 1;
        if (i < steady) return l.sketch(pick(&stay, i), .avoid);
        i -= steady;
        if (i < facts.len) return l.sketch(facts[i], .seek);
        i -= facts.len;
        if (i == 0) {
            t.sites.frame();
            c.reset();
            l.answers = 0;
        }
        if (i < steady) {
            const ex = pick(&stay, i);
            _ = try t.step(ex.ids, ex.start, .project);
            for (0..t.sites.list.len) |k| try c.add(k, t.projected(k, ex.ids.len - 1), false);
            return .{ .done = false };
        }
        i -= steady;
        if (i < facts.len) {
            const ex = facts[i];
            _ = try t.step(ex.ids, ex.start, .project);
            l.answers += @intCast(ex.ids.len - ex.start);
            for (0..t.sites.list.len) |k| try c.add(k, t.projected(k, ex.ids.len - 1)[(ex.start - 1) * adapters.candidates ..], true);
            return .{ .done = false };
        }
        try c.choose(gate_margin);
        try t.sites.open(c.coef);
        t.sites.attach(&l.b.m.weights, true);
        l.opened = true;
        l.gate();
        try l.start();
        return .{ .done = false };
    }

    fn sketch(l: *Learner, ex: Example, mode: train.Mode) !Step {
        _ = try l.trainer.?.step(ex.ids, ex.start, mode);
        return .{ .done = false };
    }

    /// The new block's gate at each layer: above every steady row's share there, shut where no fact row clears it.
    fn gate(l: *Learner) void {
        const sites = &l.trainer.?.sites;
        const c = &l.choice.?;
        const k = sites.first() / adapters.block;
        var open: usize = 0;
        var reach: u64 = 0;
        for (sites.list, c.tau, c.hits) |*site, tau, hits| {
            site.gate(k).* = tau;
            open += @intFromBool(hits > 0);
            reach += hits;
        }
        const mean = 100 * @as(f64, @floatFromInt(reach)) / @as(f64, @floatFromInt(@max(open * l.answers, 1)));
        std.log.info("slide: block {d} acts at {d} of {d} layers, on {d:.0}% of the fact's answer rows there", .{ k + 1, open, sites.list.len, mean });
    }

    /// One step on one of the fact's answers; every few steps the held-out answers are read back.
    fn stepOnce(l: *Learner) !Step {
        const t = l.trainer.?;
        const ex = l.plan.items[l.at];
        const got = try t.step(ex.ids, ex.start, .learn);
        if (!std.math.isFinite(got.loss)) return error.NonfiniteStep;
        l.at += 1;
        l.loss += got.loss;
        l.taken += 1;
        const end = l.at == l.plan.items.len;
        const back = (l.taken % check_every == 0 or end) and try l.recalled();
        if (!back and !end) return .{ .done = false, .changed = true };
        l.phase = .idle;
        l.kept_rounds += 1;
        const loss = l.loss / @as(f32, @floatFromInt(l.taken));
        return .{ .done = true, .changed = true, .report = .{ .learned = .{ .recalled = back, .steps = l.taken, .loss = loss } } };
    }

    /// Whether every held-out answer comes back token for token (each its row's likeliest).
    fn recalled(l: *Learner) !bool {
        if (l.lesson.held.len == 0) return false;
        for (l.lesson.held) |ex| if (!(try l.trainer.?.step(ex.ids, ex.start, .loss)).recalled) return false;
        return true;
    }

    /// The last lesson's answers into replay if a round of it stayed (else its block out); this lesson's are last.
    fn settleLast(l: *Learner, answers: []const Example) !void {
        const now = try own(l.gpa, answers);
        if (l.opened and l.kept_rounds == 0) {
            l.trainer.?.sites.close();
            l.trainer.?.sites.attach(&l.b.m.weights, true);
        }
        l.opened = false;
        if (l.kept_rounds > 0) {
            for (l.last) |ex| {
                if (l.replay.items.len == replay_cap) l.gpa.free(l.replay.orderedRemove(0).ids);
                try l.replay.append(l.gpa, ex);
            }
            l.gpa.free(l.last);
        } else disown(l.gpa, l.last);
        l.last = now;
        l.kept_rounds = 0;
    }

    /// The round's steps: the fact's answers in fresh orders.
    fn schedule(l: *Learner, steps: u32) !void {
        l.rounds += 1;
        var prng = std.Random.DefaultPrng.init(l.rounds);
        var facts: Deck = try .init(l.gpa, &.{l.lesson.train});
        defer facts.deinit(l.gpa);
        l.plan.clearRetainingCapacity();
        for (0..steps) |_| try l.plan.append(l.gpa, facts.draw(prng.random()));
    }
};

/// Example i of the lists in turn.
fn pick(lists: []const []const Example, i: usize) Example {
    var at = i;
    for (lists) |xs| {
        if (at < xs.len) return xs[at];
        at -= xs.len;
    }
    unreachable;
}

/// Examples dealt in shuffled passes: every one once before any comes again.
const Deck = struct {
    cards: []Example,
    next: usize,

    fn init(gpa: std.mem.Allocator, parts: []const []const Example) !Deck {
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

    fn deinit(d: *Deck, gpa: std.mem.Allocator) void {
        gpa.free(d.cards);
    }

    fn draw(d: *Deck, r: std.Random) Example {
        if (d.next == d.cards.len) {
            r.shuffle(Example, d.cards);
            d.next = 0;
        }
        d.next += 1;
        return d.cards[d.next - 1];
    }
};

fn own(gpa: std.mem.Allocator, xs: []const Example) ![]Example {
    const out = try gpa.alloc(Example, xs.len);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |x| gpa.free(x.ids);
        gpa.free(out);
    }
    for (xs, out) |x, *o| {
        o.* = .{ .ids = try gpa.dupe(u32, x.ids), .start = x.start };
        made += 1;
    }
    return out;
}

fn disown(gpa: std.mem.Allocator, xs: []Example) void {
    for (xs) |x| gpa.free(x.ids);
    gpa.free(xs);
}
