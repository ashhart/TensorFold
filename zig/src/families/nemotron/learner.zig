//! Nemotron's Sliding Weights learner on any backend: each lesson a new block of the low-rank change at every layer.
const std = @import("std");
const adapters = @import("slide_dims.zig");
const choice = @import("choice.zig");

/// Token ids whose answer starts at `start`: the rows from start - 1 on predict it.
pub const Example = struct { ids: []const u32, start: u32 };

/// One fact's examples; `more` another round, `undo` it out whole, `commit` a weight change, `save` it into the shards.
pub const Lesson = struct {
    train: []const Example = &.{},
    held: []const Example = &.{},
    near: []const Example = &.{},
    keep: []const Example = &.{},
    undo: bool = false,
    steps: u32 = max_steps,
    more: bool = false,
    commit: bool = false,
    save: bool = false,
};

pub const Report = union(enum) {
    learned: struct { recalled: bool, steps: u32, loss: f32 },
    failed: []const u8,
};

pub const Step = struct { done: bool, changed: bool = false, report: ?Report = null };

const max_steps = 400;
const check_every = 20;
const plain_steps = 60; // steps a lesson takes once it is a plain weight change, before its near misses are mined
const replay_cap = 64; // earlier lessons' answers kept steady at most, the oldest giving way

/// How far above every steady row's cosine with the gate's direction a row's must be for the block to act on it.
const gate_margin: f32 = 0.02;

/// How much each steady row weighs against the fact's rows when a lesson becomes a plain weight change.
const hold: f64 = 1;

/// A new lesson first sketches and measures what its block must avoid and read (build), then learns (steps).
const Phase = enum { idle, build, steps, undo, commit, save };

/// An example's turn: a fact answer or a twin kept as it is, and whether Adam steps after it.
const Turn = struct { ex: Example, fact: bool, last: bool };

/// The learner over a backend's training file: its Backend, Mode, max_rows and Trainer (step, sites, attach, save).
pub fn Of(comptime train: type) type {
    return struct {
        const Learner = @This();
        const Trainer = train.Trainer;
        const Mode = train.Mode;

        gpa: std.mem.Allocator,
        io: std.Io,
        b: *train.Backend,
        trainer: ?*Trainer = null,
        lesson: Lesson = .{},
        phase: Phase = .idle,
        plan: std.ArrayList(Turn) = .empty, // this round's examples in order
        at: usize = 0, // the plan's next example
        taken: u32 = 0, // Adam steps this round
        loss: f32 = 0, // the round's fact answers' summed loss
        replay: std.ArrayList(Example) = .empty, // earlier lessons' answers, owned
        last: []Example = &.{}, // the last lesson's answers, owned, joining replay when the next lesson begins
        kept_rounds: u32 = 0, // the last lesson's rounds still in the weights
        opened: bool = false, // the last lesson opened the block now last in the change
        built: usize = 0, // examples a new lesson has sketched or projected so far
        choice: ?choice.Choice = null, // the new block's directions and gates, as its examples are projected
        answers: u32 = 0, // fact answer rows projected
        plain: bool = false, // the last lesson is a plain weight change
        rounds: u64 = 0, // rounds begun, which seeds each round's order

        pub fn init(gpa: std.mem.Allocator, io: std.Io, b: *train.Backend) Learner {
            return .{ .gpa = gpa, .io = io, .b = b };
        }

        pub fn deinit(l: *Learner) void {
            if (l.trainer) |t| t.deinit(l.gpa);
            if (l.choice) |*c| c.deinit();
            l.plan.deinit(l.gpa);
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
            if (lesson.commit or lesson.save) {
                l.phase = if (lesson.save) .save else .commit;
                return;
            }
            if (lesson.train.len == 0) return;
            for ([_][]const Example{ lesson.train, lesson.held, lesson.near, lesson.keep }) |xs| for (xs) |ex| {
                if (ex.start < 1 or ex.start >= ex.ids.len or ex.ids.len - 1 > train.max_rows) return error.ExampleTooLong;
            };
            if (l.trainer == null) {
                const t = try Trainer.init(l.gpa, l.b);
                errdefer t.deinit(l.gpa);
                l.choice = try choice.Choice.init(l.gpa, t.sites.list.len);
                l.trainer = t;
            }
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

        /// A round's steps from the open block as it is now, which a failed step comes back to.
        fn start(l: *Learner) !void {
            const steps = @max(@min(l.lesson.steps, max_steps), 1);
            if (l.plain) try l.schedulePlain(steps) else try l.schedule(steps);
            l.trainer.?.sites.keep();
            l.at = 0;
            l.taken = 0;
            l.loss = 0;
            l.phase = .steps;
        }

        pub fn abort(l: *Learner) void {
            l.phase = .idle;
        }

        /// One bounded unit: one example sketched, one step of learning (held-out check every few), or the undo.
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
                    // the whole lesson, every round and its plain steps, as if it never ran; a save keeps none of it
                    l.phase = .idle;
                    if (!l.opened) return .{ .done = true };
                    l.trainer.?.sites.close();
                    l.trainer.?.attach(true);
                    l.plain = false;
                    l.opened = false;
                    l.kept_rounds = 0;
                    return .{ .done = true, .changed = true };
                },
                .build => return l.buildOnce(),
                .steps => return l.stepOnce(),
                .commit => return l.commitOnce(),
                .save => return l.saveOnce(),
            }
        }

        /// Every lesson made a weight change, folded into its layers' output projections and written into the shards.
        fn saveOnce(l: *Learner) !Step {
            l.phase = .idle;
            const t = l.trainer orelse return .{ .done = true };
            const ranks = t.sites.rank - @as(usize, if (l.opened and !l.plain) adapters.block else 0);
            if (ranks == 0) return .{ .done = true };
            const t0 = std.Io.Clock.awake.now(l.io);
            const bytes = try t.save(l.gpa, l.io, ranks);
            const took = @as(f64, @floatFromInt(std.Io.Clock.awake.now(l.io).toNanoseconds() - t0.toNanoseconds())) / 1e9;
            std.log.info("slide: {d} lessons written into the model's weights: {d} MB of output projections into its shards in {d:.1} s", .{ ranks / adapters.block, bytes >> 20, took });
            return .{ .done = true };
        }

        /// The kept lesson made a plain change of the weights: each layer's directions refit to hold steady rows still.
        fn commitOnce(l: *Learner) !Step {
            l.phase = .idle;
            if (!l.opened or l.kept_rounds == 0) return .{ .done = true };
            const t = l.trainer.?;
            const c = &l.choice.?;
            for (0..t.sites.list.len) |k| try c.refit(k, hold);
            t.sites.plain(c.coef);
            t.attach(true);
            l.plain = true;
            std.log.info("slide: lesson {d} is now a plain weight change; it learns on beside the answers it must keep", .{t.sites.first() / adapters.block + 1});
            try l.schedulePlain(plain_steps);
            t.sites.keep();
            l.at = 0;
            l.taken = 0;
            l.loss = 0;
            l.phase = .steps;
            return .{ .done = false, .changed = true };
        }

        /// One example at a time: what must stay and the fact sketched, then projected; then the block chosen, opened.
        fn buildOnce(l: *Learner) !Step {
            const t = l.trainer.?;
            const c = &l.choice.?;
            const facts = l.lesson.train;
            const stay = [_][]const Example{ l.lesson.keep, l.lesson.near, l.replay.items };
            const steady = l.lesson.keep.len + l.lesson.near.len + l.replay.items.len;
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
                for (0..t.sites.list.len) |k| try c.add(k, t.projected(k, ex.ids.len - 1), false, ex.start - 1, true);
                return .{ .done = false };
            }
            i -= steady;
            if (i < facts.len) {
                const ex = facts[i];
                _ = try t.step(ex.ids, ex.start, .project);
                l.answers += @intCast(ex.ids.len - ex.start);
                try c.head();
                for (0..t.sites.list.len) |k| try c.add(k, t.projected(k, ex.ids.len - 1)[(ex.start - 1) * adapters.candidates ..], true, 0, false);
                return .{ .done = false };
            }
            try c.choose(gate_margin);
            try t.sites.open(c.coef);
            t.attach(true);
            l.opened = true;
            l.gate();
            try l.start();
            return .{ .done = false };
        }

        fn sketch(l: *Learner, ex: Example, mode: Mode) !Step {
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
            var firsts: u64 = 0;
            for (sites.list, c.tau, c.hits, c.opens) |*site, tau, hits, opens| {
                site.gate(k).* = tau;
                open += @intFromBool(hits > 0);
                reach += hits;
                firsts += opens;
            }
            const mean = 100 * @as(f64, @floatFromInt(reach)) / @as(f64, @floatFromInt(@max(open * l.answers, 1)));
            const first = @as(f64, @floatFromInt(firsts)) / @as(f64, @floatFromInt(@max(c.heads.items.len, 1)));
            std.log.info("slide: block {d} acts at {d} of {d} layers, on {d:.0}% of the fact's answer rows there; first rows at {d:.1}", .{ k + 1, open, sites.list.len, mean, first });
        }

        /// An example's gradients, Adam after a fact answer and its twin; every few steps the held-out answers are read
        fn stepOnce(l: *Learner) !Step {
            const t = l.trainer.?;
            const turn = l.plan.items[l.at];
            const got = try t.step(turn.ex.ids, turn.ex.start, if (turn.last) .learn else .grad);
            if (!std.math.isFinite(got.loss)) return error.NonfiniteStep;
            l.at += 1;
            if (turn.fact) l.loss += got.loss;
            if (!turn.last) return .{ .done = false };
            l.taken += 1;
            const end = l.at == l.plan.items.len;
            if (l.plain and !end) return .{ .done = false, .changed = true };
            const back = (l.taken % check_every == 0 or end) and try l.recalled();
            if (!back and !end) return .{ .done = false, .changed = true };
            l.phase = .idle;
            l.kept_rounds += 1;
            const loss = l.loss / @as(f32, @floatFromInt(l.taken));
            std.log.info("slide: {d} steps{s}, the fact's answers' mean loss {d:.3}; held-out answers {s}", .{ l.taken, if (l.plain) " as a plain change" else "", loss, if (back) "back" else "not back" });
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
                l.trainer.?.attach(true);
            }
            l.opened = false;
            l.plain = false;
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

        /// A plain change's steps: a fact answer, then two steady ones as they were (twins twice as often), Adam after.
        fn schedulePlain(l: *Learner, steps: u32) !void {
            l.rounds += 1;
            var prng = std.Random.DefaultPrng.init(l.rounds);
            var facts: Deck = try .init(l.gpa, &.{l.lesson.train});
            defer facts.deinit(l.gpa);
            var steady: Deck = try .init(l.gpa, &.{ l.lesson.keep, l.lesson.near, l.lesson.near, l.replay.items });
            defer steady.deinit(l.gpa);
            l.plan.clearRetainingCapacity();
            for (0..steps) |_| {
                const alone = steady.cards.len == 0;
                try l.plan.append(l.gpa, .{ .ex = facts.draw(prng.random()), .fact = true, .last = alone });
                if (!alone) try l.plan.append(l.gpa, .{ .ex = steady.draw(prng.random()), .fact = false, .last = false });
                if (!alone) try l.plan.append(l.gpa, .{ .ex = steady.draw(prng.random()), .fact = false, .last = true });
            }
        }

        /// The round's steps: the fact's answers in fresh orders.
        fn schedule(l: *Learner, steps: u32) !void {
            l.rounds += 1;
            var prng = std.Random.DefaultPrng.init(l.rounds);
            var facts: Deck = try .init(l.gpa, &.{l.lesson.train});
            defer facts.deinit(l.gpa);
            l.plan.clearRetainingCapacity();
            for (0..steps) |_| try l.plan.append(l.gpa, .{ .ex = facts.draw(prng.random()), .fact = true, .last = true });
        }
    };
}

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
