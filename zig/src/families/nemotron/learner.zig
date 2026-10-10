//! Sliding Weights' lessons, the same on every backend: each a new block of the low-rank change at every layer, learned
//! live, over a host (slide.zig on Metal, cuda_slide.zig on CUDA) that gives the trainer, its weights and the shards.
const std = @import("std");
const adapters = @import("learn_shape.zig");
const choice = @import("choice.zig");
const subspace = @import("subspace.zig");

/// Token ids whose answer starts at `start`: the rows from start - 1 on predict it.
pub const Example = struct { ids: []const u32, start: u32 };

/// One fact's examples; `more` another round, `undo` one back, `commit` a weight change, `gate` it to the rows its
/// facts need, `save` every kept lesson into the model's files.
pub const Lesson = struct {
    train: []const Example = &.{},
    held: []const Example = &.{},
    near: []const Example = &.{},
    keep: []const Example = &.{},
    undo: bool = false,
    steps: u32 = max_steps,
    more: bool = false,
    commit: bool = false,
    gate: bool = false,
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

/// How far below its lowest fact row a kept change's gate sits, so the facts' other wordings clear it too.
const regate_margin: f32 = 0.05;

/// A new lesson first sketches and measures what its block must avoid and read (build), then learns (steps).
const Phase = enum { idle, build, steps, undo, commit, gate, save };

/// An example's turn: a fact answer or a twin kept as it is, and whether Adam steps after it.
const Turn = struct { ex: Example, fact: bool, last: bool };

/// The lessons over host H: H.Trainer (step, projected, sites, sketched), H.max_rows, H.Ctx (what they learn into),
/// H.layers, H.trainer, H.free, H.attach (the change into the forward), H.write (into the shards) and H.seconds.
pub fn Learner(comptime H: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        io: std.Io,
        b: H.Ctx,
        trainer: ?*H.Trainer = null,
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
        can_undo: bool = false,
        plain: bool = false, // the last lesson is a plain weight change, which an undo takes out whole
        rounds: u64 = 0, // rounds begun, which seeds each round's order
        decisions: []std.ArrayList(f32) = &.{}, // [layers]: the kept lesson's deciding rows along each layer's candidates

        pub fn init(gpa: std.mem.Allocator, io: std.Io, b: H.Ctx) Self {
            return .{ .gpa = gpa, .io = io, .b = b };
        }

        pub fn deinit(l: *Self) void {
            if (l.trainer) |t| H.free(t, l.gpa);
            if (l.choice) |*c| c.deinit();
            for (l.decisions) |*d| d.deinit(l.gpa);
            l.gpa.free(l.decisions);
            l.plan.deinit(l.gpa);
            disown(l.gpa, l.last);
            for (l.replay.items) |ex| l.gpa.free(ex.ids);
            l.replay.deinit(l.gpa);
        }

        /// Start a lesson's round; its examples must stay valid until its last step.
        pub fn begin(l: *Self, lesson: Lesson) !void {
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
            if (lesson.gate) {
                if (!l.plain) return error.NothingToGate;
                if (l.decisions.len == 0) {
                    l.decisions = try l.gpa.alloc(std.ArrayList(f32), H.layers(l.b));
                    for (l.decisions) |*d| d.* = .empty;
                }
                for (l.decisions) |*d| d.clearRetainingCapacity();
                l.built = 0;
                l.phase = .gate;
                return;
            }
            if (lesson.train.len == 0) return;
            for ([_][]const Example{ lesson.train, lesson.held, lesson.near, lesson.keep }) |xs| for (xs) |ex| {
                if (ex.start < 1 or ex.start >= ex.ids.len or ex.ids.len - 1 > H.max_rows) return error.ExampleTooLong;
            };
            if (l.trainer == null) {
                l.choice = try choice.Choice.init(l.gpa, H.layers(l.b));
                l.trainer = try H.trainer(l.gpa, l.io, l.b);
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

        /// A round's steps from the open block as it is now, which an undo comes back to.
        fn start(l: *Self) !void {
            const steps = @max(@min(l.lesson.steps, max_steps), 1);
            if (l.plain) try l.schedulePlain(steps) else try l.schedule(steps);
            l.trainer.?.sites.keep();
            l.can_undo = true;
            l.at = 0;
            l.taken = 0;
            l.loss = 0;
            l.phase = .steps;
        }

        pub fn abort(l: *Self) void {
            l.phase = .idle;
        }

        /// One bounded unit: one example sketched, one step of learning (with the held-out check every few), or the undo.
        pub fn step(l: *Self) Step {
            return l.advance() catch |e| {
                if (l.phase == .steps) l.trainer.?.sites.restore();
                l.phase = .idle;
                return .{ .done = true, .changed = true, .report = .{ .failed = @errorName(e) } };
            };
        }

        fn advance(l: *Self) !Step {
            switch (l.phase) {
                .idle => return .{ .done = true },
                .undo => {
                    l.phase = .idle;
                    if (l.plain) {
                        l.trainer.?.sites.close();
                        try H.attach(l.b, l.trainer.?);
                        l.plain = false;
                        l.opened = false;
                        l.kept_rounds = 0;
                        return .{ .done = true, .changed = true };
                    }
                    if (!l.can_undo) return .{ .done = true };
                    l.trainer.?.sites.restore();
                    l.can_undo = false;
                    l.kept_rounds -|= 1;
                    return .{ .done = true, .changed = true };
                },
                .build => return l.buildOnce(),
                .steps => return l.stepOnce(),
                .commit => return l.commitOnce(),
                .gate => return l.gateOnce(),
                .save => return l.saveOnce(),
            }
        }

        /// Every kept lesson's blocks written as the model's living-weights file, beside its shards.
        fn saveOnce(l: *Self) !Step {
            l.phase = .idle;
            const t = l.trainer orelse return .{ .done = true };
            const ranks = t.sites.rank - @as(usize, if (l.opened and !l.plain) adapters.block else 0);
            if (ranks == 0) return .{ .done = true };
            const t0 = H.seconds(l.io);
            const bytes = try H.write(l.b, l.gpa, l.io, t, ranks);
            std.log.info("slide: {d} lessons saved as gated blocks beside the model's shards: {d} MB in {d:.1} s", .{ ranks / adapters.block, bytes >> 20, H.seconds(l.io) - t0 });
            return .{ .done = true };
        }

        /// One fact example a step: which rows the model got wrong before this lesson (they need the change), and those
        /// rows along each layer's candidates; after the last, each layer's gate aimed between them and the steady rows
        /// (Fisher's direction) and set just below the lowest of them, so the change acts on the rows that decide its
        /// facts and on as few others as that direction can keep out.
        fn gateOnce(l: *Self) !Step {
            const t = l.trainer.?;
            const c = &l.choice.?;
            const all = [_][]const Example{ l.lesson.train, l.lesson.held };
            const total = l.lesson.train.len + l.lesson.held.len;
            if (l.built < total) {
                const ex = pick(&all, l.built);
                l.built += 1;
                const n = ex.ids.len - 1;
                _ = try t.step(ex.ids, ex.start, .base);
                var needs: [H.max_rows]bool = @splat(false);
                for (ex.start - 1..n) |r| needs[r] = t.targetProb(r) < 0.5;
                _ = try t.step(ex.ids, ex.start, .project);
                for (l.decisions, 0..) |*deciding, k| {
                    const rows = t.projected(k, n);
                    for (ex.start - 1..n) |r| if (needs[r]) try deciding.appendSlice(l.gpa, rows[r * adapters.candidates ..][0..adapters.candidates]);
                }
                return .{ .done = false };
            }
            l.phase = .idle;
            const k = t.sites.first() / adapters.block;
            var acts: usize = 0;
            var steady: f64 = 0;
            for (t.sites.list, l.decisions, 0..) |*site, deciding, layer| {
                if (site.gate(k).* >= adapters.shut or deciding.items.len == 0) continue;
                var dir: [adapters.candidates]f32 = undefined;
                try choice.Choice.fisher(l.gpa, deciding.items, c.steady_rows[layer].items, &dir);
                var low: f32 = 2;
                for (0..deciding.items.len / adapters.candidates) |r| low = @min(low, subspace.dot(&dir, deciding.items[r * adapters.candidates ..][0..adapters.candidates]));
                const tau = low - regate_margin;
                t.sites.aim(layer, &dir, tau);
                acts += 1;
                steady += c.steadyAbove(layer, &dir, tau);
            }
            try H.attach(l.b, t);
            std.log.info("slide: lesson {d} is gated again at {d} layers: the rows that decide its facts clear the gate, and {d:.0}% of the steady rows do", .{ k + 1, acts, 100 * steady / @as(f64, @floatFromInt(@max(acts, 1))) });
            return .{ .done = true, .changed = true };
        }

        /// The kept lesson made a plain change of the weights: each layer's directions refit to hold steady rows still.
        fn commitOnce(l: *Self) !Step {
            l.phase = .idle;
            if (!l.opened or l.kept_rounds == 0) return .{ .done = true };
            const t = l.trainer.?;
            const c = &l.choice.?;
            for (0..t.sites.list.len) |k| try c.refit(k, hold);
            t.sites.plain(c.coef);
            try H.attach(l.b, t);
            l.plain = true;
            std.log.info("slide: lesson {d} is now a plain weight change; it learns on beside the answers it must keep", .{t.sites.first() / adapters.block + 1});
            try l.schedulePlain(plain_steps);
            t.sites.keep();
            l.can_undo = false;
            l.at = 0;
            l.taken = 0;
            l.loss = 0;
            l.phase = .steps;
            return .{ .done = false, .changed = true };
        }

        /// One example at a time: what must stay and the fact sketched, then projected; then the block chosen and opened.
        fn buildOnce(l: *Self) !Step {
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
            try H.attach(l.b, t);
            l.opened = true;
            l.gate();
            try l.start();
            return .{ .done = false };
        }

        fn sketch(l: *Self, ex: Example, comptime mode: @EnumLiteral()) !Step {
            _ = try l.trainer.?.step(ex.ids, ex.start, mode);
            return .{ .done = false };
        }

        /// The new block's gate at each layer: above every steady row's share there, shut where no fact row clears it.
        fn gate(l: *Self) void {
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

        /// One example's gradients, Adam after each fact answer and its twin; every few steps the held-out answers are read
        fn stepOnce(l: *Self) !Step {
            const t = l.trainer.?;
            const turn = l.plan.items[l.at];
            const got = try t.step(turn.ex.ids, turn.ex.start, if (turn.fact) (if (turn.last) .learn else .grad) else (if (turn.last) .hold_learn else .hold));
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
            return .{ .done = true, .changed = true, .report = .{ .learned = .{ .recalled = back, .steps = l.taken, .loss = loss } } };
        }

        /// Whether every held-out answer comes back token for token (each its row's likeliest).
        fn recalled(l: *Self) !bool {
            if (l.lesson.held.len == 0) return false;
            for (l.lesson.held) |ex| if (!(try l.trainer.?.step(ex.ids, ex.start, .loss)).recalled) return false;
            return true;
        }

        /// The last lesson's answers into replay if a round of it stayed (else its block out); this lesson's are last.
        fn settleLast(l: *Self, answers: []const Example) !void {
            const now = try own(l.gpa, answers);
            if (l.opened and l.kept_rounds == 0) {
                l.trainer.?.sites.close();
                try H.attach(l.b, l.trainer.?);
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
        fn schedulePlain(l: *Self, steps: u32) !void {
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
        fn schedule(l: *Self, steps: u32) !void {
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
