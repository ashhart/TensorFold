//! Sliding Weights' lessons, one learner for every backend over a host that gives the trainer, weights and files.
const std = @import("std");
const adapters = @import("learn_shape.zig");
const choice = @import("choice.zig");
const router = @import("router.zig");
const lesson_route = @import("lesson_route.zig");
const Acting = lesson_route.Acting;
const train_mode = @import("train_mode.zig");

/// Token ids whose answer starts at `start` (the rows from start - 1 on predict it); lessons decide at row `decide`.
pub const Example = struct { ids: []const u32, start: u32, decide: u32 };

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
};

pub const Report = union(enum) {
    learned: struct { recalled: bool, steps: u32, loss: f32 },
    failed: []const u8,
};

pub const Step = struct { done: bool, changed: bool = false, report: ?Report = null };

const max_steps = 400;
const check_every = 20;
const per_step = 2; // fact answers whose gradients one Adam step takes together (a plain one and a thinking one)
const replay_cap = 64; // earlier lessons' questions kept as ones a new router must not take, the oldest giving way

/// A new lesson first sketches and measures what its block must read and when it acts (build), then learns (steps).
const Phase = enum { idle, build, steps, undo, reroute, drop, save };

/// The lessons over host H: its Trainer, max_rows, Ctx, layers, trainer, free, attach, write and seconds.
pub fn Learner(comptime H: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        io: std.Io,
        b: H.Ctx,
        trainer: ?*H.Trainer = null,
        lesson: Lesson = .{},
        phase: Phase = .idle,
        plan: std.ArrayList(Turn) = .empty, // this round's examples in order, Adam after each
        at: usize = 0, // the plan's next example
        loss: f32 = 0, // the round's summed loss
        replay: std.ArrayList(Example) = .empty, // earlier lessons' questions, owned
        last: []Example = &.{}, // the last lesson's questions, owned, joining replay when the next lesson begins
        kept_rounds: u32 = 0, // the last lesson's rounds still in the weights
        opened: bool = false, // the last lesson opened the block now last in the change
        built: usize = 0, // examples a new lesson has sketched or projected so far
        choice: ?choice.Choice = null, // the new block's directions, as its examples are projected
        can_undo: bool = false,
        rounds: u64 = 0, // rounds begun, which seeds each round's order
        bits: std.AutoHashMapUnmanaged(u64, Acting) = .empty, // each prompt's earlier lessons acting on its reply
        asks: [kinds][]std.ArrayList(f32) = @splat(&.{}), // [kind][layers]: decision rows along the candidates
        topic: bool = false, // the open lesson's router overlaps other prompts: those it takes, its change learns to keep
        keeps: std.ArrayList(Example) = .empty, // owned: answers the open lesson acts on and must leave as they were
        done: u32 = 0, // the round's fact steps
        answers: []Example = &.{}, // the lesson's answers, plain then thinking (the examples borrowed from the lesson)
        routing: std.ArrayList(Asked) = .empty, // each prompt a router reads once: cut after its decision row
        seen: std.AutoHashMapUnmanaged(u64, void) = .empty, // the prompts in routing

        pub fn init(gpa: std.mem.Allocator, io: std.Io, b: H.Ctx) Self {
            return .{ .gpa = gpa, .io = io, .b = b };
        }

        pub fn deinit(l: *Self) void {
            if (l.trainer) |t| H.free(t, l.gpa);
            if (l.choice) |*c| c.deinit();
            l.routing.deinit(l.gpa);
            for (l.asks) |side| {
                for (side) |*rows| rows.deinit(l.gpa);
                l.gpa.free(side);
            }
            l.bits.deinit(l.gpa);
            l.seen.deinit(l.gpa);
            l.plan.deinit(l.gpa);
            l.gpa.free(l.answers);
            for (l.keeps.items) |ex| l.gpa.free(ex.ids);
            l.keeps.deinit(l.gpa);
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
            if (lesson.save) {
                l.phase = .save;
                return;
            }
            if (lesson.drop) {
                l.phase = .drop;
                return;
            }
            if (lesson.reroute) {
                if (!l.opened) return error.NothingToRoute;
                l.built = 0;
                l.phase = .reroute;
                return;
            }
            if (lesson.train.len == 0) return;
            for ([_][]const Example{ lesson.train, lesson.think, lesson.held, lesson.near, lesson.keep, lesson.asked, lesson.unasked, lesson.kept }) |xs| for (xs) |ex| {
                if (ex.start < 1 or ex.start >= ex.ids.len or ex.ids.len - 1 > H.max_rows or ex.decide + 1 >= ex.ids.len) return error.ExampleTooLong;
            };
            if (l.trainer == null) {
                l.choice = try choice.Choice.init(l.gpa, H.layers(l.b));
                l.trainer = try H.trainer(l.gpa, l.io, l.b);
                for (&l.asks) |*side| {
                    side.* = try l.gpa.alloc(std.ArrayList(f32), H.layers(l.b));
                    for (side.*) |*rows| rows.* = .empty;
                }
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
            l.bits.clearRetainingCapacity();
            l.seen.clearRetainingCapacity();
            l.routing.clearRetainingCapacity();
            for (l.asks) |side| for (side) |*rows| rows.clearRetainingCapacity();
            for (l.keeps.items) |ex| l.gpa.free(ex.ids);
            l.keeps.clearRetainingCapacity();
            l.topic = false;
            for ([_][]const Example{ lesson.train, lesson.think, lesson.held, lesson.asked }) |xs| for (xs) |ex| try l.routes(ex, .fact);
            l.gpa.free(l.answers);
            l.answers = try std.mem.concat(l.gpa, Example, &.{ lesson.train, lesson.think });
            for (lesson.near) |ex| try l.routes(ex, .near);
            for (lesson.unasked) |ex| try l.routes(ex, .swap);
            for ([_][]const Example{ lesson.keep, l.replay.items }) |xs| for (xs) |ex| try l.routes(ex, .other);
            l.phase = .build;
        }

        /// A round's steps from the open block as it is now, which an undo comes back to.
        fn start(l: *Self) !void {
            try l.schedule(@max(@min(l.lesson.steps, max_steps), 1));
            l.trainer.?.sites.keep();
            l.can_undo = true;
            l.at = 0;
            l.done = 0;
            l.loss = 0;
            l.phase = .steps;
        }

        pub fn abort(l: *Self) void {
            l.phase = .idle;
        }

        /// One bounded unit: one example sketched or projected, one step of learning (a check every few), or the undo.
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
                    if (!l.can_undo) return .{ .done = true };
                    l.trainer.?.sites.restore();
                    l.can_undo = false;
                    l.kept_rounds -|= 1;
                    return .{ .done = true, .changed = true };
                },
                .build => return l.buildOnce(),
                .steps => return l.stepOnce(),
                .reroute => return l.rerouteOnce(),
                .drop => {
                    l.phase = .idle;
                    if (!l.opened) return .{ .done = true };
                    l.kept_rounds = 0;
                    try l.settleLast(&.{});
                    return .{ .done = true, .changed = true };
                },
                .save => return l.saveOnce(),
            }
        }

        /// Every kept lesson not yet written goes beside the model's shards, a file each.
        fn saveOnce(l: *Self) !Step {
            l.phase = .idle;
            const t = l.trainer orelse return .{ .done = true };
            const ranks = t.sites.rank - @as(usize, if (l.opened and l.kept_rounds == 0) adapters.block else 0);
            if (ranks == 0) return .{ .done = true };
            const t0 = H.seconds(l.io);
            const bytes = try H.write(l.b, l.gpa, l.io, t, ranks);
            std.log.info("slide: {d} lessons beside the model's shards, a file each with its router: {d} MB written in {d:.1} s", .{ ranks / adapters.block, bytes >> 20, H.seconds(l.io) - t0 });
            return .{ .done = true };
        }

        /// The lessons already kept that act on a reply to `ex` (their routers' scores at its decision row, read once).
        fn earlier(l: *Self, ex: Example) Acting {
            return l.bits.get(key(ex)) orelse .empty;
        }

        /// One pass over `ex` in `mode` (`acting`: the open lesson acts on its reply); its first pass reads which kept lessons do.
        fn pass(l: *Self, ex: Example, mode: train_mode.Mode, acting: bool) !f32 {
            const t = l.trainer.?;
            const k = key(ex);
            const known = l.bits.get(k);
            var bits = known orelse Acting.empty;
            if (acting) bits.set(t.sites.first() / adapters.block);
            const got = try t.step(ex.ids, ex.start, mode, .{ .decide = ex.decide, .bits = bits });
            if (known == null) try l.bits.put(l.gpa, k, latched(t.score(), t.sites.needs(), t.sites.rank / adapters.block - @intFromBool(l.opened)));
            return got.loss;
        }

        /// A prompt for the routers to read, cut after its decision row; each prompt once, a fact's question or not.
        fn routes(l: *Self, ex: Example, kind: Kind) !void {
            if ((try l.seen.getOrPut(l.gpa, key(ex))).found_existing) return;
            try l.routing.append(l.gpa, .{ .ex = .{ .ids = ex.ids[0 .. ex.decide + 2], .start = ex.decide + 1, .decide = ex.decide }, .kind = kind, .full = if (kind == .near or kind == .other) ex else null });
        }

        /// Layer by layer, a routing prompt's decision row along the candidates, as the last project pass left it.
        fn ask(l: *Self, r: Asked) !void {
            const t = l.trainer.?;
            for (l.asks[@backingInt(r.kind)], 0..) |*rows, k| try rows.appendSlice(l.gpa, t.projected(k, r.ex.ids.len - 1)[r.ex.decide * adapters.candidates ..][0..adapters.candidates]);
        }

        /// One example at a time: what must stay and the fact sketched, then projected; each routing prompt's decision
        /// rows copied out; then the block chosen and its router fitted.
        fn buildOnce(l: *Self) !Step {
            const t = l.trainer.?;
            const c = &l.choice.?;
            const facts = l.answers;
            const stay = [_][]const Example{ l.lesson.keep, l.lesson.near, l.replay.items };
            const steady = l.lesson.keep.len + l.lesson.near.len + l.replay.items.len;
            const asked = l.routing.items;
            var i = l.built;
            l.built += 1;
            if (i < steady) return l.sketch(pick(&stay, i), .avoid);
            i -= steady;
            if (i < facts.len) return l.sketch(facts[i], .seek);
            i -= facts.len;
            if (i == 0) {
                t.sites.frame();
                c.reset();
            }
            if (i < steady) {
                const ex = pick(&stay, i);
                _ = try l.pass(ex, .project, false);
                for (0..t.sites.list.len) |k| c.add(k, t.projected(k, ex.ids.len - 1), false, ex.start - 1);
                return .{ .done = false };
            }
            i -= steady;
            if (i < facts.len) {
                const ex = facts[i];
                _ = try l.pass(ex, .project, false);
                for (0..t.sites.list.len) |k| c.add(k, t.projected(k, ex.ids.len - 1)[(ex.start - 1) * adapters.candidates ..], true, 0);
                return .{ .done = false };
            }
            i -= facts.len;
            if (i < asked.len) {
                _ = try l.pass(asked[i].ex, .project, false);
                try l.ask(asked[i]);
                return .{ .done = false };
            }
            try c.choose();
            try t.sites.open(c.coef);
            l.opened = true;
            try l.route();
            try H.attach(l.b, t);
            try l.start();
            return .{ .done = false };
        }

        /// The open block's router. Where its fact's questions do not stand apart from every other prompt, it acts on
        /// what most of them clear, and every answered prompt it then takes (a near miss, an earlier lesson's question, a
        /// keep prompt) the change learns to leave as it was.
        fn route(l: *Self) !void {
            const t = l.trainer.?;
            var fit = try l.fitWith(&.{.fact}, &.{ .near, .swap, .other });
            defer fit.deinit(l.gpa);
            const layers = H.layers(l.b);
            for (0..layers) |k| t.sites.aim(k, fit.dir[k * adapters.candidates ..][0..adapters.candidates], fit.weight[k], fit.mean[k]);
            const k = t.sites.first() / adapters.block;
            t.sites.needs()[k] = if (fit.reads > 0) @max(fit.need, 1e-3) else 0; // 0: it never acts
            l.topic = !fit.apart();
            var kept: usize = 0;
            var unanswered: usize = 0;
            if (l.topic) for (l.routing.items) |r| {
                if (r.kind == .fact) continue;
                const rows = l.asks[@backingInt(r.kind)];
                const at = l.order(r);
                const parts = try l.gpa.alloc([]const f32, layers);
                defer l.gpa.free(parts);
                for (parts, rows) |*p, x| p.* = x.items;
                if (fit.score(parts, at) < fit.need) continue;
                const full = r.full orelse {
                    unanswered += 1;
                    continue;
                };
                try l.keeps.append(l.gpa, .{ .ids = try l.gpa.dupe(u32, full.ids), .start = full.start, .decide = full.decide });
                kept += 1;
            };
            std.log.info("slide: lesson {d}'s router reads {d} layers and acts on a reply that scores {d:.2} there: of its fact's questions left out of the fit, most scored at least {d:.2} (the oddest {d:.2}); of the other prompts, at most {d:.2}{s}", .{ k + 1, fit.reads, fit.need, fit.fact_low, fit.fact_least, fit.steady_high, if (fit.apart()) "" else " (they overlap)" });
            if (l.topic) std.log.info("slide: it also takes {d} answered prompts, which it learns to leave as they were, and {d} questions it has no answer for", .{ kept, unanswered });
        }

        /// Where routing prompt `r` sits among the decision rows of its kind.
        fn order(l: *const Self, r: Asked) usize {
            var i: usize = 0;
            for (l.routing.items) |x| {
                if (x.ex.ids.ptr == r.ex.ids.ptr and x.ex.ids.len == r.ex.ids.len) return i;
                if (x.kind == r.kind) i += 1;
            }
            unreachable;
        }

        /// A router fitted on the prompts of kinds `pos` against those of kinds `neg`, layer by layer.
        fn fitWith(l: *Self, pos: []const Kind, neg: []const Kind) !router.Fit {
            var arena_state = std.heap.ArenaAllocator.init(l.gpa);
            defer arena_state.deinit();
            const a = arena_state.allocator();
            const layers = H.layers(l.b);
            const facts = try a.alloc([]const f32, layers);
            const steady = try a.alloc([]const f32, layers);
            for (0..layers) |k| {
                facts[k] = try l.join(a, pos, k);
                steady[k] = try l.join(a, neg, k);
            }
            return router.fit(l.gpa, facts, steady);
        }

        /// Layer k's decision rows of the prompts of kinds `of`, one after another.
        fn join(l: *Self, a: std.mem.Allocator, of: []const Kind, k: usize) ![]const f32 {
            var out: std.ArrayList(f32) = .empty;
            for (of) |kind| try out.appendSlice(a, l.asks[@backingInt(kind)][k].items);
            return out.items;
        }

        /// A question the open lesson took: a topic lesson keeps its answer as it was (the next round learns it); a
        /// strict one reads it at its decision row and fits its router again without it.
        fn rerouteOnce(l: *Self) !Step {
            if (l.topic) {
                for (l.lesson.kept) |ex| try l.keeps.append(l.gpa, .{ .ids = try l.gpa.dupe(u32, ex.ids), .start = ex.start, .decide = ex.decide });
                l.phase = .idle;
                return .{ .done = true };
            }
            if (l.built < l.lesson.unasked.len) {
                const ex = l.lesson.unasked[l.built];
                l.built += 1;
                const r: Asked = .{ .ex = .{ .ids = ex.ids[0 .. ex.decide + 2], .start = ex.decide + 1, .decide = ex.decide }, .kind = .swap };
                _ = try l.pass(r.ex, .project, false);
                try l.ask(r);
                return .{ .done = false };
            }
            l.phase = .idle;
            try l.route();
            try H.attach(l.b, l.trainer.?);
            return .{ .done = true, .changed = true };
        }

        fn sketch(l: *Self, ex: Example, comptime mode: @EnumLiteral()) !Step {
            _ = try l.pass(ex, mode, false);
            return .{ .done = false };
        }

        /// One answer's gradients (a fact's, or one the lesson keeps), Adam after a step's last; every few steps, the
        /// held-out answers are read.
        fn stepOnce(l: *Self) !Step {
            const turn = l.plan.items[l.at];
            const loss = try l.pass(turn.ex, if (turn.last) .learn else .grad, true);
            if (!std.math.isFinite(loss)) return error.NonfiniteStep;
            l.at += 1;
            if (turn.fact) l.loss += loss;
            if (turn.last) l.done += 1;
            if (!turn.last) return .{ .done = false, .changed = true };
            const end = l.at == l.plan.items.len;
            const back = (l.done % check_every == 0 or end) and try l.recalled();
            if (!back and !end) return .{ .done = false, .changed = true };
            l.phase = .idle;
            l.kept_rounds += 1;
            return .{ .done = true, .changed = true, .report = .{ .learned = .{ .recalled = back, .steps = l.done, .loss = l.loss / @as(f32, @floatFromInt(@max(l.done * per_step, 1))) } } };
        }

        /// Whether every held-out answer comes back token for token (each its row's likeliest), the lesson acting.
        fn recalled(l: *Self) !bool {
            if (l.lesson.held.len == 0) return false;
            const t = l.trainer.?;
            for (l.lesson.held) |ex| {
                var bits = l.earlier(ex);
                bits.set(t.sites.first() / adapters.block);
                if (!(try t.step(ex.ids, ex.start, .loss, .{ .decide = ex.decide, .bits = bits })).recalled) return false;
            }
            return true;
        }

        /// The last lesson's questions into replay if a round of it stayed (else its block out); this lesson's are last.
        fn settleLast(l: *Self, answers: []const Example) !void {
            const now = try own(l.gpa, answers);
            if (l.opened and l.kept_rounds == 0) {
                l.trainer.?.sites.needs()[l.trainer.?.sites.first() / adapters.block] = 0;
                l.trainer.?.sites.close();
                try H.attach(l.b, l.trainer.?);
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

        /// The round's steps, each a plain answer and a thinking one (two plain without thinking answers), then an answer
        /// the lesson keeps if it has any; every deck in fresh orders, Adam after each step's last.
        fn schedule(l: *Self, steps: u32) !void {
            l.rounds += 1;
            var prng = std.Random.DefaultPrng.init(l.rounds);
            var plain: Deck = try .init(l.gpa, &.{l.lesson.train});
            defer plain.deinit(l.gpa);
            var think: Deck = try .init(l.gpa, &.{if (l.lesson.think.len > 0) l.lesson.think else l.lesson.train});
            defer think.deinit(l.gpa);
            var keeps: Deck = try .init(l.gpa, &.{l.keeps.items});
            defer keeps.deinit(l.gpa);
            l.plan.clearRetainingCapacity();
            const r = prng.random();
            for (0..steps) |_| {
                const keep = keeps.cards.len > 0;
                try l.plan.append(l.gpa, .{ .ex = plain.draw(r), .fact = true, .last = false });
                try l.plan.append(l.gpa, .{ .ex = think.draw(r), .fact = true, .last = !keep });
                if (keep) try l.plan.append(l.gpa, .{ .ex = keeps.draw(r), .fact = false, .last = true });
            }
        }
    };
}

/// What a routing prompt is to its lesson: its fact's question, a near miss (with the answer it had), a question about
/// someone else (prompt only), or anything else that must stay as it is.
const Kind = enum(u2) { fact, near, swap, other };
const kinds = std.enums.values(Kind).len;

/// A prompt the routers read, cut after its decision row, and what it is to the lesson.
const Asked = struct { ex: Example, kind: Kind, full: ?Example = null };

/// One example of a step: the fact's answer or one the lesson must leave as it was, and whether Adam steps after it.
const Turn = struct { ex: Example, fact: bool, last: bool };

/// The kept lessons (blocks below `blocks`) whose router scores reach what they need.
fn latched(score: []const f32, need: []const f32, blocks: usize) Acting {
    var bits: Acting = .empty;
    for (0..blocks) |b| if (need[b] > 0 and score[b] >= need[b]) bits.set(b);
    return bits;
}

/// A prompt by its rows through its decision row: what a router reads of it.
fn key(ex: Example) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ex.ids[0 .. ex.decide + 1]));
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
        o.* = .{ .ids = try gpa.dupe(u32, x.ids), .start = x.start, .decide = x.decide };
        made += 1;
    }
    return out;
}

fn disown(gpa: std.mem.Allocator, xs: []Example) void {
    for (xs) |x| gpa.free(x.ids);
    gpa.free(xs);
}

test "a kept lesson acts on a reply only once its router score reaches what it needs" {
    var score: [lesson_route.max_blocks]f32 = @splat(0);
    var need: [lesson_route.max_blocks]f32 = @splat(0);
    score[0..4].* = .{ 2.5, 1.2, 0, 3 };
    need[0..2].* = .{ 2, 1.5 };
    const four = latched(&score, &need, 4);
    try std.testing.expect(four.isSet(0) and four.count() == 1);
    try std.testing.expectEqual(@as(usize, 0), latched(&score, &need, 0).count());
}
