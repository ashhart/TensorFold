//! Sliding Weights' lessons, one learner for every backend over a host that gives the trainer, weights and files.
const std = @import("std");
const adapters = @import("learn_shape.zig");
const choice = @import("choice.zig");
const router = @import("router.zig");
const lesson_route = @import("lesson_route.zig");

/// Token ids whose answer starts at `start` (the rows from start - 1 on predict it); lessons decide at row `decide`.
pub const Example = struct { ids: []const u32, start: u32, decide: u32 };

/// One fact's examples; `more` another round, `undo` one back, `save` writes every kept lesson.
pub const Lesson = struct {
    train: []const Example = &.{},
    held: []const Example = &.{},
    near: []const Example = &.{},
    keep: []const Example = &.{},
    asked: []const Example = &.{}, // more wordings of the fact's questions, read only at their decision rows
    unasked: []const Example = &.{}, // questions it must not answer, read only at their decision rows
    undo: bool = false,
    steps: u32 = max_steps,
    more: bool = false,
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
        plan: std.ArrayList(Example) = .empty, // this round's fact answers in order, Adam after each
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
        bits: std.AutoHashMapUnmanaged(u64, u32) = .empty, // each prompt's earlier lessons acting on its reply
        asks: [2][]std.ArrayList(f32) = .{ &.{}, &.{} }, // [fact, steady][layers]: decision rows along the candidates
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
            for ([_][]const Example{ lesson.train, lesson.held, lesson.near, lesson.keep, lesson.asked, lesson.unasked }) |xs| for (xs) |ex| {
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
            for ([_][]const Example{ lesson.train, lesson.held, lesson.asked }) |xs| for (xs) |ex| try l.routes(ex, true);
            for ([_][]const Example{ lesson.keep, lesson.near, l.replay.items, lesson.unasked }) |xs| for (xs) |ex| try l.routes(ex, false);
            l.phase = .build;
        }

        /// A round's steps from the open block as it is now, which an undo comes back to.
        fn start(l: *Self) !void {
            try l.schedule(@max(@min(l.lesson.steps, max_steps), 1));
            l.trainer.?.sites.keep();
            l.can_undo = true;
            l.at = 0;
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

        /// Every kept lesson's blocks written as the model's living-weights file, beside its shards.
        fn saveOnce(l: *Self) !Step {
            l.phase = .idle;
            const t = l.trainer orelse return .{ .done = true };
            const ranks = t.sites.rank - @as(usize, if (l.opened and l.kept_rounds == 0) adapters.block else 0);
            if (ranks == 0) return .{ .done = true };
            const t0 = H.seconds(l.io);
            const bytes = try H.write(l.b, l.gpa, l.io, t, ranks);
            std.log.info("slide: {d} lessons saved beside the model's shards, each with its router: {d} MB in {d:.1} s", .{ ranks / adapters.block, bytes >> 20, H.seconds(l.io) - t0 });
            return .{ .done = true };
        }

        /// The lessons already kept that act on a reply to `ex` (their router votes at its decision row, read once).
        fn earlier(l: *Self, ex: Example) u32 {
            return l.bits.get(key(ex)) orelse 0;
        }

        /// One pass over `ex` in `mode` (`acting`: the open lesson acts on its reply); its first pass reads which kept lessons do.
        fn pass(l: *Self, ex: Example, mode: anytype, acting: bool) !f32 {
            const t = l.trainer.?;
            const k = key(ex);
            const known = l.bits.get(k);
            const mine: u32 = if (acting) @as(u32, 1) << @intCast(t.sites.first() / adapters.block) else 0;
            const got = try t.step(ex.ids, ex.start, mode, .{ .decide = ex.decide, .bits = (known orelse 0) | mine });
            if (known == null) try l.bits.put(l.gpa, k, latched(t.votes(), t.sites.needs(), t.sites.rank / adapters.block - @intFromBool(l.opened)));
            return got.loss;
        }

        /// A prompt for the routers to read, cut after its decision row; each prompt once, a fact's question or not.
        fn routes(l: *Self, ex: Example, fact: bool) !void {
            if ((try l.seen.getOrPut(l.gpa, key(ex))).found_existing) return;
            try l.routing.append(l.gpa, .{ .ex = .{ .ids = ex.ids[0 .. ex.decide + 2], .start = ex.decide + 1, .decide = ex.decide }, .fact = fact });
        }

        /// Layer by layer, a routing prompt's decision row along the candidates, as the last project pass left it.
        fn ask(l: *Self, r: Asked) !void {
            const t = l.trainer.?;
            for (l.asks[@intFromBool(!r.fact)], 0..) |*rows, k| try rows.appendSlice(l.gpa, t.projected(k, r.ex.ids.len - 1)[r.ex.decide * adapters.candidates ..][0..adapters.candidates]);
        }

        /// One example at a time: what must stay and the fact sketched, then projected; each routing prompt's decision
        /// rows copied out; then the block chosen and its router fitted.
        fn buildOnce(l: *Self) !Step {
            const t = l.trainer.?;
            const c = &l.choice.?;
            const facts = l.lesson.train;
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

        /// The open block's router: each layer's first direction and threshold, and the votes it needs.
        fn route(l: *Self) !void {
            const t = l.trainer.?;
            const layers = H.layers(l.b);
            const facts = try l.gpa.alloc([]const f32, layers);
            defer l.gpa.free(facts);
            const steady = try l.gpa.alloc([]const f32, layers);
            defer l.gpa.free(steady);
            for (facts, steady, l.asks[0], l.asks[1]) |*f, *s, a, b| {
                f.* = a.items;
                s.* = b.items;
            }
            var fit = try router.fit(l.gpa, facts, steady);
            defer fit.deinit(l.gpa);
            var open: usize = 0;
            for (0..layers) |k| {
                t.sites.aim(k, fit.dir[k * adapters.candidates ..][0..adapters.candidates], fit.tau[k]);
                open += @intFromBool(fit.tau[k] < adapters.shut);
            }
            const k = t.sites.first() / adapters.block;
            t.sites.needs()[k] = @floatFromInt(if (fit.need <= open) fit.need else 0); // more than it reads: it never acts
            std.log.info("slide: lesson {d}'s router reads {d} layers and acts on a reply when {d} agree: of {d} fact questions it never saw, most cleared at least {d} (the oddest {d}); {d} other prompts at most {d}{s}", .{ k + 1, open, fit.need, facts[0].len / adapters.candidates, fit.fact_low, fit.fact_least, steady[0].len / adapters.candidates, fit.steady_high, if (fit.apart()) "" else " (they overlap)" });
        }

        /// One question the open lesson must not take, read at its decision row; then its router fitted again.
        fn rerouteOnce(l: *Self) !Step {
            if (l.built < l.lesson.unasked.len) {
                const ex = l.lesson.unasked[l.built];
                l.built += 1;
                const r: Asked = .{ .ex = .{ .ids = ex.ids[0 .. ex.decide + 2], .start = ex.decide + 1, .decide = ex.decide }, .fact = false };
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

        /// One fact answer's gradients and Adam; every few steps (and at the end) the held-out answers are read.
        fn stepOnce(l: *Self) !Step {
            const ex = l.plan.items[l.at];
            const loss = try l.pass(ex, .learn, true);
            if (!std.math.isFinite(loss)) return error.NonfiniteStep;
            l.at += 1;
            l.loss += loss;
            const taken: u32 = @intCast(l.at);
            const end = l.at == l.plan.items.len;
            const back = (taken % check_every == 0 or end) and try l.recalled();
            if (!back and !end) return .{ .done = false, .changed = true };
            l.phase = .idle;
            l.kept_rounds += 1;
            return .{ .done = true, .changed = true, .report = .{ .learned = .{ .recalled = back, .steps = taken, .loss = l.loss / @as(f32, @floatFromInt(taken)) } } };
        }

        /// Whether every held-out answer comes back token for token (each its row's likeliest), the lesson acting.
        fn recalled(l: *Self) !bool {
            if (l.lesson.held.len == 0) return false;
            const t = l.trainer.?;
            const mine = @as(u32, 1) << @intCast(t.sites.first() / adapters.block);
            for (l.lesson.held) |ex| if (!(try t.step(ex.ids, ex.start, .loss, .{ .decide = ex.decide, .bits = l.earlier(ex) | mine })).recalled) return false;
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

        /// The round's steps: the fact's answers in fresh orders, Adam after each.
        fn schedule(l: *Self, steps: u32) !void {
            l.rounds += 1;
            var prng = std.Random.DefaultPrng.init(l.rounds);
            var facts: Deck = try .init(l.gpa, &.{l.lesson.train});
            defer facts.deinit(l.gpa);
            l.plan.clearRetainingCapacity();
            for (0..steps) |_| try l.plan.append(l.gpa, facts.draw(prng.random()));
        }
    };
}

/// A prompt the routers read, cut after its decision row, and whether it is a fact's question.
const Asked = struct { ex: Example, fact: bool };

/// The kept lessons (blocks below `blocks`) whose votes reach what they need: a bit each.
fn latched(votes: []const u32, need: []const f32, blocks: usize) u32 {
    var bits: u32 = 0;
    for (0..blocks) |b| if (need[b] > 0 and @as(f32, @floatFromInt(votes[b])) >= need[b]) {
        bits |= @as(u32, 1) << @intCast(b);
    };
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

test "a kept lesson acts on a reply only once its votes reach what it needs" {
    var votes: [32]u32 = @splat(0);
    var need: [32]f32 = @splat(0);
    votes[0..4].* = .{ 9, 3, 0, 12 };
    need[0..2].* = .{ 8, 4 };
    try std.testing.expectEqual(@as(u32, 0b0001), latched(&votes, &need, 4));
    try std.testing.expectEqual(@as(u32, 0), latched(&votes, &need, 0));
}
