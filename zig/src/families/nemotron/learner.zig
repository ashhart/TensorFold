//! Sliding Weights' lessons, one learner for every backend over a host that gives the trainer, weights and files.
const std = @import("std");
const adapters = @import("learn_shape.zig");
const choice = @import("choice.zig");
const router = @import("router.zig");
const lesson_route = @import("lesson_route.zig");
const Acting = lesson_route.Acting;
const latched = lesson_route.latched;
const train_mode = @import("train_mode.zig");
const editor = @import("editor.zig");
const probe = @import("probe.zig");
const pocket = @import("pocket.zig");
const whole_lessons = @import("whole.zig");
const examples = @import("lesson_examples.zig");

pub const Example = examples.Example;
pub const Lesson = examples.Lesson;
const max_steps = examples.max_steps;
const Deck = examples.Deck;
const key = examples.key;
const total = examples.total;
const pick = examples.pick;
const own = examples.own;
const disown = examples.disown;
const disownLesson = examples.disownLesson;

/// A projection's 4-bit codes as a backend holds them live, on the grid of their scales and biases.
pub const Codes = struct { words: []u32, scales: []const u16, biases: []const u16 };

pub const Report = union(enum) {
    learned: struct { recalled: bool, steps: u32, loss: f32 },
    failed: []const u8,
};

pub const Step = struct { done: bool, changed: bool = false, report: ?Report = null };

const check_every = 20;
const per_step = 2; // fact answers whose gradients one Adam step takes together (a plain one and a thinking one)
const replay_cap = 256; // earlier lessons' answers a new lesson leaves as they are, the oldest giving way
const replay_each = 8; // of a lesson's answers, its first (plain questions) join the replay, so every fact keeps some

/// A lesson builds (build) and learns (steps); an edit is written at its subject (edit), more rounds report (again).
const Phase = enum { idle, build, steps, undo, reroute, drop, save, edit, again, probe, pocket, retire };

/// The lessons over host H: its Trainer, max_rows, Ctx, layers, trainer, free, attach, write and seconds.
pub fn Learner(comptime H: type) type {
    return struct {
        const Self = @This();
        const has_edits = @hasDecl(H.Trainer, "key") and @hasDecl(H, "commit"); // keys, a nudged row, codes to write
        const has_pocket = @hasDecl(H.Trainer, "normed") and @hasDecl(H, "pocketSave"); // inputs, gradients, a pocket to write
        const has_whole = has_edits or has_pocket; // lessons that change the model itself
        const Edits = if (has_edits) editor.Editor(H.Trainer) else void;
        const Probes = if (has_edits) probe.Probe(H.Trainer) else void;
        const Pockets = if (has_pocket) pocket.Pocket(H.Trainer) else void;
        pub const Host = H;

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
        replay_of: std.ArrayList(u32) = .empty, // [replay]: the lesson each came from
        live: std.ArrayList(Example) = .empty, // replay but the lessons of facts a newer one replaced (borrowed)
        live_of: std.ArrayList(u32) = .empty, // [live]: the lesson each came from
        last: []Example = &.{}, // the last lesson's questions, owned, joining replay when the next lesson begins
        last_lesson: u32 = 0, // the last lesson's number, which its neurons record
        lessons: u32 = 0, // lessons begun so far
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
        whole: bool, // lessons edit the weights: they act on every row, no router (TF_SLIDE_ROUTED: a router a lesson)
        edits: Edits, // an edit of the weights' state, from its keys to its change
        extra: std.ArrayList(Example) = .empty, // owned: answers a check found moved, kept by the next solve too
        stay: [5][]const Example = @splat(&.{}), // what an edit keeps: keep, near, earlier lessons, others, extra
        edited: bool = false, // the open edit came back when its held questions were asked
        owned: Lesson = .{}, // the open edit's lesson, copied: a reroute solves it again after its request is gone
        method: whole_lessons.Method, // how a whole lesson changes the model: a pocket's neurons by default
        probe: Probes,
        pockets: Pockets, // every MoE layer's pocket of learned neurons
        pocketed: bool = false, // the last lesson put neurons in the pocket

        pub fn init(gpa: std.mem.Allocator, io: std.Io, b: H.Ctx) Self {
            return .{ .gpa = gpa, .io = io, .b = b, .whole = std.c.getenv("TF_SLIDE_ROUTED") == null, .edits = if (has_edits) Edits.init(gpa) else {}, .method = if (has_edits) whole_lessons.method() else .pocket, .probe = if (has_edits) Probes.init(gpa) else {}, .pockets = if (has_pocket) Pockets.init(gpa) else {} };
        }

        pub fn deinit(l: *Self) void {
            if (has_edits) l.edits.deinit();
            if (has_edits) l.probe.deinit();
            if (has_pocket) l.pockets.deinit();
            disownLesson(l.gpa, l.owned);
            for (l.extra.items) |ex| l.gpa.free(ex.ids);
            l.extra.deinit(l.gpa);
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
            l.replay_of.deinit(l.gpa);
            l.live.deinit(l.gpa);
            l.live_of.deinit(l.gpa);
        }

        /// The earlier lessons' questions a new lesson stays silent on, but those of facts a newer one replaced.
        pub fn liveReplay(l: *Self) ![]const Example {
            l.live.clearRetainingCapacity();
            l.live_of.clearRetainingCapacity();
            const gone: []const u32 = if (has_pocket) l.pockets.retired_lessons.items else &.{};
            for (l.replay.items, l.replay_of.items) |ex, of| if (std.mem.indexOfScalar(u32, gone, of) == null) {
                try l.live.append(l.gpa, ex);
                try l.live_of.append(l.gpa, of);
            };
            return l.live.items;
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
                if (!l.opened and !l.pocketed) return error.NothingToRoute;
                l.built = 0;
                l.phase = .reroute;
                return;
            }
            if (lesson.retire.len > 0 or lesson.unretire or lesson.forget.len > 0 or lesson.unforget) {
                try l.ready();
                l.phase = .retire;
                return;
            }
            if (lesson.train.len == 0) return;
            for ([_][]const Example{ lesson.train, lesson.think, lesson.held, lesson.near, lesson.keep, lesson.asked, lesson.unasked, lesson.kept }) |xs| for (xs) |ex| {
                if (ex.start < 1 or ex.start >= ex.ids.len or ex.ids.len - 1 > H.max_rows or ex.decide + 1 >= ex.ids.len) return error.ExampleTooLong;
            };
            try l.ready();
            if (lesson.more) {
                if (l.whole) return if (has_whole) whole_lessons.more(l) else error.EditsUnsupported;
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
            if (l.whole) return if (has_whole) whole_lessons.begin(l, lesson) else error.EditsUnsupported;
            l.gpa.free(l.answers);
            l.answers = try std.mem.concat(l.gpa, Example, &.{ lesson.train, lesson.think });
            l.phase = .build;
            for ([_][]const Example{ lesson.train, lesson.think, lesson.held, lesson.asked }) |xs| for (xs) |ex| try l.routes(ex, .fact);
            for (lesson.near) |ex| try l.routes(ex, .near);
            for (lesson.unasked) |ex| try l.routes(ex, .swap);
            for ([_][]const Example{ lesson.keep, l.replay.items }) |xs| for (xs) |ex| try l.routes(ex, .other);
        }

        /// The trainer, made the first time a lesson needs it.
        fn ready(l: *Self) !void {
            if (l.trainer != null) return;
            const t0 = H.seconds(l.io);
            l.choice = try choice.Choice.init(l.gpa, H.layers(l.b));
            l.trainer = try H.trainer(l.gpa, l.io, l.b);
            std.log.info("slide: the trainer is ready in {d:.1} s", .{H.seconds(l.io) - t0});
            l.trainer.?.sites.whole = l.whole;
            for (&l.asks) |*side| {
                side.* = try l.gpa.alloc(std.ArrayList(f32), H.layers(l.b));
                for (side.*) |*rows| rows.* = .empty;
            }
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
                    if (has_whole and l.whole and try whole_lessons.undo(l)) return .{ .done = true, .changed = true };
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
                    if (has_whole and l.whole and try whole_lessons.undo(l)) return .{ .done = true, .changed = true };
                    if (!l.opened) return .{ .done = true };
                    l.kept_rounds = 0;
                    try l.settleLast(&.{});
                    return .{ .done = true, .changed = true };
                },
                .save => return l.saveOnce(),
                .retire => {
                    l.phase = .idle;
                    if (has_whole and l.whole) if (try whole_lessons.retire(l, l.lesson)) |s| return s;
                    return .{ .done = true };
                },
                .edit => return if (has_edits) whole_lessons.editOnce(l) else unreachable,
                .probe => return if (has_edits) whole_lessons.probeOnce(l) else unreachable,
                .pocket => return if (has_pocket) whole_lessons.pocketOnce(l) else unreachable,
                .again => {
                    l.phase = .idle;
                    return .{ .done = true, .report = .{ .learned = .{ .recalled = l.edited, .steps = 0, .loss = if (has_edits) l.edits.last_loss else 0 } } };
                },
            }
        }

        /// Every kept lesson not yet written goes beside the model's shards, a file each.
        fn saveOnce(l: *Self) !Step {
            l.phase = .idle;
            const t = l.trainer orelse return .{ .done = true };
            if (l.whole) { // a lesson of the model itself goes into its own shards: the pocket, or an edit's codes
                if (has_whole and !try whole_lessons.save(l)) try whole_lessons.foldOpen(l);
                return .{ .done = true, .changed = true };
            }
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

        /// A pass of `ex` in `mode` (`acting`: the open lesson acts on its reply); its first reads which kept ones act.
        fn pass(l: *Self, ex: Example, mode: train_mode.Mode, acting: bool) !f32 {
            const t = l.trainer.?;
            const k = key(ex);
            const known = l.bits.get(k);
            var bits = if (l.whole) l.keptLessons() else known orelse Acting.empty;
            if (acting) bits.set(t.sites.first() / adapters.block);
            const got = try t.step(ex.ids, ex.start, mode, .{ .decide = ex.decide, .bits = bits, .whole = l.whole });
            if (!l.whole and known == null) try l.bits.put(l.gpa, k, latched(t.score(), t.sites.needs(), t.sites.rank / adapters.block - @intFromBool(l.opened)));
            return got.loss;
        }

        /// Every lesson kept before the open one, all acting (an in-place learner's lessons are part of the model).
        pub fn keptLessons(l: *const Self) Acting {
            var bits: Acting = .empty;
            for (0..l.trainer.?.sites.rank / adapters.block - @intFromBool(l.opened)) |b| bits.set(b);
            return bits;
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

        /// One example at a time: sketched, projected, routing rows copied; then the block chosen, its router fitted.
        fn buildOnce(l: *Self) !Step {
            const t = l.trainer.?;
            const c = &l.choice.?;
            // an edit of the weights also keeps the questions about others, and reads every wording of the fact's
            const stay_all = [_][]const Example{ l.lesson.keep, l.lesson.near, l.replay.items, l.lesson.unasked };
            const fact_all = [_][]const Example{ l.answers, l.lesson.asked };
            const stay: []const []const Example = if (l.whole) &stay_all else stay_all[0..3];
            const facts: []const []const Example = if (l.whole) &fact_all else fact_all[0..1];
            const steady = total(stay);
            const nfacts = total(facts);
            const asked = l.routing.items;
            var i = l.built;
            l.built += 1;
            if (i < steady) return l.sketch(pick(stay, i), .avoid);
            i -= steady;
            if (i < nfacts) return l.sketch(pick(facts, i), .seek);
            i -= nfacts;
            if (i == 0) {
                t.sites.frame();
                c.reset();
            }
            if (i < steady) {
                const ex = pick(stay, i);
                _ = try l.pass(ex, .project, false);
                for (0..t.sites.list.len) |k| c.add(k, t.projected(k, ex.ids.len - 1), false, ex.start - 1);
                return .{ .done = false };
            }
            i -= steady;
            if (i < nfacts) {
                const ex = pick(facts, i);
                _ = try l.pass(ex, .project, false);
                for (0..t.sites.list.len) |k| {
                    const rows = t.projected(k, ex.ids.len - 1);
                    // a weight edit: the question's rows, evenly; a routed block: the answer, its first row heavily
                    if (l.whole) c.add(k, rows[0 .. (ex.decide + 1) * adapters.candidates], true, std.math.maxInt(usize)) else c.add(k, rows[(ex.start - 1) * adapters.candidates ..], true, 0);
                }
                return .{ .done = false };
            }
            i -= nfacts;
            if (i < asked.len) {
                _ = try l.pass(asked[i].ex, .project, false);
                try l.ask(asked[i]);
                return .{ .done = false };
            }
            try c.choose();
            try t.sites.open(c.coef);
            l.opened = true;
            if (!l.whole) try l.route();
            try H.attach(l.b, t);
            try l.start();
            return .{ .done = false };
        }

        /// The open block's router: if facts don't stand apart it acts on what most clear, learning to leave the rest.
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

        /// A taken question: a topic lesson keeps its answer for next round; a strict one refits its router without it.
        fn rerouteOnce(l: *Self) !Step {
            if (l.whole) return if (has_whole) whole_lessons.reroute(l) else unreachable;
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

        /// An answer's gradients (fact or kept), Adam after a step's last; the held-out answers read every few steps.
        fn stepOnce(l: *Self) !Step {
            const turn = l.plan.items[l.at];
            const loss = try l.pass(turn.ex, if (turn.last) .learn else .grad, true);
            if (!std.math.isFinite(loss)) return error.NonfiniteStep;
            l.at += 1;
            if (turn.fact) l.loss += loss;
            if (turn.last) l.done += 1;
            if (!turn.last) return .{ .done = false, .changed = true };
            const end = l.at == l.plan.items.len;
            const back = (l.done % check_every == 0 or end) and try l.recalled(true);
            if (!back and !end) return .{ .done = false, .changed = true };
            l.phase = .idle;
            l.kept_rounds += 1;
            return .{ .done = true, .changed = true, .report = .{ .learned = .{ .recalled = back, .steps = l.done, .loss = l.loss / @as(f32, @floatFromInt(@max(l.done * per_step, 1))) } } };
        }

        pub const Placed = pocket.Placed;
        pub const Progress = if (has_pocket) Pockets.Progress else void;

        /// The open pocket lesson's progress, for a view of it; null when no pocket lesson runs.
        pub fn progress(l: *const Self) ?Progress {
            if (comptime !has_pocket) return null;
            return if (l.phase == .pocket) l.pockets.progress() else null;
        }

        /// The last pocket lesson's neurons as placed, into `out`, while they are in the pocket; the ones written.
        pub fn placed(l: *const Self, out: []Placed) []Placed {
            if (comptime !has_pocket) return out[0..0];
            return if (l.method == .pocket and l.pocketed) l.pockets.placed(out) else out[0..0];
        }

        /// Whether every held-out answer comes back token for token, the lesson acting (`open`: from its block).
        pub fn recalled(l: *Self, open: bool) !bool {
            return l.recalls(l.lesson.held, open);
        }

        pub fn recalls(l: *Self, held: []const Example, open: bool) !bool {
            if (held.len == 0) return false;
            const t = l.trainer.?;
            for (held) |ex| {
                var bits = if (l.whole) l.keptLessons() else l.earlier(ex);
                if (open) bits.set(t.sites.first() / adapters.block);
                if (!(try t.step(ex.ids, ex.start, .loss, .{ .decide = ex.decide, .bits = bits, .whole = l.whole })).recalled) return false;
            }
            return true;
        }

        /// The last lesson's questions into replay if a round stayed, else its block out; this lesson's become `last`.
        fn settleLast(l: *Self, answers: []const Example) !void {
            const now = try own(l.gpa, answers);
            if (has_whole and l.whole) try whole_lessons.settle(l);
            if (l.opened and l.kept_rounds == 0) {
                l.trainer.?.sites.needs()[l.trainer.?.sites.first() / adapters.block] = 0;
                l.trainer.?.sites.close();
                try H.attach(l.b, l.trainer.?);
            }
            l.opened = false;
            if (l.kept_rounds > 0) {
                const each = @min(l.last.len, replay_each);
                for (l.last[0..each]) |ex| {
                    if (l.replay.items.len == replay_cap) {
                        l.gpa.free(l.replay.orderedRemove(0).ids);
                        _ = l.replay_of.orderedRemove(0);
                    }
                    try l.replay.append(l.gpa, ex);
                    try l.replay_of.append(l.gpa, l.last_lesson);
                }
                for (l.last[each..]) |ex| l.gpa.free(ex.ids);
                l.gpa.free(l.last);
            } else disown(l.gpa, l.last);
            l.last = now;
            l.last_lesson = l.lessons;
            l.lessons += 1;
            l.kept_rounds = 0;
        }

        /// The round's shuffled steps: a plain and a thinking answer, then kept ones (four for an edit), then Adam.
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
            const keeping: usize = if (keeps.cards.len == 0) 0 else if (l.whole) 4 else 1; // an edit reaches every answer
            for (0..steps) |_| {
                try l.plan.append(l.gpa, .{ .ex = plain.draw(r), .fact = true, .last = false });
                try l.plan.append(l.gpa, .{ .ex = think.draw(r), .fact = true, .last = keeping == 0 });
                for (0..keeping) |j| try l.plan.append(l.gpa, .{ .ex = keeps.draw(r), .fact = false, .last = j + 1 == keeping });
            }
        }
    };
}

/// What a routing prompt is: its fact's question, a near miss (with its answer), one about someone else, or any other.
const Kind = enum(u2) { fact, near, swap, other };
const kinds = std.enums.values(Kind).len;

/// A prompt the routers read, cut after its decision row, and what it is to the lesson.
const Asked = struct { ex: Example, kind: Kind, full: ?Example = null };

/// One example of a step: the fact's answer or one the lesson must leave as it was, and whether Adam steps after it.
const Turn = struct { ex: Example, fact: bool, last: bool };
