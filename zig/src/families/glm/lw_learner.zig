//! GLM-5.3-Flash's Sliding Weights learner: Nemotron's state machine (families/nemotron/slide.zig) over one site,
//! layers.44's shared-expert down_proj. Generic over its trainer, so the machine is tested on the host with a fake one.
const std = @import("std");
const lm = @import("lw_math.zig");
const lw_sites = @import("lw_sites.zig");
const choice = @import("../nemotron/choice.zig");

pub const block = lm.block;
pub const max_rank = lm.max_rank;
pub const candidates = lm.candidates;

/// Rows a step reads at most (Nemotron's train.max_rows).
pub const max_rows = 256;

/// A step's loss alone, with the open block's gradient, or with that and Adam; or the site's inputs sketched or projected.
pub const Mode = enum { loss, grad, learn, avoid, seek, project };

/// Token ids whose answer starts at `start`: the rows from start - 1 on predict it.
pub const Example = struct { ids: []const u32, start: u32 };

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
const plain_steps = 60;
const replay_cap = 64;
const gate_margin: f32 = 0.02;
const hold: f64 = 1;

const Phase = enum { idle, build, steps, undo, commit, save };
const Turn = struct { ex: Example, fact: bool, last: bool };

/// The learner over trainer type T, which provides: `init(gpa, T.Ctx) !*T`, `deinit(gpa)`, fields `sites: lw_sites.Sites`
/// and `sketched: usize`, `step(ids, start, Mode) !lm.Result`, `projected(k, rows) []const f32`, `attach(on)`, `write(ranks) !usize`.
pub fn Learner(comptime T: type) type {
    return struct {
        const L = @This();

        gpa: std.mem.Allocator,
        ctx: T.Ctx,
        trainer: ?*T = null,
        lesson: Lesson = .{},
        phase: Phase = .idle,
        plan: std.ArrayList(Turn) = .empty,
        at: usize = 0,
        taken: u32 = 0,
        loss: f32 = 0,
        replay: std.ArrayList(Example) = .empty,
        last: []Example = &.{},
        kept_rounds: u32 = 0,
        opened: bool = false,
        built: usize = 0,
        choice: ?choice.Choice = null,
        answers: u32 = 0,
        can_undo: bool = false,
        plain: bool = false,
        rounds: u64 = 0,

        pub fn init(gpa: std.mem.Allocator, ctx: T.Ctx) L {
            return .{ .gpa = gpa, .ctx = ctx };
        }

        pub fn deinit(l: *L) void {
            if (l.trainer) |t| t.deinit(l.gpa);
            if (l.choice) |*c| c.deinit();
            l.plan.deinit(l.gpa);
            disown(l.gpa, l.last);
            for (l.replay.items) |ex| l.gpa.free(ex.ids);
            l.replay.deinit(l.gpa);
        }

        /// The trainer, made on the first lesson (it takes over a learned folder's committed blocks then).
        pub fn ensure(l: *L) !*T {
            if (l.trainer == null) {
                l.choice = try choice.Choice.init(l.gpa, 1);
                errdefer {
                    l.choice.?.deinit();
                    l.choice = null;
                }
                l.trainer = try T.init(l.gpa, l.ctx);
            }
            return l.trainer.?;
        }

        pub fn begin(l: *L, lesson: Lesson) !void {
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
                if (ex.start < 1 or ex.start >= ex.ids.len or ex.ids.len - 1 > max_rows) return error.ExampleTooLong;
            };
            const t = try l.ensure();
            if (lesson.more) {
                if (!l.opened) return error.NothingToContinue;
                return l.start();
            }
            try l.settleLast(lesson.train);
            if (t.sites.rank + block > max_rank) return error.LearnedChangeFull;
            t.sites.clear();
            t.sketched = 0;
            l.built = 0;
            l.phase = .build;
        }

        fn start(l: *L) !void {
            const steps = @max(@min(l.lesson.steps, max_steps), 1);
            if (l.plain) try l.schedulePlain(steps) else try l.schedule(steps);
            l.trainer.?.sites.keep();
            l.can_undo = true;
            l.at = 0;
            l.taken = 0;
            l.loss = 0;
            l.phase = .steps;
        }

        pub fn abort(l: *L) void {
            l.phase = .idle;
        }

        pub fn step(l: *L) Step {
            return l.advance() catch |e| {
                if (l.phase == .steps) l.trainer.?.sites.restore();
                l.phase = .idle;
                return .{ .done = true, .changed = true, .report = .{ .failed = @errorName(e) } };
            };
        }

        fn advance(l: *L) !Step {
            switch (l.phase) {
                .idle => return .{ .done = true },
                .undo => {
                    l.phase = .idle;
                    const t = l.trainer orelse return .{ .done = true };
                    if (l.plain) {
                        t.sites.close();
                        t.attach(true);
                        l.plain = false;
                        l.opened = false;
                        l.kept_rounds = 0;
                        return .{ .done = true, .changed = true };
                    }
                    if (!l.can_undo) return .{ .done = true };
                    t.sites.restore();
                    l.can_undo = false;
                    l.kept_rounds -|= 1;
                    return .{ .done = true, .changed = true };
                },
                .build => return l.buildOnce(),
                .steps => return l.stepOnce(),
                .commit => return l.commitOnce(),
                .save => return l.saveOnce(),
            }
        }

        /// Every kept lesson's block into the model folder's sidecar (a still-gated open block stays out).
        fn saveOnce(l: *L) !Step {
            l.phase = .idle;
            const t = l.trainer orelse return .{ .done = true };
            const ranks = t.sites.rank - @as(usize, if (l.opened and !l.plain) block else 0);
            if (ranks == 0) return .{ .done = true };
            const bytes = try t.write(ranks);
            t.sites.roundSaved(ranks); // what the sidecar holds, so this server and a reload of the folder agree
            std.log.info("slide: {d} lessons written into the model folder's living weights ({d} KB)", .{ ranks / block, bytes >> 10 });
            return .{ .done = true, .changed = true };
        }

        fn commitOnce(l: *L) !Step {
            l.phase = .idle;
            if (!l.opened or l.kept_rounds == 0) return .{ .done = true };
            const t = l.trainer.?;
            const c = &l.choice.?;
            for (0..t.sites.list.len) |k| try c.refit(k, hold);
            t.sites.plain(c.coef);
            t.attach(true);
            l.plain = true;
            std.log.info("slide: lesson {d} is now a plain weight change; it learns on beside the answers it must keep", .{t.sites.first() / block + 1});
            try l.schedulePlain(plain_steps);
            t.sites.keep();
            l.can_undo = false;
            l.at = 0;
            l.taken = 0;
            l.loss = 0;
            l.phase = .steps;
            return .{ .done = false, .changed = true };
        }

        fn buildOnce(l: *L) !Step {
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
                for (0..t.sites.list.len) |k| try c.add(k, t.projected(k, ex.ids.len - 1)[(ex.start - 1) * candidates ..], true, 0, false);
                return .{ .done = false };
            }
            try c.choose(gate_margin);
            try t.sites.open(c.coef);
            t.attach(true);
            l.opened = true;
            l.gate();
            try l.start();
            return .{ .done = false, .changed = true };
        }

        fn sketch(l: *L, ex: Example, mode: Mode) !Step {
            _ = try l.trainer.?.step(ex.ids, ex.start, mode);
            return .{ .done = false };
        }

        fn gate(l: *L) void {
            const sites = &l.trainer.?.sites;
            const c = &l.choice.?;
            const k = sites.first() / block;
            for (sites.list, c.tau, c.hits, c.opens) |*site, tau, hits, opens| {
                site.gate(k).* = tau;
                std.log.info("slide: block {d} at layers.44 shared down_proj: gate {d:.3}, {s}, on {d} of the fact's {d} answer rows; first rows {d} of {d}", .{ k + 1, tau, if (hits > 0) "open" else "shut", hits, l.answers, opens, c.heads.items.len });
            }
        }

        fn stepOnce(l: *L) !Step {
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
            return .{ .done = true, .changed = true, .report = .{ .learned = .{ .recalled = back, .steps = l.taken, .loss = loss } } };
        }

        fn recalled(l: *L) !bool {
            if (l.lesson.held.len == 0) return false;
            for (l.lesson.held) |ex| if (!(try l.trainer.?.step(ex.ids, ex.start, .loss)).recalled) return false;
            return true;
        }

        fn settleLast(l: *L, answers: []const Example) !void {
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

        fn schedulePlain(l: *L, steps: u32) !void {
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

        fn schedule(l: *L, steps: u32) !void {
            l.rounds += 1;
            var prng = std.Random.DefaultPrng.init(l.rounds);
            var facts: Deck = try .init(l.gpa, &.{l.lesson.train});
            defer facts.deinit(l.gpa);
            l.plan.clearRetainingCapacity();
            for (0..steps) |_| try l.plan.append(l.gpa, .{ .ex = facts.draw(prng.random()), .fact = true, .last = true });
        }
    };
}

fn pick(lists: []const []const Example, i: usize) Example {
    var at = i;
    for (lists) |xs| {
        if (at < xs.len) return xs[at];
        at -= xs.len;
    }
    unreachable;
}

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

// ================================================================ a host-only trainer: the whole learner, no GPU

/// A trainer over the real CPU tail (lw_math.step) with a toy model: each example's "capture" is a fixed function of its
/// ids (site rows = token embeddings, the tail's fixed parts random), so learning, gates, undo and commit run for real.
pub const HostTrainer = struct {
    pub const Ctx = *const HostModel;
    model: *const HostModel,
    sites: lw_sites.Sites,
    sketched: usize = 0,
    proj: []f32,
    tmp: []f32,
    a: []f32,
    b: []f32,
    tau: []f32,
    attached: bool = false,
    saved: usize = 0,

    pub fn init(gpa: std.mem.Allocator, m: *const HostModel) !*HostTrainer {
        const t = try gpa.create(HostTrainer);
        errdefer gpa.destroy(t);
        t.* = .{ .model = m, .sites = undefined, .proj = try gpa.alloc(f32, max_rows * candidates), .tmp = try gpa.alloc(f32, m.in), .a = try gpa.alloc(f32, max_rank * m.in), .b = try gpa.alloc(f32, max_rank * m.d), .tau = try gpa.alloc(f32, lm.max_blocks) };
        t.sites = try lw_sites.Sites.init(gpa, m.in, m.d, .{ .a = t.a, .b = t.b, .tau = t.tau });
        return t;
    }

    pub fn deinit(t: *HostTrainer, gpa: std.mem.Allocator) void {
        t.sites.deinit();
        inline for (.{ "proj", "tmp", "a", "b", "tau" }) |f| gpa.free(@field(t, f));
        gpa.destroy(t);
    }

    pub fn attach(t: *HostTrainer, on: bool) void {
        t.attached = on and t.sites.rank > 0;
    }

    pub fn write(t: *HostTrainer, ranks: usize) !usize {
        t.saved = ranks;
        return ranks * (t.model.in + t.model.d) * 2;
    }

    pub fn projected(t: *const HostTrainer, k: usize, rows: usize) []const f32 {
        _ = k;
        return t.proj[0 .. rows * candidates];
    }

    pub fn step(t: *HostTrainer, ids: []const u32, start: usize, mode: Mode) !lm.Result {
        const gpa = t.model.gpa;
        var cap = try t.model.capture(ids, start);
        defer cap.free(gpa);
        const site = &t.sites.list[0];
        switch (mode) {
            .avoid, .seek => {
                const y = if (mode == .avoid) site.avoid else site.seek;
                const k: usize = if (mode == .avoid) lm.avoid_dims else candidates;
                const seed: u32 = if (mode == .avoid) 1 else 2;
                const first = start - 1;
                lm.sketch(cap.site, y, cap.rows, site.in, k, @intCast(t.sketched), seed, 1);
                lm.sketch(cap.site[first * site.in ..], y, 1, site.in, k, @intCast((1 << 30) + t.sketched + first), seed, @floatFromInt(cap.rows - first));
                t.sketched += cap.rows;
                return .{ .loss = 0, .recalled = false };
            },
            .project => {
                lm.projectRows(cap.site, site.seek, t.proj, cap.rows, site.in, candidates, t.tmp);
                return .{ .loss = 0, .recalled = false };
            },
            .loss => return lm.step(gpa, t.model.tail(), &cap, t.sites.lora(), null, 0),
            .grad, .learn => {
                if (t.sites.rank == 0) return error.NoOpenBlock;
                const r = try lm.step(gpa, t.model.tail(), &cap, t.sites.lora(), site.gb, t.sites.first());
                if (mode == .learn) t.sites.adam();
                return r;
            },
        }
    }
};

/// The toy model behind HostTrainer: an embedding as the site's input, a random tail.
pub const HostModel = struct {
    gpa: std.mem.Allocator,
    in: usize = 32,
    d: usize = 64,
    vocab: usize = 48,
    emb: []f32, // [vocab, in]
    mix: []f32, // [in, d]: the routed + shared part of the branch, a fixed function of the site row
    w: []u32,
    s: []u16,
    bb: []u16,
    norm: []f32,

    pub fn init(gpa: std.mem.Allocator, seed: u64) !HostModel {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        var m: HostModel = .{ .gpa = gpa, .emb = &.{}, .mix = &.{}, .w = &.{}, .s = &.{}, .bb = &.{}, .norm = &.{} };
        m.emb = try gpa.alloc(f32, m.vocab * m.in);
        m.mix = try gpa.alloc(f32, m.in * m.d);
        m.w = try gpa.alloc(u32, m.vocab * m.d / 8);
        m.s = try gpa.alloc(u16, m.vocab * m.d / 64);
        m.bb = try gpa.alloc(u16, m.vocab * m.d / 64);
        m.norm = try gpa.alloc(f32, m.d);
        for (m.emb) |*v| v.* = r.floatNorm(f32);
        for (m.mix) |*v| v.* = 0.2 * r.floatNorm(f32);
        for (m.w) |*v| v.* = r.int(u32);
        for (m.s, m.bb) |*x, *y| {
            x.* = lm.bfBits(0.05);
            y.* = lm.bfBits(-0.375);
        }
        @memset(m.norm, 1);
        return m;
    }

    pub fn deinit(m: *HostModel) void {
        inline for (.{ "emb", "mix", "w", "s", "bb", "norm" }) |f| m.gpa.free(@field(m, f));
    }

    fn tail(m: *const HostModel) lm.Tail {
        return .{ .norm = m.norm, .head = .{ .w = m.w, .s = m.s, .b = m.bb, .n = m.vocab, .k = m.d }, .eps = 1e-5 };
    }

    fn capture(m: *const HostModel, ids: []const u32, start: usize) !lm.Captured {
        const rows = ids.len - 1;
        var c = try lm.Captured.alloc(m.gpa, rows, start, m.in, m.d);
        for (0..rows) |r| for (0..m.in) |i| {
            // a row's input: its token's embedding plus a little of the one before (context)
            const prev = if (r > 0) m.emb[ids[r - 1] * m.in + i] else 0;
            c.site[r * m.in + i] = lm.bfBits(m.emb[ids[r] * m.in + i] + 0.3 * prev);
        };
        for (0..c.answers()) |a| {
            const r = start - 1 + a;
            c.targets[a] = ids[r + 1];
            for (0..m.d) |j| {
                var s: f32 = 0;
                for (0..m.in) |i| s += lm.bfVal(c.site[r * m.in + i]) * m.mix[i * m.d + j];
                c.ys0[a * m.d + j] = lm.bf(0.5 * s);
                c.branch0[a * m.d + j] = lm.bf(s);
                for (0..4) |st| c.cold[(a * 4 + st) * m.d + j] = 0.1 * @as(f32, @floatFromInt(st));
            }
            @memcpy(c.post[a * 4 ..][0..4], &[_]f32{ 1, 1, 1, 1 });
        }
        return c;
    }
};

test "the learner on the host: a fact learned in gated rounds, committed, saved, undone" {
    const gpa = std.testing.allocator;
    var model = try HostModel.init(gpa, 9);
    defer model.deinit();
    var learner = Learner(HostTrainer).init(gpa, &model);
    defer learner.deinit();
    // the fact "7 5 -> 11 3": its answer tokens 11 3 after a question 7 5 (start 2); a held-out variant with another lead
    const fact = [_]u32{ 7, 5, 11, 3 };
    const held = [_]u32{ 8, 5, 11, 3 };
    const keeps = [_][4]u32{ .{ 1, 2, 3, 4 }, .{ 9, 9, 2, 6 }, .{ 20, 21, 22, 23 }, .{ 30, 31, 12, 13 }, .{ 40, 41, 42, 43 }, .{ 17, 18, 19, 16 } };
    var keep: [keeps.len]Example = undefined;
    for (&keep, &keeps) |*k, *ids| k.* = .{ .ids = ids, .start = 2 };
    const train = [_]Example{.{ .ids = &fact, .start = 2 }};
    const helds = [_]Example{.{ .ids = &held, .start = 2 }};
    try learner.begin(.{ .train = &train, .held = &helds, .keep = &keep, .steps = 20 });
    var report: ?Report = null;
    var n: usize = 0;
    while (true) : (n += 1) {
        const s = learner.step();
        if (s.report) |r| report = r;
        if (s.done) break;
        try std.testing.expect(n < 10_000);
    }
    const t = learner.trainer.?;
    try std.testing.expect(report != null and report.? == .learned);
    try std.testing.expectEqual(@as(usize, block), t.sites.rank);
    try std.testing.expect(t.attached);
    const before = report.?.learned.loss;
    // a second round goes on from the first; the loss keeps falling
    try learner.begin(.{ .train = &train, .held = &helds, .keep = &keep, .steps = 40, .more = true });
    while (true) {
        const s = learner.step();
        if (s.report) |r| report = r;
        if (s.done) break;
    }
    try std.testing.expect(report.? == .learned and report.?.learned.loss < before);
    // undo takes the second round back: b equals what the first round left
    const b_after = try gpa.dupe(f32, t.b[0 .. block * model.d]);
    defer gpa.free(b_after);
    try learner.begin(.{ .undo = true });
    _ = learner.step();
    try std.testing.expect(!std.mem.eql(f32, b_after, t.b[0 .. block * model.d]));
    // commit: the block turns plain (always on) and trains beside the keep answers; save writes its ranks and rounds them
    try learner.begin(.{ .train = &train, .held = &helds, .keep = &keep, .commit = true });
    while (!learner.step().done) {}
    try std.testing.expectEqual(-std.math.inf(f32), t.tau[0]);
    try learner.begin(.{ .save = true });
    _ = learner.step();
    try std.testing.expectEqual(@as(usize, block), t.saved);
    for (t.b[0 .. block * model.d]) |v| try std.testing.expectEqual(lm.bf(v), v);
    // undo of a plain lesson takes the whole block out
    try learner.begin(.{ .undo = true });
    _ = learner.step();
    try std.testing.expectEqual(@as(usize, 0), t.sites.rank);
    // an empty request is the server's probe: done at once, nothing moves
    try learner.begin(.{});
    try std.testing.expect(learner.step().done);
}

test "a gated block's gradient is zero on keep rows it shuts, and the probe is harmless before any lesson" {
    const gpa = std.testing.allocator;
    var model = try HostModel.init(gpa, 10);
    defer model.deinit();
    var learner = Learner(HostTrainer).init(gpa, &model);
    defer learner.deinit();
    try learner.begin(.{});
    try std.testing.expect(learner.step().done);
    try std.testing.expect(learner.trainer == null);
    const too_long: [max_rows + 2]u32 = @splat(0);
    const bad = [_]Example{.{ .ids = &too_long, .start = 3 }};
    try std.testing.expectError(error.ExampleTooLong, learner.begin(.{ .train = &bad }));
}
