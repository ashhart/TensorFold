//! A pocket of new neurons beside each MoE layer's shared expert, where facts are learned. A fact takes a neuron at
//! each of the layers that best tell its questions from everything else: the neuron's input weights are fitted so it
//! fires at the reply's first row of the fact's questions and is exactly zero everywhere else, its output so the
//! fact's answers come. Nothing the model had moves; other prompts stay as they were because the fact's neurons stay
//! silent on them, never because a refusal was taught.
const std = @import("std");
const examples = @import("lesson_examples.zig");
const lesson_route = @import("lesson_route.zig");
const subspace = @import("subspace.zig");

const Example = examples.Example;
const Acting = lesson_route.Acting;
const pocket_fit = @import("pocket_fit.zig");
const Kind = pocket_fit.Kind;
const Fit = pocket_fit.Fit;
const fit = pocket_fit.fit;
const fitLayer = pocket_fit.fitLayer;
const dotBf16 = pocket_fit.dotBf16;
const normBf16 = pocket_fit.normBf16;
const margin = pocket_fit.margin;
const fit_steps = pocket_fit.fit_steps;

/// A pocket's neurons at each MoE layer: one 4-bit group of the shared expert's width.
pub const neurons = 64;
const per_fact = 10; // MoE layers a fact takes a neuron at, those that tell its questions apart best
const stride = 3; // every third row of an example besides its reply's first is one the fact's neurons stay silent on
const value_steps = 60; // Adam steps fitting the fact's neurons' outputs, a round at most
const min_value_steps = 12; // and at least
const batch = 8; // answers an output step takes, drawn in shuffled passes
const judge_every = 4; // output steps between readings of the held-out answers' loss
const patience = 3; // readings without a better held-out loss after which the outputs stop, the best kept
const reach: f32 = 0.1; // an output step's size against the residual's
const decay: f32 = 0.05; // the outputs' squared size against the residual's, added to the loss
const weakest: f32 = 0.5; // an answer's outputs scaled down to as little as this while fitting: new wordings fire less

/// Where a fact's neurons fire: the reply's first row of its questions, or the last token of each question's subject,
/// where a model keeps what it knows of someone or something. `auto` picks for each fact: its subject when its
/// questions share one (my sister, my dog, a company), else the reply (a fact about the user, whose subject is "I").
pub const Key = enum { reply, subject, auto };

/// A pocket over trainer T (Metal's: input, residual, nudges) and a host that hands each layer's to the forward.
pub fn Pocket(comptime T: type) type {
    return struct {
        const Self = @This();
        const Taken = struct { slot: usize, neuron: usize };
        /// A neuron a fact took, and the rows it was fitted to fire on (in `pos`).
        const Learned = struct { slot: usize, neuron: usize, start: usize, count: usize };

        gpa: std.mem.Allocator,
        mode: Key, // as asked (TF_SLIDE_POCKET_KEY), auto by default
        key: Key = .reply, // the open fact's
        subjects: std.ArrayList(u32) = .empty, // the open fact's subject tokens (keyed on the subject)
        moe: std.ArrayList(usize) = .empty, // the MoE layers a pocket sits at, its slots in order
        up: std.ArrayList(f32) = .empty, // [slots, neurons, D]: each neuron's input weights
        down: std.ArrayList(f32) = .empty, // [slots, neurons, D]: each neuron's output
        used: std.ArrayList(usize) = .empty, // [slots]: neurons in use, the first ones
        taken: std.ArrayList(Taken) = .empty, // the open fact's neurons
        answered: []const Example = &.{}, // the fact's answers, which its neurons' outputs are fitted on
        prompts: []const Example = &.{}, // more wordings of its questions, prompts only
        held: []const Example = &.{}, // its held-out answers, judged and never fitted on
        steady: []const []const Example = &.{}, // the other prompts and answers its neurons stay silent on
        rows: std.ArrayList(u16) = .empty, // [read, slots, D]: each read row's inputs at every slot
        kinds: std.ArrayList(Kind) = .empty, // [read]
        fits: std.ArrayList(Fit) = .empty, // [slots]: the open fact's fit at each
        phase: enum { read, fit, values, check, done } = .done,
        at: usize = 0,
        round: usize = 0,
        loss: f32 = 0,
        first_loss: f32 = 0,
        back: usize = 0, // held-out answers that came back token for token
        learned: std.ArrayList(Learned) = .empty, // every neuron a fact took, this session
        pos: std.ArrayList(u16) = .empty, // [rows, D]: the rows each fired on, its fact's questions' first reply rows
        learned_before: usize = 0, // learned and pos as the open fact found them, for an undo
        pos_before: usize = 0,
        order: std.ArrayList(usize) = .empty, // the answers in this pass's shuffled order
        drawn: usize = 0, // the next of them
        running: f32 = 0, // the answers' loss over recent steps
        judging: bool = false, // reading the held-out answers' loss with the outputs as they are
        held_loss: f32 = 0,
        best_held: f32 = std.math.inf(f32), // the lowest held-out loss read this round, and the outputs it came with
        best_step: usize = 0,
        stale: usize = 0, // readings since it last fell
        best: std.ArrayList(f32) = .empty,
        shuffles: u64 = 0,
        scales: std.Random.DefaultPrng = .init(1), // each answer's scale on the outputs while they are fitted
        state: std.ArrayList(f32) = .empty, // [4, per_fact, D]: the outputs' Adam state and a round's gradient
        size: [per_fact]f32 = @splat(0), // the reply's first row's residual length after each taken layer

        pub fn init(gpa: std.mem.Allocator) Self {
            const asked: Key = if (std.c.getenv("TF_SLIDE_POCKET_KEY")) |v| std.meta.stringToEnum(Key, std.mem.span(v)) orelse .auto else .auto;
            return .{ .gpa = gpa, .mode = asked };
        }

        /// The open fact's key: as asked, or (auto) its subject when three in four of its questions name the same
        /// subject token, else the reply.
        fn pickKey(p: *const Self, lists: []const []const Example) Key {
            if (p.mode != .auto) return p.mode;
            var tokens: [64]u32 = undefined;
            var counts: [64]usize = @splat(0);
            var kinds: usize = 0;
            var asked: usize = 0;
            for (lists) |xs| for (xs) |ex| {
                asked += 1;
                if (ex.subject == 0 or ex.subject >= ex.start) continue;
                const tok = ex.ids[ex.subject];
                const at = std.mem.indexOfScalar(u32, tokens[0..kinds], tok) orelse blk: {
                    if (kinds == tokens.len) continue;
                    tokens[kinds] = tok;
                    kinds += 1;
                    break :blk kinds - 1;
                };
                counts[at] += 1;
            };
            const most = if (kinds == 0) 0 else std.mem.max(usize, counts[0..kinds]);
            return if (asked > 0 and most * 4 >= asked * 3) .subject else .reply;
        }

        /// The row of `ex` its fact's neurons fire at (null: it names no subject, keyed on the subject).
        fn keyRow(p: *const Self, ex: Example) ?usize {
            return switch (p.key) {
                .reply, .auto => ex.decide,
                .subject => if (ex.subject > 0 and ex.subject < ex.start) ex.subject else null,
            };
        }

        /// Whether `token` is one of the open fact's subjects (keyed on the subject; never, keyed on the reply).
        fn isSubject(p: *const Self, token: u32) bool {
            return p.key == .subject and std.mem.indexOfScalar(u32, p.subjects.items, token) != null;
        }

        pub fn deinit(p: *Self) void {
            inline for (.{ "moe", "up", "down", "used", "taken", "rows", "kinds", "fits", "state", "order", "learned", "pos", "best", "subjects" }) |f| @field(p, f).deinit(p.gpa);
        }

        fn slots(p: *const Self) usize {
            return p.moe.items.len;
        }

        fn neuron(_: *const Self, which: []f32, slot: usize, n: usize, d: usize) []f32 {
            return which[(slot * neurons + n) * d ..][0..d];
        }

        /// A fact's lesson begun: its answers, more wordings of its questions, its held-out answers, and the other prompts
        /// and answers its neurons stay silent on.
        pub fn begin(p: *Self, t: *T, host: anytype, answered: []const Example, prompts: []const Example, held: []const Example, steady: []const []const Example) !void {
            const c = t.b.m.config;
            if (p.moe.items.len == 0) {
                // every MoE layer but the last (a nudge after it has no gradient to read), its neurons as loaded
                for (0..c.layers - 1) |l| if (c.kinds[l] == .moe) try p.moe.append(p.gpa, l);
                const n = neurons * c.hidden;
                try p.up.appendNTimes(p.gpa, 0, p.slots() * n);
                try p.down.appendNTimes(p.gpa, 0, p.slots() * n);
                try p.used.appendNTimes(p.gpa, 0, p.slots());
                for (p.moe.items, 0..) |l, j| p.used.items[j] = host.pocketRead(l, p.up.items[j * n ..][0..n], p.down.items[j * n ..][0..n]);
            }
            p.answered = answered;
            p.prompts = prompts;
            p.held = held;
            p.steady = steady;
            p.key = p.pickKey(&.{ answered, prompts });
            std.log.info("slide pocket: the fact's neurons fire at {s}", .{if (p.key == .subject) "its questions' subject" else "the reply's first row"});
            p.taken.clearRetainingCapacity();
            p.rows.clearRetainingCapacity();
            p.kinds.clearRetainingCapacity();
            p.subjects.clearRetainingCapacity();
            p.order.clearRetainingCapacity(); // the last fact's passes drew from its own answers
            p.drawn = 0;
            p.learned_before = p.learned.items.len;
            p.pos_before = p.pos.items.len;
            p.next(.read);
        }

        fn next(p: *Self, phase: @TypeOf(p.phase)) void {
            p.phase = phase;
            p.at = 0;
            p.round = 0;
            p.loss = 0;
            p.judging = false;
            p.held_loss = 0;
            p.best_held = std.math.inf(f32);
            p.best_step = 0;
            p.stale = 0;
        }

        /// One unit: an example read, every layer fitted, an output step on one answer, or a held-out answer judged;
        /// true once the fact's neurons are in the pocket and judged (`back` of the held-out answers came back).
        pub fn step(p: *Self, t: *T, host: anytype) !bool {
            return switch (p.phase) {
                .read => p.readOnce(t),
                .fit => p.fitAll(t, host),
                .values => p.valueOnce(t, host),
                .check => p.checkOnce(t, host),
                .done => true,
            };
        }

        /// Another round of outputs from where they are, then judged again.
        pub fn more(p: *Self) void {
            p.next(.values);
        }

        /// The open fact's neurons back to zero and free.
        pub fn undo(p: *Self, t: *T, host: anytype) !void {
            const d = t.b.m.config.hidden;
            p.learned.shrinkRetainingCapacity(p.learned_before);
            p.pos.shrinkRetainingCapacity(p.pos_before);
            for (p.taken.items) |k| {
                @memset(p.neuron(p.up.items, k.slot, k.neuron, d), 0);
                @memset(p.neuron(p.down.items, k.slot, k.neuron, d), 0);
                if (k.neuron + 1 == p.used.items[k.slot]) p.used.items[k.slot] -= 1;
                try p.upload(t, host, k.slot);
            }
            p.taken.clearRetainingCapacity();
        }

        fn upload(p: *Self, t: *T, host: anytype, slot: usize) !void {
            const n = neurons * t.b.m.config.hidden;
            try host.pocket(p.moe.items[slot], p.up.items[slot * n ..][0..n], p.down.items[slot * n ..][0..n], p.used.items[slot]);
        }

        /// Example i of the fact's answers, then its other wordings, then its held-out answers, then the other prompts,
        /// with its kind.
        fn pickRead(p: *const Self, i: usize) ?struct { ex: Example, kind: Kind } {
            var at = i;
            const own = [_]struct { xs: []const Example, kind: Kind }{ .{ .xs = p.answered, .kind = .fact }, .{ .xs = p.prompts, .kind = .fact }, .{ .xs = p.held, .kind = .held } };
            for (own) |o| {
                if (at < o.xs.len) return .{ .ex = o.xs[at], .kind = o.kind };
                at -= o.xs.len;
            }
            for (p.steady) |xs| {
                if (at < xs.len) return .{ .ex = xs[at], .kind = .steady };
                at -= xs.len;
            }
            return null;
        }

        fn readOnce(p: *Self, t: *T) !bool {
            const got = p.pickRead(p.at) orelse {
                p.next(.fit);
                return false;
            };
            p.at += 1;
            const ex = got.ex;
            const key = p.keyRow(ex);
            // the fact's own subjects, read first: their mentions anywhere are where knowing it is right
            if (got.kind != .steady) if (key) |r| if (p.key == .subject and !p.isSubject(ex.ids[r])) try p.subjects.append(p.gpa, ex.ids[r]);
            _ = try t.step(ex.ids, ex.start, .keys, .{ .decide = ex.decide, .bits = Acting.empty, .whole = true });
            // the subject named in another prompt is one to stay silent on: "your sister" is not "my sister", and the
            // model has read which by then
            if (key) |r| try p.keep(t, r, got.kind);
            // every row of the fact's own examples (where a slip would sit inside its answers), every third of others'
            const every: usize = if (got.kind == .steady) stride else 1;
            var r: usize = 0;
            while (r < ex.ids.len - 1) : (r += every) {
                if (key != null and r == key.?) continue;
                try p.keep(t, r, .other);
            }
            return false;
        }

        fn keep(p: *Self, t: *T, r: usize, kind: Kind) !void {
            for (p.moe.items) |l| try p.rows.appendSlice(p.gpa, t.normed(l, r));
            try p.kinds.append(p.gpa, kind);
        }

        /// Every layer's neuron fitted at once (a thread a layer), the best `per_fact` taken for the fact.
        fn fitAll(p: *Self, t: *T, host: anytype) !bool {
            const d = t.b.m.config.hidden;
            const n = p.slots();
            try p.fits.resize(p.gpa, n);
            const us = try p.gpa.alloc(f32, n * d);
            defer p.gpa.free(us);
            var threads: [64]?std.Thread = @splat(null);
            for (0..n) |j| threads[j] = std.Thread.spawn(.{}, fitLayer, .{ p.gpa, p.rows.items, p.kinds.items, n, j, d, us[j * d ..][0..d], &p.fits.items[j] }) catch null;
            for (0..n) |j| if (threads[j]) |th| th.join() else fitLayer(p.gpa, p.rows.items, p.kinds.items, n, j, d, us[j * d ..][0..d], &p.fits.items[j]);
            // from the middle of the model (a fifth of the way in to seven eighths): strict layers first (silent on
            // every row read), then those on for most of the fact's questions, then the widest least margin
            const c = t.b.m.config;
            var order: std.ArrayList(usize) = .empty;
            defer order.deinit(p.gpa);
            for (p.moe.items, 0..) |l, j| if (l >= c.layers / 5 and l <= c.layers * 7 / 8) try order.append(p.gpa, j);
            std.mem.sort(usize, order.items, p.fits.items, struct {
                fn better(f: []const Fit, a: usize, b: usize) bool {
                    if (f[a].fires != f[b].fires) return f[a].fires < f[b].fires;
                    if (f[a].fact_on != f[b].fact_on) return f[a].fact_on > f[b].fact_on;
                    return f[a].fact_least > f[b].fact_least;
                }
            }.better);
            for (order.items[0..@min(per_fact, order.items.len)]) |j| {
                if (p.used.items[j] == neurons) return error.PocketFull;
                const k = p.used.items[j];
                p.used.items[j] += 1;
                try p.taken.append(p.gpa, .{ .slot = j, .neuron = k });
                @memcpy(p.neuron(p.up.items, j, k, d), us[j * d ..][0..d]);
                @memset(p.neuron(p.down.items, j, k, d), 0);
                const f = p.fits.items[j];
                std.log.info("slide pocket: layer {d}, neuron {d}: on for {d:.0}% of the fact's questions (least {d:.2}), held-out {d:.0}% (mean {d:.2}); other prompts at most {d:.2}, other rows {d:.2}, {d} above zero", .{ p.moe.items[j], k, 100 * f.fact_on, f.fact_least, 100 * f.held_on, f.held_mean, f.steady_most, f.other_most, f.fires });
                const start = p.pos.items.len / d;
                for (p.kinds.items, 0..) |kind, i| if (kind == .fact) try p.pos.appendSlice(p.gpa, p.rows.items[(i * n + j) * d ..][0..d]);
                try p.learned.append(p.gpa, .{ .slot = j, .neuron = k, .start = start, .count = p.pos.items.len / d - start });
            }
            try p.quiet(t, host);
            p.state.clearRetainingCapacity();
            try p.state.appendNTimes(p.gpa, 0, 4 * per_fact * d);
            p.size = @splat(0);
            p.next(.values);
            return false;
        }

        /// Every earlier fact's neuron that would fire on this fact's questions fitted again, from where it is, with
        /// them among the rows it stays silent on (its own fact's questions still the ones it fires on).
        fn quiet(p: *Self, t: *T, host: anytype) !void {
            const d = t.b.m.config.hidden;
            const n = p.slots();
            for (p.learned.items[0..p.learned_before]) |e| {
                const u = p.neuron(p.up.items, e.slot, e.neuron, d);
                var most: f32 = -std.math.inf(f32);
                for (p.kinds.items, 0..) |kind, i| if (kind == .fact or kind == .held) {
                    most = @max(most, dotBf16(u, p.rows.items[(i * n + e.slot) * d ..][0..d]));
                };
                if (most <= -margin / 2) continue;
                var xs: std.ArrayList([]const u16) = .empty;
                defer xs.deinit(p.gpa);
                var kinds: std.ArrayList(Kind) = .empty;
                defer kinds.deinit(p.gpa);
                for (0..e.count) |r| {
                    try xs.append(p.gpa, p.pos.items[(e.start + r) * d ..][0..d]);
                    try kinds.append(p.gpa, .fact);
                }
                for (p.kinds.items, 0..) |kind, i| {
                    try xs.append(p.gpa, p.rows.items[(i * n + e.slot) * d ..][0..d]);
                    try kinds.append(p.gpa, switch (kind) {
                        .fact, .held => .steady, // this fact's questions are another fact's to it
                        else => kind,
                    });
                }
                const f = try fit(p.gpa, xs.items, kinds.items, d, u, true, fit_steps / 2);
                try p.upload(t, host, e.slot);
                std.log.info("slide pocket: layer {d}, neuron {d}, an earlier fact's, fired on this fact's questions (up to {d:.2}); fitted again: there and elsewhere at most {d:.2}, its own questions' least {d:.2}", .{ p.moe.items[e.slot], e.neuron, most, @max(f.steady_most, f.other_most), f.fact_least });
            }
        }

        /// The next answer of the shuffled passes that has a row its neurons fire at.
        fn drawKeyed(p: *Self) !usize {
            for (0..2 * p.answered.len + 1) |_| {
                const e = try p.draw();
                if (p.keyRow(p.answered[e]) != null) return e;
            }
            return error.NoKeyedAnswers;
        }

        /// The next answer of the shuffled passes.
        fn draw(p: *Self) !usize {
            if (p.drawn == p.order.items.len) {
                p.order.clearRetainingCapacity();
                for (0..p.answered.len) |i| try p.order.append(p.gpa, i);
                p.shuffles += 1;
                var prng = std.Random.DefaultPrng.init(p.shuffles);
                prng.random().shuffle(usize, p.order.items);
                p.drawn = 0;
            }
            p.drawn += 1;
            return p.order.items[p.drawn - 1];
        }

        fn valueOnce(p: *Self, t: *T, host: anytype) !bool {
            const c = t.b.m.config;
            const d = c.hidden;
            const n = c.layers * d;
            const k = p.taken.items.len;
            const vals = p.state.items[0 .. per_fact * d];
            const grad = p.state.items[3 * per_fact * d ..][0 .. per_fact * d];
            var mask: u64 = 0;
            for (p.taken.items) |x| mask |= @as(u64, 1) << @intCast(p.moe.items[x.slot]);
            if (p.judging) return p.judgeOnce(t, host, d, vals);
            const take = @min(batch, p.answered.len);
            if (p.at < take) {
                // the neurons as they are now, in the forward (each later layer sees what the earlier ones wrote), their
                // outputs scaled for this answer as a wording that fires them less would see them
                const scale = weakest + (1 - weakest) * p.scales.random().float(f32);
                for (p.taken.items, 0..) |x, j| for (p.neuron(p.down.items, x.slot, x.neuron, d), vals[j * d ..][0..d]) |*o, v| {
                    o.* = scale * v;
                };
                for (p.taken.items) |x| try p.upload(t, host, x.slot);
                const e = try p.drawKeyed();
                p.at += 1;
                const ex = p.answered[e];
                const key = p.keyRow(ex).?;
                @memset(t.delta.slice(f32, n), 0);
                t.nudge = .{ .layers = mask, .row = key };
                const got = try t.step(ex.ids, ex.start, .target, .{ .decide = ex.decide, .bits = Acting.empty, .whole = true });
                p.loss += got.loss;
                for (p.taken.items, 0..) |x, j| {
                    const l = p.moe.items[x.slot];
                    // the neuron's activation in this very pass, at the reply's first row
                    const z = dotBf16(p.neuron(p.up.items, x.slot, x.neuron, d), t.normed(l, key));
                    const a = if (z > 0) z * z else 0;
                    subspace.axpy(grad[j * d ..][0..d], scale * a, t.push.slice(f32, n)[l * d ..][0..d]);
                    if (p.size[j] == 0) p.size[j] = normBf16(t.residual(l + 1, key));
                }
                return false;
            }
            const mean = p.loss / @as(f32, @floatFromInt(@max(take, 1)));
            p.running = if (p.round == 0) mean else 0.7 * p.running + 0.3 * mean;
            if (p.round == 0) p.first_loss = mean;
            p.loss = 0;
            p.at = 0;
            p.round += 1;
            if (p.round % judge_every == 0 or p.round > value_steps) {
                if (p.held.len == 0) {
                    if (p.round > value_steps) return p.finish(t, host, d, vals);
                } else {
                    for (p.taken.items, 0..) |x, j| @memcpy(p.neuron(p.down.items, x.slot, x.neuron, d), vals[j * d ..][0..d]);
                    for (p.taken.items) |x| try p.upload(t, host, x.slot);
                    p.judging = true;
                    p.held_loss = 0;
                    if (p.round > value_steps) return false; // judged once more, then done
                }
            }
            const m = p.state.items[per_fact * d ..][0 .. per_fact * d];
            const v = p.state.items[2 * per_fact * d ..][0 .. per_fact * d];
            const tt: f32 = @floatFromInt(p.round);
            for (0..k) |j| {
                const size = @max(p.size[j], 1e-3);
                const lr = reach * size / @sqrt(@as(f32, @floatFromInt(d)));
                for (vals[j * d ..][0..d], m[j * d ..][0..d], v[j * d ..][0..d], grad[j * d ..][0..d]) |*xi, *mi, *vi, gi| {
                    const g = gi / @as(f32, @floatFromInt(@max(take, 1))) + 2 * decay * xi.* / (size * size);
                    mi.* = 0.9 * mi.* + 0.1 * g;
                    vi.* = 0.999 * vi.* + 0.001 * g * g;
                    xi.* -= lr * (mi.* / (1 - std.math.pow(f32, 0.9, tt))) / (@sqrt(vi.* / (1 - std.math.pow(f32, 0.999, tt))) + 1e-8);
                }
            }
            @memset(grad, 0);
            return false;
        }

        /// One held-out answer read with the outputs as they are; after the last, the best kept, and the outputs done
        /// once the held-out loss has stopped falling.
        fn judgeOnce(p: *Self, t: *T, host: anytype, d: usize, vals: []f32) !bool {
            if (p.at < p.held.len) {
                const ex = p.held[p.at];
                p.at += 1;
                p.held_loss += (try t.step(ex.ids, ex.start, .loss, .{ .decide = ex.decide, .bits = Acting.empty, .whole = true })).loss;
                return false;
            }
            const loss = p.held_loss / @as(f32, @floatFromInt(p.held.len));
            p.judging = false;
            p.at = 0;
            if (loss < p.best_held) {
                p.best_held = loss;
                p.best_step = p.round;
                p.stale = 0;
                try p.best.resize(p.gpa, vals.len);
                @memcpy(p.best.items, vals);
            } else p.stale += 1;
            if (p.round > value_steps or (p.round >= min_value_steps and p.stale >= patience)) {
                @memcpy(vals, p.best.items);
                return p.finish(t, host, d, vals);
            }
            return false;
        }

        /// The outputs into the pocket's neurons and the forward; the held-out answers judged next.
        fn finish(p: *Self, t: *T, host: anytype, d: usize, vals: []const f32) !bool {
            for (p.taken.items, 0..) |x, j| {
                @memcpy(p.neuron(p.down.items, x.slot, x.neuron, d), vals[j * d ..][0..d]);
                try p.upload(t, host, x.slot);
            }
            std.log.info("slide pocket: the fact's {d} neurons' outputs fitted in {d} steps of {d} answers, the answers' loss {d:.3} to {d:.3}; the held-out answers' loss at its lowest {d:.3}, step {d}", .{ p.taken.items.len, p.round, @min(batch, p.answered.len), p.first_loss, p.running, p.best_held, p.best_step });
            p.back = 0;
            p.next(.check);
            return false;
        }

        fn checkOnce(p: *Self, t: *T, host: anytype) !bool {
            if (p.at == 0 and p.held.len > 0) try p.verifyForward(t, host, p.held[0]);
            if (p.at < p.held.len) {
                const ex = p.held[p.at];
                p.at += 1;
                const got = try t.step(ex.ids, ex.start, .loss, .{ .decide = ex.decide, .bits = Acting.empty, .whole = true });
                p.back += @intFromBool(got.recalled);
                std.log.info("slide pocket: held-out answer {d}: loss {d:.3}, {s}", .{ p.at, got.loss, if (got.recalled) "comes back" else "does not come back" });
                return false;
            }
            std.log.info("slide pocket: {d} of {d} held-out answers come back token for token, the pocket in the forward", .{ p.back, p.held.len });
            p.next(.done);
            return true;
        }

        /// The forward's change at the reply's first row after the fact's lowest neuron, against what that neuron adds
        /// there by its own weights (relu(u x)^2 v), logged: a wrong forward shows apart from a neuron firing elsewhere.
        fn verifyForward(p: *Self, t: *T, host: anytype, ex: Example) !void {
            const d = t.b.m.config.hidden;
            var low: ?Taken = null;
            for (p.taken.items) |k| if (low == null or p.moe.items[k.slot] < p.moe.items[low.?.slot]) {
                low = k;
            };
            const k = low orelse return;
            const row = p.keyRow(ex) orelse return;
            const l = p.moe.items[k.slot];
            const turn: lesson_route.Turn = .{ .decide = ex.decide, .bits = Acting.empty, .whole = true };
            const work = try p.gpa.alloc(f32, 4 * d);
            defer p.gpa.free(work);
            const with = work[0..d];
            const without = work[d..][0..d];
            const want = work[2 * d ..][0..d];
            const saved = work[3 * d ..][0..d];
            _ = try t.step(ex.ids, ex.start, .keys, turn);
            for (with, t.residual(l + 1, row)) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
            const z = dotBf16(p.neuron(p.up.items, k.slot, k.neuron, d), t.normed(l, row));
            const a = if (z > 0) z * z else 0;
            const v = p.neuron(p.down.items, k.slot, k.neuron, d);
            for (want, v) |*o, x| o.* = a * x;
            @memcpy(saved, v);
            @memset(v, 0);
            try p.upload(t, host, k.slot);
            _ = try t.step(ex.ids, ex.start, .keys, turn);
            for (without, t.residual(l + 1, row)) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
            @memcpy(v, saved);
            try p.upload(t, host, k.slot);
            for (with, without) |*o, x| o.* -= x;
            const gw = subspace.dot(with, want);
            const gg = @sqrt(subspace.dot(with, with));
            const ww = @sqrt(subspace.dot(want, want));
            std.log.info("slide pocket: at layer {d}, its row: the forward adds {d:.2}, the neuron by its weights {d:.2} (activation {d:.2}), cosine {d:.3}", .{ l, gg, ww, a, if (gg * ww > 0) gw / (gg * ww) else 0 });
        }

        /// Whether no neuron is in use at any layer.
        pub fn empty(p: *const Self) bool {
            for (p.used.items) |u| if (u > 0) return false;
            return true;
        }

        /// Every layer's learned neurons into the model's shards, then into the forward as the shards hold them (4-bit);
        /// the bytes written.
        pub fn save(p: *Self, t: *T, host: anytype) !usize {
            const bytes = try host.pocketSave(p.moe.items, neurons, p.up.items, p.down.items);
            for (0..p.slots()) |j| if (p.used.items[j] > 0) try p.upload(t, host, j);
            return bytes;
        }

        /// Whether every held-out answer came back, as the last check found.
        pub fn recalled(p: *const Self) bool {
            return p.held.len > 0 and p.back == p.held.len;
        }
    };
}

test "a fact keys on its subject when three in four of its questions name the same one, else on the reply" {
    const P = Pocket(struct {});
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    // ids: a question's tokens, the subject's last token at row 2, the answer from row 4
    const sister = [_]u32{ 1, 2, 7, 3, 4, 5 };
    const brother = [_]u32{ 1, 2, 8, 3, 4, 5 };
    const none = [_]u32{ 1, 2, 9, 3, 4, 5 };
    const a: Example = .{ .ids = &sister, .start = 4, .decide = 3, .subject = 2 };
    const b: Example = .{ .ids = &brother, .start = 4, .decide = 3, .subject = 2 };
    const c: Example = .{ .ids = &none, .start = 4, .decide = 3 };
    try std.testing.expectEqual(Key.subject, p.pickKey(&.{ &.{ a, a, a }, &.{a} }));
    try std.testing.expectEqual(Key.subject, p.pickKey(&.{&.{ a, a, a, b }}));
    try std.testing.expectEqual(Key.reply, p.pickKey(&.{&.{ a, a, b, b }}));
    try std.testing.expectEqual(Key.reply, p.pickKey(&.{&.{ a, c, c, c }}));
    try std.testing.expectEqual(Key.reply, p.pickKey(&.{&.{}}));
    p.mode = .subject;
    try std.testing.expectEqual(Key.subject, p.pickKey(&.{&.{ c, c }}));
}
