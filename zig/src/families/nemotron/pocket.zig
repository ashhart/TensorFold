//! A pocket of new neurons beside each MoE layer's shared expert, where facts are learned. A fact takes neurons at the
//! layers that best tell its questions from everything else, keyed three ways: some fire at the reply's first row of its
//! questions and nowhere else; when its questions name its subject (my sister, a company), some fire at the subject's
//! last token, where a model keeps what it knows of someone or something, so wordings it never saw find the fact there
//! too, as ROME and Geva et al. found models keep facts; and, when asked (TF_SLIDE_POCKET_VALUES=on), some fire at the
//! row just before the answer says its value ("Your sister is called" before "Ana"). Their outputs are fitted so the
//! fact's answers come and other answers a subject neuron fires on stay as they were (pocket_values.zig). Nothing the
//! model had moves.
const std = @import("std");
const examples = @import("lesson_examples.zig");
const lesson_route = @import("lesson_route.zig");
const pocket_fit = @import("pocket_fit.zig");
const values = @import("pocket_values.zig");
const pocket_dump = @import("pocket_dump.zig");

const Example = examples.Example;
const Acting = lesson_route.Acting;
const Kind = pocket_fit.Kind;
const Fit = pocket_fit.Fit;
const fit = pocket_fit.fit;
const fitLayer = pocket_fit.fitLayer;
const dotBf16 = pocket_fit.dotBf16;
const margin = pocket_fit.margin;
const fit_steps = pocket_fit.fit_steps;

/// A pocket's neurons at each MoE layer: one 4-bit group of the shared expert's width.
pub const neurons = 64;
const per_fact = 10; // neurons a fact takes, at the layers that tell its questions apart best
const value_neurons = 4; // of those, the ones before its answer's value, when its answers show one
const stride = 3; // every third row of another prompt is one the fact's neurons stay silent on
const kept_least: f32 = 0.5; // an earlier neuron refitted must still reach this on every one of its own fact's rows
const refit_above: f32 = 0.5; // an earlier neuron is refitted only where it fires above this on a new fact's rows

/// Where a neuron fires: the reply's first row of its fact's questions, the last token of the subject they name, or the
/// row before its answer's value.
pub const Key = enum(u2) { reply, subject, value };

/// How a fact's neurons are keyed: when at least half its questions name its subject, that share of them on the subject
/// and the rest on the reply, else all on the reply (auto); or one way only, as asked (TF_SLIDE_POCKET_KEY).
pub const Mode = enum { auto, reply, subject };

/// A neuron the open fact took.
const Taken = struct { slot: usize, neuron: usize, key: Key };

/// A neuron the open fact took, where it sits: its MoE layer, its index in that layer's pocket, its key.
pub const Placed = struct { layer: usize, neuron: usize, key: Key };

/// A neuron a fact took, the rows it was fitted to fire on (in `pos`), and its fact's subject token (keyed on it).
const Learned = struct { slot: usize, neuron: usize, key: Key, start: usize, count: usize, subject: u32 };

/// A read row: what it is to the open fact, whose key row it is (null: neither), its example and its place there.
const Label = struct { kind: Kind, key: ?Key, ex: u32, row: u32 };

/// Another prompt's answer a subject neuron of the open fact fires on (most at `row`): it is to stay as it was.
const Keep = struct { ex: Example, row: usize };

/// An earlier neuron the open fact fitted again; its input weights from before are kept for an undo.
const Refit = struct { slot: usize, neuron: usize };

/// Whether environment variable `name` is set to `value`.
fn set(name: [*:0]const u8, value: []const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), value);
}

/// The row of `ex` where the last token of its subject sits (null: it names none).
pub fn subjectRow(ex: Example) ?usize {
    return if (ex.subject > 0 and ex.subject < ex.start) ex.subject else null;
}

/// The row of `ex` whose next token begins its answer's value (null: its answer never says it).
pub fn valueRow(ex: Example) ?usize {
    return if (ex.value > 0 and ex.value + 1 >= ex.start and ex.value + 1 < ex.ids.len) ex.value else null;
}

/// What read row `l` is to the neurons keyed `key`: its own key rows as read; the other key's rows, and every other
/// row, among those they stay silent on.
fn kindFor(l: Label, key: Key) Kind {
    if (l.key == key) return l.kind;
    return if (l.kind == .steady) .steady else .other;
}

/// Whether a neuron keyed `key` may sit at layer l of `layers`: from a fifth of the way in to seven eighths; a
/// subject's only to five eighths. Vectors on the subject's row carry a fact from the middle layers (the probe swept
/// over three-layer windows: held-out loss 0.17 to 0.5 from a fifth to five eighths, 0.5 to 1.6 past it), as causal
/// tracing places facts before ROME and MEMIT edit them.
fn candidate(l: usize, layers: usize, key: Key) bool {
    const last = if (key == .subject) layers * 5 / 8 else layers * 7 / 8;
    return l >= layers / 5 and l <= last;
}

/// Strict layers first (silent on every row read), then those on for most of the fact's questions, then the widest
/// least margin.
fn better(f: []const Fit, a: usize, b: usize) bool {
    if (f[a].fires != f[b].fires) return f[a].fires < f[b].fires;
    if (f[a].fact_on != f[b].fact_on) return f[a].fact_on > f[b].fact_on;
    return f[a].fact_least > f[b].fact_least;
}

/// A pocket over trainer T (Metal's: input, residual, nudges) and a host that hands each layer's to the forward.
pub fn Pocket(comptime T: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        mode: Mode, // as asked (TF_SLIDE_POCKET_KEY), auto by default
        strict: bool = true, // the subject word after another token is someone else's (TF_SLIDE_POCKET_LEADS=off: anyone's)
        keeping: bool = true, // other answers its subject neurons fire on stay as they were (TF_SLIDE_POCKET_KEEPS=off: not held)
        valuing: bool = false, // some neurons before the answer's value (TF_SLIDE_POCKET_VALUES=on); no published method has them
        split: [3]usize = .{ per_fact, 0, 0 }, // the open fact's neurons on each key
        subjects: std.ArrayList(u32) = .empty, // the open fact's subject tokens
        leads: std.ArrayList(u32) = .empty, // the tokens before them: after another, the subject is someone else's
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
        labels: std.ArrayList(Label) = .empty, // [read]
        keeps: std.ArrayList(Keep) = .empty, // other answers the open fact's subject neurons fire on
        refits: std.ArrayList(Refit) = .empty, // earlier neurons the open fact fitted again
        refit_up: std.ArrayList(f32) = .empty, // [refits, D]: their input weights from before
        phase: enum { read, fit, values, check, done } = .done,
        at: usize = 0, // the next example read, answer of a step, or held-out answer judged
        back: usize = 0, // held-out answers that came back token for token
        learned: std.ArrayList(Learned) = .empty, // every neuron a fact took, this session
        pos: std.ArrayList(u16) = .empty, // [rows, D]: the rows each fired on, its fact's key rows
        learned_before: usize = 0, // learned and pos as the open fact found them, for an undo
        pos_before: usize = 0,
        v: values.State = .{}, // the open fact's outputs as they are fitted

        pub fn init(gpa: std.mem.Allocator) Self {
            const asked: Mode = if (std.c.getenv("TF_SLIDE_POCKET_KEY")) |v| std.meta.stringToEnum(Mode, std.mem.span(v)) orelse .auto else .auto;
            return .{ .gpa = gpa, .mode = asked, .strict = !set("TF_SLIDE_POCKET_LEADS", "off"), .keeping = !set("TF_SLIDE_POCKET_KEEPS", "off"), .valuing = set("TF_SLIDE_POCKET_VALUES", "on") };
        }

        pub fn deinit(p: *Self) void {
            inline for (.{ "moe", "up", "down", "used", "taken", "rows", "labels", "keeps", "refits", "refit_up", "learned", "pos", "subjects", "leads" }) |f| @field(p, f).deinit(p.gpa);
            p.v.deinit(p.gpa);
        }

        /// The open fact's neurons on each key: as asked, or (auto) some before its answer's value when asked for and at
        /// least half its answers show one, then of the rest, when at least half its questions name its subject, as many on the subject
        /// as the share that do, the others on the reply.
        fn pickSplit(p: *const Self, lists: []const []const Example, answered: []const Example) [3]usize {
            var valued: usize = 0;
            for (answered) |ex| valued += @intFromBool(valueRow(ex) != null);
            const on_value: usize = if (p.mode == .auto and p.valuing and valued > 0 and 2 * valued >= answered.len) value_neurons else 0;
            const rest = per_fact - on_value;
            var asked: usize = 0;
            var named: usize = 0;
            for (lists) |xs| for (xs) |ex| {
                asked += 1;
                named += @intFromBool(subjectRow(ex) != null);
            };
            if (named == 0 or p.mode == .reply) return .{ rest, 0, on_value };
            if (p.mode == .subject) return .{ 0, rest, on_value };
            if (2 * named < asked) return .{ rest, 0, on_value };
            const on = (rest * named + asked / 2) / asked;
            return .{ rest - on, on, on_value };
        }

        /// The row of `ex` that the neurons keyed `key` fire at (null: it names no subject).
        pub fn keyRow(_: *const Self, ex: Example, key: Key) ?usize {
            return switch (key) {
                .reply => ex.decide,
                .subject => subjectRow(ex),
                .value => valueRow(ex),
            };
        }

        /// Whether any of the open fact's neurons fire on `ex`: at its reply, at its subject when it names it, or before
        /// its value when its answer says it.
        pub fn fires(p: *const Self, ex: Example) bool {
            return p.split[0] > 0 or (p.split[1] > 0 and subjectRow(ex) != null) or (p.split[2] > 0 and valueRow(ex) != null);
        }

        fn slots(p: *const Self) usize {
            return p.moe.items.len;
        }

        pub fn neuron(_: *const Self, which: []f32, slot: usize, n: usize, d: usize) []f32 {
            return which[(slot * neurons + n) * d ..][0..d];
        }

        /// A fact's lesson begun: its answers, more wordings of its questions, its held-out answers, and the other prompts
        /// and answers its neurons stay silent on.
        pub fn begin(p: *Self, t: *T, host: anytype, answered: []const Example, prompts: []const Example, held: []const Example, steady: []const []const Example) !void {
            const c = t.config();
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
            p.split = p.pickSplit(&.{ answered, prompts }, answered);
            std.log.info("slide pocket: the fact's neurons: {d} at the reply's first row, {d} at its questions' subject, {d} before its answer's value", .{ p.split[0], p.split[1], p.split[2] });
            p.taken.clearRetainingCapacity();
            p.rows.clearRetainingCapacity();
            p.labels.clearRetainingCapacity();
            p.keeps.clearRetainingCapacity();
            p.subjects.clearRetainingCapacity();
            p.leads.clearRetainingCapacity();
            // the last fact's refits stay with it: kept with it, or already put back by its undo
            p.refits.clearRetainingCapacity();
            p.refit_up.clearRetainingCapacity();
            p.learned_before = p.learned.items.len;
            p.pos_before = p.pos.items.len;
            p.next(.read);
        }

        pub fn next(p: *Self, phase: @TypeOf(p.phase)) void {
            p.phase = phase;
            p.at = 0;
            p.v.restart();
        }

        /// One unit: an example read, every layer fitted, an answer of an output step, or a held-out answer judged;
        /// true once the fact's neurons are in the pocket and judged (`back` of the held-out answers came back).
        pub fn step(p: *Self, t: *T, host: anytype) !bool {
            return switch (p.phase) {
                .read => p.readOnce(t),
                .fit => p.fitAll(t, host),
                .values => values.valueOnce(p, t, host),
                .check => values.checkOnce(p, t, host),
                .done => true,
            };
        }

        /// Another round of outputs from where they are, then judged again.
        pub fn more(p: *Self) void {
            p.next(.values);
        }

        /// The open fact's neurons back to zero and free, and the earlier neurons it fitted again as they were.
        pub fn undo(p: *Self, t: *T, host: anytype) !void {
            const d = t.config().hidden;
            p.learned.shrinkRetainingCapacity(p.learned_before);
            p.pos.shrinkRetainingCapacity(p.pos_before);
            var i = p.taken.items.len;
            while (i > 0) { // the last taken first, so a slot's count comes back down past each of its neurons
                i -= 1;
                const k = p.taken.items[i];
                @memset(p.neuron(p.up.items, k.slot, k.neuron, d), 0);
                @memset(p.neuron(p.down.items, k.slot, k.neuron, d), 0);
                if (k.neuron + 1 == p.used.items[k.slot]) p.used.items[k.slot] -= 1;
                try p.upload(t, host, k.slot);
            }
            for (p.refits.items, 0..) |r, j| {
                @memcpy(p.neuron(p.up.items, r.slot, r.neuron, d), p.refit_up.items[j * d ..][0..d]);
                try p.upload(t, host, r.slot);
            }
            p.refits.clearRetainingCapacity();
            p.refit_up.clearRetainingCapacity();
            p.taken.clearRetainingCapacity();
        }

        pub fn upload(p: *Self, t: *T, host: anytype, slot: usize) !void {
            const n = neurons * t.config().hidden;
            try host.pocket(p.moe.items[slot], p.up.items[slot * n ..][0..n], p.down.items[slot * n ..][0..n], p.used.items[slot]);
        }

        /// Example i of the fact's answers, then its other wordings, then its held-out answers, then the other prompts,
        /// with its kind.
        pub fn pickRead(p: *const Self, i: usize) ?struct { ex: Example, kind: Kind } {
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
            const which: u32 = @intCast(p.at);
            p.at += 1;
            const ex = got.ex;
            const own = got.kind != .steady;
            // the fact's own subjects and the tokens before them, read first: another prompt naming the same one is where
            // its subject neurons may fire
            const subject = if (own and p.split[1] > 0) subjectRow(ex) else null;
            if (subject) |r| {
                if (std.mem.indexOfScalar(u32, p.subjects.items, ex.ids[r]) == null) try p.subjects.append(p.gpa, ex.ids[r]);
                if (r > 0 and std.mem.indexOfScalar(u32, p.leads.items, ex.ids[r - 1]) == null) try p.leads.append(p.gpa, ex.ids[r - 1]);
            }
            // its value row: the fact's own where the value comes, another answer's (an earlier fact's) one to stay silent on
            const value = if (p.split[2] > 0) valueRow(ex) else null;
            _ = try t.step(ex.ids, ex.start, .keys, .{ .decide = ex.decide, .bits = Acting.empty, .whole = true });
            try p.keep(t, ex.decide, .{ .kind = got.kind, .key = .reply, .ex = which, .row = @intCast(ex.decide) });
            if (subject) |r| try p.keep(t, r, .{ .kind = got.kind, .key = .subject, .ex = which, .row = @intCast(r) });
            if (value) |r| try p.keep(t, r, if (own) .{ .kind = got.kind, .key = .value, .ex = which, .row = @intCast(r) } else .{ .kind = .other, .key = null, .ex = which, .row = @intCast(r) });
            // every row of the fact's own examples (where a slip would sit inside its answers), every third of others',
            // and each that names the subject word. In the fact's own examples every mention is its subject (the recall
            // line's "My sister", the answer's "Your sister"): left free, never a row to stay silent on, since it reads
            // as the question's own does. In others', after the token the fact's questions put before it ("my sister",
            // the same company) it is the fact's subject, where its subject neurons may fire if that answer stays as it
            // was; after another ("your sister") someone else's, where they stay silent
            var r: usize = 0;
            while (r < ex.ids.len - 1) : (r += 1) {
                if (r == ex.decide or (subject != null and r == subject.?) or (value != null and r == value.?)) continue;
                const named = p.split[1] > 0 and std.mem.indexOfScalar(u32, p.subjects.items, ex.ids[r]) != null;
                if (named and (own or !p.strict or (r > 0 and std.mem.indexOfScalar(u32, p.leads.items, ex.ids[r - 1]) != null))) {
                    try p.keep(t, r, .{ .kind = .none, .key = .subject, .ex = which, .row = @intCast(r) });
                } else if (own or named or r % stride == 0) try p.keep(t, r, .{ .kind = .other, .key = null, .ex = which, .row = @intCast(r) });
            }
            return false;
        }

        fn keep(p: *Self, t: *T, r: usize, label: Label) !void {
            for (p.moe.items) |l| try p.rows.appendSlice(p.gpa, t.normed(l, r));
            try p.labels.append(p.gpa, label);
        }

        /// Every candidate layer's neuron fitted for each key at once (a thread a fit) and the best taken on each key;
        /// then the other answers its subject neurons fire on found, and earlier neurons that fire on its questions
        /// fitted again.
        fn fitAll(p: *Self, t: *T, host: anytype) !bool {
            const c = t.config();
            const d = c.hidden;
            const n = p.slots();
            const m = p.labels.items.len;
            if (std.c.getenv("TF_SLIDE_POCKET_DUMP")) |dir| pocket_dump.write(p, std.mem.span(dir), p.learned_before, d) catch |e| std.log.warn("slide pocket: the read rows were not written: {t}", .{e});
            const kinds = try p.gpa.alloc(Kind, 3 * m);
            defer p.gpa.free(kinds);
            for (p.labels.items, 0..) |l, i| for (0..3) |k| {
                kinds[k * m + i] = kindFor(l, @fromBackingInt(@intCast(k)));
            };
            const us = try p.gpa.alloc(f32, 3 * n * d);
            defer p.gpa.free(us);
            const fits = try p.gpa.alloc(Fit, 3 * n);
            defer p.gpa.free(fits);
            @memset(fits, .{ .fires = std.math.maxInt(usize) });
            var threads: [3 * 64]?std.Thread = @splat(null);
            for (0..3) |k| for (p.moe.items, 0..) |l, j| if (p.split[k] > 0 and candidate(l, c.layers, @fromBackingInt(@intCast(k)))) {
                threads[k * n + j] = std.Thread.spawn(.{}, fitLayer, .{ p.gpa, p.rows.items, kinds[k * m ..][0..m], n, j, d, us[(k * n + j) * d ..][0..d], &fits[k * n + j] }) catch null;
            };
            for (0..3) |k| for (p.moe.items, 0..) |l, j| if (p.split[k] > 0 and candidate(l, c.layers, @fromBackingInt(@intCast(k)))) {
                if (threads[k * n + j]) |th| th.join() else fitLayer(p.gpa, p.rows.items, kinds[k * m ..][0..m], n, j, d, us[(k * n + j) * d ..][0..d], &fits[k * n + j]);
            };
            for (0..3) |k| if (p.split[k] > 0) {
                const key: Key = @fromBackingInt(@intCast(k));
                var order: std.ArrayList(usize) = .empty;
                defer order.deinit(p.gpa);
                // a layer whose pocket is full is passed over
                for (p.moe.items, 0..) |l, j| if (candidate(l, c.layers, key) and p.used.items[j] < neurons) try order.append(p.gpa, j);
                if (order.items.len == 0) return error.PocketFull;
                std.mem.sort(usize, order.items, @as([]const Fit, fits[k * n ..][0..n]), better);
                for (order.items[0..@min(p.split[k], order.items.len)]) |j| try p.take(j, key, us[(k * n + j) * d ..][0..d], fits[k * n + j], kinds[k * m ..][0..m], d);
            };
            p.logSources(d, m, kinds);
            try p.findKeeps(d);
            try p.quiet(t, host);
            try p.v.begin(p.gpa, p.taken.items.len, d);
            p.next(.values);
            return false;
        }

        /// For each key's first neuron, the share of its fact rows it fires on (0.5 or more) from each source: the
        /// answers, the more wordings (the system and conversation variants among them, longer), the held-out answers.
        fn logSources(p: *const Self, d: usize, m: usize, kinds: []const Kind) void {
            const n = p.slots();
            for (p.taken.items) |x| {
                if (x.neuron + 1 != p.used.items[x.slot]) continue;
                const k = @backingInt(x.key);
                var on: [4]usize = @splat(0);
                var all: [4]usize = @splat(0);
                const u = p.neuron(@constCast(p.up.items), x.slot, x.neuron, d);
                for (p.labels.items, 0..) |l, i| {
                    const kind = kinds[k * m + i];
                    if (kind != .fact and kind != .held) continue;
                    const src: usize = if (l.ex < p.answered.len) 0 else if (l.ex < p.answered.len + p.prompts.len) (if (p.pickRead(l.ex).?.ex.ids.len > 60) 2 else 1) else 3;
                    all[src] += 1;
                    on[src] += @intFromBool(dotBf16(u, p.rows.items[(i * n + x.slot) * d ..][0..d]) >= 0.5);
                }
                std.log.info("slide pocket: layer {d} at the {s} fires on answers {d}/{d}, short wordings {d}/{d}, long ones {d}/{d}, held-out {d}/{d}", .{ p.moe.items[x.slot], @tagName(x.key), on[0], all[0], on[1], all[1], on[2], all[2], on[3], all[3] });
            }
        }

        /// Slot j's next neuron taken for the open fact on `key`, its input weights `u` as fitted (`f`, on `kinds`).
        fn take(p: *Self, j: usize, key: Key, u: []const f32, f: Fit, kinds: []const Kind, d: usize) !void {
            const n = p.slots();
            const k = p.used.items[j];
            p.used.items[j] += 1;
            try p.taken.append(p.gpa, .{ .slot = j, .neuron = k, .key = key });
            @memcpy(p.neuron(p.up.items, j, k, d), u);
            @memset(p.neuron(p.down.items, j, k, d), 0);
            std.log.info("slide pocket: layer {d}, neuron {d}, at the {s}: on for {d:.0}% of the fact's questions (least {d:.2}), held-out {d:.0}% (mean {d:.2}); other prompts at most {d:.2}, other rows {d:.2}, {d} above zero", .{ p.moe.items[j], k, @tagName(key), 100 * f.fact_on, f.fact_least, 100 * f.held_on, f.held_mean, f.steady_most, f.other_most, f.fires });
            const start = p.pos.items.len / d;
            for (kinds, 0..) |kind, i| if (kind == .fact) try p.pos.appendSlice(p.gpa, p.rows.items[(i * n + j) * d ..][0..d]);
            const subject: u32 = if (key == .subject and p.subjects.items.len > 0) p.subjects.items[0] else 0;
            try p.learned.append(p.gpa, .{ .slot = j, .neuron = k, .key = key, .start = start, .count = p.pos.items.len / d - start, .subject = subject });
        }

        /// Other prompts' answers that mention the open fact's subject where one of its subject neurons fires, each at
        /// the row it fires most: their answers are to stay as they were while its outputs are fitted.
        fn findKeeps(p: *Self, d: usize) !void {
            if (p.split[1] == 0 or !p.keeping) return;
            const n = p.slots();
            var i: usize = 0;
            while (i < p.labels.items.len) {
                const which = p.labels.items[i].ex;
                var most: f32 = 0;
                var at: ?usize = null;
                while (i < p.labels.items.len and p.labels.items[i].ex == which) : (i += 1) {
                    const l = p.labels.items[i];
                    // the fact's own mentions are free but already its answers, not ones to keep
                    if (l.kind != .none or l.ex < p.answered.len + p.prompts.len + p.held.len) continue;
                    var sum: f32 = 0;
                    for (p.taken.items) |x| if (x.key == .subject) {
                        const z = dotBf16(p.neuron(p.up.items, x.slot, x.neuron, d), p.rows.items[(i * n + x.slot) * d ..][0..d]);
                        sum += if (z > 0) z * z else 0;
                    };
                    if (sum > most) {
                        most = sum;
                        at = l.row;
                    }
                }
                const row = at orelse continue;
                const ex = p.pickRead(which).?.ex;
                if (ex.ids.len > ex.start + 1) try p.keeps.append(p.gpa, .{ .ex = ex, .row = row });
            }
            std.log.info("slide pocket: {d} other answers name the fact's subject where its neurons fire; they are to stay as they were", .{p.keeps.items.len});
        }

        /// Every earlier fact's neuron that would fire on this fact's key rows fitted again, from where it is, with them
        /// among the rows it stays silent on (its own fact's rows still the ones it fires on), its input weights from
        /// before kept for an undo. A subject neuron of a fact about the same subject is left as it is: what it knows of
        /// that subject may come up here, and this fact's outputs are fitted with it firing.
        fn quiet(p: *Self, t: *T, host: anytype) !void {
            const d = t.config().hidden;
            const n = p.slots();
            for (p.learned.items[0..p.learned_before]) |e| {
                if (e.key == .subject and std.mem.indexOfScalar(u32, p.subjects.items, e.subject) != null) continue;
                const u = p.neuron(p.up.items, e.slot, e.neuron, d);
                var most: f32 = -std.math.inf(f32);
                for (p.labels.items, 0..) |l, i| if (l.kind == .fact or l.kind == .held) {
                    most = @max(most, dotBf16(u, p.rows.items[(i * n + e.slot) * d ..][0..d]));
                };
                if (most <= refit_above) continue; // weak there: left exactly as it is, this fact's outputs fitted with it
                try p.refits.append(p.gpa, .{ .slot = e.slot, .neuron = e.neuron });
                try p.refit_up.appendSlice(p.gpa, u);
                var xs: std.ArrayList([]const u16) = .empty;
                defer xs.deinit(p.gpa);
                var kinds: std.ArrayList(Kind) = .empty;
                defer kinds.deinit(p.gpa);
                for (0..e.count) |r| {
                    try xs.append(p.gpa, p.pos.items[(e.start + r) * d ..][0..d]);
                    try kinds.append(p.gpa, .fact);
                }
                for (p.labels.items, 0..) |l, i| {
                    try xs.append(p.gpa, p.rows.items[(i * n + e.slot) * d ..][0..d]);
                    try kinds.append(p.gpa, switch (l.kind) {
                        .fact, .held => .steady, // this fact's questions are another fact's to it
                        .none => .other,
                        else => l.kind,
                    });
                }
                const f = try fit(p.gpa, xs.items, kinds.items, d, u, true, fit_steps / 2, e.count);
                if (f.fact_least < kept_least or @max(f.steady_most, f.other_most) > 0) {
                    // a refit that loses its own fact's questions is forgetting, and one that fires elsewhere would
                    // spread: either way it is put back as it was, and this fact's outputs are fitted with it firing
                    // where it does (no change may move what the model knew)
                    @memcpy(u, p.refit_up.items[p.refit_up.items.len - d ..]);
                    p.refits.shrinkRetainingCapacity(p.refits.items.len - 1);
                    p.refit_up.shrinkRetainingCapacity(p.refit_up.items.len - d);
                    std.log.info("slide pocket: layer {d}, neuron {d}, an earlier fact's, fires on this fact's questions (up to {d:.2}); fitted again its own would read at least {d:.2} and other rows at most {d:.2}, so it stays as it was", .{ p.moe.items[e.slot], e.neuron, most, f.fact_least, @max(f.steady_most, f.other_most) });
                    continue;
                }
                try p.upload(t, host, e.slot);
                std.log.info("slide pocket: layer {d}, neuron {d}, an earlier fact's, fired on this fact's questions (up to {d:.2}); fitted again: there and elsewhere at most {d:.2}, its own questions' least {d:.2}", .{ p.moe.items[e.slot], e.neuron, most, @max(f.steady_most, f.other_most), f.fact_least });
            }
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

        /// Where the open fact's lesson stands: its phase, its unit `at` of `of` (0: unknown), its outputs' loss.
        pub const Progress = struct { phase: @FieldType(Self, "phase"), at: usize = 0, of: usize = 0, loss: f32 = 0 };

        pub fn progress(p: *const Self) Progress {
            return switch (p.phase) {
                .read => .{ .phase = .read, .at = p.at, .of = p.reads() },
                .values => .{ .phase = .values, .at = p.v.round, .loss = p.v.running },
                .check => .{ .phase = .check, .at = p.at, .of = p.held.len },
                else => .{ .phase = p.phase },
            };
        }

        /// The examples the read phase goes through: the fact's own, then every prompt its neurons stay silent on.
        fn reads(p: *const Self) usize {
            var n = p.answered.len + p.prompts.len + p.held.len;
            for (p.steady) |xs| n += xs.len;
            return n;
        }

        /// The open fact's neurons as placed, into `out` (the first of them when it is short); the ones written.
        pub fn placed(p: *const Self, out: []Placed) []Placed {
            const n = @min(out.len, p.taken.items.len);
            for (p.taken.items[0..n], out[0..n]) |x, *o| o.* = .{ .layer = p.moe.items[x.slot], .neuron = x.neuron, .key = x.key };
            return out[0..n];
        }
    };
}

test "a fact keys as many neurons on its subject as the share of its questions that name it, at least half, else none" {
    const P = Pocket(struct {});
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    // ids: a question's tokens, the subject's last token at row 2, the answer from row 4
    const ids = [_]u32{ 1, 2, 7, 3, 4, 5 };
    const named: Example = .{ .ids = &ids, .start = 4, .decide = 3, .subject = 2 };
    const plain: Example = .{ .ids = &ids, .start = 4, .decide = 3 };
    try std.testing.expectEqual([3]usize{ 5, 5, 0 }, p.pickSplit(&.{ &.{ named, plain }, &.{} }, &.{}));
    try std.testing.expectEqual([3]usize{ 0, 10, 0 }, p.pickSplit(&.{ &.{ named, named }, &.{named} }, &.{}));
    try std.testing.expectEqual([3]usize{ 3, 7, 0 }, p.pickSplit(&.{&.{ named, named, plain }}, &.{}));
    try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{ named, plain, plain }}, &.{}));
    try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{}}, &.{}));
    // answers whose value comes after row 4: none there unless asked, then four, the rest shared as before
    const valued: Example = .{ .ids = &ids, .start = 4, .decide = 3, .subject = 2, .value = 4 };
    try std.testing.expectEqual([3]usize{ 0, 10, 0 }, p.pickSplit(&.{&.{ valued, valued }}, &.{ valued, valued }));
    p.valuing = true;
    try std.testing.expectEqual([3]usize{ 0, 6, 4 }, p.pickSplit(&.{&.{ valued, valued }}, &.{ valued, valued }));
    p.mode = .subject;
    try std.testing.expectEqual([3]usize{ 0, 10, 0 }, p.pickSplit(&.{&.{ named, plain, plain }}, &.{}));
    p.mode = .reply;
    try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{ named, named }}, &.{}));
}

test "a read row is a fact's row only to its own key's neurons; another prompt's mention of the subject is left free" {
    try std.testing.expectEqual(Kind.fact, kindFor(.{ .kind = .fact, .key = .subject, .ex = 0, .row = 2 }, .subject));
    try std.testing.expectEqual(Kind.other, kindFor(.{ .kind = .fact, .key = .subject, .ex = 0, .row = 2 }, .reply));
    try std.testing.expectEqual(Kind.held, kindFor(.{ .kind = .held, .key = .reply, .ex = 3, .row = 7 }, .reply));
    try std.testing.expectEqual(Kind.steady, kindFor(.{ .kind = .steady, .key = .reply, .ex = 1, .row = 9 }, .subject));
    try std.testing.expectEqual(Kind.none, kindFor(.{ .kind = .none, .key = .subject, .ex = 1, .row = 4 }, .subject));
    try std.testing.expectEqual(Kind.other, kindFor(.{ .kind = .none, .key = .subject, .ex = 1, .row = 4 }, .reply));
    try std.testing.expectEqual(Kind.other, kindFor(.{ .kind = .other, .key = null, .ex = 1, .row = 5 }, .subject));
}
