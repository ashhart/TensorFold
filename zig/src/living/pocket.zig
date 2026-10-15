//! A pocket of new neurons beside each MoE layer's shared expert, where facts are learned; nothing the model had moves.
const std = @import("std");
const examples = @import("lesson_examples.zig");
const lesson_route = @import("lesson_route.zig");
const pocket_fit = @import("pocket_fit.zig");
const values = @import("pocket_values.zig");
const joint = @import("pocket_joint.zig");
const pocket_dump = @import("pocket_dump.zig");
const clock = @import("pocket_clock.zig");
const reading = @import("pocket_read.zig");
const quieting = @import("pocket_quiet.zig");
const loading = @import("pocket_load.zig");
const family = @import("family.zig");
const neuron_math = @import("neuron.zig");

const Example = examples.Example;
const Acting = lesson_route.Acting;
const Kind = pocket_fit.Kind;
const Fit = pocket_fit.Fit;
const fit = pocket_fit.fit;
const fitLayer = pocket_fit.fitLayer;
const dotBf16 = pocket_fit.dotBf16;
const fit_steps = pocket_fit.fit_steps;

const per_fact = 10; // neurons a fact takes, at the layers that tell its questions apart best
const value_neurons = 4; // of those, the ones before its answer's value, when its answers show one

/// Where a neuron fires: the reply's first row, the named subject's last token, or the row before the answer's value.
pub const Key = enum(u2) { reply, subject, value };

/// Keying: auto gives the subject the share of questions naming it (if half or more), the rest to the reply.
pub const Mode = enum { auto, reply, subject, both }; // both: half at the subject, half at the reply (relation-aware)

/// A neuron the open fact took.
const Taken = struct { slot: usize, neuron: usize, key: Key };

/// A neuron the open fact took, where it sits: its MoE layer, its index in that layer's pocket, its key.
pub const Placed = struct { layer: usize, neuron: usize, key: Key };

/// A neuron a fact took, the rows it was fitted to fire on (in `pos`), and its fact's subject token (keyed on it).
const Learned = struct { slot: usize, neuron: usize, key: Key, start: usize, count: usize, subject: u32, lesson: u32 };

/// A read row: what it is to the open fact, whose key row it is (null: neither), its example and its place there.
pub const Label = struct { kind: Kind, key: ?Key, ex: u32, row: u32 };

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

/// What read row `l` is to neurons keyed `key`: their own key rows as read, every other row one they stay silent on.
fn kindFor(l: Label, key: Key) Kind {
    if (l.key == key) return l.kind;
    return if (l.kind == .steady) .steady else .other;
}

/// Whether layer l may hold a `key` neuron: a fifth to seven eighths in; a subject's to five eighths, where facts sit.
fn candidate(l: usize, layers: usize, key: Key) bool {
    const last = if (key == .subject) layers * 5 / 8 else layers * 7 / 8;
    return l >= layers / 5 and l <= last;
}

/// Strict layers first (silent on every row read), then most of the fact's questions on, then the widest least margin.
fn better(f: []const Fit, a: usize, b: usize) bool {
    if (f[a].fires != f[b].fires) return f[a].fires < f[b].fires;
    if (f[a].fact_on != f[b].fact_on) return f[a].fact_on > f[b].fact_on;
    return f[a].fact_least > f[b].fact_least;
}

/// A pocket over trainer T (its inputs, residuals, nudges) for family F, and a host handing each layer to the forward.
pub fn Pocket(comptime T: type, comptime F: family.Family) type {
    comptime F.check() catch unreachable;
    return struct {
        const Self = @This();
        /// A pocket's neurons at each layer it sits at: a whole number of quantization groups of the expert's width.
        pub const neurons = F.neurons;
        pub const of = F; // the family it learns into
        pub const swiglu = F.neuron.act == .swiglu;
        pub const gated = F.neuron.gated;

        gpa: std.mem.Allocator,
        mode: Mode, // as asked (TF_SLIDE_POCKET_KEY), auto by default
        strict: bool = true, // the subject word after another token is someone else's (TF_SLIDE_POCKET_LEADS=off: anyone's)
        keeping: bool = true, // other answers its subject neurons fire on stay as they were (TF_SLIDE_POCKET_KEEPS=off: not held)
        valuing: bool = false, // some neurons before the answer's value (TF_SLIDE_POCKET_VALUES=on); no published method has them
        silencing: bool = false, // other prompts naming the subject as its questions do stay silent too (TF_SLIDE_POCKET_SAME=silent)
        worst: bool = false, // half a step's answers to stay are those that read worst (TF_SLIDE_POCKET_REPLAY=worst)
        joining: bool = false, // earlier facts on its subject fitted with the open fact (TF_SLIDE_POCKET_GROUP=joint)
        joint: std.ArrayList(Taken) = .empty, // those facts' neurons, their outputs fitted with the open fact's
        joint_down: std.ArrayList(f32) = .empty, // [joint, D]: their outputs before, put back by an undo
        joint_answers: std.ArrayList(Example) = .empty, // their replayed answers, fitted on as the open fact's own are
        replay_lessons: []const u32 = &.{}, // the lesson each answer of steady[2] (the replay) came from
        split: [3]usize = .{ per_fact, 0, 0 }, // the open fact's neurons on each key
        subjects: std.ArrayList(u32) = .empty, // the open fact's subject tokens
        leads: std.ArrayList(u32) = .empty, // the tokens before them: after another, the subject is someone else's
        moe: std.ArrayList(usize) = .empty, // the layers a pocket learns at (the layout's), its slots in order
        hidden: usize = 0, // D, as the layout gives it
        layers: usize = 0, // the model's layers, as the layout gives them
        up: std.ArrayList(f32) = .empty, // [slots, neurons, D]: each neuron's input weights u, as fitted
        down: std.ArrayList(f32) = .empty, // [slots, neurons, D]: each neuron's output
        gate: std.ArrayList(f32) = .empty, // SwiGLU: [slots, neurons, D], each neuron's gate row (sharp u, or as written)
        scratch: std.ArrayList(f32) = .empty, // SwiGLU: [neurons, D], a slot's up rows (u / sharp) as handed over
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
        lesson: u32 = 0, // the open fact's lesson, which its neurons record
        retired_lessons: std.ArrayList(u32) = .empty, // lessons a newer fact replaced: their answers no longer stay
        retired_marks: usize = 0, // retired_lessons before the last retire, which a put-back returns to
        retired: std.ArrayList([2]usize) = .empty, // the last retire's (slot, neuron), put back if the newer fact fails
        retired_w: std.ArrayList(f32) = .empty, // [retired, 2, D]: their input and output weights from before
        refit_gate: std.ArrayList(f32) = .empty, // SwiGLU: [refits, D], their gate rows from before
        phase: enum { read, fit, values, check, done } = .done,
        at: usize = 0, // the next example read, answer of a step, or held-out answer judged
        back: usize = 0, // held-out answers that came back token for token
        learned: std.ArrayList(Learned) = .empty, // every neuron a fact took, this session
        pos: std.ArrayList(u16) = .empty, // [rows, D]: the rows each fired on, its fact's key rows
        learned_before: usize = 0, // learned and pos as the open fact found them, for an undo
        pos_before: usize = 0,
        v: values.State = .{}, // the open fact's outputs as they are fitted
        clock: clock.Clock = .{}, // where the open round's time goes
        read: reading.Seen = .{}, // what the last read pass read, which a lesson begun again over the same keeps

        pub fn init(gpa: std.mem.Allocator) Self {
            const asked: Mode = if (std.c.getenv("TF_SLIDE_POCKET_KEY")) |v| std.meta.stringToEnum(Mode, std.mem.span(v)) orelse .auto else .auto;
            return .{ .gpa = gpa, .mode = asked, .strict = !set("TF_SLIDE_POCKET_LEADS", "off"), .keeping = !set("TF_SLIDE_POCKET_KEEPS", "off"), .valuing = set("TF_SLIDE_POCKET_VALUES", "on"), .silencing = set("TF_SLIDE_POCKET_SAME", "silent"), .worst = set("TF_SLIDE_POCKET_REPLAY", "worst"), .joining = set("TF_SLIDE_POCKET_GROUP", "joint") };
        }

        pub fn deinit(p: *Self) void {
            inline for (.{ "moe", "up", "down", "gate", "scratch", "used", "taken", "rows", "labels", "keeps", "refits", "refit_up", "refit_gate", "retired_lessons", "retired", "retired_w", "joint", "joint_down", "joint_answers", "learned", "pos", "subjects", "leads" }) |f| @field(p, f).deinit(p.gpa);
            p.read.deinit(p.gpa);
            p.v.deinit(p.gpa);
        }

        /// The fact's neurons per key: as asked, or by how many answers show a value and questions name the subject.
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
            if (p.mode == .both) return .{ rest / 2, rest - rest / 2, on_value };
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

        /// Whether the fact's neurons fire on `ex`: at its reply, its subject if named, or before its value if said.
        pub fn fires(p: *const Self, ex: Example) bool {
            return p.split[0] > 0 or (p.split[1] > 0 and subjectRow(ex) != null) or (p.split[2] > 0 and valueRow(ex) != null);
        }

        fn slots(p: *const Self) usize {
            return p.moe.items.len;
        }

        pub fn neuron(_: *const Self, which: []f32, slot: usize, n: usize, d: usize) []f32 {
            return which[(slot * neurons + n) * d ..][0..d];
        }

        /// Neuron n of `slot`'s activation at input `x` (bf16): relu(u x)^2, or silu(g x) (u x) / sharp (neuron.zig).
        pub fn act(p: *const Self, slot: usize, n: usize, x: []const u16) f32 {
            const d = p.hidden;
            const z = dotBf16(p.neuron(@constCast(p.up.items), slot, n, d), x);
            const zg = if (swiglu) dotBf16(p.neuron(@constCast(p.gate.items), slot, n, d), x) else 0;
            return neuron_math.act(F.neuron, z, zg);
        }

        /// A SwiGLU neuron's gate row set from its input weights as they are now.
        pub fn gateFrom(p: *Self, slot: usize, n: usize) void {
            if (!swiglu) return;
            neuron_math.gateRow(F.neuron, p.neuron(p.up.items, slot, n, p.hidden), p.neuron(p.gate.items, slot, n, p.hidden));
        }

        fn load(p: *Self, t: *T, host: anytype) !void {
            _ = t;
            return loading.load(p, host);
        }

        /// A replaced fact's neurons, (layer, neuron) each, zeroed where they sit; their places stay taken.
        pub fn retire(p: *Self, t: *T, host: anytype, gone: []const [2]u16) !void {
            try p.load(t, host);
            const d = p.hidden;
            p.retired.clearRetainingCapacity();
            p.retired_w.clearRetainingCapacity();
            p.retired_marks = p.retired_lessons.items.len;
            for (gone) |g| {
                const slot = std.mem.indexOfScalar(usize, p.moe.items, g[0]) orelse continue;
                if (g[1] >= p.used.items[slot]) continue;
                for (p.learned.items) |e| if (e.slot == slot and e.neuron == g[1] and std.mem.indexOfScalar(u32, p.retired_lessons.items, e.lesson) == null) try p.retired_lessons.append(p.gpa, e.lesson);
                try p.retired.append(p.gpa, .{ slot, g[1] });
                try p.retired_w.appendSlice(p.gpa, p.neuron(p.up.items, slot, g[1], d));
                try p.retired_w.appendSlice(p.gpa, p.neuron(p.down.items, slot, g[1], d));
                @memset(p.neuron(p.up.items, slot, g[1], d), 0);
                @memset(p.neuron(p.down.items, slot, g[1], d), 0);
                try p.upload(t, host, slot);
            }
            std.log.info("slide pocket: {d} neurons of replaced facts zeroed", .{gone.len});
        }

        /// The last retire undone: its neurons as they were (the fact that replaced theirs was not learned).
        pub fn unretire(p: *Self, t: *T, host: anytype) !void {
            try p.load(t, host);
            const d = p.hidden;
            for (p.retired.items, 0..) |r, i| {
                @memcpy(p.neuron(p.up.items, r[0], r[1], d), p.retired_w.items[2 * i * d ..][0..d]);
                @memcpy(p.neuron(p.down.items, r[0], r[1], d), p.retired_w.items[(2 * i + 1) * d ..][0..d]);
                try p.upload(t, host, r[0]);
            }
            std.log.info("slide pocket: {d} neurons of replaced facts put back", .{p.retired.items.len});
            p.retired_lessons.shrinkRetainingCapacity(p.retired_marks);
            p.retired.clearRetainingCapacity();
            p.retired_w.clearRetainingCapacity();
        }

        /// A fact's lesson begun: its answers, more wordings, held-out answers and prompts its neurons stay silent on.
        pub fn begin(p: *Self, t: *T, host: anytype, answered: []const Example, prompts: []const Example, held: []const Example, steady: []const []const Example) !void {
            try p.load(t, host);
            p.answered = answered;
            p.prompts = prompts;
            p.held = held;
            p.steady = steady;
            p.split = p.pickSplit(&.{ answered, prompts }, answered);
            std.log.info("slide pocket: the fact's neurons: {d} at the reply's first row, {d} at its questions' subject, {d} before its answer's value", .{ p.split[0], p.split[1], p.split[2] });
            p.taken.clearRetainingCapacity();
            p.keeps.clearRetainingCapacity();
            const kept = try reading.again(p);
            if (kept == 0) {
                p.rows.clearRetainingCapacity();
                p.labels.clearRetainingCapacity();
                p.subjects.clearRetainingCapacity();
                p.leads.clearRetainingCapacity();
            }
            // the last fact's refits stay with it: kept with it, or already put back by its undo
            p.refits.clearRetainingCapacity();
            p.refit_up.clearRetainingCapacity();
            p.joint.clearRetainingCapacity();
            p.joint_down.clearRetainingCapacity();
            p.refit_gate.clearRetainingCapacity();
            p.learned_before = p.learned.items.len;
            p.pos_before = p.pos.items.len;
            p.clock = .{};
            p.next(.read);
            p.at = kept;
        }

        pub fn next(p: *Self, phase: @TypeOf(p.phase)) void {
            p.phase = phase;
            p.at = 0;
            p.v.restart();
        }

        /// One unit: a read, the fits, an output answer or a held-out answer judged; true once the neurons are judged.
        pub fn step(p: *Self, t: *T, host: anytype) !bool {
            const part: clock.Part = switch (p.phase) {
                .read => .read,
                .fit => .fit,
                .values => if (p.v.judging) .judge else .values,
                .check => .check,
                .done => return true,
            };
            const t0 = clock.now(host.io);
            const before = t.count;
            const done = switch (p.phase) {
                .read => try reading.once(p, t),
                .fit => try p.fitAll(t, host),
                .values => try values.valueOnce(p, t, host),
                .check => try values.checkOnce(p, t, host),
                .done => true,
            };
            p.clock.add(part, clock.now(host.io) - t0, before, t.count);
            if (p.phase == .done) p.clock.end();
            return done;
        }

        /// Another round of outputs from where they are, then judged again.
        pub fn more(p: *Self) void {
            p.next(.values);
        }

        /// The open fact's neurons back to zero and free, and the earlier neurons it fitted again as they were.
        pub fn undo(p: *Self, t: *T, host: anytype) !void {
            const d = p.hidden;
            p.learned.shrinkRetainingCapacity(p.learned_before);
            p.pos.shrinkRetainingCapacity(p.pos_before);
            var i = p.taken.items.len;
            while (i > 0) { // the last taken first, so a slot's count comes back down past each of its neurons
                i -= 1;
                const k = p.taken.items[i];
                @memset(p.neuron(p.up.items, k.slot, k.neuron, d), 0);
                @memset(p.neuron(p.down.items, k.slot, k.neuron, d), 0);
                if (swiglu) @memset(p.neuron(p.gate.items, k.slot, k.neuron, d), 0);
                if (k.neuron + 1 == p.used.items[k.slot]) p.used.items[k.slot] -= 1;
                try p.upload(t, host, k.slot);
            }
            for (p.refits.items, 0..) |r, j| {
                @memcpy(p.neuron(p.up.items, r.slot, r.neuron, d), p.refit_up.items[j * d ..][0..d]);
                if (swiglu) @memcpy(p.neuron(p.gate.items, r.slot, r.neuron, d), p.refit_gate.items[j * d ..][0..d]);
                try p.upload(t, host, r.slot);
            }
            p.refits.clearRetainingCapacity();
            p.refit_up.clearRetainingCapacity();
            try joint.putBack(p, t, host, d);
            p.refit_gate.clearRetainingCapacity();
            p.taken.clearRetainingCapacity();
        }

        /// Slot `slot`'s neurons into the forward: relu^2 as u, SwiGLU as its gate rows and up rows u / sharp.
        pub fn upload(p: *Self, t: *T, host: anytype, slot: usize) !void {
            _ = t;
            const n = neurons * p.hidden;
            const up = p.up.items[slot * n ..][0..n];
            const down = p.down.items[slot * n ..][0..n];
            if (swiglu) {
                neuron_math.upRow(F.neuron, up, p.scratch.items);
                try host.pocketSwiglu(p.moe.items[slot], p.gate.items[slot * n ..][0..n], p.scratch.items, down, p.used.items[slot]);
            } else try host.pocket(p.moe.items[slot], up, down, p.used.items[slot]);
        }

        /// Example i and its kind: the fact's answers, then its other wordings, held-out answers and the other prompts.
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

        /// The open fact's neurons scaled row by row in the trainer's passes: at each layer, from its first on.
        pub fn open(p: *const Self, t: *T) void {
            t.from = @splat(std.math.maxInt(u32));
            for (p.taken.items) |x| {
                const l = p.moe.items[x.slot];
                t.from[l] = @min(t.from[l], @as(u32, @intCast(x.neuron)));
            }
        }

        /// Every key's candidate layers fitted in threads, the best taken; then keeps found, earlier neurons refitted.
        fn fitAll(p: *Self, t: *T, host: anytype) !bool {
            const d = p.hidden;
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
            const t0 = clock.now(host.io);
            const threads = try p.gpa.alloc(?std.Thread, 3 * n);
            defer p.gpa.free(threads);
            @memset(threads, null);
            var fitted: usize = 0;
            const below = F.neuron.margin;
            for (0..3) |k| for (p.moe.items, 0..) |l, j| if (p.split[k] > 0 and candidate(l, p.layers, @fromBackingInt(@intCast(k)))) {
                threads[k * n + j] = std.Thread.spawn(.{}, fitLayer, .{ p.gpa, p.rows.items, kinds[k * m ..][0..m], n, j, d, us[(k * n + j) * d ..][0..d], &fits[k * n + j], below }) catch null;
                fitted += 1;
            };
            for (0..3) |k| for (p.moe.items, 0..) |l, j| if (p.split[k] > 0 and candidate(l, p.layers, @fromBackingInt(@intCast(k)))) {
                if (threads[k * n + j]) |th| th.join() else fitLayer(p.gpa, p.rows.items, kinds[k * m ..][0..m], n, j, d, us[(k * n + j) * d ..][0..d], &fits[k * n + j], below);
            };
            const t1 = clock.now(host.io);
            for (0..3) |k| if (p.split[k] > 0) {
                const key: Key = @fromBackingInt(@intCast(k));
                var order: std.ArrayList(usize) = .empty;
                defer order.deinit(p.gpa);
                // a layer whose pocket is full is passed over
                for (p.moe.items, 0..) |l, j| if (candidate(l, p.layers, key) and p.used.items[j] < neurons) try order.append(p.gpa, j);
                if (order.items.len == 0) return error.PocketFull;
                std.mem.sort(usize, order.items, @as([]const Fit, fits[k * n ..][0..n]), better);
                for (order.items[0..@min(p.split[k], order.items.len)]) |j| try p.take(j, key, us[(k * n + j) * d ..][0..d], fits[k * n + j], kinds[k * m ..][0..m], d);
            };
            p.logSources(d, m, kinds);
            try p.findKeeps(d);
            try quieting.quiet(p, t, host);
            std.log.info("slide pocket: {d} candidate neurons fitted on {d} read rows in {d:.1} s; earlier neurons checked in {d:.1} s", .{ fitted, m, t1 - t0, clock.now(host.io) - t1 });
            try joint.find(p, d);
            try p.v.begin(p.gpa, p.taken.items.len + p.joint.items.len, d, p.answered.len + p.joint_answers.items.len + p.keeps.items.len);
            joint.seed(p, d);
            p.next(.values);
            return false;
        }

        /// Logs each key's first neuron's firing share by source: answers, short and long wordings, held-out answers.
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
            p.gateFrom(j, k);
            @memset(p.neuron(p.down.items, j, k, d), 0);
            std.log.info("slide pocket: layer {d}, neuron {d}, at the {s}: on for {d:.0}% of the fact's questions (least {d:.2}), held-out {d:.0}% (mean {d:.2}); other prompts at most {d:.2}, other rows {d:.2}, {d} above zero", .{ p.moe.items[j], k, @tagName(key), 100 * f.fact_on, f.fact_least, 100 * f.held_on, f.held_mean, f.steady_most, f.other_most, f.fires });
            const start = p.pos.items.len / d;
            for (kinds, 0..) |kind, i| if (kind == .fact) try p.pos.appendSlice(p.gpa, p.rows.items[(i * n + j) * d ..][0..d]);
            const subject: u32 = if (key == .subject and p.subjects.items.len > 0) p.subjects.items[0] else 0;
            try p.learned.append(p.gpa, .{ .slot = j, .neuron = k, .key = key, .start = start, .count = p.pos.items.len / d - start, .subject = subject, .lesson = p.lesson });
        }

        /// Other answers where the fact's subject neurons fire, each at its strongest row: these stay as they were.
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
                        sum += @max(p.act(x.slot, x.neuron, p.rows.items[(i * n + x.slot) * d ..][0..d]), 0);
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

        /// Whether no neuron is in use at any layer.
        pub fn empty(p: *const Self) bool {
            for (p.used.items) |u| if (u > 0) return false;
            return true;
        }

        /// Learned neurons into the shards, then into the forward as the shards hold them (quantized); the bytes.
        pub fn save(p: *Self, t: *T, host: anytype) !usize {
            const bytes = if (swiglu) blk: {
                // written as gate rows and up rows u / sharp, handed back as written: u from the up rows again
                const rows = try p.gpa.alloc(f32, p.up.items.len);
                defer p.gpa.free(rows);
                neuron_math.upRow(F.neuron, p.up.items, rows);
                const got = try host.pocketSaveSwiglu(p.moe.items, neurons, p.gate.items, rows, p.down.items);
                neuron_math.fromUp(F.neuron, rows, p.up.items);
                break :blk got;
            } else try host.pocketSave(p.moe.items, neurons, p.up.items, p.down.items);
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
    const P = Pocket(struct {}, @import("families.zig").nemotron_h);
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
