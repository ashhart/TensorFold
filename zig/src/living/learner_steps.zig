//! The learner's one-pass requests (replies scored, the pocket read on replies, lessons saved) and a lesson's build.
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
const pack = @import("pocket_pack.zig");
const max_steps = examples.max_steps;
const Deck = examples.Deck;
const key = examples.key;
const total = examples.total;
const pick = examples.pick;
const own = examples.own;
const disown = examples.disown;
const disownLesson = examples.disownLesson;
const check_every = 20;
const learner = @import("learner.zig");
const Example = learner.Example;
const Step = learner.Step;

/// One pass of a score request's replies: whether each is its prompt's greedy reply under the model as it is.
pub fn scoreOnce(l: anytype) !Step {
    if (comptime !@TypeOf(l.*).has_greedy) unreachable;
    const t = l.trainer.?;
    const xs = l.scoring[l.same.items.len..];
    if (xs.len == 0) {
        l.phase = .idle;
        return .{ .done = true, .report = .{ .scored = l.same.items } };
    }
    if (xs[0].ids.len - 1 > @TypeOf(l.*).Host.max_rows or xs[0].start < 1 or xs[0].start >= xs[0].ids.len) {
        try l.same.append(l.gpa, false);
        return .{ .done = false };
    }
    var seqs: [@typeInfo(@TypeOf(t.spans)).array.len]train_mode.Seq = undefined;
    var n: usize = 0;
    while (n < seqs.len and n < xs.len and xs[n].ids.len - 1 <= @TypeOf(l.*).Host.max_rows and xs[n].start >= 1 and xs[n].start < xs[n].ids.len) : (n += 1) seqs[n] = pack.seq(xs[n]);
    n = pack.fits(t, seqs[0..n]);
    var got: [seqs.len]bool = undefined;
    try t.greedy(seqs[0..n], got[0..n]);
    try l.same.appendSlice(l.gpa, got[0..n]);
    return .{ .done = false };
}

/// One pass of a quiet request's examples: whether the open lesson's pocket neurons' input products stay at or.
pub fn quietOnce(l: anytype) !Step {
    if (comptime !@TypeOf(l.*).has_pocket) unreachable;
    const t = l.trainer.?;
    const xs = l.scoring[l.same.items.len..];
    if (xs.len == 0) {
        l.phase = .idle;
        return .{ .done = true, .report = .{ .scored = l.same.items } };
    }
    if (xs[0].ids.len - 1 > @TypeOf(l.*).Host.max_rows or xs[0].start < 1 or xs[0].start >= xs[0].ids.len) {
        try l.same.append(l.gpa, false);
        return .{ .done = false };
    }
    var seqs: [@typeInfo(@TypeOf(t.spans)).array.len]train_mode.Seq = undefined;
    var n: usize = 0;
    while (n < seqs.len and n < xs.len and xs[n].ids.len - 1 <= @TypeOf(l.*).Host.max_rows and xs[n].start >= 1 and xs[n].start < xs[n].ids.len) : (n += 1) seqs[n] = pack.seq(xs[n]);
    n = pack.fits(t, seqs[0..n]);
    var out: [seqs.len]pack.Result = undefined;
    try t.pass(seqs[0..n], .keys, pack.turn, out[0..n]);
    for (0..n) |i| {
        const most = l.pockets.highest(t, t.spans[i].at, xs[i].ids.len - 1);
        std.log.info("slide pocket: the open lesson's neurons' highest input product on a {d}-token reply: {d:.3}", .{ xs[i].ids.len, most });
        try l.same.append(l.gpa, most <= 0);
    }
    return .{ .done = false };
}

/// Every kept lesson not yet written goes beside the model's shards, a file each.
pub fn saveOnce(l: anytype) !Step {
    l.phase = .idle;
    const t = l.trainer orelse return .{ .done = true };
    if (l.whole) { // a lesson of the model itself goes into its own shards: the pocket, or an edit's codes
        if (@TypeOf(l.*).has_whole and !try whole_lessons.save(l)) try whole_lessons.foldOpen(l);
        return .{ .done = true, .changed = true };
    }
    if (comptime !@TypeOf(l.*).has_sites) return .{ .done = true };
    const ranks = t.sites.rank - @as(usize, if (l.opened and l.kept_rounds == 0) adapters.block else 0);
    if (ranks == 0) return .{ .done = true };
    const t0 = @TypeOf(l.*).Host.seconds(l.io);
    const bytes = try @TypeOf(l.*).Host.write(l.b, l.gpa, l.io, t, ranks);
    std.log.info("slide: {d} lessons beside the model's shards, a file each with its router: {d} MB written in {d:.1} s", .{ ranks / adapters.block, bytes >> 20, @TypeOf(l.*).Host.seconds(l.io) - t0 });
    return .{ .done = true };
}

/// One example at a time: sketched, projected, routing rows copied; then the block chosen, its router fitted.
pub fn buildOnce(l: anytype) !Step {
    if (comptime !@TypeOf(l.*).has_sites) return error.RoutedUnsupported;
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
    try @TypeOf(l.*).Host.attach(l.b, t);
    try l.start();
    return .{ .done = false };
}
