//! Whole-mode lessons: pocket neurons (default), a weight edit (TF_SLIDE_EDIT) or a probe (TF_SLIDE_PROBE).
const std = @import("std");
const platform = @import("sys");
const examples = @import("lesson_examples.zig");
const forgetting = @import("pocket_forget.zig");
const adapters = @import("learn_shape.zig");

const Example = examples.Example;
const Lesson = examples.Lesson;

pub const Method = enum { pocket, edit, probe };

/// The method the environment picks for whole lessons.
pub fn method() Method {
    if (platform.c.getenv("TF_SLIDE_PROBE") != null) return .probe;
    if (platform.c.getenv("TF_SLIDE_EDIT") != null) return .edit;
    return .pocket;
}

/// Where the editor and the pocket reach host H: a layer's codes and their commit, the rounding, a layer's pocket.
pub fn Hooks(comptime H: type) type {
    return struct {
        const Self = @This();
        b: H.Ctx,
        gpa: std.mem.Allocator,
        io: std.Io,

        pub fn codes(f: Self, layer: usize) !@import("learner.zig").Codes {
            return H.codes(f.b, layer);
        }

        pub fn commit(f: Self, layer: usize) !void {
            return H.commit(f.b, f.gpa, f.io, layer);
        }

        pub fn addRounded(f: Self, words: []u32, scales: []const u16, biases: []const u16, rows: usize, cols: usize, delta: []f32, h: []f32, add: f32) !usize {
            return H.fold4.addRounded(f.gpa, words, scales, biases, rows, cols, delta, h, add);
        }

        pub fn pocket(f: Self, layer: usize, up: []const f32, down: []const f32, used: usize) !void {
            return H.pocket(f.b, layer, up, down, used);
        }

        pub fn pocketRound(f: Self, up: []f32, down: []f32) !void {
            return H.pocketRound(f.b, f.gpa, up, down);
        }

        pub fn pocketRead(f: Self, layer: usize, up: []f32, down: []f32) !usize {
            return H.pocketRead(f.b, layer, up, down);
        }

        pub fn saveBegin(f: Self) !void {
            if (@hasDecl(H, "pocketSaveBegin")) try H.pocketSaveBegin(f.b, f.gpa, f.io);
        }
        pub fn saveFinish(f: Self) !void {
            if (@hasDecl(H, "pocketSaveFinish")) try H.pocketSaveFinish(f.b, f.gpa, f.io);
        }

        pub fn pocketSave(f: Self, layers: []const usize, neurons: usize, up: []f32, down: []f32) !usize {
            return H.pocketSave(f.b, f.gpa, f.io, layers, neurons, up, down);
        }

        /// The served checkpoint's layout: where pockets sit and learn, its width (family.zig).
        pub fn layout(f: Self) @import("family.zig").Layout {
            return H.layout(f.b);
        }

        /// A SwiGLU family's layer `layer` for the forward: gate rows, up rows (u / sharp) and outputs, [neurons, D].
        pub fn pocketSwiglu(f: Self, layer: usize, gate: []const f32, up: []const f32, down: []const f32, used: usize) !void {
            return H.pocketSwiglu(f.b, layer, gate, up, down, used);
        }

        pub fn pocketReadSwiglu(f: Self, layer: usize, gate: []f32, up: []f32, down: []f32) !usize {
            return H.pocketReadSwiglu(f.b, layer, gate, up, down);
        }

        pub fn pocketSaveSwiglu(f: Self, layers: []const usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !usize {
            return H.pocketSaveSwiglu(f.b, f.gpa, f.io, layers, neurons, gate, up, down);
        }
    };
}

fn hooks(l: anytype) Hooks(@TypeOf(l.*).Host) {
    return .{ .b = l.b, .gpa = l.gpa, .io = l.io };
}

/// A new whole lesson begun, from a copy of its own (a reroute starts it again after its request is gone).
pub fn begin(l: anytype, lesson: Lesson) !void {
    for (l.extra.items) |ex| l.gpa.free(ex.ids);
    l.extra.clearRetainingCapacity();
    examples.disownLesson(l.gpa, l.owned);
    l.owned = .{};
    l.owned = try examples.ownLesson(l.gpa, lesson);
    l.gpa.free(l.answers);
    l.answers = try std.mem.concat(l.gpa, Example, &.{ l.owned.train, l.owned.think });
    switch (l.method) {
        .probe => if (comptime !editing(@TypeOf(l.*))) return error.EditsUnsupported else {
            try l.probe.begin(l.trainer.?, l.answers, l.owned.held);
            l.phase = .probe;
        },
        .pocket => try startPocket(l),
        .edit => if (comptime !editing(@TypeOf(l.*))) return error.EditsUnsupported else {
            try openEmpty(l);
            try startEdit(l);
        },
    }
}

/// Whether learner type L has the editor and the probe beside the pocket (Metal's trainer reads keys and codes).
fn editing(comptime L: type) bool {
    return @FieldType(L, "edits") != void;
}

/// A further round of the open lesson: more of its neurons' outputs (an edit is solved once, so it only reports).
pub fn more(l: anytype) !void {
    switch (l.method) {
        .pocket => {
            if (!l.pocketed) return error.NothingToContinue;
            l.pockets.more();
            l.phase = .pocket;
        },
        else => {
            if (!l.opened) return error.NothingToContinue;
            l.phase = .again;
        },
    }
}

/// The earlier lessons' questions a new lesson stays silent on, but those of facts a newer one replaced.
fn liveReplay(l: anytype) ![]const Example {
    l.live.clearRetainingCapacity();
    l.live_of.clearRetainingCapacity();
    const gone: []const u32 = if (@TypeOf(l.pockets) != void) l.pockets.retired_lessons.items else &.{};
    for (l.replay.items, l.replay_of.items) |ex, of| if (std.mem.indexOfScalar(u32, gone, of) == null) {
        try l.live.append(l.gpa, ex);
        try l.live_of.append(l.gpa, of);
    };
    return l.live.items;
}

fn startPocket(l: anytype) !void {
    l.stay = .{ l.owned.keep, l.owned.near, try liveReplay(l), l.owned.unasked, l.extra.items };
    l.pockets.lesson = l.last_lesson;
    l.pockets.replay_lessons = l.live_of.items;
    try l.pockets.begin(l.trainer.?, hooks(l), l.answers, l.owned.asked, l.owned.held, &l.stay);
    l.pocketed = true;
    l.edited = false;
    l.phase = .pocket;
}

/// One unit of the pocket's lesson; at its end, whether its held-out answers come back with its neurons in the forward.
pub fn pocketOnce(l: anytype) !@import("learner.zig").Step {
    if (!try l.pockets.step(l.trainer.?, hooks(l))) return .{ .done = false, .changed = true };
    l.phase = .idle;
    l.edited = l.pockets.recalled();
    l.kept_rounds += 1; // the server keeps it or takes it back (an undo, a drop) by what the model says when asked
    return .{ .done = true, .changed = true, .report = .{ .learned = .{ .recalled = l.edited, .steps = 1, .loss = 0 } } };
}

pub fn probeOnce(l: anytype) !@import("learner.zig").Step {
    if (comptime !editing(@TypeOf(l.*))) return error.EditsUnsupported;
    if (!try l.probe.step(l.trainer.?, l.keptLessons())) return .{ .done = false };
    l.phase = .idle;
    return .{ .done = true, .report = .{ .failed = "a probe: nothing was learned" } };
}

/// An answer the checks found moved: the lesson again from its own copy, that answer among what it keeps.
pub fn reroute(l: anytype) !@import("learner.zig").Step {
    for (l.lesson.kept) |ex| try l.extra.append(l.gpa, .{ .ids = try l.gpa.dupe(u32, ex.ids), .start = ex.start, .decide = ex.decide, .subject = ex.subject, .value = ex.value, .context = ex.context });
    l.lesson = l.owned; // the reroute's request holds only the moved answer
    switch (l.method) {
        .pocket => {
            try l.pockets.undo(l.trainer.?, hooks(l));
            try startPocket(l);
        },
        else => if (comptime !editing(@TypeOf(l.*))) return error.EditsUnsupported else {
            l.trainer.?.sites.restore();
            try @TypeOf(l.*).Host.attach(l.b, l.trainer.?);
            try startEdit(l);
        },
    }
    return .{ .done = false };
}

/// The open pocket lesson taken back: its neurons zero and free. False when the open lesson is not a pocket's.
pub fn undo(l: anytype) !bool {
    if (l.method != .pocket) return false;
    if (l.pocketed) try l.pockets.undo(l.trainer.?, hooks(l));
    l.pocketed = false;
    l.kept_rounds = 0;
    return true;
}

/// A replaced fact's neurons zeroed or put back, or a deleted one's (the count reported); the next save writes them.
pub fn retire(l: anytype, lesson: Lesson) !?@import("learner.zig").Step {
    if (l.method != .pocket) return null;
    const t = l.trainer.?;
    if (lesson.forget.len > 0 or lesson.unforget) {
        const n = if (lesson.unforget) try forgetting.unforget(&l.pockets, t, hooks(l)) else try forgetting.forget(&l.pockets, t, hooks(l), lesson.forget);
        return .{ .done = true, .changed = n > 0, .report = .{ .learned = .{ .recalled = n > 0, .steps = @intCast(n), .loss = 0 } } };
    }
    if (lesson.unretire) try l.pockets.unretire(t, hooks(l)) else try l.pockets.retire(t, hooks(l), lesson.retire);
    return .{ .done = true, .changed = true };
}

/// A put-back the device has not taken yet, tried again; it fails until the device takes every layer of it.
pub fn heal(l: anytype) !void {
    if (l.method != .pocket) return;
    if (l.pockets.saved_mismatch) return error.PocketSavedDifferently;
    if (l.pockets.save_pending) return error.PocketSaveUncertain;
    try healLive(l);
}

/// Explicit save retries may reconcile disk uncertainty; device restoration must still complete first.
pub fn healLive(l: anytype) !void {
    if (l.method != .pocket) return;
    if (l.pockets.retire_pending) try l.pockets.unretire(l.trainer.?, hooks(l));
    if (l.pockets.restore_pending) try l.pockets.undo(l.trainer.?, hooks(l));
    if (!forgetting.stale(&l.pockets)) return;
    _ = try forgetting.unforget(&l.pockets, l.trainer.?, hooks(l));
}

/// Whether the device lacks a put-back the pocket holds: replies are refused until `heal` lands it.
pub fn stale(l: anytype) bool {
    return l.method == .pocket and (l.pockets.save_pending or l.pockets.saved_mismatch or l.pockets.retire_pending or l.pockets.restore_pending or forgetting.stale(&l.pockets));
}

/// The pocket into the shards (an unrecalled lesson out first), held-out answers asked again; false if not a pocket.
pub fn save(l: anytype) !bool {
    if (l.method != .pocket) return false;
    const H = @TypeOf(l.*).Host;
    if (l.pocketed and l.kept_rounds == 0) {
        try l.pockets.undo(l.trainer.?, hooks(l));
        l.pocketed = false;
    }
    // a pocket a delete emptied still writes: the shards hold the deleted fact until it does
    const forgot = forgetting.pending(&l.pockets);
    if (l.pockets.empty() and !forgot and !l.pockets.save_pending) return true;
    const t0 = H.seconds(l.io);
    const bytes = try l.pockets.save(l.trainer.?, hooks(l));
    forgetting.settle(&l.pockets);
    if (forgot) {
        std.log.info("slide pocket: the learned neurons are in the model's shards ({d} MB written in {d:.1} s), a deleted fact's at zero", .{ bytes >> 20, H.seconds(l.io) - t0 });
        return true;
    }
    const back = try l.recalls(l.owned.held, false);
    std.log.info("slide pocket: the learned neurons are in the model's shards ({d} MB written in {d:.1} s); the last fact's held-out answers {s} with them as written", .{ bytes >> 20, H.seconds(l.io) - t0, if (back) "come back" else "do not all come back" });
    return true;
}

/// When the next lesson begins: a kept edit folded into the codes, a pocket lesson that never came back taken out.
pub fn settle(l: anytype) !void {
    if (l.method == .pocket) {
        if (l.pocketed and l.kept_rounds == 0) try l.pockets.undo(l.trainer.?, hooks(l));
        l.pocketed = false;
        return;
    }
    try foldOpen(l);
}

/// A new block, zero everywhere, for an edit to be written into; its state as now is what an undo comes back to.
fn openEmpty(l: anytype) !void {
    const t = l.trainer.?;
    const c = &l.choice.?;
    @memset(c.coef, 0);
    try t.sites.open(c.coef);
    l.opened = true;
    t.sites.needs()[t.sites.first() / adapters.block] = 0;
    try @TypeOf(l.*).Host.attach(l.b, t);
    t.sites.keep();
    l.can_undo = true;
}

/// The edit begun from the lesson's examples and what it keeps.
fn startEdit(l: anytype) !void {
    l.stay = .{ l.lesson.keep, l.lesson.near, l.replay.items, l.lesson.unasked, l.extra.items };
    if (!try l.edits.begin(&.{ l.lesson.train, l.lesson.think }, &.{ l.lesson.held, l.lesson.asked }, &l.stay)) return error.NoSubjectInQuestions;
    l.edited = false;
    l.phase = .edit;
}

/// One unit of the edit; at its end, whether the held questions come back (a lesson that does not is never kept).
pub fn editOnce(l: anytype) !@import("learner.zig").Step {
    if (comptime !editing(@TypeOf(l.*))) return error.EditsUnsupported;
    const t = l.trainer.?;
    var acting = l.keptLessons();
    acting.set(t.sites.first() / adapters.block);
    if (!try l.edits.step(t, acting)) return .{ .done = false, .changed = true };
    l.phase = .idle;
    l.edited = try l.recalled(true);
    l.kept_rounds = @intFromBool(l.edited);
    return .{ .done = true, .changed = true, .report = .{ .learned = .{ .recalled = l.edited, .steps = 1, .loss = l.edits.last_loss } } };
}

/// A kept edit into the model's 4-bit codes, live and in its shards, keys kept for later edits; then its block closes.
pub fn foldOpen(l: anytype) !void {
    if (comptime !editing(@TypeOf(l.*))) return;
    if (!(l.opened and l.kept_rounds > 0) or l.method != .edit) return;
    const H = @TypeOf(l.*).Host;
    const t = l.trainer.?;
    try l.edits.settle();
    const t0 = H.seconds(l.io);
    try l.edits.fold(t, hooks(l));
    t.sites.close();
    try H.attach(l.b, t);
    l.opened = false;
    l.can_undo = false;
    const back = try l.recalls(l.owned.held, false);
    std.log.info("slide: the edit is in the model's own codes, live and in its shards, in {d:.1} s; its held questions {s} with no block", .{ H.seconds(l.io) - t0, if (back) "come back" else "do not come back" });
}
