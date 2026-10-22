//! Decision passes on the 27B lane backend: a free-slot synthetic stream beside live rounds, or alone when idle.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const lb = @import("qwen27_lanes.zig");
const slots = @import("qwen27_slots.zig");
const joint = @import("qwen27_joint.zig");

/// The LaneHost's busy gate: a decision pass may take a slot beside the live streams when one is free.
pub fn ready(b: *const lb.Metal) bool {
    return b.freeSlot() != null;
}

/// A queued decision's GPU work: a synthetic stream on a free slot beside the rounds, or `error.Busy` to wait.
pub fn run(b: *lb.Metal, head: ?*joint.Head, window: u32, task: *api.DecisionTask) error{ Failed, Refused, Busy }!void {
    switch (task.*) {
        .score => |*s| {
            var stream = try synthetic(b, s.prompt);
            defer stream.deinit(b.gpa);
            const slot = try bind(b, &stream);
            defer b.unbind(&stream);
            s.out = try score(b, head, slot, s.prompt, s.labels, s.logits, s.mass);
        },
        .decide => |d| {
            try check(window, d);
            var stream = try synthetic(b, d.trunk);
            defer stream.deinit(b.gpa);
            const slot = try bind(b, &stream);
            defer b.unbind(&stream);
            try decide(b, head, slot, d);
        },
    }
}

/// The pass's synthetic stream: a slot's owner for the pass's rows; drafts off, it never generates.
fn synthetic(b: *lb.Metal, ids: []const u32) error{Failed}!lanes.Stream {
    return lanes.Stream.init(b.gpa, .{ .id = "decision", .prompt = ids, .max_new = 0, .drafts = false }) catch return error.Failed;
}

/// `s` bound to a free slot as a new stream's admission binds it; `b.unbind(s)` frees it (`error.Busy` = wait).
fn bind(b: *lb.Metal, s: *lanes.Stream) error{ Failed, Busy }!u32 {
    b.settle() catch {}; // a pending verify's rows are kept before the pass touches the shared frame
    b.dropRound();
    const slot = b.freeSlot() orelse return error.Busy;
    b.ready(slot, s) catch |e| return switch (e) {
        error.LaneMemory => error.Busy,
        else => failed("a decision's slot", e),
    };
    b.dropTreeOf(slot);
    b.owners[slot] = s;
    b.runner.rope[slot] = .{};
    b.runner.images = null;
    slots.recover(b, s) catch |e| return failed("recovering the device", e);
    if (b.drafter(slot)) |d| d.reset() catch |e| return failed("resetting a decision's slot", e) else b.runner.reset(slot) catch |e| return failed("resetting a decision's slot", e);
    slots.clearContext(b, slot) catch |e| return failed("clearing a decision's slot", e);
    return slot;
}

/// `labels`' logits at the row after `prompt` on `slot` (`.last` → logits row 0); a JointHead scores option-final rows.
fn score(b: *lb.Metal, head: ?*joint.Head, slot: u32, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
    const r = b.runner;
    if (prompt.len == 0 or prompt.len > r.capacity or logits.len < labels.len) return error.Failed;
    const session = q.session.Session{ .runner = r, .slot = slot };
    if (head) |h| {
        try forward(h, b, session, 0, prompt, labels, logits);
        return massOf(logits[0..labels.len], mass);
    }
    session.prefill(prompt, b.chunk) catch |e| return failed("a score's prompt pass", e);
    return read(row0(r), labels, logits, mass);
}

/// What a packed decision needs before it takes a slot (its refusal words reach the requester).
fn check(window: u32, d: *api.Decision) error{ Failed, Refused }!void {
    if (d.pictures.len > 0) return refuse(d, "decisions with pictures are not served by this engine yet");
    if (d.trunk.len == 0 or d.branches.len == 0 or d.counts.len != d.branches.len) return error.Failed;
    var wanted: usize = 0;
    for (d.branches, d.counts) |br, n| {
        if (br.len == 0) return error.Failed;
        if (d.trunk.len + br.len > window)
            return refuse(d, std.fmt.allocPrint(d.arena orelse return error.Failed, "a decision sequence exceeds the {d}-token context", .{window}) catch return error.Failed);
        wanted = std.math.add(usize, wanted, n) catch return error.Failed;
    }
    if (d.labels.len < wanted or d.logits.len < wanted) return error.Failed;
}

/// Each branch's label logits at its last row after the shared trunk on `slot`; a `head` reads option-final positions.
fn decide(b: *lb.Metal, head: ?*joint.Head, slot: u32, d: *api.Decision) error{Failed}!void {
    const r = b.runner;
    const session = q.session.Session{ .runner = r, .slot = slot };
    session.prefill(d.trunk, b.chunk) catch |e| return failed("the trunk's prompt pass", e);
    const st = q.snapshot.saveSlot(b.gpa, r, slot, null, @intCast(d.trunk.len)) catch |e| return failed("keeping the trunk's state", e);
    defer q.snapshot.drop(b.gpa, st);
    d.read = d.trunk.len;
    var at: usize = 0;
    for (d.branches, d.counts) |br, n| {
        q.snapshot.restoreSlot(b.gpa, r, slot, null, st) catch |e| return failed("restoring the trunk's state", e);
        if (head) |h| {
            try forward(h, b, session, d.trunk.len, br, d.labels[at..][0..n], d.logits[at..][0..n]);
        } else {
            session.prefill(br, b.chunk) catch |e| return failed("a branch's prompt pass", e);
            _ = try read(row0(r), d.labels[at..][0..n], d.logits[at..][0..n], false);
        }
        d.read += br.len;
        at += n;
    }
}

/// `ids` prefilled in chunks: each `positions` entry is an option's final-token position offset by `base`.
fn forward(h: *joint.Head, b: *lb.Metal, session: q.session.Session, base: usize, ids: []const u32, positions: []const u32, logits: []f64) error{Failed}!void {
    const r = b.runner;
    for (positions) |p| if (p < base or p - base >= ids.len) return error.Failed;
    const dim: usize = @intCast(r.model.config.hidden);
    var first: usize = 0;
    while (first < ids.len) {
        const end = @min(ids.len, first + @as(usize, r.promptRows(@intCast(b.chunk))));
        session.promptChunk(ids[first..end], false) catch |e| return failed("a decision's prompt pass", e);
        const words = hiddenRows(r);
        for (positions, logits) |p, *x| {
            if (p - base < first or p - base >= end) continue;
            x.* = h.score(words[(p - base - first) * dim ..][0..dim]) catch |e| return failed("a JointHead option", e);
        }
        first = end;
    }
}

/// The last pass's logits row 0: bf16, `vocab` wide; the `.last` head leaves its one row there.
fn row0(r: *const q.decode_round.Runner) []const u16 {
    const logits = r.model.frame.get(.logits);
    const words: [*]const u16 = @ptrCast(@alignCast(logits.buffer.contents() + logits.offset));
    return words[0..r.model.config.vocab];
}

/// The last pass's post-final-norm hidden rows: the norm the head's input reads leaves every row there.
fn hiddenRows(r: *const q.decode_round.Runner) [*]const u16 {
    const input = r.model.frame.get(.input);
    return @ptrCast(@alignCast(input.buffer.contents() + input.offset));
}

/// The row's `labels` logits out, then the vocabulary logsumexp when `mass` (else 0).
fn read(row: []const u16, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
    for (labels, logits) |label, *x| {
        if (label >= row.len) return error.Failed;
        x.* = try value(row[@intCast(label)]);
    }
    if (!mass) return 0;
    var peak: f64 = -std.math.inf(f64);
    for (row) |bits| peak = @max(peak, try value(bits));
    var total: f64 = 0;
    for (row) |bits| total += @exp(try value(bits) - peak);
    return peak + @log(total);
}

/// A JointHead pass has no vocabulary: the option scores' own logsumexp stands as the label mass (it reads 1).
fn massOf(logits: []const f64, mass: bool) f64 {
    if (!mass or logits.len == 0) return 0;
    var peak: f64 = -std.math.inf(f64);
    for (logits) |x| peak = @max(peak, x);
    var total: f64 = 0;
    for (logits) |x| total += @exp(x - peak);
    return peak + @log(total);
}

/// A bf16 logit as f64; a nonfinite one fails the pass, as a draw's argmax check does.
fn value(bits: u16) error{Failed}!f64 {
    const v: f64 = @as(f32, @bitCast(@as(u32, bits) << 16));
    return if (std.math.isFinite(v)) v else error.Failed;
}

/// A pass's error logged once, the decision hooks' Failed.
fn failed(what: []const u8, err: anyerror) error{Failed} {
    std.log.err("qwen27 decision pass failed ({s}): {s}", .{ what, @errorName(err) });
    return error.Failed;
}

/// The decision refused, in `problem`'s words for the requester.
fn refuse(d: *api.Decision, problem: []const u8) error{Refused} {
    d.problem = problem;
    return error.Refused;
}
