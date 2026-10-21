//! The 27B lane backend's decision passes on an idle engine: the trunk's state kept once, each branch read over it.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lb = @import("qwen27_lanes.zig");

/// A queued decision's GPU work (LaneHost runs it alone): score reads the row after its prompt; decide, each branch's last row.
pub fn run(b: *lb.Metal, window: u32, task: *api.DecisionTask) error{ Failed, Refused }!void {
    switch (task.*) {
        .score => |*s| s.out = try score(b, s.prompt, s.labels, s.logits, s.mass),
        .decide => |d| try decide(b, window, d),
    }
}

/// `labels`' logits at the row after `prompt`: a prefill's `.last` head leaves them in logits row 0.
fn score(b: *lb.Metal, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
    const r = b.runner;
    if (prompt.len == 0 or prompt.len > r.capacity or logits.len < labels.len) return error.Failed;
    try reset(b);
    const session = q.session.Session{ .runner = r };
    session.prefill(prompt, b.chunk) catch |e| return failed("a score's prompt pass", e);
    return read(row0(r), labels, logits, mass);
}

/// Each branch's label logits at its last row after the shared trunk: the trunk's state is kept once
/// and restored a branch at a time; every branch's rows stand on the same trunk state.
fn decide(b: *lb.Metal, window: u32, d: *api.Decision) error{ Failed, Refused }!void {
    const r = b.runner;
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
    try reset(b);
    const session = q.session.Session{ .runner = r };
    session.prefill(d.trunk, b.chunk) catch |e| return failed("the trunk's prompt pass", e);
    const st = q.snapshot.saveSlot(b.gpa, r, 0, null, @intCast(d.trunk.len)) catch |e| return failed("keeping the trunk's state", e);
    defer q.snapshot.drop(b.gpa, st);
    d.read = d.trunk.len;
    var at: usize = 0;
    for (d.branches, d.counts) |br, n| {
        q.snapshot.restoreSlot(b.gpa, r, 0, null, st) catch |e| return failed("restoring the trunk's state", e);
        session.prefill(br, b.chunk) catch |e| return failed("a branch's prompt pass", e);
        _ = try read(row0(r), d.labels[at..][0..n], d.logits[at..][0..n], false);
        d.read += br.len;
        at += n;
    }
    reset(b) catch {}; // slot 0 stands empty for the next stream, whatever its prompt was
}

/// Slot 0 empty for a pass: a pending keep ends first, and the drafter's reset covers its session too.
fn reset(b: *lb.Metal) error{Failed}!void {
    b.settle() catch {};
    b.dropRound();
    b.runner.rope[0] = .{};
    b.runner.images = null;
    if (b.draft) |d| d.reset() catch |e| return failed("resetting slot 0", e);
    return b.runner.reset(0) catch |e| failed("resetting slot 0", e);
}

/// The last pass's logits row 0: bf16, `vocab` wide; the `.last` head leaves its one row there.
fn row0(r: *const q.decode_round.Runner) []const u16 {
    const logits = r.model.frame.get(.logits);
    const words: [*]const u16 = @ptrCast(@alignCast(logits.buffer.contents() + logits.offset));
    return words[0..r.model.config.vocab];
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
