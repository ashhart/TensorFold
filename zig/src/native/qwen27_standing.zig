//! Slot 0's standing state: whether a stream resumes there, what it leaves, and its rewind point.
const std = @import("std");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const lb = @import("qwen27_lanes.zig");
const resume_rules = @import("qwen27_resume.zig");
const Metal = lb.Metal;

/// Whether slot 0 resumes in place: on, and without --vision (an image prompt's rows go by its images, not its ids).
pub fn slotResume(b: *const Metal) bool {
    return b.slot_resume and b.vision == null;
}

/// A stream the prompt cache would look up: no images, no prompt logprobs (they start at 0), taught rows.
pub fn plain(s: *const lanes.Stream) bool {
    return s.images.len == 0 and s.prompt_logprobs == null and !s.untaught and s.embedding == null and resume_rules.unsalted(s.cache_salt);
}

/// Whether the cache keeps the state at the end of `s`'s prompt (a background pass's: the cache planned that mark).
pub fn keepsEnd(s: *const lanes.Stream) bool {
    return std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(s.prompt().len)) != null;
}

/// Slot 0 and its drafter stand at `at`, with no failure or round pending.
fn standsAt(b: *const Metal, at: usize) bool {
    if (b.runner.failed or b.runner.active != null or b.runner.offsets[0] != at) return false;
    const d = b.draft orelse return true;
    return !d.model.backend.failed and d.model.committed_end == at;
}

/// How `s` resumes from what slot 0 stands at, past the cache's restore point `kept_at` (qwen27_resume.zig).
pub fn resumable(b: *const Metal, s: *const lanes.Stream, kept_at: u32) resume_rules.Resume {
    if (!slotResume(b) or !plain(s) or !standsAt(b, b.standing.tokens.len)) return .none;
    return b.standing.plan(s.prompt(), s.decode_spans, kept_at);
}

/// Slot 0's position for `s` when it resumes there (0: not); the record is used up, its point and origin carried on.
pub fn takeStanding(b: *Metal, s: *lanes.Stream, kept_at: u32) !u32 {
    const plan = resumable(b, s, kept_at);
    b.standing.take(b.gpa, plan);
    errdefer b.standing = b.standing.settings(); // tokens and spans freed by take: the point and reply go too
    // rows [0, rewind_at) and the buffer hold only while slot 0 just extends past it: a restore, reset or move drops it
    const at: u32 = switch (plan) {
        .none => return 0,
        .here => |n| here: {
            if (b.draft) |d| {
                try d.model.reset(); // a new session over the kept taps, as after a restore
                d.model.committed_end = n;
            }
            break :here n;
        },
        .back => |h| back: {
            const buf = b.rewind orelse return dropRewind(b);
            if (b.draft) |d| try d.model.reset();
            q.snapshot.rewindSlot(b.gpa, b.runner, 0, b.ringOf(0), h, buf) catch return dropRewind(b); // starts at 0
            break :back h;
        },
    };
    resume_rules.inherit(&s.reuse, plan, b.standing.remote); // here or back: its rows' origin, not a pull it skipped
    return at;
}

/// A pass that could not rewind slot 0 starts it afresh, keeping no rewind point (0: resumed nowhere).
fn dropRewind(b: *Metal) u32 {
    b.standing.rewind_at = 0;
    b.standing.remote = false;
    return 0;
}

/// How far slot 0 would resume `prompt` in place, for the lane host's remote prefill (0: no resume); engine thread.
pub fn resumeAt(ptr: *anyopaque, prompt: []const u32, spans: []const [2]u32) u32 {
    const b: *Metal = @ptrCast(@alignCast(ptr));
    if (!slotResume(b) or b.owners[0] != null or !standsAt(b, b.standing.tokens.len)) return 0;
    return switch (b.standing.plan(prompt, spans, 0)) {
        .none => 0,
        .here => |n| n,
        .back => |at| if (b.rewind != null) at else 0,
    };
}

/// A stream leaves slot 0: what it stands at stays for the next prompt that extends it, if a fresh pass stands there.
pub fn leave(b: *Metal, s: *lanes.Stream) void {
    var at = b.runner.offsets[0];
    if (!slotResume(b) or !plain(s) or !standsAt(b, at)) return b.standing.deinit(b.gpa);
    var landed = s.fill_at.load(.monotonic);
    if (s.reason == .cancelled or s.isCancelled()) {
        const point = resume_rules.cancelPoint(at, landed, b.standing.rewind_at);
        if (point > 0 and point < at) {
            const buf = b.rewind orelse return b.standing.deinit(b.gpa);
            if (b.draft) |d| d.model.reset() catch return b.standing.deinit(b.gpa);
            q.snapshot.rewindSlot(b.gpa, b.runner, 0, b.ringOf(0), point, buf) catch return b.standing.deinit(b.gpa);
            at = point;
            landed = point;
        }
        s.keepInterruptedPrefill(at);
    }
    b.standing.leftLanded(b.gpa, s, at, landed, true); // a hand-off's rows too, with TF_PREFILL_KEEP
}

/// Where slot 0's pass from `at` keeps its rewind point: its history's end (0: none, or it resumed in a `replay`).
pub fn rewindPoint(b: *const Metal, s: *const lanes.Stream, slot: u32, at: usize, replay: resume_rules.Replay) u32 {
    if (slot != 0 or b.rewind == null or !slotResume(b) or !plain(s) or replay.holds(at)) return 0;
    return b.standing.point(s, at);
}
