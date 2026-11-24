//! TF_GEMMA4_ARGMAX_GRAPH, on unless 0: a greedy round's argmax in the target graph, no unread tok, pinned uploads.
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("cuda_forward.zig");
const Pick = cuda.tree_round.Pick;

/// The switch; unset or blank is on, 0 off. The startup check refused every malformed value, so a plain default stands.
pub fn on() bool {
    return @import("sys").environment.read(bool, "TF_GEMMA4_ARGMAX_GRAPH", true) catch true;
}

/// A forward's ids, positions and slots staged in pinned memory, reused only once its last copy is done.
pub const Lean = struct {
    host: cuda.HostBuffer,
    sent: cuda.Event,
    pending: bool = false,

    pub fn deinit(l: *Lean) void {
        l.sent.synchronize() catch {};
        l.sent.deinit();
        l.host.free();
    }

    /// `n` i32 words of the staging, the last forward's copies out of it first; null when it holds fewer.
    pub fn take(l: *Lean, n: usize) !?[]i32 {
        const words = l.host.slice(i32);
        if (n > words.len) return null;
        if (l.pending) try l.sent.synchronize();
        l.pending = false;
        return words[0..n];
    }

    /// The copies just queued from the staging: its next `take` waits for them.
    pub fn queued(l: *Lean, s: cuda.Stream) !void {
        try l.sent.record(s);
        l.pending = true;
    }
};

/// With the switch on, the model's staging: its widest forward's ids, positions and slots.
pub fn enable(d: *const cuda.Driver, m: *fwd.Model) !void {
    if (!on()) return;
    const words = 3 * @max(m.prompt_most, fwd.round_rows);
    var host = try cuda.HostBuffer.alloc(d, words * 4);
    errdefer host.free();
    m.lean = .{ .host = host, .sent = try cuda.Event.init(d, false) };
}

/// Whether a verify round's argmax goes into the target graph: the switch on, every window greedy and unmasked.
pub fn fold(x: anytype, windows: anytype) bool {
    if (x.m.lean == null) return false;
    for (windows) |w| {
        const l = x.lanes.getPtr(w.stream) orelse return false;
        if (l.sampled or w.masks.len != 0) return false;
    }
    return true;
}

/// Entries [at, at + out.len) as tokens, after the stream's work; a pick over non-finite logits is an error.
pub fn tokens(x: anytype, at: usize, out: []u32) !void {
    try x.s.synchronize();
    var picks: [fwd.round_rows]Pick = undefined;
    const greedy = x.greedy[at..][0..out.len];
    // a greedy entry's token is its pick: with the switch, entries that are all picks read no tok
    if (x.m.lean == null or std.mem.indexOfScalar(bool, greedy, false) != null) try x.tok.download(at * 4, std.mem.sliceAsBytes(out));
    try x.picks.download(at * @sizeOf(Pick), std.mem.sliceAsBytes(picks[0..out.len]));
    for (out, picks[0..out.len], greedy) |*t, p, g| if (g) {
        if (p.nonfinite != 0) return error.NonFiniteLogits;
        t.* = p.token;
    };
}

test {
    _ = @import("cuda_argmax_graph_test.zig");
}
