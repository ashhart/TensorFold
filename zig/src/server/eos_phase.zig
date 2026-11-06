//! A reply's end tokens under its grammar: with ignore_eos only an end token that the grammar takes ends the reply.
const std = @import("std");

/// The request's grammar at the next token: none, waiting for the token after `after` (thinking on), or on.
pub const Phase = union(enum) {
    none,
    after: u32,
    on,

    /// The phase at the token after `t`.
    pub fn next(p: Phase, t: u32) Phase {
        return switch (p) {
            .after => |end| if (t == end) .on else p,
            else => p,
        };
    }

    /// Whether `t`, at this phase, ends the reply: any end token, and with ignore_eos only one under the grammar.
    pub fn ends(p: Phase, ignore_eos: bool, eos: []const u32, t: u32) bool {
        return (!ignore_eos or p == .on) and std.mem.indexOfScalar(u32, eos, t) != null;
    }

    /// Whether the reply keeps a trailing end token as content, the negation of the condition `ends` applies.
    /// An end token the grammar takes ends the reply, so it is not content: the reply's content tokens, its
    /// logprob rows and its stream all read it the same way.
    pub fn keepsTrailing(p: Phase, ignore_eos: bool) bool {
        return ignore_eos and p != .on;
    }
};

/// A chunk's tokens that stream, and whether one of them ended the reply (it does not stream).
pub const Streamed = struct { tokens: []const u32, ended: bool };

/// A chunk's tokens that stream from `phase`: up to the first that ends the reply, each judged at its own phase.
pub fn streamed(phase: Phase, ignore_eos: bool, eos: []const u32, chunk: []const u32) Streamed {
    var p = phase;
    for (chunk, 0..) |t, i| {
        if (p.ends(ignore_eos, eos, t)) return .{ .tokens = chunk[0..i], .ended = true };
        p = p.next(t);
    }
    return .{ .tokens = chunk, .ended = false };
}

test "EOS policy: a chunk's end token is judged at its own grammar phase, so one before the grammar streams" {
    const eos = [_]u32{2};
    const close: u32 = 9; // `</think>`: the grammar takes the reply from the next token
    // thinking on, ignore_eos: an end token in the reasoning, the think end, the answer, then the grammar's stop
    const chunk = [_]u32{ 5, 2, close, 7, 2, 8 };
    const got = streamed(.{ .after = close }, true, &eos, &chunk);
    try std.testing.expect(got.ended);
    try std.testing.expectEqualSlices(u32, chunk[0..4], got.tokens);
    // the chunk's last phase judging every token would end the stream at the reasoning's end token instead
    var last: Phase = .{ .after = close };
    for (chunk) |t| last = last.next(t);
    try std.testing.expect(last == .on and last.ends(true, &eos, chunk[1]));
    const wrong = streamed(last, true, &eos, &chunk);
    try std.testing.expectEqualSlices(u32, chunk[0..1], wrong.tokens);
}

test "EOS policy: the content tokens keep a trailing end token exactly when it does not end the reply" {
    const eos = [_]u32{ 2, 3 };
    const t: u32 = 2; // an end token
    // keepsTrailing is the negation of the condition ends applies, at every phase and either ignore_eos.
    for ([_]Phase{ .none, .{ .after = 9 }, .on }) |p| {
        for ([_]bool{ false, true }) |ignore_eos| {
            try std.testing.expectEqual(!p.ends(ignore_eos, &eos, t), p.keepsTrailing(ignore_eos));
        }
    }
    // The case this pins: with ignore_eos, a grammar's own stop ends the reply, so it is not content.
    const on: Phase = .on;
    const none: Phase = .none;
    const waiting: Phase = .{ .after = 9 };
    try std.testing.expect(!on.keepsTrailing(true));
    try std.testing.expect(none.keepsTrailing(true));
    try std.testing.expect(waiting.keepsTrailing(true));
    // Without ignore_eos nothing keeps a trailing end token.
    for ([_]Phase{ .none, .{ .after = 9 }, .on }) |p| try std.testing.expect(!p.keepsTrailing(false));
}

test "EOS policy: without ignore_eos any end token ends the stream; without a grammar ignore_eos keeps them all" {
    const eos = [_]u32{ 2, 3 };
    const chunk = [_]u32{ 5, 3, 2 };
    for ([_]Phase{ .none, .{ .after = 9 }, .on }) |p| {
        const got = streamed(p, false, &eos, &chunk);
        try std.testing.expect(got.ended);
        try std.testing.expectEqualSlices(u32, chunk[0..1], got.tokens);
    }
    for ([_]Phase{ .none, .{ .after = 9 } }) |p| {
        const kept = streamed(p, true, &eos, &chunk);
        try std.testing.expect(!kept.ended);
        try std.testing.expectEqualSlices(u32, &chunk, kept.tokens);
    }
    const on = streamed(.on, true, &eos, &chunk);
    try std.testing.expect(on.ended);
    try std.testing.expectEqualSlices(u32, chunk[0..1], on.tokens);
}
