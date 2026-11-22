//! Inkling 2 Mini's admission: the streams a Mac serves by what weights, scratch and each stream's state need.
const std = @import("std");

/// What a Mini serve holds: the fixed part, the sampling scratch, and the state bytes of one stream.
pub const Demand = struct {
    /// Weights, dense layers and scratch, the image tower and the DSpark head: held whatever the stream count.
    fixed: u64,
    /// The sampling scratch the target allocates once.
    sampling: u64,
    /// One stream: KV rows and history rings of every visit at the configured window, logits and draft taps.
    stream: u64,

    /// Bytes `n` streams hold with the fixed part.
    pub fn of(d: Demand, n: u32) u64 {
        return d.fixed + d.sampling + d.stream * n;
    }
};

/// What the load limit leaves for streams: its room before the load past the fixed part or after it, less sampling.
pub fn left(room_before: u64, room_after: u64, d: Demand) u64 {
    return @min(room_after, room_before -| d.fixed) -| d.sampling;
}

/// The streams served from `room`, or why not.
pub const Verdict = union(enum) {
    /// This many streams: `--parallel` at most, or what fits when the count was not named.
    serve: u32,
    /// Not one stream fits.
    none,
    /// `--parallel` was named and this many fit, fewer.
    short: u32,
};

/// The count: whole streams of `d.stream` bytes in `room`, `asked` at most; a named count must fit whole.
pub fn verdict(room: u64, d: Demand, asked: u32, named: bool) Verdict {
    const fit: u32 = @intCast(@min(room / @max(d.stream, 1), std.math.maxInt(u32)));
    if (fit == 0 or asked == 0) return .none;
    if (named and fit < asked) return .{ .short = fit };
    return .{ .serve = @min(asked, fit) };
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// A refusal's text: what the asked streams need against what the load limit leaves, and the count that fits.
pub fn refusal(a: std.mem.Allocator, v: Verdict, d: Demand, room: u64, asked: u32, window: u64) ![]const u8 {
    const base = "Inkling 2 Mini: past {d:.2} GiB of weights, dense layers and scratch the load limit leaves {d:.2} GiB, and a stream at a {d}-token window takes {d:.2} GiB";
    const head = .{ gib(d.fixed + d.sampling), gib(room), window, gib(d.stream) };
    return switch (v) {
        .serve => unreachable,
        .none => std.fmt.allocPrint(a, base ++ ", so none fits: lower --context or free memory", head),
        .short => |fit| std.fmt.allocPrint(a, base ++ "; --parallel {d} needs {d:.2} GiB, {d} fit: serve fewer or lower --context", head ++ .{ asked, gib(d.stream * asked), fit }),
    };
}
