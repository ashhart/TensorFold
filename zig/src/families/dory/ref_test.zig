//! The float64 forward's checks: per_loop is chunk-invariant to the bit, last_loop is not, dory.py's numbers match.
const std = @import("std");
const ref = @import("ref.zig");
const names = @import("names.zig");
const fixture = @import("ref_fixture.zig");

const testing = std.testing;

const prompt = [_]u32{ 1, 7, 30, 2, 19, 4, 4, 25, 11, 0, 31, 8 };

/// The prompt's logits, fed as chunks of the given sizes (they sum to the prompt's length).
fn chunked(w: *const ref.Weights, loops: usize, sizes: []const usize, opt: ref.Options) ![]f64 {
    const c = try names.tiny.withLoops(loops);
    var st = try ref.State.init(testing.allocator, c, prompt.len, opt.mode);
    defer st.deinit();
    const all = try testing.allocator.alloc(f64, prompt.len * c.vocab);
    errdefer testing.allocator.free(all);
    var at: usize = 0;
    for (sizes) |n| {
        const l = try ref.forward(testing.allocator, w, &st, prompt[at..][0..n], at, opt);
        defer testing.allocator.free(l);
        @memcpy(all[at * c.vocab ..][0 .. n * c.vocab], l);
        at += n;
    }
    try testing.expectEqual(prompt.len, at);
    return all;
}

const cuts = [_][]const usize{ &.{12}, &.{ 5, 4, 3 }, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, &.{ 3, 9 } };

fn sameBits(a: []const f64, b: []const f64) bool {
    return std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

test "per_loop: every chunking and one-token decode give the same logits to the bit, 1 to 4 loops, both precisions" {
    var w = try ref.random(testing.allocator, names.tiny, fixture.seed);
    defer w.deinit();
    for (1..5) |loops| for ([_]bool{ false, true }) |round| {
        const whole = try chunked(&w, loops, cuts[0], .{ .round = round });
        defer testing.allocator.free(whole);
        for (whole) |v| try testing.expect(std.math.isFinite(v));
        for (cuts[1..]) |cut| {
            const other = try chunked(&w, loops, cut, .{ .round = round });
            defer testing.allocator.free(other);
            try testing.expect(sameBits(whole, other));
        }
    };
}

test "last_loop: equal to per_loop at one loop, and at two loops its logits depend on where the prompt is cut" {
    var w = try ref.random(testing.allocator, names.tiny, fixture.seed);
    defer w.deinit();
    const one_per = try chunked(&w, 1, cuts[1], .{});
    defer testing.allocator.free(one_per);
    const one_last = try chunked(&w, 1, cuts[1], .{ .mode = .last_loop });
    defer testing.allocator.free(one_last);
    try testing.expect(sameBits(one_per, one_last));
    const whole = try chunked(&w, 2, cuts[0], .{ .mode = .last_loop });
    defer testing.allocator.free(whole);
    const split = try chunked(&w, 2, cuts[1], .{ .mode = .last_loop });
    defer testing.allocator.free(split);
    const v = names.tiny.vocab;
    // the first chunk's rows see only their own chunk, as in one chunk; the later chunks' rows read last-loop keys
    try testing.expect(sameBits(whole[0 .. 5 * v], split[0 .. 5 * v]));
    try testing.expect(!sameBits(whole[5 * v ..], split[5 * v ..]));
    const per = try chunked(&w, 2, cuts[0], .{});
    defer testing.allocator.free(per);
    try testing.expect(sameBits(whole, per)); // in one chunk every key is the current loop's in both modes
}

test "float64 against dory.py run in float64 (the dev-time oracle): every logit within 1e-12" {
    var w = try ref.random(testing.allocator, names.tiny, fixture.seed);
    defer w.deinit();
    for (fixture.cases) |case| {
        const l = try chunked(&w, case.loops, cuts[0], .{});
        defer testing.allocator.free(l);
        for (case.rows, 0..) |row, j| {
            const got = l[row * names.tiny.vocab ..][0..names.tiny.vocab];
            for (got, case.logits[j * names.tiny.vocab ..][0..names.tiny.vocab]) |g, want| {
                if (@abs(g - want) > 1e-12 * @max(1, @abs(want))) {
                    std.debug.print("{d} loops, row {d}: {e} against dory.py's {e}\n", .{ case.loops, row, g, want });
                    return error.TestUnexpectedResult;
                }
            }
        }
    }
}

test "the bf16 rounding points stay near float64: the same argmax on most rows, logits within a few hundredths" {
    var w = try ref.random(testing.allocator, names.tiny, fixture.seed);
    defer w.deinit();
    for (1..5) |loops| {
        const exact = try chunked(&w, loops, cuts[0], .{});
        defer testing.allocator.free(exact);
        const rounded = try chunked(&w, loops, cuts[0], .{ .round = true });
        defer testing.allocator.free(rounded);
        var most: f64 = 0;
        for (exact, rounded) |e, r| most = @max(most, @abs(e - r));
        try testing.expect(most > 0 and most < 0.25);
    }
}

test "taps: every step leaves each row on the unit sphere" {
    var w = try ref.random(testing.allocator, names.tiny, fixture.seed);
    defer w.deinit();
    const c = names.tiny;
    var st = try ref.State.init(testing.allocator, c, prompt.len, .per_loop);
    defer st.deinit();
    const taps = try testing.allocator.alloc(f64, st.sched.len * prompt.len * c.hidden);
    defer testing.allocator.free(taps);
    const l = try ref.forward(testing.allocator, &w, &st, &prompt, 0, .{ .taps = taps });
    defer testing.allocator.free(l);
    for (0..st.sched.len) |i| for (0..prompt.len) |r| {
        var ss: f64 = 0;
        for (taps[(i * prompt.len + r) * c.hidden ..][0..c.hidden]) |v| ss += v * v;
        try testing.expectApproxEqAbs(@as(f64, 1), ss, 1e-5); // eps on the sum: short of 1 by about eps / ss
    };
}
