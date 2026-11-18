//! Queued DSpark passes: exact replies and rollback, with each effective block and sampler.
const std = @import("std");
const testing = std.testing;
const pkg = @import("dsv41_metal");
const api = @import("dsv41").api;
const fs = @import("fixture_serve.zig");
const shared = @import("serve_test.zig");
const Ask = shared.Ask;
const Served = shared.Served;
const gpa = shared.gpa;
const io = shared.io;
const seed = shared.seed;
const serve = shared.serve;
const reference = shared.reference;
const ids = shared.ids;
const same = shared.same;
const say = shared.say;
const freeAll = shared.freeAll;

/// What a model's rounds came to: the target's windows and rows, the drafted rows of the replies.
const Rounds = struct { windows: u64, rows: u64, drafted: u64 };

/// The asks one after another, drafts behind the verify on or off (every `skew` chain parted); replies into `out`.
fn served(asks: []const Ask, early: bool, skew: u32, out: [][]u32, stats: ?*pkg.early.Stats) !Rounds {
    return servedWith(asks, .{ .early = early }, skew, out, stats);
}

fn servedWith(asks: []const Ask, o: fs.Opts, skew: u32, out: [][]u32, stats: ?*pkg.early.Stats) !Rounds {
    const m = try fs.open(gpa, io, seed, o);
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    if (m.early) |e| e.skew_every = skew;
    var drafted: u64 = 0;
    try serve(s.engine(), asks, false, out, &drafted);
    if (stats) |st| st.* = m.early.?.stats;
    return .{ .windows = m.t.stats.windows, .rows = m.t.stats.rows, .drafted = drafted };
}

test "dsv41 metal serve: (k) drafts behind the verify (kept, or dropped when the GPU's acceptance parts) == serial, same rounds" {
    const p0 = try ids(160, 30);
    defer gpa.free(p0);
    const p1 = try ids(240, 31);
    defer gpa.free(p1);
    const asks = [_]Ask{ .{ .prompt = p0, .max = 48 }, .{ .prompt = p1, .max = 48 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    var off: [2][]u32 = undefined;
    const r_off = try served(&asks, false, 0, &off, null);
    defer freeAll(&off);
    var on: [2][]u32 = undefined;
    var st_on: pkg.early.Stats = .{};
    const r_on = try served(&asks, true, 0, &on, &st_on);
    defer freeAll(&on);
    var sk: [2][]u32 = undefined;
    var st_sk: pkg.early.Stats = .{};
    const r_sk = try served(&asks, true, 2, &sk, &st_sk);
    defer freeAll(&sk);
    const bad = [3]usize{ same(&want, &off), same(&want, &on), same(&want, &sk) };
    const rounds_same = std.meta.eql(r_off, r_on) and std.meta.eql(r_off, r_sk);
    say(bad[0] + bad[1] + bad[2] == 0 and rounds_same, "(g) behind the verify off / on / parted vs serial: {d} / {d} / {d} differ; windows {d} / {d} / {d}, drafted rows {d} / {d} / {d}", .{ bad[0], bad[1], bad[2], r_off.windows, r_on.windows, r_sk.windows, r_off.drafted, r_on.drafted, r_sk.drafted });
    say(st_on.used > 0 and st_sk.dropped > 0, "(g) chains: {d} queued, {d} used; parted: {d} queued, {d} swapped, {d} dropped", .{ st_on.queued, st_on.used, st_sk.queued, st_sk.skewed, st_sk.dropped });
    try testing.expectEqual([3]usize{ 0, 0, 0 }, bad);
    try testing.expect(rounds_same);
    try testing.expect(st_on.used > 0 and st_on.dropped == 0 and st_sk.skewed > 0 and st_sk.dropped >= st_sk.skewed);
}

test "dsv41 metal serve: (h) sampled drafts behind the verify (keyed picks on the GPU, kept or dropped) == serial, same rounds" {
    const p0 = try ids(170, 32);
    defer gpa.free(p0);
    const p1 = try ids(230, 33);
    defer gpa.free(p1);
    const hot: api.Sampling = .{ .seed = 4321, .temperature = 0.9, .top_k = 0, .top_p = 1.0 };
    const whole: api.Sampling = .{ .seed = 1 << 62 | 77, .temperature = 1.0, .top_k = 0, .top_p = 1.0 };
    const asks = [_]Ask{ .{ .prompt = p0, .max = 48, .sampling = hot }, .{ .prompt = p1, .max = 48, .sampling = whole } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    var off: [2][]u32 = undefined;
    const r_off = try served(&asks, false, 0, &off, null);
    defer freeAll(&off);
    var on: [2][]u32 = undefined;
    var st_on: pkg.early.Stats = .{};
    const r_on = try served(&asks, true, 0, &on, &st_on);
    defer freeAll(&on);
    var sk: [2][]u32 = undefined;
    var st_sk: pkg.early.Stats = .{};
    const r_sk = try served(&asks, true, 2, &sk, &st_sk);
    defer freeAll(&sk);
    const bad = [3]usize{ same(&want, &off), same(&want, &on), same(&want, &sk) };
    const rounds_same = std.meta.eql(r_off, r_on) and std.meta.eql(r_off, r_sk);
    say(bad[0] + bad[1] + bad[2] == 0 and rounds_same, "(h) sampled behind the verify off / on / parted vs serial: {d} / {d} / {d} differ; windows {d} / {d} / {d}, drafted rows {d} / {d} / {d}", .{ bad[0], bad[1], bad[2], r_off.windows, r_on.windows, r_sk.windows, r_off.drafted, r_on.drafted, r_sk.drafted });
    say(st_on.keyed > 0 and st_on.used > 0, "(h) chains: {d} queued ({d} keyed), {d} used, {d} dropped; parted: {d} swapped, {d} dropped", .{ st_on.queued, st_on.keyed, st_on.used, st_on.dropped, st_sk.skewed, st_sk.dropped });
    try testing.expectEqual([3]usize{ 0, 0, 0 }, bad);
    try testing.expect(rounds_same);
    try testing.expect(st_on.keyed == st_on.queued and st_on.used > 0 and st_on.dropped == 0 and st_sk.skewed > 0 and st_sk.dropped >= st_sk.skewed);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "dsv41 metal serve: (i) a block of 7 past the drafter's 5, every draft verified, with and behind the verify == serial" {
    const p0 = try ids(150, 34);
    defer gpa.free(p0);
    const p1 = try ids(210, 35);
    defer gpa.free(p1);
    const hot: api.Sampling = .{ .seed = 999, .temperature = 0.9, .top_k = 0, .top_p = 1.0 };
    const asks = [_]Ask{ .{ .prompt = p0, .max = 40 }, .{ .prompt = p1, .max = 40, .sampling = hot } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    // the static depth verifies every draft of the block, so windows of 8 rows run
    _ = setenv("TF_DSV41_DEPTH", "static", 1);
    _ = setenv("TF_DSV41_DRAFT_DEPTH", "7", 1);
    defer {
        _ = unsetenv("TF_DSV41_DEPTH");
        _ = unsetenv("TF_DSV41_DRAFT_DEPTH");
    }
    var off: [2][]u32 = undefined;
    const r_off = try servedWith(&asks, .{ .early = false, .block = 7 }, 0, &off, null);
    defer freeAll(&off);
    var on: [2][]u32 = undefined;
    var st: pkg.early.Stats = .{};
    const r_on = try servedWith(&asks, .{ .early = true, .block = 7 }, 2, &on, &st);
    defer freeAll(&on);
    const bad = [2]usize{ same(&want, &off), same(&want, &on) };
    const deep = r_off.rows > 6 * r_off.windows;
    say(bad[0] + bad[1] == 0 and deep, "(i) block 7: drafted / behind the verify (parted) vs serial: {d} / {d} differ; {d} rows in {d} windows; chains {d} queued, {d} used, {d} dropped", .{ bad[0], bad[1], r_off.rows, r_off.windows, st.queued, st.used, st.dropped });
    try testing.expectEqual([2]usize{ 0, 0 }, bad);
    try testing.expect(deep and std.meta.eql(r_off, r_on) and st.used > 0 and st.dropped > 0);
}

test "dsv41 metal serve: (j) drafts on the GPU over the whole vocabulary (greedy, keyed), with and behind the verify == serial" {
    const p0 = try ids(160, 36);
    defer gpa.free(p0);
    const p1 = try ids(220, 37);
    defer gpa.free(p1);
    const hot: api.Sampling = .{ .seed = 31, .temperature = 0.9, .top_k = 0, .top_p = 1.0 };
    const asks = [_]Ask{ .{ .prompt = p0, .max = 40 }, .{ .prompt = p1, .max = 40, .sampling = hot } };
    const ref_m = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false, .rank = 32 });
    var want: [2][]u32 = undefined;
    for (&asks, &want) |k, *o| o.* = try ref_m.t.serial(0, k.prompt, k.max, k.sampling);
    fs.close(ref_m);
    defer freeAll(&want);
    var off: [2][]u32 = undefined;
    const r_off = try servedWith(&asks, .{ .early = false, .rank = 32, .gpu_chain = true }, 0, &off, null);
    defer freeAll(&off);
    // behind the verify the chains draw the same drafts as a pass after the commit: the same rounds
    var on: [2][]u32 = undefined;
    var st_on: pkg.early.Stats = .{};
    const r_on = try servedWith(&asks, .{ .early = true, .rank = 32, .gpu_chain = true }, 0, &on, &st_on);
    defer freeAll(&on);
    var sk: [2][]u32 = undefined;
    var st: pkg.early.Stats = .{};
    const r_sk = try servedWith(&asks, .{ .early = true, .rank = 32, .gpu_chain = true }, 2, &sk, &st);
    defer freeAll(&sk);
    const bad = [3]usize{ same(&want, &off), same(&want, &on), same(&want, &sk) };
    const rounds_same = std.meta.eql(r_off, r_on) and std.meta.eql(r_off, r_sk);
    say(bad[0] + bad[1] + bad[2] == 0 and rounds_same, "(j) GPU chains: drafted / behind the verify / parted vs serial: {d} / {d} / {d} differ; drafted rows {d} / {d} / {d}; chains {d} queued, {d} used, {d} dropped", .{ bad[0], bad[1], bad[2], r_off.drafted, r_on.drafted, r_sk.drafted, st.queued, st.used, st.dropped });
    try testing.expectEqual([3]usize{ 0, 0, 0 }, bad);
    try testing.expect(rounds_same and r_off.drafted > 0 and st_on.used > 0 and st_on.dropped == 0 and st.dropped > 0);
}

test "dsv41 metal serve: (k) a block of 7 after windows of 5 drafts, 5 after others: with and behind the verify == serial" {
    const p0 = try ids(150, 38);
    defer gpa.free(p0);
    const p1 = try ids(210, 39);
    defer gpa.free(p1);
    const hot: api.Sampling = .{ .seed = 555, .temperature = 0.9, .top_k = 0, .top_p = 1.0 };
    const asks = [_]Ask{ .{ .prompt = p0, .max = 40 }, .{ .prompt = p1, .max = 40, .sampling = hot } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    // the cost depth: mostly narrow passes; the static depth of 5 (narrow windows verify 5): wide passes after
    var bad: usize = 0;
    var same_rounds = true;
    for ([_]bool{ false, true }) |static| {
        if (static) {
            _ = setenv("TF_DSV41_DEPTH", "static", 1);
            _ = setenv("TF_DSV41_DRAFT_DEPTH", "7", 1);
        }
        defer if (static) {
            _ = unsetenv("TF_DSV41_DEPTH");
            _ = unsetenv("TF_DSV41_DRAFT_DEPTH");
        };
        var off: [2][]u32 = undefined;
        const r_off = try servedWith(&asks, .{ .early = false, .block = 7, .narrow = 5 }, 0, &off, null);
        defer freeAll(&off);
        var on: [2][]u32 = undefined;
        const r_on = try servedWith(&asks, .{ .early = true, .block = 7, .narrow = 5 }, 0, &on, null);
        defer freeAll(&on);
        var sk: [2][]u32 = undefined;
        _ = try servedWith(&asks, .{ .early = true, .block = 7, .narrow = 5 }, 2, &sk, null);
        defer freeAll(&sk);
        bad += same(&want, &off) + same(&want, &on) + same(&want, &sk);
        same_rounds = same_rounds and std.meta.eql(r_off, r_on);
        say(bad == 0 and same_rounds, "(k) narrow 5 / wide 7, {s} depth: {d} replies differ; {d} rows in {d} windows", .{ if (static) "static" else "cost", bad, r_off.rows, r_off.windows });
    }
    try testing.expectEqual(@as(usize, 0), bad);
    try testing.expect(same_rounds);
}
