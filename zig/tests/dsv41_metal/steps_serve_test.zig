//! The served engine with prompts in steps (begin / piece / finish) through LaneHost: replies equal serial decode.
const std = @import("std");
const host = @import("dsv41");
const fs = @import("fixture_serve.zig");
const st = @import("serve_test.zig");

const api = host.api;
const Served = host.serve_lanes.Served;
const testing = std.testing;
const gpa = testing.allocator;
const io = testing.io;
const seed = st.seed;
const Box = st.Box;
const Ask = st.Ask;
const serve = st.serve;
const freeAll = st.freeAll;
const reference = st.reference;
const ids = st.ids;
const untagged = st.untagged;
const say = st.say;
const same = st.same;

/// Waits until `field` of the target's stats reaches `n` (read from LaneHost's thread's writes).
fn awaitStat(field: *const u64, n: u64) !void {
    for (0..60_000) |_| {
        if (@atomicLoad(u64, field, .monotonic) >= n) return;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

test "dsv41 metal serve in steps: (a) concurrent == solo == serial, drafted == draft:false, pieces between rounds" {
    const p0 = try ids(150, 1);
    defer gpa.free(p0);
    const p1 = try ids(620, 2);
    defer gpa.free(p1);
    const asks = [_]Ask{ .{ .prompt = p0 }, .{ .prompt = p1 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const m = try fs.open(gpa, io, seed, .{ .steps = true });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    const e = s.engine();
    var drafted: u64 = 0;
    var solo: [2][]u32 = undefined;
    try serve(e, &asks, false, &solo, &drafted);
    defer freeAll(&solo);
    var both: [2][]u32 = undefined;
    try serve(e, &asks, true, &both, &drafted);
    defer freeAll(&both);
    const plain_asks = [_]Ask{ .{ .prompt = p0, .drafts = false }, .{ .prompt = p1, .drafts = false } };
    var plain_drafted: u64 = 0;
    var plain: [2][]u32 = undefined;
    try serve(e, &plain_asks, true, &plain, &plain_drafted);
    defer freeAll(&plain);
    const bad = same(&want, &solo) + same(&want, &both) + same(&want, &plain);
    say(bad == 0 and m.t.stats.pieces > 0 and drafted > 0, "(a) in steps: {d} of 6 replies differ from serial; {d} pieces, {d} drafted rows", .{ bad, m.t.stats.pieces, drafted });
    try testing.expectEqual(@as(usize, 0), bad);
    try testing.expect(m.t.stats.pieces > 0 and drafted > 0 and plain_drafted == 0);
}

test "dsv41 metal serve in steps: (b) a short request during a long prompt: first in steps, after it as whole prompts" {
    const long = try ids(1100, 30);
    defer gpa.free(long);
    const short = try ids(100, 31);
    defer gpa.free(short);
    const asks = [_]Ask{ .{ .prompt = long, .max = 4 }, .{ .prompt = short, .max = 4 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    var bad: usize = 0;
    for ([_]bool{ false, true }) |steps| {
        const m = try fs.open(gpa, io, seed, .{ .slots = 2, .chunk = 16, .steps = steps });
        const s = try Served.open(gpa, io, m.model(), ".", null);
        defer Served.close(s);
        const e = s.engine();
        var boxes: [2]Box = @splat(.{});
        defer for (&boxes) |*b| b.deinit();
        const reqs = [2]api.Request{ .{ .prompt = long, .max_tokens = 4 }, .{ .prompt = short, .max_tokens = 4 } };
        try e.submit(600, &reqs[0], .{ .ctx = &boxes[0], .event = Box.event });
        // the short request arrives while the long prompt runs: two of its pieces in, or (whole) once it began
        if (steps) try awaitStat(&m.t.stats.pieces, 2) else try awaitStat(&m.t.stats.prompts, 1);
        try e.submit(601, &reqs[1], .{ .ctx = &boxes[1], .event = Box.event });
        for (&boxes) |*b| try b.wait();
        const short_first = boxes[1].first < boxes[0].first;
        const equal = std.mem.eql(u32, want[0], boxes[0].out.items) and std.mem.eql(u32, want[1], boxes[1].out.items);
        const ok = equal and short_first == steps;
        say(ok, "(b) {s}: the short reply's first token came {s} the long one's ({d} pieces); replies {s} serial", .{ if (steps) "in steps" else "whole prompts", if (short_first) "before" else "after", m.t.stats.pieces, if (equal) "equal" else "differ from" });
        bad += @intFromBool(!ok);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal serve in steps: (c) a request cancelled mid-prompt keeps nothing and the next requests == serial" {
    // no user or think tag, so no state short of the prompt is kept (Target.keepPoints)
    const long = try untagged(1100, 32);
    defer gpa.free(long);
    const p0 = try ids(120, 33);
    defer gpa.free(p0);
    const asks = [_]Ask{ .{ .prompt = p0, .max = 40 }, .{ .prompt = long } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const m = try fs.open(gpa, io, seed, .{ .slots = 2, .chunk = 16, .cache_bytes = 1 << 30, .steps = true });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    const e = s.engine();
    var bg: Box = .{};
    defer bg.deinit();
    const r0: api.Request = .{ .prompt = p0, .max_tokens = 40 };
    try e.submit(700, &r0, .{ .ctx = &bg, .event = Box.event });
    // the long prompt once the decoding reply is out, cancelled three of its pieces in
    for (0..60_000) |_| {
        bg.mutex.lockUncancelable(io);
        const out = bg.first != 0;
        bg.mutex.unlock(io);
        if (out) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    } else return error.Timeout;
    var gone: Box = .{};
    defer gone.deinit();
    const rl: api.Request = .{ .prompt = long, .max_tokens = 64 };
    const p_before = @atomicLoad(u64, &m.t.stats.pieces, .monotonic);
    try e.submit(701, &rl, .{ .ctx = &gone, .event = Box.event });
    try awaitStat(&m.t.stats.pieces, p_before + 3);
    e.cancel(701);
    try gone.wait();
    try bg.wait();
    var prefix_kept = false;
    for (m.t.cache.entries.items) |x| prefix_kept = prefix_kept or std.mem.startsWith(u32, long, x.ids);
    var drafted: u64 = 0;
    var again: [1][]u32 = undefined;
    try serve(e, asks[1..], false, &again, &drafted);
    defer freeAll(&again);
    const ok = gone.reason.? == .cancelled and m.t.stats.abandoned == 1 and std.mem.eql(u32, want[0], bg.out.items) and std.mem.eql(u32, want[1], again[0]);
    if (!ok) std.debug.print("     reason {t}, abandoned {d}, decoding reply {}, again {}\n", .{ gone.reason.?, m.t.stats.abandoned, std.mem.eql(u32, want[0], bg.out.items), std.mem.eql(u32, want[1], again[0]) });
    say(ok and !prefix_kept and st.cached_rows[0] == 0, "(c) cancelled after {d}+ pieces: {d} abandoned, a prefix of it kept: {}, the decoding reply and the prompt again (cached {d}) {s} serial", .{ 3, m.t.stats.abandoned, prefix_kept, st.cached_rows[0], if (ok) "equal" else "differ from" });
    try testing.expect(ok and !prefix_kept and st.cached_rows[0] == 0);
}

test "dsv41 metal serve in steps: (d) a decode share gives a decoding reply rounds between a prompt's pieces; replies == serial" {
    const long = try ids(1100, 34);
    defer gpa.free(long);
    const p0 = try ids(120, 35);
    defer gpa.free(p0);
    const asks = [_]Ask{ .{ .prompt = p0, .max = 300, .drafts = false }, .{ .prompt = long, .max = 8 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    var bad: usize = 0;
    // a share of 0 is one round a step (no debt); 0.75 owes replies three times each step's time (and caps its rows)
    for ([_]f64{ 0, 0.75 }) |share| {
        const m = try fs.open(gpa, io, seed, .{ .slots = 2, .chunk = 256, .steps = true, .decode_share = share });
        const s = try Served.open(gpa, io, m.model(), ".", null);
        defer Served.close(s);
        const e = s.engine();
        var boxes: [2]Box = @splat(.{});
        defer for (&boxes) |*b| b.deinit();
        const reqs = [2]api.Request{ .{ .prompt = p0, .max_tokens = 300, .drafts = false }, .{ .prompt = long, .max_tokens = 8 } };
        try e.submit(800, &reqs[0], .{ .ctx = &boxes[0], .event = Box.event });
        try awaitStat(&m.t.stats.windows, 1);
        try e.submit(801, &reqs[1], .{ .ctx = &boxes[1], .event = Box.event });
        for (&boxes) |*b| try b.wait();
        const between = m.t.stats.most_between;
        const equal = std.mem.eql(u32, want[0], boxes[0].out.items) and std.mem.eql(u32, want[1], boxes[1].out.items);
        const ok = equal and m.t.stats.pieces >= 2 and if (share > 0) between >= 3 else between <= 1;
        say(ok, "(d) decode share {d:.2}: at most {d} rounds between two of the prompt's {d} pieces; replies {s} serial", .{ share, between, m.t.stats.pieces, if (equal) "equal" else "differ from" });
        bad += @intFromBool(!ok);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "dsv41 metal serve in steps: (e) fcfs steps capped at 128 and 100 rows: replies == serial, each step on the chunk staged for it" {
    const long = try ids(1100, 36);
    defer gpa.free(long);
    const p0 = try ids(120, 37);
    defer gpa.free(p0);
    const asks = [_]Ask{ .{ .prompt = p0, .max = 300, .drafts = false }, .{ .prompt = long, .max = 8 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    _ = setenv("TENSORFOLD_PREFILL_POLICY", "fcfs", 1);
    defer _ = unsetenv("TENSORFOLD_PREFILL_POLICY");
    defer _ = unsetenv("TENSORFOLD_PREFILL_TOKENS");
    var bad: usize = 0;
    // a 256-row chunk: steps of half a chunk and of rows that divide none, while a reply decodes
    for ([_][:0]const u8{ "128", "100" }) |tokens| {
        _ = setenv("TENSORFOLD_PREFILL_TOKENS", tokens, 1);
        const m = try fs.open(gpa, io, seed, .{ .slots = 2, .chunk = 256, .steps = try host.serve_lanes.stepsFromEnv() });
        const s = try Served.open(gpa, io, m.model(), ".", null);
        defer Served.close(s);
        const e = s.engine();
        var boxes: [2]Box = @splat(.{});
        defer for (&boxes) |*b| b.deinit();
        const reqs = [2]api.Request{ .{ .prompt = p0, .max_tokens = 300, .drafts = false }, .{ .prompt = long, .max_tokens = 8 } };
        try e.submit(900, &reqs[0], .{ .ctx = &boxes[0], .event = Box.event });
        try awaitStat(&m.t.stats.windows, 1);
        try e.submit(901, &reqs[1], .{ .ctx = &boxes[1], .event = Box.event });
        for (&boxes) |*b| try b.wait();
        const equal = std.mem.eql(u32, want[0], boxes[0].out.items) and std.mem.eql(u32, want[1], boxes[1].out.items);
        const hits = m.t.pr.stage_hits;
        const ok = equal and m.t.stats.pieces >= 3 and hits + 1 >= m.t.stats.pieces;
        say(ok, "(e) fcfs, {s} rows a step: {d} pieces, {d} calls on a staged chunk; replies {s} serial", .{ tokens, m.t.stats.pieces, hits, if (equal) "equal" else "differ from" });
        bad += @intFromBool(!ok);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}
