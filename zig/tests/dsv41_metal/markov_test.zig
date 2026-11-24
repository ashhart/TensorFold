//! DSpark's greedy chain on the GPU over the whole vocabulary against the host's reference (draft/dspark.chainFull).
const std = @import("std");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");
const fds = @import("fixture_ds.zig");

const testing = std.testing;
const limit = 1280;

extern "c" fn log(x: f64) f64;

/// A draft's score as the host's race gives it: z / T, plus the keyed Gumbel noise when sampled.
fn score(mk: host.dspark.Markov, row: []const f32, prev: u32, id: u32, p: host.dspark.Params, at: u64) f64 {
    const z: f64 = row[id] + mk.bias(prev, id);
    if (!p.sampled) return z;
    return z / @max(p.temperature, 1e-6) - log(-log(host.lanes.sampling.uniform(p.seed, at, id)));
}

test "dsv41 metal markov: GPU chains (1 and 3 slots, greedy and keyed) equal the host's over the whole vocabulary" {
    // rank 32 reads a value a lane at a time, the release's 256 in word quads
    for ([_]u32{ 32, 256 }) |rank| try ranked(rank);
}

fn ranked(rank: u32) !void {
    const a = testing.allocator;
    var c = fixture.dsparkConfig();
    c.dspark_markov_rank = rank;
    const ds = try fds.Ds.initWith(a, testing.io, c, limit, 81, 3, .{ .gpu_chain = true });
    defer ds.deinit();
    try testing.expect(ds.pass.mk != null);
    const n: usize = c.dspark_block;
    const V: usize = c.vocab;
    const D: usize = c.hidden;
    const mk = ds.pass.markov();
    var rows: usize = 0;
    var near: usize = 0;
    var bad: usize = 0;
    for ([_]u32{ 1, 3 }) |slots| for (0..4) |round| {
        var asks: [3]pkg.dspark.Ask = undefined;
        var outs: [3]fds.Out = undefined;
        var props: [3]pkg.dspark.Proposal = undefined;
        for (0..slots) |s| {
            const at: u64 = 300 + 37 * round + 11 * s;
            ds.pass.reset(@intCast(s));
            try ds.ingest(@intCast(s), at - 150, 150, 60 + round);
            // every other ask keyed as the host races a row (T 0.8, no top-k or nucleus)
            const keyed: ?host.dspark.Params = if ((s + round) % 2 == 1) .{ .seed = 900 + round, .pos0 = at + 1, .sampled = true, .temperature = 0.8 } else null;
            asks[s] = fds.Ds.ask(@intCast(s), @intCast((17 * round + 5 * s) % V), at, keyed);
            outs[s] = try fds.Out.init(a, &c);
            props[s] = .{ .drafts = outs[s].drafts, .conf = outs[s].conf };
        }
        defer for (outs[0..slots]) |o| o.deinit(a);
        try ds.pass.propose(asks[0..slots], props[0..slots]);
        const logits = ds.pass.b.logitsOf(&c, slots * n);
        const hf: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, ds.pass.b.bytes(ds.pass.b.hf, 2 * slots * n * D)));
        for (0..slots) |s| {
            var hid: [16 * 256]f32 = undefined;
            for (hid[0 .. n * D], hf[s * n * D ..][0 .. n * D]) |*h, v| h.* = host.dspark.bf16(v);
            var want: [16]u32 = undefined;
            var wconf: [16]f32 = undefined;
            try host.dspark.chainFull(a, mk, logits[s * n * V ..][0 .. n * V], V, asks[s].anchor, asks[s].params, hid[0 .. n * D], want[0..n], wconf[0..n]);
            for (0..n) |i| {
                rows += 1;
                if (@abs(outs[s].conf[i] - wconf[i]) > 1e-4) bad += 1;
                if (outs[s].drafts[i] == want[i]) continue;
                // a near-tie apart: the two picks' scores within float rounding of each other
                const prev = if (i == 0) asks[s].anchor else want[i - 1];
                const row = logits[(s * n + i) * V ..][0..V];
                const at = asks[s].params.pos0 + i;
                const sw = score(mk, row, prev, want[i], asks[s].params, at);
                const sg = score(mk, row, prev, outs[s].drafts[i], asks[s].params, at);
                if (@abs(sw - sg) <= 1e-4 * @max(1.0, @abs(sw))) near += 1 else bad += 1;
                break; // later drafts follow another token
            }
        }
    };
    std.debug.print("{s} dsv41 metal markov rank {d}: {d} GPU draft rows equal the host's whole-vocabulary chain, {d} near-ties apart, {d} wrong\n", .{ if (bad == 0) "PASS" else "FAIL", rank, rows - near, near, bad });
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal markov: top-K single-TG GPU chain matches host candidateRows and filtered sampling" {
    const a = testing.allocator;
    var c = fixture.dsparkConfig();
    c.dspark_markov_rank = 256;
    const K: usize = 64;
    const ds = try fds.Ds.initWith(a, testing.io, c, limit, 83, 3, .{ .gpu_chain = true, .markov_topk = K, .candidates = K });
    defer ds.deinit();
    try testing.expect(ds.pass.mk != null);
    const n: usize = c.dspark_block;
    const V: usize = c.vocab;
    const D: usize = c.hidden;
    const mk = ds.pass.markov();
    const params = [_]host.dspark.Params{
        .{ .seed = 101, .pos0 = 301, .sampled = true, .temperature = 0.8, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 202, .pos0 = 312, .sampled = true, .temperature = 0.8, .top_k = 0, .top_p = 0.95 },
        .{ .seed = 303, .pos0 = 323, .sampled = true, .temperature = 0.8, .top_k = 0, .top_p = 1.0, .min_p = 0.05 },
    };
    var asks: [3]pkg.dspark.Ask = undefined;
    var outs: [3]fds.Out = undefined;
    var props: [3]pkg.dspark.Proposal = undefined;
    const cand_got = try a.alloc(i32, 3 * n * K);
    defer a.free(cand_got);
    const base_got = try a.alloc(f32, 3 * n * K);
    defer a.free(base_got);
    for (0..3) |s| {
        const at: u64 = 300 + 11 * s;
        ds.pass.reset(@intCast(s));
        try ds.ingest(@intCast(s), at - 150, 150, 70 + s);
        asks[s] = fds.Ds.ask(@intCast(s), @intCast((23 * s + 7) % V), at, params[s]);
        outs[s] = try fds.Out.init(a, &c);
        props[s] = .{ .drafts = outs[s].drafts, .conf = outs[s].conf, .cand = cand_got[s * n * K ..][0 .. n * K], .base = base_got[s * n * K ..][0 .. n * K] };
    }
    defer for (outs) |o| o.deinit(a);
    try ds.pass.propose(&asks, &props);
    const logits = ds.pass.b.logitsOf(&c, 3 * n);
    const hf: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, ds.pass.b.bytes(ds.pass.b.hf, 2 * 3 * n * D)));
    const cand_want = try a.alloc(i32, 3 * n * K);
    defer a.free(cand_want);
    const base_want = try a.alloc(f32, 3 * n * K);
    defer a.free(base_want);
    try host.dspark.candidateRows(logits, V, K, cand_want, base_want);
    for (cand_want, cand_got) |w, g| try testing.expectEqual(w, g);
    for (base_want, base_got) |w, g| try testing.expectEqual(w, g);
    var bad: usize = 0;
    for (0..3) |s| {
        var hid: [16 * 256]f32 = undefined;
        for (hid[0 .. n * D], hf[s * n * D ..][0 .. n * D]) |*h, v| h.* = host.dspark.bf16(v);
        var want: [16]u32 = undefined;
        var wconf: [16]f32 = undefined;
        try host.dspark.chain(a, mk, cand_want[s * n * K ..][0 .. n * K], base_want[s * n * K ..][0 .. n * K], K, asks[s].anchor, asks[s].params, hid[0 .. n * D], want[0..n], wconf[0..n]);
        for (0..n) |i| {
            if (@abs(outs[s].conf[i] - wconf[i]) > 1e-4) bad += 1;
            if (outs[s].drafts[i] != want[i]) bad += 1;
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}
