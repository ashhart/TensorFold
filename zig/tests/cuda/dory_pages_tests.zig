//! Dory's full caches in pages against contiguous caches: every row's fp32 logits bit-equal, then served replies equal.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const dory = @import("dory");
const check = @import("check.zig");
const serve = @import("dory_serve_tests.zig");
const Rig = @import("dory_tests.zig").Rig;
const Gpu = check.Gpu;

const fwd = dory.cuda_model;
const be = dory.cuda_backend;

const slots = 8;
/// Prompt lengths a slot takes in each wave: past a chunk, on and around a page edge, past a window.
const waves = [_][4]usize{ .{ 2600, 257, 1500, 3100 }, .{ 255, 256, 1024, 2049 } };

/// fp32 logits of the rows [head0, head0 + heads) of `m`'s last forward, on the host.
fn logits(gpu: Gpu, m: *const fwd.Model, heads: usize) ![]f32 {
    const out = try gpu.gpa.alloc(f32, heads * m.c.vocab);
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = m.s.logits, .len = out.len * 4, .borrowed = true }).download(0, std.mem.sliceAsBytes(out));
    return out;
}

/// The same chains through both models: false when any row's logits differ at a bit.
fn both(x: *Rig, b: *fwd.Model, chains: []const fwd.Chain, prompt: bool, head0: usize, heads: usize, what: []const u8, bad: *usize) !void {
    for ([_]*fwd.Model{ &x.m, b }) |m| try m.forward(&x.k, x.s, chains, prompt, head0, heads);
    try x.s.synchronize();
    const gpa = x.gpu.gpa;
    const l0 = try logits(x.gpu, &x.m, heads);
    defer gpa.free(l0);
    const l1 = try logits(x.gpu, b, heads);
    defer gpa.free(l1);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(l0), std.mem.sliceAsBytes(l1))) {
        bad.* += 1;
        if (bad.* < 12) std.debug.print("FAIL dory pages: {s} (slot {d} at {d}, {d} chains): {d} rows' logits differ from contiguous\n", .{ what, chains[0].slot, chains[0].p0, chains.len, heads });
    }
}

/// tf-cuda-test dory-pages MODEL LOOPS TEXT N: lock-step rows over 4 slots for N rounds, then lane-core replies.
pub fn pages(gpu: Gpu, rest: []const []const u8) !void {
    if (rest.len < 4) return error.MissingArguments;
    const gpa = gpu.gpa;
    const loops = try std.fmt.parseInt(usize, rest[1], 10);
    const count = try std.fmt.parseInt(u32, rest[3], 10);
    const ids = try serve.tokens(gpu, rest[0], rest[2], 4000);
    defer gpa.free(ids);
    const context = try be.capacity(4000 + count);
    var a: Rig = undefined;
    try a.initMode(gpu, rest[0], loops, context, slots, false);
    defer a.deinit();
    var b = try fwd.Model.initPaged(gpa, gpu.d, a.s, &a.w, &a.k, context);
    defer b.deinit();
    const per = dory.cuda_pages.pagesOf(context);
    try b.attach(a.s, slots - 1, slots * per);
    var mapped: [slots][]u32 = undefined; // slots 0..3 on shuffled runs of the pool; the rest idle
    const all = try gpa.alloc(u32, slots * per);
    defer gpa.free(all);
    for (all, 0..) |*id, i| id.* = @intCast(i);
    var prng = std.Random.DefaultPrng.init(3);
    prng.random().shuffle(u32, all);
    for (&mapped, 0..) |*mp, i| mp.* = all[i * per ..][0..per];
    for (0..4) |s| try b.mapPages(a.s, s, mapped[s], 0);
    var bad: usize = 0;
    var rows_checked: usize = 0;
    for (waves, 0..) |wave, wv| {
        var pos: [4]usize = wave;
        for (wave, 0..) |len, s| {
            var at: usize = 0;
            while (at < len) : (at += fwd.prompt_rows) {
                const end = @min(len, at + fwd.prompt_rows);
                const heads = @min(16, end - at);
                try both(&a, &b, &.{.{ .slot = s, .p0 = at, .tokens = ids[at..end] }}, true, end - at - heads, heads, "prompt chunk", &bad);
                rows_checked += heads;
            }
        }
        for (0..count) |r| {
            var chains: [4]fwd.Chain = undefined;
            var rows: usize = 0;
            for (&chains, 0..) |*ch, s| {
                const w = 1 + (r * 7 + s * 3 + wv) % fwd.window_rows;
                const from = (pos[s] * 13 + r) % 3000;
                ch.* = .{ .slot = s, .p0 = pos[s], .tokens = ids[from..][0..w] };
                pos[s] += w;
                rows += w;
            }
            try both(&a, &b, &chains, false, 0, rows, "decode round", &bad);
            rows_checked += rows;
        }
    }
    std.debug.print("RESULT dory L={d} paged == contiguous: {d} rows' fp32 logits over 4 slots (prompts to 3100, {d} rounds of 1-16 rows, page edges at 255/256/257), {d} rounds differ\n", .{ loops, rows_checked, 2 * count, bad });
    try check.expect(bad == 0, "dory L={d} pages: {d} forwards differ from contiguous", .{ loops, bad });
    var da = try be.Dory.init(gpa, gpu.d, &a.m, &a.k, a.s);
    defer da.deinit();
    var db = try be.Dory.init(gpa, gpu.d, &b, &a.k, a.s);
    defer db.deinit();
    try db.attach(slots - 1, slots * per);
    db.shufflePages(9);
    var buf: [8]serve.Case = undefined;
    const rep = try gpa.alloc(u32, 1800);
    defer gpa.free(rep);
    const cases = serve.mixed(ids, &buf, rep);
    var differ: usize = 0;
    for ([_]bool{ false, true }) |drafts| for ([_]usize{ 1, 3, 8 }) |k| {
        var cs: [8]serve.Case = undefined;
        for (cs[0..k], cases[0..k]) |*o, c| {
            o.* = c;
            o.drafts = drafts;
        }
        const ra = try serve.lanesRun(gpu, &da, cs[0..k], count);
        defer serve.freeRun(gpa, ra);
        const rb = try serve.lanesRun(gpu, &db, cs[0..k], count);
        defer serve.freeRun(gpa, rb);
        var n: usize = 0;
        for (ra.out, rb.out) |oa, ob| n += @intFromBool(!std.mem.eql(u32, oa, ob));
        differ += n;
        std.debug.print("RESULT dory L={d} paged == contiguous replies: {d} streams{s}, {d} of {d} differ ({d} tokens)\n", .{ loops, k, if (drafts) " drafted" else "", n, k, ra.tokens });
    };
    const pool = db.pool.?;
    try check.expect(differ == 0 and pool.idle(), "dory L={d} pages: {d} replies differ or the pool is not idle", .{ loops, differ });
    check.pass("dory L={d} pages: paged == contiguous, logits bit-equal and replies equal (greedy, sampled, drafted, 1/3/8 streams); pool idle", .{loops});
}
