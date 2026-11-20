//! Replay GLP edits on rewritten inputs; keep uploaded directions until graph and executable teardown.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../check.zig");
const ref = @import("glp_ref.zig");
const rig = @import("glp_rig.zig");

const W = 1032;
const R = 5;
const world = 2;
const alpha: f32 = 0.75;

fn fillInputs(rng: *ref.Rng, x: []u16, g: []u16, d: []const f32) void {
    for (0..R) |r| ref.fillRow(.random, rng, r, x[r * 4 * W ..][0 .. 4 * W], d);
    for (g) |*v| v.* = ref.spread(rng);
}

pub fn run(c: rig.Ctx) !void {
    const a = c.gpu.gpa;
    var stream = try cuda.Stream.init(c.gpu.d, true); // a captured stream is a non-blocking one
    defer stream.deinit();
    const ops: rig.glp.Ops = .{ .f = c.ops.f, .s = stream };
    var rng = ref.Rng.init(0x6001);
    const d = try a.alloc(f32, W);
    defer a.free(d);
    ref.fillDir(.random, &rng, d);
    const x = try a.alloc(u16, R * 4 * W);
    defer a.free(x);
    const g = try a.alloc(u16, world * R * W);
    defer a.free(g);
    fillInputs(&rng, x, g, d);
    // the buffers outlive the graph: declared first, so freed last
    var db = try rig.Buf.init(c.gpu, rig.bytes(d));
    defer db.free();
    var xb = try rig.Buf.init(c.gpu, rig.bytes(x));
    defer xb.free();
    var gb = try rig.Buf.init(c.gpu, rig.bytes(g));
    defer gb.free();
    const dptr = db.ptr();

    try cuda.graph.beginCapture(stream, .thread_local);
    try rig.writer(ops, gb.ptr(), R * W, W, dptr, alpha, R, W, world);
    try rig.fold(ops, xb.ptr(), 4 * W, 0, 0, dptr, alpha, R, W);
    var captured = try cuda.graph.endCapture(stream);
    defer captured.deinit();
    if (try captured.nodeCount() != 2) return rig.fail("graph: the capture holds {d} nodes, expected the writer and the fold", .{try captured.nodeCount()});
    var exec = try captured.instantiate();
    defer exec.deinit();
    try stream.synchronize();
    const idle_x = try xb.read(a);
    defer a.free(idle_x);
    const idle_g = try gb.read(a);
    defer a.free(idle_g);
    if (!std.mem.eql(u8, idle_x, rig.bytes(x)) or !std.mem.eql(u8, idle_g, rig.bytes(g))) return rig.fail("graph: the capture itself ran a launch", .{});

    const ym = try a.alloc(u16, 4 * W);
    defer a.free(ym);
    const y64 = try a.alloc(f64, 4 * W);
    defer a.free(y64);
    const o16 = try a.alloc(u16, W);
    defer a.free(o16);
    for (0..3) |rep| {
        if (rep > 0) fillInputs(&rng, x, g, d); // new values for the same buffers, nothing in the graph updated
        try xb.write(rig.bytes(x));
        try gb.write(rig.bytes(g));
        try exec.launchOn(stream);
        try stream.synchronize();
        const gx = try xb.read(a);
        defer a.free(gx);
        const gg = try gb.read(a);
        defer a.free(gg);
        var t: ref.Tally = .{};
        for (0..R) |r| {
            const xin = x[r * 4 * W ..][0 .. 4 * W];
            ref.foldRow(xin, d, alpha, .{}, ym, null);
            ref.foldRef(xin, d, alpha, y64);
            const got = rig.words(gx)[r * 4 * W ..][0 .. 4 * W];
            for (0..4 * W) |i| t.add(r * 4 * W + i, got[i], ym[i], y64[i], ref.allow(y64[i], ref.widen(xin[i]), d[i % W], alpha));
            var ranks: [world][]const u16 = undefined;
            for (0..world) |rk| ranks[rk] = g[rk * R * W + r * W ..][0..W];
            ref.writerRow(world, &ranks, d, alpha, .{}, ym[0..W]);
            ref.writerRef(world, &ranks, d, alpha, o16, y64[0..W]);
            for (0..W) |i| {
                const slot0 = rig.words(gg)[r * W + i];
                const bound = ref.allow(y64[i], ref.widen(o16[i]), d[i], alpha);
                if (!ref.same(slot0, ym[i]) or !(@abs(@as(f64, ref.widen(slot0)) - y64[i]) <= bound)) t.model_diff += 1;
                if (rig.words(gg)[R * W + r * W + i] != 0x8000) t.model_diff += 1; // rank 1's slot reads -0.0
            }
        }
        if (!t.clean()) return rig.fail("graph replay {d}: {d} of {d} fold words and writer words off the model or FP64", .{ rep, t.bits(), t.n });
        check.pass("glp graph replay {d}: writer + fold of {d} rows from one captured graph, inputs rewritten, {d} fold words bit-equal to the model and within the FP64 bound", .{ rep, R, t.n });
    }
    if (!try xb.guards(a) or !try gb.guards(a) or !try db.guards(a)) return rig.fail("graph: a band around a buffer changed", .{});
    check.pass("glp graph lifetime: capture, instantiate, 3 replays, destroy (graph and executable before the buffers); direction uploaded once, never reallocated", .{});
}
