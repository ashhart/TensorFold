//! flashnext-exl3-tpe: NCCL all-reduce inside a captured graph.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");

const Gpu = check.Gpu;
const prompt = flash.prompt;
const X = flash.exl3;
const xpass = X.pass;
const tp = X.tp;
const fpool = flash.exl3_pool;
const wire = flash.rank_wire;
const Collective = flash.collective.Collective;
const vocab = prompt.vocabulary;

const engine = @import("flashnext_exl3_tpe_engine.zig");
const cap_default = engine.cap_default;
const out_rows = engine.out_rows;
const promote = engine.promote;
const ms = engine.ms;
const gib = engine.gib;
const Ranks = engine.Ranks;
const Eng = engine.Eng;
const printPool = engine.printPool;
const Golden = engine.Golden;
const ints = engine.ints;
const num = engine.num;
const top1 = engine.top1;
const margin = engine.margin;

// -- NCCL in a captured graph ---------------------------------------------------------------------------------------

fn ready(c: *Collective) !void {
    while (true) {
        var state: cuda.nccl.Result = 0;
        try c.lib.check(c.api.ncclCommGetAsyncError(c.comm, &state), "ncclCommGetAsyncError");
        if (state == 0) return;
        if (state != 7) return c.lib.check(state, "NCCL async");
    }
}

fn allReduce(c: *Collective, s: cuda.Stream, src: u64, dst: u64, n: usize) !void {
    const r = c.lib.api.ncclAllReduce(src, dst, n, .f32, .sum, c.comm, s.handle);
    if (r != 0 and r != 7) return c.lib.check(r, "ncclAllReduce");
    try ready(c);
}

pub fn graphMode(gpu: Gpu, rank: usize, port: u16) !void {
    const a = gpu.gpa;
    const rk = try Ranks.open(gpu, rank, port);
    defer rk.close(a);
    const c = &rk.collective;
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    var all_ok = true;
    // one decode row's MoE sum (2,560 fp32), one row's per-slot y, and a 128-row block
    for ([_]usize{ 2560, 28160, 327680 }) |n| for ([_]usize{ 1, 48 }) |per_graph| {
        const mine = try a.alloc(f32, n * per_graph);
        defer a.free(mine);
        const theirs = try a.alloc(f32, n * per_graph);
        defer a.free(theirs);
        var p0 = std.Random.DefaultPrng.init(0x9a + rank);
        var p1 = std.Random.DefaultPrng.init(0x9a + (1 - rank));
        for (mine) |*v| v.* = p0.random().floatNorm(f32);
        for (theirs) |*v| v.* = p1.random().floatNorm(f32);
        var src = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(mine));
        defer src.free();
        var dst = try cuda.DeviceBuffer.alloc(gpu.d, n * per_graph * 4);
        defer dst.free();
        try dst.fill8(0, s.handle);
        try s.synchronize();
        _ = try rk.swap(n);
        // eager: per_graph all-reduces in a row
        const reps: usize = if (per_graph == 1) 50 else 5;
        for (0..3) |_| for (0..per_graph) |k| try allReduce(c, s, src.ptr + k * n * 4, dst.ptr + k * n * 4, n);
        try s.synchronize();
        try e0.record(s);
        for (0..reps) |_| for (0..per_graph) |k| try allReduce(c, s, src.ptr + k * n * 4, dst.ptr + k * n * 4, n);
        try e1.record(s);
        try e1.synchronize();
        const t_eager = (try e0.elapsedMs(e1)) / @as(f32, @floatFromInt(reps));
        try dst.fill8(0, s.handle);
        try s.synchronize();
        _ = try rk.swap(n + 1);
        // captured: the same calls in one graph, instantiated, replayed
        const cap_ok = blk: {
            cuda.graph.beginCapture(s, .thread_local) catch |err| {
                std.debug.print("rank {d}: beginCapture failed: {s}\n", .{ rank, @errorName(err) });
                break :blk false;
            };
            var failed = false;
            for (0..per_graph) |k| allReduce(c, s, src.ptr + k * n * 4, dst.ptr + k * n * 4, n) catch |err| {
                std.debug.print("rank {d}: ncclAllReduce under capture failed: {s}\n", .{ rank, @errorName(err) });
                failed = true;
                break;
            };
            var g = cuda.graph.endCapture(s) catch |err| {
                std.debug.print("rank {d}: endCapture failed: {s}\n", .{ rank, @errorName(err) });
                break :blk false;
            };
            defer g.deinit();
            if (failed) break :blk false;
            const nodes = g.nodeCount() catch 0;
            var exec = g.instantiate() catch |err| {
                std.debug.print("rank {d}: instantiate failed: {s}\n", .{ rank, @errorName(err) });
                break :blk false;
            };
            defer exec.deinit();
            exec.upload(s) catch {};
            for (0..3) |_| try exec.launchOn(s);
            try s.synchronize();
            const got = try a.alloc(f32, n * per_graph);
            defer a.free(got);
            try dst.download(0, std.mem.sliceAsBytes(got));
            var eq: usize = 0;
            for (got, mine, theirs) |x, m, t| {
                const want = if (rank == 0) m + t else t + m;
                eq += @intFromBool(@as(u32, @bitCast(x)) == @as(u32, @bitCast(want)));
            }
            try e0.record(s);
            for (0..reps) |_| try exec.launchOn(s);
            try e1.record(s);
            try e1.synchronize();
            const t_graph = (try e0.elapsedMs(e1)) / @as(f32, @floatFromInt(reps));
            std.debug.print("RESULT rank {d} graph: {d} x ncclAllReduce of {d} fp32 ({d} graph nodes): eager {d:.4} ms, graph replay {d:.4} ms ({d:.4} vs {d:.4} ms an all-reduce); replayed sums == fp32 a+b on {d}/{d}\n", .{ rank, per_graph, n, nodes, t_eager, t_graph, t_eager / @as(f32, @floatFromInt(per_graph)), t_graph / @as(f32, @floatFromInt(per_graph)), eq, got.len });
            break :blk eq == got.len;
        };
        all_ok = all_ok and cap_ok;
        const both = try rk.swap(@intFromBool(cap_ok));
        all_ok = all_ok and both == 1;
    };
    try check.expect(all_ok, "rank {d}: an all-reduce captured in a graph failed or gave other bits", .{rank});
    check.pass("rank {d}: ncclAllReduce on the nonblocking communicator captured in a CUDA graph replays on both ranks with the eager bits (SHM transport)", .{rank});
}
