//! flashnext-exl3-graph prefill, prefill-prof, prof and ngram: layer graphs in the deep prefill and profiling windows.

const std = @import("std");
const platform = @import("sys");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const Gpu = check.Gpu;
const prompt = flash.prompt;
const X = flash.exl3;
const dplan = X.dplan;
const graph = X.graph;
const harness = @import("flashnext_exl3_graph_engine.zig");
const Eng = harness.Eng;
const ms = harness.ms;

/// A long prompt through the deep prefill (TF_FLASHNEXT_EXL3_PREFILL_ROWS sets its chunks), then `steps` greedy
pub fn prefillMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, n: usize, steps: usize) !void {
    const a = gpu.gpa;
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    const ids = try a.alloc(u32, n);
    defer a.free(ids);
    var prng = std.Random.DefaultPrng.init(0x9e3779b9);
    for (ids) |*x| x.* = 200 + prng.random().uintLessThan(u32, 60000);
    const hs = try a.alloc(u64, steps + 1);
    defer a.free(hs);
    var ref: ?u64 = null;
    var ok = true;
    for ([_]graph.Mode{ .off, .off, .graph, .graph, .graph }) |m| {
        graph.force = m;
        try dplan.toHost(&g.e.exl3_dplan);
        try g.reset();
        _ = try g.swap(3);
        const p0 = graph.prefill_stats;
        const f0 = g.e.exl3_prefill.stats;
        const t0 = check.now(gpu.io);
        try g.forward(ids, .{ .logits = .last });
        const t1 = check.now(gpu.io);
        const p1 = graph.prefill_stats;
        const f1 = g.e.exl3_prefill.stats;
        try check.expect(f1.chunks > f0.chunks and f1.rows >= f0.rows + n, "rank {d}: deep prefill ran instead of falling back", .{rank});
        if (m == .graph) try check.expect(p1.captures + p1.updates + p1.replays > p0.captures + p0.updates + p0.replays, "rank {d}: prefill graph captured or replayed a layer", .{rank});
        hs[0] = std.hash.Wyhash.hash(0, g.seq.logits);
        var tok = prompt.argmax(g.seq.logits);
        for (0..steps) |i| {
            tok = try g.round(tok);
            hs[i + 1] = std.hash.Wyhash.hash(0, g.seq.logits);
        }
        const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(hs));
        if (world == 2) try check.expect(try g.swap(h) == h, "rank {d}: the ranks' prefill logits differ", .{rank});
        if (ref == null) ref = h;
        const same = ref.? == h;
        ok = ok and same;
        if (rank == 0) std.debug.print("RESULT tp={d} prefill {s}: {d} rows in {d:.1} ms = {d:.1} rows/s; layer graphs captured {d} (updates {d}) replayed {d} eager {d} (capture {d:.1} ms); last-row + {d} decode logits {s} the first eager run\n", .{ world, @tagName(m), n, ms(t0, t1), @as(f64, @floatFromInt(n)) / (ms(t0, t1) / 1e3), p1.captures - p0.captures, p1.updates - p0.updates, p1.replays - p0.replays, p1.eager - p0.eager, @as(f64, @floatFromInt(p1.capture_ns - p0.capture_ns)) / 1e6, steps, if (same) "bit-equal to" else "DIFFER from" });
    }
    graph.force = null;
    try check.expect(ok, "rank {d}: prefill under graphs differs from eager", .{rank});
    check.pass("rank {d}/{d}: deep prefill of {d} rows with layer graphs and the decode after it bit-identical to eager", .{ rank, world, n });
}

/// One warmed TP2 deep-prefill forward inside a CUDA-profiler API range for Nsight Systems.
pub fn prefillProfMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, n: usize, which: []const u8) !void {
    var stage: []const u8 = "engine initialization";
    errdefer std.debug.print("prefill-prof rank={d} failed during {s}\n", .{ rank, stage });
    const a = gpu.gpa;
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    defer graph.force = null;
    const ids = try a.alloc(u32, n);
    defer a.free(ids);
    var prng = std.Random.DefaultPrng.init(0x9e3779b9);
    for (ids) |*x| x.* = 200 + prng.random().uintLessThan(u32, 60000);
    const use_graphs = std.mem.eql(u8, which, "graph");
    if (!use_graphs and !std.mem.eql(u8, which, "eager")) return error.BadMode;
    graph.force = if (use_graphs) .graph else .off;
    try dplan.toHost(&g.e.exl3_dplan);
    stage = "graph warm-up";
    // For graph mode, capture each layer shape before the measured forward.
    for (0..if (use_graphs) @as(usize, 2) else 1) |_| {
        try g.reset();
        _ = try g.swap(3);
        try g.forward(ids, .{ .logits = .last });
        try g.stream.synchronize();
    }
    const before = graph.prefill_stats;
    const f_before = g.e.exl3_prefill.stats;
    stage = "profile setup";
    try g.reset();
    _ = try g.swap(3);
    try g.stream.synchronize();
    var prof = try Profiler.open();
    _ = prof.start();
    stage = "measured forward";
    const t0 = check.now(gpu.io);
    try g.forward(ids, .{ .logits = .last });
    const t1 = check.now(gpu.io);
    _ = prof.stop();
    try g.stream.synchronize();
    const after = graph.prefill_stats;
    const f_after = g.e.exl3_prefill.stats;
    if (use_graphs and after.replays == before.replays) return error.NoPrefillGraphReplay;
    const hash = std.hash.Wyhash.hash(0, g.seq.logits);
    stage = "rank hash check";
    if (world == 2) try check.expect(try g.swap(hash) == hash, "rank {d}: profiled prefill logits differ across ranks", .{rank});
    std.debug.print("RESULT prefill-prof rank={d}/{d} mode={s} rows={d}: {d:.1} ms wall, graphs replayed {d}, eager {d}, captured {d}, updated {d}; expert copies {d} (prefetch {d}), passes {d}, pick wait {d:.1} ms, plan {d:.1} ms; logits {x}; Nsight range excludes warm-up\n", .{ rank, world, which, n, ms(t0, t1), after.replays - before.replays, after.eager - before.eager, after.captures - before.captures, after.updates - before.updates, f_after.copies - f_before.copies, f_after.prefetched - f_before.prefetched, f_after.passes - f_before.passes, @as(f64, @floatFromInt(f_after.host_wait_ns - f_before.host_wait_ns)) / 1e6, @as(f64, @floatFromInt(f_after.plan_ns - f_before.plan_ns)) / 1e6, hash });
}

/// cuProfilerStart/Stop from libcuda (nsys --capture-range=cudaProfilerApi), looked up at run time.
const Profiler = struct {
    lib: platform.DynLib,
    start: *const fn () callconv(.c) c_int,
    stop: *const fn () callconv(.c) c_int,
    fn open() !Profiler {
        var lib = try platform.DynLib.open(platform.cuda_driver);
        return .{ .start = lib.lookup(*const fn () callconv(.c) c_int, "cuProfilerStart") orelse return error.MissingSymbol, .stop = lib.lookup(*const fn () callconv(.c) c_int, "cuProfilerStop") orelse return error.MissingSymbol, .lib = lib };
    }
};

/// A profiling window for nsys:
pub fn profMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, steps: usize, rows: usize, which: []const u8) !void {
    const a = gpu.gpa;
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    const ids = [_]u32{ 760, 6511, 314, 9338, 369, 11751, 13, 1049, 369, 3750, 364, 279, 242476, 300, 21262, 13 };
    graph.force = if (std.mem.eql(u8, which, "graph")) .graph else if (std.mem.eql(u8, which, "dplan")) .dplan else .off;
    var prof = try Profiler.open();
    const toks = try a.alloc(u32, 4 * steps * rows + 64);
    defer a.free(toks);
    var phase: usize = 0;
    while (phase < 2) : (phase += 1) {
        try dplan.toHost(&g.e.exl3_dplan);
        try g.reset();
        try g.forward(&ids, .{ .logits = .last });
        var tok = prompt.argmax(g.seq.logits);
        _ = try g.swap(5);
        if (phase == 1) _ = prof.start();
        const t0 = check.now(gpu.io);
        for (0..steps) |i| {
            tok = try g.round(tok);
            toks[i] = tok;
        }
        const t1 = check.now(gpu.io);
        for (0..steps) |i| {
            for (0..rows) |j| toks[steps + i * rows + j] = toks[(i * rows + j) % steps];
            try g.forward(toks[steps + i * rows ..][0..rows], .{ .logits = .all, .logits_out = g.out.ptr });
        }
        const t2 = check.now(gpu.io);
        if (phase == 1) _ = prof.stop();
        if (rank == 0) std.debug.print("RESULT prof tp={d} {s} phase {d}: rounds {d:.2} ms each, windows of {d} rows {d:.2} ms each; graph captures {d} replays {d}; host prepare {d:.3} ms a step\n", .{ world, which, phase, ms(t0, t1) / @as(f64, @floatFromInt(steps)), rows, ms(t1, t2) / @as(f64, @floatFromInt(steps)), graph.stats.captures, graph.stats.replays, @as(f64, @floatFromInt(graph.stats.prepare_ns)) / 1e6 / @as(f64, @floatFromInt(@max(graph.stats.prepares, 1))) });
    }
    graph.force = null;
}

/// Host cost of the n-gram rows a step gathers (Ngram.rows), 1 and 3 tokens, fresh random histories.
pub fn ngramMode(gpu: Gpu, dir: []const u8) !void {
    const a = gpu.gpa;
    var ng = try flash.ple.Ngram.init(a, gpu.io, dir);
    defer ng.deinit();
    const w = ng.width();
    const emb = try a.alloc(u16, 4 * w);
    defer a.free(emb);
    const xs = try a.alloc(f32, 4 * w / 32);
    defer a.free(xs);
    var prng = std.Random.DefaultPrng.init(7);
    for ([_]usize{ 1, 3 }) |n| for (0..2) |rep| {
        var tot: f64 = 0;
        var worst: f64 = 0;
        const reps = 64;
        for (0..reps) |_| {
            var hist = [2]i64{ prng.random().uintLessThan(u32, 240000), prng.random().uintLessThan(u32, 240000) };
            var ids: [3]u32 = undefined;
            for (&ids) |*x| x.* = prng.random().uintLessThan(u32, 240000);
            const h0 = hist;
            const t0 = check.now(gpu.io);
            try ng.rows(&hist, ids[0..n], emb[0 .. n * w], xs[0 .. n * w / 32]);
            var t = ms(t0, check.now(gpu.io));
            if (rep == 1) { // the same rows again: their pages are resident now
                hist = h0;
                const t1 = check.now(gpu.io);
                try ng.rows(&hist, ids[0..n], emb[0 .. n * w], xs[0 .. n * w / 32]);
                t = ms(t1, check.now(gpu.io));
            }
            tot += t;
            worst = @max(worst, t);
        }
        std.debug.print("RESULT ngram rows of {d} tokens (pass {d}, 1 = warm repeat): {d:.3} ms mean, {d:.3} ms worst\n", .{ n, rep, tot / reps, worst });
    };
}
