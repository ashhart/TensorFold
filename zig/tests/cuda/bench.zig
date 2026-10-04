//! Launch cost: N dependent one-thread kernels on a plain stream against the same N as one graph.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

fn enqueue(step: cuda.Function, cfg: cuda.Config, stream: cuda.Stream, counter: u64, n: usize) !void {
    for (0..n) |_| {
        var args: cuda.Args = .{};
        args.add(counter);
        args.add(@as(u64, 1));
        try cuda.launch.launch(step, cfg, stream, &args);
    }
}

/// Host cost of preparing the next replay by patching: every node's arguments, then a whole-exec update from a recapture.
fn patchCost(gpu: Gpu, step: cuda.Function, cfg: cuda.Config, stream: cuda.Stream, counter: u64, n: usize, g: cuda.graph.Graph, exec: cuda.graph.Exec) !u64 {
    const buf = try gpu.gpa.alloc(cuda.graph.Node, n);
    defer gpu.gpa.free(buf);
    const nodes = try g.nodes(buf);
    const t0 = check.now(gpu.io);
    for (nodes) |node| {
        var args: cuda.Args = .{};
        args.add(counter);
        args.add(@as(u64, 1));
        try exec.setKernel(node, step, cfg, &args);
    }
    const t1 = check.now(gpu.io);
    try cuda.graph.beginCapture(stream, .thread_local);
    try enqueue(step, cfg, stream, counter, n);
    var again = try cuda.graph.endCapture(stream);
    defer again.deinit();
    const t2 = check.now(gpu.io);
    const verdict = try exec.update(again);
    const t3 = check.now(gpu.io);
    try check.expect(verdict == .success, "whole-exec update from a recapture: {t}", .{verdict});
    try exec.launchOn(stream);
    try stream.synchronize();
    const per = @as(f64, @floatFromInt(n));
    std.debug.print("RESULT graph patch n={d}: node argument update {d:.3} us/node, recapture {d:.3} us/kernel, whole-exec update {d:.3} us/node\n", .{ n, @as(f64, @floatFromInt(t1 - t0)) / 1000 / per, @as(f64, @floatFromInt(t2 - t1)) / 1000 / per, @as(f64, @floatFromInt(t3 - t2)) / 1000 / per });
    return n;
}

/// Medians over `reps` of us per launch (host enqueue, wall, GPU events); `pdl` adds programmatic dependent launch.
pub fn overhead(gpu: Gpu, n: usize, reps: usize, pdl: bool) !void {
    const d = gpu.d;
    var probe = try cuda.Module.load(d, cuda.kernels.probe);
    defer probe.unload();
    const step = try probe.function(if (pdl) "tf_probe_pdl_step" else "tf_probe_step");
    const one: cuda.Config = .{ .grid = .{}, .block = .{}, .pdl = pdl };
    const label = if (pdl) "pdl " else "";
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var counter = try cuda.DeviceBuffer.alloc(d, 8);
    defer counter.free();
    try counter.fill8(0, null);
    var start = try cuda.Event.init(d, true);
    defer start.deinit();
    var end = try cuda.Event.init(d, true);
    defer end.deinit();

    try enqueue(step, one, stream, counter.ptr, 200);
    try stream.synchronize();
    var expected: u64 = 200;

    const enq = try gpu.gpa.alloc(f64, reps);
    defer gpu.gpa.free(enq);
    const wall = try gpu.gpa.alloc(f64, reps);
    defer gpu.gpa.free(wall);
    const dev = try gpu.gpa.alloc(f64, reps);
    defer gpu.gpa.free(dev);
    const per = @as(f64, @floatFromInt(n));

    for (0..reps) |r| {
        try start.record(stream);
        const t0 = check.now(gpu.io);
        try enqueue(step, one, stream, counter.ptr, n);
        const t1 = check.now(gpu.io);
        try end.record(stream);
        try stream.synchronize();
        const t2 = check.now(gpu.io);
        enq[r] = @as(f64, @floatFromInt(t1 - t0)) / 1000 / per;
        wall[r] = @as(f64, @floatFromInt(t2 - t0)) / 1000 / per;
        dev[r] = @as(f64, try start.elapsedMs(end)) * 1000 / per;
        expected += n;
    }
    std.debug.print("RESULT overhead {s}plain n={d} reps={d}: enqueue {d:.3} us/launch, wall {d:.3} us/launch, gpu {d:.3} us/launch\n", .{ label, n, reps, check.median(enq), check.median(wall), check.median(dev) });

    try cuda.graph.beginCapture(stream, .thread_local);
    try enqueue(step, one, stream, counter.ptr, n);
    var g = try cuda.graph.endCapture(stream);
    defer g.deinit();
    const ti0 = check.now(gpu.io);
    var exec = try g.instantiate();
    defer exec.deinit();
    const ti1 = check.now(gpu.io);
    try exec.upload(stream);
    for (0..2) |_| try exec.launchOn(stream);
    try stream.synchronize();
    expected += 2 * n;

    for (0..reps) |r| {
        try start.record(stream);
        const t0 = check.now(gpu.io);
        try exec.launchOn(stream);
        const t1 = check.now(gpu.io);
        try end.record(stream);
        try stream.synchronize();
        const t2 = check.now(gpu.io);
        enq[r] = @as(f64, @floatFromInt(t1 - t0)) / 1000;
        wall[r] = @as(f64, @floatFromInt(t2 - t0)) / 1000 / per;
        dev[r] = @as(f64, try start.elapsedMs(end)) * 1000 / per;
        expected += n;
    }
    std.debug.print("RESULT overhead {s}graph n={d} reps={d}: cuGraphLaunch {d:.1} us per graph, wall {d:.3} us/kernel, gpu {d:.3} us/kernel, instantiate {d:.1} ms\n", .{ label, n, reps, check.median(enq), check.median(wall), check.median(dev), @as(f64, @floatFromInt(ti1 - ti0)) / 1e6 });
    if (!pdl) expected += try patchCost(gpu, step, one, stream, counter.ptr, n, g, exec);

    var total: u64 = 0;
    try counter.download(0, std.mem.asBytes(&total));
    try check.expect(total == expected, "counter {d} != {d}: a launch was lost or ran twice", .{ total, expected });
    check.pass("launch cost: every one of {d} dependent kernels ran exactly once", .{expected});
}
