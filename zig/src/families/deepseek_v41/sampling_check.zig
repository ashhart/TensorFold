//! tf-dsv41-samp: the keyed sampler's device steps on one GPU against the host reference; PASS / FAIL, exit 0 on PASS.

const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const smp = @import("dsv41_serve").sampling;
const pipe = @import("sampling_pipe.zig");
const Gpu = @import("sampling_dev.zig").Gpu;

const C: usize = 64640;
const n: usize = 16;

fn rows(rnd: std.Random, out: []f32) void {
    for (0..n) |r| {
        const row = out[r * C ..][0..C];
        const kind = r % 4;
        for (row, 0..) |*v, c| {
            const x: f64 = switch (kind) {
                0 => rnd.floatNorm(f64) * 2.0,
                1 => rnd.floatNorm(f64) * 3.0 + (if (c == r * 97) @as(f64, 14) else 0),
                2 => @floor(rnd.floatNorm(f64) * 4.0) / 2.0, // coarse: ties everywhere
                else => if (c % 3 == 0) 1.5 else -2.0, // flat bands
            };
            const b: u32 = @bitCast(@as(f32, @floatCast(x)));
            v.* = @bitCast(b & 0xFFFF_0000); // bf16-valued, as kit_logits leaves them
        }
    }
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var kernels = try dk.Kernels.load(&ctx);
    defer kernels.deinit();
    var gpu = try Gpu.init(&driver, stream, &kernels);
    defer {
        drainOrExit(stream);
        gpu.deinit();
    }
    var host: pipe.HostDevice = .{ .gpa = gpa };
    defer host.deinit();
    const dev = gpu.device();
    const ref = host.device();

    var prng = std.Random.DefaultPrng.init(4101);
    const lg = try gpa.alloc(f32, n * C);
    defer gpa.free(lg);
    rows(prng.random(), lg);
    var d_lg = try cuda.DeviceBuffer.alloc(&driver, 4 * lg.len);
    defer {
        drainOrExit(stream);
        d_lg.free();
    }
    try d_lg.upload(0, std.mem.sliceAsBytes(lg));
    const h_lg: u64 = @intFromPtr(lg.ptr);
    var bad: usize = 0;

    // candidates: device packed rows == host packed rows, bit for bit
    for ([_]usize{ 9, 28, 520, 1024 }) |k| {
        const len = n * 2 * k;
        const d_out = try dev.vtable.scratch(dev.ptr, 0, 4 * len);
        const h_out = try ref.vtable.scratch(ref.ptr, 0, 4 * len);
        try dev.vtable.top(dev.ptr, d_lg.ptr, n, C, k, 64640, d_out);
        try ref.vtable.top(ref.ptr, h_lg, n, C, k, 64640, h_out);
        const got = try gpa.alloc(u32, len);
        defer gpa.free(got);
        try dev.vtable.download(dev.ptr, d_out, std.mem.sliceAsBytes(got));
        const want = @as([*]const u32, @ptrFromInt(h_out))[0..len];
        const diff: ?usize = for (got, want, 0..) |g, w, i| {
            if (g != w) break i;
        } else null;
        bad += @intFromBool(diff != null);
        std.debug.print("candidates k {d}: {s}\n", .{ k, if (diff != null) "differ" else "bit-equal" });
        if (diff) |i| std.debug.print("  first at {d}: {x} vs {x}\n", .{ i, got[i], want[i] });
    }

    // statistics: max bit-equal, sum within 1e-12 relative
    var picker = try smp.Picker.init(gpa, C);
    defer picker.deinit();
    const split = gpu.fns.split != null;
    if (!split) {
        bad += 1;
        std.debug.print("stats: the fatbin has no split kernels (dsv41_sampling_stats_*)\n", .{});
    }
    for ([_]f64{ 0.6, 1.0, 1.3 }) |t| {
        const d_st = try dev.vtable.scratch(dev.ptr, 2, 16 * n);
        const h_st = try ref.vtable.scratch(ref.ptr, 2, 16 * n);
        try ref.vtable.stats(ref.ptr, h_lg, n, C, t, h_st);
        const want = @as([*]const f64, @ptrFromInt(h_st))[0 .. 2 * n];
        var got: [2 * n]f64 = undefined;
        for ([_]bool{ false, true }) |mode| {
            if (mode and !split) continue;
            gpu.split = mode; // the split run last: the picks below read its statistics
            try dev.vtable.stats(dev.ptr, d_lg.ptr, n, C, t, d_st);
            try dev.vtable.download(dev.ptr, d_st, std.mem.sliceAsBytes(&got));
            var worst: f64 = 0;
            var max_ok = true;
            for (0..n) |r| {
                max_ok = max_ok and got[2 * r] == want[2 * r];
                worst = @max(worst, @abs(got[2 * r + 1] - want[2 * r + 1]) / want[2 * r + 1]);
            }
            const ok = max_ok and worst <= 1e-12;
            bad += @intFromBool(!ok);
            std.debug.print("stats {s} T {d}: max {s}, sum worst relative {e} ({s})\n", .{ if (mode) "split" else "row", t, if (max_ok) "bit-equal" else "DIFFERS", worst, if (ok) "ok" else "FAIL" });
        }

        // the picks from the device's candidates + statistics == the host's (512 candidates, one rank)
        const k = 512;
        const d_c = try dev.vtable.scratch(dev.ptr, 0, 4 * n * 2 * k);
        const h_c = try ref.vtable.scratch(ref.ptr, 0, 4 * n * 2 * k);
        try dev.vtable.top(dev.ptr, d_lg.ptr, n, C, k, 0, d_c);
        try ref.vtable.top(ref.ptr, h_lg, n, C, k, 0, h_c);
        const dc = try gpa.alloc(f32, n * 2 * k);
        defer gpa.free(dc);
        try dev.vtable.download(dev.ptr, d_c, std.mem.sliceAsBytes(dc));
        const hc = @as([*]const f32, @ptrFromInt(h_c))[0 .. n * 2 * k];
        var vals: [2][k]f32 = undefined;
        var ids: [2][k]u32 = undefined;
        var picks: usize = 0;
        var same: usize = 0;
        for ([_]smp.Sampling{
            .{ .seed = 11, .temperature = t, .top_k = 0, .top_p = 0.9 },
            .{ .seed = 12, .temperature = t, .top_k = 0, .top_p = 0.5, .min_p = 0.05 },
            .{ .seed = 13, .temperature = t, .top_k = 40, .top_p = 0.95 },
        }) |s| for (0..n) |r| for (0..8) |p| {
            const pos: u64 = 1000 + 37 * p;
            var out: [2]smp.Pick = undefined;
            for ([_][]const f32{ dc, hc }, [_][]const f64{ &got, want }, 0..) |cand_data, st, side| {
                const cand: smp.Candidates = .{ .data = cand_data, .world = 1, .rows = n, .k = k };
                const m = cand.row(@intCast(r), &vals[side], &ids[side]);
                const stats = [1][2]f64{.{ st[2 * r], st[2 * r + 1] }};
                const nuc = smp.isNucleus(s);
                out[side] = picker.pick(.{ .values = vals[side][0..m], .ids = ids[side][0..m], .stats = if (nuc) &stats else &.{}, .count = smp.candidateCount(s, @intCast(C), smp.nucleus_count) }, pos, s);
            }
            picks += 1;
            same += @intFromBool(std.meta.eql(out[0], out[1]));
        };
        bad += picks - same;
        std.debug.print("picks T {d}: {d}/{d} equal\n", .{ t, same, picks });
    }
    try timeStats(&driver, stream, &gpu, dev, d_lg.ptr);
    std.debug.print("{s} tf-dsv41-samp: {d} checks failed\n", .{ if (bad == 0) "PASS" else "FAIL", bad });
    return if (bad == 0) 0 else 1;
}

/// Each statistics kernel's device time a call (timing events around `reps` calls) at a segment's row counts.
fn timeStats(driver: *const cuda.Driver, stream: cuda.Stream, gpu: *Gpu, dev: pipe.Device, lg: u64) !void {
    const reps = 50;
    var e0 = try cuda.Event.init(driver, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(driver, true);
    defer e1.deinit();
    const out = try dev.vtable.scratch(dev.ptr, 2, 16 * n);
    for ([_]usize{ 1, 5, 16 }) |rows_| for ([_]bool{ false, true }) |mode| {
        if (mode and gpu.fns.split == null) continue;
        gpu.split = mode;
        try dev.vtable.stats(dev.ptr, lg, rows_, C, 0.7, out); // warm (scratch, module)
        try e0.record(stream);
        for (0..reps) |_| try dev.vtable.stats(dev.ptr, lg, rows_, C, 0.7, out);
        try e1.record(stream);
        try e1.synchronize();
        const ms = try cuda.Event.elapsedMs(e0, e1);
        std.debug.print("stats time {s} rows {d}: {d:.1} us a call\n", .{ if (mode) "split" else "row", rows_, 1000.0 * ms / reps });
    };
}

fn drainOrExit(stream: cuda.Stream) void {
    stream.synchronize() catch |err| {
        std.log.err("sampler drain failed ({t}); retaining owners until process exit", .{err});
        std.process.exit(1);
    };
}
