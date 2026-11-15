//! flashnext-exl3-graph plan: the device planner against the host planner on synthetic pools, and the copy kernel.

const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const Gpu = check.Gpu;
const X = flash.exl3;
const dplan = X.dplan;
const fpool = flash.exl3_pool;
const harness = @import("flashnext_exl3_graph_engine.zig");
const ms = harness.ms;

// -- synthetic pools: device planner == host planner -----------------------------------------------------------------

const syn_dims = 256;
const syn_width = 64;

fn synPool(gpu: Gpu, routed: [][3][]const u8, lines: usize, max_rows: usize) !fpool.Pool {
    return fpool.Pool.init(gpu.gpa, gpu.io, gpu.d, .{ .routed = routed }, .{ .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0 }, .{
        .layer = 0,
        .lines = lines,
        .dims = syn_dims,
        .width = syn_width,
        .max_rows = max_rows,
        .slots = 11,
        .direct = true,
        .double_buffer = false,
    }, null);
}

fn planCase(gpu: Gpu, lines: usize, max_rows: usize, rounds: usize, seed: u64) !void {
    const a = gpu.gpa;
    const tiles = (syn_dims / 16) * (syn_width / 16);
    // a pinned bank of 512 experts, widths 2..5 half-bits, random bytes
    var sizes: [512][3]usize = undefined;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var total: usize = 0;
    for (&sizes) |*e| for (e) |*b| {
        b.* = tiles * 16 * (2 + rnd.uintLessThan(usize, 4));
        total += (b.* + 255) / 256 * 256;
    };
    var bank = try cuda.HostBuffer.alloc(gpu.d, total);
    defer bank.free();
    rnd.bytes(bank.bytes);
    const routed = try a.alloc([3][]const u8, 512);
    defer a.free(routed);
    var at: usize = 0;
    for (routed, sizes) |*r, sz| for (0..3) |w| {
        r[w] = bank.bytes[at..][0..sz[w]];
        at += (sz[w] + 255) / 256 * 256;
    };
    var host = try synPool(gpu, routed, lines, max_rows);
    defer host.deinit();
    var dev = try synPool(gpu, routed, lines, max_rows);
    defer dev.deinit();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var pools = [_]*fpool.Pool{&dev};
    const dp = try dplan.DevPlan.init(a, gpu.d, &pools);
    defer dp.deinit();
    var picks_dev = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * 11 * 4);
    defer picks_dev.free();
    const picks = try a.alloc(i32, max_rows * 11);
    defer a.free(picks);
    // skewed routing: a hot set of 24 experts takes most picks, so rounds hit, miss and evict
    var host_ns: u64 = 0;
    for (0..rounds) |round| {
        const rows = 1 + rnd.uintLessThan(usize, max_rows);
        for (0..rows) |r| {
            var k: usize = 0;
            while (k < 10) {
                const id: i32 = if (rnd.uintLessThan(u32, 4) != 0) @intCast(rnd.uintLessThan(u32, 24) * 7) else @intCast(rnd.uintLessThan(u32, 512));
                var dup = false;
                for (picks[r * 11 ..][0..k]) |x| dup = dup or x == id;
                if (dup) continue;
                picks[r * 11 + k] = id;
                k += 1;
            }
            picks[r * 11 + 10] = if (round % 3 == 0) -1 else 512;
        }
        const n = rows * 11;
        // host
        const t0 = check.now(gpu.io);
        const rp = try host.plan(picks[0..n], rows);
        host_ns += @intCast(check.now(gpu.io) - t0);
        try check.expect(rp.n_passes == 1, "a synthetic round split into {d} passes", .{rp.n_passes});
        try host.stage(rp, 0, s);
        try host.release(rp, 0, s);
        // device
        try dp.toDevice(s);
        try picks_dev.upload(0, std.mem.sliceAsBytes(picks[0..n]));
        try dp.plan(s, 0, picks_dev.ptr, rows);
        try dp.join(s);
        try dp.restore(s, 0);
        if (round % 37 == 36) try dp.toHost(); // owner switches mid-stream keep both views consistent
    }
    try s.synchronize();
    try host.lines.drain();
    try dp.toHost();
    // residency, stamps, counts
    var same_lines: usize = 0;
    for (host.slots_tab, dev.slots_tab, host.stamp_tab, dev.stamp_tab) |hs, ds, ht, dt| same_lines += @intFromBool(hs == ds and ht == dt);
    const same_counts = std.mem.eql(u32, &host.counts[0], &dev.counts[0]);
    std.debug.print("RESULT plan lines={d} max_rows={d} rounds={d}: lines+stamps equal {d}/{d}, counts equal {}, host hits {d}/{d} copies {d} evictions {d}; device hits {d}/{d} copies {d} evictions {d}; host plan {d:.1} us a round\n", .{ lines, max_rows, rounds, same_lines, lines, same_counts, host.stats.hit_slots, host.stats.hit_slots + host.stats.miss_slots, host.stats.misses, host.stats.evictions, dev.stats.hit_slots, dev.stats.hit_slots + dev.stats.miss_slots, dev.stats.misses, dev.stats.evictions, @as(f64, @floatFromInt(host_ns)) / 1e3 / @as(f64, @floatFromInt(rounds)) });
    try check.expect(same_lines == lines and same_counts, "device planner diverged from the host planner", .{});
    try check.expect(host.stats.hit_slots == dev.stats.hit_slots and host.stats.misses == dev.stats.misses and host.stats.evictions == dev.stats.evictions, "device planner statistics differ", .{});
    // tables: the same line (relative to each pool's lines) and width for every expert
    const tb = host.lines.tables.len;
    const th = try a.alloc(u8, tb);
    defer a.free(th);
    const td = try a.alloc(u8, tb);
    defer a.free(td);
    try host.lines.tables.download(0, th);
    try dev.lines.tables.download(0, td);
    var bad_tab: usize = 0;
    for (0..512) |id| for (0..3) |w| {
        const ph = std.mem.readInt(u64, th[host.lines.tlay.ptr[w] + id * 8 ..][0..8], .little);
        const pd = std.mem.readInt(u64, td[dev.lines.tlay.ptr[w] + id * 8 ..][0..8], .little);
        const kh = std.mem.readInt(i32, th[host.lines.tlay.k2[w] + id * 4 ..][0..4], .little);
        const kd = std.mem.readInt(i32, td[dev.lines.tlay.k2[w] + id * 4 ..][0..4], .little);
        const rh = if (ph == 0) 0 else ph - host.lines.mem.ptr + 1;
        const rd = if (pd == 0) 0 else pd - dev.lines.mem.ptr + 1;
        bad_tab += @intFromBool(rh != rd or kh != kd);
    };
    // every resident line holds its expert's bytes (the copy kernel read the mapped bank)
    const mem = try a.alloc(u8, dev.lines.mem.len);
    defer a.free(mem);
    try dev.lines.mem.download(0, mem);
    var bad_lines: usize = 0;
    var resident: usize = 0;
    for (dev.slots_tab, 0..) |id, ln| {
        if (id == fpool.offload.Cache.empty_id) continue;
        resident += 1;
        for (0..3) |w| {
            const want = routed[id][w];
            const got = mem[ln * dev.lines.layout.line_bytes + dev.lines.layout.off[w] ..][0..want.len];
            if (!std.mem.eql(u8, want, got)) bad_lines += 1;
        }
    }
    std.debug.print("RESULT plan tables: {d} of 1536 entries differ; {d} resident lines, {d} projections with wrong bytes\n", .{ bad_tab, resident, bad_lines });
    try check.expect(bad_tab == 0 and bad_lines == 0, "device tables or line bytes wrong", .{});
    check.pass("device planner == host planner over {d} rounds of up to {d} rows on {d} lines (tables, lines, stamps, counts, line bytes)", .{ rounds, max_rows, lines });
}

fn copyBench(gpu: Gpu) !void {
    // Flash Next line: gate/up/down 3-bit, K 2560 x N 640: 160 x 40 tiles of 48 bytes = 614,400 bytes each
    const seg: usize = 614_400;
    const n_lines: usize = 30;
    var bank = try cuda.HostBuffer.alloc(gpu.d, n_lines * 3 * seg);
    defer bank.free();
    @memset(bank.bytes, 0x5a);
    var dst = try cuda.DeviceBuffer.alloc(gpu.d, n_lines * 3 * seg);
    defer dst.free();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_exl3_dplan);
    defer mod.unload();
    const f = try mod.function("fn_exl3_dcopy");
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var e0 = try cuda.Event.init(gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(gpu.d, true);
    defer e1.deinit();
    for ([_]usize{ 1, 3, 10 }) |k| {
        // work list: k lines of three segments
        var work: [10 * 9]i64 = undefined;
        for (0..k) |i| for (0..3) |w| {
            const off = (i * 3 + w) * seg;
            work[i * 9 + w * 3 + 0] = @intCast(@intFromPtr(bank.bytes.ptr) + off);
            work[i * 9 + w * 3 + 1] = @intCast(dst.ptr + off);
            work[i * 9 + w * 3 + 2] = @intCast(seg);
        };
        var wbuf = try cuda.DeviceBuffer.alloc(gpu.d, 10 * 9 * 8 + 16);
        defer wbuf.free();
        try wbuf.upload(0, std.mem.sliceAsBytes(work[0 .. k * 9]));
        const cnt = [1]i32{@intCast(k)};
        try wbuf.upload(10 * 9 * 8, std.mem.sliceAsBytes(&cnt));
        for ([_]u32{ 36, 72, 144, 288, 576 }) |blocks| {
            const reps = 20;
            var args: cuda.Args = .{};
            args.add(wbuf.ptr);
            args.add(wbuf.ptr + 10 * 9 * 8);
            try cuda.launch.launch(f, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, s, &args);
            try e0.record(s);
            for (0..reps) |_| try cuda.launch.launch(f, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, s, &args);
            try e1.record(s);
            try s.synchronize();
            const t = (try e0.elapsedMs(e1)) / reps;
            std.debug.print("BENCH dcopy {d} lines ({d:.2} MiB) with {d} blocks: {d:.3} ms = {d:.1} GB/s\n", .{ k, @as(f64, @floatFromInt(k * 3 * seg)) / (1 << 20), blocks, t, @as(f64, @floatFromInt(k * 3 * seg)) / (t * 1e6) });
        }
        // the copy engine, one H2D a segment
        try e0.record(s);
        for (0..20) |_| for (0..k * 3) |j| try s.d.check(s.d.api.cuMemcpyHtoDAsync_v2(dst.ptr + j * seg, bank.bytes[j * seg ..].ptr, seg, s.handle), "cuMemcpyHtoDAsync");
        try e1.record(s);
        try s.synchronize();
        const t = (try e0.elapsedMs(e1)) / 20;
        std.debug.print("BENCH dma   {d} lines ({d:.2} MiB): {d:.3} ms = {d:.1} GB/s\n", .{ k, @as(f64, @floatFromInt(k * 3 * seg)) / (1 << 20), t, @as(f64, @floatFromInt(k * 3 * seg)) / (t * 1e6) });
    }
}

pub fn planMode(gpu: Gpu) !void {
    try copyBench(gpu);
    try planCase(gpu, 10, 1, 400, 1);
    try planCase(gpu, 47, 4, 600, 2);
    try planCase(gpu, 63, 6, 600, 3);
}
