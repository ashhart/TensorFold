//! exl3-pool: gate 4, miss cost, copy/compute overlap and the decode pipeline.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("../../check.zig");

const st = core.safetensors;
const exl3e = cuda.exl3_experts;
const fpool = flash.exl3_pool;
const policy = fpool.offload;
const Gpu = check.Gpu;

const data = @import("exl3_pool_data.zig");
const dims = data.dims;
const width = data.width;
const routed_n = data.routed_n;
const table_n = data.table_n;
const top_k = data.top_k;
const slots = data.slots;
const max_rows = data.max_rows;
const shard_count = data.shard_count;
const projs = data.projs;
const Pack = data.Pack;
const LayerData = data.LayerData;
const Zipf = data.Zipf;
const bf16Bits = data.bf16Bits;
const Io = data.Io;
const call = data.call;
const poolRound = data.poolRound;
const genRound = data.genRound;

// ---------------------------------------------------------------------------------------------------- gate 4

fn ms(t0: i96, t1: i96) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) / 1e6;
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

/// A row whose first `k` routed picks are not held by the pool and the rest are.
fn rowWithMisses(pool: *const fpool.Pool, rng: std.Random, k: usize, out: []i32) void {
    var n: usize = 0;
    // held experts first (up to 10 - k)
    for (pool.slots_tab, pool.line_tab) |id, state| {
        if (n == top_k - k) break;
        if (state == .resident) {
            out[k + n] = id;
            n += 1;
        }
    }
    var m: usize = 0;
    while (m < k) {
        const id: u16 = rng.intRangeLessThan(u16, 0, routed_n);
        if (policy.lineOf(pool.v, pool.opt.layer, id) != null) continue;
        if (std.mem.indexOfScalar(i32, out[0..m], id) != null) continue;
        out[m] = id;
        m += 1;
    }
    out[top_k] = 512;
}

fn missCost(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData, src: fpool.Source, direct: bool, label: []const u8) !void {
    const gpa = gpu.gpa;
    var pool = try fpool.Pool.init(gpa, gpu.io, gpu.d, src, ld.scales(), .{ .layer = @intCast(ld.layer), .lines = 64, .direct = direct, .double_buffer = false }, null);
    defer pool.deinit();
    var vp = try Io.init(gpu, pool.tables());
    defer vp.free();
    var prng = std.Random.DefaultPrng.init(0xbe7c4);
    const rng = prng.random();
    var x: [dims]u16 = undefined;
    for (&x) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
    const wts: [slots]f32 = @splat(0.1);
    var picks: [slots]i32 = undefined;
    // warm: fill the pool with 64 experts
    for (0..7) |_| {
        rowWithMisses(&pool, rng, top_k, &picks);
        try vp.set(&x, &picks, &wts);
        _ = try poolRound(&pool, ex, s, vp, &picks, 1, false);
        try s.synchronize();
    }
    var c0 = try cuda.Event.init(gpu.d, true);
    defer c0.deinit();
    var c1 = try cuda.Event.init(gpu.d, true);
    defer c1.deinit();
    var k0 = try cuda.Event.init(gpu.d, true);
    defer k0.deinit();
    var k1 = try cuda.Event.init(gpu.d, true);
    defer k1.deinit();
    std.debug.print("  [{s}] one decode row (10 routed + shared), 64 lines, rounds of k misses (median of 9; idle card):\n", .{label});
    var base: f64 = 0;
    for ([_]usize{ 0, 1, 2, 4, 10 }) |k| {
        var wall: [9]f64 = undefined;
        var copy: [9]f64 = undefined;
        var comp: [9]f64 = undefined;
        var stage_host: [9]f64 = undefined;
        for (0..9) |it| {
            rowWithMisses(&pool, rng, k, &picks);
            try vp.set(&x, &picks, &wts);
            try s.synchronize();
            const t0 = check.now(gpu.io);
            const r = try pool.plan(&picks, 1);
            const t1 = check.now(gpu.io);
            try c0.record(pool.lines.copy);
            try pool.stage(r, 0, s);
            try c1.record(pool.lines.copy);
            const t2 = check.now(gpu.io);
            try k0.record(s);
            try call(ex, s, pool.tables(), vp, pool.picksFor(r, 0, vp.picks.ptr), 1, false);
            try k1.record(s);
            try pool.release(r, 0, s);
            try s.synchronize();
            try pool.lines.copy.synchronize();
            const t3 = check.now(gpu.io);
            wall[it] = ms(t0, t3);
            copy[it] = try c0.elapsedMs(c1);
            comp[it] = try k0.elapsedMs(k1);
            stage_host[it] = ms(t1, t2);
        }
        const w = median(&wall);
        if (k == 0) base = w;
        std.debug.print("    k={d:>2}: round wall {d:.3} ms (+{d:.3} vs all-hit), copy stream {d:.3} ms ({d:.3} ms/expert), experts compute {d:.3} ms (measured after the stream's wait on the copies), host stage {d:.3} ms\n", .{
            k, w, w - base, median(&copy), if (k > 0) median(&copy) / @as(f64, @floatFromInt(k)) else 0, median(&comp), median(&stage_host),
        });
    }
    const ls = pool.lines.stats;
    if (!direct) std.debug.print("    host staging memcpy (mmap -> pinned): {d:.3} ms/expert average over {d} copies ({d:.2} GB/s)\n", .{ @as(f64, @floatFromInt(ls.stage_ns)) / 1e6 / @as(f64, @floatFromInt(ls.copies)), ls.copies, @as(f64, @floatFromInt(ls.staged_bytes)) / @as(f64, @floatFromInt(ls.stage_ns)) });
}

/// A 16-row chunk through a 24-line pool (several passes): copies of pass j+1 overlap compute of pass j or not.
fn overlapChunk(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData, src: fpool.Source, serial: bool) !struct { wall: f64, copy: f64, comp: f64, passes: usize, copies: usize } {
    const gpa = gpu.gpa;
    var pool = try fpool.Pool.init(gpa, gpu.io, gpu.d, src, ld.scales(), .{ .layer = @intCast(ld.layer), .lines = 24, .direct = true }, null);
    defer pool.deinit();
    var vp = try Io.init(gpu, pool.tables());
    defer vp.free();
    const z = Zipf.init(0x0e1a9 + ld.layer, 0.8);
    var prng = std.Random.DefaultPrng.init(0x0e1a9);
    const rng = prng.random();
    var x: [max_rows * dims]u16 = undefined;
    var picks: [max_rows * slots]i32 = undefined;
    var wts: [max_rows * slots]f32 = undefined;
    genRound(&z, rng, max_rows, &x, &picks, &wts);
    try vp.set(&x, &picks, &wts);
    var cs: [64]cuda.Event = undefined;
    for (&cs) |*e| e.* = try cuda.Event.init(gpu.d, true);
    defer for (&cs) |*e| e.deinit();
    var ks: [64]cuda.Event = undefined;
    for (&ks) |*e| e.* = try cuda.Event.init(gpu.d, true);
    defer for (&ks) |*e| e.deinit();
    try s.synchronize();
    const t0 = check.now(gpu.io);
    const r = try pool.plan(&picks, max_rows);
    if (r.n_passes > 32) return error.TooManyPasses;
    const l = pool.tables();
    for (0..r.n_passes) |j| {
        try cs[2 * j].record(pool.lines.copy);
        try pool.stage(r, j, s);
        try cs[2 * j + 1].record(pool.lines.copy);
        if (serial) try pool.lines.copy.synchronize();
        try ks[2 * j].record(s);
        try call(ex, s, l, vp, pool.picksFor(r, j, vp.picks.ptr), max_rows, false);
        try ks[2 * j + 1].record(s);
        try pool.release(r, j, s);
        if (serial) try s.synchronize();
    }
    try s.synchronize();
    try pool.lines.copy.synchronize();
    const wall = ms(t0, check.now(gpu.io));
    var copy: f64 = 0;
    var comp: f64 = 0;
    for (0..r.n_passes) |j| {
        copy += try cs[2 * j].elapsedMs(cs[2 * j + 1]);
        comp += try ks[2 * j].elapsedMs(ks[2 * j + 1]);
    }
    return .{ .wall = wall, .copy = copy, .comp = comp, .passes = r.n_passes, .copies = r.n_copies };
}

/// 60 decode rounds (one row each, skewed picks known ahead), sequential (sync every round) vs pipelined (the host
fn decodePipe(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData, src: fpool.Source, lines: usize, pipelined: bool, double: bool) !struct { wall: f64, hit: f64, release_waits: u64, protected: u64 } {
    const gpa = gpu.gpa;
    var pool = try fpool.Pool.init(gpa, gpu.io, gpu.d, src, ld.scales(), .{ .layer = @intCast(ld.layer), .lines = lines, .direct = true, .double_buffer = double }, null);
    defer pool.deinit();
    const rounds = 60;
    var ios: [rounds]Io = undefined;
    var made: usize = 0;
    defer for (ios[0..made]) |*v| v.free();
    const z = Zipf.init(0xdec0 + ld.layer, 1.0);
    var prng = std.Random.DefaultPrng.init(0xdec0);
    const rng = prng.random();
    var picks: [rounds][slots]i32 = undefined;
    var x: [dims]u16 = undefined;
    var wts: [slots]f32 = undefined;
    for (0..rounds) |i| {
        ios[i] = try Io.init(gpu, pool.tables());
        made += 1;
        genRound(&z, rng, 1, &x, &picks[i], &wts);
        try ios[i].set(&x, &picks[i], &wts);
    }
    try s.synchronize();
    const t0 = check.now(gpu.io);
    for (0..rounds) |i| {
        _ = try poolRound(&pool, ex, s, ios[i], &picks[i], 1, false);
        if (!pipelined) try s.synchronize();
    }
    try s.synchronize();
    const wall = ms(t0, check.now(gpu.io));
    const ps = pool.stats;
    return .{ .wall = wall, .hit = @as(f64, @floatFromInt(ps.hit_slots)) / @as(f64, @floatFromInt(ps.hit_slots + ps.miss_slots)), .release_waits = pool.lines.stats.release_waits, .protected = ps.protected };
}

pub fn benchRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData) !void {
    const gpa = gpu.gpa;
    // (a) miss cost, staged from the mapped checkpoint
    try missCost(gpu, ex, s, ld, ld.source(), false, "staged: mmap -> pinned staging -> H2D");
    // the same experts in one pinned bank (what keeping the routed trellis in pinned host RAM gives): direct H2D
    var total: usize = 0;
    for (ld.routed) |e| for (e) |b| {
        total += b.len;
    };
    var bank = try cuda.HostBuffer.alloc(gpu.d, total);
    defer bank.free();
    const pinned = try gpa.alloc([3][]const u8, routed_n);
    defer gpa.free(pinned);
    var at: usize = 0;
    for (ld.routed, pinned) |e, *o| for (e, 0..) |b, p| {
        @memcpy(bank.bytes[at..][0..b.len], b);
        o[p] = bank.bytes[at..][0..b.len];
        at += b.len;
    };
    std.debug.print("  pinned bank for layer {d}: {d:.0} MiB of routed trellis copied into page-locked host memory\n", .{ ld.layer, @as(f64, @floatFromInt(total)) / 1048576.0 });
    const psrc: fpool.Source = .{ .routed = pinned, .shared = ld.shared };
    try missCost(gpu, ex, s, ld, psrc, true, "pinned bank: direct H2D");

    // (b) a 16-row chunk split into passes: serial vs overlapped copies
    const ser = try overlapChunk(gpu, ex, s, ld, psrc, true);
    const ovl = try overlapChunk(gpu, ex, s, ld, psrc, false);
    const hidden_ms = (ser.wall - ovl.wall);
    std.debug.print("  [16-row chunk, 24 lines, pinned bank] {d} passes, {d} copies: serial {d:.3} ms (copy {d:.3} + compute {d:.3}), overlapped {d:.3} ms; overlap hides {d:.3} ms = {d:.0}% of the smaller of copy/compute\n", .{
        ser.passes, ser.copies, ser.wall, ser.copy, ser.comp, ovl.wall, hidden_ms, 100.0 * hidden_ms / @min(ser.copy, ser.comp),
    });
    // (c) decode rounds: pipelined staging, double buffering on/off
    for ([_]usize{ 24, 48 }) |lines| {
        const seq = try decodePipe(gpu, ex, s, ld, psrc, lines, false, true);
        const pipe1 = try decodePipe(gpu, ex, s, ld, psrc, lines, true, false);
        const pipe2 = try decodePipe(gpu, ex, s, ld, psrc, lines, true, true);
        std.debug.print("  [60 decode rounds, {d} lines, pinned bank, hit {d:.3}] sequential {d:.2} ms; pipelined {d:.2} ms single-buffered ({d} copies waited for the previous round's compute), {d:.2} ms double-buffered ({d} waited, {d} rounds spared the previous round's lines)\n", .{
            lines, seq.hit, seq.wall, pipe1.wall, pipe1.release_waits, pipe2.wall, pipe2.release_waits, pipe2.protected,
        });
    }
    check.pass("layer {d}: miss cost, chunk overlap and decode pipelining measured (see lines above)", .{ld.layer});
}
