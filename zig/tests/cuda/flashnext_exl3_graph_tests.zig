//! flashnext-exl3-graph: the EXL3 offload pool planned on the device (cuda_exl3_dplan.zig) and the decode step as one
//! CUDA graph (cuda_exl3_graph.zig).
//!
//!   flashnext-exl3-graph plan                               synthetic pools: the device planner's lines, stamps,
//!                                                          counts and tables equal the host planner's after random
//!                                                          rounds, every resident line holds its expert's bytes;
//!                                                          the mapped-bank copy kernel's bandwidth vs the copy engine
//!   flashnext-exl3-graph RANK WORLD PORT DIR eq GOLDEN [STEPS] [MODES]
//!                                                          the real pack at tp=1/2: each golden prompt prefilled,
//!                                                          then STEPS greedy decode rounds and the same continuation
//!                                                          as verify windows (rows 3,1,4,2,8,16,...) under each mode
//!                                                          (off = host cut, dplan = device plan eager, graph):
//!                                                          every step's logits bit-equal to off's, ms/token, hit rate
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");

const Gpu = check.Gpu;
const prompt = flash.prompt;
const X = flash.exl3;
const xpass = X.pass;
const dplan = X.dplan;
const graph = X.graph;
const fpool = flash.exl3_pool;
const wire = flash.rank_wire;
const Collective = flash.collective.Collective;
const vocab = prompt.vocabulary;
const cap_default: usize = 32768;
const out_rows: usize = 64;

fn ms(t0: i96, t1: i96) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) / 1e6;
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

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

fn planMode(gpu: Gpu) !void {
    try copyBench(gpu);
    try planCase(gpu, 10, 1, 400, 1);
    try planCase(gpu, 47, 4, 600, 2);
    try planCase(gpu, 63, 6, 600, 3);
}

// -- the engine at tp=1/2 -------------------------------------------------------------------------------------------

const Ranks = struct {
    rank: usize,
    link: wire.Link,
    collective: Collective,
    seq: u64 = 10,

    fn open(gpu: Gpu, rank: usize, port: u16) !*Ranks {
        const r = try gpu.gpa.create(Ranks);
        errdefer gpu.gpa.destroy(r);
        const link = try wire.Link.open(gpu.io, "127.0.0.1", port, rank, 600_000);
        errdefer link.close();
        r.* = .{ .rank = rank, .link = link, .collective = try Collective.init(link, rank, 3_600_000) };
        return r;
    }

    fn close(r: *Ranks, gpa: std.mem.Allocator) void {
        r.collective.deinit();
        r.link.close();
        gpa.destroy(r);
    }

    fn swap(r: *Ranks, value: u64) !u64 {
        r.seq += 1;
        const until = wire.deadline(3_600_000);
        if (r.rank == 0) {
            const got = try r.link.receiveHeader(until);
            if (got.kind != .ready or got.sequence != r.seq) return error.RankOutOfStep;
            try r.link.sendHeader(.{ .kind = .ready, .sequence = r.seq, .arg = value }, until);
            return got.arg;
        }
        try r.link.sendHeader(.{ .kind = .ready, .sequence = r.seq, .arg = value }, until);
        const got = try r.link.receiveHeader(until);
        if (got.kind != .ready or got.sequence != r.seq) return error.RankOutOfStep;
        return got.arg;
    }
};

const Eng = struct {
    gpu: Gpu,
    rank: usize,
    world: usize,
    ranks: ?*Ranks,
    stream: cuda.Stream,
    glue: flash.Glue,
    e: *prompt.Engine,
    seq: prompt.Seq,
    out: cuda.DeviceBuffer,
    attached: xpass.Attached,

    fn init(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8) !*Eng {
        return initPaged(gpu, rank, world, port, dir, null);
    }

    fn initPaged(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, paging: ?X.kvpage.Config) !*Eng {
        const g = try gpu.gpa.create(Eng);
        errdefer gpu.gpa.destroy(g);
        g.gpu = gpu;
        g.rank = rank;
        g.world = world;
        g.ranks = if (world == 2) try Ranks.open(gpu, rank, port) else null;
        errdefer if (g.ranks) |ranks| ranks.close(gpu.gpa);
        g.stream = try cuda.Stream.init(gpu.d, true);
        errdefer g.stream.deinit();
        g.glue = try flash.Glue.load(gpu.d);
        errdefer g.glue.deinit();
        const t0 = check.now(gpu.io);
        g.e = try prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &g.stream, dir, world, rank);
        errdefer g.e.deinit();
        if (g.ranks) |r| g.e.collective = &r.collective;
        if (paging) |cfg| {
            g.e.page_config = cfg;
            try X.kv.setup(g.e, false, true);
        }
        _ = try g.e.reserve(@min(cap_default, g.e.chunk));
        _ = try g.e.reserveScores(cap_default, @min(cap_default, g.e.chunk));
        const common_before = cuda.usage(false).device;
        const common_bytes = try g.e.reserveSharedKv(cap_default);
        const common_resident = try std.math.sub(u64, cuda.usage(false).device, common_before);
        if (paging) |cfg| if (cfg.shared_lanes != 0 and rank == 1) {
            const arena = g.e.shared_arena orelse return error.SharedKvArenaRequired;
            const retired = try arena.register(cap_default);
            try arena.release(retired, g.stream);
        };
        g.seq = try g.e.newSeq(cap_default);
        errdefer g.seq.deinit(gpu.gpa);
        g.out = try cuda.DeviceBuffer.alloc(gpu.d, out_rows * vocab * 2);
        errdefer g.out.free();
        X.prefill.reserveFor(g.e, cap_default, world);
        var budget = try xpass.budgetFromEnv(0);
        if (paging) |_| {
            const layout = try X.kv.sequenceLayout(X.kv.of(g.e), cap_default, g.e.chunk, g.e.page_config);
            budget.other = try std.math.add(u64, budget.other, try std.math.mul(u64, layout.bytes.device(), 2));
            if (g.e.page_config.?.shared_lanes != 0) {
                const arena = g.e.shared_arena orelse return error.SharedKvArenaRequired;
                try check.expect(common_bytes == arena.layout.common_bytes and common_bytes == common_resident and common_bytes != 0, "rank {d}: shared common bytes charged once before expert pools", .{rank});
                std.debug.print("SHARED rank {d}/{d}: lanes {d}, cap {d}, window {d}+rows {d}, physical window rows {d}, slots {d}x4, K/V side rows {d}, device {d}, tables {d}, work {d}, common {d} bytes resident before pools; future two lane caches {d} bytes\n", .{ rank, world, arena.layout.options.lanes, cap_default, arena.layout.options.window, arena.layout.options.rows, arena.layout.window_rows, arena.layout.options.slots, arena.layout.side_rows, arena.layout.device_bytes, arena.layout.table_bytes, arena.layout.work_bytes, common_bytes, budget.other });
            } else try check.expect(common_bytes == 0 and g.e.shared_arena == null, "rank {d}: private control has no shared allocation", .{rank});
        }
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, g.e.layers[0..g.e.n_layers], null, g.e.chunk, budget);
        std.debug.print("RESULT rank {d}/{d} load: {d:.1} s; {d} lines a layer x {d} layers ({d:.2} GiB of pool)\n", .{ rank, world, ms(t0, check.now(gpu.io)) / 1e3, g.attached.lines, g.attached.layers, gib(g.attached.device_bytes) });
        return g;
    }

    fn deinit(g: *Eng) void {
        g.out.free();
        g.seq.deinit(g.gpu.gpa);
        g.e.deinit();
        g.glue.deinit();
        g.stream.deinit();
        if (g.ranks) |r| r.close(g.gpu.gpa);
        g.gpu.gpa.destroy(g);
    }

    fn tri(g: *Eng) flash.Tri {
        return .{ .own = &g.glue, .s = g.stream };
    }

    fn reset(g: *Eng) !void {
        const s = &g.seq;
        for ([_][]cuda.DeviceBuffer{ s.conv, s.state, s.kc, s.vc, s.ikc, s.pooled }) |bs| for (bs) |b| try b.fill8(0, g.stream.handle);
        try s.tail.fill8(0, g.stream.handle);
        try g.stream.synchronize();
        s.fed = 0;
        s.history = .{ g.e.eos, g.e.eos };
    }

    fn swap(g: *Eng, v: u64) !u64 {
        return if (g.ranks) |r| r.swap(v) else v;
    }

    fn forward(g: *Eng, ids: []const u32, hooks: prompt.batch.Hooks) !void {
        try g.e.forward(flash.Tri, g.tri(), &g.seq, ids, hooks);
        if (g.ranks) |r| try r.collective.finish(g.stream) else try g.stream.synchronize();
    }

    fn round(g: *Eng, tok: u32) !u32 {
        const q = [_]*prompt.Seq{&g.seq};
        var t = tok;
        try g.e.round(flash.Tri, g.tri(), &q, (&t)[0..1], g.out.ptr);
        if (g.ranks) |r| try r.collective.finish(g.stream);
        return prompt.argmax(g.seq.logits);
    }

    fn stats(g: *Eng) !fpool.Stats {
        try dplan.toHost();
        return xpass.poolStats(g.e.layers[0..g.e.n_layers], null);
    }
};

fn ints(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const items = v.array.items;
    const out = try a.alloc(u32, items.len);
    for (items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

const modes = [_]graph.Mode{ .off, .dplan, .graph };

const Run = struct {
    ids: []u32,
    /// Per decode step, the hash of the step's whole bf16 logits row; then per window row the same.
    round_h: []u64,
    window_h: []u64,
    ms_token: f64 = 0,
    ms_window_row: f64 = 0,
    hit: f64 = 0,
    copies: f64 = 0,
};

const win_sizes = [_]usize{ 3, 1, 4, 2, 8, 16 };

fn runMode(g: *Eng, ids: []const u32, steps: usize, r: *Run) !void {
    const a = g.gpu.gpa;
    // decode rounds
    try dplan.toHost();
    try g.reset();
    try g.forward(ids, .{ .logits = .last });
    var tok = prompt.argmax(g.seq.logits);
    r.ids[0] = tok;
    _ = try g.swap(1);
    const before = try g.stats();
    const t0 = check.now(g.gpu.io);
    for (0..steps) |i| {
        tok = try g.round(tok);
        r.ids[i + 1] = tok;
        r.round_h[i] = std.hash.Wyhash.hash(0, g.seq.logits);
    }
    const t1 = check.now(g.gpu.io);
    const after = try g.stats();
    r.ms_token = ms(t0, t1) / @as(f64, @floatFromInt(steps));
    const hits = after.hit_slots - before.hit_slots;
    const all = hits + after.miss_slots - before.miss_slots;
    r.hit = 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(all, 1)));
    r.copies = @as(f64, @floatFromInt(after.misses - before.misses)) / @as(f64, @floatFromInt(steps));
    // the same continuation teacher-forced in windows of 3, 1, 4, 2 rows (every row's logits)
    try dplan.toHost();
    try g.reset();
    try g.forward(ids, .{ .logits = .last });
    var at: usize = 0;
    var k: usize = 0;
    const host = try a.alloc(u16, 16 * vocab);
    defer a.free(host);
    const tw = check.now(g.gpu.io);
    while (at < steps) : (k += 1) {
        const n = @min(win_sizes[k % win_sizes.len], steps - at);
        try g.forward(r.ids[at..][0..n], .{ .logits = .all, .logits_out = g.out.ptr });
        try g.out.download(0, std.mem.sliceAsBytes(host[0 .. n * vocab]));
        for (0..n) |j| r.window_h[at + j] = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(host[j * vocab ..][0..vocab]));
        at += n;
    }
    r.ms_window_row = ms(tw, check.now(g.gpu.io)) / @as(f64, @floatFromInt(steps));
}

/// Two sequences in one decode round (two rows, each its own caches): `steps` rounds after both prompts, every round's
/// two logits rows hashed into `out`; ms per round.
fn pairRun(g: *Eng, b: *prompt.Seq, ids_a: []const u32, ids_b: []const u32, steps: usize, out: []u64) !f64 {
    try dplan.toHost();
    try g.reset();
    try g.forward(ids_a, .{ .logits = .last });
    for ([_][]cuda.DeviceBuffer{ b.conv, b.state, b.kc, b.vc, b.ikc, b.pooled }) |bs| for (bs) |x| try x.fill8(0, g.stream.handle);
    try b.tail.fill8(0, g.stream.handle);
    try g.stream.synchronize();
    b.fed = 0;
    b.history = .{ g.e.eos, g.e.eos };
    try g.e.forward(flash.Tri, g.tri(), b, ids_b, .{ .logits = .last });
    if (g.ranks) |r| try r.collective.finish(g.stream) else try g.stream.synchronize();
    var t = [2]u32{ prompt.argmax(g.seq.logits), prompt.argmax(b.logits) };
    const q = [_]*prompt.Seq{ &g.seq, b };
    const t0 = check.now(g.gpu.io);
    for (0..steps) |i| {
        try g.e.round(flash.Tri, g.tri(), &q, &t, g.out.ptr);
        if (g.ranks) |r| try r.collective.finish(g.stream);
        out[i] = std.hash.Wyhash.hash(0, g.seq.logits) ^ std.hash.Wyhash.hash(7, b.logits);
        t = .{ prompt.argmax(g.seq.logits), prompt.argmax(b.logits) };
    }
    return ms(t0, check.now(g.gpu.io)) / @as(f64, @floatFromInt(steps));
}

fn eqMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, golden_path: []const u8, steps: usize, which: []const u8) !void {
    const a = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, golden_path, a, .limited(1 << 26));
    defer a.free(text);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    var all_ok = true;
    for (parsed.value.object.get("prompts").?.array.items) |p| {
        const ids = try ints(a, p.object.get("ids").?);
        defer a.free(ids);
        var runs: [modes.len]Run = undefined;
        for (&runs) |*r| r.* = .{ .ids = try a.alloc(u32, steps + 1), .round_h = try a.alloc(u64, steps), .window_h = try a.alloc(u64, steps) };
        defer for (&runs) |*r| {
            a.free(r.ids);
            a.free(r.round_h);
            a.free(r.window_h);
        };
        // warm the pools once (first-touch copies are not any mode's cost)
        graph.force = .off;
        try runMode(g, ids, steps, &runs[0]);
        for (modes, 0..) |m, mi| {
            if (std.mem.indexOf(u8, which, @tagName(m)) == null and m != .off) continue;
            graph.force = m;
            const g0 = graph.stats;
            try runMode(g, ids, steps, &runs[mi]);
            const g1 = graph.stats;
            const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(runs[mi].round_h)) ^ std.hash.Wyhash.hash(1, std.mem.sliceAsBytes(runs[mi].window_h));
            const other = try g.swap(h);
            if (world == 2) try check.expect(other == h, "rank {d}: the two ranks' logits differ ({s})", .{ rank, @tagName(m) });
            const rs = runs[mi];
            var same_r: usize = 0;
            var same_w: usize = 0;
            for (rs.round_h, runs[0].round_h) |x, y| same_r += @intFromBool(x == y);
            for (rs.window_h, runs[0].window_h) |x, y| same_w += @intFromBool(x == y);
            const ok = same_r == steps and same_w == steps and std.mem.eql(u32, rs.ids, runs[0].ids);
            if (mi != 0) all_ok = all_ok and ok;
            if (rank == 0) std.debug.print("RESULT tp={d} {s} {s}: decode {d:.2} ms/token (hit {d:.1}%, {d:.1} copies a token), windows {d:.2} ms/row; logits bit-equal to off: rounds {d}/{d}, window rows {d}/{d}; greedy ids {s}; graph captures {d} (updates {d}) replays {d}, eager-dplan steps {d}, declined {d}, capture {d:.1} ms\n", .{ world, p.object.get("name").?.string, @tagName(m), rs.ms_token, rs.hit, rs.copies, rs.ms_window_row, same_r, steps, same_w, steps, if (std.mem.eql(u32, rs.ids, runs[0].ids)) "equal" else "DIFFER", g1.captures - g0.captures, g1.updates - g0.updates, g1.replays - g0.replays, g1.eager_dplan - g0.eager_dplan, g1.declined - g0.declined, @as(f64, @floatFromInt(g1.capture_ns - g0.capture_ns)) / 1e6 });
        }
        graph.force = null;
    }
    // two streams in one round
    {
        const ps = parsed.value.object.get("prompts").?.array.items;
        const ia = try ints(a, ps[0].object.get("ids").?);
        defer a.free(ia);
        const ib = try ints(a, ps[ps.len - 1].object.get("ids").?);
        defer a.free(ib);
        var b = try g.e.newSeq(cap_default);
        defer b.deinit(a);
        const pair_steps = @min(steps, 32);
        var ref: [64]u64 = undefined;
        var got: [64]u64 = undefined;
        graph.force = .off;
        _ = try pairRun(g, &b, ia, ib, pair_steps, &ref);
        for (modes) |m| {
            if (std.mem.indexOf(u8, which, @tagName(m)) == null and m != .off) continue;
            graph.force = m;
            const t = try pairRun(g, &b, ia, ib, pair_steps, &got);
            var same: usize = 0;
            for (got[0..pair_steps], ref[0..pair_steps]) |x, y| same += @intFromBool(x == y);
            if (m != .off) all_ok = all_ok and same == pair_steps;
            const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(got[0..pair_steps]));
            if (world == 2) try check.expect(try g.swap(h) == h, "rank {d}: the two ranks' two-stream logits differ", .{rank});
            if (rank == 0) std.debug.print("RESULT tp={d} two streams {s}: {d:.2} ms a round of 2 rows; both rows' logits bit-equal to off in {d}/{d} rounds\n", .{ world, @tagName(m), t, same, pair_steps });
        }
        graph.force = null;
    }
    if (dplan.current()) |dp| if (rank == 0) std.debug.print("RESULT dplan owner switches: {d} pushes, {d} pulls; {d} experts read in place (rounds with more misses than free lines)\n", .{ dp.pushes, dp.pulls, dp.inplace });
    try check.expect(all_ok, "rank {d}: a device-plan or graph step's logits differ from the host cut's", .{rank});
    check.pass("rank {d}/{d}: decode rounds and verify windows on the device plan and in graphs are bit-identical to the host cut ({d} steps a prompt)", .{ rank, world, steps });
}

/// A long prompt through the deep prefill (TF_FLASHNEXT_EXL3_PREFILL_ROWS sets its chunks), then `steps` greedy
/// rounds, twice per mode; every run's last-row logits and decode logits hashed and compared with the first eager run.
fn prefillMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, n: usize, steps: usize) !void {
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
        try dplan.toHost();
        try g.reset();
        _ = try g.swap(3);
        const p0 = graph.prefill_stats;
        const f0 = X.prefill.stats;
        const t0 = check.now(gpu.io);
        try g.forward(ids, .{ .logits = .last });
        const t1 = check.now(gpu.io);
        const p1 = graph.prefill_stats;
        const f1 = X.prefill.stats;
        try check.expect(f1.chunks > f0.chunks and f1.rows >= f0.rows + n, "rank {d}: deep prefill ran instead of falling back", .{rank});
        if (m == .graph) try check.expect(p1.captures + p1.updates + p1.replays > p0.captures + p0.updates + p0.replays, "rank {d}: prefill graph captured or replayed a layer", .{rank});
        hs[0] = std.hash.Wyhash.hash(0, g.seq.logits);
        var tok = prompt.argmax(g.seq.logits);
        for (0..steps) |i| {
            tok = try g.round(tok);
            hs[i + 1] = std.hash.Wyhash.hash(0, g.seq.logits);
        }
        if (g.seq.paged) |pg| {
            try g.stream.synchronize();
            const pager = try pg.total();
            try check.expect(!pager.overflow and pager.epoch > 0 and pager.calls > 0 and pager.rebases > 0, "rank {d}: paged prefill/decode advances epochs and crosses a window without overflow", .{rank});
            std.debug.print("PAGER rank {d}/{d} {s}: device epoch {d}, calls {d}, rebases {d}, written rows {d}, misses {d}\n", .{ rank, world, @tagName(m), pager.epoch, pager.calls, pager.rebases, pager.written_rows, pager.misses });
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
/// The profiler window excludes graph warm-up and host setup; no per-layer synchronization is added.
fn prefillProfMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, n: usize, which: []const u8) !void {
    var stage: []const u8 = "engine initialization";
    errdefer std.debug.print("prefill-prof rank={d} failed during {s}\n", .{ rank, stage });
    const a = gpu.gpa;
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    defer graph.force = null;
    try check.expect(g.seq.paged != null, "prefill profiler requires BF16 KV paging", .{});
    const ids = try a.alloc(u32, n);
    defer a.free(ids);
    var prng = std.Random.DefaultPrng.init(0x9e3779b9);
    for (ids) |*x| x.* = 200 + prng.random().uintLessThan(u32, 60000);
    const use_graphs = std.mem.eql(u8, which, "graph");
    if (!use_graphs and !std.mem.eql(u8, which, "eager")) return error.BadMode;
    graph.force = if (use_graphs) .graph else .off;
    try dplan.toHost();
    stage = "graph warm-up";
    // Warm paging and, for graph mode, capture each layer shape before the measured forward.
    for (0..if (use_graphs) @as(usize, 2) else 1) |_| {
        try g.reset();
        _ = try g.swap(3);
        try g.forward(ids, .{ .logits = .last });
        try g.stream.synchronize();
    }
    const before = graph.prefill_stats;
    const f_before = X.prefill.stats;
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
    const f_after = X.prefill.stats;
    if (use_graphs and after.replays == before.replays) return error.NoPrefillGraphReplay;
    const hash = std.hash.Wyhash.hash(0, g.seq.logits);
    stage = "rank hash check";
    if (world == 2) try check.expect(try g.swap(hash) == hash, "rank {d}: profiled prefill logits differ across ranks", .{rank});
    std.debug.print("RESULT prefill-prof rank={d}/{d} mode={s} rows={d}: {d:.1} ms wall, graphs replayed {d}, eager {d}, captured {d}, updated {d}; expert copies {d} (prefetch {d}), passes {d}, pick wait {d:.1} ms, plan {d:.1} ms; logits {x}; Nsight range excludes warm-up\n", .{ rank, world, which, n, ms(t0, t1), after.replays - before.replays, after.eager - before.eager, after.captures - before.captures, after.updates - before.updates, f_after.copies - f_before.copies, f_after.prefetched - f_before.prefetched, f_after.passes - f_before.passes, @as(f64, @floatFromInt(f_after.host_wait_ns - f_before.host_wait_ns)) / 1e6, @as(f64, @floatFromInt(f_after.plan_ns - f_before.plan_ns)) / 1e6, hash });
}

fn expectSnapshotState(rank: usize, previous: *const flash.snapshot.Snap, saved: *const flash.snapshot.Snap) !void {
    const canonical = saved.paged.?.layout.canonical;
    const state_offset = saved.paged.?.layout.total;
    try check.expect(previous.at == saved.at and std.meta.eql(previous.history, saved.history), "rank {d}: exact snapshot position and history", .{rank});
    try check.expect(std.mem.eql(u8, previous.host.?.bytes[0..canonical], saved.host.?.bytes[0..canonical]), "rank {d}: exact canonical BF16 history", .{rank});
    try check.expect(std.mem.eql(u8, previous.host.?.bytes[state_offset..previous.buf.len], saved.host.?.bytes[state_offset..saved.buf.len]), "rank {d}: exact recurrent/index/PLE state and logits", .{rank});
}

fn expectOwnedRestore(engine: *Eng, sequence: *prompt.Seq, wanted: *const flash.snapshot.Snap) !void {
    const saved = try flash.snapshot.save(engine.gpu.gpa, engine.gpu.d, engine.stream, sequence, null);
    defer flash.snapshot.drop(engine.gpu.gpa, engine.stream, saved);
    try expectSnapshotState(engine.rank, wanted, saved);
    try check.expect(std.meta.eql(wanted.paged, saved.paged) and std.mem.eql(u8, wanted.host.?.bytes[0..wanted.buf.len], saved.host.?.bytes[0..saved.buf.len]), "rank {d}: exact destination windows, marks, controls and snapshot metadata", .{engine.rank});
    const previous = wanted.owned_blocks orelse return error.SharedKvSnapshotRequired;
    const current = saved.owned_blocks orelse return error.SharedKvSnapshotRequired;
    try check.expect(previous.entries.len > 0 and previous.entries.len == current.entries.len, "rank {d}: restore retains every owned resident entry, not a cold cache", .{engine.rank});
    try check.expect(std.meta.eql(current.source, sequence.paged.?.shared_sequence), "rank {d}: restored slots belong to destination generation", .{engine.rank});
    const block_bytes = try previous.geometry.blockBytes();
    for (previous.entries, 0..) |entry, old_index| {
        const new_index = for (current.entries, 0..) |candidate, index| {
            if (std.meta.eql(entry.key, candidate.key)) break index;
        } else return error.SharedKvSnapshotMissingEntry;
        try check.expect(previous.clock - entry.stamp == current.clock - current.entries[new_index].stamp, "rank {d}: owned entry retains relative LRU age", .{engine.rank});
        try check.expect(std.mem.eql(u8, previous.payload[old_index * block_bytes ..][0..block_bytes], current.payload[new_index * block_bytes ..][0..block_bytes]), "rank {d}: owned entry remaps exact K/V payload", .{engine.rank});
    }
}

const SharedSnapshotProbe = struct {
    sequence: prompt.Seq,
    saved: *flash.snapshot.Snap,
    continued: ?*flash.snapshot.Snap = null,
    key: [6]u64,
    buffers: [3]u64,

    fn init(engine: *Eng, ids: []const u32) !*SharedSnapshotProbe {
        const probe = try engine.gpu.gpa.create(SharedSnapshotProbe);
        errdefer engine.gpu.gpa.destroy(probe);
        probe.* = .{ .sequence = try engine.e.newSeq(cap_default), .saved = undefined, .key = undefined, .buffers = undefined };
        errdefer probe.sequence.deinit(engine.gpu.gpa);
        try engine.e.forward(flash.Tri, engine.tri(), &probe.sequence, ids, .{ .logits = .last });
        probe.saved = try flash.snapshot.save(engine.gpu.gpa, engine.gpu.d, engine.stream, &probe.sequence, null);
        errdefer flash.snapshot.drop(engine.gpu.gpa, engine.stream, probe.saved);
        const arena = engine.e.shared_arena orelse return error.SharedKvArenaRequired;
        try check.expect(probe.saved.version == 2 and probe.saved.owned_blocks.?.entries.len > 0, "rank {d}: foreign probe holds shared resident blocks", .{engine.rank});
        probe.key = probe.sequence.paged.?.graphKey();
        probe.buffers = .{ arena.device.ptr, arena.tables.ptr, arena.work.ptr };
        return probe;
    }

    fn deinit(probe: *SharedSnapshotProbe, engine: *Eng) void {
        if (probe.continued) |saved| flash.snapshot.drop(engine.gpu.gpa, engine.stream, saved);
        flash.snapshot.drop(engine.gpu.gpa, engine.stream, probe.saved);
        probe.sequence.deinit(engine.gpu.gpa);
        engine.gpu.gpa.destroy(probe);
    }

    fn unchanged(probe: *SharedSnapshotProbe, engine: *Eng) !void {
        const arena = engine.e.shared_arena.?;
        try check.expect(std.meta.eql(probe.buffers, [3]u64{ arena.device.ptr, arena.tables.ptr, arena.work.ptr }), "rank {d}: arena addresses survive restore and sequence churn", .{engine.rank});
        try check.expect(std.meta.eql(probe.key, probe.sequence.paged.?.graphKey()), "rank {d}: foreign graph bindings remain legal after restore", .{engine.rank});
        const saved = try flash.snapshot.save(engine.gpu.gpa, engine.gpu.d, engine.stream, &probe.sequence, null);
        defer flash.snapshot.drop(engine.gpu.gpa, engine.stream, saved);
        try check.expect(std.meta.eql(probe.saved.paged, saved.paged), "rank {d}: foreign window origins and counters unchanged", .{engine.rank});
        try check.expect(probe.saved.at == saved.at and std.meta.eql(probe.saved.history, saved.history), "rank {d}: foreign logical position unchanged", .{engine.rank});
        try check.expect(std.mem.eql(u8, probe.saved.host.?.bytes[0..probe.saved.buf.len], saved.host.?.bytes[0..saved.buf.len]), "rank {d}: restore preserves foreign canonical history, full windows, marks, controls, ordinary state and logits", .{engine.rank});
    }

    fn continueAndRestore(probe: *SharedSnapshotProbe, engine: *Eng) !void {
        try engine.e.forward(flash.Tri, engine.tri(), &probe.sequence, &.{ 417, 819, 1237 }, .{ .logits = .last });
        const saved = try flash.snapshot.save(engine.gpu.gpa, engine.gpu.d, engine.stream, &probe.sequence, null);
        if (probe.continued) |previous| {
            defer flash.snapshot.drop(engine.gpu.gpa, engine.stream, saved);
            try expectSnapshotState(engine.rank, previous, saved);
        } else probe.continued = saved;
        try flash.snapshot.restore(engine.stream, &probe.sequence, null, probe.saved);
        try expectOwnedRestore(engine, &probe.sequence, probe.saved);
        try probe.unchanged(engine);
    }
};

const SnapshotStorage = enum { legacy, private, shared };

fn snapshotStorageMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, steps: usize, storage: SnapshotStorage) !void {
    if (world != 2) return error.SharedSnapshotRequiresTp2;
    if (steps < 8 or steps > 64) return error.SharedSnapshotDecodeSteps;
    if (X.kvpage.force_off) return error.SharedKvPagingDisabled;
    var paging = X.kvpage.configFromEnv() orelse X.kvpage.Config{ .shared_lanes = 8 };
    if (storage == .shared) {
        if (paging.shared_lanes < 3) return error.SharedSnapshotNeedsThreeLanes;
    } else paging.shared_lanes = 0;
    paging.slots = 1024;
    paging.window = 4096;
    paging.rows = 4096;
    if ((X.prefill.options orelse X.prefill.optionsFromEnv()).serve_rows > paging.rows) return error.SharedSnapshotPrefillRows;
    try paging.validate();
    const previous = X.kvpage.force;
    X.kvpage.force = paging;
    defer X.kvpage.force = previous;
    return snapshotMode(gpu, rank, world, port, dir, steps, storage);
}

fn snapshotMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, steps: usize, storage: SnapshotStorage) !void {
    const snaps = flash.snapshot;
    const allocator = gpu.gpa;
    const shared = storage == .shared;
    const paired = storage != .legacy;
    const g = if (paired) try Eng.initPaged(gpu, rank, world, port, dir, X.kvpage.force.?) else try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    defer graph.force = null;
    try check.expect(g.seq.paged != null, "snapshot test requires BF16 paging", .{});
    if (paired) try check.expect(g.seq.kv8 == null and g.seq.paged.?.cfg.slots == 1024 and g.seq.paged.?.cfg.rows == 4096, "rank {d}: paired fixture keeps fixed BF16 paging geometry", .{rank});
    if (shared) try check.expect(g.seq.paged.?.shared_arena == g.e.shared_arena and g.seq.paged.?.cfg.shared_lanes == X.kvpage.force.?.shared_lanes and g.seq.paged.?.cfg.slots == 1024 and g.seq.paged.?.cfg.rows == 4096, "rank {d}: shared snapshot fixture cannot silently use private or different paging geometry", .{rank});
    if (shared) {
        const owner = g.seq.paged.?.shared_sequence;
        const peer_generation = try g.swap(owner.generation);
        try check.expect(owner.generation != peer_generation, "rank {d}: shared TP2 fixture requires legal cross-rank generation asymmetry", .{rank});
        const logical = g.seq.paged.?.logicalGraphKey();
        const logical_hash = std.hash.Wyhash.hash(0, std.mem.asBytes(&logical));
        try check.expect(try g.swap(logical_hash) == logical_hash, "rank {d}: asymmetric physical generations retain identical logical TP geometry", .{rank});
        std.debug.print("SHARED rank {d}/{d}: primary lane {d} generation {d}, peer generation {d}; logical TP key {x}\n", .{ rank, world, owner.lane, owner.generation, peer_generation, logical_hash });
    }
    if (storage == .private) try check.expect(g.seq.paged.?.cfg.shared_lanes == 0 and g.seq.paged.?.shared_arena == null, "rank {d}: private oracle uses independent per-sequence slots", .{rank});
    var second = try g.e.newSeq(cap_default);
    var second_live = true;
    defer if (second_live) second.deinit(allocator);
    const ids = try allocator.alloc(u32, if (paired) 24576 else 16384);
    defer allocator.free(ids);
    var random = std.Random.DefaultPrng.init(0x736e6170);
    for (ids) |*token| token.* = 200 + random.random().uintLessThan(u32, 60000);
    if (paired) std.debug.print("SNAPSHOT-ORACLE store={s} rank={d}/{d} seed=736e6170 ids={x:0>16} first=8707 second=4096+8709 continuation=8192 decode={d}; compare matching rank/run/lane hashes between snapshot-private and snapshot-shared, not a golden\n", .{ @tagName(storage), rank, world, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ids)), steps });
    graph.force = .graph;
    const first_end: usize = if (paired) 8707 else 3075;
    const second_end: usize = if (paired) 12805 else 6917;
    try g.forward(ids[0..first_end], .{ .logits = .last });
    var first_saved: *snaps.Snap = undefined;
    if (paired) first_saved = try snaps.save(allocator, gpu.d, g.stream, &g.seq, null);
    defer if (paired) snaps.drop(allocator, g.stream, first_saved);
    try g.e.forward(flash.Tri, g.tri(), &second, ids[4096..second_end], .{ .logits = .last });
    if (!paired) first_saved = try snaps.save(allocator, gpu.d, g.stream, &g.seq, null);
    defer if (!paired) snaps.drop(allocator, g.stream, first_saved);
    const second_saved = try snaps.save(allocator, gpu.d, g.stream, &second, null);
    defer snaps.drop(allocator, g.stream, second_saved);
    const source_key = second.paged.?.graphKey();
    const probe: ?*SharedSnapshotProbe = if (shared) try SharedSnapshotProbe.init(g, ids[14336..23045]) else null;
    defer if (probe) |foreign| foreign.deinit(g);
    const expected = try allocator.alloc(u8, (steps + 1) * 2 * vocab * 2);
    defer allocator.free(expected);
    var reference: [2]?*snaps.Snap = .{ null, null };
    defer for (reference) |item| if (item) |saved| snaps.drop(allocator, g.stream, saved);
    for ([_]graph.Mode{ .off, .graph, .graph }, 0..) |mode, run_index| {
        graph.force = mode;
        try dplan.toHost();
        g.e.round_graph.clear();
        if (shared) {
            const previous = second.paged.?.shared_sequence;
            const window = second.paged.?.devRows(0, false);
            second_live = false;
            second.deinit(allocator);
            second = try g.e.newSeq(cap_default);
            second_live = true;
            const current = second.paged.?.shared_sequence;
            try check.expect(previous.arena == current.arena and previous.lane == current.lane and current.generation > previous.generation and current.generation > second_saved.owned_blocks.?.source.generation, "rank {d}: saved source restores into recycled lane with a new generation", .{rank});
            try check.expect(window == second.paged.?.devRows(0, false), "rank {d}: recycled lane retains its fixed window address", .{rank});
            try probe.?.unchanged(g);
            std.debug.print("SHARED rank {d}/{d}: recycled lane {d}, source generation {d}, destination generation {d}, stable window {x}\n", .{ rank, world, current.lane, second_saved.owned_blocks.?.source.generation, current.generation, window });
        }
        try snaps.restore(g.stream, &g.seq, null, first_saved);
        if (shared) {
            try expectOwnedRestore(g, &g.seq, first_saved);
            try probe.?.unchanged(g);
        }
        try snaps.restore(g.stream, &second, null, second_saved);
        if (shared) {
            try expectOwnedRestore(g, &second, second_saved);
            try check.expect(source_key[4] != second.paged.?.graphKey()[4], "rank {d}: identical restored geometry at same window address has a different strict graph generation key", .{rank});
            try probe.?.unchanged(g);
        }
        const before = graph.stats;
        const prefill_before = graph.prefill_stats;
        const deep_before = X.prefill.stats;
        try g.forward(ids[first_end..][0..8192], .{ .logits = .last });
        try g.e.forward(flash.Tri, g.tri(), &second, ids[second_end..][0..8192], .{ .logits = .last });
        if (paired) try check.expect(X.prefill.stats.chunks > deep_before.chunks and X.prefill.stats.rows >= deep_before.rows + 16384, "rank {d}: both paired lanes actually execute deep prefill", .{rank});
        var tokens = [_]u32{ prompt.argmax(g.seq.logits), prompt.argmax(second.logits) };
        const sequences = [_]*prompt.Seq{ &g.seq, &second };
        for (0..steps + 1) |step| {
            if (step > 0) {
                try g.e.round(flash.Tri, g.tri(), &sequences, &tokens, g.out.ptr);
                tokens = .{ prompt.argmax(g.seq.logits), prompt.argmax(second.logits) };
            }
            for (sequences, 0..) |sequence, lane| {
                const wanted = expected[(step * 2 + lane) * vocab * 2 ..][0 .. vocab * 2];
                if (run_index == 0) @memcpy(wanted, sequence.logits) else try check.expect(std.mem.eql(u8, wanted, sequence.logits), "rank {d} snapshot {s} lane {d} step {d} BF16 logits", .{ rank, @tagName(mode), lane, step });
            }
            if (paired) {
                for (sequences, 0..) |sequence, lane| std.debug.print("SNAPSHOT-ORACLE store={s} rank={d}/{d} mode={s} run={d} lane={d} step={d} logits={x:0>16}\n", .{ @tagName(storage), rank, world, @tagName(mode), run_index, lane, step, std.hash.Wyhash.hash(0, sequence.logits) });
                const digest = std.hash.Wyhash.hash(0, g.seq.logits) ^ std.hash.Wyhash.hash(1, second.logits);
                try check.expect(try g.swap(digest) == digest, "rank {d}: paired TP2 logits agree at step {d}", .{ rank, step });
            }
        }
        for (sequences, 0..) |sequence, lane| {
            const saved = try snaps.save(allocator, gpu.d, g.stream, sequence, null);
            if (paired) {
                const canonical_hash = std.hash.Wyhash.hash(0, saved.host.?.bytes[0..saved.paged.?.layout.canonical]);
                const state_hash = std.hash.Wyhash.hash(1, saved.host.?.bytes[saved.paged.?.layout.total..saved.buf.len]);
                const history_hash = std.hash.Wyhash.hash(2, std.mem.asBytes(&saved.history)) ^ std.hash.Wyhash.hash(3, std.mem.asBytes(&saved.at));
                std.debug.print("SNAPSHOT-ORACLE store={s} rank={d}/{d} mode={s} run={d} lane={d} at={d} canonical={x:0>16} state-logits={x:0>16} history={x:0>16} full={x:0>16}\n", .{ @tagName(storage), rank, world, @tagName(mode), run_index, lane, saved.at, canonical_hash, state_hash, history_hash, canonical_hash ^ state_hash ^ history_hash });
            }
            if (run_index == 0) {
                reference[lane] = saved;
            } else {
                defer snaps.drop(allocator, g.stream, saved);
                const previous = reference[lane].?;
                const canonical = saved.paged.?.layout.canonical;
                const state_offset = saved.paged.?.layout.total;
                try check.expect(std.mem.eql(u8, previous.host.?.bytes[0..canonical], saved.host.?.bytes[0..canonical]), "rank {d} snapshot lane {d} exact canonical BF16 history", .{ rank, lane });
                try check.expect(std.mem.eql(u8, previous.host.?.bytes[state_offset..previous.buf.len], saved.host.?.bytes[state_offset..saved.buf.len]), "rank {d} snapshot lane {d} exact recurrent/index/PLE state", .{ rank, lane });
            }
        }
        if (mode == .graph) {
            try check.expect(graph.prefill_stats.replays > prefill_before.replays, "rank {d} prefill graph replay after restore", .{rank});
            try check.expect(graph.stats.replays > before.replays, "rank {d} concurrent decode graph replay after restore", .{rank});
        }
        if (probe) |foreign| {
            for (sequences) |sequence| {
                const paged = sequence.paged.?;
                const status = try paged.total();
                try check.expect(!status.overflow and status.misses > 0 and status.rebases > 0, "rank {d}: shared paging actually fetches blocks and rebases without overflow", .{rank});
                for (paged.win0) |origin| try check.expect(origin > 0 and try X.kvpage.pagingRows(paged.cfg, origin, 256) < 256, "rank {d}: slots1024 forces paging subdivision", .{rank});
            }
            try foreign.unchanged(g);
            try foreign.continueAndRestore(g);
        }
        std.debug.print("SNAPSHOT rank {d}/{d} {s}: two paged lanes, {d} decode steps, exact state and logits; prefill replays {d}, decode replays {d}\n", .{ rank, world, @tagName(mode), steps, graph.prefill_stats.replays - prefill_before.replays, graph.stats.replays - before.replays });
    }
    if (shared) check.pass("rank {d}/{d}: shared slots1024 TP2 snapshot remapping, recycled generation, foreign windows/history/continuation, exact eager/graph state/logits and actual prefill/decode replays", .{ rank, world }) else check.pass("rank {d}/{d}: concurrent paged snapshots preserve BF16 continuation and graph prefill/decode", .{ rank, world });
}

/// cuProfilerStart/Stop from libcuda (nsys --capture-range=cudaProfilerApi), looked up at run time.
const Profiler = struct {
    lib: std.DynLib,
    start: *const fn () callconv(.c) c_int,
    stop: *const fn () callconv(.c) c_int,
    fn open() !Profiler {
        var lib = try std.DynLib.open("libcuda.so.1");
        return .{ .start = lib.lookup(*const fn () callconv(.c) c_int, "cuProfilerStart") orelse return error.MissingSymbol, .stop = lib.lookup(*const fn () callconv(.c) c_int, "cuProfilerStop") orelse return error.MissingSymbol, .lib = lib };
    }
};

/// A profiling window for nsys: `steps` decode rounds (one row) then `steps` verify windows of `rows` rows, after a
/// warm-up that captures their graphs (TF_FLASHNEXT_EXL3_GRAPH_AFTER=1 recommended), inside cuProfilerStart/Stop.
fn profMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, steps: usize, rows: usize, which: []const u8) !void {
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
        try dplan.toHost();
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
fn ngramMode(gpu: Gpu, dir: []const u8) !void {
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

/// A scheduling change's bits: every golden prompt's decode rounds and windows (runMode) under `which` (off, dplan,
/// graph) with the parallel projection parts (X.fork) off, off again and on; logits compared bit for bit, ms/token.
fn forkMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, golden_path: []const u8, steps: usize, which: []const u8) !void {
    const a = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, golden_path, a, .limited(1 << 26));
    defer a.free(text);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    graph.force = if (std.mem.eql(u8, which, "graph")) .graph else if (std.mem.eql(u8, which, "dplan")) .dplan else .off;
    defer graph.force = null;
    defer X.fork.force = null;
    var ok = true;
    for (parsed.value.object.get("prompts").?.array.items) |p| {
        const ids = try ints(a, p.object.get("ids").?);
        defer a.free(ids);
        var runs: [3]Run = undefined;
        for (&runs) |*r| r.* = .{ .ids = try a.alloc(u32, steps + 1), .round_h = try a.alloc(u64, steps), .window_h = try a.alloc(u64, steps) };
        defer for (&runs) |*r| {
            a.free(r.ids);
            a.free(r.round_h);
            a.free(r.window_h);
        };
        for ([_]bool{ false, false, true }, 0..) |on, i| {
            X.fork.force = on;
            try runMode(g, ids, steps, &runs[i]);
        }
        const same = std.mem.eql(u64, runs[1].round_h, runs[2].round_h) and std.mem.eql(u64, runs[1].window_h, runs[2].window_h) and std.mem.eql(u32, runs[1].ids, runs[2].ids);
        ok = ok and same;
        if (world == 2) {
            const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(runs[2].round_h));
            try check.expect(try g.swap(h) == h, "rank {d}: the ranks differ", .{rank});
        }
        if (rank == 0) std.debug.print("RESULT tp={d} {s} {s}: decode {d:.2} -> {d:.2} ms/token, windows {d:.2} -> {d:.2} ms/row (serial -> parallel parts); logits {s}\n", .{ world, which, p.object.get("name").?.string, runs[1].ms_token, runs[2].ms_token, runs[1].ms_window_row, runs[2].ms_window_row, if (same) "bit-equal" else "DIFFER" });
    }
    try check.expect(ok, "rank {d}: parallel parts changed the logits", .{rank});
    check.pass("rank {d}/{d}: parallel projection parts give the same bits ({s})", .{ rank, world, which });
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len >= 2 and std.mem.eql(u8, rest[0], "ngram")) return ngramMode(gpu, rest[1]);
    if (rest.len < 1) return error.MissingArgument;
    if (std.mem.eql(u8, rest[0], "plan")) return planMode(gpu);
    if (rest.len < 6) return error.MissingArgument;
    const rank = try std.fmt.parseInt(usize, rest[0], 10);
    const world = try std.fmt.parseInt(usize, rest[1], 10);
    const port = try std.fmt.parseInt(u16, rest[2], 10);
    const dir = rest[3];
    const mode = rest[4];
    if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
    const steps = if (rest.len > 6 and !std.mem.eql(u8, mode, "prefill-prof")) try std.fmt.parseInt(usize, rest[6], 10) else 48;
    const which: []const u8 = if (rest.len > 7) rest[7] else "dplan,graph";
    if (std.mem.eql(u8, mode, "snapshot")) return snapshotMode(gpu, rank, world, port, dir, steps, .legacy);
    if (std.mem.eql(u8, mode, "snapshot-shared")) return snapshotStorageMode(gpu, rank, world, port, dir, steps, .shared);
    if (std.mem.eql(u8, mode, "snapshot-private")) return snapshotStorageMode(gpu, rank, world, port, dir, steps, .private);
    if (std.mem.eql(u8, mode, "fork")) return forkMode(gpu, rank, world, port, dir, rest[5], steps, which);
    if (std.mem.eql(u8, mode, "prof")) return profMode(gpu, rank, world, port, dir, try std.fmt.parseInt(usize, rest[5], 10), steps, which);
    if (std.mem.eql(u8, mode, "prefill")) return prefillMode(gpu, rank, world, port, dir, try std.fmt.parseInt(usize, rest[5], 10), steps);
    if (std.mem.eql(u8, mode, "prefill-prof")) return prefillProfMode(gpu, rank, world, port, dir, try std.fmt.parseInt(usize, rest[5], 10), if (rest.len > 6) rest[6] else "graph");
    if (std.mem.eql(u8, mode, "eq")) return eqMode(gpu, rank, world, port, dir, rest[5], steps, which);
    return error.UnknownMode;
}
