//! exl3-tpprefill: the expert-major deep-chunk prefill (cuda_exl3_prefill.zig) at tp=1 and at tp=2 expert parallel
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");

const Gpu = check.Gpu;
const prompt = flash.prompt;
const X = flash.exl3;
const xpass = X.pass;
const xpre = X.prefill;
const wire = flash.rank_wire;
const Collective = flash.collective.Collective;
const vocab = prompt.vocabulary;

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn ms(t0: i96, t1: i96) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) / 1e6;
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

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

    fn init(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, cap: usize, deep_rows: usize) !*Eng {
        const g = try gpu.gpa.create(Eng);
        errdefer gpu.gpa.destroy(g);
        g.gpu = gpu;
        g.rank = rank;
        g.world = world;
        g.ranks = if (world == 2) try Ranks.open(gpu, rank, port) else null;
        g.stream = try cuda.Stream.init(gpu.d, true);
        g.glue = try flash.Glue.load(gpu.d);
        const t0 = check.now(gpu.io);
        g.e = try prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &g.stream, dir, world, rank);
        if (g.ranks) |r| g.e.collective = &r.collective;
        var free: [6]usize = undefined;
        var total: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free[0], &total), "cuMemGetInfo");
        _ = try g.e.reserve(@min(cap, g.e.chunk));
        _ = try g.e.reserveScores(cap, @min(cap, g.e.chunk));
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free[1], &total), "cuMemGetInfo");
        g.seq = try g.e.newSeq(cap);
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free[2], &total), "cuMemGetInfo");
        g.out = try cuda.DeviceBuffer.alloc(gpu.d, vocab * 2);
        if (deep_rows > 0) try xpre.reserveAt(gpu.gpa, gpu.d, &g.e.exl3_prefill, g.e, deep_rows, world);
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free[3], &total), "cuMemGetInfo");
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, &g.e.exl3_prefill, g.e.layers[0..g.e.n_layers], null, g.e.chunk, try xpass.budgetFromEnv(0), null);
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free[4], &total), "cuMemGetInfo");
        std.debug.print("MEM rank {d}/{d}: loaded in {d:.1} s; card {d:.2} GiB; free after weights {d:.2}; pass scratch (chunk {d}) + scores {d:.2} GiB; caches for {d} tokens {d:.2} GiB; deep chunks of {d} rows {d:.2} GiB (layout {d:.2}); pool {d} lines a layer x {d} layers = {d:.2} GiB (line {d:.2} MiB, ring {d} slots); {d:.2} GiB free after; head columns {d}\n", .{ rank, world, ms(t0, check.now(gpu.io)) / 1e3, gib(total), gib(free[0]), g.e.chunk, gib(free[0] - free[1]), cap, gib(free[1] - free[2]), deep_rows, gib(free[2] - free[3]), gib(xpre.deviceBytesAt(deep_rows, world)), g.attached.lines, g.attached.layers, gib(g.attached.device_bytes), @as(f64, @floatFromInt(g.attached.line_bytes)) / (1 << 20), g.attached.lines * 48, gib(free[4]), g.e.head_q.n });
        return g;
    }

    fn deinit(g: *Eng) void {
        xpre.release(&g.e.exl3_prefill);
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

    fn finish(g: *Eng) !void {
        if (g.ranks) |r| try r.collective.finish(g.stream) else try g.stream.synchronize();
    }

    fn forward(g: *Eng, ids: []const u32) !void {
        try g.e.forward(flash.Tri, g.tri(), &g.seq, ids, .{ .logits = .last });
        try g.finish();
    }

    fn round(g: *Eng, tok: u32) !u32 {
        const q = [_]*prompt.Seq{&g.seq};
        var t = tok;
        try g.e.round(flash.Tri, g.tri(), &q, (&t)[0..1], g.out.ptr);
        try g.finish();
        return prompt.argmax(g.seq.logits);
    }
};

fn top1(row: []const u16) u32 {
    var best: usize = 0;
    for (row, 0..) |b, i| if (promote(b) > promote(row[best])) {
        best = i;
    };
    return @intCast(best);
}

fn margin(row: []const u16) f32 {
    const b = top1(row);
    var second: f32 = -std.math.inf(f32);
    for (row, 0..) |v, i| if (i != b) {
        second = @max(second, promote(v));
    };
    return promote(row[b]) - second;
}

fn top5(row: []const u16) [5]u32 {
    var ids: [5]u32 = @splat(0);
    var vals: [5]f32 = @splat(-std.math.inf(f32));
    for (row, 0..) |b, i| {
        const v = promote(b);
        if (!(v > vals[4])) continue;
        var at: usize = 4;
        while (at > 0 and v > vals[at - 1]) : (at -= 1) {
            vals[at] = vals[at - 1];
            ids[at] = ids[at - 1];
        }
        vals[at] = v;
        ids[at] = @intCast(i);
    }
    return ids;
}

const Run = struct { ms: f64, logits: []u16, greedy: []u32, decode_ms: f64, stats: xpre.Stats };

/// A fresh sequence, `ids` prefilled (deep or the batched pass), then `steps` greedy rounds.
fn once(g: *Eng, ids: []const u32, deep: bool, rows: usize, steps: usize) !Run {
    const a = g.gpu.gpa;
    var o = xpre.optionsFromEnv();
    o.enabled = deep;
    o.rows = rows;
    g.e.exl3_prefill.options = o;
    g.e.exl3_prefill.stats = .{};
    try g.reset();
    _ = try g.swap(ids.len);
    const t0 = check.now(g.gpu.io);
    try g.forward(ids);
    const t = ms(t0, check.now(g.gpu.io));
    const st = g.e.exl3_prefill.stats;
    const logits = try a.alloc(u16, vocab);
    @memcpy(std.mem.sliceAsBytes(logits), g.seq.logits);
    const greedy = try a.alloc(u32, steps + 1);
    greedy[0] = top1(logits);
    const t1 = check.now(g.gpu.io);
    for (1..steps + 1) |i| greedy[i] = try g.round(greedy[i - 1]);
    const dt = if (steps > 0) ms(t1, check.now(g.gpu.io)) / @as(f64, @floatFromInt(steps)) else 0;
    return .{ .ms = t, .logits = logits, .greedy = greedy, .decode_ms = dt, .stats = st };
}

fn freeRun(a: std.mem.Allocator, r: Run) void {
    a.free(r.logits);
    a.free(r.greedy);
}

/// `exact`: the same bytes and greedy ids (deep chunks against the batched pass), else the same real-margin top-1.
fn compare(what: []const u8, n: usize, ref: []const u16, got: []const u16, ref_g: []const u32, got_g: []const u32, exact: bool) bool {
    var dmax: f32 = 0;
    var peak: f32 = 0;
    var same: usize = 0;
    for (ref, got) |x, y| {
        dmax = @max(dmax, @abs(promote(x) - promote(y)));
        peak = @max(peak, @abs(promote(x)));
        same += @intFromBool(x == y);
    }
    var first: ?usize = null;
    for (ref_g, got_g, 0..) |x, y, i| if (x != y and first == null) {
        first = i;
    };
    const t_eq = top1(ref) == top1(got);
    std.debug.print("CMP {s} {d} tokens: last-row logits max |d| {d:.4} (peak {d:.2}, {d:.1}% bit-equal); top-1 {d} vs {d} ({s}, reference margin {d:.4}); greedy {d}: {s}", .{ what, n, dmax, peak, 100 * @as(f64, @floatFromInt(same)) / vocab, top1(ref), top1(got), if (t_eq) "equal" else "DIFFERENT", margin(ref), ref_g.len, if (first == null) "identical" else "differ" });
    if (first) |f| std.debug.print(" from step {d} ({d} vs {d})", .{ f, ref_g[f], got_g[f] });
    std.debug.print("\n", .{});
    if (!t_eq) {
        // the two candidates in both rows, and the top-5 overlap
        const a_ = top1(ref);
        const b_ = top1(got);
        const r5 = top5(ref);
        const g5 = top5(got);
        var overlap: usize = 0;
        for (r5) |x| for (g5) |y| {
            overlap += @intFromBool(x == y);
        };
        std.debug.print("    reference: logit[{d}] {d:.4}, logit[{d}] {d:.4}; ours: logit[{d}] {d:.4}, logit[{d}] {d:.4}; top-5 overlap {d}/5 (reference {any}, ours {any})\n", .{ a_, promote(ref[a_]), b_, promote(ref[b_]), a_, promote(got[a_]), b_, promote(got[b_]), overlap, r5, g5 });
    }
    if (exact) return same == vocab and first == null;
    return t_eq or margin(ref) < 0.25;
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 10) return error.MissingArgument;
    const a = gpu.gpa;
    const rank = try std.fmt.parseInt(usize, rest[0], 10);
    const world = try std.fmt.parseInt(usize, rest[1], 10);
    const port = try std.fmt.parseInt(u16, rest[2], 10);
    const dir = rest[3];
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, rest[4], a, .limited(1 << 28));
    defer a.free(text);
    const all_ids = try a.alloc(u32, text.len / 4);
    defer a.free(all_ids);
    for (all_ids, 0..) |*x, i| x.* = std.mem.readInt(u32, text[4 * i ..][0..4], .little);
    var ns: [16]usize = undefined;
    var n_ns: usize = 0;
    var it = std.mem.splitScalar(u8, rest[5], ',');
    while (it.next()) |tok| {
        ns[n_ns] = try std.fmt.parseInt(usize, tok, 10);
        n_ns += 1;
    }
    const rows = try std.fmt.parseInt(usize, rest[6], 10);
    const cap = try std.fmt.parseInt(usize, rest[7], 10);
    const both = std.mem.eql(u8, rest[8], "both");
    const dump = rest[9];
    var steps: usize = 16;
    if (std.c.getenv("TF_TEST_GREEDY")) |v| steps = std.fmt.parseInt(usize, std.mem.span(v), 10) catch steps;
    var both_max: usize = std.math.maxInt(usize);
    if (std.c.getenv("TF_TEST_BOTH_MAX")) |v| both_max = std.fmt.parseInt(usize, std.mem.span(v), 10) catch both_max;
    for (ns[0..n_ns]) |n| if (n + steps + 1 > cap or n > all_ids.len) return error.NotEnoughRoom;
    if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
    const g = try Eng.init(gpu, rank, world, port, dir, cap, rows);
    defer g.deinit();
    // warm-up: kernels, tiles, the ring
    {
        const w = try once(g, all_ids[0..@min(1024, all_ids.len)], true, rows, 1);
        freeRun(a, w);
    }
    var all_ok = true;
    for (ns[0..n_ns]) |n| {
        const ids = all_ids[0..n];
        const d = try once(g, ids, true, rows, steps);
        defer freeRun(a, d);
        const s = d.stats;
        std.debug.print("PREFILL rank {d}/{d} {d:6} tokens deep: {d:9.1} ms = {d:7.1} rows/s; chunks {d}, passes {d}, copies {d} ({d} prefetched), all-reduces {d}; host picks wait {d:.1} ms, plan {d:.1} ms; decode after {d:.2} ms/token\n", .{ rank, world, n, d.ms, @as(f64, @floatFromInt(n)) / (d.ms / 1e3), s.chunks, s.passes, s.copies, s.prefetched, s.reduces, @as(f64, @floatFromInt(s.host_wait_ns)) / 1e6, @as(f64, @floatFromInt(s.plan_ns)) / 1e6, d.decode_ms });
        if (s.reduce_ns > 0) std.debug.print("        rank {d}: combine + all-reduce + place {d:.1} ms over {d} layers (profiled, each waited)\n", .{ rank, @as(f64, @floatFromInt(s.reduce_ns)) / 1e6, s.reduces });
        if (both and n <= both_max) {
            const b = try once(g, ids, false, rows, steps);
            defer freeRun(a, b);
            std.debug.print("PREFILL rank {d}/{d} {d:6} tokens batched: {d:9.1} ms = {d:7.1} rows/s; decode after {d:.2} ms/token\n", .{ rank, world, n, b.ms, @as(f64, @floatFromInt(n)) / (b.ms / 1e3), b.decode_ms });
            if (rank == 0) all_ok = compare("deep vs batched (same world)", n, b.logits, d.logits, b.greedy, d.greedy, true) and all_ok;
        }
        // both ranks hold the same logits and greedy ids
        const h = std.hash.Wyhash.hash(std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(d.logits)), std.mem.sliceAsBytes(d.greedy));
        const other = try g.swap(h);
        if (world == 2 and other != h) {
            std.debug.print("rank {d}: the ranks' logits or greedy ids DIFFER at {d} tokens\n", .{ rank, n });
            all_ok = false;
        }
        if (rank != 0) continue;
        var pb: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&pb, "{s}-{d}.bin", .{ dump, n });
        if (world == 1) {
            const bytes = try a.alloc(u8, vocab * 2 + d.greedy.len * 4);
            defer a.free(bytes);
            @memcpy(bytes[0 .. vocab * 2], std.mem.sliceAsBytes(d.logits));
            @memcpy(bytes[vocab * 2 ..], std.mem.sliceAsBytes(d.greedy));
            try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = bytes });
            continue;
        }
        const ref = std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 24)) catch |err| {
            std.debug.print("no tp=1 dump at {s} ({t})\n", .{ path, err });
            continue;
        };
        defer a.free(ref);
        if (ref.len != vocab * 2 + d.greedy.len * 4) {
            std.debug.print("tp=1 dump {s} has another greedy length; skipped\n", .{path});
            continue;
        }
        const rl = try a.alloc(u16, vocab);
        defer a.free(rl);
        @memcpy(std.mem.sliceAsBytes(rl), ref[0 .. vocab * 2]);
        const rg = try a.alloc(u32, d.greedy.len);
        defer a.free(rg);
        @memcpy(std.mem.sliceAsBytes(rg), ref[vocab * 2 ..]);
        all_ok = compare("tp=2 deep vs tp=1 deep", n, rl, d.logits, rg, d.greedy, false) and all_ok;
    }
    const both_ok = try g.swap(@intFromBool(all_ok));
    try check.expect(all_ok and both_ok == 1, "rank {d}: a top-1 differs past bf16 noise or the ranks disagree", .{rank});
    check.pass("rank {d}/{d}: deep prefill measured and compared", .{ rank, world });
}
