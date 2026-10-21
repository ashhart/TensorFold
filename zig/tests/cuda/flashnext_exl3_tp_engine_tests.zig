//! flashnext-exl3-tpe: the Qwen3.8-Flash-Next EXL3 pack through the Flash Next family's Engine at tp=1 or tp=2
//! (expert parallel, cuda_exl3_tp.zig) with each rank's own offload pool attached as the server attaches it
//! (cuda_exl3_pass.attach) and the lm_head split by vocabulary at tp=2 (cuda_exl3_load.shardHead). One process a card;
//! at tp=2 both ranks run the same calls in lockstep over the family's NCCL Collective (NCCL >= 2.30).
//!
//!   flashnext-exl3-tpe RANK WORLD PORT DIR tf GOLDEN DUMP   the golden's 16-id prompt teacher-forced; WORLD 1 writes
//!                                                          its bf16 logits to DUMP, WORLD 2 compares rank 0's to DUMP
//!                                                          (top-1 at every position, bit-equal count, max |d|) and
//!                                                          both to the golden
//!   flashnext-exl3-tpe RANK WORLD PORT DIR tfc GOLDEN DUMP  the same with the golden's 23 greedy ids after the prompt
//!                                                          (each continuation step teacher-forced, both margins)
//!   flashnext-exl3-tpe RANK WORLD PORT DIR gen GOLDEN DUMP  greedy 24 for each golden prompt; WORLD 1 writes the ids,
//!                                                          WORLD 2 compares (and the two ranks' ids)
//!   flashnext-exl3-tpe RANK WORLD PORT DIR bench            prefill 16/256/2048 rows, decode ms/token, each rank's
//!                                                          pool hit rate, lines and free memory
//!   flashnext-exl3-tpe 0 1 0 - hit FILES...                host replay of real router picks through the pool's planner
//!                                                          masked to each expert-parallel rank (RANKS: 1 or 2)
//!   flashnext-exl3-tpe RANK 2 PORT - graph                  ncclAllReduce on the family's nonblocking communicator
//!                                                          captured in a CUDA graph and replayed on both ranks
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");

const Gpu = check.Gpu;
const prompt = flash.prompt;
const X = flash.exl3;
const xpass = X.pass;
const tp = X.tp;
const fpool = flash.exl3_pool;
const wire = flash.rank_wire;
const Collective = flash.collective.Collective;
const vocab = prompt.vocabulary;
const cap_default: usize = 32768;
const out_rows: usize = 64;

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn ms(t0: i96, t1: i96) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) / 1e6;
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

// -- the ranks' link ------------------------------------------------------------------------------------------------

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

    /// Both ranks here; returns the other rank's `value`.
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

// -- the engine -----------------------------------------------------------------------------------------------------

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
        const t1 = check.now(gpu.io);
        _ = try g.e.reserve(@min(cap_default, g.e.chunk));
        _ = try g.e.reserveScores(cap_default, @min(cap_default, g.e.chunk));
        g.seq = try g.e.newSeq(cap_default);
        g.out = try cuda.DeviceBuffer.alloc(gpu.d, out_rows * vocab * 2);
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, g.e.layers[0..g.e.n_layers], null, g.e.chunk, try xpass.budgetFromEnv(0));
        const t2 = check.now(gpu.io);
        var free: usize = 0;
        var total: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
        const head_n = g.e.head_q.n;
        std.debug.print("RESULT rank {d}/{d} load: {d:.1} s (initRank {d:.1} s, pinned bank {d:.2} GiB copied in {d:.1} s); {d} lines a layer x {d} layers of {d:.2} MiB = {d:.2} GiB of pool; {d:.2} GiB free before the pools, {d:.2} GiB free after (card {d:.2} GiB); head columns {d}; chunk {d}\n", .{ rank, world, ms(t0, t2) / 1e3, ms(t0, t1) / 1e3, gib(g.attached.bank_bytes), @as(f64, @floatFromInt(g.attached.bank_ns)) / 1e9, g.attached.lines, g.attached.layers, @as(f64, @floatFromInt(g.attached.line_bytes)) / (1 << 20), gib(g.attached.device_bytes), gib(g.attached.free_before), gib(free), gib(total), head_n, g.e.chunk });
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

    fn stats(g: *Eng) fpool.Stats {
        return xpass.poolStats(g.e.layers[0..g.e.n_layers], null);
    }

    /// Both ranks reach this point; the other rank's value (or `v` at tp=1).
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
};

fn printPool(g: *Eng, what: []const u8) void {
    const s = g.stats();
    const routed = s.hit_slots + s.miss_slots;
    std.debug.print("RESULT rank {d}/{d} pool {s}: {d} rounds, {d} passes, hit rate {d:.1}% ({d}/{d} routed slots of this rank), {d} copies, {d} evictions; host cut {d:.1} ms over {d} blocks, plan {d:.1} ms\n", .{ g.rank, g.world, what, s.rounds, s.passes, 100 * @as(f64, @floatFromInt(s.hit_slots)) / @as(f64, @floatFromInt(@max(routed, 1))), s.hit_slots, routed, s.misses, s.evictions, @as(f64, @floatFromInt(xpass.pool_time.cut_ns)) / 1e6, xpass.pool_time.blocks, @as(f64, @floatFromInt(xpass.pool_time.plan_ns)) / 1e6 });
}

// -- golden ---------------------------------------------------------------------------------------------------------

const Golden = struct {
    parsed: std.json.Parsed(std.json.Value),

    fn open(gpu: Gpu, path: []const u8) !Golden {
        const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 26));
        defer gpu.gpa.free(text);
        return .{ .parsed = try std.json.parseFromSlice(std.json.Value, gpu.gpa, text, .{}) };
    }

    fn prompts(g: *const Golden) []std.json.Value {
        return g.parsed.value.object.get("prompts").?.array.items;
    }
};

fn ints(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const items = v.array.items;
    const out = try a.alloc(u32, items.len);
    for (items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

fn num(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => std.math.nan(f64),
    };
}

fn top1(row: []const u16) u32 {
    var best: usize = 0;
    for (row, 0..) |b, i| if (promote(b) > promote(row[best])) {
        best = i;
    };
    return @intCast(best);
}

// -- modes ----------------------------------------------------------------------------------------------------------

fn margin(row: []const u16) f32 {
    const b = top1(row);
    var second: f32 = -std.math.inf(f32);
    for (row, 0..) |v, i| if (i != b) {
        second = @max(second, promote(v));
    };
    return promote(row[b]) - second;
}

/// `cont`: the prompt followed by the golden's first 23 greedy ids (each continuation step teacher-forced).
fn tfMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, golden_path: []const u8, dump: []const u8, cont: bool) !void {
    const a = gpu.gpa;
    var golden = try Golden.open(gpu, golden_path);
    defer golden.parsed.deinit();
    const p = golden.prompts()[0];
    const prompt_ids = try ints(a, p.object.get("ids").?);
    defer a.free(prompt_ids);
    const greedy = try ints(a, p.object.get("greedy_ids").?);
    defer a.free(greedy);
    const ids = try a.alloc(u32, prompt_ids.len + if (cont) greedy.len - 1 else 0);
    defer a.free(ids);
    @memcpy(ids[0..prompt_ids.len], prompt_ids);
    if (cont) @memcpy(ids[prompt_ids.len..], greedy[0 .. greedy.len - 1]);
    const tf = p.object.get("teacher_forced").?.array.items;
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    try g.reset();
    const n = ids.len;
    const ops = g.e.layers[0].exl3.?.ops;
    const before = ops.fingerprint();
    const t0 = check.now(gpu.io);
    try g.forward(ids, .{ .logits = .all, .logits_out = g.out.ptr });
    const t1 = check.now(gpu.io);
    const logits = try a.alloc(u16, n * vocab);
    defer a.free(logits);
    try g.out.download(0, std.mem.sliceAsBytes(logits));
    std.debug.print("rank {d}/{d}: teacher-forced pass of {d} rows in {d:.1} ms (cold pool)\n", .{ rank, world, n, ms(t0, t1) });
    _ = try g.round(prompt.argmax(g.seq.logits));
    const after = ops.fingerprint();
    try check.expect(std.mem.eql(u8, std.mem.sliceAsBytes(&before), std.mem.sliceAsBytes(&after)), "rank {d}: an EXL3 scratch buffer grew inside a pass", .{rank});
    check.pass("rank {d}/{d}: EXL3 scratch (incl. expert parallel's) kept its addresses through a {d}-row prompt pass and a decode round", .{ rank, world, n });
    // both ranks hold the same gathered logits
    const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(logits));
    const other = try g.swap(h);
    if (world == 2) try check.expect(other == h, "the two ranks' gathered logits differ", .{});
    if (rank != 0) return;
    // vs the golden
    var ok: usize = 0;
    var worst: f64 = 0;
    for (tf, 0..) |row, pos| {
        const r = logits[pos * vocab ..][0..vocab];
        const g5 = try ints(a, row.object.get("top5_ids").?);
        defer a.free(g5);
        ok += @intFromBool(top1(r) == g5[0]);
        for (g5, row.object.get("top5_logits").?.array.items) |id, lv| worst = @max(worst, @abs(@as(f64, promote(r[id])) - num(lv)));
    }
    std.debug.print("RESULT tp={d} TF vs golden: top-1 {d}/{d}, max |our logit - golden| on the golden top-5 {d:.4}\n", .{ world, ok, tf.len, worst });
    if (world == 1) {
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = dump, .data = std.mem.sliceAsBytes(logits) });
        check.pass("tp=1 logits written to {s}", .{dump});
        return;
    }
    const ref_bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, dump, a, .limited(1 << 30));
    defer a.free(ref_bytes);
    if (ref_bytes.len != logits.len * 2) return error.UnexpectedTensor;
    const ref = try a.alloc(u16, logits.len);
    defer a.free(ref);
    @memcpy(std.mem.sliceAsBytes(ref), ref_bytes);
    var top_eq: usize = 0;
    var bits: usize = 0;
    var dmax: f64 = 0;
    var dsum: f64 = 0;
    std.debug.print(" pos  tp=1 top1 (margin)  tp=2 top1 (margin)  max|d| over the row  bit-equal of {d}\n", .{vocab});
    for (0..n) |pos| {
        const x = ref[pos * vocab ..][0..vocab];
        const y = logits[pos * vocab ..][0..vocab];
        var row_d: f64 = 0;
        var row_bits: usize = 0;
        for (x, y) |u, v| {
            row_bits += @intFromBool(u == v);
            const d = @abs(@as(f64, promote(u)) - promote(v));
            row_d = @max(row_d, d);
            dsum += d;
        }
        bits += row_bits;
        dmax = @max(dmax, row_d);
        const same = top1(x) == top1(y);
        top_eq += @intFromBool(same);
        std.debug.print(" {d:3}  {d:9} ({d:6.3})  {d:9} ({d:6.3})  {d:.4}  {d}{s}\n", .{ pos, top1(x), margin(x), top1(y), margin(y), row_d, row_bits, if (same) "" else "  TOP-1 DIFFERS" });
    }
    std.debug.print("RESULT tp=2 vs tp=1 logits: top-1 equal {d}/{d}; bf16 bit-equal {d}/{d} ({d:.3}%); max |d| {d:.4}, mean |d| {e:.3}\n", .{ top_eq, n, bits, n * vocab, 100 * @as(f64, @floatFromInt(bits)) / @as(f64, @floatFromInt(n * vocab)), dmax, dsum / @as(f64, @floatFromInt(n * vocab)) });
    try check.expect(top_eq == n, "tp=2 top-1 differs from tp=1", .{});
    check.pass("tp=2 teacher-forced top-1 equals tp=1 at all {d} positions", .{n});
    printPool(g, "after tf");
}

fn genMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, golden_path: []const u8, dump: []const u8) !void {
    const a = gpu.gpa;
    var golden = try Golden.open(gpu, golden_path);
    defer golden.parsed.deinit();
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    var all_golden = true;
    for (golden.prompts()) |p| {
        const ids = try ints(a, p.object.get("ids").?);
        defer a.free(ids);
        const want = try ints(a, p.object.get("greedy_ids").?);
        defer a.free(want);
        const got = try a.alloc(u32, want.len);
        defer a.free(got);
        try g.reset();
        try g.forward(ids, .{ .logits = .last });
        got[0] = prompt.argmax(g.seq.logits);
        const t0 = check.now(gpu.io);
        for (1..got.len) |i| got[i] = try g.round(got[i - 1]);
        const step = ms(t0, check.now(gpu.io)) / @as(f64, @floatFromInt(got.len - 1));
        const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(got));
        const other = try g.swap(h);
        if (world == 2) try check.expect(other == h, "the two ranks' greedy ids differ", .{});
        var first: ?usize = null;
        for (got, want, 0..) |x, y, i| if (x != y and first == null) {
            first = i;
        };
        if (first != null) all_golden = false;
        for (got) |x| try text.print(a, "{d},", .{x});
        try text.append(a, '\n');
        if (rank == 0) {
            std.debug.print("GEN tp={d} {s}: ", .{ world, p.object.get("name").?.string });
            for (got) |x| std.debug.print("{d},", .{x});
            std.debug.print("  {s} ({d:.1} ms/step)\n", .{ if (first) |_| "differs from the golden" else "equal to the golden", step });
            if (first) |d| std.debug.print("  first difference from the golden at step {d}: ours {d}, golden {d}\n", .{ d, got[d], want[d] });
        }
    }
    if (rank != 0) return;
    printPool(g, "after gen");
    if (world == 1) {
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = dump, .data = text.items });
        check.pass("tp=1 greedy ids written to {s}", .{dump});
        return;
    }
    const ref = try std.Io.Dir.cwd().readFileAlloc(gpu.io, dump, a, .limited(1 << 20));
    defer a.free(ref);
    const same = std.mem.eql(u8, ref, text.items);
    std.debug.print("RESULT tp=2 greedy 24 on the {d} golden prompts: {s} tp=1's ids; all equal the golden: {}\n", .{ golden.prompts().len, if (same) "identical to" else "DIFFERENT from", all_golden });
    try check.expect(same, "tp=2 greedy ids differ from tp=1's", .{});
    check.pass("tp=2 greedy 24 tokens identical to tp=1 on all prompts", .{});
}

fn benchMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8) !void {
    const a = gpu.gpa;
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    const max_n: usize = 2048;
    const ids = try a.alloc(u32, max_n);
    defer a.free(ids);
    var prng = std.Random.DefaultPrng.init(0x5eed);
    for (ids) |*x| x.* = 200 + prng.random().uintLessThan(u32, 60000);
    try g.reset();
    try g.forward(ids[0..64], .{ .logits = .last });
    for ([_]usize{ 16, 256, 2048 }) |n| {
        var best: f64 = std.math.inf(f64);
        for (0..2) |_| {
            try g.reset();
            _ = try g.swap(0);
            const before = g.stats();
            const t0 = check.now(gpu.io);
            try g.forward(ids[0..n], .{ .logits = .last });
            const t = ms(t0, check.now(gpu.io));
            const after = g.stats();
            const hits = after.hit_slots - before.hit_slots;
            const all = hits + after.miss_slots - before.miss_slots;
            std.debug.print("  rank {d}/{d} prefill {d:5} tokens: {d:8.1} ms = {d:7.1} rows/s  (hit {d:.1}%, {d} copies)\n", .{ rank, world, n, t, @as(f64, @floatFromInt(n)) / (t / 1e3), 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(all, 1))), after.misses - before.misses });
            best = @min(best, t);
        }
        std.debug.print("BENCH rank {d}/{d} prefill {d} tokens: best {d:.1} ms, {d:.1} rows/s\n", .{ rank, world, n, best, @as(f64, @floatFromInt(n)) / (best / 1e3) });
        var tok = prompt.argmax(g.seq.logits);
        _ = try g.swap(1);
        const before = g.stats();
        const t0 = check.now(gpu.io);
        const steps: usize = 32;
        for (0..steps) |_| tok = try g.round(tok);
        const t = ms(t0, check.now(gpu.io)) / steps;
        const after = g.stats();
        const hits = after.hit_slots - before.hit_slots;
        const all = hits + after.miss_slots - before.miss_slots;
        std.debug.print("BENCH rank {d}/{d} decode after {d} tokens: {d:.2} ms/token over {d} steps (this rank's hit {d:.1}%, {d:.1} copies a token)\n", .{ rank, world, n, t, steps, 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(all, 1))), @as(f64, @floatFromInt(after.misses - before.misses)) / steps });
    }
    printPool(g, "after bench");
}

// -- hit rates per rank (host only) ---------------------------------------------------------------------------------

const top_k = 10;
const slots = 11;
const max_rows = 16;

fn simulate(gpa: std.mem.Allocator, layer: u8, picks: []const u16, tokens: usize, prompt_n: usize, lines: usize, r: tp.Rank, counts: *[fpool.shape.layers][fpool.shape.experts]u32, dec: *[2]u64, all_: *[2]u64) !void {
    const policy = fpool.offload;
    const slots_tab = try gpa.alloc(u16, lines);
    defer gpa.free(slots_tab);
    const line_tab = try gpa.alloc(policy.Line, lines);
    defer gpa.free(line_tab);
    const stamp = try gpa.alloc(u32, lines);
    defer gpa.free(stamp);
    const lay = try gpa.alloc(u8, lines);
    defer gpa.free(lay);
    var v = policy.Vram.init(slots_tab, line_tab, stamp, lay);
    var rnd = try fpool.Round.init(gpa, lines, max_rows * slots);
    defer rnd.deinit(gpa);
    var buf: [max_rows * slots]i32 = undefined;
    var prev: [512]u16 = undefined;
    var prev_n: usize = 0;
    var t: usize = 0;
    var round: u32 = 0;
    while (t < tokens) {
        const rows: usize = if (t < prompt_n) @min(max_rows, prompt_n - t) else 1;
        for (0..rows) |i| {
            for (0..top_k) |k| {
                const id: usize = picks[(t + i) * top_k + k];
                buf[i * slots + k] = if (r.owns(id)) @intCast(id) else -1;
            }
            buf[i * slots + top_k] = if (r.owns(512)) 512 else -1;
        }
        round += 1;
        try fpool.round.plan(&v, layer, buf[0 .. rows * slots], counts, round, round, .{ .protect = prev[0..prev_n] }, &rnd);
        @memcpy(prev[0..rnd.n_ids], rnd.ids[0..rnd.n_ids]);
        prev_n = rnd.n_ids;
        all_[0] += rnd.hit_slots;
        all_[1] += rnd.miss_slots;
        if (t >= prompt_n) {
            dec[0] += rnd.hit_slots;
            dec[1] += rnd.miss_slots;
        }
        t += rows;
    }
}

fn rate(x: [2]u64) f64 {
    return if (x[0] + x[1] == 0) 0 else @as(f64, @floatFromInt(x[0])) / @as(f64, @floatFromInt(x[0] + x[1]));
}

fn hitMode(gpu: Gpu, files: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const counts = try gpa.create([fpool.shape.layers][fpool.shape.experts]u32);
    defer gpa.destroy(counts);
    for (files) |path| {
        const blob = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpa, .limited(1 << 30));
        defer gpa.free(blob);
        const hdr = std.mem.bytesAsSlice(u32, blob[0..12]);
        const tokens: usize = hdr[0];
        const prompt_n: usize = hdr[1];
        const layers: usize = hdr[2];
        const raw = try gpa.alloc(u16, tokens * layers * top_k);
        defer gpa.free(raw);
        @memcpy(std.mem.sliceAsBytes(raw), blob[12..][0 .. raw.len * 2]);
        const per = try gpa.alloc(u16, tokens * top_k);
        defer gpa.free(per);
        std.debug.print("real router picks {s}: {d} tokens ({d} prompt in 16-row rounds, {d} decode rows) x {d} layers\n", .{ path, tokens, prompt_n, tokens - prompt_n, layers });
        // routed slots a decode token each rank owns
        var own: [2]u64 = .{ 0, 0 };
        for (prompt_n..tokens) |t| for (0..layers) |l| for (0..top_k) |k| {
            own[@intFromBool(raw[(t * layers + l) * top_k + k] >= 256)] += 1;
        };
        std.debug.print("  decode routed slots on rank 0 / rank 1: {d} / {d} ({d:.1}% on rank 0)\n", .{ own[0], own[1], 100 * @as(f64, @floatFromInt(own[0])) / @as(f64, @floatFromInt(own[0] + own[1])) });
        for ([_]usize{ 10, 24, 48, 56, 64, 96, 128, 160 }) |lines| {
            var line_txt: [3][64]u8 = undefined;
            var outs: [3][]const u8 = undefined;
            const worlds = [3][2]usize{ .{ 1, 0 }, .{ 2, 0 }, .{ 2, 1 } };
            var miss_tok: [3]f64 = undefined;
            for (worlds, 0..) |w, i| {
                const r = try tp.Rank.init(w[0], w[1]);
                counts.* = @splat(@splat(0));
                var dec: [2]u64 = .{ 0, 0 };
                var all_: [2]u64 = .{ 0, 0 };
                for (0..layers) |l| {
                    for (0..tokens) |t| @memcpy(per[t * top_k ..][0..top_k], raw[(t * layers + l) * top_k ..][0..top_k]);
                    try simulate(gpa, @intCast(l), per, tokens, prompt_n, lines, r, counts, &dec, &all_);
                }
                miss_tok[i] = @as(f64, @floatFromInt(dec[1])) / @as(f64, @floatFromInt(tokens - prompt_n));
                outs[i] = try std.fmt.bufPrint(&line_txt[i], "hit {d:.3} ({d:.1} misses/token)", .{ rate(dec), miss_tok[i] });
            }
            std.debug.print("  {d:>3} lines a layer: tp=1 {s} | tp=2 rank 0 {s} | rank 1 {s} | slowest rank's misses/token {d:.1} vs tp=1 {d:.1}\n", .{ lines, outs[0], outs[1], outs[2], @max(miss_tok[1], miss_tok[2]), miss_tok[0] });
        }
    }
    check.pass("per-rank hit rates printed (host replay through the pool's own planner)", .{});
}

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

fn graphMode(gpu: Gpu, rank: usize, port: u16) !void {
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

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 5) return error.MissingArgument;
    const rank = try std.fmt.parseInt(usize, rest[0], 10);
    const world = try std.fmt.parseInt(usize, rest[1], 10);
    const port = try std.fmt.parseInt(u16, rest[2], 10);
    const dir = rest[3];
    const mode = rest[4];
    if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
    if (std.mem.eql(u8, mode, "hit")) return hitMode(gpu, rest[5..]);
    if (std.mem.eql(u8, mode, "graph")) return graphMode(gpu, rank, port);
    if (std.mem.eql(u8, mode, "bench")) return benchMode(gpu, rank, world, port, dir);
    if (rest.len < 7) return error.MissingArgument;
    if (std.mem.eql(u8, mode, "tf")) return tfMode(gpu, rank, world, port, dir, rest[5], rest[6], false);
    if (std.mem.eql(u8, mode, "tfc")) return tfMode(gpu, rank, world, port, dir, rest[5], rest[6], true);
    if (std.mem.eql(u8, mode, "gen")) return genMode(gpu, rank, world, port, dir, rest[5], rest[6]);
    return error.UnknownMode;
}
