//! flashnext-exl3-tpe: teacher-forced, free-running and timed decode through the engine.
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

pub fn tfMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, golden_path: []const u8, dump: []const u8, cont: bool) !void {
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

pub fn genMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, golden_path: []const u8, dump: []const u8) !void {
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

pub fn benchMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8) !void {
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
