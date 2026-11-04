//! flashnext-exl3-tpe: per-rank hit rates and the cache policy replay (host only).
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

pub fn hitMode(gpu: Gpu, files: []const [:0]const u8) !void {
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

pub fn policyMode(gpu: Gpu, files: []const [:0]const u8) !void {
    if (files.len == 0) return error.MissingTraceFiles;
    const allocator = gpu.gpa;
    for (files) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, allocator, .limited(X.dplan.demand_trace.pool_bytes + 1));
        defer allocator.free(bytes);
        var snapshot = try X.dplan.demand_trace.Snapshot.decode(allocator, bytes);
        defer snapshot.deinit();
        const view = try snapshot.view();
        if (view.dropped() != 0) return error.TraceDroppedRecords;
        if (view.recorded() == 0) return error.EmptyTrace;
        const first = try view.round(0);
        const result = try X.dplan.policy_eval.evaluateTrace(allocator, view);
        const lifetime = result.lifetime;
        const lru = result.lru;
        const decayed = result.decayed_frequency;
        const admission = result.decayed_admission;
        std.debug.print("POLICY {s} pool={d} layer={d} lines={d} rounds={d} drops={d}\n", .{ path, view.words[2], view.words[3], first.lines, view.recorded(), view.dropped() });
        if (view.checkpoints() != 0) std.debug.print("  {d} ownership checkpoints: each device segment starts from its recorded state; host intervals excluded\n", .{view.checkpoints()});
        std.debug.print("  lifetime picks miss={d} copies={d} evictions={d} mapped={d}\n", .{ lifetime.miss_picks, lifetime.copies, lifetime.evictions, lifetime.mapped_experts });
        std.debug.print("  lru      picks miss={d} copies={d} evictions={d} mapped={d}\n", .{ lru.miss_picks, lru.copies, lru.evictions, lru.mapped_experts });
        std.debug.print("  decayed  picks miss={d} copies={d} evictions={d} mapped={d}\n", .{ decayed.miss_picks, decayed.copies, decayed.evictions, decayed.mapped_experts });
        std.debug.print("  guarded  picks miss={d} copies={d} evictions={d} mapped={d} mapped_picks={d}\n", .{ admission.miss_picks, admission.copies, admission.evictions, admission.mapped_experts, admission.mapped_picks });
        std.debug.print("  Belady unique-miss lower bound={d} (not copy/time bound)\n", .{result.belady_unique_misses_lower_bound});
    }
}
