//! exl3-pool: hit rates of the cache policy replayed over recorded routing.
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

// ---------------------------------------------------------------------------------------------------- hit rates

const line_counts = [_]usize{ 10, 24, 48, 96, 160 };

const Sim = struct {
    hit: u64 = 0,
    miss: u64 = 0,
    dec_hit: u64 = 0,
    dec_miss: u64 = 0,
    copies: u64 = 0,
};

/// Host-only replay of a pick sequence through one layer's policy.
fn simulate(gpa: std.mem.Allocator, layer: u8, picks: []const u16, tokens: usize, prompt: usize, lines: usize, prefetch_lines: usize, counts: *[policy.shape.layers][policy.shape.experts]u32) !Sim {
    const slots_tab = try gpa.alloc(u16, lines);
    defer gpa.free(slots_tab);
    const line_tab = try gpa.alloc(policy.Line, lines);
    defer gpa.free(line_tab);
    const stamp = try gpa.alloc(u32, lines);
    defer gpa.free(stamp);
    const lay = try gpa.alloc(u8, lines);
    defer gpa.free(lay);
    var v = policy.Vram.init(slots_tab, line_tab, stamp, lay);
    var r = try fpool.Round.init(gpa, lines, max_rows * slots);
    defer r.deinit(gpa);
    var buf: [max_rows * slots]i32 = undefined;
    var sim: Sim = .{};
    var prev: [routed_n]u16 = undefined;
    var prev_n: usize = 0;
    var t: usize = 0;
    var round: u32 = 0;
    while (t < tokens) {
        const rows: usize = if (t < prompt) @min(max_rows, prompt - t) else 1;
        for (0..rows) |i| {
            for (0..top_k) |k| buf[i * slots + k] = picks[(t + i) * top_k + k];
            buf[i * slots + top_k] = 512;
        }
        round += 1;
        try fpool.round.plan(&v, layer, buf[0 .. rows * slots], counts, round, round, .{ .prefetch_bytes = prefetch_lines, .line_bytes = 1, .protect = prev[0..prev_n] }, &r);
        @memcpy(prev[0..r.n_ids], r.ids[0..r.n_ids]);
        prev_n = r.n_ids;
        // prefetch lines land before the next round (the device has a whole layer stack of time)
        for (r.prefetch()) |c| policy.reapAt(&v, layer, c.id, round) catch {};
        sim.hit += r.hit_slots;
        sim.miss += r.miss_slots;
        sim.copies += r.n_copies;
        if (t >= prompt) {
            sim.dec_hit += r.hit_slots;
            sim.dec_miss += r.miss_slots;
        }
        t += rows;
    }
    return sim;
}

fn rate(h: u64, m: u64) f64 {
    return if (h + m == 0) 0 else @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(h + m));
}

pub fn hitRun(gpu: Gpu, files: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const counts = try gpa.create([policy.shape.layers][policy.shape.experts]u32);
    defer gpa.destroy(counts);
    // synthetic: 200 tokens of Zipf(1.1) picks, decode rows only
    {
        const z = Zipf.init(0x200, 1.1);
        var prng = std.Random.DefaultPrng.init(0x200);
        var row: [slots]i32 = undefined;
        var picks: [200 * top_k]u16 = undefined;
        for (0..200) |t| {
            z.row(prng.random(), &row);
            for (0..top_k) |k| picks[t * top_k + k] = @intCast(row[k]);
        }
        std.debug.print("  synthetic Zipf(1.1) over 512 experts, 200 decode tokens, one layer:\n", .{});
        for (line_counts) |lines| {
            counts.* = @splat(@splat(0));
            const a = try simulate(gpa, 0, &picks, 200, 0, lines, 0, counts);
            counts.* = @splat(@splat(0));
            const b = try simulate(gpa, 0, &picks, 200, 0, lines, 4, counts);
            std.debug.print("    {d:>3} lines ({d:>5.1} MiB/layer at 1.76 MiB a line): hit {d:.3} (with 4-line prefetch every 4 rounds {d:.3})\n", .{ lines, @as(f64, @floatFromInt(lines)) * 1.7578, rate(a.hit, a.miss), rate(b.hit, b.miss) });
        }
    }
    for (files) |path| {
        const blob = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpa, .limited(1 << 30));
        defer gpa.free(blob);
        const hdr = std.mem.bytesAsSlice(u32, blob[0..12]);
        const tokens: usize = hdr[0];
        const prompt: usize = hdr[1];
        const layers: usize = hdr[2];
        const raw = try gpa.alloc(u16, tokens * layers * top_k);
        defer gpa.free(raw);
        @memcpy(std.mem.sliceAsBytes(raw), blob[12..][0 .. raw.len * 2]);
        std.debug.print("  real router picks {s}: {d} tokens ({d} prompt as 16-row chunks, {d} decode rows) x {d} layers\n", .{ path, tokens, prompt, tokens - prompt, layers });
        const per = try gpa.alloc(u16, tokens * top_k);
        defer gpa.free(per);
        // distinct experts a layer over the decode part (an upper bound on what any pool must hold)
        var distinct_sum: usize = 0;
        for (0..layers) |l| {
            var seen: [routed_n]bool = @splat(false);
            for (prompt..tokens) |t| for (0..top_k) |k| {
                seen[raw[(t * layers + l) * top_k + k]] = true;
            };
            for (seen) |b| distinct_sum += @intFromBool(b);
        }
        std.debug.print("    distinct experts a layer over the {d} decode tokens: {d:.1} on average\n", .{ tokens - prompt, @as(f64, @floatFromInt(distinct_sum)) / @as(f64, @floatFromInt(layers)) });
        for (line_counts) |lines| {
            for ([_]usize{ 0, 4 }) |pf| {
                counts.* = @splat(@splat(0));
                var tot: Sim = .{};
                var lo: f64 = 1;
                var hi: f64 = 0;
                var copies: u64 = 0;
                for (0..layers) |l| {
                    for (0..tokens) |t| @memcpy(per[t * top_k ..][0..top_k], raw[(t * layers + l) * top_k ..][0..top_k]);
                    const s = try simulate(gpa, @intCast(l), per, tokens, prompt, lines, pf, counts);
                    tot.hit += s.hit;
                    tot.miss += s.miss;
                    tot.dec_hit += s.dec_hit;
                    tot.dec_miss += s.dec_miss;
                    copies += s.copies;
                    const dr = rate(s.dec_hit, s.dec_miss);
                    lo = @min(lo, dr);
                    hi = @max(hi, dr);
                }
                const dec_tokens = tokens - prompt;
                const miss_per_tok = @as(f64, @floatFromInt(tot.dec_miss)) / @as(f64, @floatFromInt(dec_tokens));
                std.debug.print("    {d:>3} lines{s}: decode hit {d:.3} (layers {d:.3}..{d:.3}), {d:.1} routed misses a decode token over 48 layers; all rows hit {d:.3}; pool {d:.2} GiB for 48 layers\n", .{
                    lines, if (pf > 0) " +prefetch" else "          ", rate(tot.dec_hit, tot.dec_miss), lo, hi, miss_per_tok, rate(tot.hit, tot.miss), @as(f64, @floatFromInt(lines * 48)) * 1.7578 / 1024.0,
                });
            }
        }
    }
    check.pass("hit rates printed (host simulation of the pool's own planner)", .{});
}
