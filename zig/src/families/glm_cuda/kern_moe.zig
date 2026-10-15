//! GLM-5.3-Flash on CUDA: the grouped routed experts' plan and calls (cuda/experts.py), re-exported by kern.zig.

const std = @import("std");
const cuda = @import("cuda");
const kern = @import("kern.zig");
const K = kern.K;
const cdiv = kern.cdiv;
const EXPERTS = kern.EXPERTS;

pub fn maxItems(pairs: i64, experts: i64, tile: i64) i64 {
    return @min(pairs, experts) + @divFloor(pairs, tile);
}

pub const Plan = struct { members: u64, items: u64, counts: u64, rank: u64, hist: u64, tile: i64 };

/// experts_v7.plan: picks [R, 8] (the routed slots alone) -> the plan over the 288 experts.
pub fn expertsPlan(k: *K, picks: u64, pairs: i64, plan: Plan) !void {
    var a: cuda.Args = .{};
    const E: i32 = @intCast(EXPERTS);
    if (pairs <= 1024) {
        a.add(picks);
        a.add(@as(i32, @intCast(pairs)));
        a.add(E);
        a.add(@as(i32, @intCast(plan.tile)));
        a.add(plan.members);
        a.add(plan.items);
        a.add(plan.counts);
        try k.go(try k.ext(.experts, "11plan_kernelE"), .{ .x = 1 }, 1024, 0, &a);
    } else {
        const nblk = cdiv(pairs, 1024);
        a.add(picks);
        a.add(@as(i32, @intCast(pairs)));
        a.add(E);
        a.add(plan.rank);
        a.add(plan.hist);
        try k.go(try k.ext(.experts, "9plan_rankE"), .{ .x = @intCast(nblk) }, 1024, 0, &a);
        var b: cuda.Args = .{};
        b.add(@as(i32, @intCast(nblk)));
        b.add(E);
        b.add(@as(i32, @intCast(plan.tile)));
        b.add(plan.hist);
        b.add(plan.items);
        b.add(plan.counts);
        try k.go(try k.ext(.experts, "12plan_offsetsE"), .{ .x = 1 }, 1024, 0, &b);
        var s: cuda.Args = .{};
        s.add(picks);
        s.add(@as(i32, @intCast(pairs)));
        s.add(E);
        s.add(plan.rank);
        s.add(plan.hist);
        s.add(plan.members);
        try k.go(try k.ext(.experts, "12plan_scatterE"), .{ .x = @intCast(cdiv(pairs, 256)) }, 256, 0, &s);
    }
}

/// experts_v7.prefill / run: gate-up (epi 2) or down (epi 3 bf16 / 0 fp32) over the plan; ``slots``: pairs a row feeds.
pub fn expertsCall(k: *K, prefill: bool, epi: i64, x: u64, x_stride: i64, slots: i64, w: u64, kb: i64, nb: i64, plan: Plan, out: u64, width: i64, items: i64, limit: f32) !void {
    const gs: i64 = 64;
    const m: i64 = if (epi == 2) 2 else 1;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i32, @intCast(x_stride)));
    a.add(@as(i32, @intCast(slots)));
    a.add(w);
    a.add(@as(i32, @intCast(kb)));
    a.add(@as(i32, @intCast(nb)));
    a.add(plan.items);
    a.add(plan.counts);
    a.add(plan.members);
    a.add(out);
    a.add(@as(i32, @intCast(width)));
    a.add(limit);
    var buf: [128]u8 = undefined;
    const sbv: i64 = if (k.experts_sym) 1 else 2; // a quad's scale (and bias) uint4s a group
    if (prefill) {
        const f = try k.ext(if (k.experts3) .experts_sym3_prefill else if (k.experts_sym) .experts_sym_prefill else .experts_prefill, try std.fmt.bufPrint(&buf, "14prefill_kernelILi{d}ELi{d}ELi{d}ELi2ELi2ELi4E", .{ gs, m, epi }));
        const block: i64 = @as(i64, if (k.experts3) 24 else 32) * @divExact(4 * gs, 128) + 4 * sbv;
        const su: i64 = 64 * 8 + 4 * @divExact(64, gs) * m * block;
        const sb = su * 16;
        const stages: i64 = if (sb * 4 <= 49152) 4 else if (sb * 3 <= 49152) 3 else 2;
        const smem: u32 = @intCast(stages * sb);
        try f.allowDynamicShared(smem);
        const grid = items * cdiv(nb, 4);
        if (grid >= 1) try k.go(f, .{ .x = @intCast(grid) }, 256, smem, &a);
    } else {
        const f = try k.ext(if (k.experts3) .experts_sym3 else if (k.experts_sym) .experts_sym else .experts, try std.fmt.bufPrint(&buf, "13expert_kernelILi{d}ELi{d}ELi{d}ELi2ELi4E", .{ gs, m, epi }));
        const per_sm: i64 = @max(1, try f.occupancy(128, 0));
        const grid = @min(cdiv(items, 4), per_sm * k.sms);
        if (grid >= 1) try k.go(f, .{ .x = @intCast(grid) }, 128, 0, &a);
    }
}
