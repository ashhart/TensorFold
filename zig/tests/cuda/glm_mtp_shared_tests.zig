//! Batched shared MTP against serial twins; every TP rank executes the same numerical oracle.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const oracle = @import("glm_mtp_shared_state.zig");
const Engine = glm.engine.Engine;
const max_s = 8;
const Modes = enum { greedy, confidence, sampled, mixed };

fn samplingFor(mode: Modes, i: usize) ?@import("lanes").Sampling {
    const sampled = mode == .sampled or (mode == .mixed and i % 2 == 1);
    if (!sampled) return null;
    const temps = [_]f64{ 0.6, 0.8, 1.0, 0.7 };
    const topks = [_]u32{ 20, 40, 0, 16 };
    return .{ .seed = 0x9e3779b97f4a7c15 + @as(u64, i) * 1009, .temperature = temps[i % 4], .top_k = topks[i % 4], .top_p = 0.92 };
}

fn wantsChance(mode: Modes, i: usize) bool {
    return mode == .confidence or (mode == .mixed and i % 3 != 0);
}

fn openComm(gpu: check.Gpu, lib: *const cuda.nccl.Library, rank: c_int, world: c_int, path: ?[]const u8) !cuda.nccl.Comm {
    if (rank < 0 or rank >= world or world < 1) return error.BadRank;
    if (path) |p| return @import("glm_lanes_tests.zig").openComm(gpu, lib, rank, world, p);
    if (rank != 0 or world != 1) return error.IdFileRequired;
    var id: cuda.nccl.UniqueId = undefined;
    try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, 1, id, 0), "ncclCommInitRank");
    return comm;
}

/// Covers 2..8 streams, 1..7 absorb rows and depths 0..3, plus mixed depths and per-stream sampling rules.
pub fn run(gpu: check.Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: ?[]const u8) !void {
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    const set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    if (set.kv8) return error.CacheModeNotCovered;
    const e = try Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = 512, .mtp = true, .settings = set });
    defer e.deinit();
    var owned: [2 * max_s]cuda.DeviceBuffer = undefined;
    var allocated: usize = 0;
    defer for (owned[0..allocated]) |*b| b.free();
    for (&owned) |*b| {
        b.* = try cuda.DeviceBuffer.alloc(gpu.d, 8 * e.c.hidden * 2);
        allocated += 1;
    }
    const vocab: u32 = @intCast(@min(e.c.vocab, 4096));
    var n: oracle.Counts = .{};
    var oversized: usize = 0;
    for (std.enums.values(Modes)) |mode| for (2..max_s + 1) |s_count| {
        var firsts: [max_s]u32 = undefined;
        var prompts: [max_s][16]u32 = undefined;
        for (0..2 * s_count) |slot| {
            const i = slot % s_count;
            const plen: usize = if (i == 0) 1 else 3 + (i * 2) % 13;
            for (prompts[i][0..plen], 0..) |*tok, p| tok.* = 1 + @as(u32, @intCast((i * 37 + p * 13 + @as(usize, @backingInt(mode)) * 7) % (vocab - 1)));
            try e.bind(@intCast(slot));
            try e.begin(samplingFor(mode, i));
            try oracle.zeroMtp(e);
            const first = try e.prefill(prompts[i][0..plen], null, null, null);
            if (slot >= s_count and first != firsts[i]) n.upstream += 1;
            firsts[i] = first;
        }
        var b_out: [max_s][glm.engine.max_drafts]u32 = undefined;
        var b_ch: [max_s][glm.engine.max_drafts]f64 = undefined;
        var parts: [max_s]glm.engine.DraftPart = undefined;
        var one: [max_s][1]u32 = undefined;
        for (0..s_count) |i| {
            one[i] = .{firsts[i]};
            const depth = i % 4;
            parts[i] = .{ .slot = @intCast(i), .from_prompt = true, .follow = &one[i], .depth = depth, .out = b_out[i][0..depth], .chances = if (wantsChance(mode, i)) b_ch[i][0..depth] else null };
        }
        try e.draftShared(parts[0..s_count]);
        n.rounds += 1;
        for (parts[0..s_count], 0..) |p, i| {
            var out: [glm.engine.max_drafts]u32 = undefined;
            var ch: [glm.engine.max_drafts]f64 = undefined;
            try e.bind(@intCast(s_count + i));
            try e.draft(true, &one[i], p.depth, &out, if (p.chances != null) ch[0..p.depth] else null);
            try oracle.stream(e, i, s_count + i, p.out, out[0..p.depth], p.chances, if (p.chances != null) ch[0..p.depth] else null, &n);
        }
        for (1..8) |width| for (0..5) |depth_case| {
            const vw = if (width < 7 and depth_case % 2 == 1 and s_count * (width + 1) <= glm.engine.shared_rows) width + 1 else width;
            var v_parts: [max_s]glm.engine.Part = undefined;
            var ids: [max_s][8]u32 = undefined;
            var v_out: [max_s][8]u32 = undefined;
            var twin_out: [max_s][8]u32 = undefined;
            for (0..s_count) |i| {
                for (0..vw) |r| ids[i][r] = 1 + @as(u32, @intCast((width * 41 + depth_case * 19 + i * 11 + r * 23) % (vocab - 1)));
                v_parts[i] = .{ .slot = @intCast(i), .ids = ids[i][0..vw], .out = v_out[i][0..vw] };
            }
            if (s_count * vw <= glm.engine.shared_rows) {
                try e.verifyShared(v_parts[0..s_count]);
                for (0..s_count) |i| {
                    try e.bind(@intCast(i));
                    try oracle.saveRows(e, owned[i], vw);
                }
            } else {
                oversized += 1;
                for (0..s_count) |i| {
                    try e.bind(@intCast(i));
                    try e.verify(ids[i][0..vw], v_out[i][0..vw], 0);
                    try oracle.saveRows(e, owned[i], vw);
                }
            }
            for (0..s_count) |i| {
                try e.bind(@intCast(s_count + i));
                try e.verify(ids[i][0..vw], twin_out[i][0..vw], 0);
                try oracle.saveRows(e, owned[s_count + i], vw);
                try oracle.target(e, owned[i], owned[s_count + i], vw, v_out[i][0..vw], twin_out[i][0..vw], &n);
                const depth = if (depth_case < 4) depth_case else (i + width) % 4;
                parts[i] = .{ .slot = @intCast(i), .kept = width, .follow = v_out[i][0..width], .depth = depth, .out = b_out[i][0..depth], .chances = if (wantsChance(mode, i)) b_ch[i][0..depth] else null };
            }
            try e.draftShared(parts[0..s_count]);
            n.rounds += 1;
            for (parts[0..s_count], 0..) |p, i| {
                var out: [glm.engine.max_drafts]u32 = undefined;
                var ch: [glm.engine.max_drafts]f64 = undefined;
                try e.bind(@intCast(s_count + i));
                try e.keep(width);
                try e.draft(false, p.follow, p.depth, &out, if (p.chances != null) ch[0..p.depth] else null);
                try oracle.stream(e, i, s_count + i, p.out, out[0..p.depth], p.chances, if (p.chances != null) ch[0..p.depth] else null, &n);
            }
        };
    };
    std.debug.print("RESULT glm-mtp-shared-check rank={d} TP={d} rounds={d} streams={d} levels={d} drafts={d} chances={d} lengths={d} caches={d} upstream={d} oversized={d} cache=bf16-unpaged\n", .{ rank, world, n.rounds, n.streams, n.levels, n.drafts, n.chances, n.lengths, n.caches, n.upstream, oversized });
    if (n.drafts + n.chances + n.lengths + n.caches + n.upstream != 0) return error.TestFailed;
    check.pass("rank {d}: serial and batched MTP drafts, chances and cache bytes agree after every round", .{rank});
}
