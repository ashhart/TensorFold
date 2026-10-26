//! GLM-5.3-Flash's paired prompt steps (TF_GLM_PROMPT_PAIR) against twins a chunk a step, on one Engine and its ranks.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// `glm-pair-check`: prompts a step at a time, paired and not; first token, next window and every state byte equal.
pub fn pairCheck(gpu: Gpu, checkpoint: []const u8, dc_path: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dc = try std.json.parseFromSliceLeaky(struct { capacity: usize = 8192 }, a, try std.Io.Dir.cwd().readFileAlloc(gpu.io, dc_path, a, .limited(1 << 26)), .{ .ignore_unknown_fields = true });
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    const set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    try check.expect(set.pair, "glm-pair-check needs TF_GLM_PROMPT_PAIR on (unset or 1)", .{});
    const C = glm.engine.prompt_rows;
    const capacity = @max(dc.capacity, 4 * C);
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's paired and single prompt steps to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const V = e.share.vocab.len();
    const min = glm.prompt_pair.min_rows;
    // a pair and a tail, the shortest second chunk, a pair and a chunk, a pair to the end, a tail too short to pair
    const lens = [_]usize{ 2 * C + 300, C + min, 3 * C, 2 * C, C + min - 1 };
    var failed = false;
    for (lens, 0..) |len, i| {
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919 + i * 31) % 5000);
        var firsts: [2]u32 = undefined;
        var outs: [2][4]u32 = undefined;
        var logits: [2][]u16 = undefined;
        var paired: [2]u64 = undefined;
        for (0..2) |twin| { // twin 0 a chunk a step, twin 1 paired where it can
            e.settings.pair = twin == 1;
            try x.bind(@intCast(2 * i + twin));
            try x.begin(null);
            const before = e.pair.?.steps;
            var at: usize = 0;
            const t = while (true) {
                const r = x.chunk(ids, at, true, null, null, null) catch |err| {
                    std.debug.print("FAIL pair: prompt of {d}, twin {d}: the step at {d}: {s}\n", .{ len, twin, at, @errorName(err) });
                    return err;
                };
                if (r.first) |f| break f;
                at += r.rows;
            };
            paired[twin] = e.pair.?.steps - before;
            firsts[twin] = t;
            const win = [_]u32{ t, 101, 202, 303 };
            try x.verify(&win, &outs[twin], 0);
            logits[twin] = try a.alloc(u16, win.len * V);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(logits[twin].ptr, e.bd.logits, win.len * V * 2), "cuMemcpyDtoH");
        }
        e.settings.pair = true;
        var want: u64 = 0; // the steps glm.prompt_pair.takesTwo pairs over this prompt
        var at: usize = 0;
        while (at < len) {
            const two = glm.prompt_pair.takesTwo(at, len, e.share.world, &.{});
            want += @intFromBool(two);
            at += if (two) 2 * C else C;
        }
        if (paired[0] != 0 or paired[1] != want) {
            std.debug.print("FAIL pair: prompt of {d}: {d} paired steps a chunk a step, {d} paired (want {d})\n", .{ len, paired[0], paired[1], want });
            failed = true;
        }
        if (firsts[0] != firsts[1] or !std.mem.eql(u32, &outs[0], &outs[1]) or !std.mem.eql(u16, logits[0], logits[1])) {
            std.debug.print("FAIL pair: prompt of {d}: first {d}/{d}, the next window's tokens or logits differ\n", .{ len, firsts[0], firsts[1] });
            failed = true;
        }
        try e.stream.synchronize();
        const st = &e.seqs.items[2 * i].?.st;
        const tw = &e.seqs.items[2 * i + 1].?.st;
        const n = st.arena.buf.len;
        const h0 = try a.alloc(u8, n);
        const h1 = try a.alloc(u8, n);
        try st.arena.buf.download(0, h0);
        try tw.arena.buf.download(0, h1);
        const diff = std.mem.indexOfDiff(u8, h0, h1);
        if (st.pos != tw.pos or st.mtp_len != tw.mtp_len or !std.mem.eql(u1, st.cur, tw.cur) or diff != null) {
            std.debug.print("FAIL pair: prompt of {d}: the paired state differs (pos {d}/{d}, first byte {?d} of {d})\n", .{ len, st.pos, tw.pos, diff, n });
            failed = true;
        }
    }
    if (failed) return error.TestFailed;
    check.pass("paired prompt steps at TP={d}: prompts of {any} rows equal their twins a chunk a step in the first token, the next window's logits and tokens, and every state byte", .{ world, lens });
}
