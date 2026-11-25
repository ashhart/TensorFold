//! Full GLM-5.3's capped prompt steps (--decode-share's row cap, cuda_prompt_cuts.cappedEnd) against uncapped twins.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// A prompt and its request starts (TF_GLM_REQUEST_STARTS), in multiples of the cold chunk C plus rows.
const Case = struct { chunks: usize, tail: usize, starts: []const u32 = &.{} };

/// A state's bytes without its index rings: at pool 1 they hold a step's keys till pool_keys copies them (rebase).
fn maskRings(st: anytype, host: []u8) void {
    const n = st.ring_rows * st.ring_row;
    for (0..st.caches.len + 1) |j| {
        if (j < st.caches.len and st.caches[j].pk_rows == 0) continue; // a shared layer's cache: no rings
        for (st.rings[2 * j ..][0..2]) |r| @memset(host[r - st.arena.buf.ptr ..][0..n], 0);
    }
}

/// `glm-cap-check`: prompts a step at a time, capped and not; first token, next window and state bytes equal.
pub fn capCheck(gpu: Gpu, checkpoint: []const u8, dc_path: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
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
    const C = glm.engine.prompt_rows;
    const capacity = @max(dc.capacity, 4 * C);
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    try check.expect(e.c.dsa, "glm-cap-check takes full GLM-5.3 (glm_moe_dsa): GLM-5.3-Flash steps whole cold chunks", .{});
    try check.expect(e.c.kpool == 1, "glm-cap-check leaves the index rings out only where a pool is one key", .{});
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's capped and whole prompt steps to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const V = e.share.vocab.len();
    const cap_grid = glm.engine.prompt_cuts.cap_grid;
    // tails off and on the grid, a whole chunk, request starts off the grid, and a start a grid step past a cut
    const cases = [_]Case{
        .{ .chunks = 2, .tail = 300 },
        .{ .chunks = 2, .tail = 0 },
        .{ .chunks = 1, .tail = 5 + cap_grid },
        .{ .chunks = 1, .tail = 700, .starts = &.{ 300, C + 300 } },
        .{ .chunks = 2, .tail = 64, .starts = &.{ C + cap_grid, C + cap_grid + 1 } },
    };
    const caps = [_]u32{ 512, 128, 1000, 300, 0, 4096, 64 }; // a cap of 0 between capped steps runs to the cold cut
    var failed = false;
    var capped: usize = 0; // steps that ended inside a cold chunk
    for (cases, 0..) |cs, i| {
        const len = cs.chunks * C + cs.tail;
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919 + i * 31) % 5000);
        var firsts: [2]u32 = undefined;
        var outs: [2][4]u32 = undefined;
        var logits: [2][]u16 = undefined;
        for (0..2) |twin| { // twin 0 the cold pass, twin 1 capped
            try x.bind(@intCast(twin)); // two slots for every case: begin resets each, so no case holds another state
            try x.begin(null);
            if (cs.starts.len > 0) try x.cuts(cs.starts);
            var at: usize = 0;
            var step: usize = 0;
            const t = while (true) : (step += 1) {
                const cap: u32 = if (twin == 1) caps[(i + step) % caps.len] else 0;
                const r = x.chunkCapped(ids, at, false, null, null, null, cap) catch |err| {
                    std.debug.print("FAIL cap: prompt of {d}, twin {d}: the step at {d} (cap {d}): {s}\n", .{ len, twin, at, cap, @errorName(err) });
                    return err;
                };
                if (r.first) |f| break f;
                at += r.rows;
                if (twin == 1 and !glm.engine.prompt_cuts.coldCut(len, cs.starts, at)) capped += 1;
            };
            firsts[twin] = t;
            const win = [_]u32{ t, 101, 202, 303 };
            try x.verify(&win, &outs[twin], 0);
            logits[twin] = try a.alloc(u16, win.len * V);
            try cuda.memory.download(gpu.d, e.bd.logits, std.mem.sliceAsBytes(logits[twin])[0 .. win.len * V * 2]);
        }
        if (firsts[0] != firsts[1] or !std.mem.eql(u32, &outs[0], &outs[1]) or !std.mem.eql(u16, logits[0], logits[1])) {
            std.debug.print("FAIL cap: prompt of {d}: first {d}/{d}, the next window's tokens or logits differ\n", .{ len, firsts[0], firsts[1] });
            failed = true;
        }
        try e.stream.synchronize();
        const st = &e.seqs.items[0].?.st;
        const tw = &e.seqs.items[1].?.st;
        const n = st.arena.buf.len;
        const h0 = try a.alloc(u8, n);
        const h1 = try a.alloc(u8, n);
        try st.arena.buf.download(0, h0);
        try tw.arena.buf.download(0, h1);
        maskRings(st, h0);
        maskRings(tw, h1);
        const diff = std.mem.indexOfDiff(u8, h0, h1);
        if (st.pos != tw.pos or st.mtp_len != tw.mtp_len or diff != null) {
            std.debug.print("FAIL cap: prompt of {d}: the capped state differs (pos {d}/{d}, first byte {?d} of {d})\n", .{ len, st.pos, tw.pos, diff, n });
            failed = true;
        }
    }
    if (failed) return error.TestFailed;
    try check.expect(capped > 0, "no step ended inside a cold chunk", .{});
    check.pass("capped prompt steps at TP={d}: {d} prompts ({d} steps inside a cold chunk) equal their cold passes in the first token, the next window's logits and tokens, and every state byte but the index rings'", .{ world, cases.len, capped });
}
