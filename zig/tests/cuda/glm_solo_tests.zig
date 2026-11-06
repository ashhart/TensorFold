//! GLM-5.3-Flash's solo prompt passes (TF_GLM_CHUNK_SOLO) against twins a chunk a step, on one Engine and its ranks.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// The KDA records' copy a state reads next, both cleared in `bytes`: a pass flips them, so a solo pass's differ.
fn takeRecords(a: std.mem.Allocator, st: anytype, bytes: []u8) ![]const u8 {
    const at = st.rec - st.arena.buf.ptr;
    const one = st.kda_layers * st.rec_layer;
    const cur: usize = if (st.kda_layers > 0) st.cur[0] else 0;
    const out = try a.dupe(u8, bytes[at + cur * one ..][0..one]);
    @memset(bytes[at..][0 .. 2 * one], 0);
    return out;
}

/// `glm-solo-check`: prompts a step at a time, solo and not; first token, next window and every state byte equal.
pub fn soloCheck(gpu: Gpu, checkpoint: []const u8, dc_path: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
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
    try check.expect(set.solo, "glm-solo-check needs TF_GLM_CHUNK_SOLO=4096", .{});
    const C = glm.engine.prompt_rows;
    const capacity = @max(dc.capacity, 6 * C);
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's solo and single prompt steps to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const V = e.share.vocab.len();
    const Case = struct { len: usize, marks: []const u32 = &.{} };
    const cases = [_]Case{
        .{ .len = 2 * C + 300 }, // a solo pass and a tail
        .{ .len = C + 2 }, // the shortest solo pass
        .{ .len = C + 1 }, // a chunk and one row, a step each
        .{ .len = 3 * C }, // a solo pass and a chunk
        .{ .len = 2 * C }, // a solo pass to the end
        .{ .len = C }, // one chunk
        .{ .len = 5 * C + 7 }, // solo passes past a rebase
        .{ .len = 4 * C, .marks = &.{C} }, // a mark at chunk 1: 0 and 1 alone, then a solo pass
    };
    const Kept = struct {
        fn call(_: *anyopaque, _: u32) void {} // the check keeps no state: its marks only shape the steps
    };
    var no_ctx: u8 = 0;
    var failed = false;
    for (cases, 0..) |case, i| {
        const len = case.len;
        const marks: ?glm.engine.Marks = if (case.marks.len == 0) null else .{ .at = case.marks, .ctx = &no_ctx, .call = Kept.call };
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919 + i * 31) % 5000);
        var firsts: [2]u32 = undefined;
        var outs: [2][4]u32 = undefined;
        var logits: [2][]u16 = undefined;
        var solo: [2]u64 = @splat(0);
        for (0..2) |twin| { // twin 0 a chunk a step, twin 1 solo where it can, both with the same marks
            try x.bind(@intCast(2 * i + twin));
            try x.begin(null);
            var at: usize = 0;
            const t = while (true) {
                const r = x.chunk(ids, at, twin == 1, null, null, marks) catch |err| {
                    std.debug.print("FAIL solo: prompt of {d}, twin {d}: the step at {d}: {s}\n", .{ len, twin, at, @errorName(err) });
                    return err;
                };
                solo[twin] += @intFromBool(r.rows > C);
                if (r.first) |f| break f;
                at += r.rows;
            };
            firsts[twin] = t;
            const win = [_]u32{ t, 101, 202, 303 };
            try x.verify(&win, &outs[twin], 0);
            logits[twin] = try a.alloc(u16, win.len * V);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(logits[twin].ptr, e.bd.logits, win.len * V * 2), "cuMemcpyDtoH");
        }
        const st = &e.seqs.items[2 * i].?.st;
        const tw = &e.seqs.items[2 * i + 1].?.st;
        var want: u64 = 0; // the solo passes prompt_solo.at allows over this prompt, the rings' base followed here
        var base: usize = 0;
        var at: usize = 0;
        while (at < len) {
            const reach = @min(len - at, 2 * C);
            const two = glm.prompt_solo.takesTwo(at, len, case.marks) and glm.prompt_solo.ringsAgree(base, st.ring_rows, at, reach);
            const rows = @min(len - at, @as(usize, if (two) 2 * C else C));
            if (at + rows > base + st.ring_rows) base = at; // State.ready rebases at the step's start, a whole pool
            want += @intFromBool(two);
            at += rows;
        }
        if (solo[0] != 0 or solo[1] != want) {
            std.debug.print("FAIL solo: prompt of {d}: {d} solo passes a chunk a step, {d} solo (want {d})\n", .{ len, solo[0], solo[1], want });
            failed = true;
        }
        if (firsts[0] != firsts[1] or !std.mem.eql(u32, &outs[0], &outs[1]) or !std.mem.eql(u16, logits[0], logits[1])) {
            std.debug.print("FAIL solo: prompt of {d}: first {d}/{d}, the next window's tokens or logits differ\n", .{ len, firsts[0], firsts[1] });
            failed = true;
        }
        try e.stream.synchronize();
        const n = st.arena.buf.len;
        const h0 = try a.alloc(u8, n);
        const h1 = try a.alloc(u8, n);
        try st.arena.buf.download(0, h0);
        try tw.arena.buf.download(0, h1);
        const r0 = try takeRecords(a, st, h0);
        const r1 = try takeRecords(a, tw, h1);
        const diff = std.mem.indexOfDiff(u8, h0, h1);
        if (st.pos != tw.pos or st.mtp_len != tw.mtp_len or st.base != tw.base or st.mtp_base != tw.mtp_base or !std.mem.eql(u8, r0, r1) or diff != null) {
            std.debug.print("FAIL solo: prompt of {d}: the solo state differs (pos {d}/{d}, rings' base {d}/{d}, MTP base {d}/{d}, records {}, first byte {?d} of {d})\n", .{ len, st.pos, tw.pos, st.base, tw.base, st.mtp_base, tw.mtp_base, std.mem.eql(u8, r0, r1), diff, n });
            failed = true;
        }
    }
    if (failed) return error.TestFailed;
    check.pass("solo prompt passes at TP={d}: {d} prompts, one with a mark at chunk 1, equal their twins a chunk a step in the first token, the next window's logits and tokens, the rings' bases and every state byte", .{ world, cases.len });
}
